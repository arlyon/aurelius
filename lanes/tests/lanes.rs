use anyhow::Result;
use aurelius_lanes::{
    config::Config,
    embedding::project,
    lanes::{
        EpochRef, LaneActivities, LaneError, LaneReceipt, LaneRequest, ObjectRef, PublicationRef,
        index_lane,
    },
    mail::decode_body,
    storage,
    xberg::{XbergClient, XbergConfig},
};
use object_store::memory::InMemory;
use serde_json::json;

fn request() -> LaneRequest {
    LaneRequest {
        lane: "fixture".into(),
        profile: "a".repeat(64),
    }
}
fn epoch() -> EpochRef {
    EpochRef {
        request: request(),
        sequence: 1,
        manifest: ObjectRef {
            key: "scope/epochs/1.json".into(),
            version: "epoch-v1".into(),
        },
    }
}
fn publication() -> PublicationRef {
    PublicationRef {
        epoch: epoch(),
        manifest: ObjectRef {
            key: "scope/published/1.json".into(),
            version: "publication-v1".into(),
        },
    }
}
fn receipt() -> LaneReceipt {
    LaneReceipt {
        publication: publication(),
        receipt: ObjectRef {
            key: "scope/receipts/1.json".into(),
            version: "receipt-v1".into(),
        },
        indexed: 2,
        deferred: 1,
        ignored: 0,
    }
}
#[derive(Default)]
struct Functions {
    events: Vec<&'static str>,
    no_epoch: bool,
    fail_index: bool,
    foreign: bool,
}
impl LaneActivities for Functions {
    type Error = anyhow::Error;
    async fn discover(&mut self, _: LaneRequest) -> Result<Option<EpochRef>> {
        self.events.push("discover");
        let mut e = epoch();
        if self.foreign {
            e.request.profile = "b".repeat(64);
        }
        Ok((!self.no_epoch).then_some(e))
    }
    async fn publish(&mut self, _: EpochRef) -> Result<PublicationRef> {
        self.events.push("publish");
        Ok(publication())
    }
    async fn index(&mut self, _: PublicationRef) -> Result<LaneReceipt> {
        self.events.push("index");
        anyhow::ensure!(!self.fail_index, "injected failure");
        Ok(receipt())
    }
    async fn acknowledge(&mut self, _: LaneReceipt) -> Result<()> {
        self.events.push("acknowledge");
        Ok(())
    }
}
#[tokio::test]
async fn ordinary_function_orders_io_and_keeps_deferred_coverage() {
    let mut f = Functions::default();
    let r = index_lane(&mut f, request()).await.unwrap().unwrap();
    assert_eq!(r.deferred, 1);
    assert_eq!(f.events, ["discover", "publish", "index", "acknowledge"]);
}
#[tokio::test]
async fn incomplete_source_has_no_downstream_side_effects() {
    let mut f = Functions {
        no_epoch: true,
        ..Default::default()
    };
    assert!(index_lane(&mut f, request()).await.unwrap().is_none());
    assert_eq!(f.events, ["discover"]);
}
#[tokio::test]
async fn direct_failure_does_not_retry_or_checkpoint() {
    let mut f = Functions {
        fail_index: true,
        ..Default::default()
    };
    assert!(matches!(
        index_lane(&mut f, request()).await,
        Err(LaneError::Activity(_))
    ));
    assert_eq!(f.events, ["discover", "publish", "index"]);
}
#[tokio::test]
async fn changed_profile_is_rejected_before_publication() {
    let mut f = Functions {
        foreign: true,
        ..Default::default()
    };
    assert!(matches!(
        index_lane(&mut f, request()).await,
        Err(LaneError::Invalid(_))
    ));
    assert_eq!(f.events, ["discover"]);
}
#[tokio::test]
async fn pg_ingestion_rejects_stale_epochs_and_foreign_receipts() {
    use aurelius_lanes::ingest::{
        IngestActivities, IngestEpoch, IngestReceipt, IngestRequest, ingest_once,
    };
    struct Functions {
        sequence: u64,
        foreign: bool,
        ingested: bool,
    }
    impl IngestActivities for Functions {
        type Error = anyhow::Error;
        async fn discover(&mut self, _: IngestRequest) -> Result<Option<u64>> {
            Ok(Some(self.sequence))
        }
        async fn ingest(&mut self, mut epoch: IngestEpoch) -> Result<IngestReceipt> {
            self.ingested = true;
            if self.foreign {
                epoch.sequence += 1;
            }
            Ok(IngestReceipt {
                epoch,
                indexed: 1,
                deferred: 0,
                ignored: 0,
            })
        }
    }
    let request = IngestRequest {
        lane: request(),
        after: Some(10),
    };
    let mut functions = Functions {
        sequence: 10,
        foreign: false,
        ingested: false,
    };
    assert!(matches!(
        ingest_once(&mut functions, request.clone()).await,
        Err(LaneError::Invalid(_))
    ));
    assert!(!functions.ingested);
    functions.sequence = 11;
    functions.foreign = true;
    assert!(matches!(
        ingest_once(&mut functions, request).await,
        Err(LaneError::Invalid(_))
    ));
}

#[tokio::test]
async fn failure_codes_withhold_response_urls_credentials_and_source_context() {
    use axum::{Router, http::StatusCode, routing::get};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route("/", get(|| async { StatusCode::SERVICE_UNAVAILABLE })),
        )
        .await
    });
    let error = reqwest::get(format!("http://{address}/?credential=PRIVATE_SENTINEL"))
        .await
        .unwrap()
        .error_for_status()
        .unwrap_err();
    let error = anyhow::Error::from(error).context("PRIVATE_DOCUMENT_TEXT");
    assert_eq!(
        aurelius_lanes::ingest::failure_code(&error),
        "stage=unknown http_status=503 timeout=false postgres_code=none"
    );
    server.abort();
}

#[tokio::test]
async fn fact_pipeline_rejects_scope_changes_and_foreign_receipts() {
    use aurelius_lanes::enrichment::{
        FactActivities, FactReceipt, FactRequest, FactSource, FactStatus, enrich_once,
    };
    struct Functions {
        foreign_discovery: bool,
        foreign_receipt: bool,
    }
    impl FactActivities for Functions {
        type Error = anyhow::Error;
        async fn discover(&mut self, mut request: FactRequest) -> Result<Option<FactSource>> {
            if self.foreign_discovery {
                request.scope = "other-org".into();
            }
            Ok(Some(FactSource {
                request,
                artifact_id: "opaque-ref".into(),
            }))
        }
        async fn process(&mut self, mut source: FactSource) -> Result<FactReceipt> {
            if self.foreign_receipt {
                source.artifact_id = "other-source".into();
            }
            Ok(FactReceipt {
                source,
                status: FactStatus::Complete,
                stored_windows: 1,
                skipped_windows: 0,
                review_windows: 0,
                new_claims: 1,
            })
        }
    }
    let request = FactRequest {
        scope: "org-fixture".into(),
        profile: "a".repeat(64),
    };
    let mut functions = Functions {
        foreign_discovery: true,
        foreign_receipt: false,
    };
    assert!(matches!(
        enrich_once(&mut functions, request.clone()).await,
        Err(LaneError::Invalid(_))
    ));
    functions.foreign_discovery = false;
    functions.foreign_receipt = true;
    assert!(matches!(
        enrich_once(&mut functions, request.clone()).await,
        Err(LaneError::Invalid(_))
    ));
    functions.foreign_receipt = false;
    assert!(
        enrich_once(&mut functions, request)
            .await
            .unwrap()
            .is_some()
    );
}

#[test]
fn unsafe_object_refs_and_profiles_fail() {
    for key in ["../source", "/absolute", "safe/../source", "safe\\source"] {
        assert!(
            ObjectRef {
                key: key.into(),
                version: "v1".into()
            }
            .validate()
            .is_err()
        );
    }
    let mut r = request();
    r.profile = "wrong".into();
    assert!(r.validate().is_err());
}
#[tokio::test]
async fn object_versions_fence_reads_and_publications() {
    let store = InMemory::new();
    let first = storage::publish(&store, "scope/test.json", b"first".to_vec(), None)
        .await
        .unwrap();
    assert_eq!(storage::read(&store, &first, 100).await.unwrap(), b"first");
    assert!(
        storage::publish(&store, "scope/test.json", b"wrong".to_vec(), None)
            .await
            .is_err()
    );
    let second = storage::publish(
        &store,
        "scope/test.json",
        b"second".to_vec(),
        Some(&first.version),
    )
    .await
    .unwrap();
    assert!(storage::read(&store, &first, 100).await.is_err());
    assert!(
        storage::publish(
            &store,
            "scope/test.json",
            b"stale".to_vec(),
            Some(&first.version)
        )
        .await
        .is_err()
    );
    assert_eq!(
        storage::read(&store, &second, 100).await.unwrap(),
        b"second"
    );
    assert!(storage::read(&store, &second, 2).await.is_err());
}
#[tokio::test]
async fn exclusive_group_overwrite_keeps_version_checked_reads() {
    let store = InMemory::new();
    let first = storage::publish(&store, "scope/group.json", b"first".to_vec(), None)
        .await
        .unwrap();
    let second = storage::overwrite_exclusive(&store, "scope/group.json", b"second".to_vec())
        .await
        .unwrap();
    assert!(storage::read(&store, &first, 100).await.is_err());
    assert_eq!(
        storage::read(&store, &second, 100).await.unwrap(),
        b"second"
    );
}

#[test]
fn projection_preserves_finite_unit_norm_and_rejects_bad_vectors() {
    let v = project(vec![1.; 2560], 2560, 1280).unwrap();
    assert_eq!(v.len(), 1280);
    let norm = v.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
    assert!((norm - 1.).abs() < 1e-6);
    assert!(project(vec![0.; 2560], 2560, 1280).is_err());
    assert!(project(vec![f32::NAN; 2560], 2560, 1280).is_err());
}
#[cfg(unix)]
#[tokio::test]
async fn reducer_closes_stdin_and_drains_output_without_deadlock() -> Result<()> {
    use aurelius_lanes::reduction::ReductionConfig;
    use std::os::unix::fs::PermissionsExt;
    let path = std::env::temp_dir().join(format!(
        "aurelius-reducer-eof-{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap()
    ));
    std::fs::write(&path, "#!/bin/sh\ncat\n")?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700))?;
    let input = vec![json!({"text":"synthetic fixture ".repeat(20000)})];
    let reducer = ReductionConfig {
        executable: path.to_string_lossy().into_owned(),
        schema: "fixture.yaml".into(),
        collection: "fixture".into(),
    };
    let result =
        tokio::time::timeout(std::time::Duration::from_secs(5), reducer.combine(&input)).await;
    std::fs::remove_file(path)?;
    assert_eq!(result??, input);
    Ok(())
}

#[test]
fn base64url_mime_decoding_is_strict() {
    assert_eq!(decode_body("Tm8u").unwrap(), b"No.");
    for bad in ["%%%", "a", "YQ===", "YQ\n"] {
        assert!(decode_body(bad).is_err());
    }
}
#[tokio::test]
async fn mail_thread_orders_messages_decodes_charsets_and_preserves_attachment_gaps() -> Result<()>
{
    use aurelius_lanes::{
        document::{SourceRecord, SourceRef},
        mail,
    };
    use axum::{Json, Router, extract::State, routing::post};
    use base64::Engine;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    async fn extract(State(calls): State<Arc<AtomicUsize>>) -> Json<serde_json::Value> {
        calls.fetch_add(1, Ordering::SeqCst);
        Json(json!([{"content":"HTML evidence","mime_type":"text/html"}]))
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let state = calls.clone();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new()
                .route("/extract", post(extract))
                .with_state(state),
        )
        .await
    });
    let xberg = XbergClient::new(&XbergConfig {
        endpoint: format!("http://{address}/extract"),
        api_key_env: None,
    })?;
    let encode = |bytes: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
    let message = |id: &str, date: &str, payload: serde_json::Value| json!({"id":id,"threadId":"thread","internalDate":date,"payload":payload});
    let early = message(
        "early",
        "1000",
        json!({"mimeType":"text/plain","headers":[{"name":"Content-Type","value":"text/plain; charset=windows-1252"},{"name":"Subject","value":"Fixture"}],"body":{"data":encode(b"caf\xe9")}}),
    );
    let later = message(
        "later",
        "2000",
        json!({"mimeType":"multipart/mixed","parts":[{"mimeType":"multipart/alternative","parts":[{"mimeType":"text/plain","body":{"data":encode(b"plain evidence")}},{"mimeType":"text/html","body":{"data":encode(b"<p>duplicate HTML</p>")}}]},{"mimeType":"application/pdf","filename":"missing.pdf","body":{"attachmentId":"unavailable"}}]}),
    );
    let last = message(
        "last",
        "3000",
        json!({"mimeType":"multipart/alternative","parts":[{"mimeType":"text/plain","body":{"data":""}},{"mimeType":"text/html","body":{"data":encode(b"<p>HTML evidence</p>")}}]}),
    );
    let mut record = SourceRecord {
        organization_id: "scope".into(),
        connection_id: "connection".into(),
        stream: "threads_details".into(),
        record_id: "thread".into(),
        extracted_at: chrono::DateTime::from_timestamp_millis(4000).unwrap(),
        deleted: false,
        data: json!({"id":"thread","messages":[last,later,early]}),
        source: SourceRef {
            stream: "threads_details".into(),
            record_id: "thread".into(),
            uri: "https://mail.google.com/mail/u/0/#all/thread".into(),
            object_key: "scope/raw/thread.json".into(),
            record_index: 0,
        },
    };
    let document = mail::conversation(&record, &xberg).await?;
    assert_eq!(
        document
            .sections
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>(),
        ["café", "plain evidence", "HTML evidence"]
    );
    assert_eq!(
        document
            .sections
            .iter()
            .map(|s| s.source.record_id.as_str())
            .collect::<Vec<_>>(),
        ["early", "later", "last"]
    );
    assert_eq!(document.context["unavailable_attachments"], "1");
    assert_eq!(document.organization_id, "scope");
    assert_eq!(document.title, "Fixture");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "preferred plain text must not duplicate HTML extraction"
    );
    record.data["messages"][0]["threadId"] = json!("foreign-thread");
    assert!(mail::conversation(&record, &xberg).await.is_err());
    server.abort();
    Ok(())
}

#[test]
fn embedded_lanes_and_mime_client_config_validate() {
    let base = include_str!("fixtures/business.toml");
    let config = format!(
        "{base}\n[xberg]\nendpoint='http://127.0.0.1:8000/extract'\n[lanes.fixture]\nscope='scope'\nconnection='connection'\nnormalizer='file'\nbucket='bucket'\nraw_prefix='scope/raw/'\noutput_prefix='scope/derived/'\nstreams=['documents']\n"
    );
    let c = Config::parse(&config).unwrap();
    assert_eq!(c.lanes.len(), 1);
    assert!(c.lane_request("fixture").unwrap().validate().is_ok());
    assert!(Config::parse(&config.replace("scope/derived/", "scope/raw/")).is_err());
    assert!(
        XbergClient::new(&XbergConfig {
            endpoint: "https://user:secret@example.com".into(),
            api_key_env: None
        })
        .is_err()
    );
}
#[test]
fn lane_profile_tracks_semantics_not_database_names_or_service_locations() {
    use aurelius_lanes::{embedding::EmbeddingConfig, postgres::PostgresConfig};
    let text = format!(
        "{}\n[lanes.fixture]\nscope='scope'\nconnection='connection'\nnormalizer='file'\nbucket='bucket'\nraw_prefix='raw/'\nstreams=['files']\n",
        include_str!("fixtures/business.toml")
    );
    let mut config = Config::parse(&text).unwrap();
    config.embedding = Some(EmbeddingConfig {
        endpoint: "http://127.0.0.1:8081".into(),
        model: "fixture".into(),
        revision: "r1".into(),
        native_dimensions: 3,
        dimensions: 3,
        token_limit: 512,
    });
    let profile = config.lane_request("fixture").unwrap();
    config.embedding.as_mut().unwrap().endpoint = "https://another-service.invalid".into();
    config.postgres = Some(PostgresConfig {
        url_env: "OTHER_DATABASE_URL".into(),
        schema: "another schema".into(),
        artifacts_table: "records".into(),
        chunks_table: "pieces".into(),
        tenant_setting: None,
        pool_size: 2,
    });
    assert_eq!(profile, config.lane_request("fixture").unwrap());
    config.embedding.as_mut().unwrap().revision = "r2".into();
    assert_ne!(profile, config.lane_request("fixture").unwrap());
}

#[tokio::test]
async fn xberg_gateway_accepts_general_mime_files_and_fails_closed() {
    use axum::{Json, Router, routing::post};
    async fn respond() -> Json<serde_json::Value> {
        Json(json!([{"content":"Extracted fixture text","mime_type":"application/pdf"}]))
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, Router::new().route("/extract", post(respond)))
            .await
            .unwrap();
    });
    let client = XbergClient::new(&XbergConfig {
        endpoint: format!("http://{address}/extract"),
        api_key_env: None,
    })
    .unwrap();
    for mime in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/html",
        "text/plain",
        "text/markdown",
        "message/rfc822",
        "text/csv",
        "image/png",
        "application/epub+zip",
        "application/rtf",
    ] {
        assert!(
            !client
                .extract(b"fixture".to_vec(), "fixture", mime)
                .await
                .unwrap()
                .content
                .is_empty()
        );
    }
    assert!(
        client
            .extract(Vec::new(), "empty", "application/pdf")
            .await
            .is_err()
    );
    server.abort();
}
