#[cfg(feature = "language")]
#[path = "support/atomic_postgres.rs"]
mod atomic_postgres;
#[path = "support/fact_context.rs"]
mod fact_context;
#[cfg(feature = "temporal-tests")]
#[path = "support/notification_http.rs"]
mod notification_http;
#[path = "support/provider_properties.rs"]
mod provider_properties;
#[cfg(feature = "language")]
#[path = "support/quality_engine.rs"]
mod quality_engine;
use anyhow::{Context, Result, ensure};
use aurelius_lanes::{
    document::{CanonicalDocument, DocumentKind, Section, SourceRef},
    indexing::{Citation, IndexedChunk, IndexedDocument},
    postgres::{PostgresConfig, PostgresSink},
};
use chrono::Utc;
use sqlx::{PgPool, postgres::PgPoolOptions};
use std::collections::BTreeMap;

fn config(schema: &str, artifacts: &str, chunks: &str) -> PostgresConfig {
    PostgresConfig {
        url_env: "PG_ADAPTER_TEST_URL".into(),
        schema: schema.into(),
        artifacts_table: artifacts.into(),
        chunks_table: chunks.into(),
        tenant_setting: Some("projection.scope".into()),
        pool_size: 2,
    }
}
fn document(id: &str) -> IndexedDocument {
    let source = SourceRef {
        stream: "fixture".into(),
        record_id: id.into(),
        uri: "https://example.invalid/evidence".into(),
        object_key: "scope/raw/evidence.json".into(),
        record_index: 0,
    };
    let text = "synthetic evidence";
    let document = CanonicalDocument {
        id: id.into(),
        organization_id: "tenant-a".into(),
        connection_id: "connection-a".into(),
        kind: DocumentKind::GenericText,
        title: "Fixture".into(),
        sections: vec![Section {
            text: text.into(),
            author: None,
            occurred_at: None,
            source: source.clone(),
        }],
        context: BTreeMap::new(),
        modified_at: Utc::now(),
    };
    let chunks = (0..2)
        .map(|n| IndexedChunk {
            id: format!("piece-{n}"),
            text: text.into(),
            context_header: "Fixture".into(),
            citations: vec![Citation {
                byte_start: 0,
                byte_end: text.len(),
                source: source.clone(),
            }],
            tokens: 3,
            embedding: vec![1., 0., 0.],
        })
        .collect();
    IndexedDocument {
        content_hash: document.content_hash().unwrap(),
        document,
        profile: "a".repeat(64),
        chunks,
    }
}
async fn count(pool: &PgPool, sql: &'static str) -> Result<i64> {
    Ok(sqlx::query_scalar(sql).fetch_one(pool).await?)
}

#[tokio::test]
async fn configurable_names_atomic_replay_vectors_and_tenant_isolation() -> Result<()> {
    let url = std::env::var("PG_ADAPTER_TEST_URL")
        .context("start the disposable PG18 fixture and set PG_ADAPTER_TEST_URL")?;
    let parsed = reqwest::Url::parse(&url)?;
    ensure!(
        matches!(parsed.host_str(), Some("127.0.0.1" | "localhost"))
            && parsed.path() == "/custom_adapter_db",
        "PG tests require an isolated loopback custom_adapter_db fixture"
    );
    let admin = PgPoolOptions::new()
        .max_connections(2)
        .connect(&url)
        .await?;
    sqlx::raw_sql(r#"
DROP SCHEMA IF EXISTS "adapter-target" CASCADE;
DROP SCHEMA IF EXISTS "other target" CASCADE;
DROP ROLE IF EXISTS adapter_writer;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA "adapter-target";
CREATE SCHEMA "other target";
CREATE ROLE adapter_writer LOGIN PASSWORD 'fixture-writer-password';
CREATE TABLE "adapter-target"."source-docs" (
 id text PRIMARY KEY, organization_id text NOT NULL, source_type text NOT NULL,
 source_uri text NOT NULL, external_id text NOT NULL, revision_id text,
 content_hash text, mime_type text, title text, metadata jsonb, deleted_at timestamp,
 created_at timestamp NOT NULL, updated_at timestamp NOT NULL
);
CREATE TABLE "adapter-target"."chunk-rows" (
 id text PRIMARY KEY,organization_id text NOT NULL,artifact_source_id text NOT NULL REFERENCES "adapter-target"."source-docs"(id),
 source_path text NOT NULL,source_hash text,chunk_index integer NOT NULL,context_window_id text NOT NULL,
 child_text text NOT NULL CHECK(child_text<>'reject-chunk'),parent_block text NOT NULL,context_header text NOT NULL,
 metadata jsonb,embedding vector(3) NOT NULL,created_at timestamp NOT NULL,updated_at timestamp NOT NULL
);
ALTER TABLE "adapter-target"."source-docs" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "adapter-target"."chunk-rows" ENABLE ROW LEVEL SECURITY;
CREATE POLICY fixture_artifacts ON "adapter-target"."source-docs" TO adapter_writer
 USING(organization_id='tenant-a' AND organization_id=current_setting('projection.scope',true))
 WITH CHECK(organization_id='tenant-a' AND organization_id=current_setting('projection.scope',true));
CREATE POLICY fixture_chunks ON "adapter-target"."chunk-rows" TO adapter_writer
 USING(organization_id='tenant-a' AND organization_id=current_setting('projection.scope',true))
 WITH CHECK(organization_id='tenant-a' AND organization_id=current_setting('projection.scope',true));
GRANT USAGE ON SCHEMA "adapter-target" TO adapter_writer;
GRANT SELECT,INSERT ON ALL TABLES IN SCHEMA "adapter-target" TO adapter_writer;
CREATE TABLE "other target"."renamed""sources" (LIKE "adapter-target"."source-docs" INCLUDING ALL);
CREATE TABLE "other target"."search pieces" (LIKE "adapter-target"."chunk-rows" INCLUDING ALL);
ALTER TABLE "other target"."search pieces" ADD FOREIGN KEY(artifact_source_id) REFERENCES "other target"."renamed""sources"(id);
GRANT USAGE ON SCHEMA "other target" TO adapter_writer;
GRANT SELECT,INSERT ON ALL TABLES IN SCHEMA "other target" TO adapter_writer;
"#).execute(&admin).await?;
    let mut writer_url = parsed;
    writer_url
        .set_username("adapter_writer")
        .map_err(|_| anyhow::anyhow!("invalid fixture role"))?;
    writer_url
        .set_password(Some("fixture-writer-password"))
        .map_err(|_| anyhow::anyhow!("invalid fixture URL"))?;
    let writer = PgPoolOptions::new()
        .max_connections(2)
        .connect(writer_url.as_str())
        .await?;
    let target = config("adapter-target", "source-docs", "chunk-rows");
    let sink = PostgresSink::from_pool(target.clone(), 3, writer.clone()).await?;
    let indexed = document("doc-a");
    assert!(!sink.contains(&indexed.document, &indexed.profile).await?);
    let receipt = sink.write(&indexed).await?;
    assert_eq!(sink.write(&indexed).await?, receipt);
    assert!(sink.contains(&indexed.document, &indexed.profile).await?);
    assert_eq!(
        count(
            &admin,
            "SELECT count(*) FROM \"adapter-target\".\"source-docs\""
        )
        .await?,
        1
    );
    assert_eq!(
        count(
            &admin,
            "SELECT count(*) FROM \"adapter-target\".\"chunk-rows\""
        )
        .await?,
        2
    );
    assert_eq!(count(&admin,"SELECT count(*) FROM \"adapter-target\".\"chunk-rows\" WHERE vector_dims(embedding)=3 AND metadata->'citations'->0->>'byte_end'='18'").await?,2);
    let unset: Option<String> =
        sqlx::query_scalar("SELECT nullif(current_setting('projection.scope',true),'')")
            .fetch_one(&writer)
            .await?;
    assert!(unset.is_none(), "tenant context leaked outside transaction");
    let mut failing = document("rollback");
    failing.chunks[1].text = "reject-chunk".into();
    failing.chunks[1].citations[0].byte_end = failing.chunks[1].text.len();
    assert!(sink.write(&failing).await.is_err());
    assert_eq!(
        count(
            &admin,
            "SELECT count(*) FROM \"adapter-target\".\"source-docs\""
        )
        .await?,
        1,
        "artifact survived failed chunk transaction"
    );
    assert_eq!(
        count(
            &admin,
            "SELECT count(*) FROM \"adapter-target\".\"chunk-rows\""
        )
        .await?,
        2,
        "partial chunks survived rollback"
    );
    let mut foreign = document("doc-a");
    foreign.document.organization_id = "tenant-b".into();
    foreign.content_hash = foreign.document.content_hash()?;
    assert_ne!(
        PostgresSink::artifact_id(&indexed.document, &indexed.profile)?,
        PostgresSink::artifact_id(&foreign.document, &foreign.profile)?
    );
    assert!(!sink.contains(&foreign.document, &foreign.profile).await?);
    assert!(
        sink.write(&foreign).await.is_err(),
        "tenant RLS was bypassed"
    );
    assert!(
        PostgresSink::from_pool(target.clone(), 4, writer.clone())
            .await
            .is_err()
    );
    let mut invalid = indexed.clone();
    invalid.chunks[0].embedding[0] = f32::NAN;
    assert!(sink.write(&invalid).await.is_err());
    invalid = indexed.clone();
    invalid.chunks[0].embedding = vec![1., 0.];
    assert!(sink.write(&invalid).await.is_err());
    let mut injection = target.clone();
    injection.chunks_table = "missing\"; DROP SCHEMA public CASCADE; --".into();
    assert!(
        PostgresSink::from_pool(injection, 3, writer.clone())
            .await
            .is_err()
    );
    assert_eq!(
        count(
            &admin,
            "SELECT count(*) FROM pg_extension WHERE extname='vector'"
        )
        .await?,
        1
    );
    let alternate = PostgresSink::from_pool(
        config("other target", "renamed\"sources", "search pieces"),
        3,
        writer.clone(),
    )
    .await?;
    assert_eq!(alternate.write(&indexed).await?, receipt);
    assert_eq!(
        count(
            &admin,
            "SELECT count(*) FROM \"other target\".\"search pieces\""
        )
        .await?,
        2
    );
    sqlx::raw_sql(r#"CREATE TABLE "other target".unlinked (LIKE "other target"."search pieces" INCLUDING ALL); GRANT SELECT,INSERT ON "other target".unlinked TO adapter_writer;"#).execute(&admin).await?;
    assert!(
        PostgresSink::from_pool(
            config("other target", "renamed\"sources", "unlinked"),
            3,
            writer.clone()
        )
        .await
        .is_err()
    );
    sqlx::raw_sql(r#"ALTER TABLE "other target".unlinked DROP COLUMN context_header;"#)
        .execute(&admin)
        .await?;
    assert!(
        PostgresSink::from_pool(
            config("other target", "renamed\"sources", "unlinked"),
            3,
            writer.clone()
        )
        .await
        .is_err()
    );
    fact_context::exercise(&sink, &admin).await?;
    direct_ingestion_keeps_source_storage_read_only(sink, &admin).await?;
    writer.close().await;
    sqlx::raw_sql(r#"DROP SCHEMA "adapter-target" CASCADE; DROP SCHEMA "other target" CASCADE; DROP ROLE adapter_writer;"#).execute(&admin).await?;
    admin.close().await;
    Ok(())
}

async fn direct_ingestion_keeps_source_storage_read_only(
    sink: PostgresSink,
    admin: &PgPool,
) -> Result<()> {
    use aurelius_lanes::{
        config::Config,
        ingest::{IngestActivities, PgIngestor, ingest_once},
        storage,
    };
    use axum::{Json, Router, extract::State, routing::post};
    use futures_util::TryStreamExt;
    use object_store::{ObjectStore, memory::InMemory};
    use serde_json::{Value, json};
    use std::{
        io::Write,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };
    async fn token() -> Json<Value> {
        Json(json!({"access_token":"fixture-token","expires_in":3600}))
    }
    async fn connection() -> Json<Value> {
        Json(
            json!({"syncCatalog":{"streams":[{"stream":{"name":"channel_messages"},"config":{"selected":true,"destinationSyncMode":"append"}}]}}),
        )
    }
    fn job(id: u64, status: &str) -> Value {
        json!({"job":{"id":id,"configId":"fixture-connection","configType":"sync","status":status}})
    }
    async fn jobs() -> Json<Value> {
        Json(json!({"jobs":[job(3,"running"),job(2,"failed"),job(1,"succeeded")]}))
    }
    async fn get_job(Json(body): Json<Value>) -> Json<Value> {
        let id = body["id"].as_u64().expect("fixture job id");
        let mut value = job(
            id,
            match id {
                1 | 4 | 5 => "succeeded",
                2 => "failed",
                _ => "running",
            },
        );
        if id == 5 {
            value["job"]["configId"] = json!("another-connection");
        }
        Json(value)
    }
    async fn tokenize() -> Json<Value> {
        Json(json!({"tokens":[1,2,3,4]}))
    }
    async fn embedding(State(calls): State<Arc<AtomicUsize>>) -> Json<Value> {
        calls.fetch_add(1, Ordering::SeqCst);
        Json(json!({"data":[{"index":0,"embedding":[1.,2.,3.]}]}))
    }
    async fn fact_response(
        State(calls): State<Arc<AtomicUsize>>,
        Json(body): Json<Value>,
    ) -> (axum::http::StatusCode, Json<Value>) {
        calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(body["model"], "gpt-6-luna");
        if body.pointer("/text/format/name").and_then(Value::as_str)
            == Some(aurelius_lanes::atomic::FORMAT_NAME)
        {
            return (
                axum::http::StatusCode::OK,
                Json(
                    json!({"status":"completed","usage":{"input_tokens":20,"output_tokens":30},"output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"{\"local_entities\":[],\"assertions\":[],\"review_items\":[]}"}]}]}),
                ),
            );
        }
        let input: Value =
            serde_json::from_str(body["input"][1]["content"].as_str().expect("fixture input"))
                .expect("fixture input JSON");
        if let Some(mentions) = input.get("mentions").and_then(Value::as_array) {
            let argument = |text: &str, kind: &str| json!({"mention_id":mentions.iter().find(|mention|mention["text"]==text).unwrap_or_else(|| panic!("expected synthetic mention {text}; available: {:?}", mentions.iter().map(|mention|mention["text"].as_str().unwrap_or("?")).collect::<Vec<_>>()))["id"],"entity_type":kind});
            let fact = json!({"subject":argument("Alice","person"),"body":{"kind":"relation","predicate":"manages","object":argument("Acme","company")},"quote":"Alice manages Acme.","statement_type":"observation","polarity":"positive","modality":"asserted","qualifiers":{"adverbs":[],"conditions":[],"scope":null},"validity":{"from":null,"to":null}});
            let mut unknown = fact.clone();
            unknown["body"]["predicate"] = json!("arbitrary_activity");
            return (
                axum::http::StatusCode::OK,
                Json(
                    json!({"status":"completed","usage":{"input_tokens":20,"input_tokens_details":{"cached_tokens":10},"output_tokens":30},"output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":json!({"facts":[fact,unknown]}).to_string()}]}]}),
                ),
            );
        }
        if body.to_string().contains("provider-down") {
            return (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                Json(json!({"error":"fixture quota"})),
            );
        }
        let fact = json!({"subject":"synthetic","predicate":"owns","object":"root","statement_type":"observation","polarity":"positive","modality":"asserted","qualifiers":{"adverbs":[],"conditions":[],"scope":null},"valid_from":null,"valid_to":null,"quote":"synthetic root"});
        (
            axum::http::StatusCode::OK,
            Json(
                json!({"status":"completed","usage":{"input_tokens":20,"output_tokens":30},"output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":json!({"facts":[fact]}).to_string()}]}]}),
            ),
        )
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let router = Router::new()
        .route("/api/public/v1/applications/token", post(token))
        .route("/api/v1/connections/get", post(connection))
        .route("/api/v1/jobs/list", post(jobs))
        .route("/api/v1/jobs/get", post(get_job))
        .route("/tokenize", post(tokenize))
        .route("/v1/embeddings", post(embedding))
        .route("/responses", post(fact_response))
        .with_state(calls.clone());
    let server = tokio::spawn(async move { axum::serve(listener, router).await });
    let settings = format!(
        "{}\n[source]\nendpoint='http://{address}'\nclient_id_env='SOURCE_API_TEST_CLIENT_ID'\nclient_secret_env='SOURCE_API_TEST_CLIENT_SECRET'\n[xberg]\nendpoint='http://{address}/extract'\n[embedding]\nendpoint='http://{address}'\nmodel='fixture'\nrevision='fixture-v1'\nnative_dimensions=3\ndimensions=3\ntoken_limit=512\n[lanes.fixture]\nscope='tenant-a'\nconnection='fixture-connection'\nnormalizer='conversation'\nbucket='fixture'\nraw_prefix='fixture/raw/'\noutput_prefix='unused/'\nstreams=['channel_messages']\ncitation_base='https://example.invalid'\n",
        include_str!("fixtures/business.toml")
    );
    let settings = Config::parse(&settings)?;
    let raw = Arc::new(InMemory::new());
    let key = "fixture/raw/1/channel_messages/source.jsonl.gz";
    let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    for (n, text) in [(1, "synthetic root"), (2, "synthetic reply")] {
        writeln!(
            &mut gzip,
            "{}",
            json!({"_airbyte_extracted_at":n*1000,"_airbyte_data":{"channel_id":"channel","ts":format!("{n}.000000"),"thread_ts":"1.000000","text":text}})
        )?;
    }
    writeln!(
        &mut gzip,
        "{}",
        json!({"_airbyte_extracted_at":3000,"_airbyte_data":{"channel_id":"channel","ts":"3.000000","thread_ts":"99.000000","text":"orphan body must not be embedded"}})
    )?;
    let reference = storage::publish(raw.as_ref(), key, gzip.finish()?, None).await?;
    let before = storage::read(raw.as_ref(), &reference, 1024 * 1024).await?;
    let raw_binding: Arc<dyn ObjectStore> = raw.clone();
    let stores: BTreeMap<String, Arc<dyn ObjectStore>> =
        BTreeMap::from([("fixture".into(), raw_binding)]);
    let mut ingest = PgIngestor::with_storage(settings.clone(), stores.clone(), sink.clone())?;
    let request = ingest.request("fixture")?;
    let result = ingest_once(&mut ingest, request.clone())
        .await
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        .context("expected completed epoch")?;
    assert_eq!(result.epoch.sequence, 1);
    assert_eq!(result.indexed, 1);
    assert_eq!(result.deferred, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let repeated = ingest_once(&mut ingest, request.clone())
        .await
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    assert_eq!(repeated, None, "completed workflow lost its durable cursor");
    assert_eq!(
        sink.ingestion_cursor("tenant-a", &request.lane).await?,
        Some(1)
    );
    assert_eq!(
        sink.ingestion_cursor("tenant-b", &request.lane).await?,
        None
    );
    let mut other = request.lane.clone();
    other.profile = "b".repeat(64);
    assert_eq!(sink.ingestion_cursor("tenant-a", &other).await?, None);
    other = request.lane.clone();
    other.lane = "another-lane".into();
    assert_eq!(sink.ingestion_cursor("tenant-a", &other).await?, None);
    let mut restarted = PgIngestor::with_storage(settings.clone(), stores, sink.clone())?;
    assert!(
        ingest_once(&mut restarted, request.clone())
            .await
            .map_err(|error| anyhow::anyhow!("{error:?}"))?
            .is_none()
    );
    assert_eq!(
        ingest.ingest(result.epoch.clone()).await?,
        result,
        "replayed epoch changed receipt"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "replay recalculated a committed embedding"
    );
    let mut advanced = request;
    advanced.after = Some(1);
    assert!(
        ingest_once(&mut ingest, advanced)
            .await
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .is_none(),
        "failed/running source epochs were ingested"
    );
    assert_eq!(
        raw.list(None).try_collect::<Vec<_>>().await?.len(),
        1,
        "ingestion wrote derived S3 state"
    );
    assert_eq!(
        storage::read(raw.as_ref(), &reference, 1024 * 1024).await?,
        before
    );
    assert_eq!(count(admin,"SELECT count(*) FROM \"adapter-target\".\"chunk-rows\" WHERE metadata->'citations'->0->'source'->>'object_key'='fixture/raw/1/channel_messages/source.jsonl.gz' AND metadata->'citations'->1->'source'->>'record_index'='1'").await?,1);
    assert_eq!(count(admin,"SELECT count(*) FROM \"adapter-target\".\"source-docs\" WHERE metadata->'context'->>'status'='deferred' AND metadata->'context'->>'source_row'='2' AND metadata->>'chunk_count'='0'").await?,1);
    let source_id:String=sqlx::query_scalar("SELECT id FROM \"adapter-target\".\"source-docs\" WHERE organization_id='tenant-a' AND metadata->'sources'->0->>'object_key'='fixture/raw/1/channel_messages/source.jsonl.gz' AND metadata->'context'->>'status' IS DISTINCT FROM 'deferred'").fetch_one(admin).await?;
    let projection = sink.extraction_input("tenant-a", &source_id).await?;
    assert!(projection.document.text.contains("synthetic root"));
    assert!(sink.extraction_input("tenant-b", &source_id).await.is_err());
    let chunk = aurelius_lanes::contracts::Chunk {
        document: projection.document,
        range: projection.ranges[0],
    };
    let extractor = aurelius_lanes::facts::ChatExtractor::new(
        &aurelius_lanes::config::ChatConfig {
            endpoint: "http://127.0.0.1:1/chat".into(),
            model: "fixture".into(),
            revision: "fixture-v1".into(),
            api_key_env: None,
            api: Default::default(),
            gateway: None,
            references: Default::default(),
        },
        settings.ontology.clone(),
    )?;
    let candidates=aurelius_lanes::facts::parse_candidates(&json!({"facts":[{"subject":"synthetic","predicate":"owns","object":"root","statement_type":"observation","polarity":"positive","modality":"asserted","qualifiers":{"adverbs":[],"conditions":[],"scope":null},"valid_from":null,"valid_to":null,"quote":"synthetic root"}]}).to_string(),chunk.text()?)?;
    let graph = extractor.graph(&chunk, candidates)?;
    let references =
        aurelius_lanes::references::ReferenceConfig::default().resolve(&chunk.document)?;
    assert!(!sink.candidate_exists(&chunk, &graph.profile).await?);
    let candidate = sink.write_candidates(&chunk, &graph, &references).await?;
    assert!(sink.candidate_exists(&chunk, &graph.profile).await?);
    assert_eq!(
        sink.write_candidates(&chunk, &graph, &references).await?,
        candidate
    );
    assert!(sink.extraction_input("tenant-a", &candidate).await.is_err());
    assert_eq!(count(admin,"SELECT count(*) FROM \"adapter-target\".\"source-docs\" WHERE metadata->>'status'='candidate' AND jsonb_array_length(metadata->'candidate_graph'->'claims')=1").await?,1);
    assert_eq!(count(admin,"SELECT count(*) FROM \"adapter-target\".\"chunk-rows\" c JOIN \"adapter-target\".\"source-docs\" a ON a.id=c.artifact_source_id WHERE a.metadata->>'status'='candidate'").await?,0);
    let mut foreign = graph.clone();
    foreign.document.scope = "tenant-b".into();
    assert!(
        sink.write_candidates(&chunk, &foreign, &references)
            .await
            .is_err()
    );
    let mut fabricated = graph;
    fabricated.claims[0].evidence[0].quote = "fabricated".into();
    assert!(
        sink.write_candidates(&chunk, &fabricated, &references)
            .await
            .is_err()
    );
    let config_path = std::env::temp_dir().join(format!(
        "aurelius-pilot-fixture-{}.toml",
        std::process::id()
    ));
    let config_text = format!(
        "{}\n[postgres]\nurl_env='PILOT_TEST_DB_URL'\nschema='adapter-target'\nartifacts_table='source-docs'\nchunks_table='chunk-rows'\ntenant_setting='projection.scope'\n[embedding]\nendpoint='http://{address}'\nmodel='fixture'\nrevision='fixture-v1'\nnative_dimensions=3\ndimensions=3\ntoken_limit=512\n[chat]\nendpoint='http://{address}/responses'\nmodel='gpt-6-luna'\nrevision='cli-pilot-v1'\napi='responses'\n[chat.gateway]\ntoken_env='PILOT_TEST_GATEWAY_TOKEN'\n[lanes.fixture]\nscope='tenant-a'\nconnection='fixture-connection'\nnormalizer='conversation'\nbucket='fixture'\nraw_prefix='fixture/raw/'\nstreams=['channel_messages']\ncitation_base='https://example.invalid'\n",
        include_str!("fixtures/business.toml")
    );
    std::fs::write(&config_path, &config_text)?;
    let mut planning = Config::parse(&config_text)?;
    planning
        .postgres
        .as_mut()
        .context("fixture postgres")?
        .url_env = "INTENTIONALLY_MISSING_DATABASE_URL".into();
    planning
        .chat
        .as_mut()
        .context("fixture chat")?
        .gateway
        .as_mut()
        .context("fixture gateway")?
        .token_env = "SOURCE_API_TEST_CLIENT_SECRET".into();
    assert_eq!(
        aurelius_lanes::enrichment::FactEngine::requests(&planning)?.len(),
        1,
        "workflow identity planning must not connect to PG"
    );
    let shared = aurelius_lanes::enrichment::FactEngine::with_sink(planning, sink.clone())?;
    assert_eq!(
        shared.request("tenant-a")?.scope,
        "tenant-a",
        "shared-pool engine tried to open a second DB pool"
    );
    let mut url = reqwest::Url::parse(&std::env::var("PG_ADAPTER_TEST_URL")?)?;
    url.set_username("adapter_writer")
        .map_err(|_| anyhow::anyhow!("fixture role URL"))?;
    url.set_password(Some("fixture-writer-password"))
        .map_err(|_| anyhow::anyhow!("fixture role URL"))?;
    for _ in 0..2 {
        let result = tokio::process::Command::new(env!("CARGO_BIN_EXE_aurelius-lanes"))
            .args([
                "pilot-facts",
                config_path.to_str().context("fixture path")?,
                "tenant-a",
                &source_id,
            ])
            .env("PILOT_TEST_DB_URL", url.as_str())
            .env("PILOT_TEST_GATEWAY_TOKEN", "fixture-only-token")
            .output()
            .await?;
        ensure!(
            result.status.success(),
            "fact pilot fixture CLI failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "CLI replay made another model call"
    );
    async fn fact_cli(path: &std::path::Path, url: &reqwest::Url) -> Result<std::process::Output> {
        Ok(
            tokio::process::Command::new(env!("CARGO_BIN_EXE_aurelius-lanes"))
                .args([
                    "facts-once",
                    path.to_str().context("fixture path")?,
                    "tenant-a",
                ])
                .env("PILOT_TEST_DB_URL", url.as_str())
                .env("PILOT_TEST_GATEWAY_TOKEN", "fixture-only-token")
                .output()
                .await?,
        )
    }
    let first = fact_cli(&config_path, &url).await?;
    ensure!(
        first.status.success(),
        "first fact step: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "completed pilot repeated model inference"
    );
    let mut bad = document("bad-facts");
    bad.profile = settings.lane_request("fixture")?.profile;
    bad.document.connection_id = "fixture-connection".into();
    bad.content_hash = bad.document.content_hash()?;
    let bad_id = sink.write(&bad).await?.artifact_id;
    let rejected = fact_cli(&config_path, &url).await?;
    ensure!(
        rejected.status.success(),
        "review step: {}",
        String::from_utf8_lossy(&rejected.stderr)
    );
    let review: Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(review["review_windows"], json!(bad.chunks.len()));
    assert_eq!(review["source"]["artifact_id"], bad_id);
    let complete: Value = serde_json::from_slice(&first.stdout)?;
    let profile = complete["source"]["request"]["profile"]
        .as_str()
        .context("fact profile")?;
    assert_eq!(
        sink.fact_status("tenant-a", &bad_id, profile, None)
            .await?
            .as_deref(),
        Some("facts_complete")
    );
    assert_eq!(count(admin,"SELECT count(*) FROM \"adapter-target\".\"source-docs\" WHERE metadata->>'status'='fact_window_review'").await?,i64::try_from(bad.chunks.len())?);
    let mut good = document("good-facts");
    good.profile = bad.profile.clone();
    good.document.connection_id = "fixture-connection".into();
    good.document.sections[0].text = "synthetic root".into();
    good.chunks[0].text = "synthetic root".into();
    good.chunks[0].citations[0].byte_end = "synthetic root".len();
    good.content_hash = good.document.content_hash()?;
    sink.write(&good).await?;
    let next = fact_cli(&config_path, &url).await?;
    ensure!(
        next.status.success(),
        "next fact step: {}",
        String::from_utf8_lossy(&next.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&next.stdout)?["stored_windows"],
        1
    );
    let mut down = good;
    down.document.id = "provider-down".into();
    down.document.sections[0].text = "provider-down".into();
    down.chunks[0].text = "provider-down".into();
    down.chunks[0].citations[0].byte_end = "provider-down".len();
    down.content_hash = down.document.content_hash()?;
    let down_id = sink.write(&down).await?.artifact_id;
    let outage = fact_cli(&config_path, &url).await?;
    assert!(!outage.status.success());
    assert!(
        sink.fact_status("tenant-a", &down_id, profile, None)
            .await?
            .is_none(),
        "provider outage was quarantined as source failure"
    );
    assert_eq!(
        sink.next_fact_source("tenant-a", &[bad.profile], profile)
            .await?,
        Some(down_id)
    );
    #[cfg(feature = "temporal-tests")]
    notification_http::qualify(settings.clone(), sink.clone()).await?;
    #[cfg(feature = "language")]
    quality_engine::exercise(&config_text, &sink, admin, calls.clone()).await?;
    #[cfg(feature = "language")]
    atomic_postgres::exercise(&sink, admin, &config_text).await?;
    provider_properties::exercise(settings.clone(), &sink, &config_text, calls.clone()).await?;
    std::fs::remove_file(config_path)?;
    server.abort();
    Ok(())
}
