#![cfg(feature = "language")]
use anyhow::{Context, Result};
use aurelius_lanes::{
    atomic,
    atomic_lab::{self, Limits},
    config::{ChatApi, Config},
    contracts::{ByteRange, Document, DocumentKey},
    fact_lab::{self, LabRecord, Mode, Options},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    os::unix::fs::{DirBuilderExt, PermissionsExt},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
#[tokio::test]
async fn actual_neural_preparation_budgeted_mock_and_credential_free_bound_replay() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "atomic-lab-{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap()
    ));
    std::fs::DirBuilder::new().mode(0o700).create(&root)?;
    struct Owned(std::path::PathBuf);
    impl Drop for Owned {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _owned = Owned(root.clone());
    let text = "Reporting schedule\nDeliver report via Slack.";
    let quote = "Deliver report via Slack.";
    let start = text.find(quote).unwrap();
    let body=json!({"local_entities":[{"id":"r","entity_type":"report_schedule","identity_span_ref":"span:0-18","action_kind":null}],"assertions":[{"subject":{"ref":"r"},"predicate":"delivery_channel","object":{"kind":"code","value":"slack"},"qualification":{"mode":"required","polarity":"positive","conditions":[],"attribution":[],"validity":{"from":null,"to":null}},"evidence":[format!("span:{start}-{}",text.len())]}],"review_items":[]}).to_string();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let router=axum::Router::new().route("/responses",axum::routing::post(move |axum::Json(request):axum::Json<Value>|{observed.fetch_add(1,Ordering::SeqCst);let body=body.clone();async move {assert_eq!(request["text"]["format"]["type"],"json_schema");assert_eq!(request["text"]["format"]["strict"],true);axum::Json(json!({"status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":body}]}],"usage":{"input_tokens":100,"output_tokens":100}}))}}));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(async move { axum::serve(listener, router).await });
    let mut config: Config = toml::from_str(include_str!("../examples/fact_lab.toml"))?;
    let chat = config.chat.as_mut().unwrap();
    chat.api = ChatApi::Responses;
    chat.endpoint = format!("http://{address}/responses");
    chat.model = "fixture".into();
    chat.api_key_env = None;
    chat.gateway = None;
    config.language.as_mut().unwrap().model_dir = std::env::var("AURELIUS_TEST_LANGUAGE_MODEL_DIR")
        .context("actual local neural assets required")?;
    let registry: atomic::Registry =
        serde_json::from_str(include_str!("../examples/atomic_spo.json"))?;
    let record = LabRecord {
        id: "one".into(),
        provider: "synthetic".into(),
        document: Document {
            key: DocumentKey {
                scope: "fixture-tenant".into(),
                source: "fixture-document".into(),
                revision: "1".into(),
            },
            text: text.into(),
            context: None,
            metadata: Default::default(),
        },
        range: ByteRange {
            start: 0,
            end: text.len(),
        },
        prepared: None,
        provider_properties: None,
        canonical_context: None,
    };
    let mut second = record.clone();
    second.id = "two".into();
    let records = vec![record, second];
    let live = Options {
        mode: Mode::Live,
        output: root.join("live"),
        selected: BTreeSet::new(),
        fresh: true,
        batch_adjacent: false,
        max_calls: 1,
        max_batch_bytes: 4096,
    };
    let limits = Limits {
        max_cents: 100,
        seconds: 600,
        max_output_tokens: 8192,
    };
    let first = atomic_lab::run(&config, &records, &registry, &live, limits).await?;
    assert_eq!(first.attempts, 1);
    assert_eq!(first.assertions, 1);
    assert_eq!(first.not_run, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        std::fs::metadata(&live.output)?.permissions().mode() & 0o777,
        0o700
    );
    for entry in std::fs::read_dir(&live.output)? {
        assert_eq!(entry?.metadata()?.permissions().mode() & 0o777, 0o600);
    }
    assert!(
        atomic_lab::run(&config, &records, &registry, &live, limits)
            .await
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let cached = fact_lab::load(&live.output.join("prepared-records.json"))?;
    config.chat.as_mut().unwrap().api_key_env =
        Some("ATOMIC_REPLAY_MUST_NOT_READ_CREDENTIALS".into());
    let replay = Options {
        mode: Mode::Replay(live.output.clone()),
        output: root.join("replay"),
        selected: BTreeSet::from(["one".into()]),
        fresh: false,
        batch_adjacent: false,
        max_calls: 0,
        max_batch_bytes: 4096,
    };
    let second = atomic_lab::run(&config, &cached, &registry, &replay, limits).await?;
    assert_eq!(second.attempts, 0);
    assert_eq!(second.assertions, 1);
    assert_eq!(second.failed_records, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let before: Value =
        serde_json::from_slice(&std::fs::read(live.output.join("record-one.json"))?)?;
    let after: Value =
        serde_json::from_slice(&std::fs::read(replay.output.join("record-one.json"))?)?;
    assert_eq!(before, after);
    let mut corrupt: Value =
        serde_json::from_slice(&std::fs::read(live.output.join("response-one.json"))?)?;
    corrupt["binding"] = json!("other_tenant_request");
    std::fs::write(
        live.output.join("response-one.json"),
        serde_json::to_vec(&corrupt)?,
    )?;
    let wrong = Options {
        output: root.join("wrong"),
        ..replay
    };
    let failed = atomic_lab::run(&config, &cached, &registry, &wrong, limits).await?;
    assert_eq!(failed.failed_records, 1);
    assert_eq!(failed.assertions, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    server.abort();
    Ok(())
}
