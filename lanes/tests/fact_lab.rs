use anyhow::{Context, Result};
use aurelius_lanes::{
    config::{ChatApi, ChatConfig, Config},
    contracts::{ByteRange, Document, DocumentKey},
    fact_lab::{self, LabRecord, Mode, Options},
    language::LanguageConfig,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Result<Self> {
        let root = std::env::temp_dir().join(format!(
            "aurelius-lab-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        std::fs::create_dir(&root)?;
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700))?;
        Ok(Self(root))
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn records() -> Vec<LabRecord> {
    let first = "Alice manages Acme.";
    let text = format!("{first}\n\n{first}");
    let document = Document {
        key: DocumentKey {
            scope: "tenant-a".into(),
            source: "source-a".into(),
            revision: "r1".into(),
        },
        text,
        context: None,
        metadata: Default::default(),
    };
    [
        ByteRange {
            start: 0,
            end: first.len(),
        },
        ByteRange {
            start: first.len() + 2,
            end: document.text.len(),
        },
    ]
    .into_iter()
    .enumerate()
    .map(|(i, range)| LabRecord {
        id: format!("{i:03}"),
        provider: "fixture".into(),
        document: document.clone(),
        range,
        prepared: None,
        provider_properties: None,
        canonical_context: None,
    })
    .collect()
}
#[test]
fn minimal_lab_config_has_no_ingestion_service_dependencies() -> Result<()> {
    let temp = Temp::new()?;
    std::fs::write(
        temp.path("lab.toml"),
        include_str!("../examples/fact_lab.toml"),
    )?;
    let config = fact_lab::load_config(&temp.path("lab.toml"))?;
    assert!(
        config.postgres.is_none()
            && config.temporal.is_none()
            && config.source.is_none()
            && config.storage.is_none()
    );
    assert!(config.fact_context.is_none());
    Ok(())
}
#[test]
fn file_loader_selection_and_batching_preserve_source_boundaries() -> Result<()> {
    let temp = Temp::new()?;
    let input = records();
    std::fs::write(
        temp.path("sample.json"),
        serde_json::to_vec(&json!({"slots":input}))?,
    )?;
    let loaded = fact_lab::load(&temp.path("sample.json"))?;
    assert_eq!(loaded.len(), 2);
    let selected = BTreeSet::from(["001".into()]);
    assert_eq!(
        fact_lab::units(&loaded, &selected, false)?[0].record.range,
        loaded[1].range
    );
    assert!(fact_lab::units(&loaded, &BTreeSet::from(["missing".into()]), false).is_err());
    let merged = fact_lab::units(&loaded, &BTreeSet::new(), true)?;
    assert_eq!(merged.len(), 1);
    assert_eq!(
        fact_lab::units_bounded(&loaded, &BTreeSet::new(), true, 20)?.len(),
        2
    );
    assert!(fact_lab::units_bounded(&loaded, &BTreeSet::new(), true, 0).is_err());
    assert_eq!(merged[0].members, ["000", "001"]);
    assert_eq!(
        merged[0].record.range.text(&merged[0].record.document)?,
        loaded[0].document.text
    );
    let mut different = loaded.clone();
    different[1].document.key.scope = "tenant-b".into();
    assert_eq!(
        fact_lab::units(&different, &BTreeSet::new(), true)?.len(),
        2
    );
    let mut separated = loaded.clone();
    separated[0].document.text = "Alice manages Acme. UNSELECTED Alice manages Acme.".into();
    separated[1].document = separated[0].document.clone();
    separated[1].range = ByteRange {
        start: 30,
        end: separated[1].document.text.len(),
    };
    assert_eq!(
        fact_lab::units(&separated, &BTreeSet::new(), true)?.len(),
        2
    );
    let line = json!({"id":"unicode","provider":"fixture","document":{"key":{"scope":"t","source":"s","revision":"r"},"text":"Émilie approved it."}});
    std::fs::write(temp.path("records.jsonl"), format!("{line}\n"))?;
    let plain = fact_lab::load(&temp.path("records.jsonl"))?;
    assert_eq!(plain[0].range.end, plain[0].document.text.len());
    let mut duplicate = input.clone();
    duplicate[1].id = duplicate[0].id.clone();
    std::fs::write(temp.path("duplicate.json"), serde_json::to_vec(&duplicate)?)?;
    assert!(fact_lab::load(&temp.path("duplicate.json")).is_err());
    let mut traversal = input;
    traversal[0].id = "../escape".into();
    std::fs::write(temp.path("traversal.json"), serde_json::to_vec(&traversal)?)?;
    assert!(fact_lab::load(&temp.path("traversal.json")).is_err());
    Ok(())
}
async fn respond(
    axum::extract::State(calls): axum::extract::State<Arc<AtomicUsize>>,
    axum::Json(body): axum::Json<Value>,
) -> (axum::http::StatusCode, axum::Json<Value>) {
    calls.fetch_add(1, Ordering::SeqCst);
    let input: Value = serde_json::from_str(body["input"][1]["content"].as_str().unwrap()).unwrap();
    if input["source_text"]
        .as_str()
        .unwrap()
        .contains("provider-down")
    {
        return (
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            axum::Json(json!({"error":"fixture"})),
        );
    }
    if input["source_text"]
        .as_str()
        .unwrap()
        .contains("fixture-incomplete")
    {
        assert_eq!(body["max_output_tokens"], 8192);
        return (
            axum::http::StatusCode::OK,
            axum::Json(
                json!({"status":"incomplete","usage":{"input_tokens":12,"output_tokens":8192}}),
            ),
        );
    }
    let mentions = input["mentions"].as_array().unwrap();
    let argument = |name: &str, kind: &str| json!({"mention_id":mentions.iter().find(|m|m["text"]==name).expect("actual neural/source mention")["id"],"entity_type":kind});
    let fact = json!({"subject":argument("Alice","person"),"body":{"kind":"relation","predicate":"manages","object":argument("Acme","company")},"quote":"Alice manages Acme.","statement_type":"observation","polarity":"positive","modality":"asserted","qualifiers":{"adverbs":[],"conditions":[],"scope":null},"validity":{"from":null,"to":null}});
    let mut rejected = fact.clone();
    rejected["body"]["predicate"] = json!("unsupported_relation");
    (
        axum::http::StatusCode::OK,
        axum::Json(
            json!({"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":4},"output_tokens":5},"output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":json!({"facts":[fact,rejected]}).to_string()}]}]}),
        ),
    )
}
fn config(endpoint: String) -> Result<Config> {
    let mut config = Config::parse(include_str!("fixtures/business.toml"))?;
    config.chat = Some(ChatConfig {
        endpoint,
        model: "fixture-no-paid-inference".into(),
        revision: "v1".into(),
        api_key_env: None,
        api: ChatApi::Responses,
        gateway: None,
        references: Default::default(),
    });
    config.language = Some(LanguageConfig {
        model_dir: std::env::var("AURELIUS_TEST_LANGUAGE_MODEL_DIR")
            .context("set local test model directory")?,
        max_document_tokens: 2048,
        threads: 2,
    });
    config.fact_quality = Some(serde_json::from_value(
        json!({"context_as_of":"2026-05-01T12:00:00Z"}),
    )?);
    Ok(config)
}
fn options(output: &Path, mode: Mode) -> Options {
    Options {
        mode,
        output: output.into(),
        selected: BTreeSet::new(),
        fresh: true,
        batch_adjacent: false,
        max_calls: 1,
        max_batch_bytes: 4096,
    }
}
#[tokio::test]
async fn file_only_e2e_real_nlp_mock_transport_cap_replay_and_diagnostics() -> Result<()> {
    use axum::{Router, routing::post};
    let temp = Temp::new()?;
    let calls = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let state = calls.clone();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new()
                .route("/responses", post(respond))
                .with_state(state),
        )
        .await
    });
    let config = config(format!("http://{address}/responses"))?;
    let records = records();
    let live = options(&temp.path("live"), Mode::Live);
    let summary = fact_lab::run(&config, &records, &live).await?;
    assert_eq!(summary.attempts, 1);
    assert_eq!(summary.not_run, 1);
    assert_eq!(summary.candidates, 1);
    assert_eq!(summary.rejected_proposals, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(summary.usage_complete);
    assert_eq!(summary.cached_input_tokens, 4);
    let result: Value =
        serde_json::from_slice(&std::fs::read(live.output.join("record-000.json"))?)?;
    assert_eq!(result["proposals"].as_array().unwrap().len(), 2);
    assert_eq!(
        result["graph"]["rejections"][0]["reason"],
        "unknown_predicate"
    );
    assert_eq!(
        std::fs::metadata(&live.output)?.permissions().mode() & 0o777,
        0o700
    );
    assert_eq!(
        std::fs::metadata(live.output.join("record-000.json"))?
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert!(fact_lab::run(&config, &records, &live).await.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let cached = fact_lab::load(&live.output.join("prepared-records.json"))?;
    let mut replay = options(&temp.path("replay"), Mode::Replay(live.output.clone()));
    replay.fresh = false;
    replay.max_calls = 0;
    let replayed = fact_lab::run(&config, &cached, &replay).await?;
    assert_eq!(replayed.attempts, 0);
    assert_eq!(replayed.candidates, 1);
    assert_eq!(replayed.rejected_proposals, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let diff = fact_lab::compare(&live.output, &replay.output)?;
    assert_eq!(diff["records"].as_array().unwrap().len(), 2);
    assert!(diff["records"][1]["after"].is_null());
    let mut no_budget = options(&temp.path("no-budget"), Mode::Live);
    no_budget.max_calls = 0;
    assert!(fact_lab::run(&config, &records, &no_budget).await.is_err());
    assert!(!no_budget.output.exists());
    let mut secret = records.clone();
    secret[0].document.text = "Authorization: Bearer fixtureSecretOnly123456".into();
    secret[0].range = ByteRange {
        start: 0,
        end: secret[0].document.text.len(),
    };
    secret.truncate(1);
    let secret_run = options(&temp.path("secret"), Mode::Live);
    let rejected = fact_lab::run(&config, &secret, &secret_run).await?;
    assert_eq!(rejected.attempts, 0);
    let saved = std::fs::read_to_string(secret_run.output.join("record-000.json"))?;
    assert!(!saved.contains("fixtureSecretOnly123456"));
    assert!(!secret_run.output.join("request-000.json").exists());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let mut batched = options(&temp.path("batched"), Mode::Live);
    batched.batch_adjacent = true;
    let batch = fact_lab::run(&config, &records, &batched).await?;
    assert_eq!(batch.selected_records, 2);
    assert_eq!(batch.extraction_units, 1);
    assert_eq!(batch.attempts, 1);
    assert_eq!(batch.candidates, 1);
    let cached_batch = fact_lab::load(&batched.output.join("prepared-records.json"))?;
    let mut batch_replay = options(
        &temp.path("batch-replay"),
        Mode::Replay(batched.output.clone()),
    );
    batch_replay.fresh = false;
    batch_replay.batch_adjacent = true;
    batch_replay.max_calls = 0;
    assert_eq!(
        fact_lab::run(&config, &cached_batch, &batch_replay)
            .await?
            .candidates,
        1
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let mut incomplete = records[..1].to_vec();
    incomplete[0].document.text = "fixture-incomplete".into();
    incomplete[0].range = ByteRange {
        start: 0,
        end: incomplete[0].document.text.len(),
    };
    let mut larger_output = config.clone();
    larger_output
        .fact_quality
        .as_mut()
        .expect("quality")
        .max_output_tokens = Some(8192);
    let incomplete_options = options(&temp.path("incomplete"), Mode::Live);
    let partial = fact_lab::run(&larger_output, &incomplete, &incomplete_options).await?;
    assert_eq!(partial.review_units, 1);
    assert_eq!(partial.attempts, 1);
    assert_eq!(partial.output_tokens, 8192);
    assert!(partial.usage_complete);
    let trace_path = incomplete_options.output.join("failed-output-000.json");
    assert!(trace_path.exists());
    assert_eq!(
        std::fs::metadata(&trace_path)?.permissions().mode() & 0o777,
        0o600
    );
    assert!(serde_json::from_slice::<Value>(&std::fs::read(trace_path)?)?["output"].is_string());
    let mut failure = records.clone();
    for record in &mut failure {
        record.document.text = "provider-down".into();
        record.range = ByteRange { start: 0, end: 13 };
    }
    let mut stop = options(&temp.path("provider-stop"), Mode::Live);
    stop.max_calls = 2;
    let stopped = fact_lab::run(&config, &failure, &stop).await?;
    assert!(stopped.provider_stopped);
    assert_eq!(stopped.attempts, 1);
    assert_eq!(stopped.not_run, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    server.abort();
    Ok(())
}
