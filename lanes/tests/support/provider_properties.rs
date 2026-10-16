use anyhow::{Result, ensure};
use aurelius_lanes::{
    config::Config,
    document::{DocumentKind, SourceRef},
    lanes::Normalizer,
    postgres::PostgresSink,
    provider_metadata::{ProviderMetadataReader, ProviderValue},
    storage,
};
use futures_util::TryStreamExt;
use object_store::{ObjectStore, memory::InMemory};
use serde_json::json;
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

pub async fn exercise(
    mut config: Config,
    sink: &PostgresSink,
    model_config_text: &str,
    calls: Arc<AtomicUsize>,
) -> Result<()> {
    let before_calls = calls.load(Ordering::SeqCst);
    #[cfg(not(feature = "language"))]
    let _ = model_config_text;
    let lane = config.lanes.get_mut("fixture").expect("fixture lane");
    lane.normalizer = Normalizer::Task;
    lane.streams = vec!["tasks".into(), "stories".into()];
    let key = "fixture/raw/1/tasks/task.jsonl";
    let raw = Arc::new(InMemory::new());
    let row=json!({"_airbyte_data":{"gid":"task-1","name":"Task record","notes":"Alice manages Acme.","due_on":"2026-05-01","due_at":"2026-05-01T09:30:00+02:00","completed":false,"approval_status":"pending","assignee":{"gid":"person-1","name":"Alice"},"created_by":{"gid":"person-2"},"created_at":"2026-04-01T12:00:00Z","modified_at":"2026-04-30T08:00:00Z"}}).to_string();
    let object = storage::publish(raw.as_ref(), key, row.into_bytes(), None).await?;
    let before = storage::read(raw.as_ref(), &object, 1024 * 1024).await?;
    let source = SourceRef {
        stream: "tasks".into(),
        record_id: "tasks/task-1".into(),
        uri: "https://example.invalid/task-1".into(),
        object_key: key.into(),
        record_index: 0,
    };
    let mut indexed = super::document("provider-properties");
    indexed.document.kind = DocumentKind::AsanaTask;
    indexed.document.connection_id = "fixture-connection".into();
    indexed.document.sections[0].source = source.clone();
    let header = "Task: Task record\nAssignee: Alice\nDue date: 2026-05-01\nDue time: 2026-05-01T09:30:00+02:00\nStatus: open";
    let normalized = format!("{header}\n\nAlice manages Acme.");
    indexed.document.sections[0].text = normalized.clone();
    indexed.profile = config.lane_request("fixture")?.profile;
    for (index, chunk) in indexed.chunks.iter_mut().enumerate() {
        chunk.text = if index == 0 {
            normalized.clone()
        } else {
            "Alice manages Acme.".into()
        };
        for citation in &mut chunk.citations {
            citation.source = source.clone();
            citation.byte_start = 0;
            citation.byte_end = chunk.text.len();
        }
    }
    indexed.content_hash = indexed.document.content_hash()?;
    let id = sink.write(&indexed).await?.artifact_id;
    let input = sink.extraction_input("tenant-a", &id).await?;
    let store: Arc<dyn ObjectStore> = raw.clone();
    let stores = BTreeMap::from([("fixture".into(), store)]);
    let reader = ProviderMetadataReader::with_storage(config.clone(), stores.clone())?;
    let properties = reader.read(&input).await?.expect("Asana metadata");
    ensure!(properties.properties.len() == 5);
    let verified_header = properties.header.as_ref().expect("verified raw-row header");
    ensure!(
        verified_header.text == header,
        "provider projection changed normalized/embedded bytes"
    );
    verified_header.validate(&input.document)?;
    let mut mismatched = input.clone();
    mismatched
        .document
        .text
        .insert_str(0, "changed projection\n");
    ensure!(
        sink.write_provider_properties(&mismatched, &properties, &"a".repeat(64))
            .await
            .is_err(),
        "provider metadata accepted header/source drift"
    );
    ensure!(matches!(
        properties.properties[0].value,
        ProviderValue::Date { .. }
    ));
    ensure!(
        matches!(&properties.properties[1].value,ProviderValue::Timestamp{value} if value.offset().local_minus_utc()==7200)
    );
    ensure!(matches!(
        properties.properties[2].value,
        ProviderValue::Boolean { value: false }
    ));
    ensure!(properties.author.as_ref().expect("author").id == "person-2");
    let receipt = sink
        .write_provider_properties(&input, &properties, &"a".repeat(64))
        .await?;
    ensure!(
        sink.write_provider_properties(&input, &properties, &"a".repeat(64))
            .await?
            == receipt,
        "provider metadata replay was not idempotent"
    );
    ensure!(
        storage::read(raw.as_ref(), &object, 1024 * 1024).await? == before
            && raw.list(None).try_collect::<Vec<_>>().await?.len() == 1,
        "metadata extraction modified raw storage"
    );
    let mut other = input.clone();
    other.document.key.scope = "tenant-b".into();
    ensure!(reader.read(&other).await.is_err());
    let mut failed = input;
    failed.sources[0].object_key = "fixture/raw/2/tasks/task.jsonl".into();
    ensure!(
        reader.read(&failed).await.is_err(),
        "failed epoch authorized provider metadata"
    );
    ensure!(
        calls.load(Ordering::SeqCst) == before_calls,
        "metadata projection called embedding or generation"
    );
    #[cfg(feature = "language")]
    {
        use aurelius_lanes::{
            enrichment::{FactActivities, FactEngine, FactSource},
            fact_context::FactContextConfig,
            language::LanguageConfig,
        };
        let model = Config::parse(model_config_text)?;
        config.chat = model.chat;
        config
            .chat
            .as_mut()
            .expect("chat")
            .gateway
            .as_mut()
            .expect("gateway")
            .token_env = "SOURCE_API_TEST_CLIENT_SECRET".into();
        config.language = Some(LanguageConfig {
            model_dir: std::env::var("AURELIUS_TEST_LANGUAGE_MODEL_DIR")?,
            max_document_tokens: 2048,
            threads: 2,
        });
        config.fact_context = Some(FactContextConfig {
            schema: "adapter-target".into(),
            table: "canonical\"facts".into(),
            per_entity_limit: 16,
        });
        config.fact_quality = Some(serde_json::from_value(
            json!({"context_as_of":"2026-05-01T12:00:00Z"}),
        )?);
        let mut engine = FactEngine::with_storage(config, sink.clone(), stores)?;
        let request = engine.request("tenant-a")?;
        let source = FactSource {
            request,
            artifact_id: id,
        };
        let receipt = engine.process(source.clone()).await?;
        ensure!(
            receipt.new_claims == 2 && receipt.stored_windows == 2 && receipt.review_windows == 2,
            "Asana quality pipeline did not run through source metadata and typed validation"
        );
        engine.process(source).await?;
        ensure!(
            calls.load(Ordering::SeqCst) == before_calls + 2,
            "Asana replay repeated inference"
        );
    }
    Ok(())
}
