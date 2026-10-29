use anyhow::{Context, Result, ensure};
use aurelius_lanes::{
    atomic::{self, AnchorKind, ObjectSpec, PredicateSpec, Registry},
    contracts::Chunk,
    fact_context::FactContext,
    fact_preparation::PreparedFactSource,
    facts::AtomicExtraction,
    language::LanguageAnalysis,
    postgres::PostgresSink,
};
use serde_json::{Value, json};
use sqlx::PgPool;
use std::collections::{BTreeMap, BTreeSet};

async fn exercise_open_claims(
    sink: &PostgresSink,
    admin: &PgPool,
    chunk: &Chunk,
    prepared: &PreparedFactSource,
) -> Result<()> {
    use aurelius_lanes::{
        facts::OpenClaimsExtraction, open_claims_prototype as open, quality::QualityConfig,
    };
    let base = "c".repeat(64);
    let policy: QualityConfig = serde_json::from_value(json!({"format":"open_claims_v3",
        "context_as_of":"2026-01-01T00:00:00Z","max_output_tokens":4096}))?;
    let request = open::prepare_compact(chunk, prepared, &base, 4096)?;
    let unit = request
        .units
        .iter()
        .find(|(_, s)| s.quote.contains("$1500/month"))
        .context("open budget unit")?
        .0;
    let raw=json!({"claims":[{
        "subject":{"source":{"unit_ref":unit,"exact_quote":"Monthly budget","context_quote":null},"type_hint":null},
        "relation":"is", "object":{"source":{"unit_ref":unit,"exact_quote":"$1500/month","context_quote":null},"type_hint":null},
        "qualification":{"mode":"asserted","polarity":"positive","reported_by":[],"conditions":[],"temporal":[]},
        "evidence":[unit]}, {"unexpected":"rejected sibling"}],"unresolved":[]}).to_string();
    let graph = open::validate_partial(&raw, chunk, &request)?;
    ensure!(graph["claims"].as_array().context("claims")?.len() == 1);
    let mut extracted = OpenClaimsExtraction {
        request,
        graph,
        usage: None,
        raw_output: raw,
        profile: policy.profile(&base, &prepared.preparation_profile)?,
    };
    let id = sink
        .write_open_claims_candidates(chunk, &extracted, prepared, &base, &policy)
        .await?;
    ensure!(
        sink.write_open_claims_candidates(chunk, &extracted, prepared, &base, &policy)
            .await?
            == id
    );
    ensure!(sink.candidate_exists(chunk, &extracted.profile).await?);
    let metadata: Value =
        sqlx::query_scalar("SELECT metadata FROM \"adapter-target\".\"source-docs\" WHERE id=$1")
            .bind(&id)
            .fetch_one(admin)
            .await?;
    ensure!(
        metadata["profile"] == extracted.profile
            && metadata["protocol_profile"] == extracted.request.profile
            && metadata["validation_policy"] == open::CLAIM_LOCAL_POLICY
            && metadata["candidate_graph"]["claims"]
                .as_array()
                .context("stored claims")?
                .len()
                == 1
            && metadata["candidate_graph"]["rejected"]["claims"][0]["source_index"] == 1
            && metadata["candidate_graph"]["canonical_writes"] == 0
            && metadata["generation"] == extracted.raw_output
    );
    let mut foreign = chunk.clone();
    foreign.document.key.scope = "tenant-b".into();
    ensure!(
        sink.write_open_claims_candidates(&foreign, &extracted, prepared, &base, &policy)
            .await
            .is_err()
    );
    ensure!(
        sink.write_open_claims_candidates(chunk, &extracted, prepared, &"d".repeat(64), &policy)
            .await
            .is_err()
    );
    extracted.graph["canonical_writes"] = json!(1);
    ensure!(
        sink.write_open_claims_candidates(chunk, &extracted, prepared, &base, &policy)
            .await
            .is_err()
    );
    Ok(())
}

async fn exercise_atomic_runtime(
    sink: &PostgresSink,
    admin: &PgPool,
    chunk: &Chunk,
    prepared: &PreparedFactSource,
) -> Result<()> {
    let policy: aurelius_lanes::quality::QualityConfig = serde_json::from_value(json!({
        "format":"atomic_spo_v3", "context_as_of":"2026-01-01T00:00:00Z", "max_output_tokens":2048
    }))?;
    let registry = policy.atomic_registry()?;
    let base = "e".repeat(64);
    let request = atomic::prepare_request(chunk, prepared, &registry, &base, 2048)?;
    let start = chunk
        .document
        .text
        .find("Monthly budget")
        .context("budget")?;
    let raw = json!({
        "local_entities":[{"id":"plan","entity_type":"budget_plan","identity_span_ref":"span:0-6","action_kind":null}],
        "assertions":[{"subject":{"ref":"plan"},"predicate":"monthly_budget",
            "object":{"kind":"money","amount":"1500","currency_symbol":"$","period":"month"},
            "qualification":{"mode":"required","polarity":"positive","conditions":[],"attribution":[],"validity":{"from":null,"to":null}},
            "evidence":[format!("span:{start}-{}",chunk.document.text.len())]}],
        "review_items":[]
    }).to_string();
    let mut extracted = AtomicExtraction {
        graph: atomic::validate(&raw, &request.catalog, &registry)?,
        raw_output: raw,
        profile: request.profile.clone(),
        binding: request.binding.clone(),
        usage: None,
    };
    ensure!(extracted.graph.assertions.len() == 1);
    let profile = policy.profile(&base, &prepared.preparation_profile)?;
    ensure!(profile != request.profile);
    let id = sink
        .write_runtime_atomic_candidates(chunk, &extracted, prepared, &base, &policy)
        .await?;
    ensure!(
        sink.write_runtime_atomic_candidates(chunk, &extracted, prepared, &base, &policy)
            .await?
            == id
    );
    ensure!(sink.candidate_exists(chunk, &profile).await?);
    let metadata: Value =
        sqlx::query_scalar("SELECT metadata FROM \"adapter-target\".\"source-docs\" WHERE id=$1")
            .bind(&id)
            .fetch_one(admin)
            .await?;
    ensure!(metadata["profile"] == profile && metadata["protocol_profile"] == request.profile);
    ensure!(metadata["candidate_graph"]["contract_version"] == atomic::VERSION);
    ensure!(
        metadata["candidate_graph"]["assertions"][0]["assertion"]["object"]["amount"] == "1500"
    );
    ensure!(metadata["generation"] == extracted.raw_output);
    let mut other = chunk.clone();
    other.document.key.scope = "tenant-b".into();
    ensure!(
        sink.write_runtime_atomic_candidates(&other, &extracted, prepared, &base, &policy)
            .await
            .is_err()
    );
    ensure!(
        sink.write_runtime_atomic_candidates(chunk, &extracted, prepared, &"f".repeat(64), &policy)
            .await
            .is_err()
    );
    extracted.graph.assertions[0].rendered.push_str(" tampered");
    ensure!(
        sink.write_runtime_atomic_candidates(chunk, &extracted, prepared, &base, &policy)
            .await
            .is_err()
    );
    Ok(())
}

pub async fn exercise(sink: &PostgresSink, admin: &PgPool, config_text: &str) -> Result<()> {
    let mut indexed = super::document("atomic-postgres");
    let text = "Report\nMonthly budget $1500/month.";
    indexed.document.sections[0].text = text.into();
    for chunk in &mut indexed.chunks {
        chunk.text = text.into();
        chunk.citations[0].byte_end = text.len();
    }
    indexed.content_hash = indexed.document.content_hash()?;
    let source = sink.write(&indexed).await?.artifact_id;
    let entries = sink
        .fact_manifest("tenant-a", &[indexed.profile.clone()], "synthetic-seed")
        .await?;
    ensure!(
        entries
            .iter()
            .any(|e| e.artifact_id == source && e.windows == 2)
    );
    ensure!(
        sink.fact_manifest("tenant-b", &[indexed.profile.clone()], "synthetic-seed")
            .await?
            .is_empty()
    );
    ensure!(
        sink.fact_manifest("tenant-a", &["not-source-profile".into()], "synthetic-seed")
            .await?
            .is_empty()
    );
    let input = sink.extraction_input("tenant-a", &source).await?;
    let chunk = Chunk {
        document: input.document,
        range: input.ranges[0],
    };
    let language = LanguageAnalysis {
        document: chunk.document.key.clone(),
        content_hash: chunk.document.content_hash(),
        profile: "a".repeat(64),
        mentions: vec![],
        coreference_hypotheses: vec![],
        sentence_spans: vec![],
        verb_spans: vec![],
        number_spans: vec![],
    };
    let prepared = PreparedFactSource::from_analysis(
        &chunk.document,
        language,
        vec![],
        FactContext {
            organization_id: "tenant-a".into(),
            entities: vec![],
        },
        "2026-01-01T00:00:00Z".parse()?,
        "b".repeat(64),
    )?;
    exercise_open_claims(sink, admin, &chunk, &prepared).await?;
    exercise_atomic_runtime(sink, admin, &chunk, &prepared).await?;
    let registry = Registry {
        entity_anchors: BTreeMap::from([("report".into(), AnchorKind::Facet)]),
        predicates: BTreeMap::from([(
            "monthly_budget".into(),
            PredicateSpec {
                description: "Synthetic monthly budget".into(),
                subject_types: BTreeSet::from(["report".into()]),
                object: ObjectSpec::Money,
                render: "{subject} has monthly budget {object}".into(),
                allowed_modes: vec![],
            },
        )]),
        ..Default::default()
    };
    let request = atomic::prepare_request(&chunk, &prepared, &registry, &"c".repeat(64), 2048)?;
    let start = chunk
        .document
        .text
        .find("Monthly budget")
        .context("budget")?;
    let body = json!({"local_entities":[{"id":"report_1","entity_type":"report","identity_span_ref":"span:0-6","action_kind":null}],"assertions":[{"subject":{"ref":"report_1"},"predicate":"monthly_budget","object":{"kind":"money","amount":"1500","currency_symbol":"$","period":"month"},"qualification":{"mode":"required","polarity":"positive","conditions":[],"attribution":[],"validity":{"from":null,"to":null}},"evidence":[format!("span:{start}-{}",text.len())]}],"review_items":[]}).to_string();
    let graph = atomic::validate(&body, &request.catalog, &registry)?;
    ensure!(graph.assertions.len() == 1, "synthetic SPO not accepted");
    let mut extracted = AtomicExtraction {
        binding: request.binding.clone(),
        profile: request.profile.clone(),
        raw_output: body,
        graph,
        usage: None,
    };
    let id = sink
        .write_atomic_candidates(&chunk, &request, &registry, &extracted)
        .await?;
    ensure!(
        sink.write_atomic_candidates(&chunk, &request, &registry, &extracted)
            .await?
            == id
    );
    let metadata: Value =
        sqlx::query_scalar("SELECT metadata FROM \"adapter-target\".\"source-docs\" WHERE id=$1")
            .bind(&id)
            .fetch_one(admin)
            .await?;
    ensure!(
        metadata["candidate_graph"]["contract_version"] == atomic::VERSION
            && metadata["candidate_graph"]["assertions"]
                .as_array()
                .context("assertions")?
                .len()
                == 1
    );
    let mut context_registry = registry.clone();
    context_registry
        .entity_anchors
        .insert("source_record".into(), AnchorKind::Record);
    let later = Chunk {
        document: chunk.document.clone(),
        range: input.ranges[1],
    };
    let context_request =
        atomic::prepare_request(&later, &prepared, &context_registry, &"c".repeat(64), 2048)?;
    let context_body = json!({"local_entities":[{"id":"record_1","entity_type":"source_record","identity_span_ref":"span:0-6","action_kind":null}],"assertions":[],"review_items":[]}).to_string();
    let context_graph =
        atomic::validate(&context_body, &context_request.catalog, &context_registry)?;
    ensure!(
        context_graph.entities["record_1"].anchor.range.end < later.range.start,
        "context fixture must cross the chunk boundary"
    );
    let mut context_extracted = AtomicExtraction {
        binding: context_request.binding.clone(),
        profile: context_request.profile.clone(),
        raw_output: context_body,
        graph: context_graph,
        usage: None,
    };
    sink.write_atomic_candidates(
        &later,
        &context_request,
        &context_registry,
        &context_extracted,
    )
    .await?;
    context_extracted.raw_output = context_extracted
        .raw_output
        .replace("span:0-6", "span:7-33");
    ensure!(
        sink.write_atomic_candidates(
            &later,
            &context_request,
            &context_registry,
            &context_extracted
        )
        .await
        .is_err(),
        "unissued cross-window marker accepted"
    );
    extracted.graph.rendering_profile = "tampered".into();
    ensure!(
        sink.write_atomic_candidates(&chunk, &request, &registry, &extracted)
            .await
            .is_err()
    );
    extracted.graph = atomic::validate(&extracted.raw_output, &request.catalog, &registry)?;
    extracted.binding = "wrong".into();
    ensure!(
        sink.write_atomic_candidates(&chunk, &request, &registry, &extracted)
            .await
            .is_err()
    );
    extracted.binding = request.binding.clone();
    let mut foreign = chunk.clone();
    foreign.document.key.scope = "tenant-b".into();
    ensure!(
        sink.write_atomic_candidates(&foreign, &request, &registry, &extracted)
            .await
            .is_err()
    );
    sqlx::query(
        "UPDATE \"adapter-target\".\"source-docs\" SET revision_id='changed-source' WHERE id=$1",
    )
    .bind(&source)
    .execute(admin)
    .await?;
    ensure!(
        sink.write_atomic_candidates(&chunk, &request, &registry, &extracted)
            .await
            .is_err()
    );
    let mut config = aurelius_lanes::config::Config::parse(config_text)?;
    config.postgres.as_mut().context("postgres")?.url_env = "PG_ADAPTER_TEST_URL".into();
    config
        .chat
        .as_mut()
        .context("chat")?
        .gateway
        .as_mut()
        .context("gateway")?
        .token_env = "SOURCE_API_TEST_CLIENT_SECRET".into();
    config.language = Some(aurelius_lanes::language::LanguageConfig {
        model_dir: std::env::var("AURELIUS_TEST_LANGUAGE_MODEL_DIR")?,
        max_document_tokens: 2048,
        threads: 2,
    });
    config.fact_quality = Some(serde_json::from_value(
        json!({"context_as_of":"2026-01-01T00:00:00Z"}),
    )?);
    let original_revision = chunk
        .document
        .key
        .revision
        .strip_prefix("pg-chunk-projection-v1/")
        .context("original projection")?;
    sqlx::query("UPDATE \"adapter-target\".\"source-docs\" SET revision_id=$1 WHERE id=$2")
        .bind(original_revision)
        .bind(&source)
        .execute(admin)
        .await?;
    let recovery_input =
        std::env::temp_dir().join(format!("atomic-pg-recover-{}", std::process::id()));
    {
        use std::{
            io::Write,
            os::unix::fs::{DirBuilderExt, OpenOptionsExt},
        };
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&recovery_input)?;
        let mut recovery_config = config.clone();
        recovery_config
            .chat
            .as_mut()
            .context("chat")?
            .gateway
            .as_mut()
            .context("gateway")?
            .token_env = "INTENTIONALLY_MISSING_RECOVERY_TOKEN".into();
        let base = aurelius_lanes::facts::ChatExtractor::atomic_profile_for(
            recovery_config.chat.as_ref().context("chat")?,
            &config.ontology,
        )?;
        let request = atomic::prepare_request(&later, &prepared, &context_registry, &base, 2048)?;
        let body = json!({"local_entities":[{"id":"record_1","entity_type":"source_record","identity_span_ref":"span:0-6","action_kind":null}],"assertions":[],"review_items":[]}).to_string();
        let record = format!("{source}-1");
        for (name, value) in [
            (
                format!("source-{source}.json"),
                json!({"document":chunk.document,"ranges":input.ranges}),
            ),
            (
                format!("prepared-{record}.json"),
                serde_json::to_value(&prepared)?,
            ),
            (
                format!("request-{record}.json"),
                serde_json::to_value(&request)?,
            ),
            (
                format!("response-{record}.json"),
                json!({"contract_version":atomic::VERSION,"binding":request.binding,"body":body,"usage":null}),
            ),
        ] {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(recovery_input.join(name))?;
            file.write_all(&serde_json::to_vec(&value)?)?;
        }
        let recovered = aurelius_lanes::atomic_pg::recover(
            &recovery_config,
            &context_registry,
            "tenant-a",
            &recovery_input,
            &record,
            &recovery_input.join("output"),
        )
        .await?;
        let again = aurelius_lanes::atomic_pg::recover(
            &recovery_config,
            &context_registry,
            "tenant-a",
            &recovery_input,
            &record,
            &recovery_input.join("output-again"),
        )
        .await?;
        ensure!(recovered == again, "zero-call recovery not idempotent");
        let summary: Value =
            serde_json::from_slice(&std::fs::read(recovery_input.join("output/summary.json"))?)?;
        ensure!(summary["model_calls"] == 0 && summary["budget_settlements"] == 0);
    }
    std::fs::remove_dir_all(recovery_input)?;
    let output = std::env::temp_dir().join(format!("atomic-pg-run-{}", std::process::id()));
    let options = aurelius_lanes::atomic_pg::Options {
        scope: "tenant-a".into(),
        output: output.clone(),
        seed: "synthetic-run".into(),
        max_cents: 5,
        max_calls: 1,
        seconds: 300,
        max_output_tokens: 2048,
        request_timeout_seconds: 120,
        excluded_windows: vec![],
        source_manifest: None,
    };
    let summary = aurelius_lanes::atomic_pg::run(&config, &registry, &options).await?;
    ensure!(
        summary.completed_windows == 1 && summary.stop_reason == Some("run_limit"),
        "bounded PG runner did not persist exactly one native result"
    );
    ensure!(
        aurelius_lanes::atomic_pg::run(&config, &registry, &options)
            .await
            .is_err(),
        "implicit resume permitted"
    );
    let budget: Value = serde_json::from_slice(&std::fs::read(output.join("budget.json"))?)?;
    ensure!(budget["calls"] == 1 && budget["halted"] == false);
    let saved: Value = serde_json::from_slice(&std::fs::read(output.join("manifest.json"))?)?;
    let mut windows = Vec::new();
    for source in saved["sources"]
        .as_array()
        .context("sources")?
        .iter()
        .take(2)
    {
        let input = sink
            .extraction_input(
                "tenant-a",
                source["artifact_id"].as_str().context("source id")?,
            )
            .await?;
        for range in input.ranges {
            windows.push(aurelius_lanes::atomic_pg::ExcludedWindow {
                document: input.document.key.clone(),
                content_hash: input.document.content_hash(),
                range,
            });
        }
    }
    let second = windows.get(1).context("second synthetic window")?.clone();
    let output2 = output.with_extension("continuation");
    let options2 = aurelius_lanes::atomic_pg::Options {
        output: output2.clone(),
        excluded_windows: vec![second],
        source_manifest: Some(output.join("manifest.json")),
        ..options
    };
    let continued = aurelius_lanes::atomic_pg::run(&config, &registry, &options2).await?;
    ensure!(
        continued.already_present == 1
            && continued.skipped_uncertain_windows == 1
            && continued.completed_windows == 1,
        "continuation repeated or failed to exclude a known window"
    );
    std::fs::remove_dir_all(output2)?;
    std::fs::remove_dir_all(output)?;
    Ok(())
}
