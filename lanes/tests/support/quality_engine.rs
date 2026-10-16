use anyhow::{Context, Result, ensure};
use aurelius_lanes::{
    config::Config,
    contracts::Chunk,
    enrichment::{FactActivities, FactEngine, FactSource, FactStatus},
    fact_context::FactContextConfig,
    fact_preparation::FactPreparer,
    language::LanguageConfig,
    postgres::PostgresSink,
    quality::{QUALITY_VERSION, QualityConfig, QualityGraph},
};
use serde_json::{Value, json};
use sqlx::PgPool;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub async fn exercise(
    text: &str,
    sink: &PostgresSink,
    admin: &PgPool,
    calls: Arc<AtomicUsize>,
) -> Result<()> {
    let mut config = Config::parse(text)?;
    config.postgres.as_mut().context("postgres")?.url_env =
        "INTENTIONALLY_MISSING_DATABASE_URL".into();
    config
        .chat
        .as_mut()
        .context("chat")?
        .gateway
        .as_mut()
        .context("gateway")?
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
    let embedding_profile_before = config.lane_request("fixture")?.profile;
    ensure!(
        FactEngine::requests(&config).is_err(),
        "partial quality configuration silently used legacy extraction"
    );
    let mut legacy_config = config.clone();
    legacy_config.language = None;
    legacy_config.fact_context = None;
    let legacy_profile = FactEngine::requests(&legacy_config)?[0].profile.clone();
    config.fact_quality = Some(serde_json::from_value::<QualityConfig>(
        json!({"context_as_of":"2026-05-01T12:00:00Z"}),
    )?);
    config.validate()?;
    ensure!(
        config.lane_request("fixture")?.profile == embedding_profile_before,
        "quality configuration changed embedding profile"
    );
    ensure!(
        FactEngine::requests(&config)?[0].profile != legacy_profile,
        "typed extraction reused the legacy completion namespace"
    );
    let mut indexed = super::document("typed-facts");
    indexed.profile = config.lane_request("fixture")?.profile;
    indexed.document.connection_id = "fixture-connection".into();
    indexed.document.sections[0].text = "Alice manages Acme.".into();
    for chunk in &mut indexed.chunks {
        chunk.text = "Alice manages Acme.".into();
        chunk.citations[0].byte_end = chunk.text.len();
    }
    indexed.content_hash = indexed.document.content_hash()?;
    let id = sink.write(&indexed).await?.artifact_id;
    let mut engine = FactEngine::with_sink(config.clone(), sink.clone())?;
    let request = engine.request("tenant-a")?;
    ensure!(
        FactEngine::requests(&config)? == vec![request.clone()],
        "profile planning opened PG or disagreed with the shared-pool runtime"
    );
    let source = FactSource {
        request: request.clone(),
        artifact_id: id.clone(),
    };
    let before = calls.load(Ordering::SeqCst);
    let receipt = engine.process(source.clone()).await?;
    ensure!(
        receipt.status == FactStatus::Complete
            && receipt.stored_windows == 2
            && receipt.review_windows == 2
            && receipt.new_claims == 2,
        "FactEngine did not use typed candidates/rejections"
    );
    let again = engine.process(source).await?;
    ensure!(
        again.status == FactStatus::AlreadyComplete && calls.load(Ordering::SeqCst) == before + 2,
        "typed replay repeated inference"
    );
    let values:Vec<Value>=sqlx::query_scalar("SELECT metadata->'candidate_graph' FROM \"adapter-target\".\"source-docs\" WHERE metadata->>'source_artifact_id'=$1 AND metadata->>'profile'=$2 AND metadata->'candidate_graph'->>'contract_version'=$3 ORDER BY metadata->'source_range'->>'start'").bind(&id).bind(&request.profile).bind(QUALITY_VERSION).fetch_all(admin).await?;
    ensure!(values.len() == 2);
    let graph: QualityGraph = serde_json::from_value(values[0].clone())?;
    ensure!(
        graph.candidates.len() == 1 && graph.rejections.len() == 1 && graph.mentions.len() == 2
    );
    let chunks:i64=sqlx::query_scalar("SELECT count(*) FROM \"adapter-target\".\"chunk-rows\" c JOIN \"adapter-target\".\"source-docs\" a ON c.artifact_source_id=a.id WHERE a.metadata->>'profile'=$1 AND a.metadata->>'status'='candidate'").bind(&request.profile).fetch_one(admin).await?;
    ensure!(chunks == 0, "typed facts created fake embedding chunks");
    let input = sink.extraction_input("tenant-a", &id).await?;
    let prepared = FactPreparer::with_sink(&config, sink.clone())?
        .prepare(
            input.document.clone(),
            config
                .fact_quality
                .as_ref()
                .context("quality")?
                .context_as_of,
        )
        .await?;
    let chunk = Chunk {
        document: input.document,
        range: input.ranges[0],
    };
    let diagnostic_id = sink
        .record_fact_review(
            "tenant-a",
            &id,
            &"d".repeat(64),
            chunk.range,
            aurelius_lanes::facts::ReviewDiagnostic::TypedJson,
        )
        .await?;
    let diagnostic: String = sqlx::query_scalar(
        "SELECT metadata->>'diagnostic' FROM \"adapter-target\".\"source-docs\" WHERE id=$1",
    )
    .bind(&diagnostic_id)
    .fetch_one(admin)
    .await?;
    ensure!(
        diagnostic == "typed_json",
        "structured review cause was lost from durable receipts"
    );
    let mut aligned_policy = config.fact_quality.as_ref().context("quality")?.clone();
    aligned_policy.align_whitespace = true;
    let mut spaced = graph.candidates[0].candidate.clone();
    spaced.quote = spaced.quote.replace(' ', "\r\n");
    let aligned = aurelius_lanes::quality::parse_graph(
        &json!({"facts":[spaced]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &aligned_policy,
        &"e".repeat(64),
    )?;
    ensure!(aligned.candidates.len() == 1 && aligned.candidates[0].proposed.is_some());
    sink.write_quality_candidates(
        &chunk,
        &aligned,
        &prepared,
        &config.ontology,
        &aligned_policy,
    )
    .await?;
    let mut forged_alignment = aligned.clone();
    forged_alignment.candidates[0]
        .proposed
        .as_mut()
        .expect("original proposal")
        .quote = "fabricated source".into();
    ensure!(
        sink.write_quality_candidates(
            &chunk,
            &forged_alignment,
            &prepared,
            &config.ontology,
            &aligned_policy
        )
        .await
        .is_err(),
        "forged alignment provenance accepted"
    );
    let mut changed_layout = graph.clone();
    changed_layout.source_layout_hash = "f".repeat(64);
    ensure!(
        sink.write_quality_candidates(
            &chunk,
            &changed_layout,
            &prepared,
            &config.ontology,
            config.fact_quality.as_ref().context("quality")?
        )
        .await
        .is_err(),
        "storage accepted a changed source-layout snapshot"
    );
    let mut tampered = graph;
    tampered.candidates[0].evidence[0].quote = "fabricated source".into();
    ensure!(
        sink.write_quality_candidates(
            &chunk,
            &tampered,
            &prepared,
            &config.ontology,
            config.fact_quality.as_ref().context("quality")?
        )
        .await
        .is_err(),
        "storage accepted forged typed evidence"
    );
    Ok(())
}
