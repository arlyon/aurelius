//! Bounded, candidate-only atomic backfills over a frozen PostgreSQL manifest.
use crate::{
    atomic::{self, Registry},
    atomic_lab::SavedResponse,
    config::Config,
    contracts::{ByteRange, Chunk, DocumentKey},
    fact_lab::{LabRecord, Preparation, private_json},
    facts::{ChatExtractor, review_diagnostic, review_output, review_transport, review_usage},
    postgres::PostgresSink,
    provider_metadata::ProviderMetadataReader,
    run_budget::{RunBudget, RunStopped},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeSet, fs::DirBuilder, os::unix::fs::DirBuilderExt, path::PathBuf,
    time::Duration,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExcludedWindow {
    pub document: DocumentKey,
    pub content_hash: String,
    pub range: ByteRange,
}
#[derive(Deserialize)]
struct FrozenManifest {
    sources: Vec<crate::postgres::FactManifestEntry>,
    source_profiles: Vec<String>,
    seed: String,
}
pub struct Options {
    pub scope: String,
    pub output: PathBuf,
    pub seed: String,
    pub max_cents: u64,
    pub max_calls: u64,
    pub seconds: u64,
    pub max_output_tokens: u32,
    pub request_timeout_seconds: u64,
    pub excluded_windows: Vec<ExcludedWindow>,
    pub source_manifest: Option<PathBuf>,
}
#[derive(Default, Serialize)]
pub struct Summary {
    pub selected_documents: usize,
    pub selected_windows: i64,
    pub completed_windows: usize,
    pub already_present: usize,
    pub skipped_uncertain_windows: usize,
    pub input_review_windows: usize,
    pub failed_windows: usize,
    pub assertions: usize,
    pub review_items: usize,
    pub stop_reason: Option<&'static str>,
}

#[derive(Deserialize)]
struct SavedSource {
    document: crate::contracts::Document,
    ranges: Vec<ByteRange>,
}

/// Persist a saved, request-bound response without constructing a model client or budget.
pub async fn recover(
    config: &Config,
    registry: &Registry,
    scope: &str,
    responses: &std::path::Path,
    record: &str,
    output: &std::path::Path,
) -> Result<String> {
    let (source_id, index) = record.rsplit_once('-').context("invalid recovery record")?;
    ensure!(
        source_id.len() == 64
            && source_id.bytes().all(|b| b.is_ascii_hexdigit())
            && !index.is_empty()
            && index.bytes().all(|b| b.is_ascii_digit())
            && index.len() <= 5,
        "unsafe recovery record"
    );
    ensure!(!output.exists(), "recovery output exists; no overwrite");
    let index: usize = index.parse()?;
    let source: SavedSource = serde_json::from_slice(&crate::fact_lab::bounded(
        &responses.join(format!("source-{source_id}.json")),
        8 * 1024 * 1024,
    )?)?;
    let prepared: crate::fact_preparation::PreparedFactSource =
        serde_json::from_slice(&crate::fact_lab::bounded(
            &responses.join(format!("prepared-{record}.json")),
            8 * 1024 * 1024,
        )?)?;
    let saved: SavedResponse = serde_json::from_slice(&crate::fact_lab::bounded(
        &responses.join(format!("response-{record}.json")),
        1024 * 1024,
    )?)?;
    let original: serde_json::Value = serde_json::from_slice(&crate::fact_lab::bounded(
        &responses.join(format!("request-{record}.json")),
        2 * 1024 * 1024,
    )?)?;
    let tokens = original["max_output_tokens"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .context("recovery token bound missing")?;
    let sink = PostgresSink::connect(
        config.postgres.clone().context("Postgres required")?,
        config
            .embedding
            .as_ref()
            .context("embedding shape required")?
            .dimensions,
    )
    .await?;
    let input = sink.extraction_input(scope, source_id).await?;
    ensure!(
        source.document.key.scope == scope
            && input.document.key == source.document.key
            && input.document.content_hash() == source.document.content_hash()
            && input.document.metadata.uri == source.document.metadata.uri
            && input.ranges == source.ranges,
        "saved/current recovery source changed"
    );
    let chunk = Chunk {
        document: input.document,
        range: *source.ranges.get(index).context("recovery range missing")?,
    };
    let base = ChatExtractor::atomic_profile_for(
        config.chat.as_ref().context("chat required")?,
        &config.ontology,
    )?;
    let request = atomic::prepare_request(&chunk, &prepared, registry, &base, tokens)?;
    ensure!(
        original["binding"].as_str() == Some(&request.binding)
            && original["profile"].as_str() == Some(&request.profile),
        "saved recovery request changed"
    );
    let graph = crate::atomic_lab::replay(&saved, &request, registry)?;
    let extracted = crate::facts::AtomicExtraction {
        binding: saved.binding,
        profile: request.profile.clone(),
        raw_output: saved.body,
        usage: saved.usage,
        graph,
    };
    DirBuilder::new().mode(0o700).create(output)?;
    let artifact = sink
        .write_atomic_candidates(&chunk, &request, registry, &extracted)
        .await?;
    private_json(
        &output.join(format!("record-{record}.json")),
        &json!({"artifact_id":artifact,"profile":request.profile,"binding":request.binding,"graph":extracted.graph,"usage":extracted.usage,"recovered_without_generation":true}),
    )?;
    private_json(
        &output.join("summary.json"),
        &json!({"recovered_records":1,"model_calls":0,"budget_settlements":0,"artifact_id":artifact,"assertions":extracted.graph.assertions.len(),"review_items":extracted.graph.review_items.len()}),
    )?;
    Ok(artifact)
}

pub async fn run(config: &Config, registry: &Registry, options: &Options) -> Result<Summary> {
    registry.validate()?;
    ensure!(
        !options.scope.is_empty() && !options.seed.is_empty(),
        "scope and seed required"
    );
    ensure!(
        options.max_cents > 0
            && (1..=20000).contains(&options.max_calls)
            && (1..=259200).contains(&options.seconds)
            && (512..=16384).contains(&options.max_output_tokens)
            && (1..=600).contains(&options.request_timeout_seconds),
        "explicit bounded run limits required"
    );
    ensure!(
        !options.output.exists(),
        "run output exists; no overwrite or implicit resume"
    );
    let source_profiles = config
        .lanes
        .iter()
        .filter(|(_, l)| l.scope == options.scope)
        .map(|(name, _)| config.lane_request(name).map(|r| r.profile))
        .collect::<Result<BTreeSet<_>>>()?
        .into_iter()
        .collect::<Vec<_>>();
    let sink = PostgresSink::connect(
        config.postgres.clone().context("Postgres required")?,
        config
            .embedding
            .as_ref()
            .context("embedding shape required")?
            .dimensions,
    )
    .await?;
    let manifest = if let Some(path) = &options.source_manifest {
        let saved: FrozenManifest =
            serde_json::from_slice(&crate::fact_lab::bounded(path, 4 * 1024 * 1024)?)?;
        ensure!(
            saved.source_profiles == source_profiles && saved.seed == options.seed,
            "frozen source profiles/seed changed"
        );
        saved.sources
    } else {
        sink.fact_manifest(&options.scope, &source_profiles, &options.seed)
            .await?
    };
    ensure!(
        !manifest.is_empty()
            && manifest.len() <= 20000
            && manifest.iter().all(|e| e.windows > 0
                && !e.revision.is_empty()
                && e.artifact_id.len() == 64
                && e.artifact_id.bytes().all(|b| b.is_ascii_hexdigit()))
            && manifest
                .iter()
                .try_fold(0_i64, |n, e| n.checked_add(e.windows))
                .is_some_and(|n| n <= 20000),
        "invalid or oversized source manifest"
    );
    ensure!(
        options.excluded_windows.len() <= 1000
            && options
                .excluded_windows
                .iter()
                .all(|w| w.document.scope == options.scope
                    && w.content_hash.len() == 64
                    && w.content_hash.bytes().all(|b| b.is_ascii_hexdigit())
                    && manifest.iter().any(|e| e.artifact_id == w.document.source
                        && w.document.revision
                            == format!("pg-chunk-projection-v1/{}", e.revision))),
        "invalid uncertain-window exclusions"
    );
    let chat = config.chat.as_ref().context("chat required")?;
    let base_profile = ChatExtractor::atomic_profile_for(chat, &config.ontology)?;
    let identity = blake3::hash(&serde_json::to_vec(&(
        atomic::VERSION,
        &base_profile,
        registry.semantic_definition()?,
        &manifest,
        &options.seed,
        &options.excluded_windows,
        options.request_timeout_seconds,
    ))?)
    .to_hex()
    .to_string();
    let mut preparation = Preparation::new(config, true)?;
    preparation.warmup()?;
    let provider = ProviderMetadataReader::new(config)?;
    DirBuilder::new().mode(0o700).create(&options.output)?;
    private_json(
        &options.output.join("manifest.json"),
        &json!({"identity":identity,"sources":manifest,"source_profiles":source_profiles,"seed":options.seed,"max_cents":options.max_cents,"max_calls":options.max_calls,"seconds":options.seconds,"max_output_tokens":options.max_output_tokens,"request_timeout_seconds":options.request_timeout_seconds,"excluded_windows":options.excluded_windows,"contract_version":atomic::VERSION,"canonical_writes":0,"cost_basis":"$0.10/M input + $0.50/M output; cached full rate; gateway/infrastructure fees excluded"}),
    )?;
    let budget = RunBudget::open(
        &options.output.join("requests.jsonl"),
        identity.clone(),
        options.max_cents,
        options.max_calls,
        Duration::from_secs(options.seconds),
    )?;
    let extractor = ChatExtractor::new(chat, config.ontology.clone())?
        .with_request_timeout(Duration::from_secs(options.request_timeout_seconds))?
        .with_run_budget(budget.clone());
    let mut summary = Summary {
        selected_documents: manifest.len(),
        selected_windows: manifest.iter().map(|e| e.windows).sum(),
        ..Default::default()
    };
    let result: Result<()> = async {
        'documents: for entry in &manifest {
            if budget.check().is_err() { summary.stop_reason = Some("run_limit"); break; }
            preparation.clear_document_cache();
            let input = sink.extraction_input(&options.scope, &entry.artifact_id).await?;
            ensure!(input.document.key.revision == format!("pg-chunk-projection-v1/{}", entry.revision)
                && i64::try_from(input.ranges.len())? == entry.windows, "frozen source revision/window count changed");
            let properties = if let Some(reader) = &provider { reader.read(&input).await? } else { None };
            private_json(&options.output.join(format!("source-{}.json", entry.artifact_id)), &json!({"document":input.document,"ranges":input.ranges,"provider_properties":properties}))?;
            for exclusion in options.excluded_windows.iter().filter(|w| w.document.source == entry.artifact_id) {
                ensure!(exclusion.document == input.document.key && exclusion.content_hash == input.document.content_hash() && input.ranges.contains(&exclusion.range), "uncertain-window source/hash/range changed");
            }
            for (index, range) in input.ranges.iter().enumerate() {
                if options.excluded_windows.iter().any(|w| w.document == input.document.key && w.content_hash == input.document.content_hash() && w.range == *range) { summary.skipped_uncertain_windows += 1; continue; }
                if budget.check().is_err() { summary.stop_reason = Some("run_limit"); break 'documents; }
                let id = format!("{}-{index}", entry.artifact_id);
                let record = LabRecord { id: id.clone(), provider: format!("{:?}", input.kind), document:input.document.clone(), range:*range, prepared:None, provider_properties:properties.clone(), canonical_context:None };
                let chunk = Chunk { document:input.document.clone(), range:*range };
                let preflight = async {
                    let prepared = preparation.prepare(&record).await?;
                    let request = atomic::prepare_request(&chunk, &prepared, registry, &base_profile, options.max_output_tokens)?;
                    Ok::<_, anyhow::Error>((prepared, request))
                }.await;
                let (prepared, request) = match preflight {
                    Ok(value) => value,
                    Err(error) => {
                        private_json(&options.output.join(format!("input-review-{id}.json")), &json!({"stage":"source_preparation","diagnostic":review_diagnostic(&error)}))?;
                        sink.record_fact_status(&options.scope, &entry.artifact_id, &identity, Some(*range), "input_review").await?;
                        summary.input_review_windows += 1;
                        continue;
                    }
                };
                if sink.candidate_exists(&chunk, &request.profile).await? { summary.already_present += 1; continue; }
                private_json(&options.output.join(format!("request-{id}.json")), &request)?;
                private_json(&options.output.join(format!("prepared-{id}.json")), &prepared)?;
                match extractor.extract_atomic(&chunk, &prepared, registry, options.max_output_tokens).await {
                    Ok(extracted) => {
                        private_json(&options.output.join(format!("response-{id}.json")), &SavedResponse { contract_version:atomic::VERSION.into(), binding:extracted.binding.clone(), body:extracted.raw_output.clone(), usage:extracted.usage.clone() })?;
                        let artifact_id = sink.write_atomic_candidates(&chunk, &request, registry, &extracted).await?;
                        private_json(&options.output.join(format!("record-{id}.json")), &json!({"artifact_id":artifact_id,"profile":request.profile,"binding":request.binding,"graph":extracted.graph,"usage":extracted.usage}))?;
                        summary.assertions += extracted.graph.assertions.len();
                        summary.review_items += extracted.graph.review_items.len();
                        summary.completed_windows += 1;
                    }
                    Err(error) => {
                        if error.downcast_ref::<RunStopped>().is_some() { summary.stop_reason=Some("run_limit"); break 'documents; }
                        summary.failed_windows += 1;
                        private_json(&options.output.join(format!("failure-{id}.json")), &json!({"diagnostic":review_diagnostic(&error),"transport":review_transport(&error),"usage":review_usage(&error),"raw_output":review_output(&error)}))?;
                        if budget.snapshot()?.halted || review_usage(&error).and_then(|u| u.input_tokens.zip(u.output_tokens)).is_none() {
                            summary.stop_reason=Some("provider_or_unsettled_usage"); break 'documents;
                        }
                        sink.record_fact_review(&options.scope, &entry.artifact_id, &identity, *range, review_diagnostic(&error)).await?;
                    }
                }
            }
        }
        Ok(())
    }.await;
    if result.is_err() {
        summary.stop_reason = Some("local_or_database_failure");
    }
    private_json(&options.output.join("budget.json"), &budget.snapshot()?)?;
    private_json(&options.output.join("summary.json"), &summary)?;
    result?;
    Ok(summary)
}
