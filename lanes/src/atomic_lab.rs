//! File-only atomic candidates. Live calls need a new private durable budget;
//! replay binds the exact source, preparation, semantics and native request.
use crate::{
    atomic::{self, Registry},
    config::Config,
    contracts::Chunk,
    fact_lab::{self, LabRecord, Mode, Options, Preparation, bounded, private_json},
    facts::{ChatExtractor, ExtractionUsage, review_diagnostic, review_output, review_usage},
    run_budget::RunBudget,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{fs::DirBuilder, os::unix::fs::DirBuilderExt, path::Path, time::Duration};
#[derive(Clone, Copy)]
pub struct Limits {
    pub max_cents: u64,
    pub seconds: u64,
    pub max_output_tokens: u32,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SavedResponse {
    pub contract_version: String,
    pub binding: String,
    pub body: String,
    pub usage: Option<ExtractionUsage>,
}
#[derive(Default, Serialize)]
pub struct Summary {
    pub selected_records: usize,
    pub completed_records: usize,
    pub failed_records: usize,
    pub not_run: usize,
    pub attempts: u64,
    pub assertions: usize,
    pub review_items: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub usage_complete: bool,
    pub stopped: bool,
}
pub fn load_registry(path: &Path) -> Result<Registry> {
    let registry: Registry = serde_json::from_slice(&bounded(path, 128 * 1024)?)
        .map_err(|_| anyhow::anyhow!("invalid atomic registry"))?;
    registry.validate()?;
    Ok(registry)
}
pub fn replay(
    saved: &SavedResponse,
    request: &atomic::PreparedRequest,
    registry: &Registry,
) -> Result<atomic::AtomicGraph> {
    ensure!(
        saved.contract_version == atomic::VERSION,
        "unsupported atomic artifact version"
    );
    ensure!(
        saved.binding == request.binding,
        "atomic replay source/request binding mismatch"
    );
    atomic::validate(&saved.body, &request.catalog, registry)
}
pub async fn run(
    config: &Config,
    records: &[LabRecord],
    registry: &Registry,
    options: &Options,
    limits: Limits,
) -> Result<Summary> {
    registry.validate()?;
    ensure!(
        !options.batch_adjacent,
        "atomic lab requires one source chunk per call"
    );
    ensure!(
        (512..=16384).contains(&limits.max_output_tokens),
        "invalid atomic output limit"
    );
    ensure!(
        !options.output.exists(),
        "atomic output exists; no overwrites or implicit resume"
    );
    let live = matches!(options.mode, Mode::Live);
    ensure!(
        !live
            || (limits.max_cents > 0
                && (1..=1000).contains(&options.max_calls)
                && (1..=259200).contains(&limits.seconds)),
        "live atomic runs require explicit spend/call/deadline caps"
    );
    let selected = fact_lab::units(records, &options.selected, false)?;
    let mut ids = std::collections::BTreeSet::new();
    ensure!(
        selected.len() <= 1000
            && selected.iter().all(|u| !u.id.is_empty()
                && u.id.len() <= 80
                && u.id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
                && ids.insert(u.id.clone())),
        "unsafe or duplicate atomic record ID"
    );
    let chat = config.chat.as_ref().context("chat configuration missing")?;
    let base_profile = ChatExtractor::atomic_profile_for(chat, &config.ontology)?;
    let identity = blake3::hash(&serde_json::to_vec(&(
        atomic::VERSION,
        &base_profile,
        registry.semantic_definition()?,
        &options.selected,
    ))?)
    .to_hex()
    .to_string();
    let mut preparation = Preparation::new(config, options.fresh)?;
    DirBuilder::new().mode(0o700).create(&options.output)?;
    let budget = if live {
        Some(RunBudget::open(
            &options.output.join("requests.jsonl"),
            identity,
            limits.max_cents,
            options.max_calls as u64,
            Duration::from_secs(limits.seconds),
        )?)
    } else {
        None
    };
    let extractor = if let Some(budget) = &budget {
        Some(ChatExtractor::new(chat, config.ontology.clone())?.with_run_budget(budget.clone()))
    } else {
        None
    };
    private_json(
        &options.output.join("manifest.json"),
        &json!({"contract_version":atomic::VERSION,"semantic_definition":registry.semantic_definition()?,"live":live,"canonical_writes":0,"max_calls":options.max_calls,"max_cents":limits.max_cents,"duration_seconds":limits.seconds,"max_output_tokens":limits.max_output_tokens,"semantic_accuracy":"not measured by replay or schema validity"}),
    )?;
    let mut summary = Summary {
        selected_records: selected.len(),
        usage_complete: true,
        ..Default::default()
    };
    let mut prepared_records = Vec::new();
    let mut report = String::from(
        "# Atomic candidate extraction\n\nUnreviewed candidates, not canonical truths. Schema/replay checks do not measure model quality.\n\n| Subject | Predicate | Object | Qualified assertion | Evidence |\n|---|---|---|---|---|\n",
    );
    for unit in selected {
        if summary.stopped {
            summary.not_run += 1;
            continue;
        }
        let attempt=async {
            let prepared=preparation.prepare(&unit.record).await?;
            let chunk=Chunk{document:unit.record.document.clone(),range:unit.record.range};
            let request=atomic::prepare_request(&chunk,&prepared,registry,&base_profile,limits.max_output_tokens)?;
            let mut cached=unit.record.clone();cached.prepared=Some(prepared.clone());prepared_records.push(cached);
            private_json(&options.output.join(format!("request-{}.json",unit.id)),&request)?;
            let (saved,graph)=if let Some(extractor)=&extractor {
                if budget.as_ref().context("live budget missing")?.check().is_err(){summary.not_run+=1;summary.stopped=true;return Ok(());}
                let extracted=extractor.extract_atomic(&chunk,&prepared,registry,limits.max_output_tokens).await?;
                ensure!(extracted.binding==request.binding,"live atomic request changed");
                (SavedResponse{contract_version:atomic::VERSION.into(),binding:request.binding.clone(),body:extracted.raw_output,usage:extracted.usage},extracted.graph)
            }else{
                let directory=match &options.mode{Mode::Replay(directory)=>directory,Mode::Live=>unreachable!()};
                let saved=serde_json::from_slice::<SavedResponse>(&bounded(&directory.join(format!("response-{}.json",unit.id)),1024*1024)?).map_err(|_|anyhow::anyhow!("unsupported or malformed atomic replay artifact"))?;
                let graph=replay(&saved,&request,registry)?;(saved,graph)
            };
            // Protocol failures are retained separately in private failure artifacts.
            private_json(&options.output.join(format!("response-{}.json",unit.id)),&saved)?;
            if live {if let Some(usage)=&saved.usage {summary.input_tokens+=usage.input_tokens.unwrap_or(0);summary.output_tokens+=usage.output_tokens.unwrap_or(0);summary.usage_complete&=usage.input_tokens.is_some()&&usage.output_tokens.is_some();}else{summary.usage_complete=false;}}
            for a in &graph.assertions {
                let subject=&graph.entities[&a.assertion.subject.id];
                let subject=table_cell(&subject.anchor.quote);
                let object=if let atomic::Object::Entity{id}=&a.assertion.object{table_cell(&graph.entities[id].anchor.quote)}else{table_cell(&serde_json::to_string(&a.assertion.object)?)};
                let citations=a.evidence.iter().map(|s|format!("span:{}-{}",s.range.start,s.range.end)).collect::<Vec<_>>().join(", ");
                report.push_str(&format!("| {subject} | {} | {object} | {} | {citations} |\n",a.assertion.predicate,a.rendered.replace(['\n','\r']," ")));
            }
            summary.assertions+=graph.assertions.len();summary.review_items+=graph.review_items.len();summary.completed_records+=1;
            private_json(&options.output.join(format!("record-{}.json",unit.id)),&json!({"contract_version":atomic::VERSION,"profile":request.profile,"binding":request.binding,"graph":graph}))?;Ok::<(),anyhow::Error>(())
        }.await;
        if let Err(error) = attempt {
            summary.failed_records += 1;
            summary.stopped = true;
            if live {
                if let Some(usage) = review_usage(&error) {
                    summary.input_tokens += usage.input_tokens.unwrap_or(0);
                    summary.output_tokens += usage.output_tokens.unwrap_or(0);
                }
                summary.usage_complete = false;
            }
            private_json(
                &options.output.join(format!("failure-{}.json", unit.id)),
                &json!({"diagnostic":review_diagnostic(&error),"raw_output":review_output(&error),"usage":review_usage(&error)}),
            )?;
        }
    }
    if let Some(budget) = &budget {
        let snapshot = budget.snapshot()?;
        summary.attempts = snapshot.calls;
        private_json(&options.output.join("budget.json"), &snapshot)?;
    }
    private_json(
        &options.output.join("prepared-records.json"),
        &prepared_records,
    )?;
    private_json(&options.output.join("summary.json"), &summary)?;
    // JSON stores the authoritative graph; the Markdown table is only a view.
    use std::{io::Write, os::unix::fs::OpenOptionsExt};
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(options.output.join("report.md"))?;
    file.write_all(report.as_bytes())?;
    file.sync_all()?;
    Ok(summary)
}
fn table_cell(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace(['\n', '\r'], " ")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('[', "\\[")
        .replace(']', "\\]")
        .replace('`', "\\`")
        .replace('@', "\\@")
}
