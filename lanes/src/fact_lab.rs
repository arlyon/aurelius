//! File-only extraction experiments. No Temporal, database, object-store or
//! embedding clients are constructed here. Reports contain untrusted candidates.
use crate::{
    config::Config,
    contracts::{ByteRange, Chunk, Document},
    fact_context::FactContext,
    fact_preparation::{PreparedFactSource, alias_hypotheses},
    facts::{ChatExtractor, ExtractionReview, ExtractionUsage, ReviewDiagnostic},
    language::{LanguageAnalysis, LanguageRuntime},
    provider_metadata::ProviderProperties,
    quality::{self, QualityGraph, TypedCandidate},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::Arc,
};
const LAB_VERSION: &str = "file-fact-lab-v1";
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LabRecord {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub provider: String,
    pub document: Document,
    pub range: ByteRange,
    #[serde(default)]
    pub prepared: Option<PreparedFactSource>,
    #[serde(default)]
    pub provider_properties: Option<ProviderProperties>,
    #[serde(default)]
    pub canonical_context: Option<FactContext>,
}
#[derive(Clone, Debug)]
pub struct Unit {
    pub id: String,
    pub members: Vec<String>,
    pub record: LabRecord,
}
#[derive(Clone, Debug)]
pub enum Mode {
    Replay(PathBuf),
    Live,
}
#[derive(Clone, Debug)]
pub struct Options {
    pub mode: Mode,
    pub output: PathBuf,
    pub selected: BTreeSet<String>,
    pub fresh: bool,
    pub batch_adjacent: bool,
    pub max_calls: usize,
    pub max_batch_bytes: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordResult {
    pub id: String,
    pub members: Vec<String>,
    pub provider: String,
    pub status: String,
    pub source: crate::contracts::DocumentKey,
    pub source_content_hash: String,
    pub source_range: ByteRange,
    pub source_text: Option<String>,
    pub proposals: Vec<Value>,
    #[serde(default)]
    pub argument_texts: BTreeMap<String, String>,
    pub graph: Option<QualityGraph>,
    pub diagnostic: Option<ReviewDiagnostic>,
    pub usage: Option<ExtractionUsage>,
    pub historical_rejections_unavailable: usize,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Summary {
    pub selected_records: usize,
    pub extraction_units: usize,
    pub attempts: usize,
    pub completed_units: usize,
    pub candidates: usize,
    pub rejected_proposals: usize,
    pub review_units: usize,
    pub not_run: usize,
    pub provider_stopped: bool,
    pub usage_records: usize,
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub calculated_cost_usd: f64,
    pub usage_complete: bool,
    pub cost_basis: &'static str,
    pub semantic_accuracy: &'static str,
}
fn safe_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 80
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}
fn bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "lab file exceeds byte budget");
    Ok(bytes)
}
// Only extraction settings apply to the lab; ingestion-service configuration
// may be absent. Production Config::validate keeps its stricter worker contract.
pub fn load_config(path: &Path) -> Result<Config> {
    let bytes = bounded(path, 128 * 1024)?;
    let config: Config = toml::from_str(std::str::from_utf8(&bytes)?)
        .map_err(|_| anyhow::anyhow!("invalid lab TOML"))?;
    config.ontology.validate()?;
    config
        .chat
        .as_ref()
        .context("chat configuration required")?
        .validate()?;
    config
        .language
        .as_ref()
        .context("language configuration required")?
        .validate()?;
    config
        .fact_quality
        .as_ref()
        .context("typed policy required")?
        .validate(&config.ontology)?;
    Ok(config)
}
pub fn load(path: &Path) -> Result<Vec<LabRecord>> {
    let bytes = bounded(path, 64 * 1024 * 1024)?;
    let values = match serde_json::from_slice::<Value>(&bytes) {
        Ok(Value::Array(values)) => values,
        Ok(Value::Object(mut object)) => {
            match object.remove("slots").or_else(|| object.remove("records")) {
                Some(Value::Array(values)) => values,
                None => vec![Value::Object(object)],
                _ => anyhow::bail!("dataset records must be an array"),
            }
        }
        Ok(_) => anyhow::bail!("invalid dataset shape"),
        Err(_) => std::str::from_utf8(&bytes)?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<std::result::Result<Vec<Value>, _>>()
            .map_err(|_| anyhow::anyhow!("invalid dataset JSONL"))?,
    };
    ensure!(
        !values.is_empty() && values.len() <= 1000,
        "dataset must contain1..1000records"
    );
    let mut ids = BTreeSet::new();
    let mut output = Vec::new();
    for (index, mut value) in values.into_iter().enumerate() {
        if value.get("range").is_none()
            && let Some(text) = value.pointer("/document/text").and_then(Value::as_str)
        {
            value["range"] = json!({"start":0,"end":text.len()});
        }
        let mut record: LabRecord = serde_json::from_value(value)
            .map_err(|_| anyhow::anyhow!("invalid normalized lab record"))?;
        if record.id.is_empty() {
            record.id = format!("{index:03}");
        }
        ensure!(
            safe_id(&record.id) && ids.insert(record.id.clone()),
            "unsafe or duplicate record ID"
        );
        record.range.text(&record.document)?;
        ensure!(
            record.provider.len() <= 100 && !record.provider.chars().any(char::is_control),
            "invalid provider label"
        );
        output.push(record);
    }
    Ok(output)
}
pub fn units(records: &[LabRecord], selected: &BTreeSet<String>, batch: bool) -> Result<Vec<Unit>> {
    units_bounded(records, selected, batch, 4096)
}
pub fn units_bounded(
    records: &[LabRecord],
    selected: &BTreeSet<String>,
    batch: bool,
    max_bytes: usize,
) -> Result<Vec<Unit>> {
    ensure!(
        (1..=32768).contains(&max_bytes),
        "invalid adjacent batching byte budget"
    );
    ensure!(
        selected
            .iter()
            .all(|id| records.iter().any(|record| &record.id == id)),
        "selected record does not exist"
    );
    let mut output: Vec<Unit> = Vec::new();
    for record in records
        .iter()
        .filter(|record| selected.is_empty() || selected.contains(&record.id))
    {
        output.push(Unit {
            id: record.id.clone(),
            members: vec![record.id.clone()],
            record: record.clone(),
        });
    }
    ensure!(!output.is_empty(), "empty selection");
    if !batch {
        return Ok(output);
    }
    output.sort_by_key(|unit| {
        (
            unit.record.document.key.scope.clone(),
            unit.record.document.key.source.clone(),
            unit.record.document.key.revision.clone(),
            unit.record.range.start,
            unit.record.range.end,
            unit.id.clone(),
        )
    });
    let mut merged: Vec<Unit> = Vec::new();
    for unit in output {
        if let Some(previous) = merged.last_mut() {
            let a = &previous.record;
            let b = &unit.record;
            let same = serde_json::to_value(&a.document)? == serde_json::to_value(&b.document)?
                && a.provider == b.provider
                && serde_json::to_value(&a.prepared)? == serde_json::to_value(&b.prepared)?
                && serde_json::to_value(&a.provider_properties)?
                    == serde_json::to_value(&b.provider_properties)?
                && serde_json::to_value(&a.canonical_context)?
                    == serde_json::to_value(&b.canonical_context)?;
            let gap_ok = b.range.start <= a.range.end
                || a.document
                    .text
                    .get(a.range.end..b.range.start)
                    .is_some_and(|text| text.chars().all(char::is_whitespace));
            if same && gap_ok && b.range.end.max(a.range.end) - a.range.start <= max_bytes {
                previous.record.range.end = previous.record.range.end.max(b.range.end);
                previous.members.extend(unit.members);
                previous.id = format!(
                    "batch-{}",
                    &blake3::hash(&serde_json::to_vec(&previous.members)?).to_hex()[..16]
                );
                continue;
            }
        }
        merged.push(unit);
    }
    Ok(merged)
}
fn private_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
struct Ledger {
    file: File,
    attempts: usize,
    limit: usize,
}
impl Ledger {
    fn new(path: &Path, limit: usize) -> Result<Self> {
        Ok(Self {
            file: OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(path)?,
            attempts: 0,
            limit,
        })
    }
    fn start(&mut self, id: &str) -> Result<()> {
        ensure!(self.attempts < self.limit, "lab call budget exhausted");
        serde_json::to_writer(
            &mut self.file,
            &json!({"phase":"attempt_started","id":id,"attempt":self.attempts+1}),
        )?;
        self.file.write_all(b"\n")?;
        self.file.sync_all()?;
        self.attempts += 1;
        Ok(())
    }
}
struct Preparation<'a> {
    config: &'a Config,
    runtime: Option<Arc<LanguageRuntime>>,
    language: BTreeMap<String, LanguageAnalysis>,
    fresh: bool,
    language_profile: String,
}
impl Preparation<'_> {
    async fn prepare(&mut self, record: &LabRecord) -> Result<PreparedFactSource> {
        let policy = self
            .config
            .fact_quality
            .as_ref()
            .context("typed policy required")?;
        let language_config = self
            .config
            .language
            .as_ref()
            .context("language configuration required")?;
        let language_profile = self.language_profile.clone();
        let key = serde_json::to_string(&(
            &record.document.key,
            record.document.content_hash(),
            &language_profile,
        ))?;
        let language = if let Some(analysis) = self.language.get(&key) {
            analysis.clone()
        } else {
            let analysis = if self.fresh {
                if self.runtime.is_none() {
                    self.runtime = Some(Arc::new(LanguageRuntime::load(language_config.clone())?));
                }
                let runtime = self.runtime.as_ref().context("language runtime")?.clone();
                let document = record.document.clone();
                tokio::task::spawn_blocking(move || runtime.analyze(&document)).await??
            } else {
                let analysis = record
                    .prepared
                    .as_ref()
                    .context("cached preparation missing; use --fresh")?
                    .language
                    .clone();
                ensure!(
                    analysis.profile == language_profile,
                    "cached model/config profile changed; use --fresh"
                );
                analysis
            };
            analysis.validate(&record.document)?;
            self.language.insert(key, analysis.clone());
            analysis
        };
        let context = record
            .canonical_context
            .clone()
            .or_else(|| {
                record
                    .prepared
                    .as_ref()
                    .map(|p| p.canonical_context.clone())
            })
            .unwrap_or_else(|| FactContext {
                organization_id: record.document.key.scope.clone(),
                entities: vec![],
            });
        let references = &self
            .config
            .chat
            .as_ref()
            .context("chat configuration required")?
            .references;
        let aliases = alias_hypotheses(references, &record.document)?;
        let profile = blake3::hash(&serde_json::to_vec(&(
            LAB_VERSION,
            &language_profile,
            self.config.ontology.semantic_hash()?,
            references.semantic_hash()?,
            context.canonical_prefix()?,
        ))?)
        .to_hex()
        .to_string();
        let mut prepared = PreparedFactSource::from_analysis(
            &record.document,
            language,
            aliases,
            context,
            policy.context_as_of,
            profile,
        )?;
        if let Some(properties) = &record.provider_properties {
            prepared = prepared.with_provider_properties(&record.document, properties)?;
        } else if let Some(header) = record
            .prepared
            .as_ref()
            .and_then(|p| p.provider_header.clone())
        {
            header.validate(&record.document)?;
            prepared.provider_header = Some(header);
        }
        Ok(prepared)
    }
}
fn replay_body(directory: &Path, unit: &Unit) -> Result<(String, usize)> {
    let path = [
        directory.join(format!("record-{}.json", unit.id)),
        directory.join(format!("result-{}.json", unit.id)),
        directory.join(format!("{}.json", unit.id)),
    ]
    .into_iter()
    .find(|path| path.exists())
    .context("replay response missing")?;
    let value: Value = serde_json::from_slice(&bounded(&path, 1024 * 1024)?)
        .map_err(|_| anyhow::anyhow!("invalid replay JSON"))?;
    if let Some(status) = value.get("status").and_then(Value::as_str) {
        ensure!(status == "completed", ReviewDiagnostic::ReplayUnavailable);
    }
    if value.get("source").is_some() {
        ensure!(
            value["source"] == serde_json::to_value(&unit.record.document.key)?
                && value["source_range"] == serde_json::to_value(unit.record.range)?
                && value["source_content_hash"] == unit.record.document.content_hash(),
            "replay source identity/range changed"
        );
    } else if let Some(graph) = value.get("graph") {
        ensure!(
            graph["document"] == serde_json::to_value(&unit.record.document.key)?
                && graph["source_range"] == serde_json::to_value(unit.record.range)?
                && graph["content_hash"] == unit.record.document.content_hash(),
            "legacy replay source changed"
        );
    }
    if let Some(proposals) = value.get("proposals") {
        return Ok((json!({"facts":proposals}).to_string(), 0));
    }
    if value.get("facts").is_some() {
        return Ok((serde_json::to_string(&value)?, 0));
    }
    let graph = value
        .get("graph")
        .context("replay file has no proposals or graph")?;
    let facts: Vec<_> = graph["candidates"]
        .as_array()
        .context("legacy replay candidates")?
        .iter()
        .map(|claim| claim["candidate"].clone())
        .collect();
    Ok((
        json!({"facts":facts}).to_string(),
        graph["rejections"].as_array().map_or(0, Vec::len),
    ))
}
fn markdown_text(text: &str) -> String {
    let mut output = String::new();
    for character in text.chars() {
        if matches!(
            character,
            '\\' | '`' | '*' | '_' | '[' | ']' | '<' | '>' | '(' | ')' | '!' | '|'
        ) {
            output.push('\\');
        }
        output.push(character);
    }
    output
}
pub async fn run(config: &Config, records: &[LabRecord], options: &Options) -> Result<Summary> {
    let live = matches!(options.mode, Mode::Live);
    ensure!(
        !live || (options.max_calls > 0 && options.max_calls <= 1000),
        "live mode requires explicit1..1000call cap"
    );
    ensure!(
        !options.output.exists(),
        "output directory exists; never resume or overwrite an experiment"
    );
    config.ontology.validate()?;
    let policy = config
        .fact_quality
        .as_ref()
        .context("typed policy required")?;
    policy.validate(&config.ontology)?;
    let selected = units_bounded(
        records,
        &options.selected,
        options.batch_adjacent,
        options.max_batch_bytes,
    )?;
    std::fs::create_dir_all(&options.output)?;
    std::fs::set_permissions(&options.output, std::fs::Permissions::from_mode(0o700))?;
    let extractor = if live {
        Some(ChatExtractor::new(
            config.chat.as_ref().context("chat config required")?,
            config.ontology.clone(),
        )?)
    } else {
        None
    };
    let base_profile = ChatExtractor::profile_for(
        config
            .chat
            .as_ref()
            .context("chat configuration required")?,
        &config.ontology,
    )?;
    let mut ledger = Ledger::new(&options.output.join("attempts.jsonl"), options.max_calls)?;
    private_json(
        &options.output.join("manifest.json"),
        &json!({"version":LAB_VERSION,"live":live,"max_calls":options.max_calls,"fresh_preparation":options.fresh,"batch_adjacent":options.batch_adjacent,"max_batch_bytes":options.max_batch_bytes,"config_hash":blake3::hash(&serde_json::to_vec(config)?).to_hex().to_string(),"records_hash":blake3::hash(&serde_json::to_vec(records)?).to_hex().to_string(),"units":selected.iter().map(|unit|json!({"id":unit.id,"members":unit.members,"document":unit.record.document.key,"range":unit.record.range})).collect::<Vec<_>>(),"production_writes":0}),
    )?;
    let language_profile = LanguageRuntime::profile_for(
        config
            .language
            .as_ref()
            .context("language config required")?,
    )?;
    let mut preparation = Preparation {
        config,
        runtime: None,
        language: BTreeMap::new(),
        fresh: options.fresh,
        language_profile,
    };
    let mut cached_records = Vec::new();
    let mut summary = Summary {
        selected_records: selected.iter().map(|unit| unit.members.len()).sum(),
        extraction_units: selected.len(),
        cost_basis: "user-provided $0.10/M input + $0.50/M output, cached input full rate, no gateway fees; not an invoice",
        semantic_accuracy: "not automatically measured; compare source and expectations manually",
        ..Default::default()
    };
    let mut all_usage_fields = true;
    let mut display = String::from(
        "# Local fact extraction\n\nUnreviewed proposals, not accepted truths. No production writes.\n",
    );
    for unit in selected {
        let mut result = RecordResult {
            id: unit.id.clone(),
            members: unit.members.clone(),
            provider: unit.record.provider.clone(),
            status: String::new(),
            source: unit.record.document.key.clone(),
            source_content_hash: unit.record.document.content_hash(),
            source_range: unit.record.range,
            source_text: None,
            proposals: vec![],
            argument_texts: BTreeMap::new(),
            graph: None,
            diagnostic: None,
            usage: None,
            historical_rejections_unavailable: 0,
        };
        if live && (ledger.attempts >= options.max_calls || summary.provider_stopped) {
            result.status = "not_run_budget_or_provider_stop".into();
            summary.not_run += 1;
        } else {
            let attempt = async {
                let prepared = preparation
                    .prepare(&unit.record)
                    .await
                    .context(ExtractionReview::Input)?;
                let chunk = Chunk {
                    document: unit.record.document.clone(),
                    range: unit.record.range,
                };
                let (prompt, input) =
                    quality::model_input(&chunk, &prepared, &config.ontology, policy)
                        .context(ExtractionReview::Input)?;
                crate::facts::validate_model_input(&prompt)
                    .and_then(|()| crate::facts::validate_model_input(&input))
                    .context(ExtractionReview::Input)?;
                let profile = policy.profile(
                    &base_profile,
                    &prepared.preparation_profile,
                )?;
                private_json(&options.output.join(format!("request-{}.json", unit.id)), &json!({"system_prompt":prompt,"input":serde_json::from_str::<Value>(&input)?,"transport_credentials_included":false,"purpose":"private local debug artifact; provider payload logging remains disabled"}))?;
                result.source_text = Some(chunk.text()?.into());
                result.argument_texts = prepared
                    .language
                    .mentions
                    .iter()
                    .filter(|mention| {
                        mention.range.start >= chunk.range.start
                            && mention.range.end <= chunk.range.end
                    })
                    .map(|mention| (mention.id.clone(), mention.text.clone()))
                    .collect();
                for member in &unit.members {
                    let mut cached = records
                        .iter()
                        .find(|record| &record.id == member)
                        .context("original record missing")?
                        .clone();
                    cached.prepared = Some(prepared.clone());
                    cached_records.push(cached);
                }
                if let Some(extractor) = &extractor {
                    ledger.start(&unit.id)?;
                    let extracted = extractor
                        .extract_quality_detailed(&chunk, &prepared, policy, &profile)
                        .await?;
                    result.proposals = extracted.proposals;
                    result.graph = Some(extracted.graph);
                    result.usage = extracted.usage;
                } else {
                    let (body, unavailable) = replay_body(
                        match &options.mode {
                            Mode::Replay(path) => path,
                            Mode::Live => unreachable!(),
                        },
                        &unit,
                    )
                    .map_err(|error| {
                        if error.downcast_ref::<ReviewDiagnostic>().is_some() {
                            error
                        } else {
                            error.context(ReviewDiagnostic::ReplayUnavailable)
                        }
                    })
                    .context(ExtractionReview::Output)?;
                    result.historical_rejections_unavailable = unavailable;
                    result.proposals =
                        quality::parse_proposals(&body).context(ExtractionReview::Output)?;
                    result.graph = Some(
                        quality::parse_graph(
                            &body,
                            &chunk,
                            &prepared,
                            &config.ontology,
                            policy,
                            &profile,
                        )
                        .context(ExtractionReview::Output)?,
                    );
                }
                Ok::<_, anyhow::Error>(())
            }
            .await;
            match attempt {
                Ok(()) => {
                    result.status = "completed".into();
                    summary.completed_units += 1;
                }
                Err(error) => {
                    result.usage = crate::facts::review_usage(&error);
                    if let Some(output) = crate::facts::review_output(&error) {
                        private_json(
                            &options
                                .output
                                .join(format!("failed-output-{}.json", unit.id)),
                            &json!({"purpose":"untrusted model output for private local diagnosis only; not accepted evidence","output":output}),
                        )?;
                    }
                    result.diagnostic = Some(crate::facts::review_diagnostic(&error));
                    summary.review_units += 1;
                    match error.downcast_ref::<ExtractionReview>() {
                        Some(ExtractionReview::Input) => {
                            result.source_text = None;
                            result.status = "input_review".into();
                        }
                        Some(ExtractionReview::Output) => result.status = "output_review".into(),
                        None => {
                            result.status = "provider_or_io_failure_stop".into();
                            summary.provider_stopped = true;
                        }
                    }
                }
            }
        }
        if let Some(graph) = &result.graph {
            summary.candidates += graph.candidates.len();
            summary.rejected_proposals += graph.rejections.len();
        }
        if let Some(usage) = &result.usage {
            summary.usage_records += 1;
            all_usage_fields &= usage.input_tokens.is_some() && usage.output_tokens.is_some();
            summary.input_tokens += usage.input_tokens.unwrap_or_default();
            summary.cached_input_tokens += usage.cached_input_tokens.unwrap_or_default();
            summary.output_tokens += usage.output_tokens.unwrap_or_default();
        }
        private_json(
            &options.output.join(format!("record-{}.json", unit.id)),
            &result,
        )?;
        display.push_str(&format!(
            "\n## {} — {}\n\nStatus: {}. Members: {:?}.\n",
            unit.id,
            markdown_text(&result.provider),
            result.status,
            result.members
        ));
        if let Some(text) = &result.source_text {
            display.push_str(&format!(
                "\nSource:\n\n{}\n",
                text.lines()
                    .map(|line| format!("> {}", markdown_text(line)))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        display.push_str("\nProposals (including rejected):\n\n");
        for (index, proposal) in result.proposals.iter().enumerate() {
            let rejection = result
                .graph
                .as_ref()
                .and_then(|g| g.rejections.iter().find(|r| r.index == index));
            let status = if let Some(rejection) = rejection {
                format!("REVIEW {:?}", rejection.reason)
            } else {
                "SURVIVING CANDIDATE".into()
            };
            let label = |id: &String| {
                result
                    .argument_texts
                    .get(id)
                    .map(String::as_str)
                    .unwrap_or("unknown mention")
            };
            let Ok(proposal) = serde_json::from_value::<TypedCandidate>(proposal.clone()) else {
                display.push_str(&format!(
                    "- {status}: invalid typed candidate; raw payload retained privately.\n"
                ));
                continue;
            };
            let body = match &proposal.body {
                quality::TypedBody::Relation { predicate, object } => {
                    format!("{} → {}", predicate, label(&object.mention_id))
                }
                quality::TypedBody::Property {
                    property, value, ..
                } => format!("{} → {}", property, serde_json::to_string(value)?),
            };
            display.push_str(&format!(
                "- {status}: **{} → {}**; {:?}/{:?}. Qualifiers: {}\n",
                markdown_text(label(&proposal.subject.mention_id)),
                markdown_text(&body),
                proposal.polarity,
                proposal.modality,
                markdown_text(&serde_json::to_string(&proposal.qualifiers)?)
            ));
        }
        println!(
            "{}",
            json!({"unit":unit.id,"status":result.status,"calls":ledger.attempts,"candidates":result.graph.as_ref().map_or(0,|g|g.candidates.len())})
        );
    }
    summary.attempts = ledger.attempts;
    summary.usage_complete = summary.usage_records == summary.attempts && all_usage_fields;
    summary.calculated_cost_usd =
        (summary.input_tokens as f64 * 0.10 + summary.output_tokens as f64 * 0.50) / 1_000_000.;
    private_json(&options.output.join("summary.json"), &summary)?;
    private_json(
        &options.output.join("prepared-records.json"),
        &cached_records,
    )?;
    let mut report = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(options.output.join("report.md"))?;
    report.write_all(display.as_bytes())?;
    report.sync_all()?;
    Ok(summary)
}
pub fn compare(before: &Path, after: &Path) -> Result<Value> {
    let mut names = BTreeSet::new();
    for directory in [before, after] {
        for entry in std::fs::read_dir(directory)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if name.starts_with("record-") && name.ends_with(".json") {
                names.insert(name);
            }
        }
    }
    let mut changes = Vec::new();
    for name in names {
        let read = |directory: &Path| -> Result<Option<RecordResult>> {
            let path = directory.join(&name);
            if !path.exists() {
                return Ok(None);
            }
            Ok(Some(serde_json::from_slice(&bounded(
                &path,
                2 * 1024 * 1024,
            )?)?))
        };
        let old = read(before)?;
        let new = read(after)?;
        if let (Some(old), Some(new)) = (&old, &new) {
            ensure!(
                old.source == new.source
                    && old.source_range == new.source_range
                    && old.source_content_hash == new.source_content_hash,
                "cannot compare changed sources as the same record"
            );
        }
        let id = old
            .as_ref()
            .or(new.as_ref())
            .context("comparison record missing")?
            .id
            .clone();
        changes.push(json!({"id":id,"before":old,"after":new}));
    }
    Ok(json!({"semantic_scoring":"not performed","records":changes}))
}
