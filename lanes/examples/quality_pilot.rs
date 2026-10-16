use anyhow::{Context, Result, ensure};
use aurelius_lanes::{
    config::Config,
    contracts::{ByteRange, Chunk, Document},
    fact_preparation::{FactPreparer, PreparedFactSource},
    facts::{ChatExtractor, ExtractionReview},
    postgres::{ExtractionInput, PostgresSink},
    provider_metadata::{ProviderMetadataReader, ProviderProperties},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::{Postgres, QueryBuilder, Row, postgres::PgPoolOptions};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use tracing::{
    Subscriber,
    field::{Field, Visit},
};
use tracing_subscriber::{
    Layer,
    layer::{Context as TraceContext, SubscriberExt},
    util::SubscriberInitExt,
};
const LIMIT: usize = 100;
const PER_PROVIDER: usize = 25;
const SEED: &str = "typed-quality-balanced-100-v1";
#[derive(Clone, Serialize, Deserialize)]
struct Slot {
    provider: String,
    document: Document,
    range: ByteRange,
    prepared: PreparedFactSource,
    provider_properties: Option<ProviderProperties>,
}
#[derive(Serialize, Deserialize)]
struct Sample {
    seed: String,
    profile: String,
    config_hash: String,
    slots: Vec<Slot>,
    exclusions: BTreeMap<String, BTreeMap<String, usize>>,
}
#[derive(Clone, Default, Serialize)]
struct Usage {
    index: usize,
    input_tokens: Option<u64>,
    cached_input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}
struct UsageLayer {
    current: Arc<AtomicUsize>,
    records: Arc<Mutex<Vec<Usage>>>,
}
struct Visitor {
    usage: Usage,
    is_usage: bool,
}
impl Visit for Visitor {
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "input_tokens" => self.usage.input_tokens = Some(value),
            "cached_input_tokens" => self.usage.cached_input_tokens = Some(value),
            "output_tokens" => self.usage.output_tokens = Some(value),
            _ => (),
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.is_usage = format!("{value:?}").contains("fact extraction usage");
        }
    }
}
impl<S: Subscriber> Layer<S> for UsageLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _: TraceContext<'_, S>) {
        let mut visitor = Visitor {
            usage: Usage {
                index: self.current.load(Ordering::SeqCst),
                ..Default::default()
            },
            is_usage: false,
        };
        event.record(&mut visitor);
        if visitor.is_usage {
            self.records.lock().expect("usage lock").push(visitor.usage);
        }
    }
}
struct Ledger {
    file: File,
    started: usize,
}
impl Ledger {
    fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            file: OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(path)
                .context("pilot ledger already exists; never automatically resume or rerun")?,
            started: 0,
        })
    }
    fn start(&mut self, index: usize, provider: &str) -> Result<()> {
        ensure!(
            self.started < LIMIT && index == self.started,
            "pilot request ceiling reached or request replay attempted"
        );
        self.event(&json!({"phase":"attempt_started","index":index,"provider":provider}))?;
        self.started += 1;
        Ok(())
    }
    fn event(&mut self, value: &serde_json::Value) -> Result<()> {
        serde_json::to_writer(&mut self.file, value)?;
        self.file.write_all(b"\n")?;
        self.file.sync_all()?;
        Ok(())
    }
}
fn private_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
fn quote(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}
fn config_hash(config: &Config) -> Result<String> {
    Ok(blake3::hash(&serde_json::to_vec(config)?)
        .to_hex()
        .to_string())
}
fn profile(config: &Config, extractor: &ChatExtractor) -> Result<String> {
    config
        .fact_quality
        .as_ref()
        .context("typed policy required")?
        .profile(&extractor.profile, &FactPreparer::profile_for(config)?)
}
fn safe_reason(error: &anyhow::Error) -> &'static str {
    if error.downcast_ref::<ExtractionReview>().is_some() {
        "input_or_output_review"
    } else if error
        .chain()
        .any(|e| e.downcast_ref::<sqlx::Error>().is_some())
    {
        "database_read_failed"
    } else {
        "preparation_failed"
    }
}
async fn prepare(config: &Config, extractor: &ChatExtractor, directory: &Path) -> Result<()> {
    ensure!(!directory.exists(), "pilot output directory already exists");
    std::fs::create_dir(directory)?;
    std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;
    let settings = config.postgres.as_ref().context("Postgres required")?;
    let pool = PgPoolOptions::new()
        .max_connections(2)
        .after_connect(|connection, _| {
            Box::pin(async move {
                sqlx::query("SET default_transaction_read_only=on")
                    .execute(connection)
                    .await?;
                Ok(())
            })
        })
        .connect(&std::env::var(&settings.url_env).context("pilot database environment missing")?)
        .await
        .context("pilot read-only PG connection failed")?;
    let readonly: String = sqlx::query_scalar("SHOW transaction_read_only")
        .fetch_one(&pool)
        .await?;
    ensure!(readonly == "on", "pilot PG connection is not read-only");
    let sink = PostgresSink::from_pool(
        settings.clone(),
        config
            .embedding
            .as_ref()
            .context("embedding configuration")?
            .dimensions,
        pool.clone(),
    )
    .await?;
    let preparer = FactPreparer::with_sink(config, sink.clone())?;
    let provider_reader = ProviderMetadataReader::new(config)?;
    let policy = config
        .fact_quality
        .as_ref()
        .context("typed policy required")?;
    let mut slots = Vec::new();
    let mut exclusions = BTreeMap::new();
    for provider in ["asana", "drive", "gmail", "slack"] {
        let lane = config
            .lanes
            .get(provider)
            .context("pilot requires all four configured providers")?;
        let lane_profile = config.lane_request(provider)?.profile;
        let mut query = QueryBuilder::<Postgres>::new("SELECT a.id FROM ");
        query.push(format!("{}.{}",quote(&settings.schema),quote(&settings.artifacts_table))).push(" a WHERE organization_id=$1 AND source_type=$2 AND metadata->>'profile'=$3 AND (metadata->>'chunk_count')::int BETWEEN 1 AND 512 AND metadata->'context'->>'status' IS DISTINCT FROM 'deferred' ORDER BY md5(a.id||$4),a.id LIMIT 500");
        let mut tx = pool.begin().await?;
        if let Some(setting) = &settings.tenant_setting {
            sqlx::query("SELECT set_config($1,$2,true)")
                .bind(setting)
                .bind(&lane.scope)
                .execute(&mut *tx)
                .await?;
        }
        let rows = query
            .build()
            .bind(&lane.scope)
            .bind(format!("aurelius/{}", lane.connection))
            .bind(&lane_profile)
            .bind(SEED)
            .fetch_all(&mut *tx)
            .await?;
        tx.commit().await?;
        let mut docs: Vec<(
            ExtractionInput,
            PreparedFactSource,
            Option<ProviderProperties>,
            Vec<ByteRange>,
        )> = Vec::new();
        let mut excluded: BTreeMap<String, usize> = BTreeMap::new();
        for row in rows {
            let id: String = row.try_get("id")?;
            let result = async {
                let mut input = sink.extraction_input(&lane.scope, &id).await?;
                let properties = if input.kind == aurelius_lanes::document::DocumentKind::AsanaTask
                {
                    let value = provider_reader
                        .as_ref()
                        .context("task metadata reader required")?
                        .read(&input)
                        .await?
                        .context("task metadata missing")?;
                    input.document.metadata.authored_at = value
                        .authored_at
                        .map(|time| time.with_timezone(&chrono::Utc));
                    input.document.metadata.modified_at = value
                        .modified_at
                        .map(|time| time.with_timezone(&chrono::Utc));
                    input.document.metadata.author = value
                        .author
                        .as_ref()
                        .map(|author| format!("{}:{}", author.provider, author.id));
                    Some(value)
                } else {
                    None
                };
                let mut prepared = preparer
                    .prepare(input.document.clone(), policy.context_as_of)
                    .await?;
                if let Some(properties) = &properties {
                    prepared = prepared.with_provider_properties(&input.document, properties)?;
                }
                let mut ranges = Vec::new();
                for range in &input.ranges {
                    let chunk = Chunk {
                        document: input.document.clone(),
                        range: *range,
                    };
                    if extractor
                        .preflight_quality(&chunk, &prepared, policy)
                        .is_ok()
                    {
                        ranges.push(*range);
                    } else {
                        *excluded
                            .entry("window_preflight_review".into())
                            .or_default() += 1;
                    }
                }
                ranges.sort_by_key(|range| {
                    blake3::hash(format!("{SEED}/{id}/{}/{}", range.start, range.end).as_bytes())
                        .to_hex()
                        .to_string()
                });
                ensure!(!ranges.is_empty(), "no eligible windows");
                Ok::<_, anyhow::Error>((input, prepared, properties, ranges))
            }
            .await;
            match result {
                Ok(doc) => docs.push(doc),
                Err(error) => *excluded.entry(safe_reason(&error).into()).or_default() += 1,
            }
            if docs.len() >= PER_PROVIDER {
                break;
            }
        }
        let mut selected = 0;
        let mut round = 0;
        loop {
            let mut added = false;
            for (input, prepared, properties, ranges) in &docs {
                if let Some(range) = ranges.get(round) {
                    slots.push(Slot {
                        provider: provider.into(),
                        document: input.document.clone(),
                        range: *range,
                        prepared: prepared.clone(),
                        provider_properties: properties.clone(),
                    });
                    selected += 1;
                    added = true;
                    if selected == PER_PROVIDER {
                        break;
                    }
                }
            }
            if selected == PER_PROVIDER || !added {
                break;
            }
            round += 1;
        }
        println!(
            "{}",
            json!({"phase":"preflight","provider":provider,"eligible_documents":docs.len(),"selected_chunks":selected,"excluded":excluded})
        );
        exclusions.insert(provider.into(), excluded);
        if selected != PER_PROVIDER {
            private_json(
                &directory.join("preflight_failure.json"),
                &json!({"provider":provider,"selected_chunks":selected,"eligible_documents":docs.len(),"paid_calls":0,"reason":"insufficient_processable_whole_document_context"}),
            )?;
            anyhow::bail!(
                "not enough eligible chunks; no inference authorized for a different sampling policy"
            );
        }
    }
    ensure!(
        slots.len() == LIMIT,
        "pilot sample must have exactly 100 slots"
    );
    slots.sort_by_key(|slot| {
        blake3::hash(
            format!(
                "{SEED}/{}/{}/{}/{}",
                slot.provider, slot.document.key.source, slot.range.start, slot.range.end
            )
            .as_bytes(),
        )
        .to_hex()
        .to_string()
    });
    let sample = Sample {
        seed: SEED.into(),
        profile: profile(config, extractor)?,
        config_hash: config_hash(config)?,
        slots,
        exclusions,
    };
    private_json(&directory.join("sample.json"), &sample)?;
    println!(
        "{}",
        json!({"phase":"sample_frozen","chunks":LIMIT,"per_provider":PER_PROVIDER,"profile":sample.profile,"paid_calls":0})
    );
    Ok(())
}
async fn refresh(
    config: &Config,
    extractor: &ChatExtractor,
    original: &Path,
    directory: &Path,
) -> Result<()> {
    ensure!(!directory.exists(), "retest directory already exists");
    let original_bytes = std::fs::read(original)?;
    let original: Sample = serde_json::from_slice(&original_bytes)?;
    ensure!(
        original.slots.len() == LIMIT && original.seed == SEED,
        "original frozen sample mismatch"
    );
    std::fs::create_dir(directory)?;
    std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;
    let settings = config.postgres.as_ref().context("Postgres required")?;
    let pool = PgPoolOptions::new()
        .max_connections(2)
        .after_connect(|connection, _| {
            Box::pin(async move {
                sqlx::query("SET default_transaction_read_only=on")
                    .execute(connection)
                    .await?;
                Ok(())
            })
        })
        .connect(&std::env::var(&settings.url_env).context("pilot database environment missing")?)
        .await?;
    let readonly: String = sqlx::query_scalar("SHOW transaction_read_only")
        .fetch_one(&pool)
        .await?;
    ensure!(readonly == "on", "retest preparation is not read-only");
    let sink = PostgresSink::from_pool(
        settings.clone(),
        config
            .embedding
            .as_ref()
            .context("embedding config")?
            .dimensions,
        pool,
    )
    .await?;
    let preparer = FactPreparer::with_sink(config, sink.clone())?;
    let reader = ProviderMetadataReader::new(config)?;
    let policy = config
        .fact_quality
        .as_ref()
        .context("typed policy required")?;
    let mut cache: BTreeMap<
        String,
        (
            ExtractionInput,
            PreparedFactSource,
            Option<ProviderProperties>,
        ),
    > = BTreeMap::new();
    let mut slots = Vec::new();
    for old in &original.slots {
        let lane = config
            .lanes
            .get(&old.provider)
            .context("original provider is not configured")?;
        ensure!(lane.scope == old.document.key.scope, "retest changed scope");
        if !cache.contains_key(&old.document.key.source) {
            let mut input = sink
                .extraction_input(&lane.scope, &old.document.key.source)
                .await?;
            ensure!(
                input.connection_id == lane.connection
                    && input.embedding_profile == config.lane_request(&old.provider)?.profile,
                "retest source connection/profile drift"
            );
            let properties = if input.kind == aurelius_lanes::document::DocumentKind::AsanaTask {
                let properties = reader
                    .as_ref()
                    .context("provider reader required")?
                    .read(&input)
                    .await?
                    .context("task metadata required")?;
                let previous = old
                    .provider_properties
                    .as_ref()
                    .context("original provider provenance missing")?;
                ensure!(
                    properties.source_row_hash == previous.source_row_hash
                        && properties.object.version == previous.object.version
                        && properties.object.key == previous.object.key,
                    "immutable raw source changed since original pilot"
                );
                input.document.metadata.authored_at = properties
                    .authored_at
                    .map(|time| time.with_timezone(&chrono::Utc));
                input.document.metadata.modified_at = properties
                    .modified_at
                    .map(|time| time.with_timezone(&chrono::Utc));
                input.document.metadata.author = properties
                    .author
                    .as_ref()
                    .map(|author| format!("{}:{}", author.provider, author.id));
                Some(properties)
            } else {
                None
            };
            ensure!(
                serde_json::to_value(&input.document)? == serde_json::to_value(&old.document)?,
                "frozen source projection or source metadata changed"
            );
            let mut prepared = preparer
                .prepare(input.document.clone(), policy.context_as_of)
                .await?;
            if let Some(properties) = &properties {
                prepared = prepared.with_provider_properties(&input.document, properties)?;
            }
            ensure!(
                prepared.context_snapshot_hash == old.prepared.context_snapshot_hash,
                "canonical snapshot changed between pilots"
            );
            cache.insert(
                old.document.key.source.clone(),
                (input, prepared, properties),
            );
        }
        let (input, prepared, properties) = cache
            .get(&old.document.key.source)
            .context("prepared source cache")?;
        ensure!(
            input.ranges.contains(&old.range)
                && serde_json::to_value(&input.document)? == serde_json::to_value(&old.document)?,
            "original source window/projection changed"
        );
        let chunk = Chunk {
            document: input.document.clone(),
            range: old.range,
        };
        extractor.preflight_quality(&chunk, prepared, policy)?;
        slots.push(Slot {
            provider: old.provider.clone(),
            document: input.document.clone(),
            range: old.range,
            prepared: prepared.clone(),
            provider_properties: properties.clone(),
        });
    }
    for provider in ["asana", "drive", "gmail", "slack"] {
        ensure!(
            slots
                .iter()
                .filter(|slot| slot.provider == provider)
                .count()
                == PER_PROVIDER,
            "retest changed provider balance"
        );
    }
    let profile = profile(config, extractor)?;
    private_json(
        &directory.join("sample.json"),
        &Sample {
            seed: SEED.into(),
            profile: profile.clone(),
            config_hash: config_hash(config)?,
            slots,
            exclusions: original.exclusions,
        },
    )?;
    private_json(
        &directory.join("retest_manifest.json"),
        &json!({"original_sample_hash":blake3::hash(&original_bytes).to_hex().to_string(),"same_order_and_source_windows":true,"fresh_verified_provider_headers":true,"fresh_neural_preparation":true,"unique_documents":cache.len(),"profile":profile,"paid_calls_during_preflight":0}),
    )?;
    println!(
        "{}",
        json!({"phase":"retest_sample_frozen","chunks":LIMIT,"unique_documents":cache.len(),"same_original_windows":true,"paid_calls":0,"profile":profile})
    );
    Ok(())
}
fn replay(config: &Config, directory: &Path) -> Result<()> {
    let sample: Sample = serde_json::from_slice(&std::fs::read(directory.join("sample.json"))?)?;
    let policy = config
        .fact_quality
        .as_ref()
        .context("typed policy required")?;
    let mut cases = Vec::new();
    let mut original = 0;
    let mut surviving = 0;
    for (index, slot) in sample.slots.iter().enumerate() {
        let path = directory.join(format!("result-{index:03}.json"));
        if !path.exists() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let candidates = value["graph"]["candidates"]
            .as_array()
            .context("historical candidates")?;
        if candidates.is_empty() {
            continue;
        }
        let facts: Vec<_> = candidates
            .iter()
            .map(|candidate| candidate["candidate"].clone())
            .collect();
        let mut prepared = slot.prepared.clone();
        if slot.provider_properties.is_some() && prepared.provider_header.is_none() {
            // Explicit layout labels for the legacy frozen fixtures, not a
            // runtime parser/fallback. New runtime headers come from verified
            // raw rows through the shared normalizer rendering function.
            let text = &slot.document.text;
            ensure!(text.starts_with("Task: "), "legacy fixture header missing");
            let title_end = text.find('\n').unwrap_or(text.len());
            let header_end = text.find("\n\n").unwrap_or(text.len());
            prepared.provider_header =
                Some(aurelius_lanes::provider_metadata::ProviderTaskHeader {
                    text: text[..header_end].into(),
                    title: ByteRange {
                        start: 6,
                        end: title_end,
                    },
                });
        }
        let chunk = Chunk {
            document: slot.document.clone(),
            range: slot.range,
        };
        let graph = aurelius_lanes::quality::parse_graph(
            &json!({"facts":facts}).to_string(),
            &chunk,
            &prepared,
            &config.ontology,
            policy,
            blake3::hash(b"offline-frozen-quality-regression")
                .to_hex()
                .as_str(),
        )?;
        if [6, 38, 43, 50, 54, 63, 77].contains(&index) {
            ensure!(
                graph.candidates.is_empty(),
                "targeted pilot failure survived regression"
            );
        }
        if index == 34 {
            ensure!(
                graph.candidates.len() == 2,
                "explicit audience targets were lost"
            );
        }
        if index == 62 {
            ensure!(
                graph.candidates.len() == 1,
                "literal budget positive control was lost"
            );
        }
        original += candidates.len();
        surviving += graph.candidates.len();
        cases.push(json!({"index":index,"provider":slot.provider,"original_candidates":candidates.len(),"surviving_candidates":graph.candidates.len(),"rejections":graph.rejections}));
    }
    ensure!(original == 14, "unexpected frozen pilot survivors");
    let report = json!({"mode":"offline_frozen_replay","new_model_calls":0,"new_database_calls":0,"original_survivors":original,"surviving_candidates":surviving,"historical_rejected_proposals_not_replayable":15,"historical_review_windows_not_replayable":6,"legacy_header_layouts":"explicit fixture labels, not reverified raw rows","semantic_entailment_accuracy":"not established by validator replay","cases":cases});
    private_json(&directory.join("offline-replay-v5.json"), &report)?;
    println!("{report}");
    Ok(())
}
async fn run(
    config: &Config,
    extractor: &ChatExtractor,
    directory: &Path,
    current: Arc<AtomicUsize>,
    records: Arc<Mutex<Vec<Usage>>>,
) -> Result<()> {
    let bytes = std::fs::read(directory.join("sample.json"))?;
    let sample: Sample = serde_json::from_slice(&bytes)?;
    ensure!(
        sample.slots.len() == LIMIT
            && sample.seed == SEED
            && sample.config_hash == config_hash(config)?
            && sample.profile == profile(config, extractor)?,
        "frozen pilot manifest/config mismatch"
    );
    for provider in ["asana", "drive", "gmail", "slack"] {
        ensure!(
            sample
                .slots
                .iter()
                .filter(|slot| slot.provider == provider)
                .count()
                == PER_PROVIDER,
            "unbalanced pilot sample"
        );
    }
    let mut ledger = Ledger::new(&directory.join("attempts.jsonl"))?;
    private_json(
        &directory.join("run_manifest.json"),
        &json!({"model":"gpt-6-luna","gateway":"default","request_limit":LIMIT,"retry_count":0,"sample_hash":blake3::hash(&bytes).to_hex().to_string(),"canonical_promotions":0,"database_writes":0}),
    )?;
    let policy = config.fact_quality.as_ref().context("typed policy")?;
    let mut candidates = 0;
    let mut rejections = 0;
    let mut output_review = 0;
    let mut stopped = false;
    for (index, slot) in sample.slots.iter().enumerate() {
        let chunk = Chunk {
            document: slot.document.clone(),
            range: slot.range,
        };
        extractor.preflight_quality(&chunk, &slot.prepared, policy)?;
        current.store(index, Ordering::SeqCst);
        ledger.start(index, &slot.provider)?;
        let outcome = extractor
            .extract_quality(&chunk, &slot.prepared, policy, &sample.profile)
            .await;
        let usage = records
            .lock()
            .expect("usage lock")
            .iter()
            .find(|usage| usage.index == index)
            .cloned();
        match outcome {
            Ok(graph) => {
                candidates += graph.candidates.len();
                rejections += graph.rejections.len();
                private_json(
                    &directory.join(format!("result-{index:03}.json")),
                    &json!({"index":index,"provider":slot.provider,"graph":graph,"usage":usage}),
                )?;
                ledger.event(&json!({"phase":"completed","index":index,"provider":slot.provider,"candidates":graph.candidates.len(),"rejections":graph.rejections.len(),"usage":usage}))?;
            }
            Err(error) if error.downcast_ref::<ExtractionReview>().is_some() => {
                output_review += 1;
                ledger.event(&json!({"phase":"model_review","index":index,"provider":slot.provider,"diagnostic":aurelius_lanes::facts::review_diagnostic(&error),"usage":usage}))?;
            }
            Err(error) => {
                let status = error
                    .chain()
                    .find_map(|error| error.downcast_ref::<reqwest::Error>())
                    .and_then(|error| error.status())
                    .map(|status| status.as_u16());
                ledger.event(&json!({"phase":"provider_failure_stop","index":index,"provider":slot.provider,"http_status":status,"usage":usage,"retry":false}))?;
                stopped = true;
            }
        }
        println!(
            "{}",
            json!({"phase":"progress","attempts":ledger.started,"candidates":candidates,"rejections":rejections,"model_review_windows":output_review,"stopped":stopped})
        );
        std::io::stdout().flush()?;
        if stopped {
            break;
        }
    }
    let usage = records.lock().expect("usage lock");
    private_json(
        &directory.join("summary.json"),
        &json!({"attempts":ledger.started,"request_ceiling":LIMIT,"completed_100":ledger.started==LIMIT&&!stopped,"provider_stop":stopped,"candidates":candidates,"rejections":rejections,"model_review_windows":output_review,"usage_records":usage.len(),"input_tokens":usage.iter().filter_map(|v|v.input_tokens).sum::<u64>(),"cached_input_tokens":usage.iter().filter_map(|v|v.cached_input_tokens).sum::<u64>(),"output_tokens":usage.iter().filter_map(|v|v.output_tokens).sum::<u64>(),"cost_usd":null,"cost_status":"exact_model_dollar_pricing_unverified","database_writes":0,"accepted_facts_promoted":0}),
    )?;
    Ok(())
}
#[tokio::main]
async fn main() {
    let result = async {
        let args: Vec<_> = std::env::args().skip(1).collect();
        ensure!(
            (args.len() == 3 && args[0] != "refresh") || (args.len() == 4 && args[0] == "refresh"),
            "usage: quality_pilot prepare|run|replay config.toml output-directory; refresh config.toml original-sample.json output-directory"
        );
        let config = Config::load(Path::new(&args[1]))?;
        if args[0] == "replay" {
            return replay(&config, Path::new(&args[2]));
        }
        let chat = config.chat.as_ref().context("chat config")?;
        ensure!(
            chat.model == "gpt-6-luna"
                && chat.gateway.is_some()
                && chat.endpoint.contains("/default/openai/responses"),
            "pilot requires approved exact model/default gateway"
        );
        let extractor = ChatExtractor::new(chat, config.ontology.clone())?;
        let current = Arc::new(AtomicUsize::new(usize::MAX));
        let records = Arc::new(Mutex::new(Vec::new()));
        tracing_subscriber::registry()
            .with(UsageLayer {
                current: current.clone(),
                records: records.clone(),
            })
            .try_init()?;
        match args[0].as_str() {
            "prepare" => prepare(&config, &extractor, Path::new(&args[2])).await,
            "refresh" => refresh(&config, &extractor, Path::new(&args[2]), Path::new(&args[3])).await,
            "run" => run(&config, &extractor, Path::new(&args[2]), current, records).await,
            _ => anyhow::bail!("unknown pilot operation"),
        }
    }
    .await;
    if result.is_err() {
        eprintln!(
            "controlled pilot stopped; details withheld; inspect safe preflight/journal files before any further action"
        );
        std::process::exit(1)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ledger_hard_caps_calls_and_refuses_resume() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "aurelius-pilot-ledger-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let mut ledger = Ledger::new(&path)?;
        for index in 0..LIMIT {
            ledger.start(index, "fixture")?;
        }
        ensure!(ledger.start(LIMIT, "fixture").is_err());
        ensure!(Ledger::new(&path).is_err());
        drop(ledger);
        let text = std::fs::read_to_string(&path)?;
        ensure!(text.lines().count() == LIMIT);
        std::fs::remove_file(path)?;
        Ok(())
    }
}
