//! Read-only source/NLP export for the approved file-only paired experiment.
use anyhow::{Context, Result, ensure};
use aurelius_lanes::{
    config::Config,
    fact_context::FactContext,
    fact_preparation::{PreparedFactSource, alias_hypotheses},
    language::LanguageRuntime,
    postgres::PostgresSink,
    provider_metadata::ProviderMetadataReader,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    process::{Command, Stdio},
};

#[derive(Deserialize)]
struct Selection {
    scope: String,
    quotas: BTreeMap<String, usize>,
    queues: Vec<Entry>,
}
#[derive(Deserialize)]
struct Entry {
    artifact_id: String,
    revision: String,
    windows: i64,
    window: usize,
    kind: String,
}
fn save(path: &Path, value: &Value) -> Result<()> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
fn native(helper: &str, packet: &Value) -> Result<Value> {
    let mut child = Command::new(helper)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    child
        .stdin
        .take()
        .context("helper input absent")?
        .write_all(&serde_json::to_vec(packet)?)?;
    let result = child.wait_with_output()?;
    ensure!(result.status.success(), "native request preparation failed");
    serde_json::from_slice(&result.stdout).context("invalid prepared request")
}
#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 6,
        "atomic_snapshot CONFIG SELECTION REGISTRY HELPER OUTPUT"
    );
    let config: Config = aurelius_lanes::fact_lab::load_config(Path::new(&args[1]))?;
    let selection: Selection = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    ensure!(
        selection.quotas.values().sum::<usize>() == 100,
        "expected100-window design"
    );
    ensure!(
        config
            .lanes
            .values()
            .any(|lane| lane.scope == selection.scope),
        "unconfigured source scope"
    );
    let registry: Value = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let output = Path::new(&args[5]);
    ensure!(output.is_dir(), "private output directory required");
    ensure!(
        !output.join("summary.json").exists(),
        "snapshot output already completed"
    );
    let pg = config.postgres.clone().context("Postgres config missing")?;
    let url = std::env::var(&pg.url_env).context("runtime database credential missing")?;
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(2)
        .connect(&url)
        .await
        .map_err(|_| anyhow::anyhow!("read-only snapshot connection failed; details withheld"))?;
    let readonly: String = sqlx::query_scalar("SHOW default_transaction_read_only")
        .fetch_one(&pool)
        .await
        .map_err(|_| anyhow::anyhow!("read-only snapshot verification failed"))?;
    ensure!(
        readonly == "on",
        "snapshot requires database-enforced read-only sessions"
    );
    let sink = PostgresSink::from_pool(
        pg,
        config
            .embedding
            .as_ref()
            .context("embedding shape absent")?
            .dimensions,
        pool.clone(),
    )
    .await?;
    let provider = ProviderMetadataReader::new(&config)?;
    let runtime =
        LanguageRuntime::load(config.language.clone().context("language config absent")?)?;
    let references = &config
        .chat
        .as_ref()
        .context("chat config absent")?
        .references;
    let policy = config
        .fact_quality
        .as_ref()
        .context("preparation policy absent")?;
    let base_profile = aurelius_lanes::facts::ChatExtractor::atomic_profile_for(
        config.chat.as_ref().context("chat config absent")?,
        &config.ontology,
    )?;
    save(
        &output.join("readonly-proof.json"),
        &json!({"database_default_transaction_read_only":readonly,"model_calls":0,"database_writes":0}),
    )?;
    let mut completed = BTreeMap::<String, usize>::new();
    let mut exclusions = Vec::new();
    let mut records = Vec::new();
    for entry in selection.queues {
        if completed.get(&entry.kind).copied().unwrap_or(0) >= selection.quotas[&entry.kind] {
            continue;
        }
        let input = sink
            .extraction_input(&selection.scope, &entry.artifact_id)
            .await?;
        ensure!(
            input.document.key.revision == format!("pg-chunk-projection-v1/{}", entry.revision)
                && i64::try_from(input.ranges.len())? == entry.windows,
            "source changed after inventory"
        );
        let range = *input
            .ranges
            .get(entry.window)
            .context("selected window absent")?;
        let result: Result<Value> = async {
            let properties = if let Some(reader) = &provider { reader.read(&input).await? } else { None };
            let language = runtime.analyze(&input.document)?;
            let aliases = alias_hypotheses(references, &input.document)?;
            let context = FactContext { organization_id:input.document.key.scope.clone(), entities:vec![] };
            // Same file-lab preparation identity, without expanding canonical DB context.
            let profile = blake3::hash(&serde_json::to_vec(&("file-fact-lab-v1",runtime.profile(),config.ontology.semantic_hash()?,references.semantic_hash()?,context.canonical_prefix()?))?).to_hex().to_string();
            let mut prepared = PreparedFactSource::from_analysis(&input.document,language,aliases,context,policy.context_as_of,profile)?;
            if let Some(props) = &properties { prepared = prepared.with_provider_properties(&input.document,props)?; }
            let packet = json!({"document":input.document,"range":range,"prepared":prepared,"registry":registry,"base_profile":base_profile,"max_output_tokens":16384,"body":null,"expected_binding":null});
            let request = native(&args[4], &packet)?;
            Ok(json!({"id":format!("{}-{}",entry.artifact_id,entry.window),"provider_kind":entry.kind,"packet":packet,"request":request["request"],"provider_properties":properties,"source_windows":input.ranges.len()}))
        }.await;
        match result {
            Ok(value) => {
                let id = value["id"].as_str().context("snapshot id absent")?;
                save(&output.join(format!("{id}.json")), &value)?;
                records.push(id.to_owned());
                *completed.entry(entry.kind.clone()).or_default() += 1;
                println!(
                    "{}",
                    json!({"status":"ready","id":id,"kind":entry.kind,"ready":records.len()})
                );
            }
            Err(error) => {
                let diagnostic = format!("{:?}", aurelius_lanes::facts::review_diagnostic(&error));
                exclusions.push(json!({"artifact_id":entry.artifact_id,"window":entry.window,"kind":entry.kind,"diagnostic":diagnostic}));
                println!(
                    "{}",
                    json!({"status":"input_review","id":entry.artifact_id,"diagnostic":diagnostic})
                );
            }
        }
    }
    save(
        &output.join("summary.json"),
        &json!({"ready":records.len(),"records":records,"kind_counts":completed,"quotas":selection.quotas,"exclusions":exclusions,"model_calls":0,"database_writes":0}),
    )?;
    pool.close().await;
    ensure!(
        records.len() == 100,
        "fewer than100eligible windows; inspect source-preparation exclusions"
    );
    Ok(())
}
