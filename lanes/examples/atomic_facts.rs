use anyhow::{Context, Result, ensure};
use aurelius_lanes::{atomic_lab, atomic_pg, fact_lab};
use std::{collections::BTreeMap, path::PathBuf};

#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--help"] {
        println!(
            "atomic_facts --live --config FILE --registry FILE --scope ORG --output NEW_DIR --seed ID --max-cents N --max-calls N --deadline-seconds N --max-output-tokens N [--request-timeout-seconds 120..600] [--exclude-windows FILE] [--source-manifest FILE]\nRead-only validation: use --preflight instead of --live. Saved-response persistence: --recover --config FILE --registry FILE --scope ORG --responses DIR --record ID --output NEW_DIR (zero model calls).\nCandidate artifacts only. Fresh durable ledger; no retries, overwrites, implicit resume, canonical writes or source mutations."
        );
        return Ok(());
    }
    ensure!(
        matches!(
            args.first().map(String::as_str),
            Some("--live" | "--preflight" | "--recover")
        ),
        "explicit --live, --recover or read-only --preflight required"
    );
    let mut values = BTreeMap::new();
    for pair in args[1..].chunks(2) {
        ensure!(
            pair.len() == 2
                && matches!(
                    pair[0].as_str(),
                    "--config"
                        | "--registry"
                        | "--scope"
                        | "--output"
                        | "--seed"
                        | "--max-cents"
                        | "--max-calls"
                        | "--deadline-seconds"
                        | "--max-output-tokens"
                        | "--request-timeout-seconds"
                        | "--exclude-windows"
                        | "--source-manifest"
                        | "--responses"
                        | "--record"
                ),
            "invalid option"
        );
        ensure!(
            values.insert(pair[0].as_str(), pair[1].as_str()).is_none(),
            "duplicate option"
        );
    }
    let get = |name| values.get(name).copied().context("required option missing");
    let config = fact_lab::load_config(&PathBuf::from(get("--config")?))?;
    let registry = atomic_lab::load_registry(&PathBuf::from(get("--registry")?))?;
    if args[0] == "--recover" {
        let id = atomic_pg::recover(
            &config,
            &registry,
            get("--scope")?,
            &PathBuf::from(get("--responses")?),
            get("--record")?,
            &PathBuf::from(get("--output")?),
        )
        .await?;
        println!(
            "{}",
            serde_json::json!({"artifact_id":id,"model_calls":0,"budget_settlements":0})
        );
        return Ok(());
    }
    if args[0] == "--preflight" {
        let profiles = config
            .lanes
            .iter()
            .filter(|(_, lane)| lane.scope == get("--scope").unwrap_or(""))
            .map(|(name, _)| config.lane_request(name).map(|r| r.profile))
            .collect::<Result<Vec<_>>>()?;
        let sink = aurelius_lanes::postgres::PostgresSink::connect(
            config.postgres.clone().context("postgres")?,
            config
                .embedding
                .as_ref()
                .context("embedding shape")?
                .dimensions,
        )
        .await?;
        let manifest = sink
            .fact_manifest(get("--scope")?, &profiles, get("--seed")?)
            .await?;
        ensure!(!manifest.is_empty(), "no source documents");
        let input = sink
            .extraction_input(get("--scope")?, &manifest[0].artifact_id)
            .await?;
        let properties = if let Some(reader) =
            aurelius_lanes::provider_metadata::ProviderMetadataReader::new(&config)?
        {
            reader.read(&input).await?
        } else {
            None
        };
        aurelius_lanes::language::LanguageRuntime::load(
            config.language.clone().context("language")?,
        )?;
        println!(
            "{}",
            serde_json::json!({"read_only":true,"documents":manifest.len(),"windows":manifest.iter().map(|e|e.windows).sum::<i64>(),"sample_verified_provider_metadata":properties.is_some(),"model_calls":0})
        );
        return Ok(());
    }
    let result = atomic_pg::run(
        &config,
        &registry,
        &atomic_pg::Options {
            scope: get("--scope")?.into(),
            output: PathBuf::from(get("--output")?),
            seed: get("--seed")?.into(),
            max_cents: get("--max-cents")?.parse()?,
            max_calls: get("--max-calls")?.parse()?,
            seconds: get("--deadline-seconds")?.parse()?,
            max_output_tokens: get("--max-output-tokens")?.parse()?,
            request_timeout_seconds: values
                .get("--request-timeout-seconds")
                .copied()
                .unwrap_or("120")
                .parse()?,
            excluded_windows: if let Some(path) = values.get("--exclude-windows") {
                use std::io::Read;
                let mut bytes = Vec::new();
                std::fs::File::open(path)?
                    .take(1024 * 1024 + 1)
                    .read_to_end(&mut bytes)?;
                ensure!(bytes.len() <= 1024 * 1024, "exclusions exceed bound");
                serde_json::from_slice(&bytes)?
            } else {
                vec![]
            },
            source_manifest: values.get("--source-manifest").map(PathBuf::from),
        },
    )
    .await?;
    println!("{}", serde_json::to_string(&result)?);
    ensure!(
        !matches!(
            result.stop_reason,
            Some("provider_or_unsettled_usage" | "local_or_database_failure")
        ),
        "run stopped; inspect private artifacts, do not reset ledger"
    );
    Ok(())
}
