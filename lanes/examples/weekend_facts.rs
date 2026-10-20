use anyhow::{Context, Result, ensure};
use aurelius_lanes::{
    config::Config,
    enrichment::{FactEngine, enrich_once},
    run_budget::{RunBudget, RunStopped},
};
use std::{path::Path, time::Duration};
#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "aurelius_lanes=info".into()),
        )
        .init();
    if let Err(error) = execute().await {
        tracing::error!("bounded extraction failed; source/credential details withheld");
        if error.downcast_ref::<RunStopped>().is_none() {
            std::process::exit(1);
        }
    }
}
async fn execute() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 6,
        "usage: weekend_facts CONFIG SCOPE LEDGER MAX_CENTS MAX_CALLS DURATION_SECONDS"
    );
    let config = Config::load(Path::new(&args[0]))?;
    ensure!(
        config.fact_quality.is_some(),
        "review-only typed quality policy required"
    );
    let chat = config.chat.as_ref().context("chat required")?;
    ensure!(
        chat.model == "gpt-6-luna"
            && chat.gateway.is_some()
            && chat.endpoint.contains("/default/openai/responses"),
        "weekend run requires the approved model/default gateway"
    );
    let identity = blake3::hash(&serde_json::to_vec(&(&config, &args[1]))?)
        .to_hex()
        .to_string();
    let budget = RunBudget::open(
        Path::new(&args[2]),
        identity,
        args[3].parse()?,
        args[4].parse()?,
        Duration::from_secs(args[5].parse()?),
    )?;
    let mut engine = FactEngine::new(config)
        .await?
        .with_run_budget(budget.clone());
    let request = engine.request(&args[1])?;
    loop {
        if budget.check().is_err() {
            break;
        }
        match enrich_once(&mut engine, request.clone()).await {
            Ok(Some(receipt)) => println!(
                "{}",
                serde_json::json!({"receipt":receipt,"budget":budget.snapshot()?})
            ),
            Ok(None) => {
                let until = (budget.snapshot()?.deadline - chrono::Utc::now())
                    .to_std()
                    .unwrap_or_default();
                tokio::time::sleep(until.min(Duration::from_secs(60))).await;
            }
            Err(error) => {
                let stopped = match &error {
                    aurelius_lanes::lanes::LaneError::Activity(error)
                    | aurelius_lanes::lanes::LaneError::Invalid(error) => {
                        error.downcast_ref::<RunStopped>().is_some()
                    }
                };
                if stopped {
                    break;
                } else {
                    anyhow::bail!("bounded extraction activity failed; no automatic retry");
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"status":"stopped","budget":budget.snapshot()?})
    );
    Ok(())
}
