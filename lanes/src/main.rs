use anyhow::{Context, Result, bail, ensure};
use aurelius_lanes::{
    config::Config,
    facts::ChatExtractor,
    worker::{ExtractionRequest, extract_document},
};
#[cfg(feature = "postgres")]
use aurelius_lanes::{
    ingest::{IngestRequest, PgIngestor, ingest_once},
    lanes::LaneError,
};
use std::path::Path;
use tokio::io::{AsyncBufReadExt, AsyncReadExt};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .json()
        .flatten_event(true)
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "aurelius_lanes=info".into()),
        )
        .init();
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        if args.first().is_some_and(|a| a == "index-record") {
            args.len() == 6
        } else if args.first().is_some_and(|a| a == "pilot-facts") {
            args.len() == 4
        } else if args.first().is_some_and(|a| matches!(
            a.as_str(),
            "facts-once" | "watch-facts" | "fact-context" | "prepare-facts"
        )) {
            args.len() == 3
        } else {
            args.len() == 2
        },
        "usage: aurelius-lanes <validate|extract|store|index-lanes|watch-lanes|worker|start-lanes|notify-lanes|status-lanes|status-legacy-lanes|schedule-lanes|checkpoint-lanes|start-facts|status-facts> <config.toml>; index-record <config.toml> <lane> <job-id> <object-key> <row-index>; pilot-facts <config.toml> <org-id> <source-artifact-id>; facts-once|watch-facts <config.toml> <org-id>"
    );
    let config = Config::load(Path::new(&args[1]))?;
    tracing::info!(
        command = args[0].as_str(),
        ontology_hash = %config.ontology.semantic_hash()?,
        lanes = config.lanes.len(),
        source_revision = option_env!("AURELIUS_SOURCE_REVISION").unwrap_or("unknown"),
        "Aurelius configuration loaded"
    );
    match args[0].as_str() {
        "validate" => {
            println!("ontology_hash={}", config.ontology.semantic_hash()?);
            for name in config.lanes.keys() {
                println!("lane={name} profile={}", config.lane_request(name)?.profile);
            }
            if let Some(chat) = &config.chat {
                println!(
                    "extraction_profile={}",
                    ChatExtractor::profile_for(chat, &config.ontology)?
                );
            }
        }
        #[cfg(all(feature = "postgres", feature = "temporal"))]
        "checkpoint-lanes" => {
            let temporal = config
                .temporal
                .as_ref()
                .context("Temporal configuration required")?;
            let client = aurelius_lanes::temporal::connect(temporal).await?;
            let mut bytes = Vec::new();
            tokio::io::stdin()
                .take(256 * 1024 + 1)
                .read_to_end(&mut bytes)
                .await?;
            ensure!(bytes.len() <= 256 * 1024, "checkpoint input exceeds bound");
            let receipts: Vec<aurelius_lanes::ingest::IngestReceipt> =
                serde_json::from_slice(&bytes)?;
            ensure!(receipts.len() <= config.lanes.len(), "too many checkpoints");
            let sink = aurelius_lanes::postgres::PostgresSink::connect(
                config
                    .postgres
                    .clone()
                    .context("Postgres configuration required")?,
                config
                    .embedding
                    .as_ref()
                    .context("embedding configuration required")?
                    .dimensions,
            )
            .await?;
            let source = aurelius_lanes::source::SourceClient::new(
                config
                    .source
                    .clone()
                    .context("source configuration required")?,
            )?;
            let mut seen = std::collections::BTreeSet::new();
            for receipt in receipts {
                let name = &receipt.epoch.request.lane.lane;
                ensure!(
                    seen.insert(name.clone())
                        && config.lane_request(name)? == receipt.epoch.request.lane,
                    "checkpoint changed lane identity"
                );
                let lane = &config.lanes[name];
                let legacy = format!(
                    "aurelius-pg/{}/{}/{}",
                    lane.scope, name, receipt.epoch.request.lane.profile
                );
                ensure!(
                    !aurelius_lanes::temporal_ingest::is_running(&client, &legacy).await?,
                    "legacy observer must be stopped before importing its completed cursor"
                );
                source
                    .verify(&lane.connection, receipt.epoch.sequence)
                    .await?;
                sink.record_ingestion(&lane.scope, &receipt).await?;
                println!(
                    "lane={name} completed_epoch={} checkpointed",
                    receipt.epoch.sequence
                );
            }
        }
        #[cfg(feature = "postgres")]
        "store" => {
            let postgres = config.postgres.context("Postgres configuration required")?;
            let dimensions = config
                .embedding
                .context("embedding configuration required")?
                .dimensions;
            let sink =
                aurelius_lanes::postgres::PostgresSink::connect(postgres, dimensions).await?;
            let mut input = tokio::io::BufReader::new(tokio::io::stdin());
            loop {
                let mut line = String::new();
                let read = (&mut input)
                    .take(64 * 1024 * 1024 + 1)
                    .read_line(&mut line)
                    .await?;
                if read == 0 {
                    break;
                }
                ensure!(
                    line.len() <= 64 * 1024 * 1024,
                    "indexed document input exceeds bound"
                );
                if line.trim().is_empty() {
                    continue;
                }
                let document: aurelius_lanes::indexing::IndexedDocument =
                    serde_json::from_str(&line).map_err(|_| {
                        anyhow::anyhow!("invalid indexed document input; source details withheld")
                    })?;
                println!("{}", serde_json::to_string(&sink.write(&document).await?)?);
            }
        }
        #[cfg(feature = "postgres")]
        "index-record" => {
            let engine = PgIngestor::new(config.clone()).await?;
            let artifact = engine
                .index_record(&args[2], args[3].parse()?, &args[4], args[5].parse()?)
                .await?;
            println!(
                "{}",
                serde_json::json!({"source_artifact":artifact,"status":"indexed"})
            );
        }
        #[cfg(feature = "postgres")]
        "pilot-facts" => {
            use aurelius_lanes::{contracts::Chunk, postgres::PostgresSink};
            let settings = config
                .chat
                .as_ref()
                .context("chat configuration required")?;
            ensure!(
                settings.gateway.is_some(),
                "fact pilot requires an authenticated gateway; direct-provider fallback is disabled"
            );
            let sink = PostgresSink::connect(
                config
                    .postgres
                    .clone()
                    .context("Postgres configuration required")?,
                config
                    .embedding
                    .as_ref()
                    .context("embedding configuration required")?
                    .dimensions,
            )
            .await?;
            let input = sink.extraction_input(&args[2], &args[3]).await?;
            let references = settings.references.resolve(&input.document)?;
            let extractor = ChatExtractor::new(settings, config.ontology)?;
            for range in input.ranges.into_iter().take(2) {
                let chunk = Chunk {
                    document: input.document.clone(),
                    range,
                };
                if sink.candidate_exists(&chunk, &extractor.profile).await? {
                    println!(
                        "{}",
                        serde_json::json!({"candidate_artifact":PostgresSink::candidate_id(&chunk,&extractor.profile)?,"status":"already_stored"})
                    );
                    continue;
                }
                let graph = extractor.extract(&chunk).await?;
                let artifact = sink.write_candidates(&chunk, &graph, &references).await?;
                println!(
                    "{}",
                    serde_json::json!({"candidate_artifact":artifact,"claims":graph.claims.len(),"org_id":graph.document.scope,"status":"candidate"})
                );
            }
        }
        #[cfg(feature = "postgres")]
        "facts-once" | "watch-facts" => {
            use aurelius_lanes::enrichment::{FactEngine, enrich_once};
            let mut engine = FactEngine::new(config.clone()).await?;
            let request = engine.request(&args[2])?;
            loop {
                match enrich_once(&mut engine, request.clone()).await {
                    Ok(Some(receipt)) => println!("{}", serde_json::to_string(&receipt)?),
                    Ok(None) => {
                        if args[0] == "watch-facts" {
                            tokio::time::sleep(std::time::Duration::from_secs(300)).await;
                        }
                    }
                    Err(_) => {
                        if args[0] == "facts-once" {
                            bail!("fact processing failed; source details withheld");
                        }
                        tracing::warn!("fact processing failed; retrying after timer");
                        tokio::time::sleep(std::time::Duration::from_secs(300)).await;
                    }
                }
                if args[0] == "facts-once" {
                    break;
                }
            }
        }
        #[cfg(all(feature = "temporal", feature = "postgres"))]
        "start-facts" | "status-facts" => {
            use temporalio_client::{WorkflowIdConflictPolicy, WorkflowStartOptions};
            let temporal = config
                .temporal
                .as_ref()
                .context("Temporal configuration required")?;
            let client = aurelius_lanes::temporal::connect(temporal).await?;
            for request in aurelius_lanes::enrichment::FactEngine::requests(&config)? {
                let id = format!("aurelius-facts/{}/{}", request.scope, request.profile);
                if args[0] == "start-facts" {
                    client
                        .start_workflow(
                            aurelius_lanes::temporal_facts::FactObserver::run,
                            request,
                            WorkflowStartOptions::new(&temporal.task_queue, &id)
                                .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
                                .build(),
                        )
                        .await?;
                    println!("workflow={id} started_or_existing");
                } else {
                    println!(
                        "{}",
                        serde_json::json!({"workflow":id,"status":aurelius_lanes::temporal_facts::status(&client,&id).await?})
                    );
                }
            }
        }
        #[cfg(all(feature = "postgres", feature = "language"))]
        "prepare-facts" => {
            use std::io::Read;
            let mut bytes = Vec::new();
            std::fs::File::open(&args[2])?
                .take(8 * 1024 * 1024 + 1)
                .read_to_end(&mut bytes)?;
            ensure!(
                bytes.len() <= 8 * 1024 * 1024,
                "fact preparation request exceeds bound"
            );
            let request: aurelius_lanes::fact_preparation::PreparationRequest =
                serde_json::from_slice(&bytes)?;
            let sink = aurelius_lanes::postgres::PostgresSink::connect(
                config
                    .postgres
                    .clone()
                    .context("Postgres configuration required")?,
                config
                    .embedding
                    .as_ref()
                    .context("embedding configuration required")?
                    .dimensions,
            )
            .await?;
            let preparer =
                aurelius_lanes::fact_preparation::FactPreparer::with_sink(&config, sink)?;
            println!(
                "{}",
                serde_json::to_string(
                    &preparer
                        .prepare(request.document, request.effective_at)
                        .await?
                )?
            );
        }
        #[cfg(feature = "postgres")]
        "fact-context" => {
            use std::io::Read;
            let mut bytes = Vec::new();
            std::fs::File::open(&args[2])?
                .take(65537)
                .read_to_end(&mut bytes)?;
            ensure!(bytes.len() <= 65536, "fact-context request exceeds bound");
            let request: aurelius_lanes::fact_context::ContextRequest =
                serde_json::from_slice(&bytes)?;
            request.validate()?;
            ensure!(
                config
                    .lanes
                    .values()
                    .any(|lane| lane.scope == request.organization_id),
                "fact-context organization is not configured"
            );
            let settings = config
                .fact_context
                .as_ref()
                .context("fact-context configuration required")?;
            let sink = aurelius_lanes::postgres::PostgresSink::connect(
                config
                    .postgres
                    .clone()
                    .context("Postgres configuration required")?,
                config
                    .embedding
                    .as_ref()
                    .context("embedding configuration required")?
                    .dimensions,
            )
            .await?;
            println!(
                "{}",
                sink.fact_context(settings, &request)
                    .await?
                    .canonical_prefix()?
            );
        }
        "worker" => {
            #[cfg(feature = "temporal")]
            aurelius_lanes::temporal::run_worker(&config).await?;
            #[cfg(not(feature = "temporal"))]
            bail!("worker requires temporal build feature");
        }
        #[cfg(all(feature = "temporal", feature = "postgres"))]
        "start-lanes"
        | "notify-lanes"
        | "status-lanes"
        | "status-legacy-lanes"
        | "schedule-lanes" => {
            let temporal = config
                .temporal
                .as_ref()
                .context("Temporal configuration required")?;
            let client = aurelius_lanes::temporal::connect(temporal).await?;
            if args[0] == "schedule-lanes" {
                let id = aurelius_lanes::temporal_ready::create_schedule(&client, &config).await?;
                println!("schedule={id} interval=6h paused_pending_cutover");
                return Ok(());
            }
            for name in config.lanes.keys() {
                let request = IngestRequest {
                    lane: config.lane_request(name)?,
                    after: None,
                };
                let id = aurelius_lanes::temporal_ready::workflow_id(
                    &config.lanes[name].scope,
                    &request,
                );
                if matches!(args[0].as_str(), "start-lanes" | "notify-lanes") {
                    aurelius_lanes::temporal_ready::wake(&client, &config, request).await?;
                    println!("lane={name} workflow={id} notified");
                } else {
                    let (id, status) = if args[0] == "status-legacy-lanes" {
                        let legacy = format!(
                            "aurelius-pg/{}/{}/{}",
                            config.lanes[name].scope, name, request.lane.profile
                        );
                        let status =
                            aurelius_lanes::temporal_ingest::status(&client, &legacy).await?;
                        (legacy, status)
                    } else {
                        let status = aurelius_lanes::temporal_ready::status(&client, &id).await?;
                        (id, status)
                    };
                    println!(
                        "{}",
                        serde_json::json!({"lane":name,"workflow":id,"status":status})
                    );
                }
            }
        }
        #[cfg(feature = "postgres")]
        "index-lanes" | "watch-lanes" => {
            let mut engine = PgIngestor::new(config.clone()).await?;
            let mut requests = config
                .lanes
                .keys()
                .map(|name| Ok((name.clone(), engine.request(name)?)))
                .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
            let mut timer = tokio::time::interval(std::time::Duration::from_secs(300));
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                timer.tick().await;
                let mut failed = false;
                for name in config.lanes.keys() {
                    let request = requests.get(name).context("lane cursor missing")?.clone();
                    match ingest_once(&mut engine, request).await {
                        Ok(Some(receipt)) => {
                            requests.insert(
                                name.clone(),
                                IngestRequest {
                                    lane: receipt.epoch.request.lane.clone(),
                                    after: Some(receipt.epoch.sequence),
                                },
                            );
                            tracing::info!(lane=%name,indexed=receipt.indexed,deferred=receipt.deferred,ignored=receipt.ignored,"lane epoch stored in Postgres")
                        }
                        Ok(None) => {
                            tracing::info!(lane=%name,"lane waiting for successful source epoch")
                        }
                        Err(LaneError::Invalid(_)) | Err(LaneError::Activity(_)) => {
                            failed = true;
                            tracing::error!(lane=%name,"lane processing failed; source details withheld");
                        }
                    }
                }
                if args[0] == "index-lanes" {
                    ensure!(!failed, "some lanes failed; inspect operator metrics");
                    break;
                }
            }
        }
        "extract" => {
            let settings = config.chat.context("chat configuration required")?;
            let mut extractor = ChatExtractor::new(&settings, config.ontology)?;
            let mut input = tokio::io::BufReader::new(tokio::io::stdin());
            loop {
                let mut line = String::new();
                let read = (&mut input)
                    .take(2 * 1024 * 1024 + 1)
                    .read_line(&mut line)
                    .await?;
                if read == 0 {
                    break;
                }
                ensure!(
                    line.len() <= 2 * 1024 * 1024,
                    "extraction input exceeds bound"
                );
                if line.trim().is_empty() {
                    continue;
                }
                let request: ExtractionRequest =
                    serde_json::from_str(&line).context("invalid extraction request")?;
                println!(
                    "{}",
                    serde_json::to_string(&extract_document(&mut extractor, request).await?)?
                );
            }
        }
        _ => bail!("unknown worker command"),
    }
    Ok(())
}
