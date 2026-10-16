#[cfg(feature = "temporal")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use temporalio_client::WorkflowFetchHistoryOptions;
    use temporalio_sdk::workflow_replayer::{WorkflowReplayer, WorkflowReplayerOptions};
    let args: Vec<_> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() == 2,
        "usage: replay_lane <config.toml> <workflow-id>"
    );
    let config = aurelius_lanes::config::Config::load(std::path::Path::new(&args[0]))?;
    let client = aurelius_lanes::temporal::connect(
        config
            .temporal
            .as_ref()
            .context("Temporal configuration required")?,
    )
    .await?;
    let events = client
        .get_workflow_handle::<temporalio_common::UntypedWorkflow>(&args[1])
        .fetch_history(WorkflowFetchHistoryOptions::default())
        .into_events()
        .await?;
    let history = temporalio_client::WorkflowHistory::from(
        temporalio_common::protos::temporal::api::history::v1::History { events },
    );
    let options = WorkflowReplayerOptions::new()
        .register_workflow::<aurelius_lanes::observer::LaneObserver>()?;
    #[cfg(feature = "postgres")]
    let options = options.register_workflow::<aurelius_lanes::temporal_ingest::PgLaneObserver>()?;
    WorkflowReplayer::new(options.build())?
        .replay_workflow(history)
        .await?;
    println!("lane_history_replay=passed replay_side_effects=none");
    Ok(())
}
#[cfg(not(feature = "temporal"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("replay requires temporal feature")
}
