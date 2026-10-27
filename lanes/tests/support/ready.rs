use super::*;
use anyhow::Context;
use aurelius_lanes::{
    ingest::{IngestEpoch, IngestReceipt, IngestRequest},
    lanes::LaneRequest,
    temporal_ready::PgLaneRun,
};
use temporalio_client::{WorkflowIdConflictPolicy, WorkflowIdReusePolicy};
use temporalio_macros::activities;
use temporalio_sdk::activities::{ActivityContext, ActivityError};
#[derive(Default)]
struct State {
    pending: AtomicUsize,
    ingested: AtomicUsize,
    discoveries: AtomicUsize,
    blocked: AtomicBool,
    empty: tokio::sync::Notify,
    release: tokio::sync::Notify,
}
struct Bound(Arc<State>);
#[activities]
impl Bound {
    #[activity(name = "PgWorker::discover")]
    async fn discover(
        self: Arc<Self>,
        _: ActivityContext,
        _request: IngestRequest,
    ) -> Result<Option<u64>, ActivityError> {
        self.0.discoveries.fetch_add(1, Ordering::SeqCst);
        let next = self.0.pending.load(Ordering::SeqCst);
        if next == 0 && !self.0.blocked.swap(true, Ordering::SeqCst) {
            self.0.empty.notify_one();
            self.0.release.notified().await;
        }
        Ok((next > 0).then_some(next as u64))
    }
    #[activity(name = "PgWorker::ingest")]
    async fn ingest(
        self: Arc<Self>,
        _: ActivityContext,
        epoch: IngestEpoch,
    ) -> Result<IngestReceipt, ActivityError> {
        self.0.ingested.fetch_add(1, Ordering::SeqCst);
        self.0.pending.store(0, Ordering::SeqCst);
        Ok(IngestReceipt {
            epoch,
            indexed: 1,
            deferred: 0,
            ignored: 0,
        })
    }
}
#[tokio::test]
async fn notifications_drain_complete_race_and_replay_without_idle_timers() -> Result<()> {
    ensure!(
        Bound::discover.name() == aurelius_lanes::temporal_ingest::PgWorker::discover.name()
            && Bound::ingest.name() == aurelius_lanes::temporal_ingest::PgWorker::ingest.name()
    );
    let cli = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|path| path.join("temporal"))
        .find(|path| path.is_file())
        .context("Temporal CLI required")?;
    let env = WorkflowEnvironment::start_local(
        LocalWorkflowEnvironmentOptions::builder()
            .server_executable(EphemeralExe::ExistingPath(
                cli.to_string_lossy().into_owned(),
            ))
            .extra_args(vec!["--ip".into(), "127.0.0.1".into()])
            .build(),
    )
    .await?;
    let client = env.client().clone();
    let runtime = Runtime::from_current_tokio(Default::default())?;
    let state = Arc::new(State::default());
    state.pending.store(1, Ordering::SeqCst);
    let mut worker = Worker::new(
        &runtime,
        client.clone(),
        WorkerOptions::new("pg-ready-test")
            .register_workflow::<PgLaneRun>()?
            .register_activities(Bound(state.clone()))
            .build(),
    )?;
    let shutdown = worker.shutdown_handle();
    let exercise = async {
        let result = tokio::time::timeout(Duration::from_secs(45), async {
            let request = IngestRequest {
                lane: LaneRequest {
                    lane: "fixture".into(),
                    profile: "a".repeat(64),
                },
                after: None,
            };
            let options = || {
                WorkflowStartOptions::new("pg-ready-test", "same-lane")
                    .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
                    .id_reuse_policy(WorkflowIdReusePolicy::AllowDuplicate)
                    .build()
            };
            let handle = client
                .signal_with_start_workflow(
                    PgLaneRun::run,
                    request.clone(),
                    PgLaneRun::ready,
                    (),
                    options(),
                )
                .await?;
            // The second readiness event lands after a discovery observed no work,
            // but before that activity returns: it must not get stranded by completion.
            state.empty.notified().await;
            state.pending.store(2, Ordering::SeqCst);
            let same = client
                .signal_with_start_workflow(
                    PgLaneRun::run,
                    request.clone(),
                    PgLaneRun::ready,
                    (),
                    options(),
                )
                .await?;
            ensure!(
                handle.run_id() == same.run_id(),
                "concurrent notification started a second writer"
            );
            state.release.notify_one();
            handle
                .get_result(WorkflowGetResultOptions::default())
                .await?;
            ensure!(
                state.ingested.load(Ordering::SeqCst) == 2,
                "ready-during-drain lost work or duplicated ingestion"
            );
            let events = handle
                .fetch_history(WorkflowFetchHistoryOptions::default())
                .into_events()
                .await?;
            ensure!(
                !events.iter().any(|event| matches!(
                    event.attributes,
                    Some(Attributes::TimerStartedEventAttributes(_))
                )),
                "bounded runner created an idle timer"
            );
            let before = state.discoveries.load(Ordering::SeqCst);
            WorkflowReplayer::new(
                WorkflowReplayerOptions::new()
                    .register_workflow::<PgLaneRun>()?
                    .build(),
            )?
            .replay_workflow(temporalio_client::WorkflowHistory::from(
                temporalio_common::protos::temporal::api::history::v1::History { events },
            ))
            .await?;
            ensure!(
                state.discoveries.load(Ordering::SeqCst) == before,
                "replay performed source I/O"
            );
            let idle = client
                .signal_with_start_workflow(
                    PgLaneRun::run,
                    request,
                    PgLaneRun::ready,
                    (),
                    options(),
                )
                .await?;
            idle.get_result(WorkflowGetResultOptions::default()).await?;
            ensure!(
                state.ingested.load(Ordering::SeqCst) == 2
                    && state.discoveries.load(Ordering::SeqCst) == before + 1,
                "idle reconciliation did not finish after one discovery"
            );
            Ok::<_, anyhow::Error>(())
        })
        .await;
        shutdown();
        result?
    };
    let (result, run) = tokio::join!(exercise, worker.run());
    result?;
    run?;
    env.shutdown().await?;
    Ok(())
}
