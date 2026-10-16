use crate::{
    ingest::{IngestReceipt, IngestRequest, ingest_once},
    lanes::LaneError,
    temporal_ingest::Bindings,
};
use anyhow::{Result, ensure};
use std::{sync::Arc, time::Duration};
use temporalio_client::schedules::{
    CreateScheduleOptions, ScheduleAction, ScheduleOverlapPolicy, ScheduleSpec,
};
use temporalio_client::{
    Client, WorkflowIdConflictPolicy, WorkflowIdReusePolicy, WorkflowStartOptions,
};
use temporalio_macros::{activities, workflow, workflow_methods};
use temporalio_sdk::{
    ApplicationFailure, ContinueAsNewOptions, SyncWorkflowContext, WorkflowContext,
    WorkflowContextView, WorkflowResult,
    activities::{ActivityContext, ActivityError},
};

#[workflow]
#[derive(Default)]
pub struct PgLaneRun {
    generation: u64,
    last: Option<IngestReceipt>,
}
#[workflow_methods]
impl PgLaneRun {
    #[run]
    pub async fn run(
        ctx: &mut WorkflowContext<Self>,
        mut request: IngestRequest,
    ) -> WorkflowResult<()> {
        for _ in 0..100 {
            let generation = ctx.state(|state| state.generation);
            match ingest_once(&mut Bindings(ctx),request.clone()).await {
                Ok(Some(receipt))=> {request.after=Some(receipt.epoch.sequence);ctx.state_mut(|state|state.last=Some(receipt));}
                Ok(None)=>if ctx.state(|state|state.generation)==generation {return Ok(())},
                Err(LaneError::Activity(_))=>return Err(ApplicationFailure::non_retryable(anyhow::anyhow!("PG ingestion activity exhausted retries; next notification or reconciliation may retry")).into()),
                Err(LaneError::Invalid(_))=>return Err(ApplicationFailure::non_retryable(anyhow::anyhow!("invalid PG ingestion references")).into()),
            }
        }
        ctx.continue_as_new(request, ContinueAsNewOptions::default())?;
        Ok(())
    }
    #[signal]
    pub fn ready(&mut self, _ctx: &mut SyncWorkflowContext<Self>) {
        self.generation = self.generation.saturating_add(1);
    }
    #[query]
    pub fn status(&self, _ctx: &WorkflowContextView) -> Option<IngestReceipt> {
        self.last.clone()
    }
}

pub async fn status(client: &Client, id: &str) -> Result<Option<IngestReceipt>> {
    Ok(client
        .get_workflow_handle::<pg_lane_run::Run>(id)
        .query(
            PgLaneRun::status,
            (),
            temporalio_client::WorkflowQueryOptions::default(),
        )
        .await?)
}
pub fn workflow_id(scope: &str, request: &IngestRequest) -> String {
    format!(
        "aurelius-pg-ready/{}/{}/{}",
        scope, request.lane.lane, request.lane.profile
    )
}
pub async fn wake(
    client: &Client,
    config: &crate::config::Config,
    request: IngestRequest,
) -> Result<String> {
    let name = &request.lane.lane;
    ensure!(
        request.after.is_none() && config.lane_request(name)? == request.lane,
        "notification changed lane identity"
    );
    let lane = &config.lanes[name];
    let temporal = config
        .temporal
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Temporal configuration required"))?;
    let legacy = format!(
        "aurelius-pg/{}/{}/{}",
        lane.scope, name, request.lane.profile
    );
    ensure!(
        !crate::temporal_ingest::is_running(client, &legacy).await?,
        "legacy polling observer is still running; cutover required before notification starts"
    );
    let id = workflow_id(&lane.scope, &request);
    client
        .signal_with_start_workflow(
            PgLaneRun::run,
            request,
            PgLaneRun::ready,
            (),
            WorkflowStartOptions::new(&temporal.task_queue, &id)
                .id_reuse_policy(WorkflowIdReusePolicy::AllowDuplicate)
                .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
                .build(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("Temporal lane notification failed"))?;
    Ok(id)
}
#[derive(Clone)]
pub struct ReconcileActivities {
    pub client: Client,
    pub config: crate::config::Config,
}
#[activities]
impl ReconcileActivities {
    #[activity]
    pub async fn notify(
        self: Arc<Self>,
        ctx: ActivityContext,
        requests: Vec<IngestRequest>,
    ) -> Result<Vec<String>, ActivityError> {
        let operation = async {
            ensure!(
                requests.len() == self.config.lanes.len(),
                "reconciliation lane set changed"
            );
            let mut seen = std::collections::BTreeSet::new();
            let mut ids = Vec::new();
            for request in requests {
                ensure!(
                    seen.insert(request.lane.lane.clone()),
                    "duplicate reconciliation lane"
                );
                ids.push(wake(&self.client, &self.config, request).await?);
            }
            Ok(ids)
        };
        crate::temporal_lanes::activity(ctx, operation).await
    }
}
#[workflow]
#[derive(Default)]
pub struct PgReconcile;
#[workflow_methods]
impl PgReconcile {
    #[run]
    pub async fn run(
        ctx: &mut WorkflowContext<Self>,
        requests: Vec<IngestRequest>,
    ) -> WorkflowResult<()> {
        ctx.execute_activity(
            ReconcileActivities::notify,
            requests,
            crate::temporal_lanes::options(false),
        )
        .await?;
        Ok(())
    }
}
pub async fn create_schedule(client: &Client, config: &crate::config::Config) -> Result<String> {
    let temporal = config
        .temporal
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Temporal configuration required"))?;
    let requests = config
        .lanes
        .keys()
        .map(|name| {
            Ok(IngestRequest {
                lane: config.lane_request(name)?,
                after: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!requests.is_empty(), "no configured lanes");
    let id = format!("aurelius-pg-reconcile/{}", temporal.task_queue);
    client.create_schedule(&id,CreateScheduleOptions::builder().action(ScheduleAction::start_workflow(PgReconcile::run,requests,&temporal.task_queue,"aurelius-pg-reconciliation")).spec(ScheduleSpec::from_interval(Duration::from_secs(6*60*60))).overlap_policy(ScheduleOverlapPolicy::Skip).paused(true).note("Await verified polling-observer handoff before unpausing; ingestion only, never fact extraction").build()).await.map_err(|_|anyhow::anyhow!("schedule creation failed; inspect existing schedule before retrying"))?;
    Ok(id)
}
