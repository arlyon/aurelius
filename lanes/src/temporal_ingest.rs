use crate::{
    ingest::{
        IngestActivities, IngestEpoch, IngestReceipt, IngestRequest, PgIngestor, ingest_once,
    },
    lanes::LaneError,
    temporal_lanes::{activity, options},
};
use anyhow::Result;
use std::{sync::Arc, time::Duration};
use temporalio_client::{Client, WorkflowQueryOptions};
use temporalio_macros::{activities, workflow, workflow_methods};
use temporalio_sdk::{
    ActivityExecutionError, ApplicationFailure, ContinueAsNewOptions, WorkflowContext,
    WorkflowContextView, WorkflowResult,
    activities::{ActivityContext, ActivityError},
};

pub struct PgWorker(pub PgIngestor);
#[activities]
impl PgWorker {
    #[activity]
    pub async fn discover(
        self: Arc<Self>,
        ctx: ActivityContext,
        request: IngestRequest,
    ) -> Result<Option<u64>, ActivityError> {
        let mut ingest = self.0.clone();
        activity(ctx, ingest.discover(request)).await
    }
    #[activity]
    pub async fn ingest(
        self: Arc<Self>,
        ctx: ActivityContext,
        epoch: IngestEpoch,
    ) -> Result<IngestReceipt, ActivityError> {
        let mut ingest = self.0.clone();
        activity(ctx, ingest.ingest(epoch)).await
    }
}
#[workflow]
#[derive(Default)]
pub struct PgLaneObserver {
    last: Option<IngestReceipt>,
}
#[workflow_methods]
impl PgLaneObserver {
    #[run]
    pub async fn run(
        ctx: &mut WorkflowContext<Self>,
        mut request: IngestRequest,
    ) -> WorkflowResult<()> {
        for _ in 0..100 {
            match ingest_once(&mut Bindings(ctx), request.clone()).await {
                Ok(Some(receipt)) => {
                    request.after = Some(receipt.epoch.sequence);
                    ctx.state_mut(|state| state.last = Some(receipt));
                }
                Ok(None) => {}
                Err(LaneError::Activity(_)) => {}
                Err(LaneError::Invalid(_)) => {
                    return Err(ApplicationFailure::non_retryable(anyhow::anyhow!(
                        "invalid PG ingestion references"
                    ))
                    .into());
                }
            }
            ctx.timer(Duration::from_secs(300)).await;
        }
        ctx.continue_as_new(request, ContinueAsNewOptions::default())?;
        Ok(())
    }
    #[query]
    pub fn status(&self, _ctx: &WorkflowContextView) -> Option<IngestReceipt> {
        self.last.clone()
    }
}
pub async fn is_running(client: &Client, id: &str) -> Result<bool> {
    use temporalio_client::WorkflowExecutionStatus;
    use temporalio_client::{WorkflowDescribeOptions, errors::WorkflowInteractionError};
    match client
        .get_workflow_handle::<pg_lane_observer::Run>(id)
        .describe(WorkflowDescribeOptions::default())
        .await
    {
        Ok(description) => Ok(description.status() == WorkflowExecutionStatus::Running),
        Err(WorkflowInteractionError::NotFound(_)) => Ok(false),
        Err(_) => anyhow::bail!("cannot verify legacy observer state"),
    }
}
pub async fn status(client: &Client, id: &str) -> Result<Option<IngestReceipt>> {
    Ok(client
        .get_workflow_handle::<pg_lane_observer::Run>(id)
        .query(PgLaneObserver::status, (), WorkflowQueryOptions::default())
        .await?)
}
pub(crate) struct Bindings<'a, W>(pub(crate) &'a mut WorkflowContext<W>);
impl<W> IngestActivities for Bindings<'_, W> {
    type Error = ActivityExecutionError;
    async fn discover(&mut self, r: IngestRequest) -> Result<Option<u64>, Self::Error> {
        self.0
            .execute_activity(PgWorker::discover, r, options(false))
            .await
    }
    async fn ingest(&mut self, r: IngestEpoch) -> Result<IngestReceipt, Self::Error> {
        self.0
            .execute_activity(PgWorker::ingest, r, options(true))
            .await
    }
}
