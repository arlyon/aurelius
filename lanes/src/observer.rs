use crate::{
    lanes::{
        EpochRef, LaneActivities, LaneError, LaneReceipt, LaneRequest, PublicationRef, index_lane,
    },
    temporal_lanes::{LaneWorker, options},
};
use anyhow::Result;
use std::time::Duration;
use temporalio_macros::{workflow, workflow_methods};
use temporalio_sdk::{
    ActivityExecutionError, ApplicationFailure, ContinueAsNewOptions, WorkflowContext,
    WorkflowContextView, WorkflowResult,
};

#[workflow]
#[derive(Default)]
pub struct LaneObserver {
    last: Option<LaneReceipt>,
}
#[workflow_methods]
impl LaneObserver {
    #[run]
    pub async fn run(ctx: &mut WorkflowContext<Self>, request: LaneRequest) -> WorkflowResult<()> {
        for _ in 0..100 {
            let result = match index_lane(&mut Bindings(ctx), request.clone()).await {
                Ok(result) => result,
                Err(LaneError::Activity(error)) => {
                    if !ctx.patched("lane-observer-resume-v1") {
                        return Err(error.into());
                    }
                    ctx.timer(Duration::from_secs(300)).await;
                    continue;
                }
                Err(LaneError::Invalid(_)) => {
                    return Err(ApplicationFailure::non_retryable(anyhow::anyhow!(
                        "invalid lane references"
                    ))
                    .into());
                }
            };
            if let Some(receipt) = result {
                ctx.state_mut(|state| state.last = Some(receipt));
            }
            ctx.timer(Duration::from_secs(300)).await;
        }
        ctx.continue_as_new(request, ContinueAsNewOptions::default())?;
        Ok(())
    }
    #[query]
    pub fn status(&self, _ctx: &WorkflowContextView) -> Option<LaneReceipt> {
        self.last.clone()
    }
}
pub async fn status(client: &temporalio_client::Client, id: &str) -> Result<Option<LaneReceipt>> {
    Ok(client
        .get_workflow_handle::<lane_observer::Run>(id)
        .query(
            LaneObserver::status,
            (),
            temporalio_client::WorkflowQueryOptions::default(),
        )
        .await?)
}

struct Bindings<'a>(&'a mut WorkflowContext<LaneObserver>);
impl LaneActivities for Bindings<'_> {
    type Error = ActivityExecutionError;
    async fn discover(&mut self, r: LaneRequest) -> Result<Option<EpochRef>, Self::Error> {
        self.0
            .execute_activity(LaneWorker::discover, r, options(false))
            .await
    }
    async fn publish(&mut self, r: EpochRef) -> Result<PublicationRef, Self::Error> {
        self.0
            .execute_activity(LaneWorker::publish, r, options(false))
            .await
    }
    async fn index(&mut self, r: PublicationRef) -> Result<LaneReceipt, Self::Error> {
        let opts = options(self.0.patched("lane-index-budget-v2"));
        self.0.execute_activity(LaneWorker::index, r, opts).await
    }
    async fn acknowledge(&mut self, r: LaneReceipt) -> Result<(), Self::Error> {
        self.0
            .execute_activity(LaneWorker::acknowledge, r, options(false))
            .await
    }
}
