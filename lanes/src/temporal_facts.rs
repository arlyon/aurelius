use crate::{
    enrichment::{FactActivities, FactEngine, FactReceipt, FactRequest, FactSource, enrich_once},
    lanes::LaneError,
    temporal_lanes::activity,
};
use anyhow::Result;
use std::{sync::Arc, time::Duration};
use temporalio_client::{Client, WorkflowQueryOptions};
use temporalio_common::RetryPolicy;
use temporalio_macros::{activities, workflow, workflow_methods};
use temporalio_sdk::{
    ActivityExecutionError, ActivityOptions, ApplicationFailure, ContinueAsNewOptions,
    WorkflowContext, WorkflowContextView, WorkflowResult,
    activities::{ActivityContext, ActivityError},
};

pub struct FactWorker(pub FactEngine);
#[activities]
impl FactWorker {
    #[activity]
    pub async fn discover(
        self: Arc<Self>,
        ctx: ActivityContext,
        request: FactRequest,
    ) -> Result<Option<FactSource>, ActivityError> {
        let mut engine = self.0.clone();
        activity(ctx, engine.discover(request)).await
    }
    #[activity]
    pub async fn process(
        self: Arc<Self>,
        ctx: ActivityContext,
        source: FactSource,
    ) -> Result<FactReceipt, ActivityError> {
        let mut engine = self.0.clone();
        activity(ctx, engine.process(source)).await
    }
}
#[workflow]
#[derive(Default)]
pub struct FactObserver {
    last: Option<FactReceipt>,
}
#[workflow_methods]
impl FactObserver {
    #[run]
    pub async fn run(ctx: &mut WorkflowContext<Self>, request: FactRequest) -> WorkflowResult<()> {
        for _ in 0..100 {
            match enrich_once(&mut Bindings(ctx), request.clone()).await {
                Ok(Some(receipt)) => ctx.state_mut(|state| state.last = Some(receipt)),
                Ok(None) | Err(LaneError::Activity(_)) => {
                    ctx.timer(Duration::from_secs(300)).await;
                }
                Err(LaneError::Invalid(_)) => {
                    return Err(ApplicationFailure::non_retryable(anyhow::anyhow!(
                        "invalid fact workflow references"
                    ))
                    .into());
                }
            }
        }
        ctx.continue_as_new(request, ContinueAsNewOptions::default())?;
        Ok(())
    }
    #[query]
    pub fn status(&self, _ctx: &WorkflowContextView) -> Option<FactReceipt> {
        self.last.clone()
    }
}
pub async fn status(client: &Client, id: &str) -> Result<Option<FactReceipt>> {
    Ok(client
        .get_workflow_handle::<fact_observer::Run>(id)
        .query(FactObserver::status, (), WorkflowQueryOptions::default())
        .await?)
}
fn options() -> ActivityOptions {
    ActivityOptions::with_start_to_close_timeout(Duration::from_secs(86400))
        .heartbeat_timeout(Duration::from_secs(60))
        .retry_policy(RetryPolicy::builder().maximum_attempts(1).build())
        .build()
}
struct Bindings<'a>(&'a mut WorkflowContext<FactObserver>);
impl FactActivities for Bindings<'_> {
    type Error = ActivityExecutionError;
    async fn discover(&mut self, request: FactRequest) -> Result<Option<FactSource>, Self::Error> {
        self.0
            .execute_activity(FactWorker::discover, request, options())
            .await
    }
    async fn process(&mut self, source: FactSource) -> Result<FactReceipt, Self::Error> {
        self.0
            .execute_activity(FactWorker::process, source, options())
            .await
    }
}
