use crate::{
    engine::LaneEngine,
    lanes::{
        EpochRef, LaneActivities, LaneError, LaneReceipt, LaneRequest, PublicationRef, index_lane,
    },
};
use anyhow::Result;
use std::{
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};
use temporalio_common::RetryPolicy;
use temporalio_macros::{activities, workflow, workflow_methods};
use temporalio_sdk::{
    ActivityExecutionError, ActivityOptions, ApplicationFailure, WorkflowContext, WorkflowResult,
    activities::{ActivityContext, ActivityError},
};
use tracing::Instrument;

pub struct LaneWorker(pub LaneEngine);
pub(crate) async fn activity<T>(
    ctx: ActivityContext,
    work: impl Future<Output = Result<T>>,
) -> Result<T, ActivityError> {
    let info = ctx.info();
    let span = tracing::info_span!(
        "lane_activity",
        workflow_id = info.workflow_id.as_deref().unwrap_or(""),
        run_id = info.workflow_run_id.as_deref().unwrap_or(""),
        activity_id = %info.activity_id,
        activity_type = %info.activity_type,
        attempt = info.attempt
    );
    async move {
        let started = Instant::now();
        tracing::info!("lane activity started");
        let result = async move {
            tokio::pin!(work);
            let mut heartbeat = tokio::time::interval(Duration::from_secs(20));
            loop {
                tokio::select! {
                    _=ctx.cancelled()=>return Err(ActivityError::cancelled()),
                    result=&mut work=>return result.map_err(|error| {
                        let code=crate::ingest::failure_code(&error);
                        tracing::warn!(failure_code=%code,"lane activity failed");
                        ActivityError::from(anyhow::anyhow!("lane activity failed ({code})"))
                    }),
                    _=heartbeat.tick()=>ctx.record_heartbeat(()).await?,
                }
            }
        }
        .await;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            outcome = if result.is_ok() { "complete" } else { "failed" },
            "lane activity finished"
        );
        result
    }
    .instrument(span)
    .await
}
#[activities]
impl LaneWorker {
    #[activity]
    pub async fn discover(
        self: Arc<Self>,
        ctx: ActivityContext,
        request: LaneRequest,
    ) -> Result<Option<EpochRef>, ActivityError> {
        let mut engine = self.0.clone();
        activity(ctx, engine.discover(request)).await
    }
    #[activity]
    pub async fn publish(
        self: Arc<Self>,
        ctx: ActivityContext,
        epoch: EpochRef,
    ) -> Result<PublicationRef, ActivityError> {
        let mut engine = self.0.clone();
        activity(ctx, engine.publish(epoch)).await
    }
    #[activity]
    pub async fn index(
        self: Arc<Self>,
        ctx: ActivityContext,
        publication: PublicationRef,
    ) -> Result<LaneReceipt, ActivityError> {
        let mut engine = self.0.clone();
        activity(ctx, engine.index(publication)).await
    }
    #[activity]
    pub async fn acknowledge(
        self: Arc<Self>,
        ctx: ActivityContext,
        receipt: LaneReceipt,
    ) -> Result<(), ActivityError> {
        let mut engine = self.0.clone();
        activity(ctx, engine.acknowledge(receipt)).await
    }
}
#[workflow]
#[derive(Default)]
pub struct LaneWorkflow;
#[workflow_methods]
impl LaneWorkflow {
    #[run]
    pub async fn run(
        ctx: &mut WorkflowContext<Self>,
        request: LaneRequest,
    ) -> WorkflowResult<Option<LaneReceipt>> {
        index_lane(&mut Bindings(ctx), request)
            .await
            .map_err(|error| match error {
                LaneError::Activity(error) => error.into(),
                LaneError::Invalid(_) => ApplicationFailure::non_retryable(anyhow::anyhow!(
                    "invalid lane workflow references"
                ))
                .into(),
            })
    }
}
struct Bindings<'a>(&'a mut WorkflowContext<LaneWorkflow>);
pub(crate) fn options(extended_index_budget: bool) -> ActivityOptions {
    ActivityOptions::with_start_to_close_timeout(Duration::from_secs(if extended_index_budget {
        86400
    } else {
        3600
    }))
    .heartbeat_timeout(Duration::from_secs(60))
    .retry_policy(
        RetryPolicy::builder()
            .maximum_attempts(3)
            .initial_interval(Duration::from_secs(10))
            .build(),
    )
    .build()
}
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
