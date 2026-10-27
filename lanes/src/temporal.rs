use crate::{
    contracts::CandidateGraph,
    facts::ChatExtractor,
    worker::{Activities, ExtractionRequest, extract_document},
};
use anyhow::{Result, anyhow};
use std::{sync::Arc, time::Duration};
use temporalio_client::{Client, ClientOptions, ClientTlsOptions, ConnectionOptions, TlsOptions};
use temporalio_common::RetryPolicy;
use temporalio_macros::{activities, workflow, workflow_methods};
use temporalio_sdk::{
    ActivityExecutionError, ActivityOptions, ApplicationFailure, Runtime, Worker, WorkerOptions,
    WorkflowContext, WorkflowResult,
    activities::{ActivityContext, ActivityError},
};

pub struct ExtractionActivities(pub ChatExtractor);
#[activities]
impl ExtractionActivities {
    #[activity]
    pub async fn extract(
        self: Arc<Self>,
        ctx: ActivityContext,
        request: ExtractionRequest,
    ) -> Result<CandidateGraph, ActivityError> {
        if request.profile != self.0.profile || request.chunk.text().is_err() {
            return Err(ActivityError::application(
                ApplicationFailure::non_retryable(anyhow!(
                    "invalid extraction input or unavailable profile"
                )),
            ));
        }
        // Failure histories must not retain upstream responses or source-bearing diagnostics.
        tokio::select! {
            _ = ctx.cancelled() => Err(ActivityError::cancelled()),
            result = self.0.extract(&request.chunk) => {
                result.map_err(|_| ActivityError::from(anyhow!("extraction activity failed")))
            }
        }
    }
}

#[workflow]
#[derive(Default)]
pub struct ExtractionWorkflow;
#[workflow_methods]
impl ExtractionWorkflow {
    #[run]
    pub async fn run(
        ctx: &mut WorkflowContext<Self>,
        request: ExtractionRequest,
    ) -> WorkflowResult<CandidateGraph> {
        extract_document(&mut TemporalActivities(ctx), request)
            .await
            .map_err(Into::into)
    }
}
struct TemporalActivities<'a>(&'a mut WorkflowContext<ExtractionWorkflow>);
impl Activities for TemporalActivities<'_> {
    type Error = ActivityExecutionError;
    async fn extract(&mut self, request: ExtractionRequest) -> Result<CandidateGraph, Self::Error> {
        self.0
            .execute_activity(
                ExtractionActivities::extract,
                request,
                ActivityOptions::with_start_to_close_timeout(Duration::from_secs(150))
                    .retry_policy(RetryPolicy::builder().maximum_attempts(3).build())
                    .build(),
            )
            .await
    }
}

pub async fn run_worker(app: &crate::config::Config) -> Result<()> {
    let config = app
        .temporal
        .as_ref()
        .ok_or_else(|| anyhow!("Temporal configuration required"))?;
    let client = connect(config).await?;
    let runtime = Runtime::from_current_tokio(Default::default())?;
    let mut options = WorkerOptions::new(&config.task_queue);
    #[cfg(feature = "postgres")]
    let mut webhook = None;
    if let Some(chat) = &app.chat {
        options = options
            .register_workflow::<ExtractionWorkflow>()?
            .register_activities(ExtractionActivities(ChatExtractor::new(
                chat,
                app.ontology.clone(),
            )?));
    }
    if !app.lanes.is_empty() {
        #[cfg(feature = "postgres")]
        {
            let ingest = crate::ingest::PgIngestor::new(app.clone()).await?;
            if app.chat.is_some() {
                options = options
                    .register_workflow::<crate::temporal_facts::FactObserver>()?
                    .register_activities(crate::temporal_facts::FactWorker(
                        crate::enrichment::FactEngine::with_sink(app.clone(), ingest.sink())?,
                    ));
            }
            options = options
                .register_workflow::<crate::temporal_ingest::PgLaneObserver>()?
                .register_workflow::<crate::temporal_ready::PgLaneRun>()?
                .register_workflow::<crate::temporal_ready::PgReconcile>()?
                .register_activities(crate::temporal_ready::ReconcileActivities {
                    client: client.clone(),
                    config: app.clone(),
                })
                .register_activities(crate::temporal_ingest::PgWorker(ingest.clone()));
            if let Some(config) = &app.webhook {
                let listener = tokio::net::TcpListener::bind(config.bind).await?;
                let state = crate::notification::NotificationState::new(
                    app.clone(),
                    client.clone(),
                    ingest.sink(),
                )?;
                webhook = Some((listener, state));
            }
        }
        #[cfg(not(feature = "postgres"))]
        return Err(anyhow!(
            "lane ingestion requires the postgres build feature"
        ));
    }
    #[cfg(feature = "postgres")]
    if let Some((listener, state)) = webhook {
        return tokio::select! {
            result = serve_worker(&runtime, client, options.build()) => result,
            result = crate::notification::serve(listener, state) => result,
        };
    }
    serve_worker(&runtime, client, options.build()).await
}

pub async fn connect(config: &crate::config::TemporalConfig) -> Result<Client> {
    config.validate()?;
    let target: reqwest::Url = config.address.parse()?;
    let client_tls = match (&config.client_cert_env, &config.client_key_env) {
        (Some(cert), Some(key)) => Some(
            ClientTlsOptions::builder()
                .client_cert(crate::config::credential(cert)?.into_bytes())
                .client_private_key(crate::config::credential(key)?.into_bytes())
                .build(),
        ),
        _ => None,
    };
    let tls = (target.scheme() == "https").then(|| {
        TlsOptions::builder()
            .maybe_client_tls_options(client_tls)
            .build()
    });
    let api_key = config
        .api_key_env
        .as_deref()
        .map(crate::config::credential)
        .transpose()?;
    let client = Client::connect(
        ConnectionOptions::new(target)
            .maybe_tls_options(tls)
            .maybe_api_key(api_key)
            .connect_timeout(Duration::from_secs(10))
            .build(),
        ClientOptions::new(&config.namespace).build(),
    )
    .await?;
    Ok(client)
}

async fn serve_worker(
    runtime: &Runtime,
    client: Client,
    options: temporalio_sdk::WorkerOptions,
) -> Result<()> {
    let mut worker = Worker::new(runtime, client, options)?;
    let shutdown = worker.shutdown_handle();
    let wait = async {
        #[cfg(unix)]
        {
            let mut terminate =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
            tokio::select! {
                result = tokio::signal::ctrl_c() => result?,
                _ = terminate.recv() => {},
            }
        }
        #[cfg(not(unix))]
        tokio::signal::ctrl_c().await?;
        shutdown();
        Ok::<_, anyhow::Error>(())
    };
    let run = worker.run();
    tokio::pin!(run);
    tokio::select! {
        result = &mut run => result?,
        result = wait => { result?; run.await?; }
    }
    Ok(())
}
