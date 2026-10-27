#[cfg(feature = "postgres")]
#[path = "support/ready.rs"]
mod ready;
use anyhow::{Result, ensure};
use aurelius_lanes::{
    config::{ChatConfig, Config},
    contracts::{ByteRange, Chunk, Document, DocumentKey},
    facts::ChatExtractor,
    temporal::{ExtractionActivities, ExtractionWorkflow},
    worker::{ExtractionRequest, extract_document},
};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use serde_json::{Value, json};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};
use temporalio_client::{
    WorkflowFetchHistoryOptions, WorkflowGetResultOptions, WorkflowStartOptions,
};
use temporalio_common::protos::temporal::api::history::v1::history_event::Attributes;
use temporalio_sdk::{
    Runtime, Worker, WorkerOptions,
    testing::{EphemeralExe, LocalWorkflowEnvironmentOptions, WorkflowEnvironment},
    workflow_replayer::{WorkflowReplayer, WorkflowReplayerOptions},
};

#[cfg(feature = "postgres")]
mod continuous_facts {
    use super::*;
    use anyhow::Context;
    use aurelius_lanes::{
        enrichment::{FactReceipt, FactRequest, FactSource, FactStatus},
        temporal_facts::{FactObserver, FactWorker},
    };
    use temporalio_macros::activities;
    use temporalio_sdk::activities::{ActivityContext, ActivityError};
    #[derive(Default)]
    struct State {
        done: AtomicBool,
        calls: AtomicUsize,
    }
    struct Bound(Arc<State>);
    #[activities]
    impl Bound {
        #[activity(name = "FactWorker::discover")]
        async fn discover(
            self: Arc<Self>,
            _: ActivityContext,
            request: FactRequest,
        ) -> Result<Option<FactSource>, ActivityError> {
            self.0.calls.fetch_add(1, Ordering::SeqCst);
            Ok((!self.0.done.load(Ordering::SeqCst)).then_some(FactSource {
                request,
                artifact_id: "opaque-source-ref".into(),
            }))
        }
        #[activity(name = "FactWorker::process")]
        async fn process(
            self: Arc<Self>,
            _: ActivityContext,
            source: FactSource,
        ) -> Result<FactReceipt, ActivityError> {
            self.0.calls.fetch_add(1, Ordering::SeqCst);
            self.0.done.store(true, Ordering::SeqCst);
            Ok(FactReceipt {
                source,
                status: FactStatus::Complete,
                stored_windows: 1,
                skipped_windows: 0,
                review_windows: 1,
                new_claims: 2,
            })
        }
    }
    #[tokio::test]
    async fn fact_observer_serial_references_idle_timer_and_replay() -> Result<()> {
        ensure!(
            Bound::discover.name() == FactWorker::discover.name()
                && Bound::process.name() == FactWorker::process.name()
        );
        let cli = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .map(|p| p.join("temporal"))
            .find(|p| p.is_file())
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
        let mut worker = Worker::new(
            &runtime,
            client.clone(),
            WorkerOptions::new("fact-loop-test")
                .register_workflow::<FactObserver>()?
                .register_activities(Bound(state.clone()))
                .build(),
        )?;
        let shutdown = worker.shutdown_handle();
        let exercise = async {
            let result = async {
                let handle = client
                    .start_workflow(
                        FactObserver::run,
                        FactRequest {
                            scope: "org-fixture".into(),
                            profile: "a".repeat(64),
                        },
                        WorkflowStartOptions::new("fact-loop-test", "fact-loop-reference-test")
                            .build(),
                    )
                    .await?;
                tokio::time::timeout(Duration::from_secs(20), async {
                    loop {
                        if state.calls.load(Ordering::SeqCst) >= 3 {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                })
                .await?;
                let receipt =
                    aurelius_lanes::temporal_facts::status(&client, "fact-loop-reference-test")
                        .await?
                        .context("fact status missing")?;
                ensure!(receipt.stored_windows == 1 && receipt.review_windows == 1);
                let events = handle
                    .fetch_history(WorkflowFetchHistoryOptions::default())
                    .into_events()
                    .await?;
                let mut idle = false;
                for event in &events {
                    match &event.attributes {
                        Some(Attributes::ActivityTaskScheduledEventAttributes(a)) => {
                            ensure!(
                                a.retry_policy
                                    .as_ref()
                                    .context("retry policy")?
                                    .maximum_attempts
                                    == 1
                            );
                            for payload in &a.input.as_ref().context("activity input")?.payloads {
                                let value: Value = serde_json::from_slice(&payload.data)?;
                                ensure!(payload.data.len() < 4096);
                                ensure!(!["text", "document", "secret", "credential"].iter().any(
                                    |field| value.to_string().contains(&format!("\"{field}\":"))
                                ));
                            }
                        }
                        Some(Attributes::TimerStartedEventAttributes(t)) => {
                            idle |= t
                                .start_to_fire_timeout
                                .as_ref()
                                .is_some_and(|d| d.seconds == 300)
                        }
                        _ => (),
                    }
                }
                ensure!(idle, "idle fact observer did not persist its wait timer");
                let before = state.calls.load(Ordering::SeqCst);
                let history = temporalio_client::WorkflowHistory::from(
                    temporalio_common::protos::temporal::api::history::v1::History { events },
                );
                WorkflowReplayer::new(
                    WorkflowReplayerOptions::new()
                        .register_workflow::<FactObserver>()?
                        .build(),
                )?
                .replay_workflow(history)
                .await?;
                ensure!(
                    state.calls.load(Ordering::SeqCst) == before,
                    "fact replay executed I/O"
                );
                Ok::<_, anyhow::Error>(())
            }
            .await;
            shutdown();
            result
        };
        let (result, run) = tokio::join!(exercise, worker.run());
        result?;
        run?;
        env.shutdown().await?;
        Ok(())
    }
}

#[derive(Default)]
struct Model {
    calls: AtomicUsize,
    fail_next: AtomicBool,
}
async fn model(
    State(state): State<Arc<Model>>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    state.calls.fetch_add(1, Ordering::SeqCst);
    assert_eq!(body["messages"][0]["role"], "system");
    assert!(
        body["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("Configured ontology:")
    );
    assert_eq!(body["messages"][1]["role"], "user");
    if state.fail_next.swap(false, Ordering::SeqCst) {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({})));
    }
    let facts = json!({"facts":[{"subject":"Ada","predicate":"manages","object":"Company","statement_type":"observation","polarity":"positive","modality":"asserted","qualifiers":{"adverbs":[],"conditions":[],"scope":null},"valid_from":null,"valid_to":null,"quote":"Ada manages Company."}]});
    (
        StatusCode::OK,
        Json(json!({"choices":[{"finish_reason":"stop","message":{"content":facts.to_string()}}]})),
    )
}

#[tokio::test]
async fn direct_timer_temporal_retry_profile_guard_and_replay_agree() -> Result<()> {
    let model_state = Arc::new(Model::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let router = Router::new()
        .route("/chat", post(model))
        .with_state(model_state.clone());
    let server = tokio::spawn(async move { axum::serve(listener, router).await });
    let mut extractor = ChatExtractor::new(
        &ChatConfig {
            endpoint: format!("http://{address}/chat"),
            model: "fixture".into(),
            revision: "fixture-v1".into(),
            api_key_env: None,
            api: Default::default(),
            gateway: None,
            references: Default::default(),
        },
        Config::parse(include_str!("fixtures/business.toml"))?.ontology,
    )?;
    let text = "é context. Ada manages Company.";
    let request = ExtractionRequest {
        profile: extractor.profile.clone(),
        chunk: Chunk {
            document: Document {
                key: DocumentKey {
                    scope: "scope".into(),
                    source: "document".into(),
                    revision: "1".into(),
                },
                text: text.into(),
                context: Some("conversation".into()),
                metadata: Default::default(),
            },
            range: ByteRange {
                start: "é context. ".len(),
                end: text.len(),
            },
        },
    };
    let mut timer = tokio::time::interval(Duration::from_millis(10));
    timer.tick().await;
    let direct = extract_document(&mut extractor, request.clone()).await?;
    timer.tick().await;
    ensure!(
        serde_json::to_value(&direct)?
            == serde_json::to_value(extract_document(&mut extractor, request.clone()).await?)?
    );
    ensure!(model_state.calls.load(Ordering::SeqCst) == 2);
    model_state.fail_next.store(true, Ordering::SeqCst);

    let cli = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|p| p.join("temporal"))
        .find(|p| p.is_file())
        .expect("Temporal CLI required for integration tests");
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
    let mut worker = Worker::new(
        &runtime,
        client.clone(),
        WorkerOptions::new("knowledge-tests")
            .register_workflow::<ExtractionWorkflow>()?
            .register_activities(ExtractionActivities(extractor))
            .build(),
    )?;
    let shutdown = worker.shutdown_handle();
    let exercise = async {
        let result = async {
            let handle = client
                .start_workflow(
                    ExtractionWorkflow::run,
                    request.clone(),
                    WorkflowStartOptions::new("knowledge-tests", "candidate-graph").build(),
                )
                .await?;
            let durable = tokio::time::timeout(
                Duration::from_secs(30),
                handle.get_result(WorkflowGetResultOptions::default()),
            )
            .await??;
            ensure!(serde_json::to_value(&direct)? == serde_json::to_value(&durable)?);
            ensure!(
                model_state.calls.load(Ordering::SeqCst) == 4,
                "one direct call per tick plus failed/successful Temporal attempts"
            );
            let history = handle
                .fetch_history(WorkflowFetchHistoryOptions::default())
                .into_events()
                .await?;
            let scheduled: Vec<_> = history
                .iter()
                .filter_map(|event| match &event.attributes {
                    Some(Attributes::ActivityTaskScheduledEventAttributes(activity)) => {
                        Some(activity)
                    }
                    _ => None,
                })
                .collect();
            ensure!(
                scheduled.len() == 1
                    && scheduled[0].activity_type.as_ref().unwrap().name
                        == "ExtractionActivities::extract"
            );
            let replayer = WorkflowReplayer::new(
                WorkflowReplayerOptions::new()
                    .register_workflow::<ExtractionWorkflow>()?
                    .build(),
            )?;
            replayer
                .replay_workflow(temporalio_client::WorkflowHistory::from(
                    temporalio_common::protos::temporal::api::history::v1::History {
                        events: history,
                    },
                ))
                .await?;
            ensure!(
                model_state.calls.load(Ordering::SeqCst) == 4,
                "replay must not call model"
            );
            let mut stale = request;
            stale.profile = "unavailable-profile".into();
            let handle = client
                .start_workflow(
                    ExtractionWorkflow::run,
                    stale,
                    WorkflowStartOptions::new("knowledge-tests", "stale-profile").build(),
                )
                .await?;
            ensure!(
                tokio::time::timeout(
                    Duration::from_secs(10),
                    handle.get_result(WorkflowGetResultOptions::default())
                )
                .await?
                .is_err()
            );
            ensure!(
                model_state.calls.load(Ordering::SeqCst) == 4,
                "stale profile must fail without model I/O"
            );
            Ok::<_, anyhow::Error>(())
        }
        .await;
        shutdown();
        result
    };
    let (worker_result, result) = tokio::join!(worker.run(), exercise);
    env.shutdown().await?;
    server.abort();
    worker_result?;
    result
}

mod lane_pipeline {
    use super::*;
    use aurelius_lanes::{
        lanes::{
            EpochRef, LaneActivities, LaneReceipt, LaneRequest, ObjectRef, PublicationRef,
            index_lane,
        },
        temporal_lanes::LaneWorkflow,
    };
    use std::sync::Mutex;
    use temporalio_macros::activities;
    use temporalio_sdk::activities::{ActivityContext, ActivityError};

    #[derive(Default)]
    struct State {
        events: Mutex<Vec<&'static str>>,
        fail_index: AtomicBool,
        always_fail: AtomicBool,
    }
    #[derive(Clone)]
    struct Functions(Arc<State>);
    fn request() -> LaneRequest {
        LaneRequest {
            lane: "fixture".into(),
            profile: "a".repeat(64),
        }
    }
    fn object(key: &str) -> ObjectRef {
        ObjectRef {
            key: key.into(),
            version: "fixture-etag".into(),
        }
    }
    impl LaneActivities for Functions {
        type Error = anyhow::Error;
        async fn discover(&mut self, _: LaneRequest) -> Result<Option<EpochRef>> {
            self.0.events.lock().unwrap().push("discover");
            Ok(Some(EpochRef {
                request: request(),
                sequence: 17,
                manifest: object("scope/epochs/17.json"),
            }))
        }
        async fn publish(&mut self, epoch: EpochRef) -> Result<PublicationRef> {
            self.0.events.lock().unwrap().push("publish");
            Ok(PublicationRef {
                epoch,
                manifest: object("scope/published/17.json"),
            })
        }
        async fn index(&mut self, publication: PublicationRef) -> Result<LaneReceipt> {
            self.0.events.lock().unwrap().push("index");
            ensure!(
                !self.0.always_fail.load(Ordering::SeqCst)
                    && !self.0.fail_index.swap(false, Ordering::SeqCst),
                "injected transient index failure"
            );
            Ok(LaneReceipt {
                publication,
                receipt: object("scope/receipts/17.json"),
                indexed: 2,
                deferred: 1,
                ignored: 0,
            })
        }
        async fn acknowledge(&mut self, _: LaneReceipt) -> Result<()> {
            self.0.events.lock().unwrap().push("acknowledge");
            Ok(())
        }
    }
    struct Bound(Functions);
    #[activities]
    impl Bound {
        #[activity(name = "LaneWorker::discover")]
        async fn discover(
            self: Arc<Self>,
            _ctx: ActivityContext,
            r: LaneRequest,
        ) -> Result<Option<EpochRef>, ActivityError> {
            self.0.clone().discover(r).await.map_err(Into::into)
        }
        #[activity(name = "LaneWorker::publish")]
        async fn publish(
            self: Arc<Self>,
            _ctx: ActivityContext,
            r: EpochRef,
        ) -> Result<PublicationRef, ActivityError> {
            self.0.clone().publish(r).await.map_err(Into::into)
        }
        #[activity(name = "LaneWorker::index")]
        async fn index(
            self: Arc<Self>,
            _ctx: ActivityContext,
            r: PublicationRef,
        ) -> Result<LaneReceipt, ActivityError> {
            self.0.clone().index(r).await.map_err(Into::into)
        }
        #[activity(name = "LaneWorker::acknowledge")]
        async fn acknowledge(
            self: Arc<Self>,
            _ctx: ActivityContext,
            r: LaneReceipt,
        ) -> Result<(), ActivityError> {
            self.0.clone().acknowledge(r).await.map_err(Into::into)
        }
    }
    #[tokio::test]
    async fn lane_function_native_retry_reference_history_and_replay_agree() -> Result<()> {
        use aurelius_lanes::temporal_lanes::LaneWorker;
        ensure!(
            [
                Bound::discover.name(),
                Bound::publish.name(),
                Bound::index.name(),
                Bound::acknowledge.name()
            ] == [
                LaneWorker::discover.name(),
                LaneWorker::publish.name(),
                LaneWorker::index.name(),
                LaneWorker::acknowledge.name()
            ]
        );
        let direct_state = Arc::new(State::default());
        let direct = index_lane(&mut Functions(direct_state.clone()), request())
            .await
            .map_err(|_| anyhow::anyhow!("direct pipeline failed"))?;
        ensure!(
            *direct_state.events.lock().unwrap() == ["discover", "publish", "index", "acknowledge"]
        );
        let state = Arc::new(State::default());
        state.fail_index.store(true, Ordering::SeqCst);
        let cli = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .map(|p| p.join("temporal"))
            .find(|p| p.is_file())
            .expect("Temporal CLI required");
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
        let mut worker = Worker::new(
            &runtime,
            client.clone(),
            WorkerOptions::new("lane-pipeline-tests")
                .register_workflow::<LaneWorkflow>()?
                .register_activities(Bound(Functions(state.clone())))
                .build(),
        )?;
        let shutdown = worker.shutdown_handle();
        let exercise = async {
            let result = async {
                let handle = client
                    .start_workflow(
                        LaneWorkflow::run,
                        request(),
                        WorkflowStartOptions::new("lane-pipeline-tests", "lane-reference-pipeline")
                            .build(),
                    )
                    .await?;
                let durable = tokio::time::timeout(
                    Duration::from_secs(45),
                    handle.get_result(WorkflowGetResultOptions::default()),
                )
                .await??;
                ensure!(serde_json::to_value(direct)? == serde_json::to_value(durable)?);
                ensure!(
                    *state.events.lock().unwrap()
                        == ["discover", "publish", "index", "index", "acknowledge"],
                    "retry must not repeat publication or acknowledge before index succeeds"
                );
                let history = handle
                    .fetch_history(WorkflowFetchHistoryOptions::default())
                    .into_events()
                    .await?;
                let scheduled: Vec<_> = history
                    .iter()
                    .filter_map(|e| match &e.attributes {
                        Some(Attributes::ActivityTaskScheduledEventAttributes(a)) => Some(a),
                        _ => None,
                    })
                    .collect();
                ensure!(
                    scheduled
                        .iter()
                        .map(|a| a.activity_type.as_ref().unwrap().name.as_str())
                        .collect::<Vec<_>>()
                        == [
                            "LaneWorker::discover",
                            "LaneWorker::publish",
                            "LaneWorker::index",
                            "LaneWorker::acknowledge"
                        ]
                );
                for activity in scheduled {
                    for payload in &activity.input.as_ref().unwrap().payloads {
                        let value: Value = serde_json::from_slice(&payload.data)?;
                        let encoded = value.to_string();
                        ensure!(
                            !["document", "text", "content", "credential", "secret"]
                                .iter()
                                .any(|field| encoded.contains(&format!("\"{field}\":"))),
                            "lane history contains source bodies or credentials"
                        );
                        ensure!(
                            payload.data.len() < 4096,
                            "lane history input exceeds compact reference bound"
                        );
                    }
                }
                let before = state.events.lock().unwrap().len();
                WorkflowReplayer::new(
                    WorkflowReplayerOptions::new()
                        .register_workflow::<LaneWorkflow>()?
                        .build(),
                )?
                .replay_workflow(temporalio_client::WorkflowHistory::from(
                    temporalio_common::protos::temporal::api::history::v1::History {
                        events: history,
                    },
                ))
                .await?;
                ensure!(
                    state.events.lock().unwrap().len() == before,
                    "history replay performed I/O"
                );
                let mut stale = request();
                stale.profile = "b".repeat(64);
                let handle = client
                    .start_workflow(
                        LaneWorkflow::run,
                        stale,
                        WorkflowStartOptions::new("lane-pipeline-tests", "foreign-lane-profile")
                            .build(),
                    )
                    .await?;
                ensure!(
                    tokio::time::timeout(
                        Duration::from_secs(10),
                        handle.get_result(WorkflowGetResultOptions::default())
                    )
                    .await?
                    .is_err()
                );
                ensure!(
                    state.events.lock().unwrap().len() == before + 1,
                    "foreign profile reached publication"
                );
                Ok::<_, anyhow::Error>(())
            }
            .await;
            shutdown();
            result
        };
        let (worker_result, result) = tokio::join!(worker.run(), exercise);
        env.shutdown().await?;
        worker_result?;
        result
    }

    #[tokio::test]
    async fn observer_retains_retry_loop_after_exhausted_activity_failures() -> Result<()> {
        use aurelius_lanes::observer::LaneObserver;
        let state = Arc::new(State::default());
        state.always_fail.store(true, Ordering::SeqCst);
        let cli = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .map(|p| p.join("temporal"))
            .find(|p| p.is_file())
            .expect("Temporal CLI required");
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
        let mut worker = Worker::new(
            &runtime,
            client.clone(),
            WorkerOptions::new("lane-observer-tests")
                .register_workflow::<LaneObserver>()?
                .register_activities(Bound(Functions(state.clone())))
                .build(),
        )?;
        let shutdown = worker.shutdown_handle();
        let exercise = async {
            let result = async {
                let handle = client
                    .start_workflow(
                        LaneObserver::run,
                        request(),
                        WorkflowStartOptions::new("lane-observer-tests", "resumable-observer")
                            .build(),
                    )
                    .await?;
                let history = tokio::time::timeout(Duration::from_secs(50), async {
                    loop {
                        let history = handle
                            .fetch_history(WorkflowFetchHistoryOptions::default())
                            .into_events()
                            .await?;
                        if history.iter().any(|e| {
                            matches!(
                                &e.attributes,
                                Some(Attributes::TimerStartedEventAttributes(_))
                            )
                        }) {
                            break Ok::<_, anyhow::Error>(history);
                        }
                        tokio::time::sleep(Duration::from_millis(250)).await;
                    }
                })
                .await??;
                ensure!(
                    *state.events.lock().unwrap()
                        == ["discover", "publish", "index", "index", "index"]
                );
                ensure!(!history.iter().any(|e| matches!(
                    &e.attributes,
                    Some(Attributes::WorkflowExecutionFailedEventAttributes(_))
                )));
                let index = history
                    .iter()
                    .find_map(|e| match &e.attributes {
                        Some(Attributes::ActivityTaskScheduledEventAttributes(a))
                            if a.activity_type.as_ref().unwrap().name == "LaneWorker::index" =>
                        {
                            Some(a)
                        }
                        _ => None,
                    })
                    .unwrap();
                ensure!(index.start_to_close_timeout.as_ref().unwrap().seconds == 86400);
                let timer = history
                    .iter()
                    .find_map(|e| match &e.attributes {
                        Some(Attributes::TimerStartedEventAttributes(a)) => Some(a),
                        _ => None,
                    })
                    .unwrap();
                ensure!(timer.start_to_fire_timeout.as_ref().unwrap().seconds == 300);
                WorkflowReplayer::new(
                    WorkflowReplayerOptions::new()
                        .register_workflow::<LaneObserver>()?
                        .build(),
                )?
                .replay_workflow(temporalio_client::WorkflowHistory::from(
                    temporalio_common::protos::temporal::api::history::v1::History {
                        events: history,
                    },
                ))
                .await?;
                ensure!(
                    state.events.lock().unwrap().len() == 5,
                    "observer replay performed I/O"
                );
                Ok::<_, anyhow::Error>(())
            }
            .await;
            shutdown();
            result
        };
        let (worker_result, result) = tokio::join!(worker.run(), exercise);
        env.shutdown().await?;
        worker_result?;
        result
    }
}
