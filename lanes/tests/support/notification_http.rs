use anyhow::{Context, Result, ensure};
use aurelius_lanes::{
    config::{Config, TemporalConfig},
    ingest::IngestRequest,
    notification::{NotificationState, router},
    postgres::PostgresSink,
    temporal_ready,
};
use std::time::Duration;
use temporalio_client::{RpcOptions, UntypedWorkflow, WorkflowDescribeOptions};
use temporalio_sdk::testing::{EphemeralExe, LocalWorkflowEnvironmentOptions, WorkflowEnvironment};
pub async fn qualify(mut config: Config, sink: PostgresSink) -> Result<()> {
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
    config.temporal = Some(TemporalConfig {
        address: "http://127.0.0.1:7233".into(),
        namespace: "default".into(),
        task_queue: "notification-http-fixture".into(),
        api_key_env: None,
        client_cert_env: None,
        client_key_env: None,
    });
    let state = NotificationState::new(config.clone(), client.clone(), sink)?;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(async move { axum::serve(listener, router(state)).await });
    let result=async {
        let http=reqwest::Client::builder().timeout(Duration::from_secs(15)).build()?;
        let url=format!("http://{address}/hooks/airbyte");
        let packet=|connection:&str,job:u64,success:bool|serde_json::json!({"data":{"connection":{"id":connection},"jobId":job,"success":success,"workspace":{"id":"not-a-tenant-authority"}},"text":"ignored notification display"});
        ensure!(http.get(format!("http://{address}/health")).send().await?.status()==204);
        ensure!(http.post(&url).json(&packet("unknown",4,true)).send().await?.status()==404);
        ensure!(http.post(&url).json(&packet("fixture-connection",4,false)).send().await?.status()==204);
        ensure!(http.post(&url).json(&packet("fixture-connection",0,true)).send().await?.status()==400);
        ensure!(http.post(&url).json(&serde_json::json!({"padding":"x".repeat(17*1024)})).send().await?.status()==413);
        let request=IngestRequest{lane:config.lane_request("fixture")?,after:None};
        let id=temporal_ready::workflow_id("tenant-a",&request);
        ensure!(http.post(&url).json(&packet("fixture-connection",1,true)).send().await?.status()==202);
        ensure!(temporal_ready::status(&client,&id).await.is_err(),"already committed notification started a workflow");
        for job in [2,3,5] {ensure!(http.post(&url).json(&packet("fixture-connection",job,true)).send().await?.status()==503,"non-successful or cross-connection job accepted");}
        ensure!(http.post(&url).json(&packet("fixture-connection",4,true)).send().await?.status()==202);
        // No worker is running: a successful ACK must represent a durably accepted
        // Temporal start, not an in-memory task or successful source verification alone.
        let handle=client.get_workflow_handle::<UntypedWorkflow>(&id);
        let first=handle.describe(WorkflowDescribeOptions::default()).await?;
        ensure!(first.status()==temporalio_client::WorkflowExecutionStatus::Running);
        ensure!(http.post(&url).json(&packet("fixture-connection",4,true)).send().await?.status()==202);
        let second=handle.describe(WorkflowDescribeOptions::default()).await?;
        ensure!(first.raw_description.workflow_execution_info.as_ref().context("workflow info")?.execution==second.raw_description.workflow_execution_info.as_ref().context("workflow info")?.execution,"duplicate notification started a second writer");
        let schedule=temporal_ready::create_schedule(&client,&config).await?;
        let description=client.get_schedule_handle(schedule).describe(RpcOptions::default()).await?;
        let spec=description.raw().schedule.as_ref().context("schedule")?;
        ensure!(spec.spec.as_ref().context("schedule spec")?.interval.iter().any(|interval|interval.interval.as_ref().is_some_and(|duration|duration.seconds==21600)),"schedule is not six-hourly");
        ensure!(spec.state.as_ref().context("schedule state")?.paused,"schedule started before handoff");
        ensure!(spec.policies.as_ref().context("schedule policies")?.overlap_policy==1,"schedule does not skip overlaps");
        Ok::<_,anyhow::Error>(())
    }.await;
    server.abort();
    env.shutdown().await?;
    result
}
