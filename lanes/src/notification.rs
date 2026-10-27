use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
// Airbyte provides a URL-only callback. This listener must stay behind the
// cluster-internal Service and sender-restricted NetworkPolicy, never public ingress.
pub struct WebhookConfig {
    pub bind: std::net::SocketAddr,
}
impl WebhookConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.bind.port() >= 1024
                && (self.bind.ip().is_loopback()
                    || self.bind.ip().is_unspecified()
                    || matches!(self.bind.ip(),std::net::IpAddr::V4(ip) if ip.is_private())),
            "webhook listener requires unprivileged private bind address"
        );
        Ok(())
    }
}
#[derive(Debug, Deserialize)]
pub struct AirbyteNotification {
    pub data: JobReady,
}
#[derive(Debug, Deserialize)]
pub struct JobReady {
    pub connection: Connection,
    #[serde(rename = "jobId")]
    pub job_id: u64,
    pub success: bool,
}
#[derive(Debug, Deserialize)]
pub struct Connection {
    pub id: String,
}

#[cfg(all(feature = "temporal", feature = "postgres"))]
mod runtime {
    use super::*;
    use crate::{
        config::Config, ingest::IngestRequest, postgres::PostgresSink, source::SourceClient,
    };
    use anyhow::Context;
    use axum::{
        Json, Router,
        extract::{DefaultBodyLimit, State},
        http::StatusCode,
        routing::{get, post},
    };
    use std::sync::Arc;
    use temporalio_client::Client;
    #[derive(Clone)]
    pub struct NotificationState {
        config: Config,
        client: Client,
        source: SourceClient,
        sink: PostgresSink,
        capacity: Arc<tokio::sync::Semaphore>,
    }
    impl NotificationState {
        pub fn new(config: Config, client: Client, sink: PostgresSink) -> Result<Self> {
            let source = SourceClient::new(
                config
                    .source
                    .clone()
                    .context("source API configuration required")?,
            )?;
            Ok(Self {
                config,
                client,
                source,
                sink,
                capacity: Arc::new(tokio::sync::Semaphore::new(4)),
            })
        }
    }
    pub fn router(state: NotificationState) -> Router {
        Router::new()
            .route("/health", get(|| async { StatusCode::NO_CONTENT }))
            .route("/hooks/airbyte", post(ready))
            .layer(DefaultBodyLimit::max(16 * 1024))
            .with_state(state)
    }
    async fn ready(
        State(state): State<NotificationState>,
        Json(notification): Json<AirbyteNotification>,
    ) -> StatusCode {
        let Ok(_permit) = state.capacity.try_acquire() else {
            return StatusCode::TOO_MANY_REQUESTS;
        };
        let event = notification.data;
        if !event.success {
            return StatusCode::NO_CONTENT;
        }
        if event.job_id == 0 || event.job_id > i64::MAX as u64 {
            return StatusCode::BAD_REQUEST;
        }
        let Some((name, lane)) = state
            .config
            .lanes
            .iter()
            .find(|(_, lane)| lane.connection == event.connection.id)
        else {
            return StatusCode::NOT_FOUND;
        };
        let operation = async {
            let request = IngestRequest {
                lane: state.config.lane_request(name)?,
                after: None,
            };
            if state
                .sink
                .ingestion_cursor(&lane.scope, &request.lane)
                .await?
                .is_some_and(|after| event.job_id <= after)
            {
                return Ok(());
            }
            state.source.verify(&lane.connection, event.job_id).await?;
            crate::temporal_ready::wake(&state.client, &state.config, request).await?;
            Ok::<_, anyhow::Error>(())
        }
        .await;
        match operation {
            Ok(()) => StatusCode::ACCEPTED,
            Err(_) => {
                tracing::warn!(lane=%name,job=event.job_id,"lane notification not accepted; source, database or Temporal verification failed");
                StatusCode::SERVICE_UNAVAILABLE
            }
        }
    }
    pub async fn serve(listener: tokio::net::TcpListener, state: NotificationState) -> Result<()> {
        axum::serve(listener, router(state))
            .await
            .context("worker webhook stopped")
    }
}
#[cfg(all(feature = "temporal", feature = "postgres"))]
pub use runtime::{NotificationState, router, serve};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn airbyte_packet_ignores_display_content_and_rejects_invalid_ids() -> Result<()> {
        let packet: AirbyteNotification = serde_json::from_value(
            serde_json::json!({"data":{"connection":{"id":"configured-connection","name":"display only"},"jobId":42,"success":true,"workspace":{"id":"untrusted-not-an-org"}},"text":"ignored","blocks":[]}),
        )?;
        assert_eq!(packet.data.connection.id, "configured-connection");
        assert_eq!(packet.data.job_id, 42);
        assert!(packet.data.success);
        assert!(serde_json::from_value::<AirbyteNotification>(serde_json::json!({"data":{"connection":{"id":"connection"},"jobId":-1,"success":true}})).is_err());
        assert!(serde_json::from_value::<AirbyteNotification>(serde_json::json!({"data":{"connection":{"id":"connection"},"jobId":"42","success":true}})).is_err());
        Ok(())
    }
    #[test]
    fn listener_is_private_and_unprivileged() -> Result<()> {
        for bind in ["127.0.0.1:8080", "0.0.0.0:8080", "10.0.0.1:8080"] {
            WebhookConfig {
                bind: bind.parse()?,
            }
            .validate()?;
        }
        for bind in ["8.8.8.8:8080", "127.0.0.1:80"] {
            assert!(
                WebhookConfig {
                    bind: bind.parse()?
                }
                .validate()
                .is_err()
            );
        }
        Ok(())
    }
}
