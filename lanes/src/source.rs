use crate::config::credential;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceConfig {
    pub endpoint: String,
    pub client_id_env: String,
    pub client_secret_env: String,
}
impl SourceConfig {
    pub fn validate(&self) -> Result<()> {
        let url = reqwest::Url::parse(&self.endpoint)?;
        ensure!(
            url.scheme() == "https"
                || (url.scheme() == "http"
                    && url.host_str().is_some_and(|h| h == "127.0.0.1"
                        || h == "localhost"
                        || h.ends_with(".svc.cluster.local"))),
            "invalid source API endpoint"
        );
        ensure!(
            url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none(),
            "invalid source API URL"
        );
        crate::config::validate_env_name(&self.client_id_env)?;
        crate::config::validate_env_name(&self.client_secret_env)?;
        Ok(())
    }
}
#[derive(Clone)]
pub struct SourceClient {
    client: reqwest::Client,
    config: SourceConfig,
    token: Arc<Mutex<Option<(String, Instant)>>>,
}
#[derive(Deserialize)]
struct Token {
    access_token: String,
    expires_in: u64,
}
#[derive(Deserialize)]
struct JobList {
    jobs: Vec<JobEnvelope>,
}
#[derive(Deserialize)]
struct JobEnvelope {
    job: Job,
}
#[derive(Deserialize)]
struct Job {
    id: u64,
    #[serde(rename = "configId")]
    connection: String,
    #[serde(rename = "configType")]
    kind: String,
    status: String,
}
impl SourceClient {
    pub fn new(config: SourceConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(30))
                .build()?,
            config,
            token: Arc::new(Mutex::new(None)),
        })
    }
    async fn token(&self) -> Result<String> {
        let mut token = self.token.lock().await;
        if let Some((value, expiry)) = &*token
            && *expiry > Instant::now()
        {
            return Ok(value.clone());
        }
        let response = self.client.post(format!("{}/api/public/v1/applications/token", self.config.endpoint.trim_end_matches('/')))
            .json(&json!({"client_id":credential(&self.config.client_id_env)?,
                "client_secret":credential(&self.config.client_secret_env)?,"grant-type":"client_credentials"}))
            .send().await?;
        ensure!(
            response.status().is_success(),
            "source authentication failed"
        );
        let value: Token = response
            .json()
            .await
            .map_err(|_| anyhow::anyhow!("invalid source token response"))?;
        ensure!(
            !value.access_token.is_empty() && (61..=86400).contains(&value.expires_in),
            "invalid source token"
        );
        let result = value.access_token.clone();
        *token = Some((
            value.access_token,
            Instant::now() + Duration::from_secs(value.expires_in - 60),
        ));
        Ok(result)
    }
    async fn api<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        body: serde_json::Value,
    ) -> Result<T> {
        let mut response = self
            .client
            .post(format!(
                "{}/api/v1/{method}",
                self.config.endpoint.trim_end_matches('/')
            ))
            .bearer_auth(self.token().await?)
            .json(&body)
            .send()
            .await?;
        ensure!(
            response.status().is_success(),
            "source metadata request failed"
        );
        let mut bytes = Vec::new();
        while let Some(part) = response.chunk().await? {
            ensure!(
                bytes.len() + part.len() <= 8 * 1024 * 1024,
                "source metadata response exceeds bound"
            );
            bytes.extend_from_slice(&part);
        }
        serde_json::from_slice(&bytes).map_err(|_| anyhow::anyhow!("invalid source metadata"))
    }
    pub async fn successful_epochs(&self, connection: &str) -> Result<Vec<u64>> {
        let mut ids = Vec::new();
        for page in 0..100 {
            let jobs: JobList = self
                .api(
                    "jobs/list",
                    json!({"configTypes":["sync"],"configId":connection,
                "pagination":{"pageSize":50,"rowOffset":page*50}}),
                )
                .await?;
            let count = jobs.jobs.len();
            for entry in jobs.jobs {
                ensure!(
                    entry.job.connection == connection && entry.job.kind == "sync",
                    "source job scope mismatch"
                );
                if entry.job.status == "succeeded" {
                    ids.push(entry.job.id);
                }
            }
            if count < 50 {
                ids.sort();
                ids.dedup();
                return Ok(ids);
            }
        }
        anyhow::bail!("source job history exceeds scan bound")
    }
    pub async fn verify(&self, connection: &str, epoch: u64) -> Result<()> {
        let job: JobEnvelope = self.api("jobs/get", json!({"id":epoch})).await?;
        ensure!(
            job.job.connection == connection
                && job.job.id == epoch
                && job.job.kind == "sync"
                && job.job.status == "succeeded",
            "source epoch is not successful or changed scope"
        );
        Ok(())
    }
    pub async fn validate_connection(&self, connection: &str) -> Result<()> {
        let value: serde_json::Value = self
            .api("connections/get", json!({"connectionId":connection}))
            .await?;
        let streams = value
            .pointer("/syncCatalog/streams")
            .and_then(serde_json::Value::as_array)
            .context("source catalog unavailable")?;
        for stream in streams {
            if stream
                .pointer("/config/selected")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            {
                ensure!(
                    stream
                        .pointer("/config/destinationSyncMode")
                        .and_then(serde_json::Value::as_str)
                        == Some("append"),
                    "source delivery must remain append-only"
                );
            }
        }
        Ok(())
    }
}
