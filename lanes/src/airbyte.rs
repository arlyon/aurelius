use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct Airbyte {
    client: reqwest::Client,
    base: String,
    client_id: String,
    client_secret: String,
    token: Arc<Mutex<Option<(String, Instant)>>>,
}
#[derive(Deserialize)]
pub struct JobInfo {
    pub id: i64,
    #[serde(rename = "configId")]
    pub connection_id: String,
    #[serde(rename = "configType")]
    pub config_type: String,
    pub status: String,
}
impl JobInfo {
    pub fn terminal(&self) -> bool {
        matches!(self.status.as_str(), "succeeded" | "failed" | "cancelled")
    }
}
#[derive(Deserialize)]
struct JobRead {
    job: JobInfo,
}
#[derive(Deserialize)]
struct JobList {
    jobs: Vec<JobRead>,
}
#[derive(Deserialize)]
struct Token {
    access_token: String,
    expires_in: u64,
}

impl Airbyte {
    pub fn from_env() -> Result<Self> {
        let base = std::env::var("AIRBYTE_URL")?;
        let parsed = reqwest::Url::parse(&base)?;
        ensure!(
            parsed.scheme() == "https"
                || parsed.scheme() == "http"
                    && parsed.host_str().is_some_and(
                        |host| host.ends_with(".svc.cluster.local") || host == "127.0.0.1"
                    ),
            "Airbyte requires TLS or an explicit internal endpoint"
        );
        ensure!(
            parsed.username().is_empty() && parsed.password().is_none(),
            "credentials must not be in URL"
        );
        Ok(Self {
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(30))
                .build()?,
            base: base.trim_end_matches('/').into(),
            client_id: std::env::var("AIRBYTE_CLIENT_ID")?,
            client_secret: std::env::var("AIRBYTE_CLIENT_SECRET")?,
            token: Arc::new(Mutex::new(None)),
        })
    }
    async fn token(&self) -> Result<String> {
        let mut guard = self.token.lock().await;
        if let Some((token, expiry)) = &*guard
            && *expiry > Instant::now()
        {
            return Ok(token.clone());
        }
        let token:Token=self.client.post(format!("{}/api/public/v1/applications/token",self.base)).json(&json!({"client_id":self.client_id,"client_secret":self.client_secret,"grant-type":"client_credentials"})).send().await?.error_for_status()?.json().await?;
        ensure!(token.expires_in > 60, "invalid token expiry");
        let value = token.access_token.clone();
        *guard = Some((
            token.access_token,
            Instant::now() + Duration::from_secs(token.expires_in - 60),
        ));
        Ok(value)
    }
    async fn api<T: serde::de::DeserializeOwned>(&self, path: &str, body: Value) -> Result<T> {
        Ok(self
            .client
            .post(format!("{}/api/v1/{path}", self.base))
            .bearer_auth(self.token().await?)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }
    pub async fn verify(&self, connection: &str, id: i64) -> Result<()> {
        let job: JobRead = self.api("jobs/get", json!({"id":id})).await?;
        ensure!(
            job.job.id == id
                && job.job.connection_id == connection
                && job.job.config_type == "sync",
            "job does not belong to registered connection"
        );
        ensure!(job.job.terminal(), "sync is not terminal");
        Ok(())
    }
    pub async fn jobs(&self, connection: &str) -> Result<Vec<i64>> {
        let mut result = Vec::new();
        for page in 0..100 {
            let jobs:JobList=self.api("jobs/list",json!({"configTypes":["sync"],"configId":connection,"pagination":{"pageSize":50,"rowOffset":page*50}})).await?;
            let count = jobs.jobs.len();
            for job in jobs.jobs {
                ensure!(
                    job.job.connection_id == connection && job.job.config_type == "sync",
                    "unexpected job identity"
                );
                if job.job.terminal() {
                    result.push(job.job.id);
                }
            }
            if count < 50 {
                return Ok(result);
            }
        }
        Err(anyhow::anyhow!(
            "job history exceeds scan bound; explicit replay required"
        ))
    }
    pub async fn validate_connection(&self, id: &str) -> Result<()> {
        let value: Value = self
            .api("connections/get", json!({"connectionId":id}))
            .await?;
        let streams = value
            .get("syncCatalog")
            .and_then(|x| x.get("streams"))
            .and_then(Value::as_array)
            .context("missing connection catalog")?;
        for entry in streams {
            let config = entry
                .get("config")
                .context("missing stream configuration")?;
            if config.get("selected").and_then(Value::as_bool) == Some(true) {
                ensure!(
                    config.get("destinationSyncMode").and_then(Value::as_str) == Some("append"),
                    "archive indexing requires append-only source output"
                );
            }
        }
        Ok(())
    }
}
