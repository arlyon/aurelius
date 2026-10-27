use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    pub endpoint: String,
    pub model: String,
    pub revision: String,
    pub native_dimensions: usize,
    pub dimensions: usize,
    pub token_limit: usize,
}
impl EmbeddingConfig {
    pub fn validate(&self) -> Result<()> {
        let url = reqwest::Url::parse(&self.endpoint)?;
        ensure!(
            url.scheme() == "https"
                || (url.scheme() == "http"
                    && url
                        .host_str()
                        .is_some_and(|h| matches!(h, "localhost" | "127.0.0.1" | "[::1]")
                            || h.ends_with(".svc.cluster.local"))),
            "invalid embedding endpoint"
        );
        ensure!(
            url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none(),
            "invalid embedding URL"
        );
        ensure!(
            !self.model.trim().is_empty()
                && !self.revision.trim().is_empty()
                && self.model.len() <= 200
                && self.revision.len() <= 200,
            "model revision required"
        );
        ensure!(
            self.dimensions > 0
                && self.dimensions <= self.native_dimensions
                && self.native_dimensions <= 8192
                && (32..=4096).contains(&self.token_limit),
            "invalid embedding shape/budget"
        );
        Ok(())
    }
}
#[derive(Clone)]
pub struct Embedder {
    client: reqwest::Client,
    pub config: EmbeddingConfig,
}
impl Embedder {
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(180))
                .build()?,
        })
    }
    pub async fn count(&self, text: &str) -> Result<usize> {
        ensure!(
            text.len() <= 1024 * 1024,
            "tokenization input exceeds bound"
        );
        let value: serde_json::Value = self
            .client
            .post(format!(
                "{}/tokenize",
                self.config.endpoint.trim_end_matches('/')
            ))
            .json(&serde_json::json!({"content":text,"add_special":true,"parse_special":false}))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(value
            .get("tokens")
            .and_then(serde_json::Value::as_array)
            .context("invalid tokenization response")?
            .len())
    }
    pub async fn vector(&self, text: &str) -> Result<Vec<f32>> {
        ensure!(
            text.len() <= 16000 && self.count(text).await? <= self.config.token_limit,
            "embedding input exceeds token budget"
        );
        #[derive(Deserialize)]
        struct Item {
            index: usize,
            embedding: Vec<f32>,
        }
        #[derive(Deserialize)]
        struct Response {
            data: Vec<Item>,
        }
        let mut value: Response = self
            .client
            .post(format!(
                "{}/v1/embeddings",
                self.config.endpoint.trim_end_matches('/')
            ))
            .json(&serde_json::json!({"model":self.config.model,"input":[text]}))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        ensure!(
            value.data.len() == 1 && value.data[0].index == 0,
            "embedding response count mismatch"
        );
        let vector = value.data.remove(0).embedding;
        project(
            vector,
            self.config.native_dimensions,
            self.config.dimensions,
        )
    }
}
pub fn project(mut vector: Vec<f32>, native: usize, dimensions: usize) -> Result<Vec<f32>> {
    ensure!(
        native > 0
            && dimensions > 0
            && dimensions <= native
            && vector.len() == native
            && vector.iter().all(|n| n.is_finite()),
        "invalid native vector"
    );
    vector.truncate(dimensions);
    let norm = vector
        .iter()
        .map(|n| f64::from(*n).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(norm.is_finite() && norm > 0., "zero/nonfinite embedding");
    for n in &mut vector {
        *n = (f64::from(*n) / norm) as f32;
    }
    Ok(vector)
}
