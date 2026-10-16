use crate::config::credential;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct XbergConfig {
    pub endpoint: String,
    pub api_key_env: Option<String>,
}
impl XbergConfig {
    pub fn validate(&self) -> Result<reqwest::Url> {
        let endpoint = reqwest::Url::parse(&self.endpoint)?;
        let host = endpoint.host_str().context("xberg host required")?;
        ensure!(
            endpoint.scheme() == "https"
                || (endpoint.scheme() == "http"
                    && (matches!(host, "localhost" | "127.0.0.1" | "[::1]")
                        || host.ends_with(".svc.cluster.local"))),
            "xberg requires TLS, loopback or an explicit cluster-local service"
        );
        ensure!(
            endpoint.username().is_empty()
                && endpoint.password().is_none()
                && endpoint.query().is_none()
                && endpoint.fragment().is_none(),
            "invalid xberg endpoint"
        );
        if let Some(name) = &self.api_key_env {
            crate::config::validate_env_name(name)?;
        }
        Ok(endpoint)
    }
}

#[derive(Clone)]
pub struct XbergClient {
    client: reqwest::Client,
    endpoint: reqwest::Url,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExtractedFile {
    pub content: String,
    pub mime_type: String,
}
impl XbergClient {
    pub fn new(config: &XbergConfig) -> Result<Self> {
        let endpoint = config.validate()?;
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(reference) = &config.api_key_env {
            let mut value = reqwest::header::HeaderValue::from_str(&credential(reference)?)
                .context("invalid xberg credential")?;
            value.set_sensitive(true);
            headers.insert("x-xberg-api-key", value);
        }
        Ok(Self {
            endpoint,
            client: reqwest::Client::builder()
                .default_headers(headers)
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(120))
                .build()?,
        })
    }
    pub async fn extract(
        &self,
        bytes: Vec<u8>,
        filename: &str,
        mime_type: &str,
    ) -> Result<ExtractedFile> {
        ensure!(
            !bytes.is_empty() && bytes.len() <= 64 * 1024 * 1024,
            "invalid source file size"
        );
        ensure!(
            !filename.is_empty()
                && filename.len() <= 512
                && !filename.chars().any(char::is_control),
            "invalid extraction filename"
        );
        let part = reqwest::multipart::Part::bytes(bytes)
            .file_name(filename.to_owned())
            .mime_str(mime_type)?;
        let form = reqwest::multipart::Form::new()
            .part("files", part)
            .text("config", r#"{"output_format":"plain"}"#)
            .text("format", "json");
        let mut response = self
            .client
            .post(self.endpoint.clone())
            .multipart(form)
            .send()
            .await
            .map_err(|_| anyhow::anyhow!("xberg request failed"))?;
        response
            .error_for_status_ref()
            .context("xberg extraction failed")?;
        ensure!(
            response.status().is_success(),
            "xberg extraction rejected redirect"
        );
        let mut body = Vec::new();
        while let Some(bytes) = response.chunk().await? {
            ensure!(
                body.len() + bytes.len() <= 16 * 1024 * 1024,
                "xberg response exceeds bound"
            );
            body.extend_from_slice(&bytes);
        }
        let mut files: Vec<ExtractedFile> =
            serde_json::from_slice(&body).map_err(|_| anyhow::anyhow!("invalid xberg response"))?;
        ensure!(files.len() == 1, "ambiguous xberg extraction result");
        let file = files.remove(0);
        ensure!(
            !file.content.trim().is_empty()
                && file.content.len() <= 8 * 1024 * 1024
                && !file.mime_type.trim().is_empty(),
            "unusable xberg document"
        );
        Ok(file)
    }
}
