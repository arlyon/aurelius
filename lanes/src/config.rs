use crate::ontology::Ontology;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{io::Read, path::Path};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub ontology: Ontology,
    pub chat: Option<ChatConfig>,
    pub temporal: Option<TemporalConfig>,
    pub webhook: Option<crate::notification::WebhookConfig>,
    pub language: Option<crate::language::LanguageConfig>,
    pub fact_context: Option<crate::fact_context::FactContextConfig>,
    pub fact_quality: Option<crate::quality::QualityConfig>,
    pub xberg: Option<crate::xberg::XbergConfig>,
    pub storage: Option<crate::storage::StorageConfig>,
    pub postgres: Option<crate::postgres::PostgresConfig>,
    pub source: Option<crate::source::SourceConfig>,
    pub embedding: Option<crate::embedding::EmbeddingConfig>,
    pub reduction: Option<crate::reduction::ReductionConfig>,
    #[serde(default)]
    pub lanes: std::collections::BTreeMap<String, crate::lanes::LaneConfig>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalConfig {
    pub address: String,
    pub namespace: String,
    pub task_queue: String,
    pub api_key_env: Option<String>,
    pub client_cert_env: Option<String>,
    pub client_key_env: Option<String>,
}
impl TemporalConfig {
    pub fn validate(&self) -> Result<()> {
        let url = reqwest::Url::parse(&self.address)?;
        ensure!(
            url.scheme() == "https"
                || (url.scheme() == "http"
                    && matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"))),
            "Temporal address requires TLS or loopback HTTP"
        );
        ensure!(
            url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none(),
            "invalid Temporal address"
        );
        ensure!(
            self.client_cert_env.is_some() == self.client_key_env.is_some(),
            "Temporal mTLS requires both certificate and key references"
        );
        for name in self
            .api_key_env
            .iter()
            .chain(self.client_cert_env.iter())
            .chain(self.client_key_env.iter())
        {
            validate_env_name(name)?;
            ensure!(url.scheme() == "https", "Temporal credentials require TLS");
        }
        for value in [&self.namespace, &self.task_queue] {
            ensure!(
                !value.trim().is_empty()
                    && value.len() <= 200
                    && !value.chars().any(char::is_control),
                "invalid Temporal worker identity"
            );
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatConfig {
    pub endpoint: String,
    pub model: String,
    pub revision: String,
    // Configuration names an environment variable; credentials never enter workflow payloads.
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub api: ChatApi,
    pub gateway: Option<GatewayConfig>,
    #[serde(default)]
    pub references: crate::references::ReferenceConfig,
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatApi {
    #[default]
    ChatCompletions,
    Responses,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayConfig {
    pub token_env: String,
    pub byok_alias: Option<String>,
}
impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let mut text = String::new();
        std::fs::File::open(path)?
            .take(128 * 1024 + 1)
            .read_to_string(&mut text)?;
        ensure!(text.len() <= 128 * 1024, "configuration exceeds bound");
        Self::parse(&text)
    }
    pub fn parse(text: &str) -> Result<Self> {
        ensure!(text.len() <= 128 * 1024, "configuration exceeds bound");
        let config: Self = toml::from_str(text).context("invalid knowledge TOML")?;
        config.validate()?;
        Ok(config)
    }
    pub fn lane_request(&self, name: &str) -> Result<crate::lanes::LaneRequest> {
        let lane = self.lanes.get(name).context("unknown lane")?;
        let embedding = self.embedding.as_ref().map(|e| {
            (
                &e.model,
                &e.revision,
                e.native_dimensions,
                e.dimensions,
                e.token_limit,
            )
        });
        let signature = serde_json::to_vec(&(
            (
                &lane.scope,
                &lane.connection,
                lane.normalizer,
                &lane.bucket,
                &lane.raw_prefix,
                &lane.streams,
                &lane.citation_base,
            ),
            self.ontology.semantic_hash()?,
            embedding,
            "canonical-pg-ingestion-v1",
        ))?;
        Ok(crate::lanes::LaneRequest {
            lane: name.into(),
            profile: blake3::hash(&signature).to_hex().to_string(),
        })
    }
    pub fn extraction_profile(&self) -> Result<String> {
        let chat = self.chat.as_ref().context("chat configuration required")?;
        let base = crate::facts::ChatExtractor::profile_for(chat, &self.ontology)?;
        if let Some(policy) = &self.fact_quality {
            #[cfg(feature = "language")]
            return policy.profile(
                &base,
                &crate::fact_preparation::FactPreparer::profile_for(self)?,
            );
            #[cfg(not(feature = "language"))]
            {
                let _ = policy;
                anyhow::bail!("prepared fact extraction requires the language build feature");
            }
        }
        ensure!(
            self.language.is_none() && self.fact_context.is_none(),
            "fact preparation configured without an explicit extraction policy; refusing legacy fallback"
        );
        Ok(base)
    }
    pub fn validate(&self) -> Result<()> {
        self.ontology.validate()?;
        if let Some(chat) = &self.chat {
            chat.validate()?;
        }
        if let Some(temporal) = &self.temporal {
            temporal.validate()?;
        }
        if let Some(webhook) = &self.webhook {
            webhook.validate()?;
            ensure!(
                self.temporal.is_some() && self.postgres.is_some() && self.source.is_some(),
                "webhook requires Temporal, Postgres and source API configuration"
            );
            let connections: std::collections::BTreeSet<_> =
                self.lanes.values().map(|lane| &lane.connection).collect();
            ensure!(
                connections.len() == self.lanes.len(),
                "webhook connection maps to more than one lane"
            );
        }
        if let Some(xberg) = &self.xberg {
            xberg.validate()?;
        }
        if let Some(storage) = &self.storage {
            storage.validate()?;
        }
        if let Some(postgres) = &self.postgres {
            postgres.validate()?;
        }
        if let Some(language) = &self.language {
            language.validate()?;
        }
        if let Some(context) = &self.fact_context {
            context.validate()?;
        }
        if let Some(quality) = &self.fact_quality {
            quality.validate(&self.ontology)?;
            if quality.format == crate::quality::ExtractionFormat::OpenClaimsV3 {
                ensure!(
                    self.chat
                        .as_ref()
                        .is_some_and(|chat| matches!(chat.api, ChatApi::Responses)),
                    "open claims require the native Responses API"
                );
            }
            ensure!(
                self.language.is_some() && self.fact_context.is_some(),
                "typed fact extraction requires local language and canonical context configuration"
            );
        }
        if let Some(source) = &self.source {
            source.validate()?;
        }
        if let Some(embedding) = &self.embedding {
            embedding.validate()?;
        }
        if let Some(reduction) = &self.reduction {
            reduction.validate()?;
        }
        ensure!(self.lanes.len() <= 100, "too many lanes");
        for (name, lane) in &self.lanes {
            ensure!(
                !name.is_empty()
                    && name.len() <= 100
                    && name
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-')),
                "invalid lane name"
            );
            lane.validate()?;
        }
        Ok(())
    }
}
impl ChatConfig {
    pub fn validate(&self) -> Result<reqwest::Url> {
        self.references.validate()?;
        if let Some(gateway) = &self.gateway {
            validate_env_name(&gateway.token_env)?;
            ensure!(
                self.api_key_env.is_none(),
                "gateway BYOK cannot also supply provider credentials"
            );
            if let Some(alias) = &gateway.byok_alias {
                ensure!(
                    !alias.trim().is_empty()
                        && alias.len() <= 100
                        && alias
                            .bytes()
                            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.')),
                    "invalid gateway key alias"
                );
            }
        }
        let endpoint = reqwest::Url::parse(&self.endpoint)?;
        ensure!(
            endpoint.scheme() == "https"
                || (endpoint.scheme() == "http"
                    && matches!(
                        endpoint.host_str(),
                        Some("localhost" | "127.0.0.1" | "[::1]")
                    )),
            "chat endpoint requires HTTPS or loopback HTTP"
        );
        ensure!(
            endpoint.username().is_empty()
                && endpoint.password().is_none()
                && endpoint.query().is_none()
                && endpoint.fragment().is_none(),
            "credentials must not be embedded in endpoint"
        );
        if self.gateway.is_some() {
            ensure!(
                (endpoint.scheme() == "https"
                    && endpoint.host_str() == Some("gateway.ai.cloudflare.com"))
                    || matches!(
                        endpoint.host_str(),
                        Some("localhost" | "127.0.0.1" | "[::1]")
                    ),
                "gateway credentials require the Cloudflare gateway host or loopback"
            );
        }
        for value in [&self.model, &self.revision] {
            ensure!(
                !value.trim().is_empty()
                    && value.len() <= 200
                    && !value.chars().any(char::is_control),
                "model and immutable revision required"
            );
        }
        if let Some(name) = &self.api_key_env {
            validate_env_name(name)?;
        }
        Ok(endpoint)
    }
}
pub(crate) fn validate_env_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name.len() <= 100
            && name
                .bytes()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_')
            && !name.as_bytes()[0].is_ascii_digit(),
        "invalid credential environment reference"
    );
    Ok(())
}
pub(crate) fn credential(name: &str) -> Result<String> {
    validate_env_name(name)?;
    let value =
        std::env::var(name).map_err(|_| anyhow::anyhow!("missing or invalid credential"))?;
    ensure!(
        !value.trim().is_empty() && value.len() <= 64 * 1024,
        "invalid credential size"
    );
    Ok(value)
}
