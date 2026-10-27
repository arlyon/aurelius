use async_openai::config::Config;
use reqwest::header::HeaderMap;
use secrecy::SecretString;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct LemonadeConfig {
    api_base: String,
    api_key: SecretString,
}

impl LemonadeConfig {
    pub fn with_api_base(mut self, api_base: String) -> Self {
        self.api_base = api_base;
        self
    }
}

impl Default for LemonadeConfig {
    fn default() -> Self {
        Self {
            api_base: "http://localhost:13305/api/v1".to_string(),
            api_key: String::new().into(),
        }
    }
}

impl Config for LemonadeConfig {
    fn headers(&self) -> HeaderMap {
        HeaderMap::new()
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.api_base)
    }

    fn api_base(&self) -> &str {
        &self.api_base
    }

    fn api_key(&self) -> &SecretString {
        &self.api_key
    }

    fn query(&self) -> Vec<(&str, &str)> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;
    use swiftide_integrations::openai::GenericOpenAIBuilder;

    #[test]
    fn defaults_to_the_local_openai_compatible_endpoint_without_credentials() {
        let config = LemonadeConfig::default();
        assert_eq!(config.api_base(), "http://localhost:13305/api/v1");
        assert_eq!(
            config.url("/embeddings"),
            "http://localhost:13305/api/v1/embeddings"
        );
        assert!(config.headers().is_empty());
        assert!(config.query().is_empty());
        assert!(config.api_key().expose_secret().is_empty());
    }

    #[test]
    fn preserves_custom_endpoint_configuration() {
        let config = LemonadeConfig::default().with_api_base("http://localhost:8099/v1".into());
        assert_eq!(
            config.url("/chat/completions"),
            "http://localhost:8099/v1/chat/completions"
        );
    }

    #[test]
    fn deserializes_the_existing_config_shape() {
        let config: LemonadeConfig =
            serde_json::from_str(r#"{"api_base":"http://localhost:8099/v1"}"#).unwrap();
        assert_eq!(config.api_base(), "http://localhost:8099/v1");
        assert!(config.api_key().expose_secret().is_empty());
    }

    #[test]
    fn supports_swiftide_prompt_and_embedding_models_without_a_fork() {
        let client = GenericOpenAIBuilder::<LemonadeConfig>::default()
            .default_prompt_model("chat-model")
            .default_embed_model("embedding-model")
            .build()
            .unwrap();
        assert_eq!(client.options().prompt_model.as_deref(), Some("chat-model"));
        assert_eq!(
            client.options().embed_model.as_deref(),
            Some("embedding-model")
        );
    }
}
