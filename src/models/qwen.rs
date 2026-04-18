use anyhow::Result;
use chrono::Utc;
use ollama_client::OllamaClient;
use ollama_client::types::Options;
use serde::Deserialize;
use std::pin::pin;
use tokio_stream::StreamExt;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::metabolic::facts::Fact;

const FACT_PREDICATES: &str = "born_on, works_at, knows, spouse_of, parent_of, child_of, \
    located_in, has_skill, member_of, owns, invested_in, life_event, \
    social_trust, invoice_due, client_of, created_on, deadline_on";

pub struct Qwen {
    client: OllamaClient,
    model: String,
}

#[derive(Debug, Deserialize)]
struct RawFact {
    subject: String,
    predicate: String,
    object: String,
    confidence: f32,
}

impl Qwen {
    pub fn new() -> Self {
        Self {
            client: OllamaClient::new(),
            model: "qwen3:8b".to_string(),
        }
    }

    pub async fn extract_facts(&self, chunk: &str, chunk_id: &str) -> Result<Vec<Fact>> {
        let prompt = format!(
            r#"/no_think
Extract atomic facts from the text below. Use ONLY these predicates: {predicates}

Return ONLY a JSON array — no prose, no markdown fences:
[{{"subject":"...","predicate":"...","object":"...","confidence":0.0}}]

If no facts apply, return: []

Text:
{chunk}"#,
            predicates = FACT_PREDICATES,
            chunk = chunk,
        );

        debug!("Extracting facts from chunk (len={})", chunk.len());

        let mut options = Options::default();
        options.temperature = Some(0.1);

        let request = self
            .client
            .generate()
            .model(self.model.as_str())
            .prompt(&prompt)
            .think(false)
            .options(options);

        let stream = request
            .send_stream()
            .await
            .map_err(|e| anyhow::anyhow!("Ollama stream error: {}", e))?;

        let mut stream = pin!(stream);
        let mut full_response = String::new();

        while let Some(chunk_result) = stream.next().await {
            let c = chunk_result.map_err(|e| anyhow::anyhow!("Ollama chunk error: {}", e))?;
            full_response.push_str(&c.response);
        }

        let json_str = extract_json_array(&full_response);

        match serde_json::from_str::<Vec<RawFact>>(&json_str) {
            Ok(raw_facts) => {
                let now = Utc::now().timestamp_micros();
                let facts = raw_facts
                    .into_iter()
                    .filter(|f| !f.subject.trim().is_empty() && !f.object.trim().is_empty())
                    .map(|f| Fact {
                        id: Uuid::new_v4().to_string(),
                        chunk_id: chunk_id.to_string(),
                        subject: f.subject,
                        predicate: f.predicate,
                        object: f.object,
                        confidence: f.confidence.clamp(0.0, 1.0),
                        is_core_truth: false,
                        created_at: now,
                    })
                    .collect();
                Ok(facts)
            }
            Err(e) => {
                warn!(
                    "Failed to parse facts JSON: {} (raw: {:?})",
                    e,
                    &full_response[..full_response.len().min(300)]
                );
                Ok(vec![])
            }
        }
    }
}

fn extract_json_array(s: &str) -> String {
    if let (Some(start), Some(end)) = (s.find('['), s.rfind(']')) {
        if start <= end {
            return s[start..=end].to_string();
        }
    }
    "[]".to_string()
}
