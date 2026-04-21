use anyhow::Result;
use arrow_array::Array;
use chrono::Utc;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashSet;
use swiftide::chat_completion::{ChatCompletionRequest, ChatMessage};
use swiftide::traits::ChatCompletion;
use tracing::{debug, warn};
use tracing::{info, trace};
use uuid::Uuid;

use crate::metabolic::facts::Fact;

use crate::metabolic::facts::{get_or_create_facts_table, write_facts};

pub async fn run_extract_facts(completion: &impl ChatCompletion) -> Result<()> {
    let db_path = "aurelius_db";
    let db = lancedb::connect(db_path).execute().await?;

    let chunks_table = db
        .open_table("chunks")
        .execute()
        .await
        .map_err(|_| anyhow::anyhow!("No chunks table found — run `aurelius ingest` first"))?;

    let facts_table = get_or_create_facts_table(&db).await?;

    // Collect already-analyzed context_window_ids from the facts table
    let analyzed: HashSet<String> = {
        let batches = facts_table
            .query()
            .select(Select::columns(&["chunk_id"]))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("chunk_id")
                    .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>())
                    .map(|arr| {
                        (0..arr.len())
                            .filter_map(|i| arr.value(i).to_string().into())
                            .collect::<Vec<String>>()
                    })
                    .unwrap_or_default()
            })
            .collect()
    };

    info!(
        "Found {} already-analyzed context windows, scanning chunks...",
        analyzed.len()
    );

    // Scan chunks for unique (context_window_id, parent_block) pairs not yet analyzed
    let batches = chunks_table
        .query()
        .select(Select::columns(&["context_window_id", "parent_block"]))
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    let mut seen = HashSet::<String>::new();
    let mut unanalyzed: Vec<(String, String)> = Vec::new();

    for batch in &batches {
        let Some(cwid_col) = batch.column_by_name("context_window_id") else {
            continue;
        };
        let Some(pb_col) = batch.column_by_name("parent_block") else {
            continue;
        };
        let Some(cwid_arr) = cwid_col.as_any().downcast_ref::<arrow_array::StringArray>() else {
            continue;
        };
        let Some(pb_arr) = pb_col.as_any().downcast_ref::<arrow_array::StringArray>() else {
            continue;
        };

        for i in 0..cwid_arr.len() {
            if cwid_arr.is_null(i) || pb_arr.is_null(i) {
                continue;
            }
            let cwid = cwid_arr.value(i).to_string();
            let parent = pb_arr.value(i).to_string();

            if analyzed.contains(&cwid) || seen.contains(&cwid) {
                continue;
            }
            seen.insert(cwid.clone());
            unanalyzed.push((cwid, parent));
        }
    }

    if unanalyzed.is_empty() {
        info!("All chunks already analyzed — nothing to do");
        return Ok(());
    }

    info!(
        "Extracting facts from {} unanalyzed context windows...",
        unanalyzed.len()
    );

    let mut total_facts = 0usize;
    let extractor = Extractor { completion };

    for (i, (context_window_id, parent_block)) in unanalyzed.iter().enumerate() {
        match extractor
            .extract_facts(parent_block, context_window_id)
            .await
        {
            Ok(facts) => {
                let n = facts.len();
                if let Err(e) = write_facts(&facts_table, &facts).await {
                    tracing::warn!("Failed to write facts for {}: {}", context_window_id, e);
                } else {
                    total_facts += n;
                    info!(
                        "[{}/{}] {} facts from context window {}",
                        i + 1,
                        unanalyzed.len(),
                        n,
                        &context_window_id[..8.min(context_window_id.len())]
                    );
                }
            }
            Err(e) => tracing::warn!("Qwen error for context window {}: {}", context_window_id, e),
        }
    }

    info!(
        "Fact extraction complete: {} facts across {} context windows",
        total_facts,
        unanalyzed.len()
    );
    Ok(())
}

struct Extractor<'a> {
    completion: &'a dyn ChatCompletion,
}

const FACT_PREDICATES: &str = "born_on, works_at, knows, spouse_of, parent_of, child_of, \
    located_in, has_skill, member_of, owns, invested_in, life_event, \
    social_trust, invoice_due, client_of, created_on, deadline_on";

#[derive(Debug, Deserialize)]
struct RawFact {
    subject: String,
    predicate: String,
    object: String,
    confidence: f32,
}

impl<'a> Extractor<'a> {
    pub async fn extract_facts(&self, chunk: &str, chunk_id: &str) -> Result<Vec<Fact>> {
        let prompt = format!(
            r#"Extract atomic facts from the text below. Use ONLY these predicates: {predicates}

Return ONLY a JSON array — no prose, no markdown fences:
[{{"subject":"...","predicate":"...","object":"...","confidence":0.0}}]

If no facts apply, return: []

Text:
{chunk}"#,
            predicates = FACT_PREDICATES,
            chunk = chunk,
        );

        debug!("Extracting facts from chunk (len={})", chunk.len());
        trace!("Prompt: {}", &prompt);

        let response = self
            .completion
            .complete(&ChatCompletionRequest {
                messages: Cow::Borrowed(&[ChatMessage::System(prompt)]),
                tools_spec: Default::default(),
            })
            .await
            .map_err(|e| anyhow::anyhow!("Completion error: {}", e))?;

        let full_response = response.message.unwrap_or_default();
        debug!(
            "Response (len={}): {:?}",
            full_response.len(),
            &full_response[..full_response.len().min(200)]
        );

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

#[cfg(test)]
mod tests {

    const OLLAMA_URL: &str = "http://localhost:11434/v1/chat/completions";
    const MODEL: &str = "gemma4:26b";

    /// Sends a streaming chat request to Ollama and collects the full response text.
    /// `think` controls whether Ollama's extended thinking is enabled.
    async fn stream_ollama(prompt: &str, think: bool) -> String {
        let body = serde_json::json!({
            "model": MODEL,
            "stream": true,
            "options": {
                "think": think
            },
            "think": think,
            "messages": [{"role": "user", "content": prompt}]
        });

        let response = reqwest::Client::new()
            .post(OLLAMA_URL)
            .json(&body)
            .send()
            .await
            .expect("request to Ollama failed");

        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        assert!(status.is_success(), "Ollama returned {status}: {text}");

        // Parse SSE lines and accumulate content/thinking deltas
        let mut content = String::new();
        let mut thinking = String::new();

        for line in text.lines() {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }
            let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            if let Some(delta) = chunk["choices"][0]["delta"].as_object() {
                println!("{:?}", delta);
                if let Some(t) = delta.get("reasoning").and_then(|v| v.as_str()) {
                    thinking.push_str(t);
                }
                if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
                    content.push_str(c);
                }
            }
        }

        eprintln!(
            "[think={think}] thinking ({} chars): {:?}",
            thinking.len(),
            &thinking[..thinking.len().min(120)]
        );
        eprintln!("[think={think}] content: {content:?}");

        content
    }

    /// Run with: cargo test -- --ignored ollama_stream_no_think
    #[tokio::test]
    #[ignore = "requires running Ollama with gemma4:26b"]
    async fn ollama_stream_no_think() {
        let content = stream_ollama("What is 2 + 2? Reply with just the number.", false).await;
        assert!(!content.is_empty(), "response should not be empty");
    }

    /// Run with: cargo test -- --ignored ollama_stream_with_think
    #[tokio::test]
    #[ignore = "requires running Ollama with gemma4:26b"]
    async fn ollama_stream_with_think() {
        let content = stream_ollama("What is 2 + 2? Reply with just the number.", true).await;
        assert!(!content.is_empty(), "response should not be empty");
    }
}
