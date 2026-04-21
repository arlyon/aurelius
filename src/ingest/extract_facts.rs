use anyhow::Result;
use arrow_array::Array;
use futures::TryStreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use std::collections::{HashMap, HashSet};
use swiftide::traits::ChatCompletion;
use tracing::debug;
use tracing::info;

use crate::metabolic::facts::{
    get_or_create_facts_table, write_facts, Extractor,
};

fn create_progress_bar(len: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg:>12} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(message.to_string());
    pb
}

pub async fn run_extract_facts(completion: &dyn ChatCompletion) -> Result<()> {
    let db_path = crate::persistence::db_path();
    let db = lancedb::connect(&db_path).execute().await?;

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
        .select(Select::columns(&["context_window_id", "parent_block", "path"]))
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    let mut seen = HashSet::<String>::new();
    let mut grouped_unanalyzed: HashMap<String, Vec<(String, String)>> = HashMap::new();

    for batch in &batches {
        let Some(cwid_col) = batch.column_by_name("context_window_id") else {
            continue;
        };
        let Some(pb_col) = batch.column_by_name("parent_block") else {
            continue;
        };
        let Some(path_col) = batch.column_by_name("path") else {
            continue;
        };
        let Some(cwid_arr) = cwid_col.as_any().downcast_ref::<arrow_array::StringArray>() else {
            continue;
        };
        let Some(pb_arr) = pb_col.as_any().downcast_ref::<arrow_array::StringArray>() else {
            continue;
        };
        let Some(path_arr) = path_col.as_any().downcast_ref::<arrow_array::StringArray>() else {
            continue;
        };

        for i in 0..cwid_arr.len() {
            if cwid_arr.is_null(i) || pb_arr.is_null(i) || path_arr.is_null(i) {
                continue;
            }
            let cwid = cwid_arr.value(i).to_string();
            let parent = pb_arr.value(i).to_string();
            let path = path_arr.value(i).to_string();

            if analyzed.contains(&cwid) || seen.contains(&cwid) {
                continue;
            }
            seen.insert(cwid.clone());
            grouped_unanalyzed
                .entry(path)
                .or_default()
                .push((cwid, parent));
        }
    }

    if grouped_unanalyzed.is_empty() {
        info!("All chunks already analyzed — nothing to do");
        return Ok(());
    }

    let total_unanalyzed: usize = grouped_unanalyzed.values().map(|v| v.len()).sum();
    info!(
        "Extracting facts from {} unanalyzed context windows across {} files...",
        total_unanalyzed,
        grouped_unanalyzed.len()
    );

    let mut total_facts = 0usize;
    let extractor = Extractor { completion };

    let pb = create_progress_bar(grouped_unanalyzed.len() as u64, "Facts");

    for (path, chunks) in grouped_unanalyzed {
        for (context_window_id, parent_block) in chunks {
            match extractor
                .extract_facts(&parent_block, &context_window_id)
                .await
            {
                Ok(facts) => {
                    let n = facts.len();
                    if let Err(e) = write_facts(&facts_table, &facts).await {
                        tracing::warn!("Failed to write facts for {}: {}", context_window_id, e);
                    } else {
                        total_facts += n;
                        debug!(
                            "Extracted {} facts from context window {}",
                            n,
                            &context_window_id[..8.min(context_window_id.len())]
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!("Qwen error for context window {}: {}", context_window_id, e)
                }
            }
        }
        pb.suspend(|| {
            eprintln!("{}: fact extracted", path);
        });
        pb.inc(1);
    }

    pb.finish_and_clear();

    info!(
        "Fact extraction complete: {} facts across {} files",
        total_facts,
        total_unanalyzed
    );
    Ok(())
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
