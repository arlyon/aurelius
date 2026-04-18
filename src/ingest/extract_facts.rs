use anyhow::Result;
use arrow_array::Array;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use std::collections::HashSet;
use tracing::info;

use crate::metabolic::facts::{get_or_create_facts_table, write_facts};
use crate::models::qwen::Qwen;

pub async fn run_extract_facts() -> Result<()> {
    let db_path = "aurelius_db";
    let db = lancedb::connect(db_path).execute().await?;

    let chunks_table = db.open_table("chunks").execute().await.map_err(|_| {
        anyhow::anyhow!("No chunks table found — run `aurelius ingest` first")
    })?;

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

    let qwen = Qwen::new();
    let mut total_facts = 0usize;

    for (i, (context_window_id, parent_block)) in unanalyzed.iter().enumerate() {
        match qwen.extract_facts(parent_block, context_window_id).await {
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
            Err(e) => tracing::warn!(
                "Qwen error for context window {}: {}",
                context_window_id,
                e
            ),
        }
    }

    info!(
        "Fact extraction complete: {} facts across {} context windows",
        total_facts,
        unanalyzed.len()
    );
    Ok(())
}
