use anyhow::Result;
use arrow_array::Array;
use futures::{StreamExt, TryStreamExt};
use lancedb::connect;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use swiftide::chat_completion::ChatCompletionRequest;
use swiftide::traits::{ChatCompletion, EmbeddingModel};
use tracing::{debug, info, warn};

struct Candidate {
    chunk: String,
    context_window_id: Option<String>,
    path: Option<String>,
}

fn extract_candidates(batches: &[arrow_array::RecordBatch]) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    for batch in batches {
        let Some(chunk_col) = batch.column_by_name("chunk") else {
            continue;
        };
        let Some(chunk_arr) = chunk_col
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
        else {
            continue;
        };
        let cwid_arr = batch
            .column_by_name("context_window_id")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let path_arr = batch
            .column_by_name("path")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());

        for i in 0..chunk_arr.len() {
            if chunk_arr.is_null(i) {
                continue;
            }
            let chunk = chunk_arr.value(i).to_string();
            let context_window_id = cwid_arr
                .filter(|arr| !arr.is_null(i))
                .map(|arr| arr.value(i).to_string());
            let path = path_arr
                .filter(|arr| !arr.is_null(i))
                .map(|arr| arr.value(i).to_string());
            candidates.push(Candidate {
                chunk,
                context_window_id,
                path,
            });
        }
    }
    candidates
}

// Reciprocal Rank Fusion (k=60)
fn rrf_merge(vector_results: Vec<Candidate>, fts_results: Vec<Candidate>) -> Vec<Candidate> {
    // Map chunk text → (rrf_score, context_window_id, path)
    let mut scores: HashMap<String, (f64, Option<String>, Option<String>)> = HashMap::new();

    for (rank, c) in vector_results.iter().enumerate() {
        let entry = scores.entry(c.chunk.clone()).or_insert((
            0.0,
            c.context_window_id.clone(),
            c.path.clone(),
        ));
        entry.0 += 1.0 / (60.0 + rank as f64);
    }
    for (rank, c) in fts_results.iter().enumerate() {
        let entry = scores.entry(c.chunk.clone()).or_insert((
            0.0,
            c.context_window_id.clone(),
            c.path.clone(),
        ));
        entry.0 += 1.0 / (60.0 + rank as f64);
    }

    let mut ranked: Vec<(f64, Candidate)> = scores
        .into_iter()
        .map(|(chunk, (score, cwid, path))| {
            (
                score,
                Candidate {
                    chunk,
                    context_window_id: cwid,
                    path,
                },
            )
        })
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    ranked.into_iter().map(|(_, c)| c).collect()
}

pub async fn run_search(
    query: String,
    embedder: &impl EmbeddingModel,
    completion: &impl ChatCompletion,
    thinking: bool,
) -> Result<()> {
    println!("pulling from memory...");

    let db_path = "aurelius_db";
    let db = connect(db_path).execute().await?;
    let table_name = "chunks";

    let vector = embedder
        .embed(vec![query.clone()])
        .await?
        .into_iter()
        .next()
        .unwrap();
    let table = db.open_table(table_name).execute().await?;

    // 1. Vector search
    let vector_batches = table
        .vector_search(vector)?
        .limit(20)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    // 2. BM25 / FTS search
    let fts_batches = match table
        .query()
        .full_text_search(FullTextSearchQuery::new(query.clone()))
        .limit(20)
        .execute()
        .await
    {
        Ok(stream) => stream.try_collect::<Vec<_>>().await.unwrap_or_default(),
        Err(e) => {
            debug!("FTS search skipped (index may not exist yet): {}", e);
            vec![]
        }
    };

    // 3. RRF merge
    let candidates = rrf_merge(
        extract_candidates(&vector_batches),
        extract_candidates(&fts_batches),
    );
    let total_candidates = candidates.len();
    info!(
        "retrieved {} unique candidates (RRF merged)",
        total_candidates
    );

    // 4. Rerank top 20 → top 5
    let rerank_pool: Vec<&Candidate> = candidates.iter().take(20).collect();

    let top_indices: Vec<usize> = (0..rerank_pool.len()).collect();

    // TODO: re-enable ranking
    // let top_k = 5usize;
    // let pool_texts: Vec<String> = rerank_pool.iter().map(|c| c.chunk.clone()).collect();
    // let top_indices: Vec<usize> = if rerank_pool.len() <= top_k {
    //     (0..rerank_pool.len()).collect()
    // } else {
    //     let mut reranker = TextRerank::try_new(Default::default())?;
    //     let mut results = reranker.rerank::<String>(query.clone(), &pool_texts, false, None)?;
    //     results.sort_by(|a, b| {
    //         b.score
    //             .partial_cmp(&a.score)
    //             .unwrap_or(std::cmp::Ordering::Equal)
    //     });
    //     results.into_iter().take(top_k).map(|r| r.index).collect()
    // };

    let top_candidates: Vec<&Candidate> = top_indices.iter().map(|&i| rerank_pool[i]).collect();

    println!(
        "found {} sources (reranked from {} candidates).",
        top_candidates.len(),
        total_candidates
    );

    // 5. Small-to-big: fetch parent blocks for top candidates
    let unique_cwids: Vec<String> = top_candidates
        .iter()
        .filter_map(|c| c.context_window_id.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let synthesis_context: Vec<String> = if unique_cwids.is_empty() {
        // Old DB without context_window_id — use small chunks directly
        top_candidates.iter().map(|c| c.chunk.clone()).collect()
    } else {
        let ids_list = unique_cwids
            .iter()
            .map(|id| format!("'{}'", id))
            .collect::<Vec<_>>()
            .join(", ");
        let filter = format!("context_window_id IN ({})", ids_list);

        match table.query().only_if(&filter).execute().await {
            Ok(stream) => {
                let parent_batches = stream.try_collect::<Vec<_>>().await.unwrap_or_default();
                let mut seen: HashSet<String> = HashSet::new();
                let mut blocks: Vec<String> = Vec::new();

                for batch in &parent_batches {
                    let cwid_arr = batch
                        .column_by_name("context_window_id")
                        .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
                    let block_arr = batch
                        .column_by_name("parent_block")
                        .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
                    let path_arr = batch
                        .column_by_name("path")
                        .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());

                    if let (Some(cwids_col), Some(blocks_col)) = (cwid_arr, block_arr) {
                        for i in 0..cwids_col.len() {
                            if cwids_col.is_null(i) || blocks_col.is_null(i) {
                                continue;
                            }
                            let cwid = cwids_col.value(i).to_string();
                            if seen.insert(cwid) {
                                let path = path_arr
                                    .filter(|a| !a.is_null(i))
                                    .map(|a| a.value(i))
                                    .unwrap_or("");
                                let text = blocks_col.value(i);
                                if path.is_empty() {
                                    blocks.push(text.to_string());
                                } else {
                                    blocks.push(format!("[{}]\n{}", path, text));
                                }
                            }
                        }
                    }
                }

                if blocks.is_empty() {
                    warn!("Parent blocks not found; falling back to small chunks.");
                    top_candidates.iter().map(|c| c.chunk.clone()).collect()
                } else {
                    blocks
                }
            }
            Err(e) => {
                warn!("Small-to-big lookup failed: {}. Using small chunks.", e);
                top_candidates.iter().map(|c| c.chunk.clone()).collect()
            }
        }
    };

    // 6. Synthesis
    let mut context = String::new();
    for block in &synthesis_context {
        context.push_str(block);
        context.push_str("\n---\n");
    }
    if context.is_empty() {
        context.push_str("No relevant context found in the database.\n");
    }

    let mut completion = completion.complete_stream(&ChatCompletionRequest{
        messages: Cow::Borrowed(&[
            swiftide::chat_completion::ChatMessage::System("You are an AI assistant. Use the provided context to answer the user's query. Always include inline citations to the sources in the context (e.g. [file.txt]). If the context is insufficient to answer, say so.".to_string()),

            swiftide::chat_completion::ChatMessage::User(


            format!("Query: {}\n\nContext:\n{}", query, context),
            )
        ]),
        tools_spec: Default::default()
    }).await;

    while let Some(part) = completion.next().await {
        match part {
            Ok(chunk) => {
                print!("{}", chunk.delta.unwrap().message_chunk.unwrap_or_default());
            }
            Err(e) => {
                eprintln!("Error during completion: {}", e);
                break;
            }
        }
    }

    Ok(())
}
