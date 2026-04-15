use crate::models::gemma::Gemma;
use crate::models::zembed::Embedder;
use anyhow::Result;
use arrow_array::Array;
use futures::TryStreamExt;
use lancedb::connect;
use lancedb::query::ExecutableQuery;
use lancedb::query::QueryBase;
use tracing::{debug, info, warn};

pub async fn run_search(query: String, quantized: bool, ollama: bool) -> Result<()> {
    println!("pulling from memory...");

    let embedder = Embedder::new(quantized, ollama).await?;
    let gemma = Gemma::new(ollama).await.ok();

    let db_path = "aurelius_db";
    let db = connect(db_path).execute().await?;
    let table_name = "chunks";

    let vector = embedder.embed_query(&query).await?;

    let table = db.open_table(table_name).execute().await?;

    // 1. Vector Search
    let vector_results = table.vector_search(vector)?.limit(10).execute().await?;
    let batches = vector_results.try_collect::<Vec<_>>().await?;

    // Synthesis
    let mut context = String::new();
    let mut node_count = 0;
    for batch in batches {
        if let Some(chunk_col) = batch.column_by_name("chunk")
            && let Some(chunk_array) = chunk_col
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
        {
            for i in 0..chunk_array.len() {
                if !chunk_array.is_null(i) {
                    context.push_str(chunk_array.value(i));
                    context.push_str("\n---\n");
                    node_count += 1;
                }
            }
        }
    }

    println!("found {} sources.", node_count);

    if context.is_empty() {
        context.push_str("No relevant context found in the database.\n");
    }

    if let Some(mut g) = gemma {
        let prompt = format!(
            "<|think|>\nYou are an AI assistant. Use the following context to answer the query: {}\n\nContext:\n{}\n\nAnswer with inline citations (e.g. [file.txt]).",
            query, context
        );

        match g.generate(&prompt, 512, true).await {
            Ok(_answer) => {
                // Synthesis complete is handled by streaming output
            }
            Err(e) => {
                warn!("Synthesis failed: {}. Showing only context.", e);
                println!(
                    "\nSynthesis: (Could not generate answer, please refer to the context above.)"
                );
            }
        }
    } else {
        println!("\nSynthesis: (LLM model could not be initialized, showing context only.)");
        println!("--- Retrieved Context ---\n{}", context);
    }

    Ok(())
}
