use crate::models::gemma::Gemma;
use crate::models::zembed::Embedder;
use anyhow::Result;
use arrow_array::Array;
use futures::TryStreamExt;
use lancedb::connect;
use lancedb::query::ExecutableQuery;
use lancedb::query::QueryBase;

pub async fn run_search(query: String, quantized: bool) -> Result<()> {
    let embedder = Embedder::new(quantized).await?;
    let gemma = Gemma::new().await.ok();
    let db_path = "aurelius_db";
    let db = connect(db_path).execute().await?;
    let table_name = "chunks";

    let vector = embedder.embed_query(&query).await?;

    // Hybrid search: Vector search + BM25
    let table = db.open_table(table_name).execute().await?;

    // 1. Vector Search
    let vector_results = table.vector_search(vector)?.limit(10).execute().await?;
    let batches = vector_results.try_collect::<Vec<_>>().await?;

    // Synthesis
    let mut context = String::new();
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
                }
            }
        }
    }

    if context.is_empty() {
        context.push_str("No relevant context found in the database.\n");
    }

    println!("--- Retrieved Context ---");
    println!("{}", context);
    println!("-------------------------\n");

    if let Some(mut g) = gemma {
        let prompt = format!(
            "You are an AI assistant. Use the following context to answer the query: {}\n\nContext:\n{}\n\nAnswer with inline citations (e.g. [file.txt]).",
            query, context
        );

        match g.generate(&prompt, 512, false) {
            Ok(answer) => println!("Synthesis:\n{}", answer),
            Err(e) => {
                eprintln!("Warning: Synthesis failed: {}. Showing only context.", e);
                println!(
                    "Synthesis: (Could not generate answer, please refer to the context above.)"
                );
            }
        }
    } else {
        println!("Synthesis: (LLM model could not be initialized, showing context only.)");
    }

    Ok(())
}
