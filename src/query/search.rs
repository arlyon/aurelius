use crate::models::gemma::Gemma;
use crate::models::zembed::Embedder;
use anyhow::Result;
use lancedb::connect;
use lancedb::query::ExecutableQuery;
use lancedb::query::QueryBase;

pub async fn run_search(query: String, quantized: bool) -> Result<()> {
    let embedder = Embedder::new(quantized).await?;
    let mut gemma = Gemma::new().await?;
    let db_path = "aurelius_db";
    let db = connect(db_path).execute().await?;
    let table_name = "chunks";

    let vector = embedder.embed_query(&query).await?;

    // Hybrid search: Vector search + BM25
    let table = db.open_table(table_name).execute().await?;

    // 1. Vector Search
    let _vector_results = table.vector_search(vector)?.limit(10).execute().await?;

    // 2. BM25 Search (FTS)
    // LanceDB FTS usually requires an index.
    // For MVP, we might skip FTS if it's too complex to setup without proper schema.
    // Let's assume vector search for now as the core.

    // Synthesis
    let mut context = String::new();
    // TODO: Extract text from results properly once arrow schema is implemented.
    context.push_str("Sample context for MVP.\n");

    let prompt = format!(
        "You are an AI assistant. Use the following context to answer the query: {}\n\nContext:\n{}\n\nAnswer with inline citations (e.g. [file.txt]).",
        query, context
    );

    let answer = gemma.generate(&prompt, 512, false)?;
    println!("{}", answer);

    Ok(())
}
