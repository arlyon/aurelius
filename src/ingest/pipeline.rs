use crate::models::zembed::Embedder;
use anyhow::Result;
use lancedb::connect;
use lancedb::connection::Connection;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::info;

#[derive(Serialize, Deserialize, Clone)]
pub struct Chunk {
    pub text: String,
    pub file_path: String,
    pub vector: Vec<f32>,
}

pub async fn run_ingest(path: PathBuf) -> Result<()> {
    let mut embedder = Embedder::new().await?;
    let db_path = "aurelius_db";
    let db = connect(db_path).execute().await?;
    let table_name = "chunks";

    let files = crate::ingest::walker::walk_directory(path.clone());
    info!("Ingesting {} files from {:?}", files.len(), path);

    for file in files {
        // For MVP, if kreuzberg isn't directly exposed this way, we'll read as text
        // Or we can use the kreuzberg CLI via Command if we want the structural parsing.
        if let Ok(content) = tokio::fs::read_to_string(&file).await {
            let chunks = chunk_markdown(&content, &file.to_string_lossy());

            for mut chunk_text in chunks {
                // Prepend global context
                let global_context = format!("File: {}\n---\n", file.to_string_lossy());
                chunk_text = format!("{}{}", global_context, chunk_text);

                let vector = embedder.embed_document(&chunk_text)?;

                let chunk = Chunk {
                    text: chunk_text,
                    file_path: file.to_string_lossy().to_string(),
                    vector,
                };

                save_chunk(&db, table_name, chunk).await?;
            }
        }
    }

    Ok(())
}

fn chunk_markdown(content: &str, _file_path: &str) -> Vec<String> {
    content.split("\n\n").map(|s| s.to_string()).collect()
}

async fn save_chunk(_db: &Connection, _table_name: &str, _chunk: Chunk) -> Result<()> {
    // TODO: Implement LanceDB table creation and adding data properly with Arrow
    Ok(())
}
