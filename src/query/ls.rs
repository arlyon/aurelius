use anyhow::Result;
use arrow_array::{Array, StringArray};
use futures::StreamExt;
use lancedb::connect;
use lancedb::query::ExecutableQuery;
use std::collections::BTreeMap;

pub async fn run_ls() -> Result<()> {
    let db_path = "aurelius_db";
    let db = connect(db_path).execute().await?;
    let table_name = "chunks";

    let table = match db.open_table(table_name).execute().await {
        Ok(t) => t,
        Err(_) => {
            println!(
                "No files found in the database (table '{}' not found).",
                table_name
            );
            return Ok(());
        }
    };

    let mut stream = table.query().execute().await?;

    // Map path to a list of chunks
    let mut file_contents: BTreeMap<String, Vec<String>> = BTreeMap::new();

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;

        let path_col = batch
            .column_by_name("path")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let chunk_col = batch
            .column_by_name("chunk")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());

        if let (Some(path_array), Some(chunk_array)) = (path_col, chunk_col) {
            for i in 0..path_array.len() {
                if !path_array.is_null(i) && !chunk_array.is_null(i) {
                    let path = path_array.value(i).to_string();
                    let chunk = chunk_array.value(i).to_string();
                    file_contents.entry(path).or_default().push(chunk);
                }
            }
        }
    }

    if file_contents.is_empty() {
        println!("No files found in the database.");
    } else {
        for (path, chunks) in file_contents {
            println!("Path: {}", path);
            println!("Content:");
            for chunk in chunks {
                // If the chunk has the "File: ...\n---\n" prefix, we might want to strip it
                // but for now let's just print it as is or try to strip it if it matches.
                let display_chunk = if chunk.starts_with(&format!("File: {}\n---\n", path)) {
                    &chunk[format!("File: {}\n---\n", path).len()..]
                } else {
                    &chunk
                };
                print!("{}", display_chunk);
            }
            println!("\n---");
        }
    }

    Ok(())
}
