use anyhow::Result;
use arrow_array::{Array, StringArray, StructArray};
use futures::StreamExt;
use lancedb::connect;
use lancedb::query::ExecutableQuery;
use std::collections::HashSet;

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

    // Select all columns instead of forcing "metadata" to avoid schema errors if empty
    let mut stream = table.query().execute().await?;

    let mut paths = HashSet::new();

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;

        // LanceDB usually stores metadata as a StructArray.
        // We'll try to find the "path" field inside the metadata struct.
        if let Some(metadata_col) = batch.column_by_name("metadata") {
            if let Some(metadata_struct) = metadata_col.as_any().downcast_ref::<StructArray>() {
                if let Some(path_col) = metadata_struct.column_by_name("path") {
                    if let Some(path_array) = path_col.as_any().downcast_ref::<StringArray>() {
                        for i in 0..path_array.len() {
                            if !path_array.is_null(i) {
                                paths.insert(path_array.value(i).to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    if paths.is_empty() {
        println!("No files found in the database.");
    } else {
        let mut sorted_paths: Vec<_> = paths.into_iter().collect();
        sorted_paths.sort();
        for path in sorted_paths {
            println!("{}", path);
        }
    }

    Ok(())
}
