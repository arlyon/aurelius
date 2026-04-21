use anyhow::Result;
use arrow_array::{Array, RecordBatch, StringArray};
use futures::{StreamExt, TryStreamExt};
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::collections::HashMap;
use crate::metabolic::facts::Fact;
pub async fn run_facts(
    subject: Option<String>,
    predicate: Option<String>,
    object: Option<String>,
    source: Option<String>,
    core_only: bool,
    evicted: bool,
    sort: String,
    limit: usize,
    json: bool,
    stats: bool,
    show_source: bool,
) -> Result<()> {
    let db_path = crate::persistence::db_path();
    let db = connect(&db_path).execute().await?;
    let table_name = if evicted { "evicted_facts" } else { "facts" };

    let table = match db.open_table(table_name).execute().await {
        Ok(t) => t,
        Err(_) => {
            if json {
                println!("[]");
            } else {
                println!("No {} table found.", table_name);
            }
            return Ok(());
        }
    };

    let mut query_builder = table.query();

    let mut filters = Vec::new();
    if let Some(s) = subject {
        filters.push(format!("subject LIKE '%{}%'", s.replace('\'', "''")));
    }
    if let Some(p) = predicate {
        filters.push(format!("predicate LIKE '%{}%'", p.replace('\'', "''")));
    }
    if let Some(o) = object {
        filters.push(format!("object LIKE '%{}%'", o.replace('\'', "''")));
    }
    if core_only {
        filters.push("is_core_truth = true".to_string());
    }

    if let Some(s) = source {
        // Find matching chunk IDs first
        let chunks_table = db.open_table("chunks").execute().await?;
        let filter = format!("path LIKE '%{}%'", s.replace('\'', "''"));
        let mut stream = chunks_table.query().only_if(filter).select(lancedb::query::Select::columns(&["context_window_id"])).execute().await?;
        let mut matching_cwids = std::collections::HashSet::new();
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            if let Some(col) = batch.column_by_name("context_window_id").and_then(|c| c.as_any().downcast_ref::<StringArray>()) {
                for i in 0..col.len() {
                    if !col.is_null(i) {
                        matching_cwids.insert(col.value(i).to_string());
                    }
                }
            }
        }
        
        if matching_cwids.is_empty() {
            if json {
                println!("[]");
            } else {
                println!("No facts found for the specified source.");
            }
            return Ok(());
        }
        
        let id_list = matching_cwids.into_iter().map(|id| format!("'{}'", id.replace('\'', "''"))).collect::<Vec<_>>().join(", ");
        filters.push(format!("chunk_id IN ({})", id_list));
    }

    if !filters.is_empty() {
        query_builder = query_builder.only_if(filters.join(" AND "));
    }

    // Since we don't have order_by in this version of lancedb-rust, we sort in memory
    // For large datasets, we should ideally use a server-side order_by if supported
    // but for now we fetch all matching facts and sort them.
    // If limit is provided, we still fetch all to ensure correct global sorting.

    let batches = query_builder.execute().await?.try_collect::<Vec<RecordBatch>>().await?;
    let mut facts = extract_facts_from_batches(&batches);

    if sort == "confidence" {
        facts.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        facts.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    }

    let facts: Vec<Fact> = facts.into_iter().take(limit).collect();

    if stats {
        display_stats(&facts);
        return Ok(());
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&facts)?);
        return Ok(());
    }

    let mut sources = HashMap::new();
    if show_source && !facts.is_empty() {
        sources = fetch_sources(&db, &facts).await?;
    }

    display_table(&facts, &sources);

    Ok(())
}

fn extract_facts_from_batches(batches: &[RecordBatch]) -> Vec<Fact> {
    let mut facts = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let chunk_ids = batch
            .column_by_name("chunk_id")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let subjects = batch
            .column_by_name("subject")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let predicates = batch
            .column_by_name("predicate")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let objects = batch
            .column_by_name("object")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let confidences = batch
            .column_by_name("confidence")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>());
        let is_core_truths = batch
            .column_by_name("is_core_truth")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::BooleanArray>());
        let created_ats = batch.column_by_name("created_at").and_then(|c| {
            c.as_any()
                .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
        });

        let (
            Some(ids),
            Some(chunk_ids),
            Some(subjects),
            Some(predicates),
            Some(objects),
            Some(confidences),
            Some(is_core_truths),
            Some(created_ats),
        ) = (
            ids,
            chunk_ids,
            subjects,
            predicates,
            objects,
            confidences,
            is_core_truths,
            created_ats,
        )
        else {
            continue;
        };

        for i in 0..batch.num_rows() {
            facts.push(Fact {
                id: ids.value(i).to_string(),
                chunk_id: chunk_ids.value(i).to_string(),
                subject: subjects.value(i).to_string(),
                predicate: predicates.value(i).to_string(),
                object: objects.value(i).to_string(),
                confidence: confidences.value(i),
                is_core_truth: is_core_truths.value(i),
                created_at: created_ats.value(i),
            });
        }
    }
    facts
}

async fn fetch_sources(db: &lancedb::Connection, facts: &[Fact]) -> Result<HashMap<String, String>> {
    let mut sources = HashMap::new();
    let chunk_ids: Vec<String> = facts.iter().map(|f| f.chunk_id.clone()).collect();
    if chunk_ids.is_empty() {
        return Ok(sources);
    }

    let chunks_table = db.open_table("chunks").execute().await?;
    
    // Process in chunks to avoid too large filter string
    for chunk_group in chunk_ids.chunks(100) {
        let id_list = chunk_group
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(", ");
        let filter = format!("context_window_id IN ({})", id_list);

        let mut stream = chunks_table.query().only_if(filter).execute().await?;
        while let Some(batch_result) = stream.next().await {
            let batch: RecordBatch = batch_result?;
            let id_col = batch
                .column_by_name("context_window_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let path_col = batch
                .column_by_name("path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            if let (Some(ids), Some(paths)) = (id_col, path_col) {
                for i in 0..ids.len() {
                    if !ids.is_null(i) && !paths.is_null(i) {
                        sources.insert(ids.value(i).to_string(), paths.value(i).to_string());
                    }
                }
            }
        }
    }

    Ok(sources)
}

fn display_stats(facts: &[Fact]) {
    if facts.is_empty() {
        println!("No facts to analyze.");
        return;
    }

    let total = facts.len();
    let avg_confidence: f32 = facts.iter().map(|f| f.confidence).sum::<f32>() / total as f32;
    
    let mut subject_counts = HashMap::new();
    for f in facts {
        *subject_counts.entry(&f.subject).or_insert(0) += 1;
    }
    let mut top_entities: Vec<_> = subject_counts.into_iter().collect();
    top_entities.sort_by(|a, b| b.1.cmp(&a.1));

    println!("--- Knowledge Graph Stats ---");
    println!("Total facts:      {}", total);
    println!("Avg confidence:   {:.2}", avg_confidence);
    println!("Top entities:");
    for (entity, count) in top_entities.iter().take(5) {
        println!("  - {}: {}", entity, count);
    }
}

use crossterm::style::{Color, Stylize};

fn display_table(facts: &[Fact], sources: &HashMap<String, String>) {
    if facts.is_empty() {
        println!("No facts found.");
        return;
    }

    // Determine column widths
    let mut id_w = 4;
    let mut sub_w = 7;
    let mut pred_w = 9;
    let mut obj_w = 6;
    let conf_w = 4;
    let mut src_w = 6;

    for f in facts {
        id_w = id_w.max(f.id[..4.min(f.id.len())].len());
        sub_w = sub_w.max(f.subject.len());
        pred_w = pred_w.max(f.predicate.len());
        obj_w = obj_w.max(f.object.len());
        if !sources.is_empty() {
            let src = sources.get(&f.chunk_id).map(|s| s.as_str()).unwrap_or("[Core Truth]");
            src_w = src_w.max(src.len());
        }
    }

    // Print header
    let header = if sources.is_empty() {
        format!(
            "| {:<id_w$} | {:<sub_w$} | {:<pred_w$} | {:<obj_w$} | {:<conf_w$} |",
            "ID", "Subject", "Predicate", "Object", "Conf",
            id_w = id_w, sub_w = sub_w, pred_w = pred_w, obj_w = obj_w, conf_w = conf_w
        )
    } else {
        format!(
            "| {:<id_w$} | {:<sub_w$} | {:<pred_w$} | {:<obj_w$} | {:<conf_w$} | {:<src_w$} |",
            "ID", "Subject", "Predicate", "Object", "Conf", "Source",
            id_w = id_w, sub_w = sub_w, pred_w = pred_w, obj_w = obj_w, conf_w = conf_w, src_w = src_w
        )
    };
    println!("{}", header);
    println!("{}", "-".repeat(header.len()));

    // Print rows
    for f in facts {
        let id_short = &f.id[..4.min(f.id.len())];
        let conf_str = format!("{:.2}", f.confidence);
        
        let conf_colored = if f.confidence >= 0.8 {
            conf_str.with(Color::Green)
        } else if f.confidence >= 0.4 {
            conf_str.with(Color::Yellow)
        } else {
            conf_str.with(Color::Red)
        };
        
        if sources.is_empty() {
            print!("| {:<id_w$} | {:<sub_w$} | {:<pred_w$} | {:<obj_w$} | ",
                id_short, f.subject, f.predicate, f.object, id_w = id_w, sub_w = sub_w, pred_w = pred_w, obj_w = obj_w);
            println!("{:>conf_w$} |", conf_colored, conf_w = conf_w);
        } else {
            let src = sources.get(&f.chunk_id).map(|s| s.as_str()).unwrap_or("[Core Truth]");
            print!("| {:<id_w$} | {:<sub_w$} | {:<pred_w$} | {:<obj_w$} | ",
                id_short, f.subject, f.predicate, f.object, id_w = id_w, sub_w = sub_w, pred_w = pred_w, obj_w = obj_w);
            print!("{:>conf_w$} | ", conf_colored, conf_w = conf_w);
            println!("{:<src_w$} |", src, src_w = src_w);
        }
    }
}
