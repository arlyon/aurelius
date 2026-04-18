use anyhow::Result;
use arrow_array::{self, RecordBatch, RecordBatchIterator};
use arrow_schema;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub chunk_id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
    pub is_core_truth: bool,
    pub created_at: i64,
}

pub fn facts_schema() -> Arc<arrow_schema::Schema> {
    Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("id", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("chunk_id", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("subject", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("predicate", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("object", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("confidence", arrow_schema::DataType::Float32, false),
        arrow_schema::Field::new("is_core_truth", arrow_schema::DataType::Boolean, false),
        arrow_schema::Field::new(
            "created_at",
            arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            false,
        ),
    ]))
}

pub async fn get_or_create_facts_table(db: &lancedb::Connection) -> Result<lancedb::table::Table> {
    let table_name = "facts";
    let schema = facts_schema();
    match db.open_table(table_name).execute().await {
        Ok(t) => Ok(t),
        Err(_) => Ok(db.create_empty_table(table_name, schema).execute().await?),
    }
}

pub async fn write_facts(table: &lancedb::table::Table, facts: &[Fact]) -> Result<()> {
    if facts.is_empty() {
        return Ok(());
    }

    let schema = facts_schema();
    let now = Utc::now().timestamp_micros();

    let ids: Vec<&str> = facts.iter().map(|f| f.id.as_str()).collect();
    let chunk_ids: Vec<&str> = facts.iter().map(|f| f.chunk_id.as_str()).collect();
    let subjects: Vec<&str> = facts.iter().map(|f| f.subject.as_str()).collect();
    let predicates: Vec<&str> = facts.iter().map(|f| f.predicate.as_str()).collect();
    let objects: Vec<&str> = facts.iter().map(|f| f.object.as_str()).collect();
    let confidences: Vec<f32> = facts.iter().map(|f| f.confidence).collect();
    let is_core_truths: Vec<bool> = facts.iter().map(|f| f.is_core_truth).collect();
    let created_ats: Vec<i64> = facts
        .iter()
        .map(|f| if f.created_at == 0 { now } else { f.created_at })
        .collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(arrow_array::StringArray::from(ids)),
            Arc::new(arrow_array::StringArray::from(chunk_ids)),
            Arc::new(arrow_array::StringArray::from(subjects)),
            Arc::new(arrow_array::StringArray::from(predicates)),
            Arc::new(arrow_array::StringArray::from(objects)),
            Arc::new(arrow_array::Float32Array::from(confidences)),
            Arc::new(arrow_array::BooleanArray::from(is_core_truths)),
            Arc::new(arrow_array::TimestampMicrosecondArray::from(created_ats)),
        ],
    )?;

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    table.add(reader).execute().await?;
    Ok(())
}
