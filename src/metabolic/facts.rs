use anyhow::Result;
use arrow_array::{self, RecordBatch};
use arrow_schema;
use chrono::Utc;
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, sync::Arc};
use swiftide::{chat_completion::ChatMessage, traits::ChatCompletion};
use uuid::Uuid;

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

pub async fn get_or_create_evicted_facts_table(
    db: &lancedb::Connection,
) -> Result<lancedb::table::Table> {
    let table_name = "evicted_facts";
    let schema = facts_schema();
    match db.open_table(table_name).execute().await {
        Ok(t) => Ok(t),
        Err(_) => Ok(db.create_empty_table(table_name, schema).execute().await?),
    }
}

pub async fn load_all_facts(table: &lancedb::table::Table) -> Result<Vec<Fact>> {
    let batches = table
        .query()
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    let mut facts = Vec::new();
    for batch in &batches {
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

    Ok(facts)
}

pub async fn delete_facts_by_ids(table: &lancedb::table::Table, ids: &[String]) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    for chunk in ids.chunks(500) {
        let id_list = chunk
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(", ");
        table.delete(&format!("id IN ({})", id_list)).await?;
    }
    Ok(())
}

pub async fn delete_all_facts(table: &lancedb::table::Table) -> Result<()> {
    table.delete("id IS NOT NULL").await?;
    Ok(())
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

    table.add(vec![batch]).execute().await?;
    Ok(())
}

#[derive(Debug, PartialEq)]
pub enum ContradictionResolution {
    A,
    B,
    Unresolvable,
}

#[derive(Deserialize)]
struct ResolutionResponse {
    winner: String, // Expected "A", "B", or "Unresolvable"
}

pub async fn resolve_contradiction(
    completion: &dyn ChatCompletion,
    a: &Fact,
    b: &Fact,
) -> Result<ContradictionResolution> {
    let prompt = format!(
        r#"You are a logic engine resolving a knowledge graph contradiction.
Two facts exist for the subject "{subject}" regarding "{predicate}", but they have different objects.

Fact A: {object_a} (Confidence: {conf_a:.2})
Fact B: {object_b} (Confidence: {conf_b:.2})

Which fact is more likely to be correct? Consider that higher confidence usually wins, but if they are equally plausible, or if you cannot logically determine a winner, mark it as Unresolvable.

RESPONSE FORMAT:
Return ONLY valid JSON:
{{
    "winner": "A"
}}
(Values allowed: "A", "B", "Unresolvable")"#,
        subject = a.subject,
        predicate = a.predicate,
        object_a = a.object,
        conf_a = a.confidence,
        object_b = b.object,
        conf_b = b.confidence,
    );

    let response_text = completion
        .complete(&swiftide::chat_completion::ChatCompletionRequest {
            messages: Cow::Borrowed(&[ChatMessage::System(prompt)]),
            tools_spec: Default::default(),
        })
        .await?;

    let res: ResolutionResponse = serde_json::from_str(&response_text.message.unwrap())?;

    match res.winner.to_uppercase().as_str() {
        "A" => Ok(ContradictionResolution::A),
        "B" => Ok(ContradictionResolution::B),
        _ => Ok(ContradictionResolution::Unresolvable),
    }
}

#[derive(Deserialize)]
struct SynthesisResponse {
    predicate: Option<String>,
    object: Option<String>,
    confidence: Option<f32>,
}

/// Synthesizes a group of weak facts into a single "neuron" if they converge.
///
/// This uses an LLM to look for patterns or corroboration across multiple
/// low-confidence facts about the same subject.
pub async fn synthesize_neuron(
    completion: &dyn ChatCompletion,
    subject: &str,
    cluster: &[&Fact],
) -> Result<Option<(String, String, f32)>> {
    // 1. Format the cluster into a readable list for the LLM
    let facts_context = cluster
        .iter()
        .map(|f| {
            format!(
                "- Predicate: {}, Object: {}, Confidence: {:.2}",
                f.predicate, f.object, f.confidence
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        r#"You are a cognitive synthesis engine. You are looking at several "weak" facts (low confidence) about the subject: "{subject}".

        Your task is to determine if these facts converge into a single, stronger conclusion.
        If they do, provide a consolidated predicate and object, and assign a new confidence score (0.0 - 1.0).
        If the facts are too disjointed or contradictory to merge, return null values.

        FACTS TO ANALYZE:
        {facts_context}

        RESPONSE FORMAT:
        Return ONLY valid JSON in this format:
        {{
            "predicate": "merged_predicate",
            "object": "merged_object",
            "confidence": 0.85
        }}
        "#
    );

    // 2. Execute completion
    let response_text = completion
        .complete(&swiftide::chat_completion::ChatCompletionRequest {
            messages: Cow::Borrowed(&[ChatMessage::System(prompt)]),
            tools_spec: Default::default(),
        })
        .await?;

    let synthesis: SynthesisResponse = serde_json::from_str(response_text.message().unwrap())?;

    // 4. Return the result if valid
    if let (Some(p), Some(o), Some(c)) =
        (synthesis.predicate, synthesis.object, synthesis.confidence)
    {
        // Ensure we actually produced a useful neuron
        if c > 0.0 {
            return Ok(Some((p, o, c)));
        }
    }

    Ok(None)
}

pub struct Extractor<'a> {
    pub completion: &'a dyn ChatCompletion,
}

pub const FACT_PREDICATES: &str = "born_on, works_at, knows, spouse_of, parent_of, child_of, \
    located_in, has_skill, member_of, owns, invested_in, life_event, \
    social_trust, invoice_due, client_of, created_on, deadline_on";

pub fn is_functional_predicate(predicate: &str) -> bool {
    matches!(predicate, "born_on" | "created_on" | "deadline_on")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_functional_predicate() {
        assert!(is_functional_predicate("born_on"));
        assert!(is_functional_predicate("created_on"));
        assert!(is_functional_predicate("deadline_on"));
        assert!(!is_functional_predicate("works_at"));
        assert!(!is_functional_predicate("child_of"));
        assert!(!is_functional_predicate("client_of"));
        assert!(!is_functional_predicate("located_in"));
    }
}

#[derive(Debug, Deserialize)]
struct RawFact {
    subject: String,
    predicate: String,
    object: String,
    confidence: f32,
}

impl<'a> Extractor<'a> {
    pub async fn extract_facts(&self, chunk: &str, chunk_id: &str) -> Result<Vec<Fact>> {
        let prompt = format!(
            r#"Extract atomic facts from the text below. Use ONLY these predicates: {predicates}

Return ONLY a JSON array — no prose, no markdown fences:
[{{"subject":"...","predicate":"...","object":"...","confidence":0.0}}]

If no facts apply, return: []

Text:
{chunk}"#,
            predicates = FACT_PREDICATES,
            chunk = chunk,
        );

        let response = self
            .completion
            .complete(&swiftide::chat_completion::ChatCompletionRequest {
                messages: Cow::Borrowed(&[ChatMessage::System(prompt)]),
                tools_spec: Default::default(),
            })
            .await
            .map_err(|e| anyhow::anyhow!("Completion error: {}", e))?;

        let full_response = response.message.unwrap_or_default();
        let json_str = extract_json_array(&full_response);

        match serde_json::from_str::<Vec<RawFact>>(&json_str) {
            Ok(raw_facts) => {
                let now = Utc::now().timestamp_micros();
                let facts = raw_facts
                    .into_iter()
                    .filter(|f| !f.subject.trim().is_empty() && !f.object.trim().is_empty())
                    .map(|f| Fact {
                        id: Uuid::new_v4().to_string(),
                        chunk_id: chunk_id.to_string(),
                        subject: f.subject,
                        predicate: f.predicate,
                        object: f.object,
                        confidence: f.confidence.clamp(0.0, 1.0),
                        is_core_truth: false,
                        created_at: now,
                    })
                    .collect();
                Ok(facts)
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to parse facts JSON: {} (raw: {:?})",
                    e,
                    &full_response[..full_response.len().min(300)]
                );
                Ok(vec![])
            }
        }
    }
}

pub fn extract_json_array(s: &str) -> String {
    if let (Some(start), Some(end)) = (s.find('['), s.rfind(']')) {
        if start <= end {
            return s[start..=end].to_string();
        }
    }
    "[]".to_string()
}
