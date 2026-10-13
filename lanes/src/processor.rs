use crate::config::{Citation, Lane, Mode, RevisionUnit, Scope};
use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use kreuzberg::{ChunkingConfig, ExtractionConfig, extract_bytes};
use serde_json::Value;
use std::{
    io::Read,
    sync::{Arc, Mutex},
};
use swiftide_core::{
    EmbeddingModel, Embeddings,
    chat_completion::errors::LanguageModelError,
    indexing::{EmbeddedField, IndexingStream, TextNode},
};
use swiftide_indexing::{Pipeline, transformers::Embed};

pub const DIMENSIONS: usize = 1280;
pub const MAX_OBJECT_BYTES: usize = 64 * 1024 * 1024;
pub const FINGERPRINT: &str = "kreuzberg-4.10.4/text-1500-200/Qwen3-Embedding-4B/f4602530/Q4_K_M/last/mrl1280/l2/query-instruct-v1/review-feed-v1";

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct SourceMetadata {
    pub source_type: String,
    pub author_id: Option<String>,
    pub author_is_bot: Option<bool>,
    pub channel_id: Option<String>,
    pub thread_ts: Option<String>,
    pub source_modified_at: Option<DateTime<Utc>>,
}

fn source_metadata(data: &Value, lane: &Lane) -> Result<SourceMetadata> {
    let slack = lane.namespace.as_deref() == Some("slack")
        || matches!(lane.stream.as_str(), "channel_messages" | "threads");
    let bot = data.get("bot_id").is_some_and(|value| !value.is_null())
        || data.get("subtype").and_then(Value::as_str) == Some("bot_message");
    let author = data.get("user").and_then(scalar);
    let source_modified_at = if slack {
        data.pointer("/edited/ts")
            .or_else(|| data.get("ts"))
            .and_then(scalar)
            .map(|value| timestamp(&value))
            .transpose()?
    } else {
        data.get("modified_at")
            .or_else(|| data.get("created_at"))
            .and_then(scalar)
            .map(|value| timestamp(&value))
            .transpose()?
    };
    Ok(SourceMetadata {
        source_type: if slack { "slack" } else { "document" }.into(),
        author_id: author.clone(),
        author_is_bot: if slack {
            if bot {
                Some(true)
            } else {
                author.as_ref().map(|_| false)
            }
        } else {
            None
        },
        channel_id: if slack {
            data.get("channel_id").and_then(scalar)
        } else {
            None
        },
        thread_ts: if slack {
            data.get("thread_ts")
                .or_else(|| data.get("ts"))
                .and_then(scalar)
        } else {
            None
        },
        source_modified_at,
    })
}

#[derive(Debug)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub revision: String,
    pub modified_at: DateTime<Utc>,
    pub source_uri: String,
    pub record_index: i64,
    pub scope_value: String,
    pub deleted: bool,
    pub metadata: SourceMetadata,
}
impl Document {
    pub fn hash(&self) -> String {
        blake3::hash(
            format!(
                "{}\0{}\0{}\0{:?}",
                self.text, self.source_uri, self.deleted, self.metadata
            )
            .as_bytes(),
        )
        .to_hex()
        .to_string()
    }
}
fn scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

pub fn timestamp(value: &str) -> Result<DateTime<Utc>> {
    if let Ok(time) = DateTime::parse_from_rfc3339(value) {
        return Ok(time.with_timezone(&Utc));
    }
    let (seconds, fraction) = value.split_once('.').unwrap_or((value, ""));
    ensure!(
        fraction.len() <= 9 && fraction.chars().all(|c| c.is_ascii_digit()),
        "invalid timestamp"
    );
    DateTime::from_timestamp(seconds.parse()?, format!("{fraction:0<9}").parse()?)
        .context("timestamp out of range")
}

pub fn parse_batch(bytes: &[u8], lane: &Lane) -> Result<Vec<Document>> {
    ensure!(
        bytes.len() <= MAX_OBJECT_BYTES,
        "compressed object exceeds limit"
    );
    let mut output = Vec::new();
    GzDecoder::new(bytes)
        .take(MAX_OBJECT_BYTES as u64 + 1)
        .read_to_end(&mut output)?;
    ensure!(
        output.len() <= MAX_OBJECT_BYTES,
        "decompressed object exceeds limit"
    );
    let mut documents = Vec::new();
    for (index, line) in std::str::from_utf8(&output)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        ensure!(line.len() <= 1024 * 1024, "record exceeds limit");
        let record: Value = serde_json::from_str(line).context("invalid Airbyte JSONL")?;
        let data = record
            .get("_airbyte_data")
            .context("missing Airbyte data envelope")?;
        // The parent message also arrives in channel_messages; don't index it twice as a thread.
        if lane.stream == "threads"
            && data
                .get("thread_ts")
                .and_then(scalar)
                .is_none_or(|ts| data.get("ts").and_then(scalar).as_deref() == Some(ts.as_str()))
        {
            continue;
        }
        let ids: Vec<String> = lane
            .id_pointers
            .iter()
            .map(|p| {
                data.pointer(p)
                    .and_then(scalar)
                    .context("missing source identity")
            })
            .collect::<Result<_>>()?;
        let revision_data = if lane.revision.envelope {
            &record
        } else {
            data
        };
        let revision = revision_data
            .pointer(&lane.revision.pointer)
            .and_then(scalar)
            .context("missing source revision")?;
        let modified_at = match lane.revision.unit {
            RevisionUnit::Seconds => timestamp(&revision)?,
            RevisionUnit::Milliseconds => DateTime::from_timestamp_millis(revision.parse()?)
                .context("timestamp out of range")?,
            RevisionUnit::Rfc3339 => DateTime::parse_from_rfc3339(&revision)?.with_timezone(&Utc),
        };
        let (scope_value, in_scope) = match &lane.scope {
            Scope::Connection => (lane.connection_id.clone(), true),
            Scope::Field {
                pointer,
                allowed_values,
            } => {
                let value = data
                    .pointer(pointer)
                    .and_then(scalar)
                    .context("missing record scope")?;
                let allowed = allowed_values.contains(&value);
                (value, allowed)
            }
        };
        let deleted = !in_scope
            || match lane.deleted_pointer.as_ref().and_then(|p| data.pointer(p)) {
                Some(value) => value.as_bool().context("deletion flag must be boolean")?,
                None => false,
            };
        let text = if deleted {
            String::new()
        } else if lane.mode == Mode::Metadata {
            serde_json::to_string(data)?
        } else {
            lane.text_fields
                .iter()
                .filter_map(|pointer| {
                    data.pointer(pointer).filter(|v| !v.is_null()).map(|v| {
                        format!(
                            "{}: {}",
                            pointer.trim_start_matches('/'),
                            v.as_str()
                                .map(str::to_owned)
                                .unwrap_or_else(|| v.to_string())
                        )
                    })
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let source_uri = if deleted {
            String::new()
        } else {
            match &lane.citation {
                Citation::Slack {
                    workspace_url,
                    channel_pointer,
                    timestamp_pointer,
                } => {
                    let channel = data
                        .pointer(channel_pointer)
                        .and_then(scalar)
                        .context("missing Slack channel")?;
                    let ts = data
                        .pointer(timestamp_pointer)
                        .and_then(scalar)
                        .context("missing Slack timestamp")?;
                    ensure!(
                        channel.chars().all(|c| c.is_ascii_alphanumeric())
                            && ts.chars().all(|c| c.is_ascii_digit() || c == '.'),
                        "invalid Slack citation"
                    );
                    format!(
                        "{}/archives/{}/p{}",
                        workspace_url.trim_end_matches('/'),
                        channel,
                        ts.replace('.', "")
                    )
                }
                Citation::Url { pointers } => {
                    let url = pointers
                        .iter()
                        .find_map(|p| {
                            data.pointer(p)
                                .and_then(Value::as_str)
                                .filter(|s| !s.is_empty())
                        })
                        .context("missing source URL")?;
                    let parsed = reqwest::Url::parse(url)?;
                    ensure!(
                        parsed.scheme() == "https"
                            && parsed.username().is_empty()
                            && parsed.password().is_none(),
                        "invalid source URL"
                    );
                    url.to_string()
                }
                Citation::AsanaTask { task_pointer } => {
                    let id = data
                        .pointer(task_pointer)
                        .and_then(scalar)
                        .context("missing Asana task")?;
                    ensure!(
                        id.chars().all(|c| c.is_ascii_digit()),
                        "invalid Asana task ID"
                    );
                    format!("https://app.asana.com/0/0/{id}")
                }
                Citation::AsanaStory { story_pointer } => {
                    let id = data
                        .pointer(story_pointer)
                        .and_then(scalar)
                        .context("missing Asana story")?;
                    ensure!(
                        id.chars().all(|c| c.is_ascii_digit()),
                        "invalid Asana story ID"
                    );
                    // The connector omits the parent task ID; retain the canonical API locator.
                    format!("https://app.asana.com/api/1.0/stories/{id}")
                }
                Citation::None => String::new(),
            }
        };
        documents.push(Document {
            id: serde_json::to_string(&ids)?,
            text,
            revision,
            modified_at,
            source_uri,
            record_index: i64::try_from(index)?,
            scope_value,
            deleted,
            metadata: source_metadata(data, lane)?,
        });
    }
    Ok(documents)
}

#[derive(Clone, Debug)]
pub struct BundledEmbedder;

pub fn project(mut vector: Vec<f32>) -> Result<Vec<f32>> {
    ensure!(vector.len() == 2560, "unexpected native model dimension");
    ensure!(vector.iter().all(|n| n.is_finite()), "non-finite embedding");
    vector.truncate(DIMENSIONS);
    let norm = vector
        .iter()
        .map(|n| f64::from(*n).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(norm.is_finite() && norm > 0.0, "zero embedding");
    for value in &mut vector {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(vector)
}

impl BundledEmbedder {
    pub async fn vectors(&self, input: Vec<String>) -> Result<Embeddings> {
        #[derive(serde::Deserialize)]
        struct Item {
            index: usize,
            embedding: Vec<f32>,
        }
        #[derive(serde::Deserialize)]
        struct Response {
            data: Vec<Item>,
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(std::time::Duration::from_secs(180))
            .build()?;
        let mut vectors = Vec::with_capacity(input.len());
        // One document per request bounds CPU graph memory independently of object size.
        for text in input {
            ensure!(text.len() <= 16000, "embedding input exceeds bound");
            let mut result: Response = client
                .post("http://127.0.0.1:8081/v1/embeddings")
                .json(&serde_json::json!({"model":"Qwen3-Embedding-4B", "input":[text]}))
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            ensure!(
                result.data.len() == 1 && result.data[0].index == 0,
                "embedding count mismatch"
            );
            vectors.push(project(result.data.remove(0).embedding)?);
        }
        Ok(vectors)
    }
    pub async fn query(&self, text: &str) -> Result<Vec<f32>> {
        self.vectors(vec![format!("Instruct: Given a question, retrieve relevant workplace documents that answer the question\nQuery: {text}")])
            .await?.pop().context("missing query embedding")
    }
}
#[async_trait::async_trait]
impl EmbeddingModel for BundledEmbedder {
    async fn embed(
        &self,
        input: Vec<String>,
    ) -> std::result::Result<Embeddings, LanguageModelError> {
        self.vectors(input).await.map_err(Into::into)
    }
}

pub async fn chunks(document: &Document, lane: &Lane) -> Result<Vec<TextNode>> {
    if document.deleted || document.text.trim().is_empty() || lane.mode == Mode::Metadata {
        return Ok(Vec::new());
    }
    let result = extract_bytes(
        document.text.as_bytes(),
        &lane.mime_type,
        &ExtractionConfig {
            chunking: Some(ChunkingConfig::new(1500, 200, true)),
            ..Default::default()
        },
    )
    .await?;
    let nodes = result
        .chunks
        .context("Kreuzberg did not return chunks")?
        .into_iter()
        .map(|chunk| {
            let mut node = TextNode::new(chunk.content);
            node.offset = chunk.metadata.byte_start;
            node.metadata
                .insert("source_uri", document.source_uri.clone());
            node
        })
        .collect();
    let output = Arc::new(Mutex::new(Vec::new()));
    let destination = Arc::clone(&output);
    Pipeline::from_stream(IndexingStream::from_nodes(nodes))
        .with_concurrency(1)
        .then_in_batch(
            Embed::new(BundledEmbedder)
                .with_batch_size(1)
                .with_concurrency(1),
        )
        .then(move |node: TextNode| {
            destination
                .lock()
                .map_err(|_| anyhow::anyhow!("chunk collector poisoned"))?
                .push(node.clone());
            Ok(node)
        })
        .run()
        .await?;
    let mut guard = output
        .lock()
        .map_err(|_| anyhow::anyhow!("chunk collector poisoned"))?;
    Ok(std::mem::take(&mut *guard))
}
pub fn vector(node: &TextNode) -> Result<Vec<f32>> {
    node.vectors
        .as_ref()
        .and_then(|v| {
            v.get(&EmbeddedField::Combined)
                .or_else(|| v.get(&EmbeddedField::Chunk))
        })
        .cloned()
        .context("missing vector")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tests::fixture;
    use flate2::{Compression, write::GzEncoder};
    use std::io::Write;
    fn gzip(s: &str) -> Vec<u8> {
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(s.as_bytes()).unwrap();
        enc.finish().unwrap()
    }
    #[test]
    fn slack_author_thread_and_edit_time_are_not_extraction_time() {
        let mut lane = fixture();
        lane.namespace = Some("slack".into());
        lane.revision = crate::config::Revision {
            pointer: "/_airbyte_extracted_at".into(),
            envelope: true,
            unit: RevisionUnit::Milliseconds,
        };
        let data = r#"{"_airbyte_extracted_at":1790000000000,"_airbyte_data":{"channel_id":"CDEMO","ts":"1700000000.000001","thread_ts":"1699999999.000001","user":"U1","edited":{"ts":"1700000100.000001"},"text":"human correction"}}"#;
        let doc = parse_batch(&gzip(data), &lane).unwrap().remove(0);
        assert_eq!(doc.metadata.author_id.as_deref(), Some("U1"));
        assert_eq!(doc.metadata.author_is_bot, Some(false));
        assert_eq!(doc.metadata.thread_ts.as_deref(), Some("1699999999.000001"));
        assert!(doc.metadata.source_modified_at.unwrap() < doc.modified_at);
        assert_eq!(doc.revision, "1790000000000");
    }
    #[test]
    fn slack_bots_and_unknown_authors_are_explicit() {
        let mut lane = fixture();
        lane.namespace = Some("slack".into());
        let bot = source_metadata(
            &serde_json::json!({"user":"U1","bot_id":"B1","ts":"1700000000"}),
            &lane,
        )
        .unwrap();
        assert_eq!(bot.author_is_bot, Some(true));
        let unknown = source_metadata(&serde_json::json!({"ts":"1700000000"}), &lane).unwrap();
        assert_eq!(unknown.author_is_bot, None);
    }
    #[test]
    fn projection_is_finite_normalized_and_exact_dimension() {
        let vector = project(vec![1.0; 2560]).unwrap();
        assert_eq!(vector.len(), DIMENSIONS);
        assert!((vector.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 0.0001);
        assert!(project(vec![0.0; 2560]).is_err());
        assert!(project(vec![1.0; 1024]).is_err());
        assert!(project(vec![f32::NAN; 2560]).is_err());
    }
    #[test]
    fn scope_revocation_not_silent_skip() {
        let record =
            r#"{"_airbyte_data":{"channel_id":"CDEMO","ts":"1700000000.000001","text":"first"}}"#;
        let docs = parse_batch(&gzip(record), &fixture()).unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].source_uri.ends_with("/CDEMO/p1700000000000001"));
        let mut lane = fixture();
        lane.scope = Scope::Field {
            pointer: "/channel_id".into(),
            allowed_values: vec!["COTHER".into()],
        };
        let docs = parse_batch(&gzip(record), &lane).unwrap();
        assert!(docs[0].deleted);
        assert!(docs[0].text.is_empty());
    }
    #[test]
    fn invalid_envelope_and_timestamp() {
        assert!(parse_batch(&gzip("{}"), &fixture()).is_err());
        assert!(parse_batch(b"not gzip", &fixture()).is_err());
        assert!(timestamp("nan").is_err());
    }
    #[test]
    fn thread_parent_deduplicated() {
        let record = r#"{"_airbyte_data":{"channel_id":"CDEMO","ts":"1700000000.000001","thread_ts":"1700000000.000001","text":"parent"}}"#;
        let mut lane = fixture();
        lane.stream = "threads".into();
        assert!(parse_batch(&gzip(record), &lane).unwrap().is_empty());
    }
}
