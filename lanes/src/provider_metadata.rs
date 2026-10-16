use crate::{document::SourceRef, lanes::ObjectRef};
use anyhow::{Context, Result, ensure};
use chrono::{DateTime, FixedOffset, NaiveDate};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PROVIDER_METADATA_VERSION: &str = "source-provider-properties-v2";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderTaskHeader {
    pub text: String,
    pub title: crate::contracts::ByteRange,
}
impl ProviderTaskHeader {
    pub fn validate(&self, document: &crate::contracts::Document) -> Result<()> {
        ensure!(
            self.text.len() <= 32768
                && self.text.starts_with("Task: ")
                && document.text.starts_with(&self.text),
            "provider header does not match source projection"
        );
        ensure!(
            self.title.start == 6
                && self.title.end <= self.text.len()
                && !self.title.text(document)?.is_empty(),
            "invalid provider title boundary"
        );
        Ok(())
    }
    pub fn contains_metadata(&self, range: crate::contracts::ByteRange) -> bool {
        range.start < 6 || (range.start < self.text.len() && range.end > self.title.end)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderActor {
    pub provider: String,
    pub id: String,
    pub name: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProviderValue {
    Date { value: NaiveDate },
    Timestamp { value: DateTime<FixedOffset> },
    Boolean { value: bool },
    Text { value: String },
    Actor { value: ProviderActor },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderProperty {
    pub property: String,
    pub source_field: String,
    pub value: ProviderValue,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderProperties {
    pub contract_version: String,
    pub provider: String,
    pub task_id: String,
    pub source: SourceRef,
    pub object: ObjectRef,
    pub source_row_hash: String,
    pub authored_at: Option<DateTime<FixedOffset>>,
    pub modified_at: Option<DateTime<FixedOffset>>,
    pub author: Option<ProviderActor>,
    pub properties: Vec<ProviderProperty>,
    #[serde(default)]
    pub header: Option<ProviderTaskHeader>,
}
fn text(value: &Value) -> Result<&str> {
    let text = value.as_str().context("provider field is not text")?;
    ensure!(
        !text.is_empty() && text.len() <= 512 && !text.chars().any(char::is_control),
        "invalid provider text field"
    );
    Ok(text)
}
fn present<'a>(data: &'a Value, key: &str) -> Option<&'a Value> {
    data.get(key).filter(|value| !value.is_null())
}
fn actor(value: &Value) -> Result<ProviderActor> {
    Ok(ProviderActor {
        provider: "asana".into(),
        id: text(value.get("gid").context("provider actor ID missing")?)?.into(),
        name: present(value, "name")
            .map(text)
            .transpose()?
            .map(String::from),
    })
}
pub fn asana_properties(
    data: &Value,
    source: SourceRef,
    object: ObjectRef,
    row_hash: String,
) -> Result<ProviderProperties> {
    ensure!(
        row_hash.len() == 64
            && row_hash.bytes().all(|byte| byte.is_ascii_hexdigit())
            && !object.version.is_empty(),
        "provider content/version fingerprint missing"
    );
    ensure!(
        source.stream == "tasks" && source.record_index >= 0 && source.object_key == object.key,
        "provider provenance mismatch"
    );
    let id = text(data.get("gid").context("provider task ID missing")?)?;
    ensure!(
        format!("tasks/{id}") == source.record_id,
        "provider task identity mismatch"
    );
    let mut properties = Vec::new();
    if let Some(value) = present(data, "due_on") {
        properties.push(ProviderProperty {
            property: "due_date".into(),
            source_field: "due_on".into(),
            value: ProviderValue::Date {
                value: NaiveDate::parse_from_str(text(value)?, "%Y-%m-%d")?,
            },
        });
    }
    if let Some(value) = present(data, "due_at") {
        properties.push(ProviderProperty {
            property: "due_at".into(),
            source_field: "due_at".into(),
            value: ProviderValue::Timestamp {
                value: DateTime::parse_from_rfc3339(text(value)?)?,
            },
        });
    }
    if let Some(value) = present(data, "completed") {
        properties.push(ProviderProperty {
            property: "completed".into(),
            source_field: "completed".into(),
            value: ProviderValue::Boolean {
                value: value
                    .as_bool()
                    .context("provider completion is not boolean")?,
            },
        });
    }
    if let Some(value) = present(data, "approval_status") {
        properties.push(ProviderProperty {
            property: "approval_status".into(),
            source_field: "approval_status".into(),
            value: ProviderValue::Text {
                value: text(value)?.into(),
            },
        });
    }
    if let Some(value) = present(data, "assignee") {
        properties.push(ProviderProperty {
            property: "assigned_to".into(),
            source_field: "assignee".into(),
            value: ProviderValue::Actor {
                value: actor(value)?,
            },
        });
    }
    let timestamp = |key| -> Result<Option<DateTime<FixedOffset>>> {
        present(data, key)
            .map(|value| Ok(DateTime::parse_from_rfc3339(text(value)?)?))
            .transpose()
    };
    Ok(ProviderProperties {
        contract_version: PROVIDER_METADATA_VERSION.into(),
        provider: "asana".into(),
        task_id: id.into(),
        source,
        object,
        source_row_hash: row_hash,
        authored_at: timestamp("created_at")?,
        modified_at: timestamp("modified_at")?,
        author: present(data, "created_by").map(actor).transpose()?,
        properties,
        header: Some(ProviderTaskHeader {
            text: crate::transform::asana_header(data),
            title: crate::contracts::ByteRange {
                start: 6,
                end: 6 + data
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("Untitled Asana task")
                    .len(),
            },
        }),
    })
}

#[cfg(feature = "postgres")]
mod runtime {
    use super::*;
    use crate::{
        config::Config, document::DocumentKind, lanes::Normalizer, postgres::ExtractionInput,
        source::SourceClient, storage,
    };
    use object_store::{ObjectStore, ObjectStoreExt, path::Path};
    use std::{collections::BTreeMap, sync::Arc};
    #[derive(Clone)]
    pub struct ProviderMetadataReader {
        config: Config,
        source: SourceClient,
        stores: BTreeMap<String, Arc<dyn ObjectStore>>,
    }
    impl ProviderMetadataReader {
        pub fn new(config: &Config) -> Result<Option<Self>> {
            let tasks: Vec<_> = config
                .lanes
                .iter()
                .filter(|(_, lane)| matches!(lane.normalizer, Normalizer::Task))
                .collect();
            if tasks.is_empty() {
                return Ok(None);
            }
            let storage = config
                .storage
                .as_ref()
                .context("raw storage required for deterministic task metadata")?;
            let stores = tasks
                .into_iter()
                .map(|(name, lane)| Ok((name.clone(), storage.connect(&lane.bucket)?)))
                .collect::<Result<_>>()?;
            Ok(Some(Self::with_storage(config.clone(), stores)?))
        }
        pub fn with_storage(
            config: Config,
            stores: BTreeMap<String, Arc<dyn ObjectStore>>,
        ) -> Result<Self> {
            let source = SourceClient::new(
                config
                    .source
                    .clone()
                    .context("source API required for provider metadata")?,
            )?;
            Ok(Self {
                config,
                source,
                stores,
            })
        }
        pub async fn read(&self, input: &ExtractionInput) -> Result<Option<ProviderProperties>> {
            if input.kind != DocumentKind::AsanaTask {
                return Ok(None);
            }
            let (name, lane) = self
                .config
                .lanes
                .iter()
                .find(|(name, lane)| {
                    lane.scope == input.document.key.scope
                        && lane.connection == input.connection_id
                        && matches!(lane.normalizer, Normalizer::Task)
                        && self
                            .config
                            .lane_request(name)
                            .is_ok_and(|request| request.profile == input.embedding_profile)
                })
                .context("task metadata source lane is not configured")?;
            let sources: Vec<_> = input
                .sources
                .iter()
                .filter(|source| source.stream == "tasks")
                .collect();
            let source = sources.first().context("task source citation missing")?;
            ensure!(
                sources.iter().all(|other| *other == *source),
                "task source citation is ambiguous"
            );
            ensure!(
                crate::engine::LaneEngine::object_stream(lane, &source.object_key)? == "tasks",
                "raw task stream mismatch"
            );
            let relative = source
                .object_key
                .strip_prefix(&lane.raw_prefix)
                .context("task metadata outside configured raw prefix")?;
            let (epoch, _) = relative.split_once('/').context("raw task epoch missing")?;
            let epoch = epoch.parse::<u64>()?;
            ensure!(epoch > 0, "invalid raw task epoch");
            self.source.verify(&lane.connection, epoch).await?;
            let store = self.stores.get(name).context("raw task store missing")?;
            let head = store
                .head(&Path::from(source.object_key.as_str()))
                .await
                .map_err(|_| anyhow::anyhow!("raw task metadata HEAD failed"))?;
            let object = ObjectRef {
                key: source.object_key.clone(),
                version: head.e_tag.context("raw task ETag required")?,
            };
            let bytes = storage::read(store.as_ref(), &object, 128 * 1024 * 1024).await?;
            let decoded = crate::engine::decode_jsonl(&bytes)?;
            ensure!(
                decoded.len() <= 128 * 1024 * 1024,
                "decoded task metadata exceeds bound"
            );
            let index = usize::try_from(source.record_index)?;
            let row = std::str::from_utf8(&decoded)?
                .lines()
                .nth(index)
                .context("raw task row missing")?;
            ensure!(row.len() <= 8 * 1024 * 1024, "raw task row exceeds bound");
            let value: Value = serde_json::from_str(row)?;
            let data = value
                .get("_airbyte_data")
                .context("raw task envelope missing")?;
            Ok(Some(
                asana_properties(
                    data,
                    (**source).clone(),
                    object,
                    blake3::hash(row.as_bytes()).to_hex().to_string(),
                )
                .context(crate::facts::ExtractionReview::Input)?,
            ))
        }
    }
}
#[cfg(feature = "postgres")]
pub use runtime::ProviderMetadataReader;

#[cfg(test)]
mod tests {
    use super::*;
    fn decode(value: Value) -> Result<ProviderProperties> {
        let key = "fixture/raw/1/tasks/task.jsonl";
        let hash = blake3::hash(serde_json::to_string(&value)?.as_bytes())
            .to_hex()
            .to_string();
        asana_properties(
            &value,
            SourceRef {
                stream: "tasks".into(),
                record_id: "tasks/task-1".into(),
                uri: "https://example.invalid/task-1".into(),
                object_key: key.into(),
                record_index: 0,
            },
            ObjectRef {
                key: key.into(),
                version: "fixture-etag".into(),
            },
            hash,
        )
    }
    #[test]
    fn absent_provider_fields_do_not_invent_status_or_dates() -> Result<()> {
        let result = decode(
            serde_json::json!({"gid":"task-1","due_on":null,"due_at":null,"completed":null,"assignee":null}),
        )?;
        assert!(result.properties.is_empty());
        assert!(
            result.authored_at.is_none() && result.modified_at.is_none() && result.author.is_none()
        );
        Ok(())
    }
    #[test]
    fn malformed_or_wrong_identity_metadata_is_rejected() {
        for value in [
            serde_json::json!({"gid":"another-task","completed":false}),
            serde_json::json!({"gid":"task-1","due_on":"2026-02-30"}),
            serde_json::json!({"gid":"task-1","completed":"false"}),
            serde_json::json!({"gid":"task-1","assignee":{"name":"Alice"}}),
        ] {
            assert!(decode(value).is_err());
        }
    }
}
