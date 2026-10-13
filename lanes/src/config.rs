use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::Path};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub lanes: Vec<Lane>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Lane {
    pub name: String,
    pub organization_id: String,
    pub connection_id: String,
    pub bucket: String,
    pub root: String,
    pub namespace: Option<String>,
    pub stream: String,
    pub id_pointers: Vec<String>,
    pub text_fields: Vec<String>,
    pub revision: Revision,
    pub scope: Scope,
    pub citation: Citation,
    pub deleted_pointer: Option<String>,
    pub mime_type: String,
    pub mode: Mode,
    pub enabled: bool,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Scope {
    Connection,
    Field {
        pointer: String,
        allowed_values: Vec<String>,
    },
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Revision {
    pub pointer: String,
    pub envelope: bool,
    pub unit: RevisionUnit,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Citation {
    Slack {
        workspace_url: String,
        channel_pointer: String,
        timestamp_pointer: String,
    },
    Url {
        pointers: Vec<String>,
    },
    AsanaTask {
        task_pointer: String,
    },
    AsanaStory {
        story_pointer: String,
    },
    None,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RevisionUnit {
    Seconds,
    Milliseconds,
    Rfc3339,
}

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Content,
    Metadata,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let mut names = HashSet::new();
        let mut streams = HashSet::new();
        for lane in &self.lanes {
            ensure!(
                !lane.name.is_empty() && names.insert(&lane.name),
                "invalid or duplicate lane"
            );
            ensure!(
                streams.insert((&lane.connection_id, &lane.namespace, &lane.stream)),
                "duplicate stream mapping"
            );
            ensure!(
                !lane.organization_id.is_empty()
                    && lane
                        .organization_id
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "invalid organization ID"
            );
            ensure!(
                uuid::Uuid::parse_str(&lane.connection_id).is_ok(),
                "connection must be UUID"
            );
            ensure!(!lane.bucket.is_empty(), "bucket required");
            ensure!(
                lane.root == format!("orgs/{}/raw/", lane.organization_id),
                "root must be scoped to organization"
            );
            for part in std::iter::once(&lane.stream).chain(lane.namespace.iter()) {
                ensure!(
                    !part.is_empty()
                        && part
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
                        && part != "."
                        && part != "..",
                    "unsafe stream or namespace"
                );
            }
            ensure!(!lane.id_pointers.is_empty(), "identity mapping required");
            ensure!(
                lane.mode == Mode::Metadata || !lane.text_fields.is_empty(),
                "content mapping required"
            );
            let mut pointers: Vec<&String> = lane
                .id_pointers
                .iter()
                .chain(&lane.text_fields)
                .chain([&lane.revision.pointer])
                .chain(lane.deleted_pointer.iter())
                .collect();
            match &lane.scope {
                Scope::Connection => {}
                Scope::Field {
                    pointer,
                    allowed_values,
                } => {
                    pointers.push(pointer);
                    ensure!(
                        !lane.enabled || !allowed_values.is_empty(),
                        "enabled field scope requires explicit values"
                    );
                }
            }
            match &lane.citation {
                Citation::Slack {
                    workspace_url,
                    channel_pointer,
                    timestamp_pointer,
                } => {
                    let url = reqwest::Url::parse(workspace_url)?;
                    ensure!(
                        url.scheme() == "https"
                            && url.username().is_empty()
                            && url.password().is_none(),
                        "invalid Slack URL"
                    );
                    pointers.extend([channel_pointer, timestamp_pointer]);
                }
                Citation::Url { pointers: urls } => {
                    ensure!(!urls.is_empty(), "citation mapping required");
                    pointers.extend(urls);
                }
                Citation::AsanaTask { task_pointer } => pointers.push(task_pointer),
                Citation::AsanaStory { story_pointer } => pointers.push(story_pointer),
                Citation::None => {
                    ensure!(lane.mode == Mode::Metadata, "content requires a citation")
                }
            }
            ensure!(
                pointers.iter().all(|p| p.starts_with('/')),
                "mappings must be JSON pointers"
            );
            ensure!(
                matches!(
                    lane.mime_type.as_str(),
                    "text/plain" | "text/markdown" | "text/html"
                ),
                "unsupported record MIME type"
            );
        }
        for left in &self.lanes {
            for right in &self.lanes {
                ensure!(
                    left.connection_id != right.connection_id
                        || (left.organization_id == right.organization_id
                            && left.bucket == right.bucket),
                    "connection registered to multiple tenants or buckets"
                );
            }
        }
        Ok(())
    }

    pub fn connection(&self, id: &str) -> Result<Vec<&Lane>> {
        let lanes: Vec<_> = self
            .lanes
            .iter()
            .filter(|lane| lane.connection_id == id && lane.enabled)
            .collect();
        ensure!(!lanes.is_empty(), "unknown or disabled connection");
        Ok(lanes)
    }
    pub fn lane(&self, name: &str) -> Result<&Lane> {
        self.lanes
            .iter()
            .find(|lane| lane.name == name && lane.enabled)
            .context("unknown or disabled lane")
    }
}

impl Lane {
    pub fn batch_prefix(&self, sync_id: i64) -> Result<String> {
        ensure!(sync_id > 0, "invalid sync ID");
        let namespace = self
            .namespace
            .as_ref()
            .map(|n| format!("{n}/"))
            .unwrap_or_default();
        Ok(format!(
            "{}{}/{namespace}{}/",
            self.root, sync_id, self.stream
        ))
    }
    pub fn fingerprint(&self) -> Result<String> {
        Ok(blake3::hash(&serde_json::to_vec(self)?)
            .to_hex()
            .to_string())
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    pub fn fixture() -> Lane {
        Lane {
            name: "demo".into(),
            organization_id: "org_demo".into(),
            connection_id: "11111111-1111-4111-8111-111111111111".into(),
            bucket: "demo".into(),
            root: "orgs/org_demo/raw/".into(),
            namespace: None,
            stream: "messages".into(),
            id_pointers: vec!["/channel_id".into(), "/ts".into()],
            text_fields: vec!["/text".into()],
            revision: Revision {
                pointer: "/ts".into(),
                envelope: false,
                unit: RevisionUnit::Seconds,
            },
            scope: Scope::Field {
                pointer: "/channel_id".into(),
                allowed_values: vec!["CDEMO".into()],
            },
            citation: Citation::Slack {
                workspace_url: "https://demo.slack.com".into(),
                channel_pointer: "/channel_id".into(),
                timestamp_pointer: "/ts".into(),
            },
            deleted_pointer: None,
            mime_type: "text/plain".into(),
            mode: Mode::Content,
            enabled: true,
        }
    }
    #[test]
    fn exact_batch_scope() {
        assert_eq!(
            fixture().batch_prefix(12).unwrap(),
            "orgs/org_demo/raw/12/messages/"
        );
        assert!(fixture().batch_prefix(-1).is_err());
    }
    #[test]
    fn reject_cross_org_root_and_unscoped_lane() {
        let mut lane = fixture();
        lane.root = "orgs/other/raw/".into();
        assert!(Config { lanes: vec![lane] }.validate().is_err());
        let mut lane = fixture();
        lane.scope = Scope::Field {
            pointer: "/channel_id".into(),
            allowed_values: vec![],
        };
        assert!(Config { lanes: vec![lane] }.validate().is_err());
    }
    #[test]
    fn multi_stream_not_multi_tenant() {
        let first = fixture();
        let mut second = fixture();
        second.name = "other".into();
        second.stream = "threads".into();
        assert!(
            Config {
                lanes: vec![first.clone(), second.clone()]
            }
            .validate()
            .is_ok()
        );
        second.organization_id = "org_other".into();
        second.root = "orgs/org_other/raw/".into();
        assert!(
            Config {
                lanes: vec![first, second]
            }
            .validate()
            .is_err()
        );
    }
}
