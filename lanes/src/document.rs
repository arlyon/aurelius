use anyhow::{Result, ensure};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const TRANSFORM_VERSION: &str = "document-sections-v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocumentKind {
    SlackConversation,
    AsanaTask,
    GenericText,
    MailThread,
    File,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceRef {
    pub stream: String,
    pub record_id: String,
    pub uri: String,
    pub object_key: String,
    pub record_index: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Section {
    pub text: String,
    pub author: Option<String>,
    pub occurred_at: Option<DateTime<Utc>>,
    pub source: SourceRef,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalDocument {
    pub id: String,
    pub organization_id: String,
    pub connection_id: String,
    pub kind: DocumentKind,
    pub title: String,
    pub sections: Vec<Section>,
    pub context: BTreeMap<String, String>,
    pub modified_at: DateTime<Utc>,
}

impl CanonicalDocument {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.id.is_empty(), "document identity required");
        ensure!(
            !self.organization_id.is_empty() && !self.connection_id.is_empty(),
            "document scope required"
        );
        ensure!(
            self.sections.len() <= 20000,
            "document section bound exceeded"
        );
        ensure!(
            self.sections.iter().map(|s| s.text.len()).sum::<usize>() <= 128 * 1024 * 1024,
            "document text bound exceeded"
        );
        for section in &self.sections {
            ensure!(!section.text.trim().is_empty(), "empty section");
            ensure!(section.text.len() <= 1024 * 1024, "section bound exceeded");
            ensure!(
                !section.source.record_id.is_empty() && section.source.record_index >= 0,
                "section provenance required"
            );
            let uri = reqwest::Url::parse(&section.source.uri)?;
            ensure!(
                matches!(uri.scheme(), "https" | "s3")
                    && uri.username().is_empty()
                    && uri.password().is_none(),
                "safe citation required"
            );
        }
        Ok(())
    }

    pub fn content_hash(&self) -> Result<String> {
        // Extraction clocks/object locations do not change the embedding input.
        let sections: Vec<_> = self
            .sections
            .iter()
            .map(|section| {
                (
                    &section.text,
                    &section.author,
                    section.occurred_at,
                    &section.source.uri,
                    &section.source.record_id,
                )
            })
            .collect();
        let bytes = serde_json::to_vec(&(TRANSFORM_VERSION, &self.title, &self.context, sections))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    pub fn plaintext(&self) -> String {
        self.sections
            .iter()
            .map(Section::render)
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl Section {
    pub fn render(&self) -> String {
        let header = match (&self.author, self.occurred_at) {
            (Some(author), Some(time)) => format!("{} — {}", author, time.to_rfc3339()),
            (Some(author), None) => author.clone(),
            (None, Some(time)) => time.to_rfc3339(),
            (None, None) => String::new(),
        };
        if header.is_empty() {
            self.text.clone()
        } else {
            format!("{header}\n{}", self.text)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceRecord {
    pub organization_id: String,
    pub connection_id: String,
    pub stream: String,
    pub record_id: String,
    pub extracted_at: DateTime<Utc>,
    pub deleted: bool,
    pub data: Value,
    pub source: SourceRef,
}

impl SourceRecord {
    pub fn text(&self, field: &str) -> Option<&str> {
        self.data
            .get(field)
            .and_then(Value::as_str)
            .filter(|text| !text.trim().is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> CanonicalDocument {
        CanonicalDocument {
            id: "channel/thread".into(),
            organization_id: "org_test".into(),
            connection_id: "connection".into(),
            kind: DocumentKind::SlackConversation,
            title: "Alice in #marketing".into(),
            context: BTreeMap::new(),
            modified_at: DateTime::from_timestamp(1700000000, 0).unwrap(),
            sections: vec![Section {
                text: "Budget approved".into(),
                author: Some("Alice".into()),
                occurred_at: Some(DateTime::from_timestamp(1700000000, 0).unwrap()),
                source: SourceRef {
                    stream: "channel_messages".into(),
                    record_id: "message".into(),
                    uri: "https://example.com/message".into(),
                    object_key: "object".into(),
                    record_index: 0,
                },
            }],
        }
    }
    #[test]
    fn extraction_replay_does_not_change_content_identity() {
        let first = fixture();
        let mut replay = first.clone();
        replay.modified_at += chrono::Duration::hours(6);
        replay.sections[0].source.object_key = "replayed-object".into();
        replay.sections[0].source.record_index = 12;
        assert_eq!(
            first.content_hash().unwrap(),
            replay.content_hash().unwrap()
        );
        replay.sections[0].text = "Budget revised".into();
        assert_ne!(
            first.content_hash().unwrap(),
            replay.content_hash().unwrap()
        );
    }
    #[test]
    fn empty_sections_and_credential_bearing_citations_are_invalid() {
        let mut doc = fixture();
        assert!(doc.validate().is_ok());
        doc.sections[0].text.clear();
        assert!(doc.validate().is_err());
        doc.sections[0].text = "Valid text".into();
        doc.sections[0].source.uri = "https://user:password@example.com/message".into();
        assert!(doc.validate().is_err());
    }
}
