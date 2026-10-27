use crate::document::{CanonicalDocument, DocumentKind, Section, SourceRecord};
use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Utc};
use regex::{Captures, Regex};
use std::{
    collections::{BTreeMap, HashMap},
    sync::LazyLock,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeferredReason {
    MissingThreadRoot,
    MissingTask,
}

#[derive(Clone, Debug)]
pub enum TransformResult {
    Ready(CanonicalDocument),
    Ignored,
    Deferred(DeferredReason),
}

#[derive(Default)]
pub struct SlackNames {
    pub users: HashMap<String, String>,
    pub channels: HashMap<String, String>,
}

static MENTIONS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<@([A-Z0-9]+)>").unwrap());
static CHANNELS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<#([A-Z0-9]+)\|([^>]+)>").unwrap());
static LINKS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<(https?://[^>|]+)(?:\|([^>]+))?>").unwrap());

pub fn clean_slack(text: &str, names: &SlackNames) -> String {
    let text = MENTIONS.replace_all(text, |capture: &Captures| {
        format!(
            "@{}",
            names
                .users
                .get(&capture[1])
                .map(String::as_str)
                .unwrap_or(&capture[1])
        )
    });
    let text = CHANNELS.replace_all(&text, |capture: &Captures| format!("#{}", &capture[2]));
    let text = LINKS.replace_all(&text, |capture: &Captures| {
        capture
            .get(2)
            .map_or_else(|| capture[1].to_owned(), |label| label.as_str().to_owned())
    });
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("<!channel>", "@channel")
        .replace("<!here>", "@here")
        .replace("<!everyone>", "@everyone")
        .trim()
        .to_owned()
}

fn scope(records: &[SourceRecord]) -> Result<(&str, &str)> {
    let first = records.first().context("source records required")?;
    ensure!(
        records
            .iter()
            .all(|record| record.organization_id == first.organization_id
                && record.connection_id == first.connection_id),
        "mixed source scope rejected"
    );
    Ok((&first.organization_id, &first.connection_id))
}

pub fn timestamp(value: &str) -> Result<DateTime<Utc>> {
    if let Ok(time) = DateTime::parse_from_rfc3339(value) {
        return Ok(time.with_timezone(&Utc));
    }
    let (seconds, fraction) = value.split_once('.').unwrap_or((value, ""));
    ensure!(
        fraction.len() <= 9 && fraction.chars().all(|c| c.is_ascii_digit()),
        "invalid source timestamp"
    );
    DateTime::from_timestamp(seconds.parse()?, format!("{fraction:0<9}").parse()?)
        .context("source timestamp outside range")
}
fn slack_time(ts: &str) -> Result<DateTime<Utc>> {
    timestamp(ts)
}

fn keep_slack(record: &SourceRecord) -> bool {
    !matches!(
        record.text("subtype"),
        Some(
            "channel_join"
                | "channel_leave"
                | "channel_archive"
                | "channel_unarchive"
                | "pinned_item"
                | "unpinned_item"
        )
    )
}

pub fn slack_conversation(
    records: &[SourceRecord],
    channel: &str,
    root_ts: &str,
    names: &SlackNames,
) -> Result<TransformResult> {
    let (organization, connection) = scope(records)?;
    let mut messages: BTreeMap<DateTime<Utc>, &SourceRecord> = BTreeMap::new();
    for record in records {
        if !matches!(record.stream.as_str(), "channel_messages" | "threads")
            || record.text("channel_id") != Some(channel)
        {
            continue;
        }
        let ts = record
            .text("ts")
            .context("Slack message timestamp required")?;
        let thread = record.text("thread_ts").unwrap_or(ts);
        if thread != root_ts && ts != root_ts {
            continue;
        }
        let time = slack_time(ts)?;
        match messages.get(&time) {
            Some(previous) if previous.extracted_at > record.extracted_at => {}
            Some(previous) if previous.extracted_at == record.extracted_at => {
                ensure!(
                    previous.text("text") == record.text("text"),
                    "conflicting Slack revision"
                );
            }
            _ => {
                messages.insert(time, record);
            }
        }
    }
    let root = slack_time(root_ts)?;
    if !messages.contains_key(&root) {
        return Ok(TransformResult::Deferred(DeferredReason::MissingThreadRoot));
    }
    if messages[&root].deleted {
        return Ok(TransformResult::Ignored);
    }
    let modified = messages
        .values()
        .map(|record| record.extracted_at)
        .max()
        .unwrap();
    let mut sections = Vec::new();
    for (time, record) in messages {
        if record.deleted || !keep_slack(record) {
            continue;
        }
        let text = clean_slack(record.text("text").unwrap_or_default(), names);
        if text.is_empty() {
            continue;
        }
        let author = record.text("user").map(|id| {
            names
                .users
                .get(id)
                .cloned()
                .unwrap_or_else(|| format!("Slack user {id}"))
        });
        sections.push(Section {
            text,
            author,
            occurred_at: Some(time),
            source: record.source.clone(),
        });
    }
    if sections.is_empty() {
        return Ok(TransformResult::Ignored);
    }
    let channel_name = names
        .channels
        .get(channel)
        .map(String::as_str)
        .unwrap_or(channel);
    let title = format!("Conversation in #{channel_name} — {}", root.to_rfc3339());
    let doc = CanonicalDocument {
        id: serde_json::to_string(&(channel, root_ts))?,
        organization_id: organization.into(),
        connection_id: connection.into(),
        kind: DocumentKind::SlackConversation,
        title,
        sections,
        context: BTreeMap::from([
            ("channel".into(), channel_name.into()),
            ("thread_started_at".into(), root.to_rfc3339()),
        ]),
        modified_at: modified,
    };
    doc.validate()?;
    Ok(TransformResult::Ready(doc))
}

// Shared with verified raw-row metadata extraction: boundaries must match the
// normalized header byte-for-byte, without changing existing embedding input.
pub(crate) fn asana_header(data: &serde_json::Value) -> String {
    let title = data
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("Untitled Asana task");
    let mut text = format!("Task: {title}");
    for (pointer, label) in [
        ("/assignee/name", "Assignee"),
        ("/due_on", "Due date"),
        ("/due_at", "Due time"),
    ] {
        if let Some(value) = data
            .pointer(pointer)
            .and_then(serde_json::Value::as_str)
            .filter(|s| !s.is_empty())
        {
            text.push_str(&format!("\n{label}: {value}"));
        }
    }
    if let Some(projects) = data.get("projects").and_then(serde_json::Value::as_array) {
        let names: Vec<_> = projects
            .iter()
            .filter_map(|p| p.get("name").and_then(serde_json::Value::as_str))
            .collect();
        if !names.is_empty() {
            text.push_str(&format!("\nProjects: {}", names.join(", ")));
        }
    }
    if let Some(done) = data.get("completed").and_then(serde_json::Value::as_bool) {
        text.push_str(if done {
            "\nStatus: completed"
        } else {
            "\nStatus: open"
        });
    }
    text
}
pub fn asana_task(records: &[SourceRecord], task_gid: &str) -> Result<TransformResult> {
    let (organization, connection) = scope(records)?;
    let task = records
        .iter()
        .filter(|record| record.stream == "tasks" && record.text("gid") == Some(task_gid))
        .max_by_key(|record| record.extracted_at);
    let Some(task) = task else {
        return Ok(TransformResult::Deferred(DeferredReason::MissingTask));
    };
    if task.deleted {
        return Ok(TransformResult::Ignored);
    }
    let title = task
        .text("name")
        .unwrap_or("Untitled Asana task")
        .to_owned();
    let notes = task.text("notes").unwrap_or_default();
    let mut text = asana_header(&task.data);
    if !notes.is_empty() {
        text.push_str("\n\n");
        text.push_str(notes);
    }
    let mut sections = vec![Section {
        text,
        author: None,
        occurred_at: None,
        source: task.source.clone(),
    }];
    let mut comments: BTreeMap<String, &SourceRecord> = BTreeMap::new();
    for record in records.iter().filter(|record| {
        !record.deleted
            && record.stream == "stories"
            && record.text("parent_task_gid") == Some(task_gid)
    }) {
        let gid = record
            .text("gid")
            .context("Asana story identity required")?;
        if comments
            .get(gid)
            .is_none_or(|previous| previous.extracted_at <= record.extracted_at)
        {
            comments.insert(gid.to_owned(), record);
        }
    }
    let modified_at = comments
        .values()
        .map(|record| record.extracted_at)
        .chain([task.extracted_at])
        .max()
        .unwrap();
    let mut ordered: Vec<_> = comments.values().copied().collect();
    ordered.sort_by_key(|record| {
        (
            record.text("created_at").unwrap_or_default(),
            record.record_id.as_str(),
        )
    });
    for record in ordered {
        // System actions are represented by task state; comments carry conversational evidence.
        if record.text("type") != Some("comment")
            && record.text("resource_subtype") != Some("comment_added")
        {
            continue;
        }
        let Some(text) = record.text("text") else {
            continue;
        };
        let author = record
            .data
            .pointer("/created_by/name")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let occurred_at = record
            .text("created_at")
            .map(DateTime::parse_from_rfc3339)
            .transpose()?
            .map(|time| time.with_timezone(&Utc));
        sections.push(Section {
            text: text.into(),
            author,
            occurred_at,
            source: record.source.clone(),
        });
    }
    let document = CanonicalDocument {
        id: task_gid.into(),
        organization_id: organization.into(),
        connection_id: connection.into(),
        kind: DocumentKind::AsanaTask,
        title,
        sections,
        context: BTreeMap::new(),
        modified_at,
    };
    document.validate()?;
    Ok(TransformResult::Ready(document))
}

pub fn generic_text(
    record: &SourceRecord,
    title: &str,
    fields: &[String],
) -> Result<TransformResult> {
    let text = fields
        .iter()
        .filter_map(|pointer| {
            record
                .data
                .pointer(pointer)
                .and_then(serde_json::Value::as_str)
        })
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if text.trim().is_empty() {
        return Ok(TransformResult::Ignored);
    }
    let document = CanonicalDocument {
        id: record.record_id.clone(),
        organization_id: record.organization_id.clone(),
        connection_id: record.connection_id.clone(),
        kind: DocumentKind::GenericText,
        title: title.into(),
        sections: vec![Section {
            text,
            author: None,
            occurred_at: None,
            source: record.source.clone(),
        }],
        context: BTreeMap::new(),
        modified_at: record.extracted_at,
    };
    document.validate()?;
    Ok(TransformResult::Ready(document))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::SourceRef;
    use serde_json::json;
    fn record(stream: &str, data: serde_json::Value) -> SourceRecord {
        let id = data
            .get("ts")
            .or_else(|| data.get("gid"))
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        SourceRecord {
            organization_id: "org_test".into(),
            connection_id: "conn".into(),
            stream: stream.into(),
            record_id: id.clone(),
            extracted_at: DateTime::from_timestamp(1800000000, 0).unwrap(),
            deleted: false,
            data,
            source: SourceRef {
                stream: stream.into(),
                record_id: id.clone(),
                uri: format!("https://example.com/{id}"),
                object_key: "object".into(),
                record_index: 0,
            },
        }
    }
    fn ready(result: TransformResult) -> CanonicalDocument {
        match result {
            TransformResult::Ready(document) => document,
            _ => panic!("expected document"),
        }
    }
    #[test]
    fn shuffled_streams_form_one_conversation_with_root_and_reply_citations() {
        let root = record(
            "channel_messages",
            json!({"channel_id":"C1","ts":"1700000000.000001","text":"What is the budget?","user":"U1"}),
        );
        let duplicate = record("threads", root.data.clone());
        let reply = record(
            "threads",
            json!({"channel_id":"C1","ts":"1700000001.000001","thread_ts":"1700000000.000001","text":"Approved: $42000","user":"U2"}),
        );
        let names = SlackNames {
            users: HashMap::from([("U1".into(), "Alice".into()), ("U2".into(), "Bob".into())]),
            channels: HashMap::from([("C1".into(), "marketing".into())]),
        };
        let first = ready(
            slack_conversation(
                &[reply.clone(), duplicate, root.clone()],
                "C1",
                "1700000000.000001",
                &names,
            )
            .unwrap(),
        );
        let second =
            ready(slack_conversation(&[root, reply], "C1", "1700000000.000001", &names).unwrap());
        assert_eq!(first.sections.len(), 2);
        assert_eq!(
            first.content_hash().unwrap(),
            second.content_hash().unwrap()
        );
        assert_eq!(first.sections[0].author.as_deref(), Some("Alice"));
        assert_ne!(first.sections[0].source.uri, first.sections[1].source.uri);
        assert!(first.title.contains("marketing"));
    }
    #[test]
    fn missing_parent_is_deferred_and_cross_tenant_input_rejected() {
        let mut reply = record(
            "threads",
            json!({"channel_id":"C1","ts":"1700000001.0","thread_ts":"1700000000.0","text":"yes"}),
        );
        assert!(matches!(
            slack_conversation(
                &[reply.clone()],
                "C1",
                "1700000000.0",
                &SlackNames::default()
            )
            .unwrap(),
            TransformResult::Deferred(DeferredReason::MissingThreadRoot)
        ));
        let root = record(
            "channel_messages",
            json!({"channel_id":"C1","ts":"1700000000.0","text":"Question"}),
        );
        reply.organization_id = "org_other".into();
        assert!(
            slack_conversation(&[root, reply], "C1", "1700000000.0", &SlackNames::default())
                .is_err()
        );
    }
    #[test]
    fn cleans_slack_and_preserves_meaningful_bot_content() {
        let names = SlackNames {
            users: HashMap::from([("U1".into(), "Alice".into())]),
            ..Default::default()
        };
        assert_eq!(
            clean_slack(
                "<@U1> review <https://example.com|the plan> in <#C1|marketing> &amp; reply",
                &names
            ),
            "@Alice review the plan in #marketing & reply"
        );
        let root = record(
            "channel_messages",
            json!({"channel_id":"C1","ts":"1700000000.0","text":"Budget anomaly: overspend 30%","bot_id":"B1"}),
        );
        assert_eq!(
            ready(slack_conversation(&[root], "C1", "1700000000.0", &names).unwrap())
                .sections
                .len(),
            1
        );
    }
    #[test]
    fn task_comments_require_exported_relationship_and_exclude_system_chatter() {
        let task = record(
            "tasks",
            json!({"gid":"123","name":"Campaign launch","notes":"Plan","completed":false}),
        );
        let comment = record(
            "stories",
            json!({"gid":"10","parent_task_gid":"123","type":"comment","text":"Budget approved","created_at":"2026-01-01T00:00:00Z","created_by":{"name":"Alice"}}),
        );
        let system = record(
            "stories",
            json!({"gid":"11","parent_task_gid":"123","type":"system","text":"Assigned to Bob"}),
        );
        let unrelated = record(
            "stories",
            json!({"gid":"12","type":"comment","text":"Unlinked"}),
        );
        let doc = ready(asana_task(&[system, unrelated, comment, task], "123").unwrap());
        assert_eq!(doc.sections.len(), 2);
        assert!(doc.plaintext().contains("Budget approved"));
        assert!(!doc.plaintext().contains("Unlinked"));
        assert!(!doc.plaintext().contains("Assigned to Bob"));
    }
    #[test]
    fn fallback_only_uses_explicit_text_not_json_or_credentials() {
        let raw = record(
            "unknown",
            json!({"gid":"123","body":"Useful document","token":"secret","nested":{"key":"secret"}}),
        );
        let doc = ready(generic_text(&raw, "Document", &["/body".into()]).unwrap());
        assert_eq!(doc.plaintext(), "Useful document");
        assert!(matches!(
            generic_text(&raw, "Document", &["/nested".into()]).unwrap(),
            TransformResult::Ignored
        ));
    }
}
