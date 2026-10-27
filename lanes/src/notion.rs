//! Deterministic Notion page projection. Unsupported or unlinked records remain
//! explicit coverage deferrals; they are never invented text for the extractor.
use crate::document::{CanonicalDocument, DocumentKind, Section, SourceRecord};
use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub struct Projection {
    pub documents: Vec<CanonicalDocument>,
    pub deferred: Vec<String>,
}

fn rich(value: &Value) -> String {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|v| {
            v.get("plain_text")
                .and_then(Value::as_str)
                .or_else(|| v.pointer("/text/content").and_then(Value::as_str))
        })
        .collect::<String>()
}
fn title(data: &Value) -> String {
    if data.get("title").is_some() {
        return rich(&data["title"]);
    }
    data.get("properties")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|p| p.values())
        .find(|v| v.get("type").and_then(Value::as_str) == Some("title"))
        .map(|v| rich(&v["title"]))
        .unwrap_or_default()
}
fn text(record: &SourceRecord) -> Option<String> {
    let data = &record.data;
    match record.stream.as_str() {
        "pages" | "data_sources" => {
            let mut lines = vec![title(data)];
            if let Some(properties) = data.get("properties").and_then(Value::as_object) {
                for (name, property) in properties {
                    let kind = property.get("type").and_then(Value::as_str).unwrap_or("");
                    let value = &property[kind];
                    let rendered = match kind {
                        "rich_text" => rich(value),
                        "select" | "status" => value
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .into(),
                        "multi_select" => value
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter_map(|v| v.get("name").and_then(Value::as_str))
                            .collect::<Vec<_>>()
                            .join(", "),
                        "date" => [value.get("start"), value.get("end")]
                            .into_iter()
                            .flatten()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join(" – "),
                        "number" if value.is_number() => value.to_string(),
                        "checkbox" if value.is_boolean() => value.to_string(),
                        _ => String::new(),
                    };
                    if !rendered.is_empty() {
                        lines.push(format!("{name}: {rendered}"));
                    }
                }
            }
            if data.get("description").is_some() {
                lines.push(rich(&data["description"]));
            }
            Some(
                lines
                    .into_iter()
                    .filter(|l| !l.trim().is_empty())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        }
        "comments" => Some(rich(&data["rich_text"])),
        "blocks" => {
            let kind = data.get("type")?.as_str()?;
            let body = &data[kind];
            match kind {
                "paragraph" | "heading_1" | "heading_2" | "heading_3" | "bulleted_list_item"
                | "numbered_list_item" | "quote" | "callout" | "code" | "toggle" => {
                    Some(rich(&body["rich_text"]))
                }
                "to_do" => Some(format!(
                    "[{}] {}",
                    if body.get("checked").and_then(Value::as_bool) == Some(true) {
                        "x"
                    } else {
                        " "
                    },
                    rich(&body["rich_text"])
                )),
                "table_row" => Some(
                    body.get("cells")?
                        .as_array()?
                        .iter()
                        .map(rich)
                        .collect::<Vec<_>>()
                        .join("\t"),
                ),
                "child_page" | "child_database" => {
                    body.get("title").and_then(Value::as_str).map(str::to_owned)
                }
                "equation" => body
                    .get("expression")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                // These are structural containers; their children are traversed separately.
                "column_list" | "column" | "table" | "divider" | "table_of_contents"
                | "breadcrumb" => Some(String::new()),
                _ => None,
            }
        }
        _ => None,
    }
}
fn parent(data: &Value) -> Option<&str> {
    let p = data.get("parent")?;
    let kind = p.get("type")?.as_str()?;
    p.get(kind)?.as_str()
}
fn modified(record: &SourceRecord) -> Result<DateTime<Utc>> {
    match record.data.get("last_edited_time").and_then(Value::as_str) {
        Some(value) => Ok(DateTime::parse_from_rfc3339(value)?.with_timezone(&Utc)),
        None => Ok(record.extracted_at),
    }
}
fn root<'a>(id: &'a str, records: &BTreeMap<&str, &'a SourceRecord>) -> Option<&'a str> {
    let mut current = id;
    let mut seen = BTreeSet::new();
    for _ in 0..100 {
        if !seen.insert(current) {
            return None;
        }
        let record = records.get(current)?;
        if matches!(record.stream.as_str(), "pages" | "data_sources") {
            return Some(current);
        }
        current = parent(&record.data)?;
    }
    None
}

pub fn project(records: &[SourceRecord]) -> Result<Projection> {
    let mut result = Projection {
        documents: Vec::new(),
        deferred: Vec::new(),
    };
    let Some(first) = records.first() else {
        return Ok(result);
    };
    let mut by_id: BTreeMap<&str, &SourceRecord> = BTreeMap::new();
    for record in records {
        ensure!(
            record.organization_id == first.organization_id
                && record.connection_id == first.connection_id,
            "Notion projection cannot cross tenant/source scope"
        );
        if record.deleted
            || record.data.get("archived").and_then(Value::as_bool) == Some(true)
            || record.data.get("in_trash").and_then(Value::as_bool) == Some(true)
        {
            continue;
        }
        let id = record
            .data
            .get("id")
            .and_then(Value::as_str)
            .context("Notion source identity required")?;
        ensure!(
            !id.is_empty()
                && id.len() <= 100
                && id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-'),
            "invalid Notion identifier"
        );
        if let Some(prior) = by_id.get(id) {
            let page_block = |r: &SourceRecord| {
                r.stream == "blocks"
                    && r.data.get("type").and_then(Value::as_str) == Some("child_page")
            };
            // Notion pages are also blocks: the child_page stub can share the
            // page UUID. Prefer the full page instead of merging unrelated IDs.
            if prior.stream == "pages" && page_block(record) {
                continue;
            }
            ensure!(
                record.stream == "pages" && page_block(prior),
                "duplicate Notion source identity"
            );
        }
        by_id.insert(id, record);
    }
    let mut groups: BTreeMap<&str, Vec<&SourceRecord>> = BTreeMap::new();
    for (id, record) in &by_id {
        // User metadata is not free-standing business evidence.
        if record.stream == "users" {
            continue;
        }
        if let Some(page) = root(id, &by_id) {
            groups.entry(page).or_default().push(record);
        } else {
            result.deferred.push(record.record_id.clone());
        }
    }
    for (id, mut group) in groups {
        let page = by_id[id];
        let page_url = page
            .data
            .get("url")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("https://www.notion.so/{}", id.replace('-', "")));
        let url = reqwest::Url::parse(&page_url)?;
        ensure!(
            url.scheme() == "https"
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none(),
            "unsafe Notion page citation"
        );
        // Parent/sequence paths preserve child ordering, independent of S3 file order.
        let path = |record: &SourceRecord| {
            let mut steps = Vec::new();
            let mut current = record;
            for _ in 0..100 {
                if current.data.get("id").and_then(Value::as_str) == Some(id) {
                    break;
                }
                steps.push((
                    current
                        .data
                        .pointer("/parent/sequence_number")
                        .and_then(Value::as_u64)
                        .unwrap_or(u64::MAX),
                    current.data["id"].as_str().unwrap_or("").to_owned(),
                ));
                let Some(next) = parent(&current.data).and_then(|p| by_id.get(p)) else {
                    break;
                };
                current = next;
            }
            steps.reverse();
            steps
        };
        group.sort_by_key(|record| path(record));
        let mut sections = Vec::new();
        let mut time = modified(page)?;
        for record in group {
            time = time.max(modified(record)?);
            let Some(text) = text(record) else {
                result.deferred.push(record.record_id.clone());
                continue;
            };
            if text.trim().is_empty() {
                continue;
            }
            let mut source = record.source.clone();
            source.uri = if record.stream == "blocks" {
                format!(
                    "{page_url}#{}",
                    record.data["id"]
                        .as_str()
                        .context("Notion block identity required")?
                        .replace('-', "")
                )
            } else {
                page_url.clone()
            };
            sections.push(Section {
                text,
                author: None,
                occurred_at: None,
                source,
            });
        }
        if sections.is_empty() {
            continue;
        }
        let document = CanonicalDocument {
            id: format!("notion/{id}"),
            organization_id: first.organization_id.clone(),
            connection_id: first.connection_id.clone(),
            kind: DocumentKind::GenericText,
            title: title(&page.data),
            sections,
            context: BTreeMap::from([("provider".into(), "notion".into())]),
            modified_at: time,
        };
        document.validate()?;
        result.documents.push(document);
    }
    result.deferred.sort();
    result.deferred.dedup();
    Ok(result)
}
