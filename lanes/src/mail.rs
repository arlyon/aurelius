use crate::{
    document::{CanonicalDocument, DocumentKind, Section, SourceRecord},
    transform::{DeferredReason, TransformResult},
    xberg::XbergClient,
};
use anyhow::{Context, Result, ensure};
use base64::Engine;
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
struct Thread {
    id: String,
    messages: Vec<Message>,
}
#[derive(Deserialize)]
struct Message {
    id: String,
    #[serde(rename = "threadId")]
    thread: String,
    #[serde(rename = "internalDate")]
    date: String,
    payload: Part,
}
#[derive(Deserialize)]
struct Part {
    #[serde(rename = "mimeType")]
    mime: String,
    #[serde(default)]
    filename: String,
    #[serde(default)]
    headers: Vec<Header>,
    body: Option<Body>,
    #[serde(default)]
    parts: Vec<Part>,
}
#[derive(Deserialize)]
struct Header {
    name: String,
    value: String,
}
#[derive(Deserialize)]
struct Body {
    data: Option<String>,
    #[serde(rename = "attachmentId")]
    attachment: Option<String>,
}
fn field<'a>(headers: &'a [Header], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case(name))
        .map(|h| h.value.as_str())
}
fn collect<'a>(
    part: &'a Part,
    depth: usize,
    parts: &mut usize,
    attachments: &mut usize,
    output: &mut Vec<&'a Part>,
) -> Result<()> {
    *parts += 1;
    ensure!(
        depth <= 16 && *parts <= 1000,
        "mail MIME nesting exceeds bound"
    );
    if !part.filename.is_empty()
        || part
            .body
            .as_ref()
            .and_then(|b| b.attachment.as_ref())
            .is_some()
    {
        if part
            .body
            .as_ref()
            .and_then(|b| b.data.as_ref())
            .is_some_and(|data| !data.is_empty())
        {
            output.push(part);
        } else {
            *attachments += 1;
        }
        return Ok(());
    }
    if !part.parts.is_empty() {
        if part.mime.eq_ignore_ascii_case("multipart/alternative") {
            if let Some(child) = part
                .parts
                .iter()
                .find(|p| {
                    p.mime.eq_ignore_ascii_case("text/plain")
                        && p.body
                            .as_ref()
                            .and_then(|body| body.data.as_ref())
                            .is_some_and(|data| !data.is_empty())
                })
                .or_else(|| {
                    part.parts.iter().find(|p| {
                        p.mime.eq_ignore_ascii_case("text/html")
                            && p.body
                                .as_ref()
                                .and_then(|body| body.data.as_ref())
                                .is_some_and(|data| !data.is_empty())
                    })
                })
                .or_else(|| part.parts.iter().find(|p| !p.parts.is_empty()))
            {
                collect(child, depth + 1, parts, attachments, output)?;
            }
        } else {
            for child in &part.parts {
                collect(child, depth + 1, parts, attachments, output)?;
            }
        }
    } else if matches!(
        part.mime.to_lowercase().as_str(),
        "text/plain" | "text/html"
    ) {
        output.push(part);
    }
    Ok(())
}
pub fn decode_body(encoded: &str) -> Result<Vec<u8>> {
    ensure!(
        encoded.len() <= 8 * 1024 * 1024
            && encoded
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'=')),
        "invalid base64url mail body"
    );
    let trimmed = encoded.trim_end_matches('=');
    ensure!(encoded.len() - trimmed.len() <= 2, "invalid base64 padding");
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(trimmed)?;
    ensure!(
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&bytes) == trimmed,
        "noncanonical base64url body"
    );
    Ok(bytes)
}
fn decode_text(part: &Part, bytes: &[u8]) -> Result<String> {
    let content_type = field(&part.headers, "content-type").unwrap_or(&part.mime);
    let charset = content_type
        .split(';')
        .skip(1)
        .find_map(|p| {
            let (key, value) = p.trim().split_once('=')?;
            key.trim()
                .eq_ignore_ascii_case("charset")
                .then(|| value.trim().trim_matches(['\"', '\'']))
        })
        .unwrap_or("utf-8");
    let encoding =
        encoding_rs::Encoding::for_label(charset.as_bytes()).context("unsupported mail charset")?;
    let (text, _, errors) = encoding.decode(bytes);
    ensure!(!errors, "invalid mail text encoding");
    Ok(text.into_owned())
}

pub async fn project(record: &SourceRecord, xberg: &XbergClient) -> Result<TransformResult> {
    if record.deleted {
        return Ok(TransformResult::Ignored);
    }
    let document = conversation(record, xberg).await?;
    if document.sections.is_empty() {
        return Ok(TransformResult::Deferred(
            DeferredReason::NoReadableMailContent,
        ));
    }
    Ok(TransformResult::Ready(document))
}

pub async fn conversation(record: &SourceRecord, xberg: &XbergClient) -> Result<CanonicalDocument> {
    let thread: Thread =
        serde_json::from_value(record.data.clone()).context("invalid mail thread")?;
    ensure!(
        !thread.id.is_empty()
            && thread.id.len() <= 200
            && !thread.messages.is_empty()
            && thread.messages.len() <= 1000,
        "invalid mail thread size"
    );
    let mut messages = thread.messages;
    messages.sort_by(|a, b| {
        a.date
            .parse::<i64>()
            .unwrap_or(i64::MIN)
            .cmp(&b.date.parse::<i64>().unwrap_or(i64::MIN))
            .then(a.id.cmp(&b.id))
    });
    let title = field(&messages[0].payload.headers, "subject")
        .unwrap_or("Email conversation")
        .to_owned();
    let mut seen = BTreeSet::new();
    let mut sections = Vec::new();
    let mut total = 0;
    let mut attachments = 0;
    let mut parts = 0;
    for message in messages {
        ensure!(
            message.thread == thread.id
                && !message.id.is_empty()
                && message.id.len() <= 200
                && message
                    .id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
                && seen.insert(message.id.clone()),
            "ambiguous mail thread membership"
        );
        let authored = DateTime::<Utc>::from_timestamp_millis(message.date.parse()?)
            .context("invalid mail source date")?;
        let mut leaves = Vec::new();
        collect(
            &message.payload,
            0,
            &mut parts,
            &mut attachments,
            &mut leaves,
        )?;
        let mut texts = Vec::new();
        for part in leaves {
            let Some(data) = part.body.as_ref().and_then(|b| b.data.as_deref()) else {
                continue;
            };
            let bytes = decode_body(data)?;
            total += bytes.len();
            ensure!(total <= 8 * 1024 * 1024, "decoded mail exceeds bound");
            let text = if !part.filename.is_empty()
                || !matches!(
                    part.mime.to_lowercase().as_str(),
                    "text/plain" | "text/html"
                ) {
                xberg
                    .extract(
                        bytes,
                        if part.filename.is_empty() {
                            "attachment"
                        } else {
                            &part.filename
                        },
                        &part.mime,
                    )
                    .await?
                    .content
            } else {
                let text = decode_text(part, &bytes)?;
                if part.mime.eq_ignore_ascii_case("text/html") && !text.trim().is_empty() {
                    xberg
                        .extract(text.into_bytes(), "message.html", "text/html")
                        .await?
                        .content
                } else {
                    text
                }
            };
            if !text.trim().is_empty() {
                texts.push(text);
            }
        }
        let text = texts.join("\n\n");
        if text.trim().is_empty() {
            continue;
        }
        ensure!(text.len() <= 1024 * 1024, "mail section exceeds bound");
        let mut source = record.source.clone();
        source.record_id = message.id.clone();
        source.uri = format!("https://mail.google.com/mail/u/0/#all/{}", message.id);
        sections.push(Section {
            text,
            author: field(&message.payload.headers, "from").map(str::to_owned),
            occurred_at: Some(authored),
            source,
        });
    }
    let mut context = BTreeMap::new();
    context.insert("unavailable_attachments".into(), attachments.to_string());
    let document = CanonicalDocument {
        id: thread.id,
        organization_id: record.organization_id.clone(),
        connection_id: record.connection_id.clone(),
        kind: DocumentKind::MailThread,
        title,
        sections,
        context,
        modified_at: record.extracted_at,
    };
    document.validate()?;
    Ok(document)
}
