use crate::{
    document::{CanonicalDocument, SourceRef},
    embedding::Embedder,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Citation {
    pub byte_start: usize,
    pub byte_end: usize,
    pub source: SourceRef,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct IndexedChunk {
    pub id: String,
    pub text: String,
    pub context_header: String,
    pub citations: Vec<Citation>,
    pub tokens: usize,
    pub embedding: Vec<f32>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct IndexedDocument {
    pub document: CanonicalDocument,
    pub profile: String,
    pub content_hash: String,
    pub chunks: Vec<IndexedChunk>,
}

pub async fn index_document(
    document: CanonicalDocument,
    profile: &str,
    embedder: &Embedder,
) -> Result<IndexedDocument> {
    document.validate()?;
    let content_hash = document.content_hash()?;
    let mut header = document.title.clone();
    for (k, v) in &document.context {
        header.push_str(&format!("\n{k}: {v}"));
    }
    if header.len() > 2000 || embedder.count(&header).await? > embedder.config.token_limit / 4 {
        header.clear();
    }
    let mut chunks = Vec::new();
    let mut text = String::new();
    let mut citations = Vec::new();
    for section in &document.sections {
        let rendered = section.render();
        let mut start = 0;
        while start < rendered.len() {
            let mut end = rendered.len().min(start + 12000);
            while !rendered.is_char_boundary(end) {
                end -= 1;
            }
            while end > start {
                let piece = &rendered[start..end];
                let candidate = if text.is_empty() {
                    piece.to_owned()
                } else {
                    format!("{text}\n\n{piece}")
                };
                if candidate.len() <= 16000
                    && embedder.count(&input(&header, &candidate)).await?
                        <= embedder.config.token_limit
                {
                    let offset = if text.is_empty() { 0 } else { text.len() + 2 };
                    text = candidate;
                    citations.push(Citation {
                        byte_start: offset,
                        byte_end: text.len(),
                        source: section.source.clone(),
                    });
                    start = end;
                    break;
                }
                if !text.is_empty() {
                    flush(
                        &mut chunks,
                        &mut text,
                        &mut citations,
                        &header,
                        &content_hash,
                        profile,
                        embedder,
                    )
                    .await?;
                    continue;
                }
                let chars = rendered[start..end].chars().count();
                ensure!(chars > 1, "single character exceeds token budget");
                end = start
                    + rendered[start..end]
                        .char_indices()
                        .nth(chars / 2)
                        .map(|(n, _)| n)
                        .unwrap_or(0);
            }
            ensure!(end > 0, "chunk splitting made no progress");
        }
    }
    flush(
        &mut chunks,
        &mut text,
        &mut citations,
        &header,
        &content_hash,
        profile,
        embedder,
    )
    .await?;
    Ok(IndexedDocument {
        document,
        profile: profile.into(),
        content_hash,
        chunks,
    })
}
fn input(header: &str, text: &str) -> String {
    if header.is_empty() {
        text.into()
    } else {
        format!("{header}\n\n{text}")
    }
}
async fn flush(
    chunks: &mut Vec<IndexedChunk>,
    text: &mut String,
    citations: &mut Vec<Citation>,
    header: &str,
    hash: &str,
    profile: &str,
    embedder: &Embedder,
) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    let formatted = input(header, text);
    let tokens = embedder.count(&formatted).await?;
    let vector = embedder.vector(&formatted).await?;
    let id = blake3::hash(&serde_json::to_vec(&(
        hash,
        profile,
        chunks.len(),
        text.as_str(),
    ))?)
    .to_hex()
    .to_string();
    chunks.push(IndexedChunk {
        id,
        text: std::mem::take(text),
        context_header: header.into(),
        citations: std::mem::take(citations),
        tokens,
        embedding: vector,
    });
    Ok(())
}
