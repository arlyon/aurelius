use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const CONTRACT_VERSION: &str = "knowledge-v1";
pub const MAX_DOCUMENT_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DocumentKey {
    pub scope: String,
    pub source: String,
    pub revision: String,
}
impl DocumentKey {
    pub fn validate(&self) -> Result<()> {
        for value in [&self.scope, &self.source, &self.revision] {
            ensure!(
                !value.trim().is_empty()
                    && value.len() <= 512
                    && !value.chars().any(char::is_control),
                "invalid source identity"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    pub key: DocumentKey,
    pub text: String,
    // Identifies a coherent conversation/document supplied by the source adapter.
    pub context: Option<String>,
    #[serde(default)]
    pub metadata: SourceMetadata,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceMetadata {
    pub author: Option<String>,
    pub authored_at: Option<chrono::DateTime<chrono::Utc>>,
    pub modified_at: Option<chrono::DateTime<chrono::Utc>>,
    pub uri: Option<String>,
}
impl Document {
    pub fn validate(&self) -> Result<()> {
        self.key.validate()?;
        ensure!(
            !self.text.trim().is_empty() && self.text.len() <= MAX_DOCUMENT_BYTES,
            "invalid document text"
        );
        for value in self.metadata.author.iter().chain(self.metadata.uri.iter()) {
            ensure!(
                !value.trim().is_empty()
                    && value.len() <= 2048
                    && !value.chars().any(char::is_control),
                "invalid source metadata"
            );
        }
        if let Some(context) = &self.context {
            ensure!(
                !context.trim().is_empty() && context.len() <= 512,
                "invalid context identity"
            );
        }
        Ok(())
    }
    pub fn content_hash(&self) -> String {
        blake3::hash(self.text.as_bytes()).to_hex().to_string()
    }
}

// Offsets always refer to UTF-8 bytes in the original document, end-exclusive.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ByteRange {
    pub start: usize,
    pub end: usize,
}
impl ByteRange {
    // Python coreference offsets count Unicode code points, not bytes or UTF-16 units.
    pub fn from_codepoints(document: &Document, start: usize, end: usize) -> Result<Self> {
        document.validate()?;
        ensure!(start < end, "empty or inverted character range");
        let byte_at = |index| {
            document
                .text
                .char_indices()
                .map(|(offset, _)| offset)
                .chain(std::iter::once(document.text.len()))
                .nth(index)
                .ok_or_else(|| anyhow::anyhow!("character offset outside document"))
        };
        let range = Self {
            start: byte_at(start)?,
            end: byte_at(end)?,
        };
        range.text(document)?;
        Ok(range)
    }
    pub fn text<'a>(&self, document: &'a Document) -> Result<&'a str> {
        ensure!(self.start < self.end, "empty or inverted source range");
        document
            .text
            .get(self.start..self.end)
            .ok_or_else(|| anyhow::anyhow!("range is outside the document or splits UTF-8"))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceSpan {
    pub document: DocumentKey,
    pub content_hash: String,
    pub range: ByteRange,
    pub quote: String,
}
impl SourceSpan {
    pub fn validate(&self, document: &Document) -> Result<()> {
        document.validate()?;
        ensure!(
            self.document == document.key && self.content_hash == document.content_hash(),
            "source revision or content mismatch"
        );
        ensure!(
            self.range.text(document)? == self.quote,
            "source quote mismatch"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Chunk {
    pub document: Document,
    pub range: ByteRange,
}
impl Chunk {
    pub fn text(&self) -> Result<&str> {
        self.document.validate()?;
        let text = self.range.text(&self.document)?;
        ensure!(
            !text.trim().is_empty() && text.len() <= 16_000,
            "invalid extraction chunk"
        );
        Ok(text)
    }
    pub fn spans(&self, quote: &str) -> Result<Vec<SourceSpan>> {
        ensure!(!quote.is_empty(), "empty quote");
        let text = self.text()?;
        // Retain every occurrence; choosing the first would invent disambiguation.
        let offsets: Vec<_> = text
            .char_indices()
            .filter_map(|(offset, _)| text[offset..].starts_with(quote).then_some(offset))
            .take(17)
            .collect();
        ensure!(
            offsets.len() <= 16,
            "quote is too ambiguous; provide a larger exact quote"
        );
        let spans: Vec<_> = offsets
            .into_iter()
            .map(|offset| SourceSpan {
                document: self.document.key.clone(),
                content_hash: self.document.content_hash(),
                range: ByteRange {
                    start: self.range.start + offset,
                    end: self.range.start + offset + quote.len(),
                },
                quote: quote.into(),
            })
            .collect();
        ensure!(!spans.is_empty(), "fabricated source quote");
        Ok(spans)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Node {
    Mention {
        id: String,
        text: String,
        evidence: SourceSpan,
    },
    // A proposed identity is not an accepted merge or an assertion of sameness.
    EntityHypothesis {
        id: String,
        label: String,
        entity_type: String,
        mentions: Vec<String>,
    },
    Literal {
        id: String,
        value: serde_json::Value,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Argument {
    pub text: String,
    // Empty means the model's label has no verified verbatim mention. Never fabricate a span.
    pub mentions: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum PredicateMapping {
    Declared { predicate: String },
    Unmapped { predicate: String },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimEdge {
    pub id: String,
    pub subject: Argument,
    pub object: Argument,
    pub mapping: PredicateMapping,
    pub claim: crate::facts::Candidate,
    pub evidence: Vec<SourceSpan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateGraph {
    pub contract_version: String,
    pub document: DocumentKey,
    pub content_hash: String,
    pub profile: String,
    pub ontology_hash: String,
    pub source_metadata: SourceMetadata,
    pub context: Option<String>,
    pub nodes: Vec<Node>,
    pub claims: Vec<ClaimEdge>,
}
