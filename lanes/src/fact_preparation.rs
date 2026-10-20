use crate::{
    contracts::{ByteRange, Document, DocumentKey},
    fact_context::{ContextEntity, FactContext},
    language::LanguageAnalysis,
    references::ReferenceConfig,
};
use anyhow::{Result, ensure};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const PREPARATION_VERSION: &str = "source-aligned-fact-preparation-v1";
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreparationRequest {
    pub document: Document,
    pub effective_at: DateTime<Utc>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AliasHypothesis {
    pub alias: String,
    pub range: ByteRange,
    pub candidate_refs: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedFactSource {
    pub document: DocumentKey,
    pub content_hash: String,
    pub preparation_profile: String,
    pub language: LanguageAnalysis,
    pub alias_hypotheses: Vec<AliasHypothesis>,
    pub canonical_context: FactContext,
    pub context_effective_at: DateTime<Utc>,
    pub context_snapshot_hash: String,
    #[serde(default)]
    pub provider_header: Option<crate::provider_metadata::ProviderTaskHeader>,
}
impl PreparedFactSource {
    pub fn from_analysis(
        document: &Document,
        language: LanguageAnalysis,
        aliases: Vec<AliasHypothesis>,
        canonical_context: FactContext,
        effective_at: DateTime<Utc>,
        preparation_profile: String,
    ) -> Result<Self> {
        document.validate()?;
        language.validate(document)?;
        ensure!(
            canonical_context.organization_id == document.key.scope,
            "context changed tenant"
        );
        for alias in &aliases {
            ensure!(
                alias.range.text(document)? == alias.alias,
                "alias changed source span"
            );
        }
        let prefix = canonical_context.canonical_prefix()?;
        let result = Self {
            document: document.key.clone(),
            content_hash: document.content_hash(),
            preparation_profile,
            language,
            alias_hypotheses: aliases,
            canonical_context,
            context_effective_at: effective_at,
            context_snapshot_hash: blake3::hash(prefix.as_bytes()).to_hex().to_string(),
            provider_header: None,
        };
        result.canonical_prefix()?;
        Ok(result)
    }
    pub fn with_provider_properties(
        mut self,
        document: &Document,
        properties: &crate::provider_metadata::ProviderProperties,
    ) -> Result<Self> {
        ensure!(
            self.document == document.key
                && self.content_hash == document.content_hash()
                && properties.provider == "asana",
            "provider preparation changed source"
        );
        let header = properties
            .header
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("verified provider header missing"))?;
        header.validate(document)?;
        self.provider_header = Some(header.clone());
        Ok(self)
    }
    pub fn canonical_prefix(&self) -> Result<String> {
        ensure!(
            self.canonical_context.organization_id == self.document.scope,
            "prepared context changed tenant"
        );
        let prefix = self.canonical_context.canonical_prefix()?;
        ensure!(
            blake3::hash(prefix.as_bytes()).to_hex().as_str() == self.context_snapshot_hash,
            "prepared canonical snapshot changed"
        );
        Ok(prefix)
    }
}
/// Request-local evidence catalog. Generated links never alter source coordinates.
#[derive(Clone, Debug, Serialize)]
pub struct SpanCatalog {
    pub document: DocumentKey,
    pub content_hash: String,
    pub links: Vec<String>,
    spans: BTreeMap<String, crate::contracts::SourceSpan>,
    noun_refs: std::collections::BTreeSet<String>,
    action_refs: std::collections::BTreeSet<String>,
    clause_refs: std::collections::BTreeSet<String>,
    record_ref: Option<String>,
    business_start: usize,
}
impl SpanCatalog {
    pub fn resolve(&self, href: &str) -> Result<&crate::contracts::SourceSpan> {
        self.spans
            .get(href)
            .ok_or_else(|| anyhow::anyhow!("unissued evidence marker"))
    }
    pub fn hrefs(&self) -> Vec<String> {
        self.spans.keys().cloned().collect()
    }
    pub fn quote(&self, href: &str) -> Result<&str> {
        Ok(&self.resolve(href)?.quote)
    }
    pub fn anchor_hrefs(&self, kind: &crate::atomic::AnchorKind) -> Vec<String> {
        match kind {
            crate::atomic::AnchorKind::Mention => self.noun_refs.iter().cloned().collect(),
            crate::atomic::AnchorKind::Action => self.action_refs.iter().cloned().collect(),
            crate::atomic::AnchorKind::Facet => {
                self.clause_refs.union(&self.noun_refs).cloned().collect()
            }
            crate::atomic::AnchorKind::Record => self.record_ref.iter().cloned().collect(),
        }
    }
    pub fn record_anchor(&self) -> Result<&crate::contracts::SourceSpan> {
        self.resolve(
            self.record_ref
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("source record anchor absent"))?,
        )
    }
    pub fn business_quote(&self, href: &str) -> Result<&str> {
        let span = self.resolve(href)?;
        if span.range.end <= self.business_start {
            return Ok("");
        }
        Ok(&span.quote[self.business_start.saturating_sub(span.range.start)..])
    }
}
impl PreparedFactSource {
    pub fn span_catalog(&self, chunk: &crate::contracts::Chunk) -> Result<SpanCatalog> {
        chunk.text()?;
        ensure!(
            self.document == chunk.document.key
                && self.content_hash == chunk.document.content_hash(),
            "span catalog source changed"
        );
        self.language.validate(&chunk.document)?;
        let mut ranges = std::collections::BTreeSet::new();
        let business_start = self.provider_header.as_ref().map_or(0, |h| h.text.len());
        let mut clause_refs = std::collections::BTreeSet::new();
        let root_range = if let Some(h) = &self.provider_header {
            Some(h.title)
        } else {
            let mut offset = 0;
            let mut found = None;
            for line in chunk.document.text.split_inclusive('\n') {
                let leading = line.len() - line.trim_start().len();
                let end = line.trim_end().len();
                if end > leading {
                    found = Some(ByteRange {
                        start: offset + leading,
                        end: offset + end,
                    });
                    break;
                }
                offset += line.len();
            }
            found
        };
        if let Some(r) = root_range {
            ranges.insert((r.start, r.end));
        }
        ranges.insert((chunk.range.start, chunk.range.end));
        let mut offset = chunk.range.start;
        for line in chunk.text()?.split_inclusive('\n') {
            let leading = line.len() - line.trim_start().len();
            let end = line.trim_end().len();
            if end > leading {
                ranges.insert((offset + leading, offset + end));
                if offset + leading >= business_start {
                    clause_refs.insert(format!("span:{}-{}", offset + leading, offset + end));
                }
            }
            offset += line.len();
        }
        for m in &self.language.mentions {
            if m.range.start >= chunk.range.start && m.range.end <= chunk.range.end {
                ranges.insert((m.range.start, m.range.end));
            }
        }
        for r in self
            .language
            .verb_spans
            .iter()
            .chain(self.language.number_spans.iter())
        {
            if r.start >= chunk.range.start && r.end <= chunk.range.end {
                ranges.insert((r.start, r.end));
            }
        }
        if let Some(header) = &self.provider_header {
            header.validate(&chunk.document)?;
            ranges.insert((header.title.start, header.title.end));
        }
        ensure!(ranges.len() <= 2048, "span catalog exceeds bound");
        let mut spans = BTreeMap::new();
        let mut links = Vec::new();
        for (start, end) in ranges {
            let range = ByteRange { start, end };
            let quote = range.text(&chunk.document)?.to_owned();
            let href = format!("span:{start}-{end}");
            let label = quote
                .replace('\\', "\\\\")
                .replace('[', "\\[")
                .replace(']', "\\]");
            links.push(format!("[{label}]({href})"));
            spans.insert(
                href,
                crate::contracts::SourceSpan {
                    document: chunk.document.key.clone(),
                    content_hash: self.content_hash.clone(),
                    range,
                    quote,
                },
            );
        }
        ensure!(
            serde_json::to_vec(&links)?.len() <= 128 * 1024,
            "annotated source exceeds bound"
        );
        let noun_refs = self
            .language
            .mentions
            .iter()
            .filter(|m| m.noun && !m.pronoun)
            .map(|m| format!("span:{}-{}", m.range.start, m.range.end))
            .filter(|href| spans.contains_key(href))
            .collect();
        let action_refs = spans
            .iter()
            .filter(|(_, span)| {
                span.range.start >= business_start
                    && self
                        .language
                        .verb_spans
                        .iter()
                        .any(|r| r.start >= span.range.start && r.end <= span.range.end)
            })
            .map(|(href, _)| href.clone())
            .collect();
        let record_ref = root_range.map(|r| format!("span:{}-{}", r.start, r.end));
        Ok(SpanCatalog {
            document: chunk.document.key.clone(),
            content_hash: self.content_hash.clone(),
            links,
            spans,
            noun_refs,
            action_refs,
            clause_refs,
            record_ref,
            business_start,
        })
    }
}
// Matching a trusted alias dictionary only proposes retrieval candidates. Neither
// a unique spelling nor a neural coreference cluster proves an entity identity.
pub fn alias_hypotheses(
    config: &ReferenceConfig,
    document: &Document,
) -> Result<Vec<AliasHypothesis>> {
    config.validate()?;
    document.validate()?;
    let mut output = Vec::new();
    for (alias, targets) in config
        .verified_aliases
        .get(&document.key.scope)
        .into_iter()
        .flatten()
    {
        let mut refs = targets.clone();
        refs.sort();
        refs.dedup();
        for (start, _) in document.text.match_indices(alias) {
            let end = start + alias.len();
            if document.text[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_')
                || document.text[end..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_')
            {
                continue;
            }
            ensure!(output.len() < 2048, "alias hypothesis budget exceeded");
            output.push(AliasHypothesis {
                alias: alias.clone(),
                range: ByteRange { start, end },
                candidate_refs: refs.clone(),
            });
        }
    }
    output.sort_by(|a, b| {
        (a.range.start, a.range.end, &a.alias).cmp(&(b.range.start, b.range.end, &b.alias))
    });
    Ok(output)
}
pub fn context_entities(
    aliases: &[AliasHypothesis],
    ontology: &crate::ontology::Ontology,
) -> Result<Vec<ContextEntity>> {
    ontology.validate()?;
    let mut entities: BTreeMap<String, String> = BTreeMap::new();
    for alias in aliases {
        // Ambiguous aliases and provider IDs without a declared canonical namespace
        // remain source-local hints. Never query canonical facts by raw noun strings.
        if alias.candidate_refs.len() != 1 {
            continue;
        }
        let id = &alias.candidate_refs[0];
        let Some((kind, _)) = id.split_once(':') else {
            continue;
        };
        if !ontology.entities.contains_key(kind) {
            continue;
        }
        entities
            .entry(id.clone())
            .and_modify(|name| {
                if (alias.alias.to_lowercase(), alias.alias.as_str())
                    < (name.to_lowercase(), name.as_str())
                {
                    *name = alias.alias.clone();
                }
            })
            .or_insert_with(|| alias.alias.clone());
    }
    ensure!(
        entities.len() <= 32,
        "canonical retrieval entity budget exceeded"
    );
    let mut entities: Vec<_> = entities
        .into_iter()
        .map(|(id, name)| ContextEntity { id, name })
        .collect();
    entities.sort_by_key(|entity| (entity.name.to_lowercase(), entity.id.clone()));
    Ok(entities)
}

#[cfg(all(feature = "language", feature = "postgres"))]
mod runtime {
    use super::*;
    use crate::{
        config::Config,
        fact_context::{ContextRequest, FactContextConfig},
        language::LanguageRuntime,
        postgres::PostgresSink,
    };
    use anyhow::Context;
    use std::{collections::BTreeSet, sync::Arc};
    #[derive(Clone)]
    pub struct FactPreparer {
        language: Arc<LanguageRuntime>,
        sink: PostgresSink,
        context: FactContextConfig,
        ontology: crate::ontology::Ontology,
        references: ReferenceConfig,
        scopes: BTreeSet<String>,
        profile: String,
    }
    impl FactPreparer {
        pub fn profile_for(config: &Config) -> Result<String> {
            config.validate()?;
            let language = LanguageRuntime::profile_for(
                config
                    .language
                    .as_ref()
                    .context("local language configuration required")?,
            )?;
            let context = config
                .fact_context
                .as_ref()
                .context("canonical context configuration required")?;
            let references = &config
                .chat
                .as_ref()
                .context("fact chat configuration required")?
                .references;
            Ok(blake3::hash(&serde_json::to_vec(&(
                PREPARATION_VERSION,
                language,
                config.ontology.semantic_hash()?,
                references.semantic_hash()?,
                context,
            ))?)
            .to_hex()
            .to_string())
        }
        pub fn with_sink(config: &Config, sink: PostgresSink) -> Result<Self> {
            config.validate()?;
            let language = Arc::new(LanguageRuntime::load(
                config
                    .language
                    .clone()
                    .context("local language configuration required")?,
            )?);
            let context = config
                .fact_context
                .clone()
                .context("canonical context configuration required")?;
            let references = config
                .chat
                .as_ref()
                .context("fact chat configuration required")?
                .references
                .clone();
            let profile = blake3::hash(&serde_json::to_vec(&(
                PREPARATION_VERSION,
                language.profile(),
                config.ontology.semantic_hash()?,
                references.semantic_hash()?,
                &context,
            ))?)
            .to_hex()
            .to_string();
            let scopes = config
                .lanes
                .values()
                .map(|lane| lane.scope.clone())
                .collect();
            Ok(Self {
                language,
                sink,
                context,
                ontology: config.ontology.clone(),
                references,
                scopes,
                profile,
            })
        }
        pub async fn prepare(
            &self,
            document: Document,
            effective_at: DateTime<Utc>,
        ) -> Result<PreparedFactSource> {
            document.validate()?;
            ensure!(
                self.scopes.contains(&document.key.scope),
                "unconfigured preparation scope"
            );
            let source = document.clone();
            let runtime = self.language.clone();
            let language = tokio::task::spawn_blocking(move || runtime.analyze(&source))
                .await
                .context("local language task failed")??;
            let aliases = alias_hypotheses(&self.references, &document)
                .context(crate::facts::ExtractionReview::Input)?;
            let entities = context_entities(&aliases, &self.ontology)
                .context(crate::facts::ExtractionReview::Input)?;
            let canonical_context = self
                .sink
                .fact_context(
                    &self.context,
                    &ContextRequest {
                        organization_id: document.key.scope.clone(),
                        entities,
                        effective_at,
                    },
                )
                .await
                .map_err(|error| {
                    if error
                        .downcast_ref::<crate::fact_context::ContextReview>()
                        .is_some()
                    {
                        error.context(crate::facts::ExtractionReview::Input)
                    } else {
                        error
                    }
                })?;
            PreparedFactSource::from_analysis(
                &document,
                language,
                aliases,
                canonical_context,
                effective_at,
                self.profile.clone(),
            )
        }
    }
}
#[cfg(all(feature = "language", feature = "postgres"))]
pub use runtime::FactPreparer;
