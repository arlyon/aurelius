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
