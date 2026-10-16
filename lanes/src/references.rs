use crate::contracts::{ByteRange, Chunk, Document, DocumentKey, SourceSpan};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const REFERENCE_VERSION: &str = "explicit-source-references-v1";

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceConfig {
    #[serde(default)]
    pub verified_aliases: BTreeMap<String, BTreeMap<String, Vec<String>>>,
}
impl ReferenceConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.verified_aliases.len() <= 128,
            "too many verified aliases"
        );
        for (scope, aliases) in &self.verified_aliases {
            ensure!(
                !scope.trim().is_empty()
                    && scope.len() <= 512
                    && !scope.chars().any(char::is_control),
                "invalid alias scope"
            );
            ensure!(aliases.len() <= 128, "too many scope aliases");
            for (alias, targets) in aliases {
                ensure!(
                    !alias.trim().is_empty()
                        && alias.len() <= 200
                        && !alias.chars().any(char::is_control),
                    "invalid verified alias"
                );
                ensure!(
                    !targets.is_empty()
                        && targets.len() <= 8
                        && targets.iter().all(|t| !t.trim().is_empty()
                            && t.len() <= 200
                            && !t.chars().any(char::is_control)),
                    "invalid alias targets"
                );
                ensure!(!is_pronoun(alias), "pronouns cannot be verified aliases");
            }
        }
        Ok(())
    }
    pub fn semantic_hash(&self) -> Result<String> {
        self.validate()?;
        let mut aliases = self.verified_aliases.clone();
        for aliases in aliases.values_mut() {
            for targets in aliases.values_mut() {
                targets.sort();
                targets.dedup();
            }
        }
        Ok(
            blake3::hash(&serde_json::to_vec(&(REFERENCE_VERSION, aliases))?)
                .to_hex()
                .to_string(),
        )
    }
    pub fn resolve(&self, document: &Document) -> Result<ReferenceContext> {
        self.validate()?;
        document.validate()?;
        let content_hash = document.content_hash();
        let mut annotations = 0;
        let mut span = |start: usize, end: usize| -> Result<SourceSpan> {
            ensure!(annotations < 2048, "reference annotation bound exceeded");
            annotations += 1;
            let range = ByteRange { start, end };
            Ok(SourceSpan {
                document: document.key.clone(),
                content_hash: content_hash.clone(),
                quote: range.text(document)?.into(),
                range,
            })
        };
        let mut groups: BTreeMap<(String, ReferenceRule), Vec<SourceSpan>> = BTreeMap::new();
        let mut unresolved = Vec::new();
        let ids =
            regex::Regex::new(r"<@([UW][A-Z0-9]{2,63})>|\bSlack user ([UW][A-Z0-9]{2,63})\b")?;
        for matched in ids.captures_iter(&document.text) {
            let id = matched
                .get(1)
                .or_else(|| matched.get(2))
                .expect("reference regex capture");
            let whole = matched.get(0).expect("reference regex match");
            groups
                .entry((
                    format!("slack:{}", id.as_str()),
                    ReferenceRule::ExplicitProviderId,
                ))
                .or_default()
                .push(span(whole.start(), whole.end())?);
        }
        for (alias, targets) in self
            .verified_aliases
            .get(&document.key.scope)
            .into_iter()
            .flatten()
        {
            let mut targets = targets.clone();
            targets.sort();
            targets.dedup();
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
                let mention = span(start, end)?;
                if targets.len() == 1 {
                    groups
                        .entry((targets[0].clone(), ReferenceRule::VerifiedAlias))
                        .or_default()
                        .push(mention);
                } else {
                    unresolved.push(UnresolvedReference {
                        mention,
                        reason: "ambiguous_alias".into(),
                    });
                }
            }
        }
        let pronouns = regex::Regex::new(
            r"(?i)\b(i|me|my|mine|myself|we|us|our|ours|he|him|his|she|her|hers|they|them|their|it|its|you|your)\b",
        )?;
        for found in pronouns.find_iter(&document.text) {
            unresolved.push(UnresolvedReference {
                mention: span(found.start(), found.end())?,
                reason: "no_verified_antecedent".into(),
            });
        }
        let mut groups: Vec<_> = groups
            .into_iter()
            .map(|((target, rule), mut mentions)| {
                mentions.sort_by_key(|m| (m.range.start, m.range.end));
                mentions.dedup();
                ReferenceGroup {
                    target,
                    rule,
                    mentions,
                }
            })
            .collect();
        groups.sort_by(|a, b| (&a.target, &a.rule).cmp(&(&b.target, &b.rule)));
        unresolved.sort_by_key(|u| (u.mention.range.start, u.mention.range.end));
        ensure!(
            groups.len() <= 128
                && groups.iter().map(|g| g.mentions.len()).sum::<usize>() + unresolved.len()
                    <= 2048,
            "reference annotation bound exceeded"
        );
        Ok(ReferenceContext {
            version: REFERENCE_VERSION.into(),
            document: document.key.clone(),
            content_hash: document.content_hash(),
            groups,
            unresolved,
            context_limited: false,
        })
    }
}
fn is_pronoun(text: &str) -> bool {
    matches!(
        text.to_ascii_lowercase().as_str(),
        "i" | "me"
            | "my"
            | "mine"
            | "myself"
            | "we"
            | "us"
            | "our"
            | "ours"
            | "he"
            | "him"
            | "his"
            | "she"
            | "her"
            | "hers"
            | "they"
            | "them"
            | "their"
            | "it"
            | "its"
            | "you"
            | "your"
    )
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceRule {
    ExplicitProviderId,
    VerifiedAlias,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReferenceGroup {
    pub target: String,
    pub rule: ReferenceRule,
    pub mentions: Vec<SourceSpan>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UnresolvedReference {
    pub mention: SourceSpan,
    pub reason: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReferenceContext {
    pub version: String,
    pub document: DocumentKey,
    pub content_hash: String,
    pub groups: Vec<ReferenceGroup>,
    pub unresolved: Vec<UnresolvedReference>,
    #[serde(default)]
    pub context_limited: bool,
}
impl ReferenceContext {
    pub fn for_chunk(&self, chunk: &Chunk) -> Result<Self> {
        chunk.text()?;
        ensure!(
            self.version == REFERENCE_VERSION
                && self.document == chunk.document.key
                && self.content_hash == chunk.document.content_hash(),
            "reference context changed document/revision"
        );
        for mention in self
            .groups
            .iter()
            .flat_map(|g| &g.mentions)
            .chain(self.unresolved.iter().map(|u| &u.mention))
        {
            ensure!(
                mention.document == chunk.document.key
                    && mention.content_hash == self.content_hash
                    && mention.range.text(&chunk.document)? == mention.quote,
                "reference mention changed source evidence"
            );
        }
        ensure!(
            self.groups.len() <= 128
                && self.groups.iter().map(|g| g.mentions.len()).sum::<usize>()
                    + self.unresolved.len()
                    <= 2048,
            "reference context exceeds bound"
        );
        let local =
            |m: &SourceSpan| m.range.start >= chunk.range.start && m.range.end <= chunk.range.end;
        let context_limited = self.context_limited
            || self
                .groups
                .iter()
                .filter(|g| g.mentions.iter().any(&local))
                .count()
                > 16
            || self
                .groups
                .iter()
                .any(|g| g.mentions.iter().filter(|m| local(m)).count() > 4)
            || self.unresolved.iter().filter(|u| local(&u.mention)).count() > 32;
        let groups = self
            .groups
            .iter()
            .filter(|g| g.mentions.iter().any(&local))
            .take(16)
            .map(|g| {
                let mut mentions: Vec<_> = g
                    .mentions
                    .iter()
                    .filter(|m| local(m))
                    .take(4)
                    .cloned()
                    .collect();
                if let Some(antecedent) = g.mentions.iter().find(|m| !local(m)) {
                    mentions.push(antecedent.clone());
                }
                ReferenceGroup {
                    target: g.target.clone(),
                    rule: g.rule.clone(),
                    mentions,
                }
            })
            .collect();
        Ok(Self {
            groups,
            context_limited,
            unresolved: self
                .unresolved
                .iter()
                .filter(|u| local(&u.mention))
                .take(32)
                .cloned()
                .collect(),
            ..self.clone()
        })
    }
}
