use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Ontology {
    pub id: String,
    pub version: String,
    #[serde(default)]
    pub entities: BTreeMap<String, EntityType>,
    #[serde(default)]
    pub predicates: BTreeMap<String, Predicate>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EntityType {
    pub description: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Predicate {
    pub description: String,
    #[serde(default)]
    pub subject_types: BTreeSet<String>,
    #[serde(default)]
    pub object_types: BTreeSet<String>,
}
fn name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 80
        && value
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
        && value.as_bytes()[0].is_ascii_lowercase()
}
impl Ontology {
    pub fn validate(&self) -> Result<()> {
        ensure!(name(&self.id), "invalid ontology identity");
        ensure!(
            !self.version.trim().is_empty()
                && self.version.len() <= 80
                && !self.version.chars().any(char::is_control),
            "invalid ontology version"
        );
        ensure!(
            self.entities.len() <= 200 && self.predicates.len() <= 500,
            "ontology exceeds bound"
        );
        for (kind, entity) in &self.entities {
            ensure!(name(kind), "invalid entity type");
            description(&entity.description)?;
        }
        for (relation, predicate) in &self.predicates {
            ensure!(name(relation), "invalid predicate");
            description(&predicate.description)?;
            for kind in predicate
                .subject_types
                .iter()
                .chain(&predicate.object_types)
            {
                ensure!(
                    self.entities.contains_key(kind),
                    "predicate references unknown entity type"
                );
            }
        }
        ensure!(
            serde_json::to_vec(self)?.len() <= 64 * 1024,
            "ontology exceeds prompt budget"
        );
        Ok(())
    }
    pub fn semantic_hash(&self) -> Result<String> {
        self.validate()?;
        // Ordered maps/sets make comments, declaration order and whitespace irrelevant.
        Ok(blake3::hash(&serde_json::to_vec(self)?)
            .to_hex()
            .to_string())
    }
    pub fn mapping(&self, predicate: &str) -> crate::contracts::PredicateMapping {
        if self.predicates.contains_key(predicate) {
            crate::contracts::PredicateMapping::Declared {
                predicate: predicate.into(),
            }
        } else {
            crate::contracts::PredicateMapping::Unmapped {
                predicate: predicate.into(),
            }
        }
    }
}
fn description(value: &str) -> Result<()> {
    ensure!(
        !value.trim().is_empty() && value.len() <= 2000,
        "invalid ontology description"
    );
    Ok(())
}
