use anyhow::{Result, ensure};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct ContextReview;
impl std::fmt::Display for ContextReview {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("canonical context exceeds safe input bounds and requires review")
    }
}
impl std::error::Error for ContextReview {}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FactContextConfig {
    pub schema: String,
    pub table: String,
    pub per_entity_limit: usize,
}
impl FactContextConfig {
    pub fn validate(&self) -> Result<()> {
        crate::postgres::quote_identifier(&self.schema)?;
        crate::postgres::quote_identifier(&self.table)?;
        ensure!(
            (1..=32).contains(&self.per_entity_limit),
            "invalid per-entity fact limit"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContextEntity {
    pub id: String,
    pub name: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextRequest {
    pub organization_id: String,
    pub entities: Vec<ContextEntity>,
    pub effective_at: DateTime<Utc>,
}
fn identifier(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 512 && !value.chars().any(char::is_control)
}
impl ContextRequest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            identifier(&self.organization_id),
            "invalid fact-context organization"
        );
        ensure!(self.entities.len() <= 32, "too many context entities");
        let mut ids = BTreeSet::new();
        for entity in &self.entities {
            ensure!(
                identifier(&entity.id) && identifier(&entity.name),
                "invalid context entity"
            );
            ensure!(ids.insert(&entity.id), "duplicate context entity identity");
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnownFact {
    pub id: String,
    pub subject_ref: String,
    pub predicate: String,
    pub object_ref: Option<String>,
    pub object_value: Option<String>,
    pub evidence: Vec<serde_json::Value>,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_to: Option<DateTime<Utc>>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EntityFacts {
    pub entity: ContextEntity,
    pub facts: Vec<KnownFact>,
    pub has_more: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FactContext {
    pub organization_id: String,
    pub entities: Vec<EntityFacts>,
}
impl FactContext {
    pub fn canonical_prefix(&self) -> Result<String> {
        ensure!(
            identifier(&self.organization_id),
            "invalid context organization"
        );
        let mut context = self.clone();
        let mut entities = BTreeSet::new();
        for entry in &mut context.entities {
            ensure!(
                entities.insert(&entry.entity.id),
                "duplicate context entity"
            );
            let mut facts = BTreeSet::new();
            for fact in &entry.facts {
                ensure!(
                    fact.subject_ref == entry.entity.id && facts.insert(&fact.id),
                    "invalid context fact identity"
                );
            }
            entry.facts.sort_by(|a, b| {
                (&a.predicate, &a.object_ref, &a.object_value, &a.id).cmp(&(
                    &b.predicate,
                    &b.object_ref,
                    &b.object_value,
                    &b.id,
                ))
            });
        }
        context
            .entities
            .sort_by_key(|entry| (entry.entity.name.to_lowercase(), entry.entity.id.clone()));
        let encoded = serde_json::to_string(&context)?;
        ensure!(encoded.len() <= 32768, ContextReview);
        Ok(encoded)
    }
}

#[cfg(feature = "postgres")]
pub(crate) async fn read(
    pool: &sqlx::PgPool,
    config: &FactContextConfig,
    request: &ContextRequest,
) -> Result<FactContext> {
    use anyhow::Context;
    use chrono::NaiveDateTime;
    use sqlx::{Postgres, QueryBuilder, Row};
    use std::collections::BTreeMap;
    config.validate()?;
    request.validate()?;
    let mut context = FactContext {
        organization_id: request.organization_id.clone(),
        entities: request
            .entities
            .iter()
            .cloned()
            .map(|entity| EntityFacts {
                entity,
                facts: Vec::new(),
                has_more: false,
            })
            .collect(),
    };
    if request.entities.is_empty() {
        return Ok(context);
    }
    let table = format!(
        "{}.{}",
        crate::postgres::quote_identifier(&config.schema)?,
        crate::postgres::quote_identifier(&config.table)?
    );
    let fail = |error: sqlx::Error| {
        let code = error
            .as_database_error()
            .and_then(|e| e.code())
            .map(|s| s.into_owned())
            .unwrap_or_else(|| "transport".into());
        anyhow::anyhow!("fact-context read failed ({code}); database details withheld")
    };
    let mut tx = pool.begin().await.map_err(fail)?;
    sqlx::query("SET TRANSACTION READ ONLY")
        .execute(&mut *tx)
        .await
        .map_err(fail)?;
    sqlx::query("SET LOCAL statement_timeout='5s'")
        .execute(&mut *tx)
        .await
        .map_err(fail)?;
    let ids: Vec<_> = request
        .entities
        .iter()
        .map(|entity| entity.id.clone())
        .collect();
    let mut query = QueryBuilder::<Postgres>::new(
        "WITH eligible AS (SELECT id,subject_ref,predicate,object_ref,object_value,evidence,valid_from,valid_to,row_number() OVER (PARTITION BY subject_ref ORDER BY predicate COLLATE \"C\",coalesce(object_ref,object_value,'') COLLATE \"C\",id COLLATE \"C\") AS n FROM ",
    );
    query.push(table).push(" WHERE organization_id=$1 AND scope='org' AND status='active' AND subject_ref=ANY($2::text[]) AND (valid_from IS NULL OR valid_from<=$3) AND (valid_to IS NULL OR valid_to>$3) AND jsonb_typeof(evidence)='array' AND EXISTS(SELECT 1 FROM jsonb_array_elements(CASE WHEN jsonb_typeof(evidence)='array' THEN evidence ELSE '[]'::jsonb END) e WHERE e->>'kind' IN ('app_database','oauth_mapping','admin_statement','user_statement','review_confirmation','repeated_observation') AND length(trim(coalesce(e->>'sourceRef','')))>0)) SELECT id,subject_ref,predicate,left(object_ref,2001) AS object_ref,left(object_value,2001) AS object_value,CASE WHEN octet_length(evidence::text)<=8192 THEN evidence ELSE NULL END AS evidence,valid_from,valid_to FROM eligible WHERE n<=$4 ORDER BY subject_ref COLLATE \"C\",n");
    let rows = query
        .build()
        .bind(&request.organization_id)
        .bind(ids)
        .bind(request.effective_at.naive_utc())
        .bind(i64::try_from(config.per_entity_limit + 1)?)
        .fetch_all(&mut *tx)
        .await
        .map_err(fail)?;
    let indices: BTreeMap<_, _> = context
        .entities
        .iter()
        .enumerate()
        .map(|(index, entry)| (entry.entity.id.clone(), index))
        .collect();
    for row in rows {
        let subject: String = row.try_get("subject_ref").map_err(fail)?;
        let entry =
            &mut context.entities[*indices.get(&subject).context("unexpected fact subject")?];
        if entry.facts.len() == config.per_entity_limit {
            entry.has_more = true;
            continue;
        }
        let object_ref: Option<String> = row.try_get("object_ref").map_err(fail)?;
        let object_value: Option<String> = row.try_get("object_value").map_err(fail)?;
        ensure!(
            object_ref
                .as_ref()
                .or(object_value.as_ref())
                .is_some_and(|s| !s.trim().is_empty()),
            "fact object is missing"
        );
        ensure!(
            object_ref
                .iter()
                .chain(object_value.iter())
                .all(|s| s.len() <= 2000),
            ContextReview
        );
        let evidence: Option<serde_json::Value> = row.try_get("evidence").map_err(fail)?;
        let evidence = evidence.context(ContextReview)?;
        let id: String = row.try_get("id").map_err(fail)?;
        let predicate: String = row.try_get("predicate").map_err(fail)?;
        ensure!(
            identifier(&id) && !predicate.is_empty() && predicate.len() <= 80,
            "invalid canonical fact identity/predicate"
        );
        let time = |name: &str| -> Result<Option<DateTime<Utc>>> {
            Ok(row
                .try_get::<Option<NaiveDateTime>, _>(name)
                .map_err(fail)?
                .map(|time| DateTime::from_naive_utc_and_offset(time, Utc)))
        };
        entry.facts.push(KnownFact {
            id,
            subject_ref: subject,
            predicate,
            object_ref,
            object_value,
            evidence: serde_json::from_value(evidence).context("invalid context evidence")?,
            valid_from: time("valid_from")?,
            valid_to: time("valid_to")?,
        });
    }
    tx.commit().await.map_err(fail)?;
    context.canonical_prefix()?;
    Ok(context)
}
