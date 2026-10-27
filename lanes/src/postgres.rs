use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostgresConfig {
    pub url_env: String,
    pub schema: String,
    pub artifacts_table: String,
    pub chunks_table: String,
    pub tenant_setting: Option<String>,
    #[serde(default = "default_pool_size")]
    pub pool_size: u32,
}
fn default_pool_size() -> u32 {
    2
}
#[derive(Debug)]
struct DatabaseFailure {
    code: String,
}
impl std::fmt::Display for DatabaseFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Postgres operation failed ({}); database details withheld",
            self.code
        )
    }
}
impl std::error::Error for DatabaseFailure {}
pub(crate) fn database_failure_code(error: &anyhow::Error) -> Option<&str> {
    error.chain().find_map(|cause| {
        cause
            .downcast_ref::<DatabaseFailure>()
            .map(|failure| failure.code.as_str())
    })
}
impl PostgresConfig {
    pub fn validate(&self) -> Result<()> {
        crate::config::validate_env_name(&self.url_env)?;
        ensure!(
            (1..=4).contains(&self.pool_size),
            "invalid Postgres pool size"
        );
        for name in [&self.schema, &self.artifacts_table, &self.chunks_table] {
            quote_identifier(name)?;
        }
        ensure!(
            self.artifacts_table != self.chunks_table,
            "Postgres targets must be distinct"
        );
        if let Some(setting) = &self.tenant_setting {
            ensure!(
                setting.len() <= 128
                    && setting.split('.').count() == 2
                    && setting.split('.').all(|part| !part.is_empty()
                        && part
                            .bytes()
                            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')),
                "invalid tenant setting"
            );
        }
        Ok(())
    }
    #[cfg(feature = "postgres")]
    fn relation(&self, table: &str) -> Result<String> {
        Ok(format!(
            "{}.{}",
            quote_identifier(&self.schema)?,
            quote_identifier(table)?
        ))
    }
}
pub(crate) fn quote_identifier(name: &str) -> Result<String> {
    ensure!(
        !name.is_empty() && name.len() <= 63 && !name.chars().any(char::is_control),
        "invalid Postgres identifier"
    );
    Ok(format!("\"{}\"", name.replace('"', "\"\"")))
}

#[cfg(feature = "postgres")]
mod driver {
    use super::*;
    use crate::{document::CanonicalDocument, indexing::IndexedDocument};
    use anyhow::Context;
    use serde_json::json;
    use sqlx::{PgPool, Postgres, QueryBuilder, Row, Transaction, postgres::PgPoolOptions};
    use std::{
        collections::{BTreeMap, BTreeSet},
        time::Duration,
    };

    #[derive(Clone)]
    pub struct PostgresSink {
        pool: PgPool,
        config: PostgresConfig,
        artifacts: String,
        chunks: String,
        vector_type: String,
        dimensions: usize,
    }
    #[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
    pub struct StoredDocument {
        pub artifact_id: String,
        pub chunks: usize,
    }

    #[derive(Clone, Debug)]
    pub struct ExtractionInput {
        pub document: crate::contracts::Document,
        pub ranges: Vec<crate::contracts::ByteRange>,
        pub connection_id: String,
        pub embedding_profile: String,
        pub kind: crate::document::DocumentKind,
        pub sources: Vec<crate::document::SourceRef>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct FactManifestEntry {
        pub artifact_id: String,
        pub revision: String,
        pub windows: i64,
    }

    fn database_error(error: sqlx::Error) -> anyhow::Error {
        let code = match &error {
            sqlx::Error::PoolTimedOut => "pool_timeout".into(),
            sqlx::Error::PoolClosed => "pool_closed".into(),
            _ => error
                .as_database_error()
                .and_then(|e| e.code())
                .filter(|code| {
                    code.len() == 5
                        && code
                            .bytes()
                            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
                })
                .map(|code| code.into_owned())
                .unwrap_or_else(|| "transport".into()),
        };
        DatabaseFailure { code }.into()
    }
    #[test]
    fn pool_failure_diagnostics_never_retain_private_driver_details() {
        let error = database_error(sqlx::Error::Protocol("PRIVATE_DATABASE_SECRET".into()));
        assert_eq!(super::database_failure_code(&error), Some("transport"));
        assert!(!format!("{error:#}").contains("PRIVATE_DATABASE_SECRET"));
        let error = database_error(sqlx::Error::PoolTimedOut);
        assert_eq!(super::database_failure_code(&error), Some("pool_timeout"));
    }
    impl PostgresSink {
        pub async fn connect(config: PostgresConfig, dimensions: usize) -> Result<Self> {
            config.validate()?;
            ensure!(
                (1..=16000).contains(&dimensions),
                "invalid embedding dimensions"
            );
            let pool = PgPoolOptions::new()
                .max_connections(config.pool_size)
                .idle_timeout(Some(Duration::from_secs(60)))
                .acquire_timeout(Duration::from_secs(15))
                .connect(&crate::config::credential(&config.url_env)?)
                .await
                .map_err(database_error)?;
            Self::from_pool(config, dimensions, pool).await
        }
        pub async fn from_pool(
            config: PostgresConfig,
            dimensions: usize,
            pool: PgPool,
        ) -> Result<Self> {
            config.validate()?;
            ensure!(
                (1..=16000).contains(&dimensions),
                "invalid embedding dimensions"
            );
            let artifacts = config.relation(&config.artifacts_table)?;
            let chunks = config.relation(&config.chunks_table)?;
            let vector_schema: String = sqlx::query_scalar("SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid=e.extnamespace WHERE e.extname='vector'")
                .fetch_one(&pool).await.map_err(database_error)?;
            let sink = Self {
                pool,
                config,
                artifacts,
                chunks,
                vector_type: format!("{}.\"vector\"", quote_identifier(&vector_schema)?),
                dimensions,
            };
            sink.validate_structure().await?;
            Ok(sink)
        }
        async fn validate_structure(&self) -> Result<()> {
            self.validate_table(
                &self.artifacts,
                &[
                    ("id", "text"),
                    ("organization_id", "text"),
                    ("source_type", "text"),
                    ("source_uri", "text"),
                    ("external_id", "text"),
                    ("revision_id", "text"),
                    ("content_hash", "text"),
                    ("mime_type", "text"),
                    ("title", "text"),
                    ("metadata", "jsonb"),
                    ("deleted_at", "timestamp"),
                    ("created_at", "timestamp"),
                    ("updated_at", "timestamp"),
                ],
            )
            .await?;
            self.validate_table(
                &self.chunks,
                &[
                    ("id", "text"),
                    ("organization_id", "text"),
                    ("artifact_source_id", "text"),
                    ("source_path", "text"),
                    ("source_hash", "text"),
                    ("chunk_index", "int4"),
                    ("context_window_id", "text"),
                    ("child_text", "text"),
                    ("parent_block", "text"),
                    ("context_header", "text"),
                    ("metadata", "jsonb"),
                    ("embedding", "vector"),
                    ("created_at", "timestamp"),
                    ("updated_at", "timestamp"),
                ],
            )
            .await?;
            let width: i32 = sqlx::query_scalar("SELECT atttypmod FROM pg_attribute WHERE attrelid=to_regclass($1) AND attname='embedding' AND NOT attisdropped")
                .bind(&self.chunks).fetch_one(&self.pool).await.map_err(database_error)?;
            ensure!(
                usize::try_from(width)? == self.dimensions,
                "Postgres vector dimensions differ from embedding profile"
            );
            let linked: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM pg_constraint c JOIN pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=ANY(c.conkey) JOIN pg_attribute r ON r.attrelid=c.confrelid AND r.attnum=ANY(c.confkey) WHERE c.contype='f' AND c.conrelid=to_regclass($1) AND c.confrelid=to_regclass($2) AND cardinality(c.conkey)=1 AND a.attname='artifact_source_id' AND r.attname='id')")
                .bind(&self.chunks).bind(&self.artifacts).fetch_one(&self.pool).await.map_err(database_error)?;
            ensure!(
                linked,
                "chunk artifact foreign key does not reference configured artifact table"
            );
            Ok(())
        }
        async fn validate_table(&self, relation: &str, required: &[(&str, &str)]) -> Result<()> {
            let rows = sqlx::query("SELECT a.attname,t.typname FROM pg_attribute a JOIN pg_type t ON t.oid=a.atttypid WHERE a.attrelid=to_regclass($1) AND a.attnum>0 AND NOT a.attisdropped")
                .bind(relation).fetch_all(&self.pool).await.map_err(database_error)?;
            let columns: BTreeMap<String, String> = rows
                .into_iter()
                .map(|r| Ok((r.try_get("attname")?, r.try_get("typname")?)))
                .collect::<std::result::Result<_, sqlx::Error>>()
                .map_err(database_error)?;
            for (column, kind) in required {
                ensure!(
                    columns.get(*column).map(String::as_str) == Some(*kind),
                    "Postgres table {relation} requires column {column} of type {kind}"
                );
            }
            let key: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM pg_index i JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=i.indkey[0] WHERE i.indrelid=to_regclass($1) AND i.indisunique AND i.indisvalid AND i.indpred IS NULL AND i.indnkeyatts=1 AND a.attname='id')")
                .bind(relation).fetch_one(&self.pool).await.map_err(database_error)?;
            ensure!(key, "Postgres target requires a unique id key");
            let permitted: bool = sqlx::query_scalar(
                "SELECT has_table_privilege($1,'SELECT') AND has_table_privilege($1,'INSERT')",
            )
            .bind(relation)
            .fetch_one(&self.pool)
            .await
            .map_err(database_error)?;
            ensure!(
                permitted,
                "Postgres target requires SELECT and INSERT grants"
            );
            Ok(())
        }
        async fn tenant(&self, scope: &str) -> Result<Transaction<'_, Postgres>> {
            let mut tx = self.pool.begin().await.map_err(database_error)?;
            if let Some(setting) = &self.config.tenant_setting {
                sqlx::query("SELECT set_config($1,$2,true)")
                    .bind(setting)
                    .bind(scope)
                    .execute(&mut *tx)
                    .await
                    .map_err(database_error)?;
            }
            Ok(tx)
        }
        pub fn artifact_id(document: &CanonicalDocument, profile: &str) -> Result<String> {
            Ok(blake3::hash(&serde_json::to_vec(&(
                "canonical-pg-v1",
                &document.organization_id,
                &document.connection_id,
                &document.id,
                document.content_hash()?,
                profile,
            ))?)
            .to_hex()
            .to_string())
        }
        pub async fn contains(&self, document: &CanonicalDocument, profile: &str) -> Result<bool> {
            let id = Self::artifact_id(document, profile)?;
            let mut tx = self.tenant(&document.organization_id).await?;
            let mut query = QueryBuilder::<Postgres>::new(
                "SELECT metadata->>'chunk_count' AS expected,(SELECT count(*) FROM ",
            );
            query
                .push(&self.chunks)
                .push(
                    " c WHERE c.artifact_source_id=a.id AND c.organization_id=$2) AS actual FROM ",
                )
                .push(&self.artifacts)
                .push(" a WHERE a.id=$1 AND a.organization_id=$2");
            let row = query
                .build()
                .bind(id)
                .bind(&document.organization_id)
                .fetch_optional(&mut *tx)
                .await
                .map_err(database_error)?;
            let complete = if let Some(row) = row {
                let expected: Option<String> = row.try_get("expected").map_err(database_error)?;
                let actual: i64 = row.try_get("actual").map_err(database_error)?;
                expected
                    .and_then(|n| n.parse::<i64>().ok())
                    .is_some_and(|n| n > 0 && n == actual)
            } else {
                false
            };
            tx.commit().await.map_err(database_error)?;
            Ok(complete)
        }
        pub async fn fact_context(
            &self,
            config: &crate::fact_context::FactContextConfig,
            request: &crate::fact_context::ContextRequest,
        ) -> Result<crate::fact_context::FactContext> {
            crate::fact_context::read(&self.pool, config, request).await
        }
        pub async fn fact_manifest(
            &self,
            scope: &str,
            source_profiles: &[String],
            seed: &str,
        ) -> Result<Vec<FactManifestEntry>> {
            ensure!(
                !source_profiles.is_empty() && source_profiles.len() <= 100,
                "source profiles required"
            );
            let mut tx = self.tenant(scope).await?;
            let mut query = QueryBuilder::<Postgres>::new(
                "SELECT id,revision_id,(metadata->>'chunk_count')::bigint AS windows FROM ",
            );
            query.push(&self.artifacts).push(" WHERE organization_id=$1 AND deleted_at IS NULL AND metadata->>'profile'=ANY($2::text[]) AND (metadata->>'chunk_count')::bigint>0 AND NOT(metadata ? 'candidate_graph') ORDER BY md5(id || $3),id LIMIT 20001");
            let rows = query
                .build()
                .bind(scope)
                .bind(source_profiles)
                .bind(seed)
                .fetch_all(&mut *tx)
                .await
                .map_err(database_error)?;
            ensure!(
                rows.len() <= 20000,
                "source manifest exceeds document bound"
            );
            let entries = rows
                .into_iter()
                .map(|row| {
                    Ok(FactManifestEntry {
                        artifact_id: row.try_get("id").map_err(database_error)?,
                        revision: row.try_get("revision_id").map_err(database_error)?,
                        windows: row.try_get("windows").map_err(database_error)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                entries.iter().all(|e| e.windows > 0
                    && e.artifact_id.len() == 64
                    && e.artifact_id.bytes().all(|b| b.is_ascii_hexdigit())
                    && !e.revision.is_empty())
                    && entries
                        .iter()
                        .try_fold(0_i64, |sum, e| sum.checked_add(e.windows))
                        .is_some_and(|sum| sum <= 20000),
                "unsafe or oversized source manifest"
            );
            tx.commit().await.map_err(database_error)?;
            Ok(entries)
        }
        pub async fn next_fact_source(
            &self,
            scope: &str,
            source_profiles: &[String],
            profile: &str,
        ) -> Result<Option<String>> {
            ensure!(
                !source_profiles.is_empty() && source_profiles.len() <= 100,
                "fact source profiles required"
            );
            let mut tx = self.tenant(scope).await?;
            let mut query = QueryBuilder::<Postgres>::new("SELECT a.id FROM ");
            query.push(&self.artifacts).push(" a WHERE a.organization_id=$1 AND a.metadata->>'profile'=ANY($2::text[]) AND (a.metadata->>'chunk_count')::bigint>0 AND NOT EXISTS(SELECT 1 FROM ");
            query.push(&self.artifacts).push(" r WHERE r.organization_id=a.organization_id AND r.metadata->>'source_artifact_id'=a.id AND r.metadata->>'profile'=$3 AND r.metadata->>'projection'='pg-chunk-projection-v1' AND r.metadata->>'status' IN ('facts_complete','facts_deferred')) ORDER BY a.created_at,a.id LIMIT 1");
            let id = query
                .build_query_scalar::<String>()
                .bind(scope)
                .bind(source_profiles)
                .bind(profile)
                .fetch_optional(&mut *tx)
                .await
                .map_err(database_error)?;
            tx.commit().await.map_err(database_error)?;
            Ok(id)
        }
        pub async fn fact_status(
            &self,
            scope: &str,
            source: &str,
            profile: &str,
            range: Option<crate::contracts::ByteRange>,
        ) -> Result<Option<String>> {
            let mut tx = self.tenant(scope).await?;
            let mut query = QueryBuilder::<Postgres>::new("SELECT metadata->>'status' FROM ");
            query.push(&self.artifacts).push(" WHERE organization_id=$1 AND metadata->>'source_artifact_id'=$2 AND metadata->>'profile'=$3 AND metadata->>'projection'='pg-chunk-projection-v1' AND ");
            if range.is_some() {
                query.push("metadata->>'status'='fact_window_review' AND metadata->'source_range'=$4::jsonb");
            } else {
                query.push("metadata->>'status' IN ('facts_complete','facts_deferred')");
            }
            query.push(" LIMIT 1");
            let mut q = query
                .build_query_scalar::<String>()
                .bind(scope)
                .bind(source)
                .bind(profile);
            if let Some(range) = range {
                q = q.bind(serde_json::to_value(range)?);
            }
            let status = q.fetch_optional(&mut *tx).await.map_err(database_error)?;
            tx.commit().await.map_err(database_error)?;
            Ok(status)
        }
        pub async fn ingestion_cursor(
            &self,
            scope: &str,
            request: &crate::lanes::LaneRequest,
        ) -> Result<Option<u64>> {
            request.validate()?;
            let mut tx = self.tenant(scope).await?;
            let mut query =
                QueryBuilder::<Postgres>::new("SELECT metadata->'receipt' AS receipt FROM ");
            query.push(&self.artifacts).push(" WHERE organization_id=$1 AND source_type='aurelius/ingestion-checkpoint' AND revision_id=$2 AND metadata->>'lane'=$3 ORDER BY (metadata->>'sequence')::bigint DESC LIMIT 1");
            let row = query
                .build()
                .bind(scope)
                .bind(&request.profile)
                .bind(&request.lane)
                .fetch_optional(&mut *tx)
                .await
                .map_err(database_error)?;
            let sequence = row
                .map(|row| -> Result<u64> {
                    let value: serde_json::Value =
                        row.try_get("receipt").map_err(database_error)?;
                    let receipt: crate::ingest::IngestReceipt =
                        serde_json::from_value(value).context("invalid ingestion checkpoint")?;
                    ensure!(
                        receipt.epoch.request.lane == *request,
                        "checkpoint changed lane identity"
                    );
                    Ok(receipt.epoch.sequence)
                })
                .transpose()?;
            tx.commit().await.map_err(database_error)?;
            Ok(sequence)
        }
        pub async fn record_ingestion(
            &self,
            scope: &str,
            receipt: &crate::ingest::IngestReceipt,
        ) -> Result<()> {
            let request = &receipt.epoch.request.lane;
            request.validate()?;
            ensure!(
                receipt.epoch.sequence > 0 && receipt.epoch.sequence <= i64::MAX as u64,
                "invalid ingestion checkpoint sequence"
            );
            let id = blake3::hash(&serde_json::to_vec(&(
                "ingestion-checkpoint-v1",
                scope,
                request,
                receipt.epoch.sequence,
            ))?)
            .to_hex()
            .to_string();
            let metadata = json!({"status":"ingestion_complete","lane":request.lane,"sequence":receipt.epoch.sequence,"receipt":receipt,"chunk_count":0});
            let mut tx = self.tenant(scope).await?;
            let mut query = QueryBuilder::<Postgres>::new("INSERT INTO ");
            query.push(&self.artifacts).push(" (id,organization_id,source_type,source_uri,external_id,revision_id,content_hash,mime_type,title,metadata,created_at,updated_at) VALUES ($1,$2,'aurelius/ingestion-checkpoint',$3,$1,$4,$1,'application/vnd.aurelius.ingestion-checkpoint+json','Completed ingestion checkpoint',$5,timezone('UTC',now()),timezone('UTC',now())) ON CONFLICT(id) DO NOTHING");
            query
                .build()
                .bind(&id)
                .bind(scope)
                .bind(format!(
                    "aurelius-checkpoint://{}/{}",
                    request.lane, receipt.epoch.sequence
                ))
                .bind(&request.profile)
                .bind(metadata)
                .execute(&mut *tx)
                .await
                .map_err(database_error)?;
            tx.commit().await.map_err(database_error)?;
            Ok(())
        }
        pub async fn record_fact_status(
            &self,
            scope: &str,
            source: &str,
            profile: &str,
            range: Option<crate::contracts::ByteRange>,
            reason: &str,
        ) -> Result<String> {
            self.record_fact_status_details(scope, source, profile, range, reason, None)
                .await
        }
        pub async fn record_fact_review(
            &self,
            scope: &str,
            source: &str,
            profile: &str,
            range: crate::contracts::ByteRange,
            diagnostic: crate::facts::ReviewDiagnostic,
        ) -> Result<String> {
            self.record_fact_status_details(
                scope,
                source,
                profile,
                Some(range),
                "output_rejected",
                Some(diagnostic),
            )
            .await
        }
        async fn record_fact_status_details(
            &self,
            scope: &str,
            source: &str,
            profile: &str,
            range: Option<crate::contracts::ByteRange>,
            reason: &str,
            diagnostic: Option<crate::facts::ReviewDiagnostic>,
        ) -> Result<String> {
            let status = match (range.is_some(), reason) {
                (false, "complete") => "facts_complete",
                (false, "input_review") => "facts_deferred",
                (true, "input_review" | "output_rejected") => "fact_window_review",
                _ => anyhow::bail!("invalid fact receipt reason"),
            };
            ensure!(!profile.is_empty(), "fact profile required");
            let mut tx = self.tenant(scope).await?;
            let mut parent = QueryBuilder::<Postgres>::new("SELECT revision_id,source_uri FROM ");
            parent
                .push(&self.artifacts)
                .push(" WHERE id=$1 AND organization_id=$2");
            let row = parent
                .build()
                .bind(source)
                .bind(scope)
                .fetch_optional(&mut *tx)
                .await
                .map_err(database_error)?
                .context("fact receipt parent not found")?;
            let revision: String = row.try_get("revision_id").map_err(database_error)?;
            let uri: String = row.try_get("source_uri").map_err(database_error)?;
            let id = blake3::hash(&serde_json::to_vec(&(
                "fact-receipt-v1",
                scope,
                source,
                &revision,
                profile,
                range,
                status,
            ))?)
            .to_hex()
            .to_string();
            let metadata = serde_json::json!({"status":status,"profile":profile,"source_artifact_id":source,"source_revision":revision,"projection":"pg-chunk-projection-v1","source_range":range,"reason":reason,"diagnostic":diagnostic,"chunk_count":0});
            let mut query = QueryBuilder::<Postgres>::new("INSERT INTO ");
            query.push(&self.artifacts).push(" (id,organization_id,source_type,source_uri,external_id,revision_id,content_hash,mime_type,title,metadata,created_at,updated_at) VALUES ($1,$2,'aurelius/fact-status',$3,$4,$5,$1,'application/vnd.aurelius.fact-status+json','Fact extraction receipt',$6,timezone('UTC',now()),timezone('UTC',now())) ON CONFLICT(id) DO NOTHING");
            query
                .build()
                .bind(&id)
                .bind(scope)
                .bind(uri)
                .bind(format!("fact-status/{id}"))
                .bind(profile)
                .bind(metadata)
                .execute(&mut *tx)
                .await
                .map_err(database_error)?;
            tx.commit().await.map_err(database_error)?;
            Ok(id)
        }
        pub async fn extraction_input(
            &self,
            scope: &str,
            artifact_id: &str,
        ) -> Result<ExtractionInput> {
            ensure!(
                !scope.trim().is_empty() && !artifact_id.is_empty(),
                "extraction source scope/identity required"
            );
            let mut tx = self.tenant(scope).await?;
            let mut query =
                QueryBuilder::<Postgres>::new("SELECT revision_id,source_uri,metadata FROM ");
            query
                .push(&self.artifacts)
                .push(" WHERE id=$1 AND organization_id=$2");
            let source = query
                .build()
                .bind(artifact_id)
                .bind(scope)
                .fetch_optional(&mut *tx)
                .await
                .map_err(database_error)?
                .context("scoped extraction source not found")?;
            let metadata: serde_json::Value = source.try_get("metadata").map_err(database_error)?;
            ensure!(
                metadata.get("candidate_graph").is_none()
                    && metadata
                        .pointer("/context/status")
                        .and_then(serde_json::Value::as_str)
                        != Some("deferred"),
                crate::facts::ExtractionReview::Input
            );
            let mut query = QueryBuilder::<Postgres>::new(
                "SELECT chunk_index,octet_length(child_text) AS bytes,left(child_text,16001) AS child_text FROM ",
            );
            query.push(&self.chunks).push(" WHERE artifact_source_id=$1 AND organization_id=$2 ORDER BY chunk_index,id LIMIT 513");
            let rows = query
                .build()
                .bind(artifact_id)
                .bind(scope)
                .fetch_all(&mut *tx)
                .await
                .map_err(database_error)?;
            let expected = metadata
                .get("chunk_count")
                .and_then(serde_json::Value::as_u64)
                .context(crate::facts::ExtractionReview::Input)?;
            ensure!(
                !rows.is_empty() && rows.len() <= 512 && u64::try_from(rows.len())? == expected,
                crate::facts::ExtractionReview::Input
            );
            let mut text = String::new();
            let mut ranges = Vec::new();
            for (index, row) in rows.iter().enumerate() {
                ensure!(
                    row.try_get::<i32, _>("chunk_index")
                        .map_err(database_error)?
                        == i32::try_from(index)?,
                    crate::facts::ExtractionReview::Input
                );
                let bytes: i32 = row.try_get("bytes").map_err(database_error)?;
                ensure!(
                    (1..=16000).contains(&bytes),
                    crate::facts::ExtractionReview::Input
                );
                let part: String = row.try_get("child_text").map_err(database_error)?;
                ensure!(
                    part.len() == usize::try_from(bytes)?,
                    crate::facts::ExtractionReview::Input
                );
                if index > 0 {
                    text.push_str("\n\n");
                }
                let start = text.len();
                text.push_str(&part);
                ensure!(
                    text.len() <= crate::contracts::MAX_DOCUMENT_BYTES,
                    crate::facts::ExtractionReview::Input
                );
                ranges.push(crate::contracts::ByteRange {
                    start,
                    end: text.len(),
                });
            }
            let document = crate::contracts::Document {
                key: crate::contracts::DocumentKey {
                    scope: scope.into(),
                    source: artifact_id.into(),
                    revision: format!(
                        "pg-chunk-projection-v1/{}",
                        source
                            .try_get::<String, _>("revision_id")
                            .map_err(database_error)?
                    ),
                },
                text,
                context: Some(artifact_id.into()),
                metadata: crate::contracts::SourceMetadata {
                    uri: Some(source.try_get("source_uri").map_err(database_error)?),
                    ..Default::default()
                },
            };
            document
                .validate()
                .context(crate::facts::ExtractionReview::Input)?;
            tx.commit().await.map_err(database_error)?;
            let connection_id = metadata
                .get("connection_id")
                .and_then(serde_json::Value::as_str)
                .context(crate::facts::ExtractionReview::Input)?
                .to_owned();
            let embedding_profile = metadata
                .get("profile")
                .and_then(serde_json::Value::as_str)
                .context(crate::facts::ExtractionReview::Input)?
                .to_owned();
            let kind = serde_json::from_value(
                metadata
                    .get("kind")
                    .cloned()
                    .context(crate::facts::ExtractionReview::Input)?,
            )
            .context(crate::facts::ExtractionReview::Input)?;
            let sources: Vec<crate::document::SourceRef> = serde_json::from_value(
                metadata
                    .get("sources")
                    .cloned()
                    .context(crate::facts::ExtractionReview::Input)?,
            )
            .context(crate::facts::ExtractionReview::Input)?;
            ensure!(
                !connection_id.is_empty() && sources.len() <= 20000,
                crate::facts::ExtractionReview::Input
            );
            Ok(ExtractionInput {
                document,
                ranges,
                connection_id,
                embedding_profile,
                kind,
                sources,
            })
        }
        pub fn candidate_id(chunk: &crate::contracts::Chunk, profile: &str) -> Result<String> {
            chunk.text()?;
            ensure!(!profile.is_empty(), "candidate profile required");
            Ok(blake3::hash(&serde_json::to_vec(&(
                "candidate-graph-v1",
                &chunk.document.key,
                chunk.document.content_hash(),
                chunk.range,
                profile,
            ))?)
            .to_hex()
            .to_string())
        }
        pub async fn candidate_exists(
            &self,
            chunk: &crate::contracts::Chunk,
            profile: &str,
        ) -> Result<bool> {
            let id = Self::candidate_id(chunk, profile)?;
            let mut tx = self.tenant(&chunk.document.key.scope).await?;
            let mut query = QueryBuilder::<Postgres>::new("SELECT EXISTS(SELECT 1 FROM ");
            query.push(&self.artifacts).push(" WHERE id=$1 AND organization_id=$2 AND metadata->>'status'='candidate' AND metadata ? 'candidate_graph')");
            let exists = query
                .build_query_scalar::<bool>()
                .bind(id)
                .bind(&chunk.document.key.scope)
                .fetch_one(&mut *tx)
                .await
                .map_err(database_error)?;
            tx.commit().await.map_err(database_error)?;
            Ok(exists)
        }
        pub async fn write_candidates(
            &self,
            chunk: &crate::contracts::Chunk,
            graph: &crate::contracts::CandidateGraph,
            references: &crate::references::ReferenceContext,
        ) -> Result<String> {
            chunk.text()?;
            ensure!(
                graph.document == chunk.document.key
                    && graph.content_hash == chunk.document.content_hash()
                    && graph.contract_version == crate::contracts::CONTRACT_VERSION,
                "candidate source identity mismatch"
            );
            let local = |span: &crate::contracts::SourceSpan| -> Result<()> {
                span.validate(&chunk.document)?;
                ensure!(
                    span.range.start >= chunk.range.start && span.range.end <= chunk.range.end,
                    "candidate evidence outside source chunk"
                );
                Ok(())
            };
            for node in &graph.nodes {
                if let crate::contracts::Node::Mention { evidence, .. } = node {
                    local(evidence)?;
                }
            }
            for claim in &graph.claims {
                ensure!(!claim.evidence.is_empty(), "candidate evidence missing");
                for evidence in &claim.evidence {
                    ensure!(
                        evidence.quote == claim.claim.quote,
                        "claim evidence does not match its quote"
                    );
                    local(evidence)?;
                }
            }
            crate::facts::parse_candidates(
                &serde_json::to_string(
                    &serde_json::json!({"facts":graph.claims.iter().map(|c|&c.claim).collect::<Vec<_>>()}),
                )?,
                chunk.text()?,
            )?;
            ensure!(
                graph.nodes.len() <= 256 && serde_json::to_vec(graph)?.len() <= 512 * 1024,
                "candidate graph bound exceeded"
            );
            let references = references.for_chunk(chunk)?;
            let metadata = serde_json::json!({"status":"candidate","profile":graph.profile,"ontology_hash":graph.ontology_hash,"source_artifact_id":chunk.document.key.source,"source_revision":chunk.document.key.revision,"source_range":chunk.range,"source_content_hash":graph.content_hash,"reference_context":references,"candidate_graph":graph,"chunk_count":0});
            self.write_candidate_artifact(chunk, &graph.profile, metadata)
                .await
        }
        pub async fn write_atomic_candidates(
            &self,
            chunk: &crate::contracts::Chunk,
            request: &crate::atomic::PreparedRequest,
            registry: &crate::atomic::Registry,
            extracted: &crate::facts::AtomicExtraction,
        ) -> Result<String> {
            chunk.text()?;
            ensure!(
                request.binding == extracted.binding && request.profile == extracted.profile,
                "atomic request binding mismatch"
            );
            let graph = crate::atomic::validate(&extracted.raw_output, &request.catalog, registry)?;
            ensure!(
                graph.document == chunk.document.key
                    && graph.content_hash == chunk.document.content_hash()
                    && serde_json::to_value(&graph)? == serde_json::to_value(&extracted.graph)?,
                "atomic graph/source mismatch"
            );
            let local = |span: &crate::contracts::SourceSpan| -> Result<()> {
                span.validate(&chunk.document)?;
                let inside =
                    span.range.start >= chunk.range.start && span.range.end <= chunk.range.end;
                let issued_context = request
                    .catalog
                    .record_anchor()
                    .is_ok_and(|record| record.range == span.range)
                    || request.catalog.is_instruction_title(span);
                ensure!(
                    inside || issued_context,
                    "atomic span outside window or issued context"
                );
                Ok(())
            };
            for entity in graph.entities.values() {
                local(&entity.anchor)?;
            }
            for assertion in &graph.assertions {
                for span in &assertion.evidence {
                    local(span)?;
                }
            }
            let metadata = json!({"status":"candidate","profile":request.profile,"source_artifact_id":chunk.document.key.source,"source_revision":chunk.document.key.revision,"source_range":chunk.range,"source_content_hash":graph.content_hash,"candidate_graph":graph,"request_binding":request.binding,"generation":extracted.raw_output,"usage":extracted.usage,"chunk_count":0});
            self.write_candidate_artifact(chunk, &request.profile, metadata)
                .await
        }
        pub async fn write_provider_properties(
            &self,
            input: &ExtractionInput,
            properties: &crate::provider_metadata::ProviderProperties,
            quality_profile: &str,
        ) -> Result<String> {
            ensure!(
                input.kind == crate::document::DocumentKind::AsanaTask
                    && properties.contract_version
                        == crate::provider_metadata::PROVIDER_METADATA_VERSION
                    && input.sources.contains(&properties.source)
                    && format!("tasks/{}", properties.task_id) == properties.source.record_id
                    && properties.object.key == properties.source.object_key,
                "provider property source mismatch"
            );
            properties
                .header
                .as_ref()
                .context("verified provider header required")
                .and_then(|header| header.validate(&input.document))
                .context(crate::facts::ExtractionReview::Input)?;
            let profile = blake3::hash(&serde_json::to_vec(&(
                crate::provider_metadata::PROVIDER_METADATA_VERSION,
                quality_profile,
                &properties.object,
                &properties.source_row_hash,
            ))?)
            .to_hex()
            .to_string();
            let chunk = crate::contracts::Chunk {
                document: input.document.clone(),
                range: *input
                    .ranges
                    .first()
                    .context("provider property parent range missing")?,
            };
            let metadata = json!({"status":"candidate","profile":profile,"source_artifact_id":input.document.key.source,"source_revision":input.document.key.revision,"source_range":chunk.range,"source_content_hash":input.document.content_hash(),"candidate_graph":properties,"chunk_count":0});
            self.write_candidate_artifact(&chunk, &profile, metadata)
                .await
        }
        pub async fn write_open_claims_candidates(
            &self,
            chunk: &crate::contracts::Chunk,
            extracted: &crate::facts::OpenClaimsExtraction,
            prepared: &crate::fact_preparation::PreparedFactSource,
            base_profile: &str,
            policy: &crate::quality::QualityConfig,
        ) -> Result<String> {
            ensure!(
                policy.format == crate::quality::ExtractionFormat::OpenClaimsV3,
                "open-claims policy required"
            );
            let request = crate::open_claims_prototype::prepare_compact(
                chunk,
                prepared,
                base_profile,
                policy.max_output_tokens.unwrap_or(4096),
            )?;
            let profile = policy.profile(base_profile, &prepared.preparation_profile)?;
            ensure!(
                extracted.profile == profile
                    && extracted.request.binding == request.binding
                    && extracted.request.profile == request.profile,
                "open candidate request/profile mismatch"
            );
            let checked = crate::open_claims_prototype::validate_partial(
                &extracted.raw_output,
                chunk,
                &request,
            )?;
            ensure!(
                checked == extracted.graph,
                "open candidate graph validation failed"
            );
            let metadata = json!({"status":"candidate", "profile":profile,
                "protocol_profile":request.profile,
                "contract_version":crate::open_claims_prototype::COMPACT_VERSION,
                "validation_policy":crate::open_claims_prototype::CLAIM_LOCAL_POLICY,
                "source_artifact_id":chunk.document.key.source,
                "source_revision":chunk.document.key.revision,"source_range":chunk.range,
                "source_content_hash":chunk.document.content_hash(),
                "request_binding":request.binding,"candidate_graph":checked,
                "generation":extracted.raw_output,"usage":extracted.usage,
                "preparation_profile":prepared.preparation_profile,
                "canonical_context":prepared.canonical_context,
                "provider_layout":prepared.provider_header,"chunk_count":0});
            self.write_candidate_artifact(chunk, &profile, metadata)
                .await
        }
        pub async fn write_quality_candidates(
            &self,
            chunk: &crate::contracts::Chunk,
            graph: &crate::quality::QualityGraph,
            prepared: &crate::fact_preparation::PreparedFactSource,
            ontology: &crate::ontology::Ontology,
            policy: &crate::quality::QualityConfig,
        ) -> Result<String> {
            ensure!(
                graph.contract_version == crate::quality::QUALITY_VERSION
                    && graph.document == chunk.document.key
                    && graph.content_hash == chunk.document.content_hash()
                    && graph.source_range == chunk.range,
                "typed candidate source identity mismatch"
            );
            let checked = crate::quality::parse_graph(
                &serde_json::to_string(
                    &json!({"facts":graph.candidates.iter().map(|candidate|candidate.proposed.as_ref().unwrap_or(&candidate.candidate)).collect::<Vec<_>>()}),
                )?,
                chunk,
                prepared,
                ontology,
                policy,
                &graph.profile,
            )?;
            ensure!(
                checked.rejections.is_empty()
                    && serde_json::to_value(&checked.candidates)?
                        == serde_json::to_value(&graph.candidates)?
                    && serde_json::to_value(&checked.mentions)?
                        == serde_json::to_value(&graph.mentions)?
                    && checked.ontology_hash == graph.ontology_hash
                    && checked.language_profile == graph.language_profile
                    && checked.context_snapshot_hash == graph.context_snapshot_hash
                    && checked.source_layout_hash == graph.source_layout_hash
                    && graph.rejections.len() <= 20
                    && graph
                        .rejections
                        .iter()
                        .all(|rejection| rejection.index < 20),
                "typed candidate graph validation failed"
            );
            let metadata = json!({"status":"candidate","profile":graph.profile,"ontology_hash":graph.ontology_hash,"source_artifact_id":chunk.document.key.source,"source_revision":chunk.document.key.revision,"source_range":chunk.range,"source_content_hash":graph.content_hash,"candidate_graph":graph,"canonical_context":prepared.canonical_context,"provider_layout":prepared.provider_header,"preparation_profile":prepared.preparation_profile,"chunk_count":0});
            self.write_candidate_artifact(chunk, &graph.profile, metadata)
                .await
        }
        async fn write_candidate_artifact(
            &self,
            chunk: &crate::contracts::Chunk,
            profile: &str,
            metadata: serde_json::Value,
        ) -> Result<String> {
            let id = Self::candidate_id(chunk, profile)?;
            let uri = chunk
                .document
                .metadata
                .uri
                .as_ref()
                .context("candidate source URI required")?;
            let parsed = reqwest::Url::parse(uri)?;
            ensure!(
                matches!(parsed.scheme(), "https" | "s3")
                    && parsed.username().is_empty()
                    && parsed.password().is_none(),
                "unsafe candidate source URI"
            );
            ensure!(
                serde_json::to_vec(&metadata)?.len() <= 1024 * 1024,
                "candidate artifact exceeds bound"
            );
            let mut tx = self.tenant(&chunk.document.key.scope).await?;
            let mut parent = QueryBuilder::<Postgres>::new("SELECT revision_id,source_uri FROM ");
            parent
                .push(&self.artifacts)
                .push(" WHERE id=$1 AND organization_id=$2 AND deleted_at IS NULL");
            let source = parent
                .build()
                .bind(&chunk.document.key.source)
                .bind(&chunk.document.key.scope)
                .fetch_optional(&mut *tx)
                .await
                .map_err(database_error)?
                .context("candidate parent source not found")?;
            ensure!(
                chunk.document.key.revision
                    == format!(
                        "pg-chunk-projection-v1/{}",
                        source
                            .try_get::<String, _>("revision_id")
                            .map_err(database_error)?
                    )
                    && source
                        .try_get::<String, _>("source_uri")
                        .map_err(database_error)?
                        == *uri,
                "candidate parent revision/URI mismatch"
            );
            let mut query = QueryBuilder::<Postgres>::new("INSERT INTO ");
            query.push(&self.artifacts).push(" (id,organization_id,source_type,source_uri,external_id,revision_id,content_hash,mime_type,title,metadata,created_at,updated_at) VALUES ($1,$2,'aurelius/candidates',$3,$4,$5,$6,'application/vnd.aurelius.candidate-graph+json','Ontology fact candidates',$7,timezone('UTC',now()),timezone('UTC',now())) ON CONFLICT (id) DO NOTHING");
            query
                .build()
                .bind(&id)
                .bind(&chunk.document.key.scope)
                .bind(uri)
                .bind(format!("candidate/{id}"))
                .bind(profile)
                .bind(chunk.document.content_hash())
                .bind(metadata)
                .execute(&mut *tx)
                .await
                .map_err(database_error)?;
            tx.commit().await.map_err(database_error)?;
            Ok(id)
        }
        pub async fn write(&self, indexed: &IndexedDocument) -> Result<StoredDocument> {
            self.write_artifact(indexed, false).await
        }
        pub async fn write_deferred(
            &self,
            document: CanonicalDocument,
            profile: &str,
        ) -> Result<StoredDocument> {
            ensure!(
                document
                    .context
                    .get("status")
                    .is_some_and(|s| s == "deferred")
                    && document
                        .context
                        .get("reason")
                        .is_some_and(|s| !s.is_empty())
                    && document
                        .context
                        .get("source_etag")
                        .is_some_and(|s| !s.is_empty())
                    && document.sections.len() == 1,
                "invalid deferred source reference"
            );
            let indexed = IndexedDocument {
                content_hash: document.content_hash()?,
                document,
                profile: profile.into(),
                chunks: vec![],
            };
            self.write_artifact(&indexed, true).await
        }
        async fn write_artifact(
            &self,
            indexed: &IndexedDocument,
            deferred: bool,
        ) -> Result<StoredDocument> {
            let document = &indexed.document;
            document.validate()?;
            ensure!(
                indexed.content_hash == document.content_hash()?,
                "indexed document hash mismatch"
            );
            ensure!(
                !indexed.profile.is_empty() && (deferred || !indexed.chunks.is_empty()),
                "empty indexed document/profile"
            );
            let mut ids = BTreeSet::new();
            for chunk in &indexed.chunks {
                ensure!(
                    !chunk.id.is_empty() && ids.insert(&chunk.id) && !chunk.text.is_empty(),
                    "invalid or duplicate chunk identity"
                );
                ensure!(
                    chunk.embedding.len() == self.dimensions
                        && chunk.embedding.iter().all(|v| v.is_finite())
                        && chunk.embedding.iter().any(|v| *v != 0.),
                    "invalid chunk embedding"
                );
                for citation in &chunk.citations {
                    ensure!(
                        citation.byte_start < citation.byte_end
                            && citation.byte_end <= chunk.text.len()
                            && chunk.text.is_char_boundary(citation.byte_start)
                            && chunk.text.is_char_boundary(citation.byte_end),
                        "invalid chunk citation range"
                    );
                }
            }
            let artifact_id = Self::artifact_id(document, &indexed.profile)?;
            let source_uri = &document
                .sections
                .first()
                .context("document source required")?
                .source
                .uri;
            let source_type = format!("aurelius/{}", document.connection_id);
            let metadata = json!({"profile":indexed.profile,"connection_id":document.connection_id,"kind":document.kind,
                "context":document.context,"modified_at":document.modified_at,"chunk_count":indexed.chunks.len(),
                "sources":document.sections.iter().map(|s|&s.source).collect::<Vec<_>>()});
            let mut tx = self.tenant(&document.organization_id).await?;
            let mut artifact_query = QueryBuilder::<Postgres>::new("INSERT INTO ");
            artifact_query.push(&self.artifacts).push(" (id,organization_id,source_type,source_uri,external_id,revision_id,content_hash,mime_type,title,metadata,created_at,updated_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,timezone('UTC',now()),timezone('UTC',now())) ON CONFLICT (id) DO NOTHING");
            artifact_query
                .build()
                .bind(&artifact_id)
                .bind(&document.organization_id)
                .bind(source_type)
                .bind(source_uri)
                .bind(&document.id)
                .bind(format!("{}/{}", indexed.content_hash, indexed.profile))
                .bind(&indexed.content_hash)
                .bind(document.context.get("mime_type"))
                .bind(&document.title)
                .bind(metadata)
                .execute(&mut *tx)
                .await
                .map_err(database_error)?;
            for (index, chunk) in indexed.chunks.iter().enumerate() {
                let id = blake3::hash(&serde_json::to_vec(&(&artifact_id, &chunk.id))?)
                    .to_hex()
                    .to_string();
                let vector = serde_json::to_string(&chunk.embedding)?;
                let source = chunk
                    .citations
                    .first()
                    .map(|c| c.source.uri.as_str())
                    .unwrap_or(source_uri);
                let mut chunk_query = QueryBuilder::<Postgres>::new("INSERT INTO ");
                chunk_query.push(&self.chunks).push(" (id,organization_id,artifact_source_id,source_path,source_hash,chunk_index,context_window_id,child_text,parent_block,context_header,metadata,embedding,created_at,updated_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$8,$9,$10,$11::text::")
                    .push(&self.vector_type).push(",timezone('UTC',now()),timezone('UTC',now())) ON CONFLICT (id) DO NOTHING");
                chunk_query.build().bind(id).bind(&document.organization_id).bind(&artifact_id).bind(source)
                    .bind(&indexed.content_hash).bind(i32::try_from(index)?).bind(&artifact_id).bind(&chunk.text)
                    .bind(&chunk.context_header).bind(json!({"profile":indexed.profile,"citations":chunk.citations,"tokens":chunk.tokens}))
                    .bind(vector).execute(&mut *tx).await.map_err(database_error)?;
            }
            tx.commit().await.map_err(database_error)?;
            Ok(StoredDocument {
                artifact_id,
                chunks: indexed.chunks.len(),
            })
        }
    }
}
#[cfg(feature = "postgres")]
pub use driver::{ExtractionInput, FactManifestEntry, PostgresSink, StoredDocument};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pool_size_is_bounded_and_defaults_to_two() {
        let mut config: PostgresConfig = serde_json::from_value(serde_json::json!({
            "url_env":"DATABASE_URL", "schema":"public", "artifacts_table":"artifacts",
            "chunks_table":"chunks"
        }))
        .unwrap();
        assert_eq!(config.pool_size, 2);
        for size in [1, 2, 4] {
            config.pool_size = size;
            config.validate().unwrap();
        }
        for size in [0, 5, u32::MAX] {
            config.pool_size = size;
            assert!(config.validate().is_err());
        }
    }
    #[test]
    fn sqlstate_survives_private_context_without_leaking_it() {
        let error = anyhow::Error::new(DatabaseFailure {
            code: "53300".into(),
        })
        .context("PRIVATE_QUERY_AND_CREDENTIAL");
        assert_eq!(
            crate::ingest::failure_code(&error),
            "stage=unknown http_status=0 timeout=false postgres_code=53300"
        );
    }
    #[test]
    fn identifiers_are_quoted_not_interpolated_as_sql() {
        assert_eq!(
            quote_identifier("custom-table").unwrap(),
            "\"custom-table\""
        );
        assert_eq!(
            quote_identifier("name\"; SELECT 1;--").unwrap(),
            "\"name\"\"; SELECT 1;--\""
        );
        assert!(quote_identifier("bad\nname").is_err());
        assert!(quote_identifier(&"a".repeat(64)).is_err());
    }
}
