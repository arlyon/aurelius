use crate::{
    config::{Config, Lane},
    processor::{DIMENSIONS, Document, FINGERPRINT, vector},
};
use anyhow::{Result, ensure};
use chrono::{DateTime, Utc};
use object_store::ObjectMeta;
use pgvector::Vector;
use serde::Serialize;
use sqlx::{PgPool, Postgres, Row, Transaction};
use swiftide_core::indexing::TextNode;
use uuid::Uuid;

pub static MIGRATIONS: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

#[derive(Clone)]
pub struct Store {
    pub pool: PgPool,
}

#[derive(Clone, Debug)]
pub struct Job {
    pub id: i64,
    pub lane: String,
    pub sync_id: i64,
    pub lease: Uuid,
    pub manual: bool,
}

#[derive(Clone, Debug)]
pub struct ObjectTask {
    pub job_id: i64,
    pub lane: String,
    pub key: String,
    pub etag: String,
    pub version: Option<String>,
    pub modified_at: DateTime<Utc>,
    pub size: i64,
    pub lease: Uuid,
}

#[derive(Serialize)]
pub struct Hit {
    pub chunk_id: String,
    pub artifact_source_id: String,
    pub source_id: String,
    pub lane: String,
    pub content: String,
    pub source_uri: String,
    pub revision: String,
    pub object_key: String,
    pub record_index: i64,
    pub score: f64,
}

pub fn fingerprint(lane: &Lane) -> Result<String> {
    Ok(format!("{FINGERPRINT}/{}", lane.fingerprint()?))
}

impl Store {
    pub async fn configure_lanes(&self, config: &Config) -> Result<()> {
        let organizations: std::collections::HashSet<_> = config
            .lanes
            .iter()
            .map(|lane| lane.organization_id.as_str())
            .collect();
        for organization in organizations {
            let mut tx = self.pool.begin().await?;
            sqlx::query("SELECT set_config('lake.organization_id',$1,true)")
                .bind(organization)
                .execute(&mut *tx)
                .await?;
            sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))")
                .bind(organization)
                .execute(&mut *tx)
                .await?;
            let old =
                sqlx::query("SELECT lane,fingerprint,enabled FROM lanes WHERE organization_id=$1")
                    .bind(organization)
                    .fetch_all(&mut *tx)
                    .await?;
            let configured: Vec<_> = config
                .lanes
                .iter()
                .filter(|lane| lane.organization_id == organization)
                .collect();
            for row in &old {
                let name: String = row.get("lane");
                let next = configured.iter().find(|lane| lane.name == name);
                let changed = match next {
                    Some(lane) => {
                        !lane.enabled || fingerprint(lane)? != row.get::<String, _>("fingerprint")
                    }
                    None => true,
                };
                if changed && row.get::<bool, _>("enabled") {
                    // Revoke visibility immediately; rebuilding is not an authorization window.
                    sqlx::query("INSERT INTO document_changes (organization_id,lane,source_id,artifact_source_id,revision,content_hash,extracted_at,source_uri,deleted,metadata) SELECT organization_id,lane,source_id,active_artifact_id,active_revision,active_hash,latest_modified_at,coalesce(source_uri,''),true,'{\"scope_revoked\":true}'::jsonb FROM lake_documents WHERE organization_id=$1 AND lane=$2 AND active_artifact_id IS NOT NULL AND NOT desired_deleted").bind(organization).bind(&name).execute(&mut *tx).await?;
                    sqlx::query("UPDATE lake_documents SET desired_deleted=true WHERE organization_id=$1 AND lane=$2").bind(organization).bind(&name).execute(&mut *tx).await?;
                }
                if next.is_none() {
                    sqlx::query(
                        "UPDATE lanes SET enabled=false WHERE organization_id=$1 AND lane=$2",
                    )
                    .bind(organization)
                    .bind(&name)
                    .execute(&mut *tx)
                    .await?;
                }
            }
            for lane in configured {
                sqlx::query("INSERT INTO lanes (organization_id,lane,fingerprint,enabled) VALUES ($1,$2,$3,$4) ON CONFLICT (organization_id,lane) DO UPDATE SET fingerprint=EXCLUDED.fingerprint,enabled=EXCLUDED.enabled")
                    .bind(organization).bind(&lane.name).bind(fingerprint(lane)?).bind(lane.enabled).execute(&mut *tx).await?;
            }
            tx.commit().await?;
        }
        Ok(())
    }
    pub async fn request_reconcile(&self, organization: &str) -> Result<()> {
        sqlx::query("INSERT INTO reconcile_requests (organization_id) VALUES ($1) ON CONFLICT (organization_id) DO UPDATE SET generation=reconcile_requests.generation+1,status=CASE WHEN reconcile_requests.status='running' THEN 'running' ELSE 'pending' END,available_at=now()")
            .bind(organization).execute(&self.pool).await?;
        Ok(())
    }
    pub async fn claim_reconcile(&self) -> Result<Option<(String, i64, Uuid)>> {
        let lease = Uuid::new_v4();
        let row=sqlx::query("WITH candidate AS (SELECT organization_id FROM reconcile_requests WHERE status='pending' AND available_at<=now() OR status='running' AND lease_until<now() ORDER BY available_at FOR UPDATE SKIP LOCKED LIMIT 1) UPDATE reconcile_requests r SET status='running',lease_token=$1,lease_until=now()+interval '10 minutes' FROM candidate c WHERE r.organization_id=c.organization_id RETURNING r.organization_id,r.generation")
            .bind(lease).fetch_optional(&self.pool).await?;
        Ok(row.map(|r| (r.get("organization_id"), r.get("generation"), lease)))
    }
    pub async fn finish_reconcile(
        &self,
        organization: &str,
        generation: i64,
        lease: Uuid,
        success: bool,
    ) -> Result<()> {
        sqlx::query("UPDATE reconcile_requests SET status=CASE WHEN generation=$2 AND $4 THEN 'completed' ELSE 'pending' END,available_at=CASE WHEN $4 THEN now() ELSE now()+interval '1 minute' END,lease_until=NULL WHERE organization_id=$1 AND lease_token=$3 AND status='running'")
            .bind(organization).bind(generation).bind(lease).bind(success).execute(&self.pool).await?;
        Ok(())
    }
    pub async fn enqueue(&self, lane: &Lane, sync_id: i64, manual: bool) -> Result<()> {
        ensure!(sync_id > 0, "invalid sync ID");
        sqlx::query("INSERT INTO ingestion_jobs (lane,sync_id,fingerprint,manual_replay) VALUES ($1,$2,$3,$4) ON CONFLICT (lane,sync_id,fingerprint) DO UPDATE SET status=CASE WHEN ingestion_jobs.status='running' THEN 'running' ELSE 'pending' END, available_at=now(), error_code=NULL,attempts=CASE WHEN ingestion_jobs.status='running' THEN ingestion_jobs.attempts ELSE 0 END,manual_replay=ingestion_jobs.manual_replay OR EXCLUDED.manual_replay WHERE EXCLUDED.manual_replay")
            .bind(&lane.name).bind(sync_id).bind(fingerprint(lane)?).bind(manual).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn claim_job(&self) -> Result<Option<Job>> {
        sqlx::query("UPDATE ingestion_jobs SET status='failed',error_code='attempt_limit' WHERE status='running' AND lease_until<now() AND attempts>=10").execute(&self.pool).await?;
        let lease = Uuid::new_v4();
        let row = sqlx::query("WITH candidate AS (SELECT id FROM ingestion_jobs WHERE (status='pending' AND available_at<=now() OR status='running' AND lease_until<now()) AND attempts<10 ORDER BY available_at,id FOR UPDATE SKIP LOCKED LIMIT 1) UPDATE ingestion_jobs j SET status='running',attempts=attempts+1,lease_token=$1,lease_until=now()+interval '5 minutes' FROM candidate c WHERE j.id=c.id RETURNING j.id,j.lane,j.sync_id,j.manual_replay")
            .bind(lease).fetch_optional(&self.pool).await?;
        Ok(row.map(|r| Job {
            id: r.get("id"),
            lane: r.get("lane"),
            sync_id: r.get("sync_id"),
            manual: r.get("manual_replay"),
            lease,
        }))
    }

    pub async fn manifest(&self, job: &Job, objects: &[ObjectMeta]) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        let valid = sqlx::query("SELECT id FROM ingestion_jobs WHERE id=$1 AND lease_token=$2 AND status='running' AND lease_until>now() FOR UPDATE").bind(job.id).bind(job.lease).fetch_optional(&mut *tx).await?;
        ensure!(valid.is_some(), "manifest lease lost");
        for object in objects {
            let etag = object
                .e_tag
                .clone()
                .unwrap_or_else(|| format!("{}:{}", object.last_modified, object.size));
            sqlx::query("INSERT INTO ingestion_objects (job_id,object_key,etag,version_id,modified_at,size_bytes) VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT DO NOTHING")
                .bind(job.id).bind(object.location.to_string()).bind(etag).bind(&object.version).bind(object.last_modified).bind(i64::try_from(object.size)?).execute(&mut *tx).await?;
        }
        sqlx::query("UPDATE ingestion_jobs SET status='completed',completed_at=now(),lease_token=NULL,lease_until=NULL WHERE id=$1").bind(job.id).execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn job_failed(&self, job: &Job) -> Result<()> {
        sqlx::query("UPDATE ingestion_jobs SET status=CASE WHEN attempts>=10 THEN 'failed' ELSE 'pending' END,available_at=now()+interval '1 minute',error_code='discovery_failed',lease_until=NULL WHERE id=$1 AND lease_token=$2 AND status='running'")
            .bind(job.id).bind(job.lease).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn claim_object(&self) -> Result<Option<ObjectTask>> {
        sqlx::query("UPDATE ingestion_objects SET status='failed',error_code='attempt_limit' WHERE status='running' AND lease_until<now() AND attempts>=10").execute(&self.pool).await?;
        let lease = Uuid::new_v4();
        let row=sqlx::query("WITH candidate AS (SELECT job_id,object_key,etag FROM ingestion_objects WHERE (status='pending' AND available_at<=now() OR status='running' AND lease_until<now()) AND attempts<10 ORDER BY available_at,job_id,object_key FOR UPDATE SKIP LOCKED LIMIT 1) UPDATE ingestion_objects o SET status='running',attempts=o.attempts+1,lease_token=$1,lease_until=now()+interval '5 minutes' FROM candidate c,ingestion_jobs j WHERE o.job_id=c.job_id AND o.object_key=c.object_key AND o.etag=c.etag AND j.id=o.job_id RETURNING o.*,j.lane")
            .bind(lease).fetch_optional(&self.pool).await?;
        Ok(row.map(|r| ObjectTask {
            job_id: r.get("job_id"),
            lane: r.get("lane"),
            key: r.get("object_key"),
            etag: r.get("etag"),
            version: r.get("version_id"),
            modified_at: r.get("modified_at"),
            size: r.get("size_bytes"),
            lease,
        }))
    }

    pub async fn heartbeat(&self, task: &ObjectTask) -> Result<()> {
        let result=sqlx::query("UPDATE ingestion_objects SET lease_until=now()+interval '5 minutes' WHERE job_id=$1 AND object_key=$2 AND etag=$3 AND lease_token=$4 AND status='running' AND lease_until>now()")
            .bind(task.job_id).bind(&task.key).bind(&task.etag).bind(task.lease).execute(&self.pool).await?;
        ensure!(result.rows_affected() == 1, "object lease lost");
        Ok(())
    }

    async fn tenant<'a>(
        &'a self,
        organization: &str,
        task: &ObjectTask,
    ) -> Result<Transaction<'a, Postgres>> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("SELECT set_config('lake.organization_id',$1,true)")
            .bind(organization)
            .execute(&mut *tx)
            .await?;
        // Every document mutation for a tenant shares this lock. Cursor allocation
        // then cannot race ahead of an earlier uncommitted activation.
        sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
            .bind(organization)
            .execute(&mut *tx)
            .await?;
        let guard=sqlx::query("SELECT job_id FROM ingestion_objects WHERE job_id=$1 AND object_key=$2 AND etag=$3 AND lease_token=$4 AND status='running' AND lease_until>now() FOR UPDATE")
            .bind(task.job_id).bind(&task.key).bind(&task.etag).bind(task.lease).fetch_optional(&mut *tx).await?;
        ensure!(guard.is_some(), "object lease lost");
        Ok(tx)
    }

    pub async fn observe(&self, lane: &Lane, task: &ObjectTask, doc: &Document) -> Result<bool> {
        let mut tx = self.tenant(&lane.organization_id, task).await?;
        let row=sqlx::query("SELECT latest_modified_at,latest_source_modified_at,latest_hash,active_hash,active_fingerprint,desired_deleted FROM lake_documents WHERE organization_id=$1 AND lane=$2 AND source_id=$3 FOR UPDATE")
            .bind(&lane.organization_id).bind(&lane.name).bind(&doc.id).fetch_optional(&mut *tx).await?;
        let hash = doc.hash();
        let profile = fingerprint(lane)?;
        if let Some(row) = row {
            let previous: DateTime<Utc> = row.get("latest_modified_at");
            if doc.modified_at.timestamp_micros() < previous.timestamp_micros() {
                return Ok(false);
            }
            if !doc.deleted
                && !row.get::<bool, _>("desired_deleted")
                && let (Some(incoming), Some(current)) = (
                    doc.metadata.source_modified_at,
                    row.get::<Option<DateTime<Utc>>, _>("latest_source_modified_at"),
                )
                && incoming < current
            {
                return Ok(false);
            }
            if doc.modified_at.timestamp_micros() == previous.timestamp_micros() {
                ensure!(
                    row.get::<String, _>("latest_hash") == hash
                        || row
                            .get::<Option<String>, _>("active_fingerprint")
                            .is_some_and(|old| old != profile),
                    "ambiguous source revision"
                );
            }
            let unchanged = row.get::<bool, _>("desired_deleted") == doc.deleted
                && row.get::<Option<String>, _>("active_hash").as_deref() == Some(hash.as_str())
                && row
                    .get::<Option<String>, _>("active_fingerprint")
                    .as_deref()
                    == Some(profile.as_str());
            self.update_observation(&mut tx, lane, task, doc, &hash)
                .await?;
            tx.commit().await?;
            return Ok(!unchanged);
        }
        self.update_observation(&mut tx, lane, task, doc, &hash)
            .await?;
        tx.commit().await?;
        Ok(true)
    }

    async fn update_observation(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        lane: &Lane,
        task: &ObjectTask,
        doc: &Document,
        hash: &str,
    ) -> Result<()> {
        sqlx::query("INSERT INTO lake_documents (organization_id,lane,source_id,scope_value,latest_modified_at,latest_revision,latest_hash,latest_object_key,latest_record_index,desired_deleted,latest_source_modified_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) ON CONFLICT (organization_id,lane,source_id) DO UPDATE SET scope_value=EXCLUDED.scope_value,latest_modified_at=EXCLUDED.latest_modified_at,latest_revision=EXCLUDED.latest_revision,latest_hash=EXCLUDED.latest_hash,latest_object_key=EXCLUDED.latest_object_key,latest_record_index=EXCLUDED.latest_record_index,desired_deleted=EXCLUDED.desired_deleted,latest_source_modified_at=coalesce(EXCLUDED.latest_source_modified_at,lake_documents.latest_source_modified_at) WHERE EXCLUDED.latest_modified_at>lake_documents.latest_modified_at OR EXCLUDED.latest_modified_at=lake_documents.latest_modified_at AND EXCLUDED.latest_hash=lake_documents.latest_hash")
            .bind(&lane.organization_id).bind(&lane.name).bind(&doc.id).bind(&doc.scope_value).bind(doc.modified_at).bind(&doc.revision).bind(hash).bind(&task.key).bind(doc.record_index).bind(doc.deleted).bind(doc.metadata.source_modified_at).execute(&mut **tx).await?;
        Ok(())
    }

    pub async fn activate(
        &self,
        lane: &Lane,
        task: &ObjectTask,
        doc: &Document,
        nodes: Vec<TextNode>,
    ) -> Result<()> {
        let mut tx = self.tenant(&lane.organization_id, task).await?;
        let row=sqlx::query("SELECT latest_hash,latest_modified_at,active_artifact_id,(SELECT deleted FROM document_changes e WHERE e.organization_id=$1 AND e.lane=$2 AND e.source_id=$3 ORDER BY cursor DESC LIMIT 1) AS published_deleted FROM lake_documents WHERE organization_id=$1 AND lane=$2 AND source_id=$3 FOR UPDATE")
            .bind(&lane.organization_id).bind(&lane.name).bind(&doc.id).fetch_one(&mut *tx).await?;
        if row.get::<String, _>("latest_hash") != doc.hash()
            || row
                .get::<DateTime<Utc>, _>("latest_modified_at")
                .timestamp_micros()
                != doc.modified_at.timestamp_micros()
        {
            return Ok(());
        }
        let profile = fingerprint(lane)?;
        let artifact_id = blake3::hash(
            format!(
                "{}\0{}\0{}\0{}\0{}",
                lane.organization_id,
                lane.name,
                doc.id,
                doc.hash(),
                profile
            )
            .as_bytes(),
        )
        .to_hex()
        .to_string();
        if row
            .get::<Option<String>, _>("active_artifact_id")
            .as_deref()
            == Some(artifact_id.as_str())
            && row.get::<Option<bool>, _>("published_deleted") == Some(doc.deleted)
        {
            return Ok(());
        }
        let source_type = format!("airbyte/{}/{}", lane.connection_id, lane.stream);
        let revision_id = format!("{}/{}", doc.hash(), profile);
        let provenance = serde_json::json!({"lane":lane.name,"connection_id":lane.connection_id,"embedding_profile":crate::processor::FINGERPRINT,"bucket":lane.bucket,"object_key":task.key,"record_index":doc.record_index,"source_revision":doc.revision,"document_text":doc.text,"source_metadata":doc.metadata});
        sqlx::query("INSERT INTO \"ad-spend-memory_artifact_sources\" (id,organization_id,source_type,source_uri,external_id,revision_id,content_hash,mime_type,metadata,deleted_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,CASE WHEN $10 THEN now() ELSE NULL END) ON CONFLICT (id) DO NOTHING")
            .bind(&artifact_id).bind(&lane.organization_id).bind(source_type).bind(&doc.source_uri).bind(&doc.id).bind(revision_id).bind(doc.hash()).bind(&lane.mime_type).bind(&provenance).bind(doc.deleted).execute(&mut *tx).await?;
        // Immutable revision IDs keep existing memory evidence valid across edits and retries.
        for (index, node) in nodes.into_iter().enumerate() {
            let embedding = vector(&node)?;
            ensure!(
                embedding.len() == DIMENSIONS && embedding.iter().all(|v| v.is_finite()),
                "invalid vector"
            );
            let chunk_id = blake3::hash(format!("{artifact_id}/{index}").as_bytes())
                .to_hex()
                .to_string();
            let mut metadata = serde_json::json!({"lane":lane.name,"embedding_profile":crate::processor::FINGERPRINT,"bucket":lane.bucket,"object_key":task.key,"record_index":doc.record_index,"byte_start":node.offset});
            metadata["source_revision"] = serde_json::json!(doc.revision);
            let parent = node.chunk.clone();
            sqlx::query("INSERT INTO \"ad-spend-memory_chunks\" (id,organization_id,artifact_source_id,source_path,source_hash,chunk_index,context_window_id,child_text,parent_block,context_header,metadata,embedding) VALUES ($1,$2,$3,$4,$5,$6,$3,$7,$8,$4,$9,$10) ON CONFLICT (id) DO NOTHING")
                .bind(chunk_id).bind(&lane.organization_id).bind(&artifact_id).bind(&doc.source_uri).bind(doc.hash()).bind(i32::try_from(index)?).bind(node.chunk).bind(parent).bind(metadata).bind(Vector::from(embedding)).execute(&mut *tx).await?;
        }
        sqlx::query("UPDATE lake_documents SET active_revision=$4,active_hash=$5,active_object_key=$6,active_record_index=$7,active_fingerprint=$8,source_uri=$9,deleted=$10,active_artifact_id=$11 WHERE organization_id=$1 AND lane=$2 AND source_id=$3")
            .bind(&lane.organization_id).bind(&lane.name).bind(&doc.id).bind(&doc.revision).bind(doc.hash()).bind(&task.key).bind(doc.record_index).bind(fingerprint(lane)?).bind(&doc.source_uri).bind(doc.deleted).bind(&artifact_id).execute(&mut *tx).await?;
        sqlx::query("INSERT INTO document_changes (organization_id,lane,source_id,artifact_source_id,revision,content_hash,extracted_at,source_modified_at,source_uri,deleted,metadata) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)")
            .bind(&lane.organization_id).bind(&lane.name).bind(&doc.id).bind(&artifact_id).bind(&doc.revision).bind(doc.hash()).bind(doc.modified_at).bind(doc.metadata.source_modified_at).bind(&doc.source_uri).bind(doc.deleted).bind(serde_json::to_value(&doc.metadata)?).execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn finish_object(&self, task: &ObjectTask, success: bool) -> Result<()> {
        let result=sqlx::query("UPDATE ingestion_objects SET status=CASE WHEN $5 THEN 'completed' WHEN attempts>=10 THEN 'failed' ELSE 'pending' END,available_at=now()+interval '1 minute',error_code=CASE WHEN $5 THEN NULL ELSE 'processing_failed' END,lease_until=NULL WHERE job_id=$1 AND object_key=$2 AND etag=$3 AND lease_token=$4 AND status='running' AND lease_until>now()")
            .bind(task.job_id).bind(&task.key).bind(&task.etag).bind(task.lease).bind(success).execute(&self.pool).await?;
        ensure!(result.rows_affected() == 1, "object lease lost");
        Ok(())
    }

    pub async fn search(
        &self,
        organization: &str,
        lanes: &[String],
        fingerprints: &[String],
        query: &str,
        embedding: Vec<f32>,
    ) -> Result<Vec<Hit>> {
        ensure!(
            embedding.len() == DIMENSIONS && embedding.iter().all(|v| v.is_finite()),
            "invalid query embedding"
        );
        let mut tx = self.pool.begin().await?;
        sqlx::query("SELECT set_config('lake.organization_id',$1,true)")
            .bind(organization)
            .execute(&mut *tx)
            .await?;
        // Apply tenant and active-document filters inside both retrieval branches, before ranking.
        let rows=sqlx::query("WITH eligible AS MATERIALIZED (SELECT c.*,d.lane,d.source_id,d.source_uri,d.active_revision,d.active_object_key,d.active_record_index FROM \"ad-spend-memory_chunks\" c JOIN lake_documents d ON c.artifact_source_id=d.active_artifact_id AND c.organization_id=d.organization_id WHERE c.organization_id=$1 AND d.lane=ANY($2) AND NOT d.deleted AND NOT d.desired_deleted AND d.active_hash=d.latest_hash AND d.active_fingerprint=ANY($5)), semantic AS (SELECT lane,source_id,chunk_index,row_number() OVER (ORDER BY embedding <=> $4::vector) AS rank FROM eligible ORDER BY embedding <=> $4::vector LIMIT 40), lexical AS (SELECT lane,source_id,chunk_index,row_number() OVER (ORDER BY to_tsvector('english',child_text) <@> to_bm25query(to_tsvector('english',$3),'public.memory_chunks_lake_bm25')) AS rank FROM eligible WHERE to_tsvector('english',child_text) <@> to_bm25query(to_tsvector('english',$3),'public.memory_chunks_lake_bm25') < 0 ORDER BY to_tsvector('english',child_text) <@> to_bm25query(to_tsvector('english',$3),'public.memory_chunks_lake_bm25') LIMIT 40), scores AS (SELECT lane,source_id,chunk_index,sum(score)::float8 AS score FROM (SELECT lane,source_id,chunk_index,1.0/(60+rank) AS score FROM semantic UNION ALL SELECT lane,source_id,chunk_index,1.0/(60+rank) AS score FROM lexical) combined GROUP BY lane,source_id,chunk_index) SELECT e.*,s.score FROM scores s JOIN eligible e USING (lane,source_id,chunk_index) ORDER BY s.score DESC LIMIT 10")
            .bind(organization).bind(lanes).bind(query).bind(Vector::from(embedding)).bind(fingerprints).fetch_all(&mut *tx).await?;
        let hits = rows
            .into_iter()
            .map(|r| Hit {
                chunk_id: r.get("id"),
                artifact_source_id: r.get("artifact_source_id"),
                source_id: r.get("source_id"),
                lane: r.get("lane"),
                content: r.get("child_text"),
                source_uri: r.get("source_uri"),
                revision: r.get("active_revision"),
                object_key: r.get("active_object_key"),
                record_index: r.get("active_record_index"),
                score: r.get("score"),
            })
            .collect();
        tx.commit().await?;
        Ok(hits)
    }
}
