use crate::{
    airbyte::Airbyte,
    config::Config,
    processor::{MAX_OBJECT_BYTES, chunks, parse_batch},
    store::{Job, ObjectTask, Store},
};
use anyhow::{Context, Result, ensure};
use futures::StreamExt;
use object_store::{
    ClientOptions, GetOptions, ObjectStore, RetryConfig, aws::AmazonS3Builder, path::Path,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

#[derive(Clone)]
pub struct Worker {
    pub config: Arc<Config>,
    pub store: Store,
    pub airbyte: Option<Airbyte>,
    stores: Arc<HashMap<String, Arc<dyn ObjectStore>>>,
}
fn same_modified_second(
    actual: chrono::DateTime<chrono::Utc>,
    listed: chrono::DateTime<chrono::Utc>,
) -> bool {
    actual.timestamp() == listed.timestamp()
}

impl Worker {
    pub fn new(config: Config, store: Store, airbyte: Option<Airbyte>) -> Result<Self> {
        let endpoint = std::env::var("S3_ENDPOINT")?;
        ensure!(endpoint.starts_with("https://"), "S3 requires HTTPS");
        let mut stores: HashMap<String, Arc<dyn ObjectStore>> = HashMap::new();
        for lane in config.lanes.iter().filter(|l| l.enabled) {
            if !stores.contains_key(&lane.bucket) {
                let backend = AmazonS3Builder::new()
                    .with_bucket_name(&lane.bucket)
                    .with_endpoint(&endpoint)
                    .with_region("us-east-1")
                    .with_access_key_id(std::env::var("AWS_ACCESS_KEY_ID")?)
                    .with_secret_access_key(std::env::var("AWS_SECRET_ACCESS_KEY")?)
                    .with_virtual_hosted_style_request(false)
                    .with_allow_http(false)
                    .with_client_options(ClientOptions::new().with_timeout(Duration::from_secs(30)))
                    .with_retry(RetryConfig {
                        max_retries: 2,
                        retry_timeout: Duration::from_secs(60),
                        ..Default::default()
                    })
                    .build()?;
                stores.insert(lane.bucket.clone(), Arc::new(backend));
            }
        }
        Ok(Self {
            config: Arc::new(config),
            store,
            airbyte,
            stores: Arc::new(stores),
        })
    }
    pub async fn request_daily(&self) -> Result<()> {
        let organizations: HashSet<_> = self
            .config
            .lanes
            .iter()
            .filter(|l| l.enabled)
            .map(|l| l.organization_id.as_str())
            .collect();
        for organization in organizations {
            self.store.request_reconcile(organization).await?;
        }
        Ok(())
    }
    async fn reconcile(&self, organization: &str) -> Result<()> {
        let airbyte = self
            .airbyte
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Airbyte API unavailable in worker-only mode"))?;
        let connections: HashSet<_> = self
            .config
            .lanes
            .iter()
            .filter(|l| l.enabled && l.organization_id == organization)
            .map(|l| l.connection_id.as_str())
            .collect();
        for connection in connections {
            airbyte.validate_connection(connection).await?;
            for id in airbyte.jobs(connection).await? {
                for lane in self.config.connection(connection)? {
                    self.store.enqueue(lane, id, false).await?;
                }
            }
        }
        Ok(())
    }
    pub async fn reconcile_next(&self) -> Result<bool> {
        let Some((organization, generation, lease)) = self.store.claim_reconcile().await? else {
            return Ok(false);
        };
        let success = tokio::time::timeout(Duration::from_secs(540), self.reconcile(&organization))
            .await
            .is_ok_and(|r| r.is_ok());
        if !success {
            tracing::error!(code="reconciliation_failed",organization_id=%organization);
        }
        self.store
            .finish_reconcile(&organization, generation, lease, success)
            .await?;
        Ok(true)
    }
    async fn discover(&self, job: &Job) -> Result<()> {
        let lane = self.config.lane(&job.lane)?;
        if !job.manual {
            self.airbyte
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Airbyte API unavailable"))?
                .verify(&lane.connection_id, job.sync_id)
                .await?;
        }
        let prefix = lane.batch_prefix(job.sync_id)?;
        let store = self
            .stores
            .get(&lane.bucket)
            .ok_or_else(|| anyhow::anyhow!("unknown bucket"))?;
        let path = Path::from(prefix.clone());
        let mut stream = store.list(Some(&path));
        let mut objects = Vec::new();
        while let Some(object) = stream.next().await {
            let object = object?;
            ensure!(
                object.location.as_ref().starts_with(&prefix),
                "object escaped batch prefix"
            );
            if object.location.as_ref().ends_with(".jsonl.gz") {
                ensure!(objects.len() < 20000, "batch exceeds manifest bound");
                objects.push(object);
            }
        }
        self.store.manifest(job, &objects).await?;
        tracing::info!(lane=%lane.name,sync_id=job.sync_id,objects=objects.len(),"manifest persisted");
        Ok(())
    }
    pub async fn discover_next(&self) -> Result<bool> {
        let Some(job) = self.store.claim_job().await? else {
            return Ok(false);
        };
        let success = tokio::time::timeout(Duration::from_secs(240), self.discover(&job))
            .await
            .is_ok_and(|r| r.is_ok());
        if !success {
            tracing::error!(lane=%job.lane,sync_id=job.sync_id,code="discovery_failed");
            self.store.job_failed(&job).await?;
        }
        Ok(true)
    }
    async fn process(&self, task: &ObjectTask) -> Result<()> {
        let lane = self.config.lane(&task.lane)?;
        ensure!(
            task.key.starts_with(&lane.root)
                && task.size >= 0
                && usize::try_from(task.size)? <= MAX_OBJECT_BYTES,
            "object outside scope or size limit"
        );
        let backend = self
            .stores
            .get(&lane.bucket)
            .ok_or_else(|| anyhow::anyhow!("unknown bucket"))?;
        let result = backend
            .get_opts(
                &Path::from(task.key.clone()),
                GetOptions {
                    if_match: Some(task.etag.clone()),
                    version: task.version.clone(),
                    ..Default::default()
                },
            )
            .await
            .context("s3_fetch")?;
        ensure!(
            i64::try_from(result.meta.size)? == task.size
                // IDrive LIST has milliseconds; HTTP Last-Modified has only whole seconds.
                && same_modified_second(result.meta.last_modified, task.modified_at),
            "object changed since discovery"
        );
        let etag = result
            .meta
            .e_tag
            .clone()
            .unwrap_or_else(|| format!("{}:{}", result.meta.last_modified, result.meta.size));
        ensure!(etag == task.etag, "object changed since discovery");
        let bytes = result.bytes().await.context("s3_read")?;
        let parser_lane = lane.clone();
        let documents = tokio::task::spawn_blocking(move || parse_batch(&bytes, &parser_lane))
            .await
            .context("parse_task")?
            .context("parse_records")?;
        let count = documents.len();
        for doc in documents {
            if self
                .store
                .observe(lane, task, &doc)
                .await
                .context("observe")?
            {
                let nodes = chunks(&doc, lane).await.context("embedding")?;
                self.store
                    .activate(lane, task, &doc, nodes)
                    .await
                    .context("persist")?;
            }
        }
        tracing::info!(lane=%lane.name,records=count,"object processed");
        Ok(())
    }
    pub async fn index_next(&self) -> Result<bool> {
        let Some(task) = self.store.claim_object().await? else {
            return Ok(false);
        };
        let process = self.process(&task);
        tokio::pin!(process);
        let mut heartbeat = tokio::time::interval(Duration::from_secs(30));
        let success = loop {
            tokio::select! {
                result=&mut process=>{
                    match result {
                        Ok(()) => break true,
                        Err(error) => {
                            let db_code = error.chain().find_map(|cause| cause.downcast_ref::<sqlx::Error>())
                                .and_then(|error| error.as_database_error())
                                .and_then(|error| error.code()).map(|code| code.into_owned());
                            // Stage labels are static; never log source bodies or upstream responses.
                            tracing::error!(lane=%task.lane, stage=%error, db_code, code="object_stage_failed");
                            break false;
                        }
                    }
                },
                _=heartbeat.tick()=>{self.store.heartbeat(&task).await?;}
            }
        };
        if !success {
            tracing::error!(lane=%task.lane,code="processing_failed");
        }
        self.store.finish_object(&task, success).await?;
        if success {
            // Lakebase BM25 refreshes corpus statistics at VACUUM, not at insert time.
            sqlx::query("VACUUM (ANALYZE) public.\"ad-spend-memory_chunks\"")
                .execute(&self.store.pool)
                .await?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn idrive_list_milliseconds_match_http_seconds_but_not_changed_second() {
        let listed = chrono::DateTime::from_timestamp_millis(1791844154722).unwrap();
        let header = chrono::DateTime::from_timestamp(1791844154, 0).unwrap();
        assert!(same_modified_second(header, listed));
        assert!(!same_modified_second(
            header + chrono::Duration::seconds(1),
            listed
        ));
    }
}
