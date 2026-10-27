use crate::lanes::{LaneError, LaneRequest};
#[cfg(feature = "postgres")]
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct IngestRequest {
    pub lane: LaneRequest,
    pub after: Option<u64>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct IngestEpoch {
    pub request: IngestRequest,
    pub sequence: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct IngestReceipt {
    pub epoch: IngestEpoch,
    pub indexed: u64,
    pub deferred: u64,
    pub ignored: u64,
}

#[derive(Debug)]
pub(crate) struct IngestStage(pub &'static str);
impl std::fmt::Display for IngestStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for IngestStage {}
pub fn failure_code(error: &anyhow::Error) -> String {
    let stage = error
        .downcast_ref::<IngestStage>()
        .map_or("unknown", |s| s.0);
    let http = error
        .chain()
        .find_map(|e| e.downcast_ref::<reqwest::Error>());
    format!(
        "stage={stage} http_status={} timeout={}",
        http.and_then(|e| e.status()).map_or(0, |s| s.as_u16()),
        http.is_some_and(|e| e.is_timeout())
    )
}

pub trait IngestActivities {
    type Error;
    fn discover(
        &mut self,
        request: IngestRequest,
    ) -> impl std::future::Future<Output = std::result::Result<Option<u64>, Self::Error>>;
    fn ingest(
        &mut self,
        epoch: IngestEpoch,
    ) -> impl std::future::Future<Output = std::result::Result<IngestReceipt, Self::Error>>;
}
pub async fn ingest_once<A: IngestActivities>(
    activities: &mut A,
    request: IngestRequest,
) -> std::result::Result<Option<IngestReceipt>, LaneError<A::Error>> {
    request.lane.validate().map_err(LaneError::Invalid)?;
    let Some(sequence) = activities
        .discover(request.clone())
        .await
        .map_err(LaneError::Activity)?
    else {
        return Ok(None);
    };
    if sequence == 0 || request.after.is_some_and(|after| sequence <= after) {
        return Err(LaneError::Invalid(anyhow::anyhow!(
            "source epoch did not advance"
        )));
    }
    let epoch = IngestEpoch { request, sequence };
    let receipt = activities
        .ingest(epoch.clone())
        .await
        .map_err(LaneError::Activity)?;
    if receipt.epoch != epoch {
        return Err(LaneError::Invalid(anyhow::anyhow!(
            "ingestion receipt changed source identity"
        )));
    }
    Ok(Some(receipt))
}

#[cfg(feature = "postgres")]
mod runtime {
    use super::*;
    use crate::{
        config::Config,
        document::{CanonicalDocument, DocumentKind, Section, SourceRef},
        embedding::Embedder,
        engine::{GroupFile, LaneEngine, RawObject},
        indexing::index_document,
        lanes::{LaneConfig, Normalizer, ObjectRef},
        mail,
        postgres::PostgresSink,
        source::SourceClient,
        storage,
        transform::{self, SlackNames, TransformResult},
        xberg::XbergClient,
    };
    use anyhow::Context;
    use futures_util::StreamExt;
    use object_store::{ObjectStore, path::Path};
    use std::{collections::BTreeMap, sync::Arc};

    #[derive(Clone)]
    pub struct PgIngestor {
        config: Arc<Config>,
        source: SourceClient,
        xberg: XbergClient,
        embedder: Embedder,
        sink: PostgresSink,
        stores: Arc<BTreeMap<String, Arc<dyn ObjectStore>>>,
    }
    impl PgIngestor {
        pub async fn new(config: Config) -> Result<Self> {
            config.validate()?;
            let embedding = config
                .embedding
                .clone()
                .context("embedding configuration required")?;
            let sink = PostgresSink::connect(
                config
                    .postgres
                    .clone()
                    .context("Postgres configuration required")?,
                embedding.dimensions,
            )
            .await?;
            let storage = config
                .storage
                .as_ref()
                .context("raw storage configuration required")?;
            let mut stores = BTreeMap::new();
            for lane in config.lanes.values() {
                if !stores.contains_key(&lane.bucket) {
                    stores.insert(lane.bucket.clone(), storage.connect(&lane.bucket)?);
                }
            }
            Self::with_storage(config, stores, sink)
        }
        pub fn with_storage(
            config: Config,
            stores: BTreeMap<String, Arc<dyn ObjectStore>>,
            sink: PostgresSink,
        ) -> Result<Self> {
            config.validate()?;
            ensure!(
                config
                    .lanes
                    .values()
                    .all(|lane| stores.contains_key(&lane.bucket)),
                "raw bucket binding missing"
            );
            let source = SourceClient::new(
                config
                    .source
                    .clone()
                    .context("source configuration required")?,
            )?;
            let xberg = XbergClient::new(
                config
                    .xberg
                    .as_ref()
                    .context("xberg configuration required")?,
            )?;
            let embedder = Embedder::new(
                config
                    .embedding
                    .clone()
                    .context("embedding configuration required")?,
            )?;
            Ok(Self {
                config: Arc::new(config),
                source,
                xberg,
                embedder,
                sink,
                stores: Arc::new(stores),
            })
        }
        #[cfg(feature = "temporal")]
        pub(crate) fn sink(&self) -> PostgresSink {
            self.sink.clone()
        }
        pub fn request(&self, name: &str) -> Result<IngestRequest> {
            Ok(IngestRequest {
                lane: self.config.lane_request(name)?,
                after: None,
            })
        }
        /// Index one real archived mail thread for an operator-selected pilot.
        /// Source job, configured stream, object ETag and original row provenance
        /// are verified exactly as for a full epoch; no synthetic lane is used.
        pub async fn index_record(
            &self,
            name: &str,
            sequence: u64,
            key: &str,
            row: usize,
        ) -> Result<String> {
            let request = self.request(name)?;
            let (lane, store) = self.context(&request)?;
            ensure!(
                matches!(lane.normalizer, Normalizer::MailThread),
                "record pilot requires a mail thread lane"
            );
            ensure!(row < 1_000_000, "pilot row exceeds bound");
            self.source.validate_connection(&lane.connection).await?;
            self.source.verify(&lane.connection, sequence).await?;
            let object = self
                .objects(lane, store, sequence)
                .await?
                .into_iter()
                .find(|object| object.reference.key == key && object.stream == "threads_details")
                .context("pilot object outside verified source epoch/stream")?;
            let bytes = storage::read(store, &object.reference, 128 * 1024 * 1024).await?;
            let output = crate::engine::decode_jsonl(&bytes)?;
            ensure!(
                output.len() <= 128 * 1024 * 1024,
                "pilot object exceeds bound"
            );
            let line = std::str::from_utf8(&output)?
                .lines()
                .nth(row)
                .context("pilot row absent")?;
            ensure!(
                line.len() <= 8 * 1024 * 1024,
                "pilot row exceeds byte bound"
            );
            let (group, record_key, record) =
                LaneEngine::record(lane, &object, row, serde_json::from_str(line)?)?;
            let file = GroupFile {
                scope: format!("{}/{}", lane.scope, lane.connection),
                group_key: serde_json::to_string(&group)?,
                group,
                records: BTreeMap::from([(record_key, record)]),
            };
            let mut records = LaneEngine::source_records(lane, &file, &object.reference)?;
            let record = records.first_mut().context("pilot record required")?;
            ensure!(!record.deleted, "pilot record was deleted");
            record.source.record_index = i64::try_from(row)?;
            let document = mail::conversation(record, &self.xberg).await?;
            let id = PostgresSink::artifact_id(&document, &request.lane.profile)?;
            self.write_document(document, &request.lane.profile).await?;
            Ok(id)
        }
        fn context(&self, request: &IngestRequest) -> Result<(&LaneConfig, &dyn ObjectStore)> {
            request.lane.validate()?;
            ensure!(
                self.config.lane_request(&request.lane.lane)? == request.lane,
                "unavailable ingestion profile"
            );
            let lane = self
                .config
                .lanes
                .get(&request.lane.lane)
                .context("unknown lane")?;
            Ok((
                lane,
                self.stores
                    .get(&lane.bucket)
                    .context("unknown raw bucket")?
                    .as_ref(),
            ))
        }
        async fn objects(
            &self,
            lane: &LaneConfig,
            store: &dyn ObjectStore,
            sequence: u64,
        ) -> Result<Vec<RawObject>> {
            self.source.verify(&lane.connection, sequence).await?;
            let prefix = format!("{}{sequence}/", lane.raw_prefix);
            let path = Path::parse(&prefix)?;
            let mut listing = store.list(Some(&path));
            let mut objects = Vec::new();
            while let Some(meta) = listing.next().await {
                let meta = meta?;
                ensure!(
                    objects.len() < 20000 && meta.location.as_ref().starts_with(&prefix),
                    "raw source listing exceeds bound/scope"
                );
                if meta.size == 0 {
                    continue;
                }
                if !matches!(lane.normalizer, Normalizer::File) {
                    let relative = meta
                        .location
                        .as_ref()
                        .strip_prefix(&prefix)
                        .context("raw path mismatch")?;
                    let stream = relative
                        .split('/')
                        .next()
                        .context("raw stream path required")?;
                    if !lane.streams.iter().any(|selected| selected == stream) {
                        continue;
                    }
                }
                let stream = LaneEngine::object_stream(lane, meta.location.as_ref())?;
                objects.push(RawObject {
                    reference: ObjectRef {
                        key: meta.location.to_string(),
                        version: meta.e_tag.context("source ETag required")?,
                    },
                    size: meta.size,
                    stream,
                    observed_at: meta.last_modified,
                });
            }
            objects.sort_by(|a, b| a.reference.key.cmp(&b.reference.key));
            Ok(objects)
        }
        async fn write_document(&self, document: CanonicalDocument, profile: &str) -> Result<()> {
            if !self
                .sink
                .contains(&document, profile)
                .await
                .context(IngestStage("pg_lookup"))?
            {
                tracing::info!(
                    profile,
                    source_bytes = document
                        .sections
                        .iter()
                        .map(|s| s.text.len())
                        .sum::<usize>(),
                    "embedding document started"
                );
                let indexed = index_document(document, profile, &self.embedder)
                    .await
                    .context(IngestStage("embedding"))?;
                self.sink
                    .write(&indexed)
                    .await
                    .context(IngestStage("pg_write"))?;
                tracing::info!(
                    profile,
                    chunks = indexed.chunks.len(),
                    "document stored in Postgres"
                );
            }
            Ok(())
        }
        async fn defer(
            &self,
            lane: &LaneConfig,
            reference: &ObjectRef,
            profile: &str,
            row: Option<usize>,
            observed_at: chrono::DateTime<chrono::Utc>,
            reason: &str,
        ) -> Result<()> {
            let id = blake3::hash(&serde_json::to_vec(&(reference, row))?)
                .to_hex()
                .to_string();
            let document = CanonicalDocument {
                id: format!("deferred/{id}"),
                organization_id: lane.scope.clone(),
                connection_id: lane.connection.clone(),
                kind: DocumentKind::GenericText,
                title: "Deferred source coverage".into(),
                modified_at: observed_at,
                sections: vec![Section {
                    text: "Deferred source reference; no extracted content".into(),
                    author: None,
                    occurred_at: None,
                    source: SourceRef {
                        stream: LaneEngine::object_stream(lane, &reference.key)?,
                        record_id: id,
                        uri: format!("s3://{}/{}", lane.bucket, reference.key),
                        object_key: reference.key.clone(),
                        record_index: i64::try_from(row.unwrap_or(0))?,
                    },
                }],
                context: BTreeMap::from([
                    ("status".into(), "deferred".into()),
                    ("reason".into(), reason.into()),
                    ("source_etag".into(), reference.version.clone()),
                    (
                        "source_row".into(),
                        row.map_or_else(|| "all".into(), |n| n.to_string()),
                    ),
                ]),
            };
            self.sink
                .write_deferred(document, profile)
                .await
                .context(IngestStage("pg_deferred_write"))?;
            Ok(())
        }
        async fn run_epoch(&self, epoch: IngestEpoch) -> Result<IngestReceipt> {
            let (lane, store) = self.context(&epoch.request)?;
            ensure!(
                epoch.sequence > epoch.request.after.unwrap_or(0),
                "source epoch did not advance"
            );
            self.source.validate_connection(&lane.connection).await?;
            self.source.verify(&lane.connection, epoch.sequence).await?;
            let sequences: Vec<_> = self
                .source
                .successful_epochs(&lane.connection)
                .await?
                .into_iter()
                .filter(|n| *n <= epoch.sequence)
                .collect();
            ensure!(
                sequences.contains(&epoch.sequence),
                "source epoch not in completed history"
            );
            let mut receipt = IngestReceipt {
                epoch: epoch.clone(),
                indexed: 0,
                deferred: 0,
                ignored: 0,
            };
            let mut groups: BTreeMap<String, GroupFile> = BTreeMap::new();
            let mut origins: BTreeMap<(String, String), (ObjectRef, usize)> = BTreeMap::new();
            let mut files: BTreeMap<String, RawObject> = BTreeMap::new();
            for sequence in sequences {
                for object in self.objects(lane, store, sequence).await? {
                    if matches!(lane.normalizer, Normalizer::File) {
                        let relative = object
                            .reference
                            .key
                            .strip_prefix(&format!("{}{sequence}/", lane.raw_prefix))
                            .context("raw path mismatch")?
                            .to_owned();
                        files.insert(relative, object);
                        continue;
                    }
                    if (matches!(lane.normalizer, Normalizer::MailThread)
                        && object.stream == "labels")
                        || (matches!(lane.normalizer, Normalizer::Task)
                            && !matches!(object.stream.as_str(), "tasks" | "stories"))
                    {
                        receipt.ignored += 1;
                        continue;
                    }
                    let bytes = storage::read(store, &object.reference, 128 * 1024 * 1024).await?;
                    let output = crate::engine::decode_jsonl(&bytes)?;
                    if output.len() > 128 * 1024 * 1024 {
                        self.defer(
                            lane,
                            &object.reference,
                            &epoch.request.lane.profile,
                            None,
                            object.observed_at,
                            "decompressed_object_bound",
                        )
                        .await?;
                        receipt.deferred += 1;
                        continue;
                    }
                    for (row, line) in std::str::from_utf8(&output)?
                        .lines()
                        .enumerate()
                        .filter(|(_, l)| !l.trim().is_empty())
                    {
                        if line.len() > 8 * 1024 * 1024 {
                            self.defer(
                                lane,
                                &object.reference,
                                &epoch.request.lane.profile,
                                Some(row),
                                object.observed_at,
                                "raw_row_bound",
                            )
                            .await?;
                            receipt.deferred += 1;
                            continue;
                        }
                        let (group, key, record) =
                            LaneEngine::record(lane, &object, row, serde_json::from_str(line)?)?;
                        let id = blake3::hash(&serde_json::to_vec(&group)?)
                            .to_hex()
                            .to_string();
                        let file = groups.entry(id.clone()).or_insert_with(|| GroupFile {
                            scope: format!("{}/{}", lane.scope, lane.connection),
                            group_key: serde_json::to_string(&group).unwrap_or_default(),
                            group,
                            records: BTreeMap::new(),
                        });
                        if let Some(prior) = file.records.get(&key) {
                            ensure!(
                                prior.version != record.version || prior.data == record.data,
                                "conflicting source revision"
                            );
                            if prior.version > record.version {
                                continue;
                            }
                        }
                        origins.insert((id, key.clone()), (object.reference.clone(), row));
                        file.records.insert(key, record);
                    }
                }
            }
            for (relative, object) in files {
                let filename = relative
                    .rsplit('/')
                    .next()
                    .context("source filename required")?;
                let extracted = self
                    .xberg
                    .extract(
                        storage::read(store, &object.reference, 64 * 1024 * 1024)
                            .await
                            .context(IngestStage("file_read"))?,
                        filename,
                        "application/octet-stream",
                    )
                    .await
                    .context(IngestStage("file_extract"))?;
                let id = blake3::hash(relative.as_bytes()).to_hex().to_string();
                let document = CanonicalDocument {
                    id: id.clone(),
                    organization_id: lane.scope.clone(),
                    connection_id: lane.connection.clone(),
                    kind: DocumentKind::File,
                    title: filename.into(),
                    sections: vec![Section {
                        text: extracted.content,
                        author: None,
                        occurred_at: None,
                        source: SourceRef {
                            stream: object.stream,
                            record_id: id,
                            uri: format!("s3://{}/{}", lane.bucket, object.reference.key),
                            object_key: object.reference.key,
                            record_index: 0,
                        },
                    }],
                    context: BTreeMap::from([
                        ("path".into(), relative),
                        ("mime_type".into(), extracted.mime_type),
                    ]),
                    modified_at: object.observed_at,
                };
                self.write_document(document, &epoch.request.lane.profile)
                    .await?;
                receipt.indexed += 1;
            }
            for (id, file) in groups {
                let first_key = file.records.keys().next().context("empty source group")?;
                let (reference, _) = origins
                    .get(&(id.clone(), first_key.clone()))
                    .context("source origin missing")?;
                let mut records = LaneEngine::source_records(lane, &file, reference)?;
                for record in &mut records {
                    let (source, row) = origins
                        .get(&(id.clone(), record.record_id.clone()))
                        .context("source record origin missing")?;
                    record.source.stream = LaneEngine::object_stream(lane, &source.key)?;
                    record.source.object_key = source.key.clone();
                    record.source.record_index = i64::try_from(*row)?;
                }
                if matches!(lane.normalizer, Normalizer::Notion) {
                    let projection = crate::notion::project(&records)?;
                    for document in projection.documents {
                        self.write_document(document, &epoch.request.lane.profile)
                            .await?;
                        receipt.indexed += 1;
                    }
                    for record_id in projection.deferred {
                        let (source, row) = origins
                            .get(&(id.clone(), record_id))
                            .context("Notion deferred origin missing")?;
                        self.defer(
                            lane,
                            source,
                            &epoch.request.lane.profile,
                            Some(*row),
                            records
                                .iter()
                                .map(|r| r.extracted_at)
                                .max()
                                .context("Notion source clock required")?,
                            "notion_unsupported_or_unlinked",
                        )
                        .await?;
                        receipt.deferred += 1;
                    }
                    continue;
                }
                let outcome = match lane.normalizer {
                    Normalizer::Conversation => transform::slack_conversation(
                        &records,
                        &file.group[0],
                        &file.group[1],
                        &SlackNames::default(),
                    )?,
                    Normalizer::Task => transform::asana_task(&records, &file.group[0])?,
                    Normalizer::MailThread => {
                        let record = records
                            .iter()
                            .max_by_key(|r| r.extracted_at)
                            .context("empty mail group")?;
                        if record.deleted {
                            TransformResult::Ignored
                        } else {
                            TransformResult::Ready(mail::conversation(record, &self.xberg).await?)
                        }
                    }
                    Normalizer::Notion | Normalizer::File => unreachable!(),
                };
                match outcome {
                    TransformResult::Ready(document) => {
                        self.write_document(document, &epoch.request.lane.profile)
                            .await?;
                        receipt.indexed += 1;
                    }
                    TransformResult::Deferred(reason) => {
                        for record in &records {
                            let (source, row) = origins
                                .get(&(id.clone(), record.record_id.clone()))
                                .context("deferred origin missing")?;
                            self.defer(
                                lane,
                                source,
                                &epoch.request.lane.profile,
                                Some(*row),
                                record.extracted_at,
                                &format!("{reason:?}"),
                            )
                            .await?;
                        }
                        receipt.deferred += 1;
                    }
                    TransformResult::Ignored => receipt.ignored += 1,
                }
            }
            Ok(receipt)
        }
    }
    impl IngestActivities for PgIngestor {
        type Error = anyhow::Error;
        async fn discover(&mut self, request: IngestRequest) -> Result<Option<u64>> {
            let (lane, _) = self.context(&request)?;
            let after = request.after.max(
                self.sink
                    .ingestion_cursor(&lane.scope, &request.lane)
                    .await?,
            );
            self.source.validate_connection(&lane.connection).await?;
            Ok(self
                .source
                .successful_epochs(&lane.connection)
                .await?
                .into_iter()
                .find(|n| *n > after.unwrap_or(0)))
        }
        async fn ingest(&mut self, epoch: IngestEpoch) -> Result<IngestReceipt> {
            let receipt = self.run_epoch(epoch).await?;
            let (lane, _) = self.context(&receipt.epoch.request)?;
            self.sink.record_ingestion(&lane.scope, &receipt).await?;
            Ok(receipt)
        }
    }
}
#[cfg(feature = "postgres")]
pub use runtime::PgIngestor;
