use crate::{
    config::Config,
    document::{CanonicalDocument, DocumentKind, Section, SourceRecord, SourceRef},
    embedding::Embedder,
    indexing::{IndexedDocument, index_document},
    lanes::{
        EpochRef, LaneActivities, LaneConfig, LaneReceipt, LaneRequest, Normalizer, ObjectRef,
        PublicationRef,
    },
    mail,
    source::SourceClient,
    storage,
    transform::{self, SlackNames, TransformResult},
    xberg::XbergClient,
};
use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use object_store::{ObjectStore, ObjectStoreExt, path::Path};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::Read, sync::Arc};

#[derive(Clone)]
pub struct LaneEngine {
    config: Arc<Config>,
    source: SourceClient,
    xberg: XbergClient,
    embedder: Embedder,
    stores: Arc<BTreeMap<String, Arc<dyn ObjectStore>>>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawObject {
    pub(crate) reference: ObjectRef,
    pub(crate) size: u64,
    pub(crate) stream: String,
    pub(crate) observed_at: DateTime<Utc>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EpochManifest {
    request: LaneRequest,
    sequence: u64,
    objects: Vec<RawObject>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupRecord {
    pub(crate) version: i64,
    pub(crate) data: Value,
    pub(crate) deleted: bool,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupFile {
    pub(crate) scope: String,
    pub(crate) group_key: String,
    pub(crate) group: Vec<String>,
    pub(crate) records: BTreeMap<String, GroupRecord>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Published {
    epoch: EpochRef,
    files: Vec<ObjectRef>,
}

/// Both Airbyte JSONL and legacy gzip JSONL are accepted, with the same
/// decompressed bound. A malformed gzip is an error, never a plaintext fallback.
pub(crate) fn decode_jsonl(bytes: &[u8]) -> Result<Vec<u8>> {
    let reader: Box<dyn Read + '_> = if bytes.starts_with(&[0x1f, 0x8b]) {
        Box::new(flate2::read::GzDecoder::new(bytes))
    } else {
        Box::new(bytes)
    };
    let mut output = Vec::new();
    reader
        .take(128 * 1024 * 1024 + 1)
        .read_to_end(&mut output)?;
    Ok(output)
}

impl LaneEngine {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        ensure!(
            config
                .lanes
                .values()
                .all(|lane| !lane.output_prefix.is_empty()),
            "legacy shadow engine requires explicit output prefixes; use PG ingestion"
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
        let storage = config
            .storage
            .as_ref()
            .context("storage configuration required")?;
        let mut stores = BTreeMap::new();
        for lane in config.lanes.values() {
            if !stores.contains_key(&lane.bucket) {
                stores.insert(lane.bucket.clone(), storage.connect(&lane.bucket)?);
            }
        }
        Ok(Self {
            config: Arc::new(config),
            source,
            xberg,
            embedder,
            stores: Arc::new(stores),
        })
    }
    pub fn request(&self, name: &str) -> Result<LaneRequest> {
        self.config.lane_request(name)
    }
    fn context(&self, request: &LaneRequest) -> Result<(&LaneConfig, &dyn ObjectStore, String)> {
        request.validate()?;
        ensure!(
            self.request(&request.lane)? == *request,
            "unavailable lane profile"
        );
        let lane = self
            .config
            .lanes
            .get(&request.lane)
            .context("unknown lane")?;
        let store = self
            .stores
            .get(&lane.bucket)
            .context("unknown lane bucket")?;
        Ok((
            lane,
            store.as_ref(),
            format!("{}{}/", lane.output_prefix, request.profile),
        ))
    }
    fn epoch_context(&self, epoch: &EpochRef) -> Result<(&LaneConfig, &dyn ObjectStore, String)> {
        let (lane, store, base) = self.context(&epoch.request)?;
        ensure!(
            epoch.sequence > 0
                && epoch.manifest.key == format!("{base}epochs/{}.json", epoch.sequence),
            "epoch reference escaped configured scope"
        );
        Ok((lane, store, base))
    }
    async fn read_value<T: serde::de::DeserializeOwned>(
        store: &dyn ObjectStore,
        reference: &ObjectRef,
    ) -> Result<T> {
        serde_json::from_slice(&storage::read(store, reference, 128 * 1024 * 1024).await?)
            .context("invalid stored artifact")
    }
    async fn once(store: &dyn ObjectStore, key: &str, value: &impl Serialize) -> Result<ObjectRef> {
        let bytes = serde_json::to_vec(value)?;
        match storage::publish(store, key, bytes.clone(), None).await {
            Ok(reference) => Ok(reference),
            Err(error) => {
                let meta = store.head(&Path::parse(key)?).await;
                let Ok(meta) = meta else {
                    return Err(error);
                };
                let reference = ObjectRef {
                    key: key.into(),
                    version: meta.e_tag.context("stored ETag required")?,
                };
                ensure!(
                    storage::read(store, &reference, 128 * 1024 * 1024).await? == bytes,
                    "conflicting immutable publication"
                );
                Ok(reference)
            }
        }
    }
    async fn current<T: serde::de::DeserializeOwned>(
        store: &dyn ObjectStore,
        key: &str,
    ) -> Result<Option<(ObjectRef, T)>> {
        match store.head(&Path::parse(key)?).await {
            Ok(meta) => {
                let r = ObjectRef {
                    key: key.into(),
                    version: meta.e_tag.context("stored ETag required")?,
                };
                let value = Self::read_value(store, &r).await?;
                Ok(Some((r, value)))
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }
    async fn defer_source(
        store: &dyn ObjectStore,
        base: &str,
        source: &ObjectRef,
        row: Option<usize>,
        reason: &str,
    ) -> Result<ObjectRef> {
        let id = blake3::hash(&serde_json::to_vec(&(source, row, reason))?)
            .to_hex()
            .to_string();
        Self::once(
            store,
            &format!("{base}documents/{id}/deferred.json"),
            &json!({"status":"deferred","source":source,"record_index":row,"reason":reason}),
        )
        .await
    }
    pub(crate) fn object_stream(lane: &LaneConfig, key: &str) -> Result<String> {
        if matches!(lane.normalizer, Normalizer::File) {
            return Ok(lane.streams[0].clone());
        }
        let parts: Vec<_> = key.split('/').collect();
        let hits: Vec<_> = lane
            .streams
            .iter()
            .filter(|s| parts.contains(&s.as_str()))
            .collect();
        ensure!(hits.len() == 1, "object has unknown or ambiguous stream");
        Ok(hits[0].clone())
    }
    pub(crate) fn record(
        lane: &LaneConfig,
        object: &RawObject,
        row: usize,
        envelope: Value,
    ) -> Result<(Vec<String>, String, GroupRecord)> {
        let data = envelope
            .get("_airbyte_data")
            .context("missing source envelope")?
            .clone();
        let version = envelope
            .get("_airbyte_extracted_at")
            .and_then(Value::as_i64)
            .context("missing source version")?;
        DateTime::<Utc>::from_timestamp_millis(version).context("invalid source clock")?;
        let get = |name: &str| {
            data.get(name)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .context("missing source identity")
        };
        let (group, key) = match lane.normalizer {
            Normalizer::Conversation => {
                let channel = get("channel_id")?;
                let ts = get("ts")?;
                let root = data
                    .get("thread_ts")
                    .and_then(Value::as_str)
                    .unwrap_or(&ts)
                    .to_owned();
                (
                    vec![channel.clone(), root],
                    serde_json::to_string(&(channel, ts))?,
                )
            }
            Normalizer::Task => {
                let gid = get("gid")?;
                let parent = if object.stream == "stories" {
                    data.get("parent_task_gid")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("unlinked-story-{gid}"))
                } else {
                    gid.clone()
                };
                (vec![parent], format!("{}/{}", object.stream, gid))
            }
            Normalizer::MailThread => (vec![get("id")?], get("id")?),
            Normalizer::Notion => (
                vec!["workspace".into()],
                format!("{}/{}", object.stream, get("id")?),
            ),
            Normalizer::File => anyhow::bail!("raw file is not a record envelope"),
        };
        let mut projected = data;
        if matches!(lane.normalizer, Normalizer::Conversation) {
            let names = [
                "channel_id",
                "ts",
                "thread_ts",
                "text",
                "user",
                "bot_id",
                "subtype",
                "edited",
                "deleted",
            ];
            projected = Value::Object(
                projected
                    .as_object()
                    .context("invalid source record")?
                    .iter()
                    .filter(|(k, _)| names.contains(&k.as_str()))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            );
        }
        let _ = row;
        Ok((
            group,
            key,
            GroupRecord {
                version,
                deleted: projected.get("deleted").and_then(Value::as_bool) == Some(true)
                    || (matches!(lane.normalizer, Normalizer::Notion)
                        && ["archived", "in_trash"]
                            .iter()
                            .any(|key| projected.get(key).and_then(Value::as_bool) == Some(true))),
                data: projected,
            },
        ))
    }
    pub(crate) fn source_records(
        lane: &LaneConfig,
        file: &GroupFile,
        reference: &ObjectRef,
    ) -> Result<Vec<SourceRecord>> {
        ensure!(
            file.scope == format!("{}/{}", lane.scope, lane.connection)
                && serde_json::to_string(&file.group)? == file.group_key,
            "group file changed scope/key"
        );
        file.records
            .iter()
            .enumerate()
            .map(|(row, (id, record))| {
                let stream = match lane.normalizer {
                    Normalizer::Conversation => "threads",
                    Normalizer::Task => {
                        id.split_once('/').context("task record stream required")?.0
                    }
                    Normalizer::MailThread => "threads_details",
                    Normalizer::Notion => {
                        id.split_once('/')
                            .context("Notion record stream required")?
                            .0
                    }
                    Normalizer::File => "file",
                };
                let uri = match lane.normalizer {
                    Normalizer::Conversation => {
                        let channel = record
                            .data
                            .get("channel_id")
                            .and_then(Value::as_str)
                            .context("channel required")?;
                        let ts = record
                            .data
                            .get("ts")
                            .and_then(Value::as_str)
                            .context("message time required")?;
                        transform::timestamp(ts)?;
                        format!(
                            "{}/archives/{channel}/p{}",
                            lane.citation_base
                                .as_deref()
                                .context("conversation citation base required")?
                                .trim_end_matches('/'),
                            ts.replace('.', "")
                        )
                    }
                    Normalizer::Task => record
                        .data
                        .get("permalink_url")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("https://app.asana.com/0/0/{}", file.group[0])),
                    Normalizer::MailThread => {
                        format!("https://mail.google.com/mail/u/0/#all/{}", file.group[0])
                    }
                    Normalizer::Notion => format!(
                        "https://www.notion.so/{}",
                        record
                            .data
                            .get("id")
                            .and_then(Value::as_str)
                            .context("Notion identity required")?
                            .replace('-', "")
                    ),
                    Normalizer::File => format!("s3://{}/{}", lane.bucket, reference.key),
                };
                Ok(SourceRecord {
                    organization_id: lane.scope.clone(),
                    connection_id: lane.connection.clone(),
                    stream: stream.into(),
                    record_id: id.clone(),
                    extracted_at: DateTime::from_timestamp_millis(record.version)
                        .context("source clock required")?,
                    deleted: record.deleted,
                    data: record.data.clone(),
                    source: SourceRef {
                        stream: stream.into(),
                        record_id: id.clone(),
                        uri,
                        object_key: reference.key.clone(),
                        record_index: i64::try_from(row)?,
                    },
                })
            })
            .collect()
    }
}
impl LaneActivities for LaneEngine {
    type Error = anyhow::Error;
    async fn discover(&mut self, request: LaneRequest) -> Result<Option<EpochRef>> {
        let (lane, store, base) = self.context(&request)?;
        self.source.validate_connection(&lane.connection).await?;
        for sequence in self.source.successful_epochs(&lane.connection).await? {
            if store
                .head(&Path::parse(format!("{base}complete/{sequence}.json"))?)
                .await
                .is_ok()
            {
                continue;
            }
            self.source.verify(&lane.connection, sequence).await?;
            let prefix = format!("{}{sequence}/", lane.raw_prefix);
            let prefix_path = Path::parse(&prefix)?;
            let mut entries = store.list(Some(&prefix_path));
            let mut objects = Vec::new();
            while let Some(entry) = entries.next().await {
                let entry = entry?;
                ensure!(
                    objects.len() < 20000 && entry.location.as_ref().starts_with(&prefix),
                    "source manifest exceeds scope/bound"
                );
                if entry.size == 0 {
                    continue;
                }
                let stream = Self::object_stream(lane, entry.location.as_ref())?;
                objects.push(RawObject {
                    reference: ObjectRef {
                        key: entry.location.to_string(),
                        version: entry.e_tag.context("source ETag required")?,
                    },
                    size: entry.size,
                    stream,
                    observed_at: entry.last_modified,
                });
            }
            ensure!(
                !objects.is_empty(),
                "successful source epoch has no objects"
            );
            objects.sort_by(|a, b| a.reference.key.cmp(&b.reference.key));
            let manifest = Self::once(
                store,
                &format!("{base}epochs/{sequence}.json"),
                &EpochManifest {
                    request: request.clone(),
                    sequence,
                    objects,
                },
            )
            .await?;
            return Ok(Some(EpochRef {
                request,
                sequence,
                manifest,
            }));
        }
        Ok(None)
    }
    async fn publish(&mut self, epoch: EpochRef) -> Result<PublicationRef> {
        let (lane, store, base) = self.epoch_context(&epoch)?;
        self.source.verify(&lane.connection, epoch.sequence).await?;
        let manifest: EpochManifest = Self::read_value(store, &epoch.manifest).await?;
        ensure!(
            manifest.request == epoch.request && manifest.sequence == epoch.sequence,
            "manifest changed source epoch"
        );
        let mut files = Vec::new();
        let mut groups: BTreeMap<String, GroupFile> = BTreeMap::new();
        for object in manifest.objects {
            ensure!(
                object
                    .reference
                    .key
                    .starts_with(&format!("{}{}/", lane.raw_prefix, epoch.sequence)),
                "source object escaped epoch"
            );
            let bytes = storage::read(store, &object.reference, 128 * 1024 * 1024).await?;
            if matches!(lane.normalizer, Normalizer::File) {
                let relative = object
                    .reference
                    .key
                    .strip_prefix(&format!("{}{}/", lane.raw_prefix, epoch.sequence))
                    .context("source path mismatch")?;
                let filename = relative
                    .rsplit('/')
                    .next()
                    .context("source filename required")?;
                let extracted = self
                    .xberg
                    .extract(bytes, filename, "application/octet-stream")
                    .await?;
                let id = blake3::hash(relative.as_bytes()).to_hex().to_string();
                let doc = CanonicalDocument {
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
                            record_id: id.clone(),
                            uri: format!("s3://{}/{}", lane.bucket, object.reference.key),
                            object_key: object.reference.key.clone(),
                            record_index: 0,
                        },
                    }],
                    context: BTreeMap::from([
                        ("path".into(), relative.into()),
                        ("mime_type".into(), extracted.mime_type),
                    ]),
                    modified_at: object.observed_at,
                };
                doc.validate()?;
                let revision = doc.content_hash()?;
                files.push(
                    Self::once(
                        store,
                        &format!("{base}documents/{id}/{revision}.json"),
                        &doc,
                    )
                    .await?,
                );
                continue;
            }
            if matches!(lane.normalizer, Normalizer::MailThread) && object.stream == "labels" {
                continue;
            }
            if matches!(lane.normalizer, Normalizer::Task)
                && !matches!(object.stream.as_str(), "tasks" | "stories")
            {
                continue;
            }
            let output = decode_jsonl(&bytes)?;
            if output.len() > 128 * 1024 * 1024 {
                files.push(
                    Self::defer_source(
                        store,
                        &base,
                        &object.reference,
                        None,
                        "decompressed_object_exceeds_bound",
                    )
                    .await?,
                );
                continue;
            }
            for (row, line) in std::str::from_utf8(&output)?
                .lines()
                .enumerate()
                .filter(|(_, line)| !line.trim().is_empty())
            {
                if line.len() > 8 * 1024 * 1024 {
                    files.push(
                        Self::defer_source(
                            store,
                            &base,
                            &object.reference,
                            Some(row),
                            "record_exceeds_bound",
                        )
                        .await?,
                    );
                    continue;
                }
                let (group, key, record) =
                    Self::record(lane, &object, row, serde_json::from_str(line)?)?;
                let id = blake3::hash(&serde_json::to_vec(&group)?)
                    .to_hex()
                    .to_string();
                let file = groups.entry(id).or_insert_with(|| GroupFile {
                    scope: format!("{}/{}", lane.scope, lane.connection),
                    group_key: serde_json::to_string(&group).unwrap_or_default(),
                    group,
                    records: BTreeMap::new(),
                });
                if let Some(old) = file.records.get(&key) {
                    ensure!(
                        old.version != record.version || old.data == record.data,
                        "conflicting source revision"
                    );
                    if old.version > record.version {
                        continue;
                    }
                }
                file.records.insert(key, record);
            }
        }
        if !groups.is_empty() {
            let reducer = self
                .config
                .reduction
                .as_ref()
                .context("native reducer required for record lanes")?;
            let mut input = Vec::new();
            let mut previous = BTreeMap::new();
            for (id, file) in &groups {
                let key = format!("{base}groups/{id}.json");
                if let Some((reference, old)) = Self::current::<GroupFile>(store, &key).await? {
                    for (record_id, new) in &file.records {
                        if let Some(prior) = old.records.get(record_id) {
                            ensure!(
                                prior.version != new.version || prior.data == new.data,
                                "conflicting replay revision"
                            );
                        }
                    }
                    input.push(serde_json::to_value(&old)?);
                    previous.insert(id.clone(), reference);
                }
                input.push(serde_json::to_value(file)?);
            }
            for value in reducer.combine(&input).await? {
                let file: GroupFile = serde_json::from_value(value)?;
                let id = blake3::hash(&serde_json::to_vec(&file.group)?)
                    .to_hex()
                    .to_string();
                ensure!(
                    groups.contains_key(&id),
                    "reducer returned unexpected group"
                );
                let key = format!("{base}groups/{id}.json");
                let payload = serde_json::to_vec(&file)?;
                let reference = if let Some(old) = previous.get(&id) {
                    if storage::read(store, old, 128 * 1024 * 1024).await? == payload {
                        old.clone()
                    } else {
                        storage::overwrite_exclusive(store, &key, payload).await?
                    }
                } else {
                    storage::publish(store, &key, payload, None).await?
                };
                let records = Self::source_records(lane, &file, &reference)?;
                let document = match lane.normalizer {
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
                            .context("mail record required")?;
                        if record.deleted {
                            TransformResult::Ignored
                        } else {
                            TransformResult::Ready(mail::conversation(record, &self.xberg).await?)
                        }
                    }
                    Normalizer::Notion => anyhow::bail!("Notion projection requires PG ingestion"),
                    Normalizer::File => unreachable!(),
                };
                let state_key = format!("{base}documents/{id}/{}.json", epoch.sequence);
                match document {
                    TransformResult::Ready(doc) => files.push(
                        Self::once(store, &state_key, &json!({"status":"ready","document":doc}))
                            .await?,
                    ),
                    TransformResult::Deferred(_) => files
                        .push(Self::once(store, &state_key, &json!({"status":"deferred"})).await?),
                    TransformResult::Ignored => files
                        .push(Self::once(store, &state_key, &json!({"status":"ignored"})).await?),
                }
            }
        }
        let manifest = Self::once(
            store,
            &format!("{base}published/{}.json", epoch.sequence),
            &Published {
                epoch: epoch.clone(),
                files,
            },
        )
        .await?;
        Ok(PublicationRef { epoch, manifest })
    }
    async fn index(&mut self, publication: PublicationRef) -> Result<LaneReceipt> {
        let (_, store, base) = self.epoch_context(&publication.epoch)?;
        ensure!(
            publication.manifest.key
                == format!("{base}published/{}.json", publication.epoch.sequence),
            "publication escaped scope"
        );
        let published: Published = Self::read_value(store, &publication.manifest).await?;
        ensure!(
            published.epoch == publication.epoch,
            "published epoch changed identity"
        );
        let mut indexed = 0;
        let mut deferred = 0;
        let mut ignored = 0;
        for reference in published.files {
            ensure!(
                reference.key.starts_with(&format!("{base}documents/")),
                "document reference escaped profile"
            );
            let value: Value = Self::read_value(store, &reference).await?;
            match value.get("status").and_then(Value::as_str) {
                Some("deferred") => {
                    deferred += 1;
                    continue;
                }
                Some("ignored") => {
                    ignored += 1;
                    continue;
                }
                _ => {}
            }
            let document: CanonicalDocument =
                serde_json::from_value(value.get("document").cloned().unwrap_or(value))?;
            let id = blake3::hash(document.id.as_bytes()).to_hex().to_string();
            let hash = document.content_hash()?;
            let key = format!("{base}index/{id}/{hash}.json");
            if let Some((_, prior)) = Self::current::<IndexedDocument>(store, &key).await? {
                ensure!(
                    prior.profile == publication.epoch.request.profile
                        && prior.content_hash == hash,
                    "cached index profile changed"
                );
            } else {
                let result =
                    index_document(document, &publication.epoch.request.profile, &self.embedder)
                        .await?;
                Self::once(store, &key, &result).await?;
            }
            indexed += 1;
        }
        let receipt_key = format!("{base}receipts/{}.json", publication.epoch.sequence);
        let receipt=Self::once(store,&receipt_key,&json!({"publication":publication,"indexed":indexed,"deferred":deferred,"ignored":ignored})).await?;
        Ok(LaneReceipt {
            publication,
            receipt,
            indexed,
            deferred,
            ignored,
        })
    }
    async fn acknowledge(&mut self, receipt: LaneReceipt) -> Result<()> {
        let (_, store, base) = self.epoch_context(&receipt.publication.epoch)?;
        ensure!(
            receipt.receipt.key
                == format!("{base}receipts/{}.json", receipt.publication.epoch.sequence),
            "receipt escaped scope"
        );
        let stored: Value = Self::read_value(store, &receipt.receipt).await?;
        ensure!(
            stored.get("publication") == Some(&serde_json::to_value(&receipt.publication)?),
            "receipt publication mismatch"
        );
        Self::once(
            store,
            &format!("{base}complete/{}.json", receipt.publication.epoch.sequence),
            &receipt,
        )
        .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_jsonl_and_gzip_decode_identically_and_corruption_fails() -> Result<()> {
        use std::io::Write;
        let raw = b"{\"_airbyte_data\":{\"id\":\"fixture\"}}\n";
        let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gzip.write_all(raw)?;
        assert_eq!(decode_jsonl(raw)?, raw);
        assert_eq!(decode_jsonl(&gzip.finish()?)?, raw);
        assert!(decode_jsonl(&[0x1f, 0x8b, 0x00]).is_err());
        Ok(())
    }
    #[tokio::test]
    async fn deferred_source_retains_exact_revision_and_record_without_payload() -> Result<()> {
        let store = object_store::memory::InMemory::new();
        let source = ObjectRef {
            key: "scope/raw/27/thread.jsonl.gz".into(),
            version: "source-etag".into(),
        };
        let reference = LaneEngine::defer_source(
            &store,
            "scope/profile/",
            &source,
            Some(42),
            "record_exceeds_bound",
        )
        .await?;
        let replay = LaneEngine::defer_source(
            &store,
            "scope/profile/",
            &source,
            Some(42),
            "record_exceeds_bound",
        )
        .await?;
        ensure!(reference == replay);
        let state: Value = LaneEngine::read_value(&store, &reference).await?;
        ensure!(state["status"] == "deferred" && state["record_index"] == 42);
        ensure!(state["source"] == serde_json::to_value(source)?);
        ensure!(state.get("document").is_none() && state.get("text").is_none());
        Ok(())
    }
}
