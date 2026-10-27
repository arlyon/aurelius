use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::future::Future;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaneConfig {
    pub scope: String,
    pub connection: String,
    pub normalizer: Normalizer,
    pub bucket: String,
    pub raw_prefix: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub output_prefix: String,
    pub streams: Vec<String>,
    pub citation_base: Option<String>,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Normalizer {
    Conversation,
    Task,
    MailThread,
    Notion,
    File,
}
impl LaneConfig {
    pub fn validate(&self) -> Result<()> {
        for value in [&self.scope, &self.connection, &self.bucket] {
            ensure!(
                !value.trim().is_empty()
                    && value.len() <= 200
                    && !value.chars().any(char::is_control),
                "invalid lane identity"
            );
        }
        ensure!(
            !self.streams.is_empty() && self.streams.len() <= 32,
            "lane streams required"
        );
        if let Some(base) = &self.citation_base {
            let uri = reqwest::Url::parse(base)?;
            ensure!(
                uri.scheme() == "https"
                    && uri.username().is_empty()
                    && uri.password().is_none()
                    && uri.query().is_none()
                    && uri.fragment().is_none(),
                "invalid citation base"
            );
        }
        for stream in &self.streams {
            ensure!(
                !stream.is_empty()
                    && stream.len() <= 200
                    && stream
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.'))
                    && stream != "."
                    && stream != "..",
                "invalid lane stream"
            );
        }
        for prefix in std::iter::once(&self.raw_prefix)
            .chain((!self.output_prefix.is_empty()).then_some(&self.output_prefix))
        {
            ensure!(
                !prefix.starts_with('/')
                    && prefix.ends_with('/')
                    && prefix.len() <= 1000
                    && !prefix.contains('\\')
                    && !prefix.chars().any(char::is_control)
                    && prefix
                        .trim_end_matches('/')
                        .split('/')
                        .all(|part| !part.is_empty() && part != "." && part != ".."),
                "invalid lane object prefix"
            );
        }
        ensure!(
            self.output_prefix.is_empty()
                || (!self.output_prefix.starts_with(&self.raw_prefix)
                    && !self.raw_prefix.starts_with(&self.output_prefix)),
            "source and output prefixes overlap"
        );
        Ok(())
    }
    pub fn profile(&self, ontology_hash: &str) -> Result<String> {
        self.validate()?;
        Ok(blake3::hash(&serde_json::to_vec(&(
            "lane-files-v1",
            self,
            ontology_hash,
        ))?)
        .to_hex()
        .to_string())
    }
}

// Workflow inputs/results carry immutable handles, not downloaded files or credentials.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LaneRequest {
    pub lane: String,
    pub profile: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObjectRef {
    pub key: String,
    pub version: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EpochRef {
    pub request: LaneRequest,
    pub sequence: u64,
    pub manifest: ObjectRef,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PublicationRef {
    pub epoch: EpochRef,
    pub manifest: ObjectRef,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LaneReceipt {
    pub publication: PublicationRef,
    pub receipt: ObjectRef,
    pub indexed: u64,
    pub deferred: u64,
    pub ignored: u64,
}
impl LaneRequest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.lane.is_empty()
                && self.lane.len() <= 100
                && self
                    .lane
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-')),
            "invalid lane name"
        );
        ensure!(
            self.profile.len() == 64
                && self
                    .profile
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
            "invalid lane profile"
        );
        Ok(())
    }
}
impl ObjectRef {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.key.is_empty()
                && self.key.len() <= 2048
                && !self.key.starts_with('/')
                && !self.key.contains('\\')
                && !self.key.chars().any(char::is_control)
                && self
                    .key
                    .split('/')
                    .all(|part| !part.is_empty() && part != "." && part != "..")
                && !self.version.is_empty()
                && self.version.len() <= 512
                && !self.version.chars().any(char::is_control),
            "invalid immutable object reference"
        );
        Ok(())
    }
}

pub trait LaneActivities {
    type Error;
    fn discover(
        &mut self,
        request: LaneRequest,
    ) -> impl Future<Output = Result<Option<EpochRef>, Self::Error>>;
    fn publish(
        &mut self,
        epoch: EpochRef,
    ) -> impl Future<Output = Result<PublicationRef, Self::Error>>;
    fn index(
        &mut self,
        publication: PublicationRef,
    ) -> impl Future<Output = Result<LaneReceipt, Self::Error>>;
    fn acknowledge(
        &mut self,
        receipt: LaneReceipt,
    ) -> impl Future<Output = Result<(), Self::Error>>;
}
#[derive(Debug)]
pub enum LaneError<E> {
    Invalid(anyhow::Error),
    Activity(E),
}

// One ordinary function, used unchanged by direct hosts and the native Temporal binding.
pub async fn index_lane<A: LaneActivities>(
    activities: &mut A,
    request: LaneRequest,
) -> Result<Option<LaneReceipt>, LaneError<A::Error>> {
    request.validate().map_err(LaneError::Invalid)?;
    let Some(epoch) = activities
        .discover(request.clone())
        .await
        .map_err(LaneError::Activity)?
    else {
        return Ok(None);
    };
    let validate = || -> Result<()> {
        ensure!(
            epoch.request == request && epoch.sequence > 0,
            "source epoch changed lane/profile"
        );
        epoch.manifest.validate()
    };
    validate().map_err(LaneError::Invalid)?;
    let publication = activities
        .publish(epoch.clone())
        .await
        .map_err(LaneError::Activity)?;
    publication
        .manifest
        .validate()
        .map_err(LaneError::Invalid)?;
    ensure_same_epoch(&epoch, &publication.epoch).map_err(LaneError::Invalid)?;
    let receipt = activities
        .index(publication.clone())
        .await
        .map_err(LaneError::Activity)?;
    receipt.receipt.validate().map_err(LaneError::Invalid)?;
    if receipt.publication != publication {
        return Err(LaneError::Invalid(anyhow::anyhow!(
            "index receipt changed publication"
        )));
    }
    activities
        .acknowledge(receipt.clone())
        .await
        .map_err(LaneError::Activity)?;
    Ok(Some(receipt))
}
fn ensure_same_epoch(expected: &EpochRef, actual: &EpochRef) -> Result<()> {
    ensure!(expected == actual, "publication changed source epoch");
    Ok(())
}
