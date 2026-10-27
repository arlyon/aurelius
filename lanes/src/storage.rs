use crate::{config::credential, lanes::ObjectRef};
use anyhow::{Context, Result, ensure};
use futures_util::StreamExt;
use object_store::{
    GetOptions, ObjectStore, PutMode, PutOptions, UpdateVersion, aws::AmazonS3Builder, path::Path,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageConfig {
    pub endpoint: String,
    pub region: String,
    pub access_key_env: String,
    pub secret_key_env: String,
}
impl StorageConfig {
    pub fn validate(&self) -> Result<()> {
        let url = reqwest::Url::parse(&self.endpoint)?;
        ensure!(
            url.scheme() == "https"
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none(),
            "invalid object storage endpoint"
        );
        ensure!(
            !self.region.trim().is_empty() && self.region.len() <= 100,
            "invalid storage region"
        );
        crate::config::validate_env_name(&self.access_key_env)?;
        crate::config::validate_env_name(&self.secret_key_env)?;
        Ok(())
    }
    pub fn connect(&self, bucket: &str) -> Result<Arc<dyn ObjectStore>> {
        self.validate()?;
        Ok(Arc::new(
            AmazonS3Builder::new()
                .with_endpoint(&self.endpoint)
                .with_region(&self.region)
                .with_bucket_name(bucket)
                .with_access_key_id(credential(&self.access_key_env)?)
                .with_secret_access_key(credential(&self.secret_key_env)?)
                .with_virtual_hosted_style_request(false)
                .with_allow_http(false)
                .build()?,
        ))
    }
}

pub async fn read(
    store: &dyn ObjectStore,
    reference: &ObjectRef,
    max_bytes: usize,
) -> Result<Vec<u8>> {
    reference.validate()?;
    ensure!(
        max_bytes > 0 && max_bytes <= 128 * 1024 * 1024,
        "invalid object read bound"
    );
    let result = store
        .get_opts(
            &Path::parse(&reference.key)?,
            GetOptions {
                if_match: Some(reference.version.clone()),
                ..Default::default()
            },
        )
        .await?;
    ensure!(
        result.meta.e_tag.as_deref() == Some(&reference.version)
            && result.meta.size <= max_bytes as u64,
        "object changed or exceeds bound"
    );
    let expected = result.meta.size;
    let mut stream = result.into_stream();
    let mut bytes = Vec::new();
    while let Some(part) = stream.next().await {
        let part = part?;
        ensure!(
            bytes.len() + part.len() <= max_bytes,
            "object stream exceeds bound"
        );
        bytes.extend_from_slice(&part);
    }
    ensure!(bytes.len() as u64 == expected, "incomplete object stream");
    Ok(bytes)
}

/// The caller must serialize all writers for this sync group, including direct runs.
pub async fn overwrite_exclusive(
    store: &dyn ObjectStore,
    key: &str,
    bytes: Vec<u8>,
) -> Result<ObjectRef> {
    ObjectRef {
        key: key.into(),
        version: "replacement".into(),
    }
    .validate()?;
    ensure!(
        bytes.len() <= 128 * 1024 * 1024,
        "object publication exceeds bound"
    );
    let result = store
        .put_opts(
            &Path::parse(key)?,
            bytes.into(),
            PutOptions {
                mode: PutMode::Overwrite,
                ..Default::default()
            },
        )
        .await?;
    Ok(ObjectRef {
        key: key.into(),
        version: result.e_tag.context("publication ETag required")?,
    })
}

pub async fn publish(
    store: &dyn ObjectStore,
    key: &str,
    bytes: Vec<u8>,
    previous: Option<&str>,
) -> Result<ObjectRef> {
    ObjectRef {
        key: key.into(),
        version: previous.unwrap_or("new").into(),
    }
    .validate()?;
    ensure!(
        bytes.len() <= 128 * 1024 * 1024,
        "object publication exceeds bound"
    );
    let mode = match previous {
        None => PutMode::Create,
        Some(version) => PutMode::Update(UpdateVersion {
            e_tag: Some(version.into()),
            version: None,
        }),
    };
    let result = store
        .put_opts(
            &Path::parse(key)?,
            bytes.into(),
            PutOptions {
                mode,
                ..Default::default()
            },
        )
        .await?;
    Ok(ObjectRef {
        key: key.into(),
        version: result.e_tag.context("publication ETag required")?,
    })
}
