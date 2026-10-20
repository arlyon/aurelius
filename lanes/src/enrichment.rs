use crate::lanes::LaneError;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FactRequest {
    pub scope: String,
    pub profile: String,
}
impl FactRequest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.scope.trim().is_empty()
                && self.scope.len() <= 512
                && !self.scope.chars().any(char::is_control),
            "invalid fact scope"
        );
        ensure!(
            self.profile.len() == 64 && self.profile.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid fact profile"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FactSource {
    pub request: FactRequest,
    pub artifact_id: String,
}
impl FactSource {
    pub fn validate(&self) -> Result<()> {
        self.request.validate()?;
        ensure!(
            !self.artifact_id.is_empty()
                && self.artifact_id.len() <= 512
                && !self.artifact_id.chars().any(char::is_control),
            "invalid fact source reference"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FactStatus {
    Complete,
    Deferred,
    AlreadyComplete,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FactReceipt {
    pub source: FactSource,
    pub status: FactStatus,
    pub stored_windows: usize,
    pub skipped_windows: usize,
    pub review_windows: usize,
    pub new_claims: usize,
}
pub trait FactActivities {
    type Error;
    fn discover(
        &mut self,
        request: FactRequest,
    ) -> impl std::future::Future<Output = Result<Option<FactSource>, Self::Error>>;
    fn process(
        &mut self,
        source: FactSource,
    ) -> impl std::future::Future<Output = Result<FactReceipt, Self::Error>>;
}
pub async fn enrich_once<A: FactActivities>(
    activities: &mut A,
    request: FactRequest,
) -> Result<Option<FactReceipt>, LaneError<A::Error>> {
    request.validate().map_err(LaneError::Invalid)?;
    let Some(source) = activities
        .discover(request.clone())
        .await
        .map_err(LaneError::Activity)?
    else {
        return Ok(None);
    };
    source.validate().map_err(LaneError::Invalid)?;
    if source.request != request {
        return Err(LaneError::Invalid(anyhow::anyhow!(
            "fact discovery changed scope/profile"
        )));
    }
    let receipt = activities
        .process(source.clone())
        .await
        .map_err(LaneError::Activity)?;
    if receipt.source != source {
        return Err(LaneError::Invalid(anyhow::anyhow!(
            "fact receipt changed source identity"
        )));
    }
    Ok(Some(receipt))
}

#[cfg(feature = "postgres")]
mod runtime {
    use super::*;
    use crate::{
        config::Config,
        contracts::Chunk,
        facts::{ChatExtractor, ExtractionReview},
        postgres::PostgresSink,
    };
    use anyhow::Context;
    use std::collections::BTreeMap;
    #[derive(Clone)]
    pub struct FactEngine {
        sink: PostgresSink,
        extractor: ChatExtractor,
        profile: String,
        #[cfg(feature = "language")]
        preparer: Option<crate::fact_preparation::FactPreparer>,
        #[cfg(feature = "language")]
        provider_metadata: Option<crate::provider_metadata::ProviderMetadataReader>,
        config: Config,
        profiles: BTreeMap<String, Vec<String>>,
    }
    fn extraction_profile(config: &Config, base: &str) -> Result<String> {
        if let Some(policy) = &config.fact_quality {
            #[cfg(feature = "language")]
            return policy.profile(
                base,
                &crate::fact_preparation::FactPreparer::profile_for(config)?,
            );
            #[cfg(not(feature = "language"))]
            {
                let _ = policy;
                anyhow::bail!("typed fact extraction requires the language build feature");
            }
        }
        ensure!(
            config.language.is_none() && config.fact_context.is_none(),
            "fact preparation configured without a typed quality policy; refusing legacy extraction fallback"
        );
        Ok(base.into())
    }
    impl FactEngine {
        pub async fn new(config: Config) -> Result<Self> {
            let sink = PostgresSink::connect(
                config
                    .postgres
                    .clone()
                    .context("Postgres configuration required")?,
                config
                    .embedding
                    .as_ref()
                    .context("embedding configuration required")?
                    .dimensions,
            )
            .await?;
            Self::with_sink(config, sink)
        }
        pub fn with_run_budget(mut self, budget: crate::run_budget::RunBudget) -> Self {
            self.extractor = self.extractor.with_run_budget(budget);
            self
        }
        pub fn with_sink(config: Config, sink: PostgresSink) -> Result<Self> {
            Self::with_reader(config, sink, None)
        }
        pub fn with_storage(
            config: Config,
            sink: PostgresSink,
            stores: BTreeMap<String, std::sync::Arc<dyn object_store::ObjectStore>>,
        ) -> Result<Self> {
            let reader = crate::provider_metadata::ProviderMetadataReader::with_storage(
                config.clone(),
                stores,
            )?;
            Self::with_reader(config, sink, Some(reader))
        }
        fn with_reader(
            config: Config,
            sink: PostgresSink,
            reader: Option<crate::provider_metadata::ProviderMetadataReader>,
        ) -> Result<Self> {
            config.validate()?;
            #[cfg(not(feature = "language"))]
            let _ = reader;
            let chat = config
                .chat
                .as_ref()
                .context("chat configuration required")?;
            ensure!(
                chat.gateway.is_some(),
                "background facts require an authenticated gateway"
            );
            let extractor = ChatExtractor::new(chat, config.ontology.clone())?;
            let profile = extraction_profile(&config, &extractor.profile)?;
            #[cfg(feature = "language")]
            let preparer = if config.fact_quality.is_some() {
                Some(crate::fact_preparation::FactPreparer::with_sink(
                    &config,
                    sink.clone(),
                )?)
            } else {
                None
            };
            #[cfg(feature = "language")]
            let provider_metadata = if config.fact_quality.is_some() {
                match reader {
                    Some(reader) => Some(reader),
                    None => crate::provider_metadata::ProviderMetadataReader::new(&config)?,
                }
            } else {
                None
            };
            let mut profiles: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for (name, lane) in &config.lanes {
                profiles
                    .entry(lane.scope.clone())
                    .or_default()
                    .push(config.lane_request(name)?.profile);
            }
            ensure!(
                !profiles.is_empty(),
                "fact scopes require configured source lanes"
            );
            Ok(Self {
                sink,
                extractor,
                profile,
                #[cfg(feature = "language")]
                preparer,
                #[cfg(feature = "language")]
                provider_metadata,
                config,
                profiles,
            })
        }
        pub fn requests(config: &Config) -> Result<Vec<FactRequest>> {
            config.validate()?;
            let chat = config
                .chat
                .as_ref()
                .context("chat configuration required")?;
            ensure!(
                chat.gateway.is_some(),
                "background facts require an authenticated gateway"
            );
            let extractor = ChatExtractor::new(chat, config.ontology.clone())?;
            let profile = extraction_profile(config, &extractor.profile)?;
            let scopes: std::collections::BTreeSet<_> = config
                .lanes
                .values()
                .map(|lane| lane.scope.clone())
                .collect();
            ensure!(
                !scopes.is_empty(),
                "fact scopes require configured source lanes"
            );
            Ok(scopes
                .into_iter()
                .map(|scope| FactRequest {
                    scope,
                    profile: profile.clone(),
                })
                .collect())
        }
        pub fn request(&self, scope: &str) -> Result<FactRequest> {
            let request = FactRequest {
                scope: scope.into(),
                profile: self.profile.clone(),
            };
            self.validate(&request)?;
            Ok(request)
        }
        fn validate(&self, request: &FactRequest) -> Result<()> {
            request.validate()?;
            ensure!(
                request.profile == self.profile && self.profiles.contains_key(&request.scope),
                "stale or unconfigured fact scope/profile"
            );
            Ok(())
        }
        #[cfg(feature = "language")]
        async fn run_quality(
            &self,
            source: FactSource,
            mut receipt: FactReceipt,
            mut input: crate::postgres::ExtractionInput,
        ) -> Result<FactReceipt> {
            let provider_properties = if input.kind == crate::document::DocumentKind::AsanaTask {
                let properties = self
                    .provider_metadata
                    .as_ref()
                    .context("deterministic task metadata reader required")?
                    .read(&input)
                    .await?
                    .context("task metadata missing")?;
                self.sink
                    .write_provider_properties(&input, &properties, &self.profile)
                    .await?;
                input.document.metadata.authored_at = properties
                    .authored_at
                    .map(|time| time.with_timezone(&chrono::Utc));
                input.document.metadata.modified_at = properties
                    .modified_at
                    .map(|time| time.with_timezone(&chrono::Utc));
                input.document.metadata.author = properties
                    .author
                    .as_ref()
                    .map(|author| format!("{}:{}", author.provider, author.id));
                Some(properties)
            } else {
                None
            };
            let policy = self
                .config
                .fact_quality
                .as_ref()
                .context("typed fact policy required")?;
            let prepared = match self
                .preparer
                .as_ref()
                .context("typed fact preparer required")?
                .prepare(input.document.clone(), policy.context_as_of)
                .await
                .and_then(|prepared| match &provider_properties {
                    Some(properties) => prepared
                        .with_provider_properties(&input.document, properties)
                        .context(ExtractionReview::Input),
                    None => Ok(prepared),
                }) {
                Ok(prepared) => prepared,
                Err(error) if error.downcast_ref::<ExtractionReview>().is_some() => {
                    self.sink
                        .record_fact_status(
                            &source.request.scope,
                            &source.artifact_id,
                            &source.request.profile,
                            None,
                            "input_review",
                        )
                        .await?;
                    receipt.status = FactStatus::Deferred;
                    return Ok(receipt);
                }
                Err(error) => return Err(error),
            };
            for range in input.ranges {
                let chunk = Chunk {
                    document: input.document.clone(),
                    range,
                };
                if self.sink.candidate_exists(&chunk, &self.profile).await? {
                    receipt.skipped_windows += 1;
                    continue;
                }
                if self
                    .sink
                    .fact_status(
                        &source.request.scope,
                        &source.artifact_id,
                        &self.profile,
                        Some(range),
                    )
                    .await?
                    .is_some()
                {
                    receipt.review_windows += 1;
                    continue;
                }
                match self
                    .extractor
                    .extract_quality(&chunk, &prepared, policy, &self.profile)
                    .await
                {
                    Ok(graph) => {
                        self.sink
                            .write_quality_candidates(
                                &chunk,
                                &graph,
                                &prepared,
                                &self.config.ontology,
                                policy,
                            )
                            .await?;
                        receipt.stored_windows += 1;
                        receipt.new_claims += graph.candidates.len();
                        if !graph.rejections.is_empty() {
                            self.sink
                                .record_fact_status(
                                    &source.request.scope,
                                    &source.artifact_id,
                                    &self.profile,
                                    Some(range),
                                    "output_rejected",
                                )
                                .await?;
                            receipt.review_windows += 1;
                        }
                    }
                    Err(error) => {
                        match error.downcast_ref::<ExtractionReview>() {
                            Some(ExtractionReview::Input) => {
                                self.sink
                                    .record_fact_status(
                                        &source.request.scope,
                                        &source.artifact_id,
                                        &self.profile,
                                        Some(range),
                                        "input_review",
                                    )
                                    .await?;
                            }
                            Some(ExtractionReview::Output) => {
                                self.sink
                                    .record_fact_review(
                                        &source.request.scope,
                                        &source.artifact_id,
                                        &self.profile,
                                        range,
                                        crate::facts::review_diagnostic(&error),
                                    )
                                    .await?;
                            }
                            None => return Err(error),
                        }
                        receipt.review_windows += 1;
                    }
                }
            }
            self.sink
                .record_fact_status(
                    &source.request.scope,
                    &source.artifact_id,
                    &self.profile,
                    None,
                    "complete",
                )
                .await?;
            Ok(receipt)
        }
        async fn run_source(&self, source: FactSource) -> Result<FactReceipt> {
            source.validate()?;
            self.validate(&source.request)?;
            let scope = &source.request.scope;
            let profile = &source.request.profile;
            let mut receipt = FactReceipt {
                source: source.clone(),
                status: FactStatus::Complete,
                stored_windows: 0,
                skipped_windows: 0,
                review_windows: 0,
                new_claims: 0,
            };
            if self
                .sink
                .fact_status(scope, &source.artifact_id, profile, None)
                .await?
                .is_some()
            {
                receipt.status = FactStatus::AlreadyComplete;
                return Ok(receipt);
            }
            let input = match self.sink.extraction_input(scope, &source.artifact_id).await {
                Ok(input) => input,
                Err(error) if error.downcast_ref::<ExtractionReview>().is_some() => {
                    self.sink
                        .record_fact_status(
                            scope,
                            &source.artifact_id,
                            profile,
                            None,
                            "input_review",
                        )
                        .await?;
                    receipt.status = FactStatus::Deferred;
                    return Ok(receipt);
                }
                Err(error) => return Err(error),
            };
            if self.config.fact_quality.is_some() {
                #[cfg(feature = "language")]
                {
                    return match self
                        .run_quality(source.clone(), receipt.clone(), input)
                        .await
                    {
                        Err(error)
                            if matches!(
                                error.downcast_ref::<ExtractionReview>(),
                                Some(ExtractionReview::Input)
                            ) =>
                        {
                            self.sink
                                .record_fact_status(
                                    scope,
                                    &source.artifact_id,
                                    profile,
                                    None,
                                    "input_review",
                                )
                                .await?;
                            receipt.status = FactStatus::Deferred;
                            Ok(receipt)
                        }
                        result => result,
                    };
                }
                #[cfg(not(feature = "language"))]
                anyhow::bail!("typed fact extraction requires the language build feature");
            }
            let references = match self
                .config
                .chat
                .as_ref()
                .context("chat configuration required")?
                .references
                .resolve(&input.document)
            {
                Ok(references) => references,
                Err(_) => {
                    self.sink
                        .record_fact_status(
                            scope,
                            &source.artifact_id,
                            profile,
                            None,
                            "input_review",
                        )
                        .await?;
                    receipt.status = FactStatus::Deferred;
                    return Ok(receipt);
                }
            };
            for range in input.ranges {
                let chunk = Chunk {
                    document: input.document.clone(),
                    range,
                };
                if self.sink.candidate_exists(&chunk, profile).await? {
                    receipt.skipped_windows += 1;
                    continue;
                }
                if self
                    .sink
                    .fact_status(scope, &source.artifact_id, profile, Some(range))
                    .await?
                    .is_some()
                {
                    receipt.review_windows += 1;
                    continue;
                }
                match self.extractor.extract(&chunk).await {
                    Ok(graph) => {
                        self.sink
                            .write_candidates(&chunk, &graph, &references)
                            .await?;
                        receipt.stored_windows += 1;
                        receipt.new_claims += graph.claims.len();
                    }
                    Err(error) => {
                        let reason = match error.downcast_ref::<ExtractionReview>() {
                            Some(ExtractionReview::Input) => "input_review",
                            Some(ExtractionReview::Output) => "output_rejected",
                            None => return Err(error),
                        };
                        self.sink
                            .record_fact_status(
                                scope,
                                &source.artifact_id,
                                profile,
                                Some(range),
                                reason,
                            )
                            .await?;
                        receipt.review_windows += 1;
                    }
                }
            }
            self.sink
                .record_fact_status(scope, &source.artifact_id, profile, None, "complete")
                .await?;
            tracing::info!(org_id=%scope,profile=%profile,stored_windows=receipt.stored_windows,review_windows=receipt.review_windows,new_claims=receipt.new_claims,"fact source processed");
            Ok(receipt)
        }
    }
    impl FactActivities for FactEngine {
        type Error = anyhow::Error;
        async fn discover(&mut self, request: FactRequest) -> Result<Option<FactSource>> {
            self.validate(&request)?;
            Ok(self
                .sink
                .next_fact_source(
                    &request.scope,
                    &self.profiles[&request.scope],
                    &request.profile,
                )
                .await?
                .map(|artifact_id| FactSource {
                    request,
                    artifact_id,
                }))
        }
        async fn process(&mut self, source: FactSource) -> Result<FactReceipt> {
            self.run_source(source).await
        }
    }
}
#[cfg(feature = "postgres")]
pub use runtime::FactEngine;
