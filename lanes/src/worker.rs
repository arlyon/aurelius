use crate::{
    contracts::{CandidateGraph, Chunk},
    facts::ChatExtractor,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::future::Future;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionRequest {
    pub chunk: Chunk,
    pub profile: String,
}

pub trait Activities {
    type Error;
    fn extract(
        &mut self,
        request: ExtractionRequest,
    ) -> impl Future<Output = Result<CandidateGraph, Self::Error>>;
}

// Hosts may call this directly from their timer. Temporal supplies only activity bindings.
pub async fn extract_document<A: Activities>(
    activities: &mut A,
    request: ExtractionRequest,
) -> Result<CandidateGraph, A::Error> {
    activities.extract(request).await
}

impl Activities for ChatExtractor {
    type Error = anyhow::Error;
    async fn extract(&mut self, request: ExtractionRequest) -> Result<CandidateGraph> {
        ensure!(
            request.profile == self.profile,
            "extraction profile mismatch"
        );
        ChatExtractor::extract(self, &request.chunk).await
    }
}
