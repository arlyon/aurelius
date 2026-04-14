use crate::models::zembed::Embedder;
use anyhow::Result;
use async_trait::async_trait;
use blake3;
use kreuzberg::{ExtractionConfig, extract_file};
use std::path::PathBuf;
use swiftide::indexing::{self, IndexingStream, Node};
use swiftide::integrations::lancedb::LanceDB;
use swiftide::traits::{Loader, NodeCache, Transformer, WithIndexingDefaults};
use tracing::{debug, info, warn};

#[derive(Clone, Default, Debug)]
pub struct GitignoreLoader {
    path: Vec<PathBuf>,
}

impl GitignoreLoader {
    pub fn new(path: Vec<PathBuf>) -> Self {
        Self { path }
    }
}

#[async_trait]
impl Loader for GitignoreLoader {
    type Output = String;

    fn into_stream(self) -> IndexingStream<Self::Output> {
        let nodes = self
            .path
            .into_iter()
            .flat_map(|p| {
                let files = crate::ingest::walker::walk_directory(p);
                info!("Found {} files to process", files.len());
                files.into_iter().filter_map(|p| {
                    let path_str = p.to_string_lossy().to_string();
                    debug!("Preparing node for {:?}", path_str);

                    // Hash the file for caching
                    let hash = match std::fs::read(&p) {
                        Ok(bytes) => blake3::hash(&bytes).to_hex().to_string(),
                        Err(e) => {
                            warn!("Failed to read file for hashing at {:?}: {}", p, e);
                            return None;
                        }
                    };

                    let mut node = Node::<String>::new("".to_string());
                    node.metadata.insert("path".to_string(), path_str);
                    node.metadata.insert("hash".to_string(), hash);
                    Some(node)
                })
            })
            .collect();

        IndexingStream::from_nodes(nodes)
    }
}

#[derive(Clone, Default)]
pub struct KreuzbergTransformer;

impl WithIndexingDefaults for KreuzbergTransformer {}

#[async_trait]
impl Transformer for KreuzbergTransformer {
    type Input = String;
    type Output = String;

    async fn transform_node(&self, mut node: Node<Self::Input>) -> Result<Node<Self::Output>> {
        let path_str = node
            .metadata
            .get("path")
            .ok_or_else(|| anyhow::anyhow!("Node missing path metadata"))?
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Path metadata is not a string"))?;

        let path = PathBuf::from(path_str);
        debug!("Extracting content from {:?}", path);

        let config = ExtractionConfig::default();
        match extract_file(&path, None, &config).await {
            Ok(result) => {
                debug!(
                    "Successfully extracted {} characters from {:?}",
                    result.content.len(),
                    path
                );
                node.chunk = result.content;
            }
            Err(e) => {
                warn!(
                    "Failed to extract content from {:?}: {}. Skipping file.",
                    path, e
                );
                // Silently skip files that can't be processed (e.g. unknown mime types)
                node.chunk = "".to_string();
            }
        }

        Ok(node)
    }
}

#[derive(Clone, Default)]
pub struct ContextPrependTransformer;

impl WithIndexingDefaults for ContextPrependTransformer {}

#[async_trait]
impl Transformer for ContextPrependTransformer {
    type Input = String;
    type Output = String;

    async fn transform_node(&self, mut node: Node<Self::Input>) -> Result<Node<Self::Output>> {
        let path_str = node
            .metadata
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let global_context = format!("File: {}\n---\n", path_str);
        node.chunk = format!("{}{}", global_context, node.chunk);

        Ok(node)
    }
}

#[derive(Clone, Default)]
pub struct LogTransformer;

impl WithIndexingDefaults for LogTransformer {}

#[async_trait]
impl Transformer for LogTransformer {
    type Input = String;
    type Output = String;

    async fn transform_node(&self, node: Node<Self::Input>) -> Result<Node<Self::Output>> {
        let path = node
            .metadata
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        info!("Processing file: {}", path);
        Ok(node)
    }
}

#[derive(Clone, Default, Debug)]
struct Cache;

#[async_trait::async_trait]
impl NodeCache for Cache {
    type Input = String;

    async fn get(&self, _node: &Node<Self::Input>) -> bool {
        false
    }
    async fn set(&self, _node: &Node<Self::Input>) {}
}

pub async fn run_ingest(path: Vec<PathBuf>, quantized: bool) -> Result<()> {
    info!("Starting ingestion for {:?}", path);

    let embedder = Embedder::new(quantized).await?;

    let lancedb = LanceDB::builder()
        .uri("aurelius_db")
        .table_name("chunks")
        .vector_size(2560) // zembed-1 dimension
        .with_vector(indexing::EmbeddedField::Combined)
        .with_metadata("path")
        .build()?;

    let loader = GitignoreLoader::new(path);

    indexing::Pipeline::from_loader(loader)
        .with_concurrency(1)
        .filter_cached(Cache)
        .then(KreuzbergTransformer::default())
        .then(LogTransformer::default())
        .then_chunk(indexing::transformers::ChunkMarkdown::from_chunk_range(
            100..500,
        ))
        .filter(|node| node.as_ref().map(|n| !n.chunk.is_empty()).unwrap_or(false))
        .then(ContextPrependTransformer::default())
        .then_in_batch(indexing::transformers::Embed::new(embedder))
        .then_store_with(lancedb)
        .run()
        .await?;

    info!("Ingestion complete");
    Ok(())
}
