use crate::models::zembed::Embedder;
use uuid;
use anyhow::Result;
use arrow_array;
use arrow_array::RecordBatchIterator;
use arrow_schema;
use async_trait::async_trait;
use blake3;
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use kreuzberg::{ChunkerType, ChunkingConfig, ExtractionConfig, extract_file};
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::path::PathBuf;
use std::sync::Arc;
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
                files.into_iter().filter_map(|p| {
                    let path_str = p.to_string_lossy().to_string();

                    // Hash the file for caching
                    let hash = match std::fs::read(&p) {
                        Ok(bytes) => {
                            let h = blake3::hash(&bytes).to_hex().to_string();
                            debug!("Hashed file {:?}: {}", p, h);
                            h
                        }
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
impl swiftide::traits::ChunkerTransformer for KreuzbergTransformer {
    type Input = String;
    type Output = String;

    async fn transform_node(&self, mut node: Node<Self::Input>) -> IndexingStream<Self::Output> {
        let path_str = node
            .metadata
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if path_str.is_empty() {
            return IndexingStream::from_nodes(vec![node]);
        }

        let path = PathBuf::from(path_str);
        debug!("Extracting content from {:?}", path);

        // Try to determine a good chunker type based on extension
        let chunker_type = match path.extension().and_then(|ext| ext.to_str()) {
            Some("md") => Some(ChunkerType::Markdown),
            Some("yaml") | Some("yml") => Some(ChunkerType::Yaml),
            _ => Some(ChunkerType::Text),
        };

        let mut config = ExtractionConfig::default();
        if let Some(ct) = chunker_type {
            config.chunking = Some(ChunkingConfig {
                chunker_type: ct,
                max_characters: 1500,
                overlap: 0,
                ..Default::default()
            });
        }

        match extract_file(&path, None, &config).await {
            Ok(result) => {
                debug!("Successfully extracted content from {:?}", path);

                // Common metadata for all chunks
                node.metadata
                    .insert("mime_type".to_string(), result.mime_type.to_string());

                if let Some(title) = &result.metadata.title {
                    node.metadata.insert("title".to_string(), title.to_string());
                }
                if let Some(authors) = &result.metadata.authors {
                    node.metadata
                        .insert("authors".to_string(), authors.join(", "));
                }
                if let Some(lang) = &result.metadata.language {
                    node.metadata
                        .insert("language".to_string(), lang.to_string());
                }

                #[cfg(any(feature = "keywords-yake", feature = "keywords-rake"))]
                if let Some(keywords) = &result.extracted_keywords {
                    let kw_str = keywords
                        .iter()
                        .map(|k| k.text.clone())
                        .collect::<Vec<_>>()
                        .join(", ");
                    node.metadata.insert("keywords".to_string(), kw_str);
                }

                #[cfg(feature = "tree-sitter")]
                if let Some(code) = &result.code_intelligence {
                    let mut symbols = Vec::new();
                    for func in &code.imports {
                        symbols.push(func.source.clone());
                    }
                    for class in &code.exports {
                        symbols.push(class.name.clone());
                    }
                    for sym in &code.symbols {
                        symbols.push(sym.name.clone());
                    }

                    if !symbols.is_empty() {
                        node.metadata
                            .insert("code_symbols".to_string(), symbols.join(" "));
                    }
                }

                if let Some(chunks) = result.chunks {
                    if !chunks.is_empty() {
                        let mut nodes = Vec::new();
                        for (i, chunk) in chunks.into_iter().enumerate() {
                            let mut chunk_node = node.clone();
                            chunk_node.chunk = chunk.content;
                            chunk_node
                                .metadata
                                .insert("chunk_index".to_string(), i as i64);
                            chunk_node
                                .metadata
                                .insert("chunked_by".to_string(), "kreuzberg".to_string());

                            if let Some(hc) = chunk.metadata.heading_context {
                                let breadcrumbs = hc
                                    .headings
                                    .iter()
                                    .map(|h| h.text.clone())
                                    .collect::<Vec<_>>()
                                    .join(" > ");
                                chunk_node
                                    .metadata
                                    .insert("heading_context".to_string(), breadcrumbs);
                            }

                            nodes.push(chunk_node);
                        }
                        return IndexingStream::from_nodes(nodes);
                    }
                }

                // Fallback: use the whole content and let the next pipeline step handle it
                node.chunk = result.content;
                IndexingStream::from_nodes(vec![node])
            }
            Err(e) => {
                warn!(
                    "Failed to extract content from {:?}: {}. Skipping file.",
                    path, e
                );
                node.chunk = "".to_string();
                IndexingStream::from_nodes(vec![node])
            }
        }
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

        let mut header = format!("File: {}\n", path_str);

        if let Some(title) = node.metadata.get("title").and_then(|v| v.as_str()) {
            header.push_str(&format!("Title: {}\n", title));
        }
        if let Some(authors) = node.metadata.get("authors").and_then(|v| v.as_str()) {
            header.push_str(&format!("Authors: {}\n", authors));
        }
        if let Some(mime) = node.metadata.get("mime_type").and_then(|v| v.as_str()) {
            header.push_str(&format!("Format: {}\n", mime));
        }
        if let Some(lang) = node.metadata.get("language").and_then(|v| v.as_str()) {
            header.push_str(&format!("Language: {}\n", lang));
        }

        if let Some(context) = node
            .metadata
            .get("heading_context")
            .and_then(|v| v.as_str())
        {
            header.push_str(&format!("Context: {}\n", context));
        }

        let global_context = format!("{}---\n", header);
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

#[derive(Clone, Debug)]
struct Cache {
    cache_table: lancedb::table::Table,
    chunks_table: lancedb::table::Table,
}

impl Cache {
    pub async fn new(
        db: &lancedb::Connection,
        chunks_table: lancedb::table::Table,
    ) -> Result<Self> {
        let table_name = "cache";
        let cache_table = match db.open_table(table_name).execute().await {
            Ok(t) => t,
            Err(_) => {
                let schema = Arc::new(arrow_schema::Schema::new(vec![
                    arrow_schema::Field::new("path", arrow_schema::DataType::Utf8, false),
                    arrow_schema::Field::new("hash", arrow_schema::DataType::Utf8, false),
                    arrow_schema::Field::new(
                        "last_ingested_at",
                        arrow_schema::DataType::Timestamp(
                            arrow_schema::TimeUnit::Microsecond,
                            None,
                        ),
                        false,
                    ),
                ]));
                db.create_empty_table(table_name, schema).execute().await?
            }
        };
        Ok(Self {
            cache_table,
            chunks_table,
        })
    }

    pub async fn invalidate(
        &self,
        path_glob: Option<&str>,
        before: Option<DateTime<Utc>>,
    ) -> Result<()> {
        let mut conditions = Vec::new();

        if let Some(glob) = path_glob {
            let sql_glob = glob.replace('*', "%").replace('?', "_");
            conditions.push(format!("path LIKE '{}'", sql_glob));
        }

        if let Some(before_dt) = before {
            conditions.push(format!(
                "last_ingested_at < {}",
                before_dt.timestamp_micros()
            ));
        }

        if !conditions.is_empty() {
            let filter = conditions.join(" AND ");
            info!("Invalidating cache with filter: {}", filter);
            self.cache_table.delete(&filter).await?;
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl NodeCache for Cache {
    type Input = String;

    async fn get(&self, node: &Node<Self::Input>) -> bool {
        let path = node.metadata.get("path").and_then(|v| v.as_str());
        let hash = node.metadata.get("hash").and_then(|v| v.as_str());

        if let (Some(path), Some(hash)) = (path, hash) {
            let query = format!("path = '{}' AND hash = '{}'", path, hash);
            if let Ok(result) = self.cache_table.query().only_if(query).execute().await {
                if let Ok(batches) = result.try_collect::<Vec<arrow_array::RecordBatch>>().await {
                    if batches.iter().any(|b| b.num_rows() > 0) {
                        return true;
                    }
                }
            }

            // If we are here, the file is either new or modified.
            // Invalidate old chunks for this path.
            debug!("Invalidating old chunks for path: {}", path);
            let _ = self
                .chunks_table
                .delete(&format!("path = '{}'", path))
                .await;

            // Also remove from cache table to be clean, though set() will overwrite/append
            let _ = self.cache_table.delete(&format!("path = '{}'", path)).await;
        }
        false
    }

    async fn set(&self, node: &Node<Self::Input>) {
        let path = node.metadata.get("path").and_then(|v| v.as_str());
        let hash = node.metadata.get("hash").and_then(|v| v.as_str());

        if let (Some(path), Some(hash)) = (path, hash) {
            let last_ingested_at = Utc::now().timestamp_micros();
            let batch = arrow_array::RecordBatch::try_new(
                self.cache_table.schema().await.unwrap(),
                vec![
                    Arc::new(arrow_array::StringArray::from(vec![path])),
                    Arc::new(arrow_array::StringArray::from(vec![hash])),
                    Arc::new(arrow_array::TimestampMicrosecondArray::from(vec![
                        last_ingested_at,
                    ])),
                ],
            )
            .unwrap();

            let reader =
                RecordBatchIterator::new(vec![Ok(batch)], self.cache_table.schema().await.unwrap());
            let _ = self.cache_table.add(reader).execute().await;
        }
    }
}

#[derive(Clone, Default)]
pub struct HierarchicalChunker;

impl WithIndexingDefaults for HierarchicalChunker {}

#[async_trait]
impl swiftide::traits::ChunkerTransformer for HierarchicalChunker {
    type Input = String;
    type Output = String;

    async fn transform_node(&self, node: Node<Self::Input>) -> IndexingStream<Self::Output> {
        if node.chunk.is_empty() {
            return IndexingStream::from_nodes(vec![]);
        }

        let context_window_id = uuid::Uuid::new_v4().to_string();
        let parent_text = node.chunk.clone();
        let chars: Vec<char> = parent_text.chars().collect();
        let total = chars.len();

        const CHILD_SIZE: usize = 300;
        const STEP: usize = 250;

        if total <= CHILD_SIZE {
            let mut child = node;
            child.metadata.insert("context_window_id".to_string(), context_window_id);
            child.metadata.insert("parent_block".to_string(), parent_text);
            return IndexingStream::from_nodes(vec![child]);
        }

        let mut children = Vec::new();
        let mut start = 0usize;
        loop {
            let end = (start + CHILD_SIZE).min(total);
            let child_text: String = chars[start..end].iter().collect();

            let mut child = node.clone();
            child.chunk = child_text;
            child.metadata.insert("context_window_id".to_string(), context_window_id.clone());
            child.metadata.insert("parent_block".to_string(), parent_text.clone());
            children.push(child);

            if end >= total {
                break;
            }
            start += STEP;
        }

        IndexingStream::from_nodes(children)
    }
}

pub async fn run_ingest(
    path: Vec<PathBuf>,
    quantized: bool,
    ollama: bool,
    lemonade: Option<(String, String)>,
    invalidate_path: Option<String>,
    invalidate_before: Option<String>,
) -> Result<()> {
    info!("Starting ingestion for {:?}", path);

    let lemonade_ref = lemonade.as_ref().map(|(url, model)| (url.as_str(), model.as_str()));
    let embedder = Embedder::new(quantized, ollama, lemonade_ref).await?;

    let db_path = "aurelius_db";
    let db = lancedb::connect(db_path).execute().await?;

    let table_name = "chunks";
    let chunks_schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new(
            "vector_combined",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    true,
                )),
                2560,
            ),
            true,
        ),
        arrow_schema::Field::new(
            "id",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::UInt8,
                    true,
                )),
                16,
            ),
            false,
        ),
        arrow_schema::Field::new("chunk", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("path", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("mime_type", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("authors", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("language", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("hash", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("keywords", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("code_symbols", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("context_window_id", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("parent_block", arrow_schema::DataType::Utf8, true),
    ]));
    let chunks_table = match db.open_table(table_name).execute().await {
        Ok(t) => {
            // Drop and recreate if schema is missing required fields
            let existing_schema = t.schema().await?;
            if existing_schema.field_with_name("vector_combined").is_err()
                || existing_schema.field_with_name("context_window_id").is_err()
            {
                db.drop_table(table_name, &[]).await?;
                db.create_empty_table(table_name, chunks_schema)
                    .execute()
                    .await?
            } else {
                t
            }
        }
        Err(_) => {
            db.create_empty_table(table_name, chunks_schema)
                .execute()
                .await?
        }
    };

    let cache = Cache::new(&db, chunks_table).await?;

    // Apply invalidation rules
    let before_dt = if let Some(before_str) = invalidate_before {
        if let Ok(dt) = DateTime::parse_from_rfc3339(&before_str) {
            Some(dt.with_timezone(&Utc))
        } else if before_str.ends_with('h') {
            let hours: i64 = before_str.trim_end_matches('h').parse()?;
            Some(Utc::now() - chrono::Duration::hours(hours))
        } else if before_str.ends_with('d') {
            let days: i64 = before_str.trim_end_matches('d').parse()?;
            Some(Utc::now() - chrono::Duration::days(days))
        } else {
            anyhow::bail!("Invalid date format for invalidate-before. Use RFC3339 or '1h', '2d'.");
        }
    } else {
        None
    };

    if invalidate_path.is_some() || before_dt.is_some() {
        cache
            .invalidate(invalidate_path.as_deref(), before_dt)
            .await?;
    }

    let lancedb = LanceDB::builder()
        .uri(db_path)
        .table_name(table_name)
        .vector_size(2560) // zembed-1 dimension
        .with_vector(indexing::EmbeddedField::Combined)
        .with_metadata("path")
        .with_metadata("mime_type")
        .with_metadata("title")
        .with_metadata("authors")
        .with_metadata("language")
        .with_metadata("hash")
        .with_metadata("keywords")
        .with_metadata("code_symbols")
        .with_metadata("context_window_id")
        .with_metadata("parent_block")
        .build()?;

    let loader = GitignoreLoader::new(path);

    indexing::Pipeline::from_loader(loader)
        .with_concurrency(1)
        .filter_cached(cache)
        .then_chunk(KreuzbergTransformer::default())
        .then(LogTransformer::default())
        .then_chunk(HierarchicalChunker::default())
        .filter(|node| node.as_ref().map(|n| !n.chunk.is_empty()).unwrap_or(false))
        .then(ContextPrependTransformer::default())
        .then_in_batch(indexing::transformers::Embed::new(embedder))
        .then_store_with(lancedb)
        .run()
        .await?;

    // Ensure FTS index exists on the chunk column for BM25 search
    let chunks_table = db.open_table(table_name).execute().await?;
    if let Err(e) = chunks_table
        .create_index(
            &["chunk"],
            lancedb::index::Index::FTS(FtsIndexBuilder::default()),
        )
        .execute()
        .await
    {
        warn!(
            "Failed to create FTS index (search will fall back to vector only): {}",
            e
        );
    } else {
        info!("FTS index on 'chunk' column ready");
    }

    info!("Ingestion complete");
    Ok(())
}
