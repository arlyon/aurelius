use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "aurelius")]
#[command(about = "Rust-Recall: A fully on-device RAG CLI tool.", long_about = None)]
pub struct Cli {
    /// Use quantized version of the embedding model.
    #[arg(long, global = true)]
    pub quantized: bool,

    /// Use ollama for embeddings.
    #[arg(long, global = true)]
    pub ollama: bool,

    /// Use lemonade for embeddings and LLM (OpenAI-compatible local server).
    #[arg(long, global = true)]
    pub lemonade: bool,

    /// Lemonade server base URL.
    #[arg(long, global = true, default_value = "http://localhost:13305")]
    pub lemonade_url: String,

    /// Lemonade embedding model name.
    #[arg(long, global = true, default_value = "user.zembed-1")]
    pub lemonade_embed_model: String,

    /// Lemonade LLM model name.
    #[arg(long, global = true, default_value = "user.Qwen3.6-35B-A3B-GGUF-UD-Q4_K_M")]
    pub lemonade_llm_model: String,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Download the required models (zembed-1 and gemma-4-E4B).
    Models,

    /// Ingest a directory of files into the local knowledge graph, then extract facts.
    /// Pass --no-extract to skip fact extraction.
    Ingest {
        /// The path to the directory to ingest.
        path: Vec<PathBuf>,

        /// Invalidate cache for paths matching this glob.
        #[arg(long)]
        invalidate_path: Option<String>,

        /// Invalidate cache for entries added before this date (RFC3339 or '1h', '2d' offset).
        #[arg(long)]
        invalidate_before: Option<String>,

        /// Skip fact extraction after ingestion (do not load Qwen).
        #[arg(long)]
        no_extract: bool,
    },

    /// Extract typed atomic facts from ingested chunks via Qwen (qwen3:8b, Ollama).
    /// Idempotent: skips context windows already present in the facts table.
    ExtractFacts,

    /// Search the knowledge graph and generate a response.
    Search {
        /// The natural language query.
        query: String,

        /// Disable thinking mode for the Gemma synthesis call.
        #[arg(long)]
        no_think: bool,
    },

    /// List all ingested files in the database.
    Ls,
}
