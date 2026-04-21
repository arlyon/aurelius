use clap::{Parser, Subcommand};
use std::{fmt, path::PathBuf, str::FromStr};

/// A backend+model specifier in the form `backend` or `backend:model-name`.
/// Supported backends: `ollama`, `lemonade`.
#[derive(Clone, Debug)]
pub enum ModelSpec {
    Ollama(String),
    Lemonade(String),
}

impl FromStr for ModelSpec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.split_once(':') {
            Some(("ollama", model)) => Ok(Self::Ollama(model.to_string())),
            Some(("lemonade", model)) => Ok(Self::Lemonade(model.to_string())),
            Some((backend, _)) => Err(format!(
                "unknown backend '{backend}'; expected 'ollama' or 'lemonade'"
            )),
            None => Err(format!(
                "unknown backend '{s}'; expected 'ollama:<model>' or 'lemonade:<model>'"
            )),
        }
    }
}

impl fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ollama(m) => write!(f, "ollama:{m}"),
            Self::Lemonade(m) => write!(f, "lemonade:{m}"),
        }
    }
}

#[derive(Parser)]
#[command(name = "aurelius")]
#[command(about = "Rust-Recall: A fully on-device RAG CLI tool.", long_about = None)]
pub struct Cli {
    /// Embedding backend and model (e.g. 'ollama', 'ollama:nomic-embed-text', 'lemonade:user.zembed-1').
    #[arg(
        long,
        global = true,
        default_value = "ollama:hf.co/Abiray/zembed-1-Q4_K_M-GGUF",
        value_name = "BACKEND[:MODEL]"
    )]
    pub embed: ModelSpec,

    /// Chat backend and model (e.g. 'ollama', 'ollama:qwen3:8b', 'lemonade:user.Qwen3-35B').
    #[arg(
        long,
        global = true,
        default_value = "ollama:gemma4:26b",
        value_name = "BACKEND[:MODEL]"
    )]
    pub chat: ModelSpec,

    /// Ollama server base URL.
    #[arg(long, global = true)]
    pub ollama_url: Option<String>,

    /// Lemonade server base URL.
    #[arg(long, global = true)]
    pub lemonade_url: Option<String>,

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

    /// Run the nightly metabolic cycle: decay stale facts, resolve contradictions,
    /// merge weak fact clusters into neurons, and write a morning pulse briefing.
    Dream {
        /// Print what would change without writing anything.
        #[arg(long)]
        dry_run: bool,

        /// Apply changes automatically without prompting.
        #[arg(short, long)]
        yes: bool,
    },

    /// Interactively resolve contradictions in the facts table, locking in Core Truths.
    Teach {
        /// Maximum number of contradiction pairs to surface (default: 10).
        #[arg(long, default_value_t = 10)]
        limit: usize,

        /// An optional prompt to ad-hoc pull facts from.
        prompt: Option<String>,
    },

    /// Display the morning pulse dashboard (run `aurelius dream` first).
    Today,

    /// List all ingested files in the database.
    Ls,

    /// List and filter extracted facts from the knowledge graph.
    Facts {
        /// Filter by subject.
        #[arg(long)]
        subject: Option<String>,

        /// Filter by predicate.
        #[arg(long)]
        predicate: Option<String>,

        /// Filter by object.
        #[arg(long)]
        object: Option<String>,

        /// Filter by source file path.
        #[arg(long)]
        source: Option<String>,

        /// Show only "Core Truths" (confidence = 1.0).
        #[arg(long)]
        core_only: bool,

        /// Show evicted facts instead of current ones.
        #[arg(long)]
        evicted: bool,

        /// Sort by 'confidence' or 'date' (default: date).
        #[arg(long, default_value = "date")]
        sort: String,

        /// Limit the number of facts displayed (default: 50).
        #[arg(long, default_value_t = 50)]
        limit: usize,

        /// Output in JSON format.
        #[arg(long)]
        json: bool,

        /// Display summary statistics instead of facts.
        #[arg(long)]
        stats: bool,

        /// Show the source file path for each fact.
        #[arg(long)]
        show_source: bool,
    },
}
