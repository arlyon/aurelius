use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "aurelius")]
#[command(about = "Rust-Recall: A fully on-device RAG CLI tool.", long_about = None)]
pub struct Cli {
    /// Use quantized version of the embedding model.
    #[arg(long, global = true)]
    pub quantized: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Download the required models (zembed-1 and gemma-4-E4B).
    Models,

    /// Ingest a directory of files into the local knowledge graph.
    Ingest {
        /// The path to the directory to ingest.
        path: PathBuf,
    },

    /// Search the knowledge graph and generate a response.
    Search {
        /// The natural language query.
        query: String,
    },

    /// List all ingested files in the database.
    Ls,
}
