mod cli;
mod ingest;
mod models;
mod query;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};
use models::download::download_models;
use ingest::pipeline::run_ingest;
use query::search::run_search;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Models => {
            println!("Downloading models...");
            download_models().await?;
            println!("Models downloaded successfully.");
        }
        Commands::Ingest { path } => {
            println!("Ingesting path: {:?}", path);
            run_ingest(path).await?;
            println!("Ingestion complete.");
        }
        Commands::Search { query } => {
            println!("Searching for: {}", query);
            run_search(query).await?;
        }
    }

    Ok(())
}
