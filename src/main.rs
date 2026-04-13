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
use query::ls::run_ls;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Models => {
            println!("Downloading models...");
            download_models(cli.quantized).await?;
            println!("Models downloaded successfully.");
        }
        Commands::Ingest { path } => {
            println!("Ingesting path: {:?} (quantized: {})", path, cli.quantized);
            run_ingest(path, cli.quantized).await?;
            println!("Ingestion complete.");
        }
        Commands::Search { query } => {
            println!("Searching for: {} (quantized: {})", query, cli.quantized);
            run_search(query, cli.quantized).await?;
        }
        Commands::Ls => {
            run_ls().await?;
        }
    }

    Ok(())
}
