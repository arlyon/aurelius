#![allow(clippy::default_constructed_unit_structs)]

mod cli;
mod ingest;
mod models;
mod query;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};
use ingest::pipeline::run_ingest;
use models::download::download_models;
use query::ls::run_ls;
use query::search::run_search;
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Models => {
            info!("Downloading models...");
            download_models(cli.quantized).await?;
            info!("Models downloaded successfully.");
        }
        Commands::Ingest { path, invalidate_path, invalidate_before } => {
            let lemonade = cli.lemonade.then(|| (cli.lemonade_url.clone(), cli.lemonade_embed_model.clone()));
            info!("Ingesting path: {:?} (quantized: {}, ollama: {}, lemonade: {})", path, cli.quantized, cli.ollama, cli.lemonade);
            run_ingest(path, cli.quantized, cli.ollama, lemonade, invalidate_path, invalidate_before).await?;
            info!("Ingestion complete.");
        }
        Commands::Search { query, no_think } => {
            let lemonade = cli.lemonade.then(|| (cli.lemonade_url.clone(), cli.lemonade_embed_model.clone(), cli.lemonade_llm_model.clone()));
            info!("Searching for: {} (quantized: {}, ollama: {}, lemonade: {})", query, cli.quantized, cli.ollama, cli.lemonade);
            run_search(query, cli.quantized, cli.ollama, lemonade, !no_think).await?;
        }
        Commands::Ls => {
            run_ls().await?;
        }
    }

    Ok(())
}
