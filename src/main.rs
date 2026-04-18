#![allow(clippy::default_constructed_unit_structs)]

mod cli;
mod ingest;
mod metabolic;
mod models;
mod query;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};
use ingest::extract_facts::run_extract_facts;
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
        Commands::Ingest { path, invalidate_path, invalidate_before, no_extract } => {
            let lemonade = cli.lemonade.then(|| (cli.lemonade_url.clone(), cli.lemonade_embed_model.clone()));
            info!("Ingesting path: {:?} (quantized: {}, ollama: {}, lemonade: {})", path, cli.quantized, cli.ollama, cli.lemonade);
            run_ingest(path, cli.quantized, cli.ollama, lemonade, invalidate_path, invalidate_before).await?;
            if !no_extract {
                info!("Running fact extraction (pass --no-extract to skip)...");
                run_extract_facts().await?;
            }
            info!("Ingestion complete.");
        }
        Commands::Search { query, no_think } => {
            let lemonade = cli.lemonade.then(|| (cli.lemonade_url.clone(), cli.lemonade_embed_model.clone(), cli.lemonade_llm_model.clone()));
            info!("Searching for: {} (quantized: {}, ollama: {}, lemonade: {})", query, cli.quantized, cli.ollama, cli.lemonade);
            run_search(query, cli.quantized, cli.ollama, lemonade, !no_think).await?;
        }
        Commands::ExtractFacts => {
            info!("Extracting facts from ingested chunks...");
            run_extract_facts().await?;
        }
        Commands::Ls => {
            run_ls().await?;
        }
    }

    Ok(())
}
