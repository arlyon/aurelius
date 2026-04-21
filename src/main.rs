#![allow(clippy::default_constructed_unit_structs)]

mod cli;
mod ingest;
mod metabolic;
mod query;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands, ModelSpec};
use ingest::extract_facts::run_extract_facts;
use ingest::pipeline::run_ingest;
use metabolic::dream::run_dream;
use query::ls::run_ls;
use query::search::run_search;
use query::teach::run_teach;
use query::today::run_today;
use swiftide::traits::{ChatCompletion, EmbeddingModel};
use swiftide_integrations::{lemonade::config::LemonadeConfig, ollama::config::OllamaConfig};
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

fn build_embedder(
    spec: &ModelSpec,
    ollama_url: Option<String>,
    lemonade_url: Option<String>,
) -> Result<Box<dyn EmbeddingModel>> {
    match spec {
        ModelSpec::Ollama(model) => {
            let mut b = swiftide_integrations::ollama::Ollama::builder();

            b.default_embed_model(model);

            if let Some(url) = ollama_url {
                use async_openai::Client;
                b.client(Client::<OllamaConfig>::with_config(
                    OllamaConfig::builder().api_base(url).build()?,
                ));
            }

            Ok(Box::new(b.build()?))
        }
        ModelSpec::Lemonade(model) => {
            let mut b = swiftide_integrations::lemonade::Lemonade::builder();
            b.default_embed_model(model);

            if let Some(url) = lemonade_url {
                use async_openai::Client;
                b.client(Client::<LemonadeConfig>::with_config(
                    LemonadeConfig::builder().api_base(url).build()?,
                ));
            }

            Ok(Box::new(b.build()?))
        }
    }
}

fn build_chat(
    spec: &ModelSpec,
    ollama_url: Option<String>,
    lemonade_url: Option<String>,
) -> Result<Box<dyn ChatCompletion>> {
    match spec {
        ModelSpec::Ollama(model) => {
            let mut b = swiftide_integrations::ollama::Ollama::builder();

            b.default_prompt_model(model);

            if let Some(url) = ollama_url {
                use async_openai::Client;
                b.client(Client::<OllamaConfig>::with_config(
                    OllamaConfig::builder().api_base(url).build()?,
                ));
            }

            Ok(Box::new(b.build()?))
        }
        ModelSpec::Lemonade(model) => {
            let mut b = swiftide_integrations::lemonade::Lemonade::builder();
            b.default_embed_model(model);

            if let Some(url) = lemonade_url {
                use async_openai::Client;
                b.client(Client::<LemonadeConfig>::with_config(
                    LemonadeConfig::builder().api_base(url).build()?,
                ));
            }

            Ok(Box::new(b.build()?))
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let ollama_url = cli.ollama_url;
    let lemonade_url = cli.lemonade_url;

    match cli.command {
        Commands::Models => {
            info!("Downloading models...");
            // TODO: re-enable when we re-enable local models
            // download_models(cli.quantized).await?;
            info!("Models downloaded successfully.");
        }
        Commands::Ingest {
            path,
            invalidate_path,
            invalidate_before,
            no_extract,
        } => {
            info!(
                "Ingesting path: {:?} (embed: {}, chat: {})",
                path, cli.embed, cli.chat
            );

            let embedder = build_embedder(&cli.embed, ollama_url.clone(), lemonade_url.clone())?;
            run_ingest(path, embedder, invalidate_path, invalidate_before).await?;

            if !no_extract {
                info!("Running fact extraction (pass --no-extract to skip)...");
                let completion = build_chat(&cli.chat, ollama_url, lemonade_url)?;
                run_extract_facts(&completion).await?;
            }
            info!("Ingestion complete.");
        }
        Commands::Search { query, no_think } => {
            info!(
                "Searching for: {} (embed: {}, chat: {})",
                query, cli.embed, cli.chat
            );

            let embedder = build_embedder(&cli.embed, ollama_url.clone(), lemonade_url.clone())?;
            let completion = build_chat(&cli.chat, ollama_url, lemonade_url)?;
            run_search(query, &embedder, &completion, !no_think).await?;
        }
        Commands::ExtractFacts => {
            info!("Extracting facts from ingested chunks...");
            let completion = build_chat(&cli.chat, ollama_url, lemonade_url)?;
            run_extract_facts(&completion).await?;
        }
        Commands::Dream { dry_run } => {
            info!("Running dream cycle (dry_run: {})...", dry_run);
            let completion = build_chat(&cli.chat, ollama_url, lemonade_url)?;
            run_dream(dry_run, &completion).await?;
        }
        Commands::Teach { limit } => {
            run_teach(limit).await?;
        }
        Commands::Today => {
            run_today().await?;
        }
        Commands::Ls => {
            run_ls().await?;
        }
    }

    Ok(())
}
