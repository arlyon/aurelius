#![allow(clippy::default_constructed_unit_structs)]

mod cli;
mod ingest;
mod metabolic;
mod query;
mod persistence;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands, ModelSpec};
use ingest::extract_facts::run_extract_facts;
use ingest::pipeline::run_ingest;
use metabolic::dream::run_dream;
use query::facts::run_facts;
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
            b.default_prompt_model(model);
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
    let ollama_url = cli.ollama_url.clone();
    let lemonade_url = cli.lemonade_url.clone();

    match cli.command {
        Commands::Models => {
            info!("Models are usually pre-downloaded or pulled on demand by backends.");
        }
        Commands::Ingest {
            path,
            invalidate_path,
            invalidate_before,
            no_extract,
        } => {
            let embedder = build_embedder(&cli.embed, ollama_url.clone(), lemonade_url.clone())?;
            run_ingest(path, embedder, invalidate_path, invalidate_before).await?;

            if !no_extract {
                let completion = build_chat(&cli.chat, ollama_url, lemonade_url)?;
                run_extract_facts(completion.as_ref()).await?;
            }
        }
        Commands::Search { query, no_think } => {
            let embedder = build_embedder(&cli.embed, ollama_url.clone(), lemonade_url.clone())?;
            let completion = build_chat(&cli.chat, ollama_url, lemonade_url)?;
            run_search(query, &embedder, &completion, !no_think).await?;
        }
        Commands::ExtractFacts => {
            let completion = build_chat(&cli.chat, ollama_url, lemonade_url)?;
            run_extract_facts(completion.as_ref()).await?;
        }
        Commands::Dream { dry_run, yes } => {
            let completion = build_chat(&cli.chat, ollama_url, lemonade_url)?;
            run_dream(dry_run, yes, completion.as_ref()).await?;
        }
        Commands::Teach { limit, prompt } => {
            let completion = build_chat(&cli.chat, ollama_url, lemonade_url)?;
            run_teach(limit, prompt, completion.as_ref()).await?;
        }
        Commands::Today => {
            run_today().await?;
        }
        Commands::Ls => {
            run_ls().await?;
        }
        Commands::Facts {
            subject,
            predicate,
            object,
            source,
            core_only,
            evicted,
            sort,
            limit,
            json,
            stats,
            show_source,
        } => {
            run_facts(
                subject,
                predicate,
                object,
                source,
                core_only,
                evicted,
                sort,
                limit,
                json,
                stats,
                show_source,
            )
            .await?;
        }
    }

    Ok(())
}
