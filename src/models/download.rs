use anyhow::Result;
use hf_hub::api::tokio::ApiBuilder;
use tracing::info;

pub async fn download_models(quantized: bool) -> Result<()> {
    let api = ApiBuilder::new().with_progress(true).build()?;

    if quantized {
        info!("Downloading Abiray/zembed-1-Q4_K_M-GGUF...");
        let repo = api.model("Abiray/zembed-1-Q4_K_M-GGUF".to_string());
        repo.get("zembed-1-Q4_K_M.gguf").await?;

        // We still need the tokenizer and config from the base repo
        info!("Downloading tokenizer and config from zeroentropy/zembed-1...");
        let repo = api.model("zeroentropy/zembed-1".to_string());
        repo.get("tokenizer.json").await?;
        repo.get("config.json").await?;
    } else {
        // Download zembed-1
        info!("Downloading zeroentropy/zembed-1...");
        let repo = api.model("zeroentropy/zembed-1".to_string());
        repo.get("model-00001-of-00002.safetensors").await?;
        repo.get("model-00002-of-00002.safetensors").await?;
        repo.get("tokenizer.json").await?;
        repo.get("config.json").await?;
    }

    // Download Gemma-4-E4B
    info!("Downloading google/gemma-4-E4B...");
    let repo = api.model("google/gemma-4-E4B".to_string());
    repo.get("model.safetensors").await?;
    repo.get("tokenizer.json").await?;
    repo.get("config.json").await?;

    Ok(())
}
