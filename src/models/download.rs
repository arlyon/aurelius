use anyhow::Result;
use hf_hub::api::tokio::ApiBuilder;
use tracing::info;

pub async fn download_models() -> Result<()> {
    let api = ApiBuilder::new().with_progress(true).build()?;

    // Download zembed-1
    info!("Downloading zeroentropy/zembed-1...");
    let repo = api.model("zeroentropy/zembed-1".to_string());
    repo.get("model-00001-of-00002.safetensors").await?;
    repo.get("model-00002-of-00002.safetensors").await?;
    repo.get("tokenizer.json").await?;
    repo.get("config.json").await?;

    // Download Gemma-2-2b-it
    info!("Downloading google/gemma-2-2b-it...");
    let repo = api.model("google/gemma-2-2b-it".to_string());
    repo.get("model-00001-of-00002.safetensors").await?;
    repo.get("model-00002-of-00002.safetensors").await?;
    repo.get("tokenizer.json").await?;
    repo.get("config.json").await?;

    Ok(())
}
