use anyhow::{Context, Result};
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights as QModel;
use fastembed::{Qwen3Config, Qwen3Model};
use hf_hub::api::tokio::ApiBuilder;
use std::sync::Arc;
use swiftide::chat_completion::errors::LanguageModelError;
use swiftide::traits::EmbeddingModel;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::info;

pub enum Model {
    Full(Qwen3Model),
    Quantized(QModel),
}

pub struct Embedder {
    model: Arc<Mutex<Model>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish()
    }
}

fn l2_normalize(t: &Tensor) -> candle_core::Result<Tensor> {
    let sum_sq = t.sqr()?.sum_keepdim(1)?;
    t.broadcast_div(&sum_sq.sqrt()?)
}

// Replicating fastembed's attention mask logic
fn build_attention_mask_4d(mask: &Tensor) -> candle_core::Result<Tensor> {
    let (b_sz, seq_len) = mask.dims2()?;
    // [B, 1, 1, S]
    let mask = mask.reshape((b_sz, 1, 1, seq_len))?;
    // Convert to additive mask: 0 for keep, -1e9 for mask
    let mask = (mask.to_dtype(DType::F32)? - 1.0)?;
    let mask = (mask * 1e9)?;
    Ok(mask)
}

impl Embedder {
    pub async fn new(quantized: bool) -> Result<Self> {
        let api = ApiBuilder::new().build()?;

        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        let tokenizer_repo = api.model("zeroentropy/zembed-1".to_string());
        let tokenizer_file = tokenizer_repo.get("tokenizer.json").await?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams::default()));

        let model = if quantized {
            info!("Initializing quantized zembed-1 (GGUF)...");
            let repo = api.model("Abiray/zembed-1-Q4_K_M-GGUF".to_string());
            let model_file = repo
                .get("zembed-1-Q4_K_M.gguf")
                .await
                .context("getting model")?;

            let mut file = std::fs::File::open(&model_file)?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut file)
                .map_err(|e| anyhow::anyhow!("Failed to read GGUF content: {:?}", e))?;
            let model = QModel::from_gguf(content, &mut file, &device)
                .map_err(|e| anyhow::anyhow!("Failed to load GGUF model: {:?}", e))?;

            Model::Quantized(model)
        } else {
            info!("Initializing zembed-1 with manual prefix fix...");
            let repo = api.model("zeroentropy/zembed-1".to_string());
            let config_file = repo.get("config.json").await?;
            let weights_files = vec![
                repo.get("model-00001-of-00002.safetensors").await?,
                repo.get("model-00002-of-00002.safetensors").await?,
            ];

            let config_bytes = std::fs::read(config_file)?;
            let cfg: Qwen3Config = serde_json::from_slice(&config_bytes)?;

            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&weights_files, DType::F32, &device)?
            };
            let vb = vb.pp("model");

            let model = Qwen3Model::new(cfg, vb)
                .map_err(|e| anyhow::anyhow!("Failed to create Qwen3Model: {:?}", e))?;
            Model::Full(model)
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }

    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self
            .embed(vec![text.to_string()])
            .await
            .map_err(|e| anyhow::anyhow!("Embedding error: {:?}", e))?;
        Ok(embeddings
            .into_iter()
            .next()
            .context("No embedding returned")?)
    }
}

#[async_trait]
impl EmbeddingModel for Embedder {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LanguageModelError> {
        let mut model_guard = self.model.lock().await;

        // Zembed-1 specific formatting: append <|im_end|>\n
        let augmented_texts: Vec<String> =
            input.iter().map(|t| format!("{}<|im_end|>\n", t)).collect();

        let encodings = self
            .tokenizer
            .encode_batch(augmented_texts, true)
            .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

        if encodings.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = encodings.len();
        let seq_len = encodings[0].len();

        match &mut *model_guard {
            Model::Full(model) => {
                let device = model.device();
                let mut input_ids_vec = Vec::with_capacity(batch_size * seq_len);
                let mut attention_mask_vec = Vec::with_capacity(batch_size * seq_len);

                for enc in &encodings {
                    input_ids_vec.extend(enc.get_ids().iter().copied());
                    attention_mask_vec.extend(enc.get_attention_mask().iter().map(|&m| m as f32));
                }

                let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, seq_len), device)
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?;
                let attention_mask_2d =
                    Tensor::from_vec(attention_mask_vec, (batch_size, seq_len), device)
                        .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                let attention_mask_4d = build_attention_mask_4d(&attention_mask_2d)
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                let hidden = model
                    .forward(&input_ids, Some(&attention_mask_4d))
                    .map_err(|e| LanguageModelError::permanent(format!("{:?}", e)))?;

                // Last token pooling
                let pooled = hidden
                    .i((.., seq_len - 1))
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                let normalized = l2_normalize(&pooled)
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                let data = normalized
                    .to_dtype(DType::F32)
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?
                    .to_vec2::<f32>()
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                Ok(data)
            }
            Model::Quantized(model) => {
                // For quantized model, we process one by one for simplicity as candle's ModelWeights
                // usually expects single sequence or handled differently for batching.
                // Also, zembed-1 is last token pooling.
                let mut results = Vec::with_capacity(batch_size);

                for enc in &encodings {
                    let ids = enc.get_ids();
                    let input_ids = Tensor::new(ids, &self.device)
                        .map_err(|e| LanguageModelError::permanent(e.to_string()))?
                        .unsqueeze(0)
                        .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                    // forward returns logits for the LAST token
                    let pooled = model
                        .forward(&input_ids, 0)
                        .map_err(|e| LanguageModelError::permanent(format!("{:?}", e)))?;

                    let normalized = l2_normalize(&pooled)
                        .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                    let data = normalized
                        .to_dtype(DType::F32)
                        .map_err(|e| LanguageModelError::permanent(e.to_string()))?
                        .to_vec1::<f32>()
                        .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                    results.push(data);
                }

                Ok(results)
            }
        }
    }
}

impl Clone for Embedder {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            tokenizer: Arc::clone(&self.tokenizer),
            device: self.device.clone(),
        }
    }
}
