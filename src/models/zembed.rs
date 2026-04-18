use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use fastembed::{Qwen3Config, Qwen3Model};
use hf_hub::api::tokio::ApiBuilder;
use ndarray::{Array2, ArrayViewD};
use ort::session::Session;
use ort::session::SessionInputValue;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::borrow::Cow;
use std::sync::Arc;
use swiftide::chat_completion::errors::LanguageModelError;
use swiftide::traits::EmbeddingModel;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info};

pub enum Model {
    Full(Qwen3Model),
    Quantized(Session),
    Ollama(swiftide::integrations::ollama::Ollama),
    Lemonade {
        client: reqwest::Client,
        base_url: String,
        model: String,
    },
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

fn l2_normalize_tensor(t: &Tensor) -> candle_core::Result<Tensor> {
    let sum_sq = t.sqr()?.sum_keepdim(1)?;
    t.broadcast_div(&sum_sq.sqrt()?)
}

// Helper for manual L2 normalization of ndarrays (used for ORT output)
fn l2_normalize_vec(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let eps = 1e-12;
    v.iter_mut().for_each(|x| *x /= norm + eps);
    v
}

fn build_attention_mask_4d(mask: &Tensor) -> candle_core::Result<Tensor> {
    let (b_sz, seq_len) = mask.dims2()?;
    let mask = mask.reshape((b_sz, 1, 1, seq_len))?;
    let mask = (mask.to_dtype(DType::F32)? - 1.0)?;
    let mask = (mask * 1e9)?;
    Ok(mask)
}

impl Embedder {
    pub async fn new(quantized: bool, ollama: bool, lemonade: Option<(&str, &str)>) -> Result<Self> {
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

        let model = if let Some((base_url, embed_model)) = lemonade {
            info!("Initializing Lemonade embeddings (model: {embed_model}, url: {base_url})...");
            Model::Lemonade {
                client: reqwest::Client::new(),
                base_url: base_url.to_string(),
                model: embed_model.to_string(),
            }
        } else if ollama {
            if !quantized {
                anyhow::bail!("ollama + non-quantized is not supported right now");
            }
            info!("Initializing Ollama hf.co/Abiray/zembed-1-Q4_K_M-GGUF...");
            let ollama_client = swiftide::integrations::ollama::Ollama::builder()
                .default_embed_model("hf.co/Abiray/zembed-1-Q4_K_M-GGUF")
                .build()?;
            Model::Ollama(ollama_client)
        } else if quantized {
            info!("Initializing 4-bit ONNX zembed-1...");
            // Load your local 4-bit ONNX file
            let model_path = "zembed_base/model_int4.onnx";

            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(num_cpus::get())?
                .commit_from_file(model_path)
                .context("Failed to load ONNX model. Ensure model_int4.onnx exists.")?;

            Model::Quantized(session)
        } else {
            info!("Initializing full zembed-1 (Candle)...");
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
            let model = Qwen3Model::new(cfg, vb.pp("model"))
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
        self.embed(vec![text.to_string()])
            .await
            .map_err(|e| anyhow::anyhow!("Embedding error: {:?}", e))?
            .into_iter()
            .next()
            .context("No embedding returned")
    }
}

#[async_trait]
impl EmbeddingModel for Embedder {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LanguageModelError> {
        let mut model_guard = self.model.lock().await;

        let batch_size = input.len();
        let start = std::time::Instant::now();

        let augmented_texts: Vec<String> =
            input.iter().map(|t| format!("{}<|im_end|>\n", t)).collect();
        let encodings = self
            .tokenizer
            .encode_batch(augmented_texts, true)
            .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

        if encodings.is_empty() {
            return Ok(vec![]);
        }

        let total_tokens: usize = encodings.iter().map(|e| e.get_ids().len()).sum();
        let seq_len = encodings[0].len();

        match &mut *model_guard {
            Model::Ollama(ollama) => {
                info!(
                    "Submitting batch of {} chunks (~{} tokens) to Ollama",
                    batch_size, total_tokens
                );
                let result = ollama.embed(input).await;
                let duration = start.elapsed();
                let secs = duration.as_secs_f64();
                let tok_per_sec = if secs > 0.0 {
                    total_tokens as f64 / secs
                } else {
                    0.0
                };

                info!(
                    "Ollama embedding complete: {} chunks in {:.2?} ({:.2} tok/s)",
                    batch_size, duration, tok_per_sec
                );
                result
            }
            Model::Lemonade { client, base_url, model } => {
                info!(
                    "Submitting batch of {} chunks (~{} tokens) to Lemonade",
                    batch_size, total_tokens
                );

                #[derive(Serialize)]
                struct EmbedRequest<'a> {
                    model: &'a str,
                    input: &'a [String],
                }

                #[derive(Deserialize)]
                struct EmbedData {
                    embedding: Vec<f32>,
                }

                #[derive(Deserialize)]
                struct EmbedResponse {
                    data: Vec<EmbedData>,
                }

                let url = format!("{}/api/v1/embeddings", base_url);
                let resp = client
                    .post(&url)
                    .json(&EmbedRequest { model, input: &input })
                    .send()
                    .await
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    return Err(LanguageModelError::permanent(format!(
                        "Lemonade embeddings error {status}: {body}"
                    )));
                }

                let embed_resp: EmbedResponse = resp
                    .json()
                    .await
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                let duration = start.elapsed();
                let tok_per_sec = if duration.as_secs_f64() > 0.0 {
                    total_tokens as f64 / duration.as_secs_f64()
                } else {
                    0.0
                };
                info!(
                    "Lemonade embedding complete: {} chunks in {:.2?} ({:.2} tok/s)",
                    batch_size, duration, tok_per_sec
                );

                let embeddings: Vec<Vec<f32>> = embed_resp.data.into_iter().map(|d| d.embedding).collect();
                // Sort by original order (Lemonade returns in order, but be safe)
                if embeddings.len() != batch_size {
                    return Err(LanguageModelError::permanent(format!(
                        "Lemonade returned {} embeddings for {} inputs",
                        embeddings.len(),
                        batch_size
                    )));
                }
                Ok(embeddings)
            }
            Model::Full(model) => {
                info!(
                    "Using full model (Candle) for embedding batch of {} chunks",
                    batch_size
                );
                let mut input_ids_vec = Vec::with_capacity(batch_size * seq_len);
                let mut attention_mask_vec = Vec::with_capacity(batch_size * seq_len);
                for enc in &encodings {
                    input_ids_vec.extend(enc.get_ids().iter().copied());
                    attention_mask_vec.extend(enc.get_attention_mask().iter().map(|&m| m as f32));
                }
                let input_ids =
                    Tensor::from_vec(input_ids_vec, (batch_size, seq_len), &self.device)
                        .map_err(|e| LanguageModelError::permanent(e.to_string()))?;
                let attention_mask_2d =
                    Tensor::from_vec(attention_mask_vec, (batch_size, seq_len), &self.device)
                        .map_err(|e| LanguageModelError::permanent(e.to_string()))?;
                let attention_mask_4d = build_attention_mask_4d(&attention_mask_2d)
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                let hidden = model
                    .forward(&input_ids, Some(&attention_mask_4d))
                    .map_err(|e| LanguageModelError::permanent(format!("{:?}", e)))?;

                let pooled = hidden
                    .i((.., seq_len - 1))
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?;
                let normalized = l2_normalize_tensor(&pooled)
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                normalized
                    .to_dtype(DType::F32)
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?
                    .to_vec2::<f32>()
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))
            }
            Model::Quantized(session) => {
                debug!("Using quantized model (ORT) for embedding");
                // Prepare inputs for ONNX Runtime (requires i64 and ndarray)
                let ids_vec: Vec<i64> = encodings
                    .iter()
                    .flat_map(|e| e.get_ids().iter().map(|&i| i as i64))
                    .collect();
                let mask_vec: Vec<i64> = encodings
                    .iter()
                    .flat_map(|e| e.get_attention_mask().iter().map(|&m| m as i64))
                    .collect();

                let input_ids = Array2::from_shape_vec((batch_size, seq_len), ids_vec).map_err(
                    |e: ndarray::ShapeError| LanguageModelError::permanent(e.to_string()),
                )?;
                let attention_mask = Array2::from_shape_vec((batch_size, seq_len), mask_vec)
                    .map_err(|e: ndarray::ShapeError| {
                        LanguageModelError::permanent(e.to_string())
                    })?;

                let mut position_ids = Array2::<i64>::zeros((batch_size, seq_len));
                for b in 0..batch_size {
                    for s in 0..seq_len {
                        position_ids[[b, s]] = s as i64;
                    }
                }

                let input_ids_val = Value::from_array(input_ids)
                    .map_err(|e: ort::Error| LanguageModelError::permanent(e.to_string()))?;
                let attention_mask_val = Value::from_array(attention_mask)
                    .map_err(|e: ort::Error| LanguageModelError::permanent(e.to_string()))?;
                let position_ids_val = Value::from_array(position_ids)
                    .map_err(|e: ort::Error| LanguageModelError::permanent(e.to_string()))?;

                let outputs = session
                    .run(vec![
                        (
                            Cow::from("input_ids"),
                            SessionInputValue::from(input_ids_val),
                        ),
                        (
                            Cow::from("attention_mask"),
                            SessionInputValue::from(attention_mask_val),
                        ),
                        (
                            Cow::from("position_ids"),
                            SessionInputValue::from(position_ids_val),
                        ),
                    ])
                    .map_err(|e: ort::Error| LanguageModelError::permanent(e.to_string()))?;

                let hidden_states_val = outputs
                    .get("last_hidden_state")
                    .context("Missing last_hidden_state output")
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                let (shape, data) = hidden_states_val
                    .try_extract_tensor::<f32>()
                    .map_err(|e: ort::Error| LanguageModelError::permanent(e.to_string()))?;

                let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                let view = ArrayViewD::from_shape(shape_vec, data)
                    .map_err(|e| LanguageModelError::permanent(e.to_string()))?;

                let mut results = Vec::with_capacity(batch_size);
                for b in 0..batch_size {
                    let last_token_vec = view.slice(ndarray::s![b, seq_len - 1, ..]).to_vec();
                    results.push(l2_normalize_vec(last_token_vec));
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
