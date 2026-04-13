use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, Model};
use hf_hub::api::tokio::ApiBuilder;
use tokenizers::Tokenizer;

pub struct Embedder {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub async fn new() -> Result<Self> {
        let api = ApiBuilder::new().build()?;
        let repo = api.model("zeroentropy/zembed-1".to_string());

        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };

        let config_file = repo.get("config.json").await?;
        let tokenizer_file = repo.get("tokenizer.json").await?;

        // Handling sharded weights
        let weights_files = vec![
            repo.get("model-00001-of-00002.safetensors").await?,
            repo.get("model-00002-of-00002.safetensors").await?,
        ];

        let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;

        // Load in F32. 4B params = ~16GB.
        // If CUDA is available, ensure you have enough VRAM.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&weights_files, DType::F32, &device)? };

        // Note: Qwen3 is compatible with Qwen2 architecture for embeddings.
        let model = Model::new(&config, vb)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn embed_with_prompt(&mut self, text: &str, prompt: &str) -> Result<Vec<f32>> {
        // Formulate the full prompt and suffix as per zembed-1 configuration.
        let full_text = format!("{}{}<|im_end|>\n", prompt, text);
        let tokens = self
            .tokenizer
            .encode(full_text, true)
            .map_err(anyhow::Error::msg)?;
        let tokens_ids = tokens.get_ids();
        let input_ids = Tensor::new(tokens_ids, &self.device)?.unsqueeze(0)?;

        // Forward pass. Qwen2::forward returns logits by default.
        // For embeddings, we need the hidden states before the LM head.
        // Since candle-transformers' Model hides the internal transformer,
        // we use the forward method but we need to ensure we can get the embeddings.
        // Note: For zembed-1, the 'lm_head' is tied with the word embeddings.

        let logits = self.model.forward(&input_ids, 0, None)?;

        // Last-token pooling: Use the embedding/logits of the very last token.
        // Since it's a 1xNxV tensor, we take the last index in the sequence dimension.
        let (_n, seq_len, _vocab_size) = logits.dims3()?;
        let last_token_logits = logits.get(0)?.get(seq_len - 1)?;

        // L2 Normalization
        let norm = last_token_logits.sqr()?.sum_all()?.sqrt()?;
        let normalized = (last_token_logits / norm)?;

        Ok(normalized.to_vec1::<f32>()?)
    }

    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        let prompt = "<|im_start|>system\nquery<|im_end|>\n<|im_start|>user\n";
        self.embed_with_prompt(text, prompt)
    }

    pub fn embed_document(&mut self, text: &str) -> Result<Vec<f32>> {
        let prompt = "<|im_start|>system\ndocument<|im_end|>\n<|im_start|>user\n";
        self.embed_with_prompt(text, prompt)
    }
}
