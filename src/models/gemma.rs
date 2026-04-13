use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gemma::{Config, Model};
use hf_hub::api::tokio::ApiBuilder;
use tokenizers::Tokenizer;

pub struct Gemma {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl Gemma {
    pub async fn new() -> Result<Self> {
        let api = ApiBuilder::new().build()?;
        let repo = api.model("google/gemma-2-2b-it".to_string());

        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };

        let config_file = repo.get("config.json").await?;
        let tokenizer_file = repo.get("tokenizer.json").await?;
        let weights_files = vec![
            repo.get("model-00001-of-00002.safetensors").await?,
            repo.get("model-00002-of-00002.safetensors").await?,
        ];

        let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_files, candle_core::DType::F32, &device)? };

        let model = Model::new(false, &config, vb)?;

        Ok(Self { model, tokenizer, device })
    }

    pub fn generate(&mut self, prompt: &str, max_len: usize) -> Result<String> {
        let mut tokens = self.tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        let mut logits_processor = LogitsProcessor::new(1337, Some(0.7), None);
        let mut generated_text = String::new();

        for _i in 0..max_len {
            let input = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() - 1)?;
            let logits = logits.squeeze(0)?;
            let token = logits_processor.sample(&logits)?;
            tokens.push(token);

            if let Some(text) = self.tokenizer.decode(&[token], true).ok() {
                generated_text.push_str(&text);
            }

            if token == 1 { // Assuming 1 is EOS
                break;
            }
        }

        Ok(generated_text)
    }
}
