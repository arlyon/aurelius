use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gemma::{Config, Model};
use hf_hub::api::tokio::ApiBuilder;
use tokenizers::Tokenizer;
use tracing::info;

pub struct Gemma {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl Gemma {
    pub async fn new() -> Result<Self> {
        let api = ApiBuilder::new().build()?;
        let repo = api.model("google/gemma-1.1-2b-it".to_string());

        let device = if candle_core::utils::cuda_is_available() {
            let d = Device::new_cuda(0).unwrap_or(Device::Cpu);
            info!("Using device: {:?}", d);
            d
        } else if candle_core::utils::metal_is_available() {
            let d = Device::new_metal(0).unwrap_or(Device::Cpu);
            info!("Using device: {:?}", d);
            d
        } else {
            info!("Using device: CPU");
            Device::Cpu
        };

        let config_file = repo.get("config.json").await?;
        let tokenizer_file = repo.get("tokenizer.json").await?;
        let w1 = repo.get("model-00001-of-00002.safetensors").await?;
        let w2 = repo.get("model-00002-of-00002.safetensors").await?;

        let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[w1, w2], candle_core::DType::F32, &device)?
        };

        let model = Model::new(false, &config, vb)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn generate(&mut self, prompt: &str, max_len: usize, thinking: bool) -> Result<String> {
        // Gemma 4 supports native thinking mode with <|think|> token
        let full_prompt = if thinking {
            format!(
                "<|im_start|>system\n<|think|>\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt
            )
        } else {
            format!(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt
            )
        };

        let mut tokens = self
            .tokenizer
            .encode(full_prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let mut logits_processor = LogitsProcessor::new(1337, Some(1.0), Some(0.95)); // Gemma 4 recommended: temp 1.0, top_p 0.95
        let mut generated_text = String::new();

        for _i in 0..max_len {
            let input = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() - 1)?;
            let logits = logits.squeeze(0)?;
            let token = logits_processor.sample(&logits)?;
            tokens.push(token);

            if let Ok(text) = self.tokenizer.decode(&[token], true) {
                generated_text.push_str(&text);
            }

            if token == 1 {
                // Assuming 1 is EOS
                break;
            }
        }

        Ok(generated_text)
    }
}
