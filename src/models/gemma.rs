use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gemma::{Config, Model};
use hf_hub::api::tokio::ApiBuilder;
use ollama_client::OllamaClient;
use ollama_client::types::Options;
use std::io::{self, Write};
use std::pin::pin;
use tokenizers::Tokenizer;
use tokio_stream::StreamExt;
use tracing::{debug, info};

pub enum ModelKind {
    Full {
        model: Model,
        tokenizer: Tokenizer,
        device: Device,
    },
    Ollama {
        client: OllamaClient,
        model: String,
    },
}

pub struct Gemma {
    kind: ModelKind,
    buffer: String,
    in_thinking: bool,
}

impl Gemma {
    pub async fn new(ollama: bool) -> Result<Self> {
        let kind = if ollama {
            let model_name = "gemma4:26b";
            info!("Initializing Ollama Gemma ({model_name})...");
            let client = OllamaClient::new();
            ModelKind::Ollama {
                client,
                model: model_name.to_string(),
            }
        } else {
            info!("Initializing Gemma model (Candle)...");
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

            debug!("Fetching model files from HF Hub...");
            let config_file = repo.get("config.json").await?;
            let tokenizer_file = repo.get("tokenizer.json").await?;
            let w1 = repo.get("model-00001-of-00002.safetensors").await?;
            let w2 = repo.get("model-00002-of-00002.safetensors").await?;

            debug!("Loading config and tokenizer...");
            let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
            let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;

            debug!("Loading weights (this may take a moment)...");
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[w1, w2], candle_core::DType::F32, &device)?
            };

            let model = Model::new(false, &config, vb)?;
            info!("Gemma model initialized successfully.");

            ModelKind::Full {
                model,
                tokenizer,
                device,
            }
        };

        Ok(Self {
            kind,
            buffer: String::new(),
            in_thinking: false,
        })
    }

    pub async fn generate(
        &mut self,
        prompt: &str,
        max_len: usize,
        thinking: bool,
    ) -> Result<String> {
        let mut full_response = String::new();
        self.in_thinking = false;
        self.buffer.clear();

        // Take kind out of self to avoid borrow checker issues with self.process_token
        let mut kind = std::mem::replace(
            &mut self.kind,
            ModelKind::Ollama {
                client: OllamaClient::new(),
                model: "".to_string(),
            },
        );

        let res: Result<()> = match &mut kind {
            ModelKind::Ollama { client, model } => {
                debug!("Starting Ollama streaming generation with model: {}", model);

                let mut options = Options::default();
                options.temperature = Some(1.0);
                options.top_p = Some(0.95);
                options.top_k = Some(64);

                let request = client
                    .generate()
                    .model(model.as_str())
                    .prompt(prompt)
                    .think(thinking)
                    .options(options);

                let stream = request
                    .send_stream()
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to start Ollama stream: {}", e))?;

                let mut stream = pin!(stream);

                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.map_err(|e| anyhow::anyhow!("Ollama stream error: {}", e))?;

                    // Direct access to thinking field from ollama-client's GenerateStreamChunk
                    if let Some(thought) = &chunk.thinking {
                        if !thought.is_empty() {
                            if !self.in_thinking {
                                print!("Thinking...\n\x1b[90;3m");
                                io::stdout().flush()?;
                                self.in_thinking = true;
                            }
                            print!("{thought}");
                            full_response.push_str(thought);
                            io::stdout().flush()?;
                            continue;
                        }
                    }

                    let response = &chunk.response;
                    if !response.is_empty() {
                        if self.in_thinking {
                            print!("\x1b[0m\n...done thinking.\n\n");
                            io::stdout().flush()?;
                            self.in_thinking = false;
                        }
                        let output = self.process_token(response);
                        if !output.is_empty() {
                            print!("{output}");
                            full_response.push_str(&output);
                            io::stdout().flush()?;
                        }
                    }
                }
                Ok(())
            }
            ModelKind::Full {
                model,
                tokenizer,
                device,
            } => {
                debug!("Starting Candle generation (thinking: {})", thinking);
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

                let mut tokens = tokenizer
                    .encode(full_prompt, true)
                    .map_err(anyhow::Error::msg)?
                    .get_ids()
                    .to_vec();
                let mut logits_processor = LogitsProcessor::new(1337, Some(1.0), Some(0.95));

                for _ in 0..max_len {
                    let input = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
                    let logits = model.forward(&input, tokens.len() - 1)?;
                    let logits = logits.squeeze(0)?;
                    let token = logits_processor.sample(&logits)?;
                    tokens.push(token);

                    if let Ok(text) = tokenizer.decode(&[token], true) {
                        let output = self.process_token(&text);
                        if !output.is_empty() {
                            print!("{output}");
                            full_response.push_str(&output);
                            io::stdout().flush()?;
                        }
                    }

                    if token == 1 {
                        break;
                    }
                }
                Ok(())
            }
        };

        // Put kind back
        self.kind = kind;
        res?;

        if self.in_thinking {
            print!("\x1b[0m\n...done thinking.\n\n");
            io::stdout().flush()?;
            self.in_thinking = false;
        }

        if !self.buffer.is_empty() {
            let last = self.buffer.clone();
            print!("{last}");
            full_response.push_str(&last);
            io::stdout().flush()?;
        }

        println!();
        Ok(full_response)
    }

    fn process_token(&mut self, token: &str) -> String {
        self.buffer.push_str(token);
        let mut output = String::new();

        loop {
            if !self.in_thinking {
                if let Some(pos) = self.buffer.find("<think>") {
                    output.push_str(&self.buffer[..pos]);
                    output.push_str("Thinking...\n\x1b[90;3m");
                    self.in_thinking = true;
                    self.buffer.drain(..pos + 7);
                    continue;
                }
            } else if let Some(pos) = self.buffer.find("</think>") {
                output.push_str(&self.buffer[..pos]);
                output.push_str("\x1b[0m\n...done thinking.\n\n");
                self.in_thinking = false;
                self.buffer.drain(..pos + 8);
                continue;
            }

            let tag = if self.in_thinking {
                "</think>"
            } else {
                "<think>"
            };
            let tag_len = tag.len();

            if self.buffer.len() > tag_len {
                let safe_len = self.buffer.len() - tag_len + 1;
                let to_output = self.buffer[..safe_len].to_string();

                output.push_str(&to_output);
                self.buffer.drain(..safe_len);
            }
            break;
        }

        output
    }
}
