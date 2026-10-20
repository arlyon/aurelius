use crate::contracts::{ByteRange, Document, DocumentKey};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const LANGUAGE_VERSION: &str = "source-aligned-fcoref-v3";
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageConfig {
    pub model_dir: String,
    pub max_document_tokens: usize,
    pub threads: usize,
}
impl LanguageConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.model_dir.is_empty() && self.model_dir.len() <= 2048,
            "invalid language model directory"
        );
        ensure!(
            (2..=4096).contains(&self.max_document_tokens),
            "invalid language document token budget"
        );
        ensure!(
            (1..=2).contains(&self.threads),
            "language inference requires one or two threads"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Mention {
    pub id: String,
    pub range: ByteRange,
    pub text: String,
    pub noun: bool,
    // Includes compound spans containing a linguistically tagged pronoun.
    pub pronoun: bool,
    pub learned: bool,
    pub raw_mention_score: Option<f32>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageAnalysis {
    pub document: DocumentKey,
    pub content_hash: String,
    pub profile: String,
    pub mentions: Vec<Mention>,
    pub coreference_hypotheses: Vec<Vec<String>>,
    pub verb_spans: Vec<ByteRange>,
    pub number_spans: Vec<ByteRange>,
    #[serde(default)]
    pub sentence_spans: Vec<ByteRange>,
}
impl LanguageAnalysis {
    pub fn validate(&self, document: &Document) -> Result<()> {
        use std::collections::BTreeSet;
        ensure!(
            self.document == document.key && self.content_hash == document.content_hash(),
            "language analysis changed source identity"
        );
        ensure!(
            self.mentions.len() <= 2048 && self.coreference_hypotheses.len() <= 512,
            "language annotation budget exceeded"
        );
        let mut ids = BTreeSet::new();
        for mention in &self.mentions {
            ensure!(
                ids.insert(&mention.id) && mention.range.text(document)? == mention.text,
                "language mention is not original source text"
            );
            ensure!(
                mention.raw_mention_score.is_none_or(f32::is_finite)
                    && mention.learned == mention.raw_mention_score.is_some(),
                "invalid learned mention score"
            );
        }
        for cluster in &self.coreference_hypotheses {
            ensure!(
                cluster.len() >= 2 && cluster.iter().all(|id| ids.contains(id)),
                "invalid source-local coreference hypothesis"
            );
            ensure!(
                cluster.iter().collect::<BTreeSet<_>>().len() == cluster.len(),
                "duplicate coreference mention"
            );
        }
        for range in self
            .verb_spans
            .iter()
            .chain(&self.number_spans)
            .chain(&self.sentence_spans)
        {
            range.text(document)?;
        }
        Ok(())
    }
}

#[cfg(feature = "language")]
mod runtime {
    use super::*;
    use anno::{FCoref, FCorefConfig};
    use anyhow::Context;
    use std::{collections::BTreeMap, io::Read, path::Path};
    use tokenizers::Tokenizer;
    pub struct LanguageRuntime {
        coref: FCoref,
        nlp: nlprule::Tokenizer,
        tokenizer: Tokenizer,
        config: LanguageConfig,
        profile: String,
    }
    fn digest(path: &Path) -> Result<String> {
        let mut file = std::fs::File::open(path).context("local language artifact missing")?;
        ensure!(
            file.metadata()?.len() <= 512 * 1024 * 1024,
            "language artifact exceeds bound"
        );
        let mut hasher = blake3::Hasher::new();
        let mut buffer = [0; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
    impl LanguageRuntime {
        pub fn profile_for(config: &LanguageConfig) -> Result<String> {
            config.validate()?;
            let dir = Path::new(&config.model_dir);
            let encoder = if dir.join("encoder_quantized.onnx").is_file() {
                "encoder_quantized.onnx"
            } else {
                "encoder.onnx"
            };
            let mut artifacts = BTreeMap::new();
            for name in [
                encoder,
                "scorer_weights.safetensors",
                "tokenizer.json",
                "config.json",
                "en_tokenizer.bin",
            ] {
                artifacts.insert(name, digest(&dir.join(name))?);
            }
            Ok(blake3::hash(&serde_json::to_vec(&(
                LANGUAGE_VERSION,
                config.max_document_tokens,
                config.threads,
                artifacts,
            ))?)
            .to_hex()
            .to_string())
        }
        pub fn load(config: LanguageConfig) -> Result<Self> {
            let profile = Self::profile_for(&config)?;
            let dir = Path::new(&config.model_dir);
            let coref = FCoref::from_path_with_config(
                &config.model_dir,
                FCorefConfig {
                    num_threads: config.threads,
                    ..Default::default()
                },
            )?;
            let nlp = nlprule::Tokenizer::new(dir.join("en_tokenizer.bin"))?;
            ensure!(
                nlp.chunker().is_some(),
                "local noun-phrase chunker is missing"
            );
            let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
                .map_err(|_| anyhow::anyhow!("local language tokenizer invalid"))?;
            ensure!(
                tokenizer.get_truncation().is_none(),
                "local language tokenizer must not truncate source"
            );
            Ok(Self {
                coref,
                nlp,
                tokenizer,
                config,
                profile,
            })
        }
        pub fn profile(&self) -> &str {
            &self.profile
        }
        pub fn analyze(&self, document: &Document) -> Result<LanguageAnalysis> {
            document
                .validate()
                .context(crate::facts::ExtractionReview::Input)?;
            let mut mentions: BTreeMap<(usize, usize), Mention> = BTreeMap::new();
            let id = |range: ByteRange| -> Result<String> {
                Ok(blake3::hash(&serde_json::to_vec(&(
                    &document.key,
                    document.content_hash(),
                    &self.profile,
                    range,
                ))?)
                .to_hex()
                .to_string())
            };
            let mut words = Vec::new();
            let mut verbs = Vec::new();
            let mut numbers = Vec::new();
            let mut sentences = Vec::new();
            for sentence in self.nlp.pipe(&document.text) {
                let mut phrase = None;
                let mut sentence_start = None;
                let mut sentence_end = 0;
                for token in sentence.tokens() {
                    let range = token.span().byte();
                    if range.is_empty() {
                        continue;
                    }
                    let span = ByteRange {
                        start: range.start,
                        end: range.end,
                    };
                    ensure!(
                        span.text(document)? == token.word().text().as_str(),
                        "word segmentation rewrote source text"
                    );
                    sentence_start.get_or_insert(span.start);
                    sentence_end = span.end;
                    words.push((span.start, span.end));
                    ensure!(
                        words.len() <= self.config.max_document_tokens,
                        crate::facts::ExtractionReview::Input
                    );
                    let tags: Vec<_> = token
                        .word()
                        .tags()
                        .iter()
                        .map(|tag| tag.pos().as_str().split(':').next().unwrap_or(""))
                        .collect();
                    let noun = tags
                        .iter()
                        .any(|tag| matches!(*tag, "NN" | "NNS" | "NNP" | "NNPS"));
                    let pronoun = tags
                        .iter()
                        .any(|tag| matches!(*tag, "PRP" | "PRP$" | "WP" | "WP$"));
                    if noun || pronoun {
                        mentions.insert(
                            (span.start, span.end),
                            Mention {
                                id: id(span)?,
                                range: span,
                                text: span.text(document)?.into(),
                                noun,
                                pronoun,
                                learned: false,
                                raw_mention_score: None,
                            },
                        );
                    }
                    if tags
                        .iter()
                        .any(|tag| matches!(*tag, "VB" | "VBD" | "VBG" | "VBN" | "VBP" | "VBZ"))
                    {
                        verbs.push(span);
                    }
                    if tags.contains(&"CD") {
                        numbers.push(span);
                    }
                    if token
                        .chunks()
                        .iter()
                        .any(|label| label.split('-').take(2).eq(["B", "NP"]))
                    {
                        phrase = Some(span.start);
                    }
                    if token
                        .chunks()
                        .iter()
                        .any(|label| label.split('-').take(2).eq(["E", "NP"]))
                        && let Some(start) = phrase.take()
                    {
                        let range = ByteRange {
                            start,
                            end: span.end,
                        };
                        let text = range.text(document)?.to_owned();
                        mentions.entry((start, span.end)).or_insert(Mention {
                            id: id(range)?,
                            range,
                            text,
                            noun,
                            pronoun,
                            learned: false,
                            raw_mention_score: None,
                        });
                    }
                }
                if let Some(start) = sentence_start {
                    sentences.push(ByteRange {
                        start,
                        end: sentence_end,
                    });
                }
            }
            let surfaces: Vec<_> = words
                .iter()
                .map(|&(start, end)| &document.text[start..end])
                .collect();
            let encoding = self
                .tokenizer
                .encode(surfaces, true)
                .map_err(|_| anyhow::anyhow!("word tokenizer failed"))?;
            ensure!(
                encoding.len() <= self.config.max_document_tokens,
                crate::facts::ExtractionReview::Input
            );
            let analysis = self.coref.analyze_words(&document.text, &words)?;
            for mention in analysis.mentions {
                let range = ByteRange::from_codepoints(
                    document,
                    mention.character_start,
                    mention.character_end,
                )?;
                ensure!(
                    range.text(document)? == mention.text,
                    "learned mention changed source span"
                );
                let entry = mentions.entry((range.start, range.end)).or_insert(Mention {
                    id: id(range)?,
                    range,
                    text: mention.text,
                    noun: false,
                    pronoun: false,
                    learned: true,
                    raw_mention_score: Some(mention.raw_mention_score),
                });
                entry.learned = true;
                entry.raw_mention_score = Some(mention.raw_mention_score);
            }
            let mut clusters = Vec::new();
            for cluster in analysis.clusters {
                let mut ids = Vec::new();
                for (start, end) in cluster.spans {
                    let range = ByteRange::from_codepoints(document, start, end)?;
                    ids.push(
                        mentions
                            .get(&(range.start, range.end))
                            .context("coreference mention not found")?
                            .id
                            .clone(),
                    );
                }
                ids.sort();
                ids.dedup();
                if ids.len() >= 2 {
                    clusters.push(ids);
                }
            }
            let pronouns: Vec<_> = mentions
                .values()
                .filter(|mention| mention.pronoun)
                .map(|mention| mention.range)
                .collect();
            for mention in mentions.values_mut() {
                mention.pronoun |= pronouns.iter().any(|range| {
                    range.start >= mention.range.start && range.end <= mention.range.end
                });
            }
            let result = LanguageAnalysis {
                document: document.key.clone(),
                content_hash: document.content_hash(),
                profile: self.profile.clone(),
                mentions: mentions.into_values().collect(),
                coreference_hypotheses: clusters,
                verb_spans: verbs,
                number_spans: numbers,
                sentence_spans: sentences,
            };
            result.validate(document)?;
            Ok(result)
        }
    }
}
#[cfg(feature = "language")]
pub use runtime::LanguageRuntime;
