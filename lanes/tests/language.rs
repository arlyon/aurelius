#![cfg(feature = "language")]
use anyhow::{Context, Result, ensure};
use aurelius_lanes::{
    contracts::{Document, DocumentKey},
    language::{LanguageConfig, LanguageRuntime},
};

fn document(text: &str) -> Document {
    Document {
        key: DocumentKey {
            scope: "fixture-org".into(),
            source: "fixture-source".into(),
            revision: "fixture-revision".into(),
        },
        text: text.into(),
        context: None,
        metadata: Default::default(),
    }
}
#[test]
fn local_neural_annotations_are_source_aligned_and_not_identity_merges() -> Result<()> {
    let model_dir = std::env::var("AURELIUS_TEST_LANGUAGE_MODEL_DIR")
        .context("set AURELIUS_TEST_LANGUAGE_MODEL_DIR to the reviewed local FCoref assets; no download or heuristic fallback")?;
    let runtime = LanguageRuntime::load(LanguageConfig {
        model_dir: model_dir.clone(),
        max_document_tokens: 2048,
        threads: 2,
    })?;
    let source = document("Alice reviewed the campaign. She approved its monthly budget.");
    let analysis = runtime.analyze(&source)?;
    analysis.validate(&source)?;
    ensure!(
        analysis
            .mentions
            .iter()
            .any(|mention| mention.learned && mention.raw_mention_score.is_some()),
        "neural mention head produced no annotations"
    );
    ensure!(
        analysis
            .mentions
            .iter()
            .any(|mention| mention.pronoun && mention.text == "She"),
        "linguistic pronoun annotation missing"
    );
    let pronoun = analysis
        .mentions
        .iter()
        .find(|mention| mention.text == "She")
        .context("source pronoun missing")?;
    for mention in &analysis.mentions {
        if mention.range.start <= pronoun.range.start && mention.range.end >= pronoun.range.end {
            ensure!(
                mention.pronoun,
                "compound learned span lost its linguistic pronoun annotation"
            );
        }
    }
    // nlprule exposes lexical POS alternatives here, not contextual certainty.
    // Preserve the known recall cost of the conservative conflict gate rather
    // than treating an ambiguous placeholder as a clean named-entity control.
    let ambiguous = runtime.analyze(&document("Alice manages Company."))?;
    let company = ambiguous
        .mentions
        .iter()
        .find(|mention| mention.text == "Company")
        .context("ambiguous noun fixture missing")?;
    ensure!(
        company.noun && ambiguous.verb_spans.contains(&company.range),
        "pinned lexical-ambiguity fixture changed"
    );
    let again = runtime.analyze(&source)?;
    ensure!(
        serde_json::to_value(&analysis)? == serde_json::to_value(&again)?,
        "same local model/source changed annotations"
    );
    for text in [
        "Émilie reviewed the proposal. She approved it.",
        "Ok. Thank you! Happy Friday.",
        "Alice: I approved the budget.\nBob: I rejected the proposal.",
    ] {
        let source = document(text);
        let result = runtime.analyze(&source)?;
        result.validate(&source)?;
        let serialized = serde_json::to_value(&result)?;
        ensure!(
            serialized.get("canonical_entities").is_none()
                && serialized.get("identity_merges").is_none()
        );
    }
    let mut tampered = analysis.clone();
    tampered.mentions[0].text = "not in source".into();
    ensure!(tampered.validate(&source).is_err());
    let oversized = document(&"Unrelated context. ".repeat(900));
    ensure!(
        runtime.analyze(&oversized).is_err(),
        "oversized source was silently truncated"
    );
    let tiny = LanguageRuntime::load(LanguageConfig {
        model_dir,
        max_document_tokens: 8,
        threads: 2,
    })?;
    ensure!(
        runtime.profile() != tiny.profile(),
        "language budget omitted from profile"
    );
    ensure!(tiny.analyze(&source).is_err());
    Ok(())
}
