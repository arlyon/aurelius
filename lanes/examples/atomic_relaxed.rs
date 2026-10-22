//! Credential-free, file-only preparation and validation for the approved span experiment.
use anyhow::{Context, Result, ensure};
use aurelius_lanes::{
    atomic::{self, AnchorKind, Registry},
    contracts::{ByteRange, Chunk, Document},
    fact_preparation::PreparedFactSource,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{Read, Write};

const POLICY: &str = "experimental-relaxed-source-spans-v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    document: Document,
    range: ByteRange,
    prepared: PreparedFactSource,
    registry: Registry,
    base_profile: String,
    max_output_tokens: Option<u32>,
    body: Option<String>,
    expected_binding: Option<String>,
}
fn main() -> Result<()> {
    let mut bytes = Vec::new();
    std::io::stdin()
        .take(4 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 4 * 1024 * 1024,
        "experiment input exceeds bound"
    );
    let input: Input = serde_json::from_slice(&bytes).context("invalid experiment input")?;
    let chunk = Chunk {
        document: input.document,
        range: input.range,
    };
    let mut request = atomic::prepare_request(
        &chunk,
        &input.prepared,
        &input.registry,
        &input.base_profile,
        input.max_output_tokens.unwrap_or(8192),
    )?;
    let hints = json!({"mentions":request.catalog.anchor_hrefs(&AnchorKind::Mention),"actions":request.catalog.anchor_hrefs(&AnchorKind::Action)});
    request.catalog.relax_entity_anchors(&chunk)?;
    // Shared enums avoid repeating every source token for every entity type.
    // This is schema factoring, not JSON mode or a relaxation of native strict output.
    let allowed = request.catalog.anchor_hrefs(&AnchorKind::Mention);
    for variant in request.schema["properties"]["local_entities"]["items"]["anyOf"]
        .as_array_mut()
        .context("entity schema absent")?
    {
        let kind = variant["properties"]["entity_type"]["const"]
            .as_str()
            .context("entity type absent")?;
        if !matches!(input.registry.entity_anchors[kind], AnchorKind::Record) {
            variant["properties"]["identity_span_ref"] = json!({"$ref":"#/$defs/source_identity"});
        }
    }
    request.schema["$defs"]["source_identity"] = json!({"type":"string","enum":allowed});
    request.prompt.push_str("\nExperimental source-span policy: NLP mention/action tags are optional hints, not identity restrictions. Choose the shortest issued span that actually denotes the declared entity, not an unrelated token or a sentence merely containing its name. Local IDs are opaque references, not proof of identity. Retain the same strict predicate, graph, qualification, source-byte and review contracts.");
    let mut model_input: Value = serde_json::from_str(&request.input)?;
    model_input["span_links"] = json!(request.catalog.links);
    model_input["source_text"] = json!(chunk.text()?);
    model_input["nlp_hints_not_authority"] = hints;
    request.input = serde_json::to_string(&model_input)?;
    request.profile = blake3::hash(&serde_json::to_vec(&(POLICY, &request.profile))?)
        .to_hex()
        .to_string();
    request.binding = blake3::hash(&serde_json::to_vec(&(
        POLICY,
        &request.profile,
        &request.schema,
        &request.prompt,
        &request.input,
        request.max_output_tokens,
    ))?)
    .to_hex()
    .to_string();
    let result = if let Some(body) = input.body {
        ensure!(
            input.expected_binding.as_deref() == Some(&request.binding),
            "experimental request binding mismatch"
        );
        let graph = atomic::validate(&body, &request.catalog, &input.registry)?;
        json!({"binding":request.binding,"graph":graph,"canonical_writes":0})
    } else {
        ensure!(
            input.expected_binding.is_none(),
            "unexpected preparation binding"
        );
        json!({"policy":POLICY,"request":request,"canonical_writes":0})
    };
    serde_json::to_writer(std::io::stdout().lock(), &result)?;
    std::io::stdout().write_all(b"\n")?;
    Ok(())
}
