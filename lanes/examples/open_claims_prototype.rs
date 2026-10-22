//! File-only prototype: cargo run --locked --example open_claims_prototype < packet.json
//! No credentials, model client, database connection or production integration.
use anyhow::{Result, ensure};
use aurelius_lanes::{
    contracts::{ByteRange, Chunk, Document},
    fact_preparation::PreparedFactSource,
    open_claims_prototype,
};
use serde::Deserialize;
use serde_json::json;
use std::io::{Read, Write};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    document: Document,
    range: ByteRange,
    prepared: PreparedFactSource,
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
        "prototype packet exceeds bound"
    );
    let input: Input = serde_json::from_slice(&bytes)?;
    let chunk = Chunk {
        document: input.document,
        range: input.range,
    };
    let request = open_claims_prototype::prepare(
        &chunk,
        &input.prepared,
        &input.base_profile,
        input.max_output_tokens.unwrap_or(16384),
    )?;
    let result = if let Some(body) = input.body {
        ensure!(
            input.expected_binding.as_deref() == Some(&request.binding),
            "prototype binding mismatch"
        );
        json!({"binding":request.binding,"graph":open_claims_prototype::validate(&body,&chunk,&request)?,"canonical_writes":0})
    } else {
        ensure!(
            input.expected_binding.is_none(),
            "unexpected preparation binding"
        );
        json!({"request":request,"canonical_writes":0})
    };
    serde_json::to_writer(std::io::stdout().lock(), &result)?;
    std::io::stdout().write_all(b"\n")?;
    Ok(())
}
