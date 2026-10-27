//! File-only prototype: cargo run --locked --example open_claims_prototype < packet.json
//! --claim-local isolates v1 items; --contextual selects v2; --compact selects compact v3.
//! --claim-local-v1 reproduces the earlier validation policy for saved v1 artifacts.
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
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (contextual, claim_local, compact, legacy_partial) = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => (false, false, false, false),
        ["--claim-local"] => (false, true, false, false),
        ["--claim-local-v1"] => (false, true, false, true),
        ["--contextual"] => (true, true, false, false),
        ["--compact"] => (true, true, true, false),
        _ => anyhow::bail!(
            "use no flags, --claim-local, --claim-local-v1, --contextual or --compact"
        ),
    };
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
    let prepare = if compact {
        open_claims_prototype::prepare_compact
    } else if contextual {
        open_claims_prototype::prepare_contextual
    } else {
        open_claims_prototype::prepare
    };
    let request = prepare(
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
        let validate = if legacy_partial {
            open_claims_prototype::validate_partial_legacy
        } else if claim_local {
            open_claims_prototype::validate_partial
        } else {
            open_claims_prototype::validate
        };
        json!({"binding":request.binding,"graph":validate(&body,&chunk,&request)?,"canonical_writes":0})
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
