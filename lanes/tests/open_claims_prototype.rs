use anyhow::{Context, Result};
use aurelius_lanes::{
    contracts::{ByteRange, Chunk, Document, DocumentKey},
    fact_context::FactContext,
    fact_preparation::PreparedFactSource,
    language::LanguageAnalysis,
    open_claims_prototype::{self as open, PreparedRequest},
};
use serde_json::{Value, json};
fn fixture(text: &str) -> Result<(Chunk, PreparedFactSource)> {
    let document = Document {
        key: DocumentKey {
            scope: "tenant".into(),
            source: "source".into(),
            revision: "1".into(),
        },
        text: text.into(),
        context: None,
        metadata: Default::default(),
    };
    let analysis = LanguageAnalysis {
        document: document.key.clone(),
        content_hash: document.content_hash(),
        profile: "a".repeat(64),
        mentions: vec![],
        coreference_hypotheses: vec![],
        verb_spans: vec![],
        number_spans: vec![],
        sentence_spans: vec![],
    };
    let prepared = PreparedFactSource::from_analysis(
        &document,
        analysis,
        vec![],
        FactContext {
            organization_id: "tenant".into(),
            entities: vec![],
        },
        "2026-01-01T00:00:00Z".parse()?,
        "b".repeat(64),
    )?;
    Ok((
        Chunk {
            range: ByteRange {
                start: 0,
                end: document.text.len(),
            },
            document,
        },
        prepared,
    ))
}
fn unit(request: &PreparedRequest, text: &str) -> Result<String> {
    request
        .units
        .iter()
        .find(|(_, s)| s.quote == text)
        .map(|(id, _)| id.clone())
        .context("test source unit absent")
}
fn quote(unit: &str, text: &str) -> Value {
    json!({"unit_ref":unit,"exact_quote":text})
}
fn claim(unit: &str, subject: &str, relation: &str, object: &str) -> Value {
    json!({"subject":{"source":quote(unit,subject),"type_hint":null},"relation":relation,"object":{"source":quote(unit,object),"type_hint":null},"qualification":{"mode":"asserted","polarity":"positive","attribution":[],"conditions":[],"temporal":[]},"evidence":[unit]})
}
fn body(claims: Vec<Value>) -> Value {
    json!({"claims":claims,"unresolved":[]})
}
#[test]
fn full_phrases_free_relations_and_source_literals_need_no_ontology() -> Result<()> {
    let (chunk, prepared) = fixture(
        "Header\nMegan manages client communications.\nCreate Training Agendas.\nGoogle Ads spent €150.\n",
    )?;
    let request = open::prepare(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let a = unit(&request, "Megan manages client communications.")?;
    let b = unit(&request, "Create Training Agendas.")?;
    let c = unit(&request, "Google Ads spent €150.")?;
    let mut one = claim(&a, "Megan", "manages", "client communications");
    one["object"]["type_hint"] = json!("communications_workflow");
    let mut two = claim(&b, "Create", "targets", "Training Agendas");
    two["qualification"]["mode"] = json!("requested");
    let value = body(vec![one, two, claim(&c, "Google Ads", "spent", "€150")]);
    let graph = open::validate(&value.to_string(), &chunk, &request)?;
    assert_eq!(graph["claims"].as_array().unwrap().len(), 3);
    assert_eq!(
        graph["claims"][0]["object_span"]["quote"],
        "client communications"
    );
    assert_eq!(
        graph["claims"][2]["object_span"]["range"]["end"]
            .as_u64()
            .unwrap()
            - graph["claims"][2]["object_span"]["range"]["start"]
                .as_u64()
                .unwrap(),
        "€150".len() as u64
    );
    assert_eq!(graph["canonical_writes"], 0);
    // Schema/bytes do not establish truth: no handwritten initial-spend veto.
    let mut wrong = value;
    wrong["claims"][2]["relation"] = json!("initial_spend");
    assert!(open::validate(&wrong.to_string(), &chunk, &request).is_ok());
    Ok(())
}
#[test]
fn partial_roles_unknowns_and_arbitrary_source_conditions_are_retained() -> Result<()> {
    let text = "Megan: Send the plan if no response by tomorrow.";
    let (chunk, prepared) = fixture(text)?;
    let request = open::prepare(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let u = unit(&request, text)?;
    let mut c = claim(&u, "Send", "targets", "the plan");
    c["qualification"]["mode"] = json!("requested");
    c["qualification"]["conditions"] = json!([quote(&u, "if no response")]);
    c["qualification"]["temporal"] = json!([quote(&u, "by tomorrow")]);
    let graph = open::validate(&body(vec![c]).to_string(), &chunk, &request)?;
    assert_eq!(
        graph["claims"][0]["condition_spans"][0]["quote"],
        "if no response"
    );
    assert!(graph["claims"][0]["claim"]["subject"]["type_hint"].is_null());
    Ok(())
}
#[test]
fn quote_changes_ambiguity_and_unissued_units_fail_without_repair() -> Result<()> {
    let text = "Google Ads spent €150; aa overlaps aaa.";
    let (chunk, prepared) = fixture(text)?;
    let request = open::prepare(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let u = unit(&request, text)?;
    for object in ["€151", "150 EUR", "aa"] {
        assert!(
            open::validate(
                &body(vec![claim(&u, "Google Ads", "spent", object)]).to_string(),
                &chunk,
                &request
            )
            .is_err()
        );
    }
    let mut c = claim(&u, "Google Ads", "spent", "€150");
    c["object"]["source"]["unit_ref"] = json!("span:999-1000");
    assert!(open::validate(&body(vec![c]).to_string(), &chunk, &request).is_err());
    Ok(())
}
#[test]
fn changed_source_scope_range_context_or_request_binding_is_rejected() -> Result<()> {
    let (chunk, prepared) = fixture("Header\nGoogle Ads spent €150.\n")?;
    let request = open::prepare(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let empty = body(vec![]).to_string();
    for change in [0, 1, 2] {
        let mut altered = chunk.clone();
        match change {
            0 => altered.document.key.scope = "other".into(),
            1 => altered.document.text.push('!'),
            _ => altered.range.start = 1,
        };
        assert!(open::validate(&empty, &altered, &request).is_err());
        assert!(open::prepare(&altered, &prepared, &"c".repeat(64), 16384).is_err() || change == 2);
    }
    let mut changed = prepared.clone();
    changed.canonical_context.organization_id = "other".into();
    assert!(open::prepare(&chunk, &changed, &"c".repeat(64), 16384).is_err());
    let mut request = request;
    request.binding = "0".repeat(64);
    assert!(open::validate(&empty, &chunk, &request).is_err());
    Ok(())
}
#[test]
fn envelope_array_and_nullable_constraints_remain_strict() -> Result<()> {
    let (chunk, prepared) = fixture("Google Ads spent €150.")?;
    let request = open::prepare(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let u = unit(&request, &chunk.document.text)?;
    let c = claim(&u, "Google Ads", "spent", "€150");
    assert_eq!(request.schema["properties"]["claims"]["maxItems"], 128);
    assert!(open::validate(&body(vec![c.clone(); 128]).to_string(), &chunk, &request).is_ok());
    assert!(open::validate(&body(vec![c.clone(); 129]).to_string(), &chunk, &request).is_err());
    let mut missing = c.clone();
    missing["object"]
        .as_object_mut()
        .unwrap()
        .remove("type_hint");
    assert!(open::validate(&body(vec![missing]).to_string(), &chunk, &request).is_err());
    let mut no_evidence = c.clone();
    no_evidence["evidence"] = json!([]);
    assert!(open::validate(&body(vec![no_evidence]).to_string(), &chunk, &request).is_err());
    let mut extra = body(vec![c]);
    extra["repair"] = json!(true);
    assert!(open::validate(&extra.to_string(), &chunk, &request).is_err());
    Ok(())
}
#[test]
fn security_guard_and_window_boundaries_are_not_relaxed() -> Result<()> {
    let (chunk, prepared) = fixture("Header\napi_key=fixture_secret_value\n")?;
    assert!(open::prepare(&chunk, &prepared, &"c".repeat(64), 16384).is_err());
    let (mut chunk, prepared) = fixture("Header\nGoogle Ads spent €150.\nOther unrelated text.")?;
    chunk.range.end = "Header\nGoogle Ads spent €150.".len();
    let request = open::prepare(&chunk, &prepared, &"c".repeat(64), 16384)?;
    assert!(
        request
            .units
            .values()
            .all(|u| !u.quote.contains("Other unrelated"))
    );
    Ok(())
}
