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
fn claim_local_policy_keeps_unmodified_siblings_and_rejects_whole_bad_claim() -> Result<()> {
    let text = "Google Ads spent €150.";
    let (chunk, prepared) = fixture(text)?;
    let request = open::prepare(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let u = unit(&request, text)?;
    let good = claim(&u, "Google Ads", "spent", "€150");
    let mut bad = good.clone();
    bad["qualification"]["conditions"] = json!([quote(&u, "if approved")]);
    let mut value = body(vec![good.clone(), bad, good.clone()]);
    value["unresolved"] = json!([{"source":quote(&u,"absent"),"reason":"ambiguous referent"},{"source":quote(&u,"Google Ads"),"reason":"source-local scope"}]);
    assert!(open::validate(&value.to_string(), &chunk, &request).is_err());
    let graph = open::validate_partial(&value.to_string(), &chunk, &request)?;
    assert_eq!(graph["validation_policy"], open::CLAIM_LOCAL_POLICY);
    assert_eq!(graph["claims"].as_array().unwrap().len(), 2);
    assert_eq!(graph["claims"][0]["source_index"], 0);
    assert_eq!(graph["claims"][1]["source_index"], 2);
    assert_eq!(graph["claims"][0]["claim"], good);
    assert_eq!(graph["claims"][1]["claim"], good);
    assert_eq!(graph["rejected"]["claims"][0]["source_index"], 1);
    assert_eq!(graph["rejected"]["unresolved"][0]["source_index"], 0);
    assert_eq!(graph["unresolved"][0]["source_index"], 1);
    assert_eq!(graph["emitted_counts"]["claims"], 3);
    Ok(())
}
#[test]
fn partial_policy_does_not_salvage_invalid_envelopes_or_source_bindings() -> Result<()> {
    let (chunk, prepared) = fixture("Google Ads spent €150.")?;
    let request = open::prepare(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let u = unit(&request, &chunk.document.text)?;
    let c = claim(&u, "Google Ads", "spent", "€150");
    for value in [
        json!({"claims":[c.clone()],"unresolved":[],"repair":true}),
        body(vec![c.clone(); 129]),
    ] {
        assert!(open::validate_partial(&value.to_string(), &chunk, &request).is_err());
    }
    assert!(open::validate_partial("{", &chunk, &request).is_err());
    let mut wrong = chunk.clone();
    wrong.document.key.scope = "another".into();
    assert!(open::validate_partial(&body(vec![c]).to_string(), &wrong, &request).is_err());
    Ok(())
}
fn contextual_quotes(value: &mut Value) {
    match value {
        Value::Object(properties) => {
            if properties.contains_key("unit_ref") && properties.contains_key("exact_quote") {
                properties.insert("context_quote".into(), Value::Null);
            }
            for child in properties.values_mut() {
                contextual_quotes(child);
            }
        }
        Value::Array(values) => {
            for child in values {
                contextual_quotes(child);
            }
        }
        _ => {}
    }
}
#[test]
fn explicit_exact_context_selects_repeated_quotes_without_guessing() -> Result<()> {
    let text = "I'll update High Intent; Liz asked Liz.";
    let (chunk, prepared) = fixture(text)?;
    let old = open::prepare(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let request = open::prepare_contextual(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let u = unit(&request, text)?;
    assert_ne!(old.binding, request.binding);
    assert_eq!(request.contract_version, open::CONTEXT_VERSION);
    let mut c = claim(&u, "I", "will update", "High Intent");
    contextual_quotes(&mut c);
    let no_context = open::validate_partial(&body(vec![c.clone()]).to_string(), &chunk, &request)?;
    assert_eq!(
        no_context["rejected"]["claims"].as_array().unwrap().len(),
        1
    );
    c["subject"]["source"]["context_quote"] = json!("I'll update");
    let graph = open::validate_partial(&body(vec![c]).to_string(), &chunk, &request)?;
    assert_eq!(graph["claims"][0]["subject_span"]["range"]["start"], 0);
    assert_eq!(graph["claims"][0]["subject_span"]["quote"], "I");
    let mut c = claim(&u, "Liz", "asked", "Liz");
    contextual_quotes(&mut c);
    c["subject"]["source"]["context_quote"] = json!("Liz asked");
    c["object"]["source"]["context_quote"] = json!("asked Liz");
    let graph = open::validate_partial(&body(vec![c]).to_string(), &chunk, &request)?;
    assert_ne!(
        graph["claims"][0]["subject_span"]["range"],
        graph["claims"][0]["object_span"]["range"]
    );
    Ok(())
}
#[test]
fn context_must_be_exact_unique_in_same_unit_and_contain_quote_once() -> Result<()> {
    let text = "Café met Café; Liz said Liz said.";
    let (chunk, prepared) = fixture(text)?;
    let request = open::prepare_contextual(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let u = unit(&request, text)?;
    let mut c = claim(&u, "Café", "met", "Café");
    contextual_quotes(&mut c);
    c["subject"]["source"]["context_quote"] = json!("Café met");
    c["object"]["source"]["context_quote"] = json!("met Café");
    let graph = open::validate_partial(&body(vec![c.clone()]).to_string(), &chunk, &request)?;
    assert_eq!(graph["claims"].as_array().unwrap().len(), 1);
    for context in ["Cafe met", text, "Liz said", "not in source"] {
        let mut wrong = c.clone();
        wrong["subject"]["source"]["context_quote"] = json!(context);
        let graph = open::validate_partial(&body(vec![wrong]).to_string(), &chunk, &request)?;
        assert!(graph["claims"].as_array().unwrap().is_empty());
        assert_eq!(graph["rejected"]["claims"].as_array().unwrap().len(), 1);
    }
    let mut wrong = c;
    wrong["subject"]["source"]["unit_ref"] = json!("span:999-1000");
    assert!(
        open::validate_partial(&body(vec![wrong]).to_string(), &chunk, &request)?["claims"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    Ok(())
}
#[test]
fn versioned_context_fields_do_not_relabel_or_repair_v1_bodies() -> Result<()> {
    let text = "Google Ads spent €150.";
    let (chunk, prepared) = fixture(text)?;
    let old = open::prepare(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let new = open::prepare_contextual(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let u = unit(&old, text)?;
    let c = claim(&u, "Google Ads", "spent", "€150");
    let wrong_version = open::validate_partial(&body(vec![c.clone()]).to_string(), &chunk, &new)?;
    assert!(wrong_version["claims"].as_array().unwrap().is_empty());
    assert_eq!(
        wrong_version["rejected"]["claims"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let mut contextual = c;
    contextual_quotes(&mut contextual);
    let wrong_version =
        open::validate_partial(&body(vec![contextual.clone()]).to_string(), &chunk, &old)?;
    assert!(wrong_version["claims"].as_array().unwrap().is_empty());
    assert_eq!(
        wrong_version["rejected"]["claims"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(open::validate_partial(&body(vec![contextual]).to_string(), &chunk, &new).is_ok());
    assert!(
        new.schema["$defs"]["quote"]["required"]
            .as_array()
            .unwrap()
            .contains(&json!("context_quote"))
    );
    Ok(())
}
#[test]
fn malformed_items_do_not_discard_valid_siblings() -> Result<()> {
    let text = "Google Ads spent €150.";
    let (chunk, prepared) = fixture(text)?;
    let request = open::prepare(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let u = unit(&request, text)?;
    let good = claim(&u, "Google Ads", "spent", "€150");
    let mut bad_mode = good.clone();
    bad_mode["qualification"]["mode"] = json!("invalid");
    let mut missing = good.clone();
    missing["object"]
        .as_object_mut()
        .unwrap()
        .remove("type_hint");
    let mut extra = good.clone();
    extra["repair"] = json!(true);
    let value = body(vec![
        good.clone(),
        bad_mode,
        missing,
        extra,
        Value::Null,
        good.clone(),
    ]);
    assert!(open::validate_partial_legacy(&value.to_string(), &chunk, &request).is_err());
    let graph = open::validate_partial(&value.to_string(), &chunk, &request)?;
    assert_eq!(graph["claims"].as_array().unwrap().len(), 2);
    assert_eq!(graph["rejected"]["claims"].as_array().unwrap().len(), 4);
    assert_eq!(graph["claims"][0]["claim"], good);
    assert_eq!(graph["claims"][1]["claim"], good);
    assert_eq!(graph["claims"][1]["source_index"], 5);
    assert_eq!(graph["validation_policy"], open::CLAIM_LOCAL_POLICY);
    Ok(())
}
#[test]
fn compact_model_view_retains_source_once_and_keeps_raw_nlp_backend_only() -> Result<()> {
    let text = "\r\nHeader\r\n\r\nMegan manages client communications.\r\nShe sends the report.\r\n\r\nGoogle Ads spent €150.\r\n\r\n";
    let (chunk, mut prepared) = fixture(text)?;
    for index in 0..100 {
        prepared
            .language
            .mentions
            .push(aurelius_lanes::language::Mention {
                id: format!("long{index}"),
                range: chunk.range,
                text: text.into(),
                noun: true,
                pronoun: false,
                learned: true,
                raw_mention_score: Some(0.5),
            });
    }
    for (id, word) in [("name", "Megan"), ("pronoun", "She")] {
        let start = text.find(word).unwrap();
        prepared
            .language
            .mentions
            .push(aurelius_lanes::language::Mention {
                id: id.into(),
                range: ByteRange {
                    start,
                    end: start + word.len(),
                },
                text: word.into(),
                noun: true,
                pronoun: id == "pronoun",
                learned: true,
                raw_mention_score: Some(0.5),
            });
    }
    prepared
        .language
        .coreference_hypotheses
        .push(vec!["name".into(), "pronoun".into()]);
    let before = serde_json::to_value(&prepared)?;
    let legacy = open::prepare(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let compact = open::prepare_compact(&chunk, &prepared, &"c".repeat(64), 16384)?;
    assert_eq!(serde_json::to_value(&prepared)?, before);
    assert!(compact.input.len() * 4 < legacy.input.len());
    let view: Value = serde_json::from_str(&compact.input)?;
    assert!(
        view["units"]
            .as_object()
            .unwrap()
            .values()
            .all(Value::is_string)
    );
    let reconstructed = compact
        .units
        .iter()
        .filter(|(id, _)| id.starts_with('u'))
        .map(|(_, span)| span.quote.as_str())
        .collect::<String>();
    assert_eq!(reconstructed, text);
    assert_eq!(
        view["coreference_hypotheses_not_identity"],
        json!([["Megan", "She"]])
    );
    for field in [
        "raw_mention_score",
        "content_hash",
        "source_range",
        "source_text",
        "noun",
    ] {
        assert!(!view.as_object().unwrap().contains_key(field));
    }
    assert!(!compact.input.contains(&chunk.document.content_hash()));
    assert_eq!(
        compact.source_binding.as_ref().unwrap().document,
        chunk.document.key
    );
    Ok(())
}
#[test]
fn compact_reported_by_is_explicit_versioned_and_preserves_wire_claim() -> Result<()> {
    let text = "Megan said Google Ads spent €150.";
    let (chunk, prepared) = fixture(text)?;
    let request = open::prepare_compact(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let u = request
        .units
        .iter()
        .find(|(_, s)| s.quote.contains("Megan said"))
        .unwrap()
        .0;
    let mut c = claim(u, "Google Ads", "spent", "€150");
    contextual_quotes(&mut c);
    let q = c["qualification"].as_object_mut().unwrap();
    q.remove("attribution");
    let mut reporter = quote(u, "Megan");
    contextual_quotes(&mut reporter);
    q.insert("reported_by".into(), json!([reporter]));
    let graph = open::validate_partial(&body(vec![c.clone()]).to_string(), &chunk, &request)?;
    assert_eq!(graph["claims"][0]["claim"], c);
    assert_eq!(graph["claims"][0]["reported_by_spans"][0]["quote"], "Megan");
    assert!(graph["claims"][0].get("attribution_spans").is_none());
    let schema = &request.schema["$defs"]["qualification"]["properties"];
    assert!(schema.get("attribution").is_none());
    assert!(
        schema["reported_by"]["description"]
            .as_str()
            .unwrap()
            .contains("Not its assigned actor")
    );
    let mut wrong = c;
    wrong["qualification"]["attribution"] = json!([]);
    let graph = open::validate_partial(&body(vec![wrong]).to_string(), &chunk, &request)?;
    assert!(graph["claims"].as_array().unwrap().is_empty());
    Ok(())
}
#[test]
fn compact_backend_binding_and_resource_security_checks_remain_fatal() -> Result<()> {
    let (chunk, prepared) = fixture("Header\nGoogle Ads spent €150.")?;
    let mut request = open::prepare_compact(&chunk, &prepared, &"c".repeat(64), 16384)?;
    let empty = body(vec![]).to_string();
    request.source_binding.as_mut().unwrap().document.scope = "other".into();
    assert!(open::validate_partial(&empty, &chunk, &request).is_err());
    let mut request = open::prepare_compact(&chunk, &prepared, &"c".repeat(64), 16384)?;
    request.units.get_mut("u0000").unwrap().quote.push('!');
    assert!(open::validate_partial(&empty, &chunk, &request).is_err());
    let (secret, prepared) = fixture("api_key=fixture_secret_value")?;
    assert!(open::prepare_compact(&secret, &prepared, &"c".repeat(64), 16384).is_err());
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
