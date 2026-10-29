use anyhow::Result;
use aurelius_lanes::{
    atomic::{self, ObjectSpec, PredicateSpec, Registry},
    contracts::{ByteRange, Chunk, Document, DocumentKey},
    fact_context::FactContext,
    fact_preparation::PreparedFactSource,
    language::LanguageAnalysis,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn atomic_runtime_restores_the_preserved_registry_with_a_distinct_profile() -> Result<()> {
    use aurelius_lanes::{
        config::Config,
        quality::{ExtractionFormat, QualityConfig},
    };
    let ontology = Config::parse(include_str!("fixtures/business.toml"))?.ontology;
    let mut policy: QualityConfig = serde_json::from_value(json!({
        "context_as_of":"2026-01-01T00:00:00Z", "max_output_tokens":4096
    }))?;
    let base = "c".repeat(64);
    let preparation = "b".repeat(64);
    let typed = policy.profile(&base, &preparation)?;
    policy.format = ExtractionFormat::OpenClaimsV3;
    let open = policy.profile(&base, &preparation)?;
    policy.format = ExtractionFormat::AtomicSpoV3;
    policy.validate(&ontology)?;
    let runtime = policy.profile(&base, &preparation)?;
    assert_ne!(runtime, typed);
    assert_ne!(runtime, open);
    assert_eq!(serde_json::to_value(&policy)?["format"], "atomic_spo_v3");
    let registry = policy.atomic_registry()?;
    let preserved: Registry = serde_json::from_str(include_str!("../examples/atomic_spo.json"))?;
    assert_eq!(
        registry.semantic_definition()?,
        preserved.semantic_definition()?
    );
    assert_eq!(
        registry.rendering_profile()?,
        preserved.rendering_profile()?
    );
    for predicate in [
        "monthly_budget",
        "pull_deadline",
        "executor",
        "recipient",
        "target",
        "access_level",
    ] {
        assert!(registry.predicates.contains_key(predicate));
    }
    policy.max_output_tokens = Some(2048);
    assert_ne!(runtime, policy.profile(&base, &preparation)?);
    assert_ne!(runtime, policy.profile(&"d".repeat(64), &preparation)?);
    assert_ne!(runtime, policy.profile(&base, &"a".repeat(64))?);
    policy.align_mentions = true;
    assert!(policy.validate(&ontology).is_err());
    policy.align_mentions = false;
    policy.extra_instructions = Some("Do not silently change the atomic prompt".into());
    assert!(policy.validate(&ontology).is_err());
    Ok(())
}

#[test]
fn atomic_runtime_refuses_legacy_chat_api_instead_of_falling_back() -> Result<()> {
    use aurelius_lanes::config::Config;
    let mut config = Config::parse(include_str!("fixtures/business.toml"))?;
    config.chat = Some(serde_json::from_value(json!({
        "endpoint":"http://127.0.0.1:1234/v1/chat/completions", "model":"fixture",
        "revision":"fixture", "api":"chat_completions"
    }))?);
    config.fact_quality = Some(serde_json::from_value(json!({
        "format":"atomic_spo_v3", "context_as_of":"2026-01-01T00:00:00Z"
    }))?);
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("Responses API")
    );
    Ok(())
}

#[test]
fn relaxed_identity_spans_do_not_depend_on_noun_or_verb_tags() -> Result<()> {
    let (mut chunk, mut prepared, _) = fixture()?;
    chunk.document.text.push_str("\n[Tim] Café\n");
    chunk.range.end = chunk.document.text.len();
    prepared.content_hash = chunk.document.content_hash();
    prepared.language.content_hash = prepared.content_hash.clone();
    let mut catalog = prepared.span_catalog(&chunk)?;
    let tim = chunk.document.text.find("Tim").unwrap();
    let href = format!("span:{tim}-{}", tim + 3);
    assert!(
        !catalog
            .anchor_hrefs(&atomic::AnchorKind::Mention)
            .contains(&href)
    );
    let evidence = catalog.evidence_hrefs();
    let reviews = catalog.review_hrefs();
    let record = catalog.record_anchor()?.clone();
    catalog.relax_entity_anchors(&chunk)?;
    for kind in [
        atomic::AnchorKind::Mention,
        atomic::AnchorKind::Action,
        atomic::AnchorKind::Facet,
    ] {
        assert!(catalog.anchor_hrefs(&kind).contains(&href));
    }
    assert_eq!(catalog.quote(&href)?, "Tim");
    let cafe = chunk.document.text.find("Café").unwrap();
    let unicode = format!("span:{cafe}-{}", cafe + "Café".len());
    assert_eq!(catalog.quote(&unicode)?, "Café");
    catalog.resolve(&unicode)?.validate(&chunk.document)?;
    assert_eq!(catalog.evidence_hrefs(), evidence);
    assert_eq!(catalog.review_hrefs(), reviews);
    assert_eq!(catalog.record_anchor()?, &record);
    assert!(catalog.resolve("span:999999-1000000").is_err());
    Ok(())
}

#[test]
fn relaxed_catalog_rejects_changed_source_and_does_not_add_other_window_tokens() -> Result<()> {
    let (chunk, prepared, _) = fixture()?;
    let mut catalog = prepared.span_catalog(&chunk)?;
    let mut changed = chunk.clone();
    changed.document.key.scope = "another-tenant".into();
    assert!(catalog.relax_entity_anchors(&changed).is_err());
    changed = chunk.clone();
    changed.document.text.push_str("changed");
    assert!(catalog.relax_entity_anchors(&changed).is_err());
    let mut first = chunk.clone();
    first.range.end = chunk.document.text.find('\n').unwrap();
    let mut first_catalog = prepared.span_catalog(&first)?;
    first_catalog.relax_entity_anchors(&first)?;
    let ada = chunk.document.text.find("Ada").unwrap();
    assert!(
        first_catalog
            .resolve(&format!("span:{ada}-{}", ada + 3))
            .is_err()
    );
    Ok(())
}

fn set(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|s| s.to_string()).collect()
}
fn fixture() -> Result<(Chunk, PreparedFactSource, Registry)> {
    let text = "Report\nPull stats Monday by 4pm PT.\nMonthly budget $50,000/month for 2 months.\nAd groups ranked by FORM FILLS, not CTR.\nAda\nSend report to Ada.\n";
    let document = Document {
        key: DocumentKey {
            scope: "tenant".into(),
            source: "document".into(),
            revision: "1".into(),
        },
        text: text.into(),
        context: None,
        metadata: Default::default(),
    };
    // Validator fixtures only, not a replacement NLP runtime.
    let language = LanguageAnalysis {
        sentence_spans: vec![],
        document: document.key.clone(),
        content_hash: document.content_hash(),
        profile: "a".repeat(64),
        mentions: vec![aurelius_lanes::language::Mention {
            id: "ada".into(),
            range: ByteRange {
                start: text.find("Ada").unwrap(),
                end: text.find("Ada").unwrap() + 3,
            },
            text: "Ada".into(),
            noun: true,
            pronoun: false,
            learned: true,
            raw_mention_score: Some(1.0),
        }],
        coreference_hypotheses: vec![],
        verb_spans: vec![ByteRange {
            start: text.find("Send").unwrap(),
            end: text.find("Send").unwrap() + 4,
        }],
        number_spans: vec![],
    };
    let prepared = PreparedFactSource::from_analysis(
        &document,
        language,
        vec![],
        FactContext {
            organization_id: "tenant".into(),
            entities: vec![],
        },
        "2026-01-01T00:00:00Z".parse()?,
        "b".repeat(64),
    )?;
    let chunk = Chunk {
        range: ByteRange {
            start: 0,
            end: document.text.len(),
        },
        document,
    };
    let mut registry = Registry {
        entity_anchors: BTreeMap::from([
            ("report".into(), atomic::AnchorKind::Facet),
            ("person".into(), atomic::AnchorKind::Mention),
            ("action".into(), atomic::AnchorKind::Action),
        ]),
        ..Default::default()
    };
    for (name, types, object, render) in [
        (
            "pull_deadline",
            set(&["report"]),
            ObjectSpec::WeeklyLocalTime,
            "{subject} has pull deadline {object}",
        ),
        (
            "monthly_budget",
            set(&["report"]),
            ObjectSpec::Money,
            "{subject} has monthly budget {object}",
        ),
        (
            "ramp_duration",
            set(&["report"]),
            ObjectSpec::Duration,
            "{subject} has ramp duration {object}",
        ),
        (
            "ranking_metric",
            set(&["report"]),
            ObjectSpec::Code(set(&["form_fills", "ctr"])),
            "{subject} is ranked by {object}",
        ),
        (
            "executor",
            set(&["action"]),
            ObjectSpec::Entity(set(&["person"])),
            "{subject} has executor {object}",
        ),
        (
            "target",
            set(&["action"]),
            ObjectSpec::Entity(set(&["report"])),
            "{subject} targets {object}",
        ),
    ] {
        registry.predicates.insert(
            name.into(),
            PredicateSpec {
                description: format!("Synthetic {name} contract"),
                allowed_modes: vec![],
                subject_types: types,
                object,
                render: render.into(),
            },
        );
    }
    registry.action_requirements.insert(
        "send".into(),
        atomic::ActionSpec {
            required_roles: set(&["executor", "target"]),
            target_types: set(&["report"]),
        },
    );
    Ok((chunk, prepared, registry))
}
fn href(chunk: &Chunk, text: &str) -> String {
    let start = chunk.document.text.find(text).unwrap();
    format!("span:{start}-{}", start + text.len())
}
fn entity(id: &str, kind: &str, anchor: String) -> Value {
    json!({"id":id,"entity_type":kind,"identity_span_ref":anchor,"action_kind":null})
}
fn qualification(mode: &str) -> Value {
    json!({"mode":mode,"polarity":"positive","conditions":[],"attribution":[],"validity":{"from":null,"to":null}})
}
fn fact(subject: &str, predicate: &str, object: Value, evidence: String) -> Value {
    json!({"subject":{"ref":subject},"predicate":predicate,"object":object,"qualification":qualification("required"),"evidence":[evidence]})
}
#[test]
fn unknown_currency_is_required_null_and_keeps_distinct_identity() -> Result<()> {
    let (mut chunk, mut prepared, mut registry) = fixture()?;
    chunk.document.text = "Report\nMonthly budget 4000 per month.\nInitial spend 50.\n".into();
    chunk.range.end = chunk.document.text.len();
    prepared.content_hash = chunk.document.content_hash();
    prepared.language.content_hash = prepared.content_hash.clone();
    prepared.language.mentions.clear();
    prepared.language.verb_spans.clear();
    prepared.language.number_spans.clear();
    prepared.language.sentence_spans.clear();
    registry.predicates.insert(
        "initial_spend".into(),
        PredicateSpec {
            description: "Synthetic initial spend".into(),
            allowed_modes: vec![],
            subject_types: set(&["report"]),
            object: ObjectSpec::MoneyAmount,
            render: "{subject} has initial spend {object}".into(),
        },
    );
    let catalog = prepared.span_catalog(&chunk)?;
    let budget = fact(
        "r",
        "monthly_budget",
        json!({"kind":"money","amount":"4000","currency_symbol":null,"period":"month"}),
        href(&chunk, "Monthly budget 4000 per month."),
    );
    let spend = fact(
        "r",
        "initial_spend",
        json!({"kind":"money_amount","amount":"50","currency_symbol":null}),
        href(&chunk, "Initial spend 50."),
    );
    let body = json!({"local_entities":[entity("r","report",href(&chunk,"Report"))],"assertions":[budget,spend],"review_items":[]});
    let graph = atomic::validate(&body.to_string(), &catalog, &registry)?;
    assert_eq!(graph.assertions.len(), 2);
    assert!(
        graph
            .assertions
            .iter()
            .all(|a| a.rendered.contains("currency unknown") && !a.rendered.contains('$'))
    );
    assert_eq!(
        serde_json::to_value(&graph.assertions[0].assertion.object)?["currency_symbol"],
        Value::Null
    );
    let mut known = body.clone();
    known["assertions"][0]["object"]["currency_symbol"] = json!("$");
    // Deterministic validation does not establish that the guessed symbol is true.
    let known_graph = atomic::validate(&known.to_string(), &catalog, &registry)?;
    assert_ne!(graph.assertions[0].id, known_graph.assertions[0].id);
    for symbol in [json!(""), json!("USD"), json!("unknown")] {
        let mut invalid = body.clone();
        invalid["assertions"][0]["object"]["currency_symbol"] = symbol;
        assert!(atomic::validate(&invalid.to_string(), &catalog, &registry).is_err());
    }
    for index in [0, 1] {
        let mut missing = body.clone();
        missing["assertions"][index]["object"]
            .as_object_mut()
            .unwrap()
            .remove("currency_symbol");
        assert!(atomic::validate(&missing.to_string(), &catalog, &registry).is_err());
    }
    Ok(())
}

#[test]
fn native_array_limits_match_runtime_root_and_qualification_limits() -> Result<()> {
    let (chunk, prepared, mut registry) = fixture()?;
    registry
        .condition_predicates
        .insert("ranking_metric".into());
    let catalog = prepared.span_catalog(&chunk)?;
    let schema = registry.schema(&catalog)?;
    for key in ["local_entities", "assertions", "review_items"] {
        assert_eq!(schema["properties"][key]["maxItems"], 128);
        assert_eq!(schema["properties"][key]["minItems"], 0);
    }
    let q = &schema["$defs"]["qualification"]["properties"];
    assert_eq!(q["conditions"]["maxItems"], 8);
    assert_eq!(q["attribution"]["maxItems"], 8);
    for variant in schema["properties"]["assertions"]["items"]["anyOf"]
        .as_array()
        .unwrap()
    {
        assert_eq!(variant["properties"]["evidence"]["minItems"], 1);
        assert_eq!(variant["properties"]["evidence"]["maxItems"], 16);
    }
    let review_ref = catalog.review_hrefs()[0].clone();
    let review = json!({"intended_predicate":"unsupported","span_ref":review_ref,"exact_quote":catalog.quote(&review_ref)?,"reason":"unsupported_predicate","action_ref":null});
    let body = json!({"local_entities":[],"assertions":[],"review_items":vec![review.clone();128]});
    assert_eq!(
        atomic::validate(&body.to_string(), &catalog, &registry)?
            .review_items
            .len(),
        128
    );
    let mut too_many = body;
    too_many["review_items"]
        .as_array_mut()
        .unwrap()
        .push(review);
    assert!(atomic::validate(&too_many.to_string(), &catalog, &registry).is_err());
    let evidence_ref = href(&chunk, "Ad groups ranked by FORM FILLS, not CTR.");
    let condition = json!({"subject":{"ref":"r"},"predicate":"ranking_metric","object":{"kind":"code","value":"form_fills"},"polarity":"positive","evidence":[evidence_ref]});
    let mut assertion = fact(
        "r",
        "ranking_metric",
        json!({"kind":"code","value":"form_fills"}),
        evidence_ref.clone(),
    );
    assertion["evidence"] = json!(vec![evidence_ref; 16]);
    assertion["qualification"]["conditions"] = json!(vec![condition; 8]);
    assertion["qualification"]["attribution"] = json!(vec![json!({"ref":"ada"}); 8]);
    let valid = json!({"local_entities":[entity("r","report",href(&chunk,"Report")),entity("ada","person",href(&chunk,"Ada"))],"assertions":[assertion],"review_items":[]});
    assert_eq!(
        atomic::validate(&valid.to_string(), &catalog, &registry)?
            .assertions
            .len(),
        1
    );
    for field in ["conditions", "attribution"] {
        let mut invalid = valid.clone();
        let values = invalid["assertions"][0]["qualification"][field]
            .as_array_mut()
            .unwrap();
        values.push(values[0].clone());
        assert!(atomic::validate(&invalid.to_string(), &catalog, &registry).is_err());
    }
    for count in [0, 17] {
        let mut invalid = valid.clone();
        invalid["assertions"][0]["evidence"] =
            json!(vec![
                href(&chunk, "Ad groups ranked by FORM FILLS, not CTR.");
                count
            ]);
        assert!(atomic::validate(&invalid.to_string(), &catalog, &registry).is_err());
    }
    Ok(())
}

#[test]
fn entity_definitions_are_model_metadata_and_versioned_not_source_vetoes() -> Result<()> {
    let (chunk, prepared, mut registry) = fixture()?;
    let before = atomic::prepare_request(&chunk, &prepared, &registry, &"c".repeat(64), 2048)?;
    registry.entity_descriptions.insert(
        "report".into(),
        "A source-local content artifact, not the action producing it.".into(),
    );
    let after = atomic::prepare_request(&chunk, &prepared, &registry, &"c".repeat(64), 2048)?;
    assert_ne!(before.profile, after.profile);
    assert_ne!(before.binding, after.binding);
    let variants = after.schema["properties"]["local_entities"]["items"]["anyOf"]
        .as_array()
        .unwrap();
    let report = variants
        .iter()
        .find(|v| v["properties"]["entity_type"]["const"] == "report")
        .unwrap();
    assert_eq!(
        report["properties"]["entity_type"]["description"],
        registry.entity_descriptions["report"]
    );
    registry
        .entity_descriptions
        .insert("undeclared".into(), "Invalid type".into());
    assert!(registry.validate().is_err());
    Ok(())
}

#[cfg(feature = "language")]
#[test]
fn v2_saved_contract_is_not_relabelled_as_v3() -> Result<()> {
    let (chunk, prepared, registry) = fixture()?;
    let request = atomic::prepare_request(&chunk, &prepared, &registry, &"c".repeat(64), 2048)?;
    let saved = aurelius_lanes::atomic_lab::SavedResponse {
        contract_version: "atomic-spo-v2".into(),
        binding: request.binding.clone(),
        body: json!({"local_entities":[],"assertions":[],"review_items":[]}).to_string(),
        usage: None,
    };
    assert!(aurelius_lanes::atomic_lab::replay(&saved, &request, &registry).is_err());
    Ok(())
}

#[test]
fn typed_atoms_schema_and_rendering_preserve_original_values() -> Result<()> {
    let (c, p, r) = fixture()?;
    let catalog = p.span_catalog(&c)?;
    let schema = r.schema(&catalog)?;
    assert_eq!(schema["additionalProperties"], false);
    assert!(schema["properties"].get("notes").is_none());
    let body = json!({"local_entities":[entity("report_1","report",href(&c,"Report"))],"assertions":[fact("report_1","pull_deadline",json!({"kind":"weekly_local_time","weekday":"monday","local_time":"16:00","timezone_text":"PT"}),href(&c,"Pull stats Monday by 4pm PT.")),fact("report_1","monthly_budget",json!({"kind":"money","amount":"50000","currency_symbol":"$","period":"month"}),href(&c,"Monthly budget $50,000/month for 2 months.")),fact("report_1","ramp_duration",json!({"kind":"duration","amount":2,"unit":"month"}),href(&c,"Monthly budget $50,000/month for 2 months."))],"review_items":[]});
    let graph = atomic::validate(&body.to_string(), &catalog, &r)?;
    assert_eq!(graph.assertions.len(), 3);
    assert!(graph.review_items.is_empty());
    assert!(graph.assertions[0].rendered.contains("16:00 PT"));
    assert!(!graph.assertions[1].rendered.contains("USD"));
    Ok(())
}
#[test]
fn coordinate_identity_ignores_temporary_ids_and_renderer() -> Result<()> {
    let (c, p, mut r) = fixture()?;
    let catalog = p.span_catalog(&c)?;
    let f = fact(
        "local_1",
        "ranking_metric",
        json!({"kind":"code","value":"form_fills"}),
        href(&c, "Ad groups ranked by FORM FILLS, not CTR."),
    );
    let first = json!({"local_entities":[entity("local_1","report",href(&c,"Report"))],"assertions":[f.clone(),f],"review_items":[]});
    let a = atomic::validate(&first.to_string(), &catalog, &r)?;
    assert_eq!(a.assertions.len(), 1);
    let second = first.to_string().replace("local_1", "renamed_99");
    r.predicates.get_mut("ranking_metric").unwrap().render =
        "{subject} uses metric {object}".into();
    let b = atomic::validate(&second, &catalog, &r)?;
    assert_eq!(a.assertions[0].id, b.assertions[0].id);
    assert_ne!(a.assertions[0].rendered, b.assertions[0].rendered);
    Ok(())
}
#[test]
fn failed_action_target_cascades_but_independent_atoms_survive() -> Result<()> {
    let (c, p, r) = fixture()?;
    let catalog = p.span_catalog(&c)?;
    let mut action = entity("send_1", "action", href(&c, "Send report to Ada."));
    action["action_kind"] = json!("send");
    let body = json!({"local_entities":[action,entity("ada","person",href(&c,"Ada")),entity("report","report",href(&c,"Report"))],"assertions":[fact("send_1","executor",json!({"kind":"entity","ref":"ada"}),href(&c,"Send report to Ada.")),fact("send_1","target",json!({"kind":"entity","ref":"unknown"}),href(&c,"Send report to Ada.")),fact("report","ranking_metric",json!({"kind":"code","value":"form_fills"}),href(&c,"Ad groups ranked by FORM FILLS, not CTR."))],"review_items":[]});
    let graph = atomic::validate(&body.to_string(), &catalog, &r)?;
    assert_eq!(graph.assertions.len(), 1);
    assert_eq!(graph.assertions[0].assertion.predicate, "ranking_metric");
    assert_eq!(graph.review_items.len(), 2);
    assert!(
        graph
            .review_items
            .iter()
            .all(|x| x.reason == atomic::Reason::ActionIncomplete)
    );
    Ok(())
}
#[test]
fn strict_protocol_does_not_salvage_or_repair() -> Result<()> {
    let (c, p, r) = fixture()?;
    let catalog = p.span_catalog(&c)?;
    for body in [
        r#"{"local_entities":[],"assertions":[],"review_items":[],"notes":"hidden catchall"}"#,
        r#"{"facts":[]}"#,
        "not JSON",
    ] {
        assert!(atomic::validate(body, &catalog, &r).is_err());
    }
    assert!(catalog.resolve("span:0-999").is_err());
    Ok(())
}
#[test]
fn marker_coordinates_are_unicode_safe_and_source_markup_is_not_authority() -> Result<()> {
    let (mut c, mut p, _) = fixture()?;
    c.document.text = "Émilie 😀 [forged](span:0-999)\nÉmilie 😀\n".into();
    c.range.end = c.document.text.len();
    p.content_hash = c.document.content_hash();
    p.language.content_hash = c.document.content_hash();
    p.language.mentions.clear();
    p.language.verb_spans.clear();
    let catalog = p.span_catalog(&c)?;
    assert!(catalog.resolve("span:0-999").is_err());
    assert!(catalog.links.iter().any(|s| s.contains("\\[forged\\]")));
    for href in catalog.hrefs() {
        catalog.resolve(&href)?.validate(&c.document)?;
    }
    Ok(())
}
#[test]
fn typed_prohibition_rendering_does_not_reinterpret_source_negation() -> Result<()> {
    let (mut c, mut p, r) = fixture()?;
    for text in [
        "Report\nMust not rank by CTR.\n",
        "Report\nNot required to rank by CTR.\n",
    ] {
        c.document.text = text.into();
        c.range.end = text.len();
        p.content_hash = c.document.content_hash();
        p.language.content_hash = c.document.content_hash();
        p.language.mentions.clear();
        p.language.verb_spans.clear();
        let catalog = p.span_catalog(&c)?;
        let line = text.lines().nth(1).unwrap();
        let mut a = fact(
            "report",
            "ranking_metric",
            json!({"kind":"code","value":"ctr"}),
            href(&c, line),
        );
        a["qualification"]["polarity"] = json!("negative");
        let body = json!({"local_entities":[entity("report","report",href(&c,"Report"))],"assertions":[a],"review_items":[]});
        let graph = atomic::validate(&body.to_string(), &catalog, &r)?;
        // The second interpretation is deliberately wrong: this pass validates the typed
        // claim and source bytes, not entailment. The model must choose review for that source.
        assert_eq!(graph.assertions.len(), 1);
        assert!(
            graph.assertions[0]
                .rendered
                .contains("Prohibited (must not)")
        );
    }
    Ok(())
}
#[test]
fn missing_nullable_review_fields_and_prose_explanations_are_protocol_failures() -> Result<()> {
    let (c, p, r) = fixture()?;
    let catalog = p.span_catalog(&c)?;
    let item = json!({"intended_predicate":"unsupported_business_relation","span_ref":href(&c,"Report"),"exact_quote":"Report","reason":"unsupported_predicate","action_ref":null});
    let good = json!({"local_entities":[],"assertions":[],"review_items":[item]});
    assert_eq!(
        atomic::validate(&good.to_string(), &catalog, &r)?
            .review_items
            .len(),
        1
    );
    let mut missing = good.clone();
    missing["review_items"][0]
        .as_object_mut()
        .unwrap()
        .remove("action_ref");
    assert!(atomic::validate(&missing.to_string(), &catalog, &r).is_err());
    let mut prose = good.clone();
    prose["review_items"][0]["explanation"] = json!("a paragraph must not become a fact");
    assert!(atomic::validate(&prose.to_string(), &catalog, &r).is_err());
    let mut wrong = good;
    wrong["review_items"][0]["exact_quote"] = json!("report");
    assert!(atomic::validate(&wrong.to_string(), &catalog, &r).is_err());
    Ok(())
}
#[tokio::test]
async fn native_transport_sends_one_strict_request_and_never_downgrades() -> Result<()> {
    use aurelius_lanes::{
        config::{ChatApi, ChatConfig, Config},
        facts::{ChatExtractor, ReviewDiagnostic, review_diagnostic},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let router=axum::Router::new().route("/responses",axum::routing::post(move |axum::Json(body):axum::Json<Value>| {observed.fetch_add(1,Ordering::SeqCst);async move {
        assert_eq!(body["text"]["format"]["type"],"json_schema");assert_eq!(body["text"]["format"]["strict"],true);assert_eq!(body["text"]["format"]["schema"]["additionalProperties"],false);assert!(!body["text"]["format"]["schema"]["$defs"]["span_ref"]["enum"].as_array().unwrap().is_empty());
        axum::Json(json!({"status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"{\"local_entities\":[],\"assertions\":[],\"review_items\":[]}"}]}],"usage":{"input_tokens":10,"output_tokens":10}}))
    }}));
    let unsupported = Arc::new(AtomicUsize::new(0));
    let rejected = unsupported.clone();
    let router = router.route(
        "/unsupported",
        axum::routing::post(move || {
            rejected.fetch_add(1, Ordering::SeqCst);
            async { axum::http::StatusCode::BAD_REQUEST }
        }),
    );
    let bad_calls = Arc::new(AtomicUsize::new(0));
    let counted = bad_calls.clone();
    let router=router.route("/bad/{case}",axum::routing::post(move|axum::extract::Path(case):axum::extract::Path<String>|{counted.fetch_add(1,Ordering::SeqCst);async move{
        let content=if case=="refusal"{json!({"type":"refusal","refusal":"fixture refusal"})}else{json!({"type":"output_text","text":"{\"facts\":[]}"})};
        axum::Json(json!({"status":if case=="incomplete"{"incomplete"}else{"completed"},"output":[{"type":"message","role":"assistant","status":"completed","content":[content]}],"usage":{"input_tokens":10,"output_tokens":10}}))
    }}));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(async move { axum::serve(listener, router).await });
    let (c, p, r) = fixture()?;
    let config = Config::parse(include_str!("fixtures/business.toml"))?;
    let mut settings = ChatConfig {
        endpoint: format!("http://{address}/responses"),
        model: "fixture".into(),
        revision: "fixture".into(),
        api: ChatApi::Responses,
        api_key_env: None,
        gateway: None,
        references: Default::default(),
    };
    let result = ChatExtractor::new(&settings, config.ontology.clone())?
        .extract_atomic(&c, &p, &r, 4096)
        .await?;
    assert!(result.graph.assertions.is_empty());
    assert!(serde_json::to_value(&result)?.get("raw_output").is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    settings.endpoint = format!("http://{address}/unsupported");
    let error = ChatExtractor::new(&settings, config.ontology.clone())?
        .extract_atomic(&c, &p, &r, 4096)
        .await
        .err()
        .unwrap();
    assert_eq!(
        review_diagnostic(&error),
        ReviewDiagnostic::NativeSchemaUnsupported
    );
    assert_eq!(unsupported.load(Ordering::SeqCst), 1);
    for (index, (case, diagnostic)) in [
        ("malformed", ReviewDiagnostic::NativeSchemaViolation),
        ("refusal", ReviewDiagnostic::RefusalOrAmbiguous),
        ("incomplete", ReviewDiagnostic::IncompleteResponse),
    ]
    .into_iter()
    .enumerate()
    {
        settings.endpoint = format!("http://{address}/bad/{case}");
        let error = ChatExtractor::new(&settings, config.ontology.clone())?
            .extract_atomic(&c, &p, &r, 4096)
            .await
            .err()
            .unwrap();
        assert_eq!(review_diagnostic(&error), diagnostic);
        assert!(aurelius_lanes::facts::review_output(&error).is_some());
        assert_eq!(
            aurelius_lanes::facts::review_usage(&error)
                .unwrap()
                .output_tokens,
            Some(10)
        );
        assert_eq!(bad_calls.load(Ordering::SeqCst), index + 1);
    }
    settings.api = ChatApi::ChatCompletions;
    let error = ChatExtractor::new(&settings, config.ontology)?
        .extract_atomic(&c, &p, &r, 4096)
        .await
        .err()
        .unwrap();
    assert_eq!(
        review_diagnostic(&error),
        ReviewDiagnostic::NativeSchemaUnsupported
    );
    assert_eq!(unsupported.load(Ordering::SeqCst), 1);
    server.abort();
    Ok(())
}
