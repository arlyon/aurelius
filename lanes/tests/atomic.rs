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
        facet_forms: BTreeMap::from([("report".into(), vec!["report".into()])]),
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
            ObjectSpec::Code(BTreeMap::from([
                ("form_fills".into(), vec!["FORM FILLS".into()]),
                ("ctr".into(), vec!["CTR".into()]),
            ])),
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
            verb_forms: vec!["send".into()],
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
fn prohibition_is_not_absence_of_obligation() -> Result<()> {
    let (mut c, mut p, r) = fixture()?;
    for (text, expected) in [
        ("Report\nMust not rank by CTR.\n", false),
        ("Report\nNot required to rank by CTR.\n", true),
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
        if expected {
            assert!(graph.assertions.is_empty());
            assert_eq!(
                graph.review_items[0].reason,
                atomic::Reason::UnsupportedOuterNegation
            );
        } else {
            assert_eq!(graph.assertions.len(), 1);
            assert!(
                graph.assertions[0]
                    .rendered
                    .contains("Prohibited (must not)")
            );
        }
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
