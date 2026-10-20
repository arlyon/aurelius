use anyhow::{Context, Result};
use aurelius_lanes::{
    atomic::{self, Registry},
    contracts::{ByteRange, Chunk, Document, DocumentKey},
    fact_context::FactContext,
    fact_preparation::PreparedFactSource,
    language::{LanguageAnalysis, Mention},
};
use serde_json::{Value, json};
use std::collections::BTreeSet;
fn registry() -> Result<Registry> {
    Ok(serde_json::from_str(include_str!(
        "../examples/atomic_spo.json"
    ))?)
}
fn cases() -> Result<Vec<Value>> {
    Ok(serde_json::from_str(include_str!(
        "fixtures/atomic-golden.json"
    ))?)
}
fn range(text: &str, quote: &str) -> ByteRange {
    let start = text
        .find(quote)
        .expect("explicit fixture annotation exists");
    ByteRange {
        start,
        end: start + quote.len(),
    }
}
fn href(text: &str, quote: &str) -> String {
    let r = range(text, quote);
    format!("span:{}-{}", r.start, r.end)
}
// Hand-labelled validator fixtures, not another NLP runtime or extraction method.
fn fixture(case: &Value) -> Result<(Chunk, PreparedFactSource, Value)> {
    let text = case["source"].as_str().context("fixture source")?;
    let document = Document {
        key: DocumentKey {
            scope: "synthetic-tenant".into(),
            source: "golden-document".into(),
            revision: "r1".into(),
        },
        text: text.into(),
        context: None,
        metadata: Default::default(),
    };
    let entities = case["entities"].as_array().context("fixture entities")?;
    let mentions = entities
        .iter()
        .filter(|e| e[1] != "action")
        .enumerate()
        .map(|(i, e)| {
            let quote = e[2].as_str().unwrap();
            Mention {
                id: format!("noun_{i}"),
                range: range(text, quote),
                text: quote.into(),
                noun: true,
                pronoun: false,
                learned: true,
                raw_mention_score: Some(1.0),
            }
        })
        .collect();
    let verbs = case
        .get("verbs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|s| range(text, s.as_str().unwrap()))
        .collect();
    let language = LanguageAnalysis {
        sentence_spans: vec![],
        document: document.key.clone(),
        content_hash: document.content_hash(),
        profile: "a".repeat(64),
        mentions,
        verb_spans: verbs,
        number_spans: vec![],
        coreference_hypotheses: vec![],
    };
    let prepared = PreparedFactSource::from_analysis(
        &document,
        language,
        vec![],
        FactContext {
            organization_id: document.key.scope.clone(),
            entities: vec![],
        },
        "2026-01-01T00:00:00Z".parse()?,
        "b".repeat(64),
    )?;
    let declarations=entities.iter().map(|e|json!({"id":e[0],"entity_type":e[1],"identity_span_ref":href(text,e[2].as_str().unwrap()),"action_kind":e[3]})).collect::<Vec<_>>();
    let facts=case["facts"].as_array().context("fixture facts")?.iter().map(|f|{
        let conditions=f.get("conditions").and_then(Value::as_array).into_iter().flatten().map(|c|json!({"subject":{"ref":c["subject"]},"predicate":c["predicate"],"object":c["object"],"polarity":c["polarity"],"evidence":[href(text,c["quote"].as_str().unwrap())]})).collect::<Vec<_>>();
        let attribution=f.get("attribution").and_then(Value::as_array).into_iter().flatten().map(|id|json!({"ref":id})).collect::<Vec<_>>();
        json!({"subject":{"ref":f["subject"]},"predicate":f["predicate"],"object":f["object"],"evidence":[href(text,f["quote"].as_str().unwrap())],"qualification":{"mode":f["mode"],"polarity":f.get("polarity").cloned().unwrap_or(json!("positive")),"conditions":conditions,"attribution":attribution,"validity":{"from":null,"to":null}}})
    }).collect::<Vec<_>>();
    let reviews=case.get("review").map(|r|vec![json!({"intended_predicate":r["intended_predicate"],"span_ref":href(text,r["quote"].as_str().unwrap()),"exact_quote":r["quote"],"reason":r["reason"],"action_ref":null})]).unwrap_or_default();
    let chunk = Chunk {
        range: ByteRange {
            start: 0,
            end: text.len(),
        },
        document,
    };
    Ok((
        chunk,
        prepared,
        json!({"local_entities":declarations,"assertions":facts,"review_items":reviews}),
    ))
}
fn check_schema(value: &Value) {
    match value {
        Value::Object(object) => {
            if let Some(properties) = object.get("properties") {
                assert_eq!(object["additionalProperties"], false);
                let required = object["required"].as_array().unwrap();
                assert_eq!(required.len(), properties.as_object().unwrap().len());
            }
            if let Some(reference) = object.get("$ref") {
                assert!(reference.as_str().unwrap().starts_with("#/$defs/"));
            }
            if let Some(alternatives) = object.get("anyOf") {
                assert!(!alternatives.as_array().unwrap().is_empty());
            }
            for child in object.values() {
                check_schema(child);
            }
        }
        Value::Array(items) => {
            for child in items {
                check_schema(child);
            }
        }
        _ => {}
    }
}
#[test]
fn synthetic_source_goldens_cover_every_initial_predicate() -> Result<()> {
    let registry = registry()?;
    registry.validate()?;
    let cases = cases()?;
    assert!(cases.len() >= 20);
    let mut covered = BTreeSet::new();
    for case in &cases {
        let (chunk, prepared, body) = fixture(case)?;
        let catalog = prepared.span_catalog(&chunk)?;
        check_schema(&registry.schema(&catalog)?);
        let graph = atomic::validate(&body.to_string(), &catalog, &registry)
            .with_context(|| format!("fixture {}", case["name"]))?;
        let mut actual = graph
            .assertions
            .iter()
            .map(|a| a.assertion.predicate.clone())
            .collect::<Vec<_>>();
        actual.sort();
        let mut expected = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(actual, expected, "{}", case["name"]);
        assert_eq!(
            graph.review_items.len(),
            case["reviews"].as_u64().unwrap() as usize,
            "{}",
            case["name"]
        );
        for a in &graph.assertions {
            covered.insert(a.assertion.predicate.clone());
            assert_eq!(
                atomic::render(&a.assertion, &graph.entities, &registry)?,
                a.rendered
            );
            assert!(
                body["assertions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|original| *original == serde_json::to_value(&a.assertion).unwrap())
            );
            for s in &a.evidence {
                s.validate(&chunk.document)?;
            }
        }
    }
    assert_eq!(covered, registry.predicates.keys().cloned().collect());
    Ok(())
}
#[test]
fn qualification_refs_ids_order_and_renderer_do_not_change_coordinate_hashes() -> Result<()> {
    let cases = cases()?;
    let mut registry = registry()?;
    for name in ["condition_scope", "requester_is_not_executor_or_completed"] {
        let case = cases.iter().find(|c| c["name"] == name).unwrap();
        let (c, p, body) = fixture(case)?;
        let catalog = p.span_catalog(&c)?;
        let first = atomic::validate(&body.to_string(), &catalog, &registry)?;
        let mut renamed = body.clone();
        renamed["local_entities"].as_array_mut().unwrap().reverse();
        renamed["assertions"].as_array_mut().unwrap().reverse();
        let mut text = renamed.to_string();
        for e in body["local_entities"].as_array().unwrap() {
            let id = e["id"].as_str().unwrap();
            text = text.replace(&format!("\"{id}\""), &format!("\"renamed_{id}\""));
        }
        for predicate in registry.predicates.values_mut() {
            predicate.render = "{subject} — {object}".into();
        }
        let second = atomic::validate(&text, &catalog, &registry)?;
        assert_eq!(
            first
                .assertions
                .iter()
                .map(|a| a.id.clone())
                .collect::<BTreeSet<_>>(),
            second.assertions.iter().map(|a| a.id.clone()).collect()
        );
    }
    Ok(())
}
#[test]
fn semantic_versions_and_money_normalization_are_part_of_identity() -> Result<()> {
    let case = cases()?
        .into_iter()
        .find(|c| c["name"] == "planned_amount_and_ramp")
        .unwrap();
    let (c, p, body) = fixture(&case)?;
    let catalog = p.span_catalog(&c)?;
    let mut registry = registry()?;
    let original = atomic::validate(&body.to_string(), &catalog, &registry)?;
    let mut equivalent = body.clone();
    equivalent["assertions"][0]["object"]["amount"] = json!("050000.00");
    let same = atomic::validate(&equivalent.to_string(), &catalog, &registry)?;
    assert_eq!(original.assertions[0].id, same.assertions[0].id);
    equivalent["assertions"][0]["qualification"]["mode"] = json!("asserted");
    let different = atomic::validate(&equivalent.to_string(), &catalog, &registry)?;
    assert_ne!(same.assertions[0].id, different.assertions[0].id);
    registry
        .predicates
        .get_mut("monthly_budget")
        .unwrap()
        .description
        .push_str(" Version 2.");
    let changed = atomic::validate(&body.to_string(), &catalog, &registry)?;
    assert_ne!(original.assertions[0].id, changed.assertions[0].id);
    Ok(())
}
#[test]
fn replay_is_bound_to_source_revision_request_and_contract_version() -> Result<()> {
    let case = cases()?.remove(0);
    let (c, p, body) = fixture(&case)?;
    let registry = registry()?;
    let request = atomic::prepare_request(&c, &p, &registry, "base", 8192)?;
    assert_ne!(
        request.binding,
        atomic::prepare_request(&c, &p, &registry, "different_model", 8192)?.binding
    );
    assert_ne!(
        request.binding,
        atomic::prepare_request(&c, &p, &registry, "base", 4096)?.binding
    );
    let mut other = c.clone();
    other.document.key.scope = "other_tenant".into();
    let mut prepared = p.clone();
    prepared.document = other.document.key.clone();
    prepared.language.document = other.document.key.clone();
    prepared.canonical_context.organization_id = other.document.key.scope.clone();
    prepared.context_snapshot_hash =
        blake3::hash(prepared.canonical_context.canonical_prefix()?.as_bytes())
            .to_hex()
            .to_string();
    let other_request = atomic::prepare_request(&other, &prepared, &registry, "base", 8192)?;
    assert_ne!(request.binding, other_request.binding);
    assert!(atomic::validate(&body.to_string(), &request.catalog, &registry).is_ok());
    #[cfg(feature = "language")]
    {
        use aurelius_lanes::atomic_lab::{SavedResponse, replay};
        let mut saved = SavedResponse {
            contract_version: atomic::VERSION.into(),
            binding: request.binding.clone(),
            body: body.to_string(),
            usage: None,
        };
        assert!(replay(&saved, &request, &registry).is_ok());
        assert!(replay(&saved, &other_request, &registry).is_err());
        saved.contract_version = "v11-paragraphs".into();
        assert!(replay(&saved, &request, &registry).is_err());
    }
    Ok(())
}
#[test]
fn action_kind_and_conflicting_core_modalities_cannot_hide_in_hashes() -> Result<()> {
    let case = cases()?
        .into_iter()
        .find(|c| c["name"] == "complete_send_roles")
        .unwrap();
    let (c, p, body) = fixture(&case)?;
    let catalog = p.span_catalog(&c)?;
    let mut registry = registry()?;
    registry
        .action_requirements
        .insert("post".into(), registry.action_requirements["send"].clone());
    let original = atomic::validate(&body.to_string(), &catalog, &registry)?;
    assert_eq!(original.assertions.len(), 3);
    let mut changed = body.clone();
    changed["local_entities"][1]["action_kind"] = json!("post");
    let other = atomic::validate(&changed.to_string(), &catalog, &registry)?;
    assert_eq!(other.assertions.len(), 3);
    assert_ne!(original.assertions[0].id, other.assertions[0].id);
    changed = body;
    changed["assertions"][1]["qualification"]["mode"] = json!("asserted");
    let conflicting = atomic::validate(&changed.to_string(), &catalog, &registry)?;
    assert!(conflicting.assertions.is_empty());
    assert_eq!(conflicting.review_items.len(), 3);
    Ok(())
}
#[test]
fn same_action_anchor_under_alias_cannot_bypass_core_failure() -> Result<()> {
    let case = cases()?
        .into_iter()
        .find(|c| c["name"] == "failed_target_cascades")
        .unwrap();
    let (c, p, mut body) = fixture(&case)?;
    let catalog = p.span_catalog(&c)?;
    let registry = registry()?;
    let mut alias = body["local_entities"][0].clone();
    alias["id"] = json!("alias");
    body["local_entities"].as_array_mut().unwrap().push(alias);
    body["assertions"][0]["subject"]["ref"] = json!("alias");
    let graph = atomic::validate(&body.to_string(), &catalog, &registry)?;
    assert_eq!(graph.assertions.len(), 1);
    assert_eq!(graph.review_items.len(), 3);
    Ok(())
}
#[test]
fn rendering_inserts_values_once_and_never_interprets_label_slots() -> Result<()> {
    let mut case = cases()?
        .into_iter()
        .find(|c| c["name"] == "reported_ownership")
        .unwrap();
    case["source"] = json!(
        case["source"]
            .as_str()
            .unwrap()
            .replace("Acme Media", "Acme {object} Media")
    );
    case["entities"][1][2] = json!("Acme {object} Media");
    case["facts"][0]["quote"] = json!(
        case["facts"][0]["quote"]
            .as_str()
            .unwrap()
            .replace("Acme Media", "Acme {object} Media")
    );
    let (c, p, body) = fixture(&case)?;
    let graph = atomic::validate(&body.to_string(), &p.span_catalog(&c)?, &registry()?)?;
    assert_eq!(graph.assertions.len(), 1);
    assert!(graph.assertions[0].rendered.contains("{object}"));
    Ok(())
}
#[test]
fn paid_probe_regressions_are_constrained_during_native_generation() -> Result<()> {
    let cases = cases()?;
    let registry = registry()?;
    let (c, p, mut body) = fixture(&cases[0])?;
    let catalog = p.span_catalog(&c)?;
    let schema = registry.schema(&catalog)?;
    let variants = schema["properties"]["assertions"]["items"]["anyOf"]
        .as_array()
        .unwrap();
    let deadline = variants
        .iter()
        .find(|v| v["properties"]["predicate"]["const"] == "pull_deadline")
        .unwrap();
    let pattern = deadline["properties"]["object"]["properties"]["local_time"]["pattern"]
        .as_str()
        .unwrap();
    let time = regex::Regex::new(pattern)?;
    assert!(time.is_match("16:00"));
    assert!(!time.is_match("4pm"));
    let money = variants
        .iter()
        .find(|v| v["properties"]["predicate"]["const"] == "monthly_budget")
        .unwrap();
    let decimal = regex::Regex::new(
        money["properties"]["object"]["properties"]["amount"]["pattern"]
            .as_str()
            .unwrap(),
    )?;
    assert!(decimal.is_match("50000"));
    assert!(!decimal.is_match("50,000"));
    body["assertions"][0]["object"]["local_time"] = json!("4pm");
    assert!(atomic::validate(&body.to_string(), &catalog, &registry).is_err());
    for review in schema["properties"]["review_items"]["items"]["anyOf"]
        .as_array()
        .unwrap()
    {
        let props = &review["properties"];
        let href = props["span_ref"]["const"].as_str().unwrap();
        assert_eq!(props["exact_quote"]["const"], catalog.quote(href)?);
    }
    let (c, p, _) = fixture(
        cases
            .iter()
            .find(|c| c["name"] == "complete_send_roles")
            .unwrap(),
    )?;
    let catalog = p.span_catalog(&c)?;
    let nora = href(&c.document.text, "Nora");
    assert!(catalog.resolve(&nora).is_ok());
    assert!(!catalog.evidence_hrefs().contains(&nora));
    Ok(())
}
#[test]
fn noun_like_action_headings_are_not_named_entity_anchors() -> Result<()> {
    let case = json!({"source":"Reporting schedule\n[Jordan] Provide Lead Flows","entities":[["j","person","[Jordan] Provide Lead Flows",null]],"facts":[],"verbs":["Provide"]});
    let (c, p, _) = fixture(&case)?;
    let catalog = p.span_catalog(&c)?;
    let heading = href(&c.document.text, "[Jordan] Provide Lead Flows");
    assert!(catalog.resolve(&heading).is_ok());
    assert!(
        !catalog
            .anchor_hrefs(&atomic::AnchorKind::Mention)
            .contains(&heading)
    );
    assert!(
        catalog
            .anchor_hrefs(&atomic::AnchorKind::Action)
            .contains(&heading)
    );
    Ok(())
}
#[test]
fn native_review_identifiers_and_money_punctuation_match_the_contract() -> Result<()> {
    let registry = registry()?;
    for (quote, amount) in [
        ("Monthly budget $1,500, with KPIs.", "1500"),
        ("Monthly budget $5,000.", "5000"),
    ] {
        let case = json!({"source":format!("Budget plan\n{quote}"),"entities":[["b","budget_plan","Budget plan",null]],"facts":[{"subject":"b","predicate":"monthly_budget","object":{"kind":"money","amount":amount,"currency_symbol":"$","period":"month"},"mode":"asserted","quote":quote}]});
        let (c, p, body) = fixture(&case)?;
        let catalog = p.span_catalog(&c)?;
        let graph = atomic::validate(&body.to_string(), &catalog, &registry)?;
        assert_eq!(graph.assertions.len(), 1);
        let schema = registry.schema(&catalog)?;
        let review = &schema["properties"]["review_items"]["items"]["anyOf"][0]["properties"];
        let identifier =
            regex::Regex::new(review["intended_predicate"]["pattern"].as_str().unwrap())?;
        assert!(identifier.is_match("requested_push_action"));
        assert!(!identifier.is_match("requested push action"));
    }
    let case = json!({"source":"HubSpot tracking\nHubSpot events don’t have UTMs.","entities":[["t","tracking_configuration","HubSpot tracking",null]],"facts":[{"subject":"t","predicate":"operational_status","object":{"kind":"code","value":"missing"},"mode":"asserted","quote":"HubSpot events don’t have UTMs."}]});
    let (c, p, body) = fixture(&case)?;
    assert_eq!(
        atomic::validate(&body.to_string(), &p.span_catalog(&c)?, &registry)?
            .assertions
            .len(),
        1
    );
    Ok(())
}
#[test]
fn reporting_hypotheses_require_exact_source_reporter_without_autofill() -> Result<()> {
    let mut case = cases()?
        .into_iter()
        .find(|c| c["name"] == "reported_ownership")
        .unwrap();
    case["verbs"] = json!(["reported"]);
    let (c, p, mut body) = fixture(&case)?;
    let catalog = p.span_catalog(&c)?;
    let registry = registry()?;
    assert!(
        catalog.requires_attribution(
            body["assertions"][0]["evidence"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect::<Vec<_>>()
                .as_slice()
        )
    );
    body["assertions"][0]["qualification"]["attribution"] = json!([]);
    let rejected = atomic::validate(&body.to_string(), &catalog, &registry)?;
    assert!(rejected.assertions.is_empty());
    assert_eq!(
        rejected.review_items[0].reason,
        atomic::Reason::MissingAttribution
    );
    body["assertions"][0]["qualification"]["attribution"] = json!([{"ref":"j"}]);
    let accepted = atomic::validate(&body.to_string(), &catalog, &registry)?;
    assert_eq!(accepted.assertions.len(), 1);
    Ok(())
}
#[test]
fn task_title_requests_do_not_become_completed_business_actions() -> Result<()> {
    let source = "Task: Create artist profile\nStatus: completed\n\nNo artist page exists.";
    let case = json!({"source":source,"verbs":["Create"],"entities":[["a","action","Create artist profile","create"],["p","artist_profile","artist profile",null]],"facts":[{"subject":"a","predicate":"target","object":{"kind":"entity","ref":"p"},"mode":"required","quote":source}]});
    let (c, mut p, mut body) = fixture(&case)?;
    p.provider_header = Some(aurelius_lanes::provider_metadata::ProviderTaskHeader {
        text: "Task: Create artist profile\nStatus: completed\n\n".into(),
        title: range(source, "Create artist profile"),
    });
    let catalog = p.span_catalog(&c)?;
    let registry = registry()?;
    assert_eq!(
        atomic::validate(&body.to_string(), &catalog, &registry)?
            .assertions
            .len(),
        1
    );
    body["assertions"][0]["qualification"]["mode"] = json!("asserted");
    let rejected = atomic::validate(&body.to_string(), &catalog, &registry)?;
    assert!(rejected.assertions.is_empty());
    assert_eq!(
        rejected.review_items[0].reason,
        atomic::Reason::ActionIncomplete
    );
    Ok(())
}
#[test]
fn typed_values_cannot_drop_negation_units_or_forecast_mode() -> Result<()> {
    let cases = cases()?;
    let registry = registry()?;
    for (name, index, value) in [
        ("artist_profile_explicitly_absent", 0, "exists"),
        ("worked_hours_and_unbilled_period", 1, "billed"),
    ] {
        let case = cases.iter().find(|c| c["name"] == name).unwrap();
        let (c, p, mut body) = fixture(case)?;
        body["assertions"][index]["object"]["value"] = json!(value);
        let g = atomic::validate(&body.to_string(), &p.span_catalog(&c)?, &registry)?;
        assert!(
            g.review_items
                .iter()
                .any(|r| r.reason == atomic::Reason::InvalidTarget)
        );
    }
    let case = cases
        .iter()
        .find(|c| c["name"] == "field_layout_budget_and_forecast_are_separate")
        .unwrap();
    let (c, p, mut body) = fixture(case)?;
    body["assertions"][1]["qualification"]["mode"] = json!("asserted");
    let g = atomic::validate(&body.to_string(), &p.span_catalog(&c)?, &registry)?;
    assert_eq!(g.assertions.len(), 1);
    assert_eq!(g.review_items[0].reason, atomic::Reason::AmbiguousScope);
    let case = cases
        .iter()
        .find(|c| c["name"] == "initial_spend_and_zero_conversions")
        .unwrap();
    let (c, p, mut body) = fixture(case)?;
    body["assertions"][1]["object"]["unit"] = json!("hour");
    assert!(atomic::validate(&body.to_string(), &p.span_catalog(&c)?, &registry).is_err());
    Ok(())
}
