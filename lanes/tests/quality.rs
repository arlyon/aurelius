use anyhow::Result;
use aurelius_lanes::{
    config::Config,
    contracts::{ByteRange, Chunk, Document, DocumentKey},
    fact_context::FactContext,
    fact_preparation::PreparedFactSource,
    language::{LanguageAnalysis, Mention},
    quality::{self, QualityConfig, RejectionReason},
    time_value::{TimeValue, Validity},
};
use serde_json::{Value, json};

fn fixture() -> Result<(Chunk, PreparedFactSource, Config, QualityConfig)> {
    let config = Config::parse(include_str!("fixtures/business.toml"))?;
    let document = Document {
        key: DocumentKey {
            scope: "tenant-a".into(),
            source: "source-a".into(),
            revision: "revision-a".into(),
        },
        text: "Ada manages Company if approved. Deadline 2026-05-01.".into(),
        context: None,
        metadata: Default::default(),
    };
    // Explicit labelled validator fixtures, not an alternative runtime NLP backend.
    let mentions = [("ada", "Ada", false), ("company", "Company", false)]
        .into_iter()
        .map(|(id, text, pronoun)| {
            let start = document.text.find(text).unwrap();
            Mention {
                id: id.into(),
                range: ByteRange {
                    start,
                    end: start + text.len(),
                },
                text: text.into(),
                noun: true,
                pronoun,
                learned: true,
                raw_mention_score: Some(1.0),
            }
        })
        .collect();
    let context = FactContext {
        organization_id: document.key.scope.clone(),
        entities: vec![],
    };
    let hash = blake3::hash(context.canonical_prefix()?.as_bytes())
        .to_hex()
        .to_string();
    let policy: QualityConfig = serde_json::from_value(
        json!({"context_as_of":"2026-05-01T12:00:00Z","properties":{"deadline":{"description":"An explicitly stated deadline","subject_types":["person"],"value_kind":"time"},"status":{"description":"Provider status","subject_types":["person"],"value_kind":"text","metadata_only":true}}}),
    )?;
    let prepared = PreparedFactSource {
        provider_header: None,
        document: document.key.clone(),
        content_hash: document.content_hash(),
        preparation_profile: "a".repeat(64),
        language: LanguageAnalysis {
            sentence_spans: vec![],
            document: document.key.clone(),
            content_hash: document.content_hash(),
            profile: "b".repeat(64),
            mentions,
            coreference_hypotheses: vec![],
            verb_spans: vec![],
            number_spans: vec![],
        },
        alias_hypotheses: vec![],
        canonical_context: context,
        context_effective_at: policy.context_as_of,
        context_snapshot_hash: hash,
    };
    let chunk = Chunk {
        range: ByteRange {
            start: 0,
            end: document.text.len(),
        },
        document,
    };
    Ok((chunk, prepared, config, policy))
}
fn relation() -> Value {
    json!({"subject":{"mention_id":"ada","entity_type":"person"},"body":{"kind":"relation","predicate":"manages","object":{"mention_id":"company","entity_type":"company"}},"quote":"Ada manages Company if approved.","statement_type":"observation","polarity":"positive","modality":"may","qualifiers":{"adverbs":[],"conditions":["if approved"],"scope":null},"validity":{"from":null,"to":null}})
}
fn graph(facts: Vec<Value>) -> Result<quality::QualityGraph> {
    let (c, p, config, policy) = fixture()?;
    quality::parse_graph(
        &json!({"facts":facts}).to_string(),
        &c,
        &p,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )
}
#[test]
fn closed_relations_validate_types_and_preserve_qualifiers() -> Result<()> {
    let mut unknown = relation();
    unknown["body"]["predicate"] = json!("random_verb");
    let mut wrong = relation();
    wrong["body"]["object"]["entity_type"] = json!("person");
    let result = graph(vec![relation(), unknown, wrong])?;
    assert_eq!(result.candidates.len(), 1);
    assert_eq!(result.mentions.len(), 2);
    assert_eq!(
        result.rejections[0].reason,
        RejectionReason::UnknownPredicate
    );
    assert_eq!(result.rejections[1].reason, RejectionReason::EndpointType);
    assert_eq!(
        serde_json::to_value(&result.candidates[0].candidate)?["polarity"],
        "positive"
    );
    assert_eq!(
        result.candidates[0].candidate.qualifiers.conditions,
        ["if approved"]
    );
    Ok(())
}
#[test]
fn source_missing_mentions_and_fabricated_conditions_are_not_candidates() -> Result<()> {
    let mut missing = relation();
    missing["subject"]["mention_id"] = json!("invented");
    let mut quote = relation();
    quote["quote"] = json!("Ada definitely owns Company.");
    let mut condition = relation();
    condition["qualifiers"]["conditions"] = json!(["unless revenue falls"]);
    let result = graph(vec![missing, quote, condition])?;
    assert!(result.candidates.is_empty());
    assert_eq!(
        result
            .rejections
            .iter()
            .map(|r| r.reason.clone())
            .collect::<Vec<_>>(),
        [
            RejectionReason::UnknownMention,
            RejectionReason::MissingSourceEvidence,
            RejectionReason::UnanchoredQualifier
        ]
    );
    Ok(())
}
#[test]
fn pronouns_are_reviewed_even_with_a_model_cluster() -> Result<()> {
    let (c, mut p, config, policy) = fixture()?;
    p.language.mentions[0].pronoun = true;
    p.language.coreference_hypotheses = vec![vec!["ada".into(), "company".into()]];
    let result = quality::parse_graph(
        &json!({"facts":[relation()]}).to_string(),
        &c,
        &p,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(
        result.rejections[0].reason,
        RejectionReason::AmbiguousReference
    );
    assert!(result.candidates.is_empty());
    Ok(())
}
#[test]
fn property_dates_preserve_precision_and_provider_metadata_is_not_llm_work() -> Result<()> {
    let mut fact = relation();
    fact["quote"] = json!("Ada manages Company if approved. Deadline 2026-05-01.");
    fact["body"] = json!({"kind":"property","property":"deadline","value":{"kind":"time","value":{"kind":"date","value":"2026-05-01"}},"value_quote":"2026-05-01"});
    let mut invented = fact.clone();
    invented["body"]["value"]["value"] = json!({"kind":"timestamp","value":"2026-05-01T00:00:00Z"});
    let mut metadata = fact.clone();
    metadata["body"] = json!({"kind":"property","property":"status","value":{"kind":"text","value":"approved"},"value_quote":"approved"});
    let result = graph(vec![fact, invented, metadata])?;
    assert_eq!(result.candidates.len(), 1);
    assert_eq!(result.rejections[0].reason, RejectionReason::PropertyValue);
    assert_eq!(result.rejections[1].reason, RejectionReason::MetadataOnly);
    Ok(())
}
#[test]
fn relative_dates_periods_and_validity_do_not_invent_anchors() -> Result<()> {
    let relative: TimeValue =
        serde_json::from_value(json!({"kind":"relative","text":"next Friday"}))?;
    relative.validate()?;
    assert!(relative.grounded_by("by next Friday"));
    let period: TimeValue = serde_json::from_value(
        json!({"kind":"period","start":{"kind":"month","year":2026,"month":5},"end":{"kind":"month","year":2026,"month":6},"end_inclusive":true}),
    )?;
    period.validate()?;
    assert!(period.grounded_by("May 2026 through June 2026"));
    assert!(
        serde_json::from_value::<TimeValue>(json!({"kind":"date","value":"2026-02-30"})).is_err()
    );
    assert!(
        serde_json::from_value::<Validity>(
            json!({"from":{"kind":"relative","text":"tomorrow"},"to":null})
        )
        .is_err()
    );
    let mixed: Validity = serde_json::from_value(
        json!({"from":{"kind":"date","value":"2026-05-01"},"to":{"kind":"timestamp","value":"2026-06-01T00:00:00Z"}}),
    )?;
    assert!(mixed.validate().is_err());
    Ok(())
}
#[test]
fn source_metadata_moves_only_in_suffix_and_empty_extraction_is_normal() -> Result<()> {
    let (c, p, config, policy) = fixture()?;
    let (a, b) = quality::model_input(&c, &p, &config.ontology, &policy)?;
    let mut changed = c.clone();
    changed.document.metadata.modified_at = Some("2026-07-01T00:00:00Z".parse()?);
    let (x, y) = quality::model_input(&changed, &p, &config.ontology, &policy)?;
    assert_eq!(a, x);
    assert_ne!(b, y);
    let input: Value = serde_json::from_str(&b)?;
    assert_eq!(input["mentions"][0].as_object().unwrap().len(), 3);
    assert!(input["mentions"][0].get("raw_mention_score").is_none());
    let mut reordered = p.clone();
    reordered.language.coreference_hypotheses = vec![vec!["company".into(), "ada".into()]];
    let (_, left) = quality::model_input(&c, &reordered, &config.ontology, &policy)?;
    reordered.language.coreference_hypotheses[0].reverse();
    let (_, right) = quality::model_input(&c, &reordered, &config.ontology, &policy)?;
    assert_eq!(left, right);
    let empty = graph(vec![])?;
    assert!(empty.candidates.is_empty() && empty.rejections.is_empty());
    let mut forged = relation();
    forged["subject"]["canonical_id"] = json!("user:invented");
    let rejected = graph(vec![forged])?;
    assert!(rejected.candidates.is_empty());
    assert_eq!(
        rejected.rejections[0].reason,
        RejectionReason::InvalidCandidateShape
    );
    Ok(())
}
#[test]
fn duplicate_text_in_different_source_revisions_is_not_deduplicated() -> Result<()> {
    let original = graph(vec![relation()])?;
    let (mut c, mut p, config, policy) = fixture()?;
    c.document.key.revision = "another-revision".into();
    p.document = c.document.key.clone();
    p.language.document = c.document.key.clone();
    let another = quality::parse_graph(
        &json!({"facts":[relation()]}).to_string(),
        &c,
        &p,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_ne!(original.candidates[0].id, another.candidates[0].id);
    Ok(())
}
#[tokio::test]
async fn canonical_context_credentials_are_blocked_before_transport() -> Result<()> {
    use aurelius_lanes::{
        config::{ChatApi, ChatConfig},
        fact_context::{ContextEntity, EntityFacts, KnownFact},
        facts::{ChatExtractor, ExtractionReview},
    };
    let (chunk, mut prepared, config, policy) = fixture()?;
    prepared.canonical_context.entities.push(EntityFacts {
        entity: ContextEntity {
            id: "person:ada".into(),
            name: "Ada".into(),
        },
        facts: vec![KnownFact {
            id: "fixture-context".into(),
            subject_ref: "person:ada".into(),
            predicate: "has_business_goal".into(),
            object_ref: None,
            object_value: Some("OPENAI_API_KEY=fixtureCredentialValue".into()),
            evidence: vec![],
            valid_from: None,
            valid_to: None,
        }],
        has_more: false,
    });
    prepared.context_snapshot_hash =
        blake3::hash(prepared.canonical_context.canonical_prefix()?.as_bytes())
            .to_hex()
            .to_string();
    let extractor = ChatExtractor::new(
        &ChatConfig {
            endpoint: "http://127.0.0.1:9/responses".into(),
            model: "no-paid-inference".into(),
            revision: "fixture".into(),
            api_key_env: None,
            api: ChatApi::Responses,
            gateway: None,
            references: Default::default(),
        },
        config.ontology,
    )?;
    let error = extractor
        .extract_quality(&chunk, &prepared, &policy, &"c".repeat(64))
        .await
        .unwrap_err();
    assert!(
        matches!(
            error.downcast_ref::<ExtractionReview>(),
            Some(ExtractionReview::Input)
        ),
        "canonical credentials reached transport or were treated as a provider error"
    );
    Ok(())
}
#[test]
fn temporal_literals_cannot_invent_offsets_or_change_precision() -> Result<()> {
    let timestamp: TimeValue =
        serde_json::from_value(json!({"kind":"timestamp","value":"2026-05-01T09:30:00+02:00"}))?;
    assert!(timestamp.literal_matches("2026-05-01T09:30:00+02:00"));
    assert!(!timestamp.literal_matches("2026-05-01T07:30:00Z"));
    let date: TimeValue = serde_json::from_value(json!({"kind":"date","value":"2026-05-01"}))?;
    assert!(!date.literal_matches("2026-05-01T00:00:00Z"));
    let month: TimeValue = serde_json::from_value(json!({"kind":"month","year":2026,"month":5}))?;
    assert!(!month.literal_matches("2026-05-01"));
    Ok(())
}
#[test]
fn conflicting_pos_and_incomplete_noun_spans_are_reviewed() -> Result<()> {
    let (chunk, mut prepared, config, policy) = fixture()?;
    prepared
        .language
        .verb_spans
        .push(prepared.language.mentions[0].range);
    let conflict = quality::parse_graph(
        &json!({"facts":[relation()]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(
        conflict.rejections[0].reason,
        RejectionReason::LinguisticConflict
    );
    let mut chunk = chunk;
    chunk.document.text = "Ada manages Company Holdings if approved.".into();
    chunk.range.end = chunk.document.text.len();
    prepared.content_hash = chunk.document.content_hash();
    prepared.language.content_hash = prepared.content_hash.clone();
    prepared.language.verb_spans.clear();
    for (id, text) in [
        ("full-company", "Company Holdings"),
        ("holdings", "Holdings"),
    ] {
        let start = chunk.document.text.find(text).unwrap();
        prepared.language.mentions.push(Mention {
            id: id.into(),
            text: text.into(),
            range: ByteRange {
                start,
                end: start + text.len(),
            },
            noun: true,
            pronoun: false,
            learned: false,
            raw_mention_score: None,
        });
    }
    let mut partial = relation();
    partial["quote"] = json!(chunk.document.text);
    let mut complete = partial.clone();
    complete["body"]["object"]["mention_id"] = json!("full-company");
    let graph = quality::parse_graph(
        &json!({"facts":[partial,complete]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(graph.rejections[0].reason, RejectionReason::PartialMention);
    assert_eq!(graph.candidates.len(), 1);
    Ok(())
}
#[test]
fn metadata_aliases_and_header_crossings_are_blocked_but_narrative_dates_survive() -> Result<()> {
    let (mut chunk, mut prepared, config, policy) = fixture()?;
    let header = "Task: Ada\nDue date: 2026-05-01\nStatus: open";
    chunk.document.text = format!("{header}\n\nAda deadline 2026-05-02.");
    chunk.range.end = chunk.document.text.len();
    prepared.content_hash = chunk.document.content_hash();
    prepared.language.content_hash = prepared.content_hash.clone();
    prepared.language.mentions.truncate(1);
    let subject = &mut prepared.language.mentions[0];
    let start = chunk.document.text.rfind("Ada").unwrap();
    subject.range = ByteRange {
        start,
        end: start + 3,
    };
    prepared.provider_header = Some(aurelius_lanes::provider_metadata::ProviderTaskHeader {
        text: header.into(),
        title: ByteRange { start: 6, end: 9 },
    });
    let property = |date: &str| json!({"subject":{"mention_id":"ada","entity_type":"person"},"body":{"kind":"property","property":"deadline","value":{"kind":"time","value":{"kind":"date","value":date}},"value_quote":date},"quote":chunk.document.text,"statement_type":"observation","polarity":"positive","modality":"asserted","qualifiers":{"adverbs":[],"conditions":[],"scope":null},"validity":{"from":null,"to":null}});
    let graph = quality::parse_graph(
        &json!({"facts":[property("2026-05-01"),property("2026-05-02")]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(graph.rejections[0].reason, RejectionReason::MetadataOnly);
    assert_eq!(graph.candidates.len(), 1);
    let mut crossed = prepared.language.mentions[0].clone();
    crossed.range = ByteRange { start: 6, end: 18 };
    crossed.text = crossed.range.text(&chunk.document)?.into();
    prepared.language.mentions[0] = crossed;
    let graph = quality::parse_graph(
        &json!({"facts":[property("2026-05-02")]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(
        graph.rejections[0].reason,
        RejectionReason::ProviderBoundary
    );
    let mut corrupt = prepared.clone();
    corrupt
        .provider_header
        .as_mut()
        .unwrap()
        .text
        .push_str("tampered");
    assert!(quality::model_input(&chunk, &corrupt, &config.ontology, &policy).is_err());
    Ok(())
}
#[test]
fn unverified_action_scope_is_reviewed_without_relabeling_the_action() -> Result<()> {
    let (chunk, prepared, mut config, policy) = fixture()?;
    config.ontology.predicates.insert(
        "assigned_to".into(),
        config.ontology.predicates["manages"].clone(),
    );
    let mut assignment = relation();
    assignment["body"]["predicate"] = json!("assigned_to");
    let graph = quality::parse_graph(
        &json!({"facts":[assignment,relation()]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(
        graph.rejections[0].reason,
        RejectionReason::UnverifiedActionScope
    );
    assert_eq!(graph.candidates.len(), 1);
    Ok(())
}
#[test]
fn a_literal_cannot_ground_an_entity_by_repeating_its_type_as_property() -> Result<()> {
    let (chunk, prepared, config, mut policy) = fixture()?;
    policy.properties.insert(
        "person".into(),
        serde_json::from_value(
            json!({"description":"Person literal","subject_types":["person"],"value_kind":"text"}),
        )?,
    );
    let mut fact = relation();
    fact["body"] = json!({"kind":"property","property":"person","value":{"kind":"text","value":"Ada"},"value_quote":"Ada"});
    let graph = quality::parse_graph(
        &json!({"facts":[fact]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(graph.rejections[0].reason, RejectionReason::SelfReference);
    Ok(())
}
#[test]
fn malformed_candidate_does_not_discard_independent_well_typed_claims() -> Result<()> {
    let (chunk, prepared, config, policy) = fixture()?;
    let body = json!({"facts":[relation(),{"secret_untrusted_field":"do-not-log"}]}).to_string();
    assert_eq!(quality::parse_proposals(&body)?.len(), 2);
    let graph = quality::parse_graph(
        &body,
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(graph.candidates.len(), 1);
    assert_eq!(graph.rejections.len(), 1);
    assert_eq!(graph.rejections[0].index, 1);
    assert_eq!(
        graph.rejections[0].reason,
        RejectionReason::InvalidCandidateShape
    );
    assert!(!serde_json::to_string(&graph)?.contains("do-not-log"));
    Ok(())
}
#[test]
fn invalid_typed_json_has_a_safe_persistable_diagnostic() -> Result<()> {
    let (chunk, prepared, config, policy) = fixture()?;
    let error = quality::parse_graph(
        r#"{"facts":[],"secret_untrusted_field":"do-not-log"}"#,
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )
    .unwrap_err();
    let code = aurelius_lanes::facts::review_diagnostic(&error);
    assert_eq!(code, aurelius_lanes::facts::ReviewDiagnostic::TypedJson);
    assert_eq!(serde_json::to_string(&code)?, "\"typed_json\"");
    assert!(!format!("{code}").contains("do-not-log"));
    assert!(!format!("{error:#}").contains("do-not-log"));
    assert!(!format!("{error:?}").contains("secret_untrusted_field"));
    let error = quality::parse_graph(
        &json!({"facts":vec![relation();21]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )
    .unwrap_err();
    assert_eq!(
        aurelius_lanes::facts::review_diagnostic(&error),
        aurelius_lanes::facts::ReviewDiagnostic::CandidateBudget
    );
    Ok(())
}
#[test]
fn opt_in_whitespace_alignment_restores_original_bytes_and_retains_proposal() -> Result<()> {
    let (chunk, prepared, config, mut policy) = fixture()?;
    let mut proposed = relation();
    proposed["quote"] = json!("Ada\r\nmanages Company if\tapproved.");
    proposed["qualifiers"]["conditions"] = json!(["if\tapproved"]);
    let strict = quality::parse_graph(
        &json!({"facts":[proposed.clone()]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert!(strict.candidates.is_empty());
    policy.align_whitespace = true;
    let graph = quality::parse_graph(
        &json!({"facts":[proposed.clone()]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(graph.candidates.len(), 1);
    let checked = &graph.candidates[0];
    assert_eq!(
        serde_json::to_value(checked.proposed.as_ref().unwrap())?,
        proposed
    );
    assert_eq!(checked.candidate.quote, "Ada manages Company if approved.");
    assert_eq!(
        checked.evidence[0].quote,
        checked.evidence[0].range.text(&chunk.document)?
    );
    assert_eq!(checked.candidate.qualifiers.conditions, ["if approved"]);
    policy.properties.insert(
        "note".into(),
        serde_json::from_value(
            json!({"description":"Verbatim note","subject_types":["person"],"value_kind":"text"}),
        )?,
    );
    let mut literal = proposed.clone();
    literal["body"] = json!({"kind":"property","property":"note","value":{"kind":"text","value":"manages\r\nCompany if\tapproved"},"value_quote":"manages\r\nCompany if\tapproved"});
    let resolved = quality::parse_graph(
        &json!({"facts":[literal]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(resolved.candidates.len(), 1);
    let value = serde_json::to_value(&resolved.candidates[0].candidate.body)?;
    assert_eq!(value["value_quote"], "manages Company if approved");
    assert_eq!(value["value"]["value"], value["value_quote"]);
    let mut bad = proposed;
    bad["quote"] = json!("Ada never manages Company if approved.");
    assert!(
        quality::parse_graph(
            &json!({"facts":[bad]}).to_string(),
            &chunk,
            &prepared,
            &config.ontology,
            &policy,
            &"c".repeat(64)
        )?
        .candidates
        .is_empty()
    );
    Ok(())
}
#[test]
fn literal_occurrence_alignment_uses_real_quote_local_nlp_mentions_not_identity_merges()
-> Result<()> {
    let (mut chunk, mut prepared, config, mut policy) = fixture()?;
    let mut later = prepared
        .language
        .mentions
        .iter()
        .find(|m| m.id == "ada")
        .unwrap()
        .clone();
    chunk.document.text.push_str(" Ada");
    later.id = "later-ada".into();
    later.range = ByteRange {
        start: chunk.document.text.len() - 3,
        end: chunk.document.text.len(),
    };
    chunk.range.end = chunk.document.text.len();
    prepared.language.content_hash = chunk.document.content_hash();
    prepared.content_hash = chunk.document.content_hash();
    prepared.language.mentions.push(later);
    let mut fact = relation();
    fact["subject"]["mention_id"] = json!("later-ada");
    assert!(
        quality::parse_graph(
            &json!({"facts":[fact.clone()]}).to_string(),
            &chunk,
            &prepared,
            &config.ontology,
            &policy,
            &"c".repeat(64)
        )?
        .candidates
        .is_empty()
    );
    policy.align_mentions = true;
    let graph = quality::parse_graph(
        &json!({"facts":[fact]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(graph.candidates.len(), 1);
    assert_eq!(graph.candidates[0].candidate.subject.mention_id, "ada");
    assert_eq!(
        graph.candidates[0]
            .proposed
            .as_ref()
            .unwrap()
            .subject
            .mention_id,
        "later-ada"
    );
    assert_eq!(
        graph
            .mentions
            .iter()
            .find(|m| m.id == "ada")
            .unwrap()
            .raw_mention_score,
        Some(1.0)
    );
    Ok(())
}
#[test]
fn quote_anchors_include_original_intervening_bytes_and_reject_ambiguous_bounds() -> Result<()> {
    let (chunk, prepared, config, mut policy) = fixture()?;
    let mut fact = relation();
    fact.as_object_mut().unwrap().remove("quote");
    fact["quote_start"] = json!("Ada");
    fact["quote_end"] = json!("approved.");
    assert!(
        quality::parse_graph(
            &json!({"facts":[fact.clone()]}).to_string(),
            &chunk,
            &prepared,
            &config.ontology,
            &policy,
            &"c".repeat(64)
        )?
        .candidates
        .is_empty()
    );
    policy.quote_anchors = true;
    let graph = quality::parse_graph(
        &json!({"facts":[fact.clone()]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(graph.candidates.len(), 1);
    assert_eq!(
        graph.candidates[0].candidate.quote,
        "Ada manages Company if approved."
    );
    assert!(graph.candidates[0].candidate.quote_start.is_none());
    assert!(
        graph.candidates[0]
            .proposed
            .as_ref()
            .unwrap()
            .quote_start
            .is_some()
    );
    let mut mixed = fact.clone();
    mixed["quote"] = json!("Ada manages Company if approved.");
    assert!(
        quality::parse_graph(
            &json!({"facts":[mixed]}).to_string(),
            &chunk,
            &prepared,
            &config.ontology,
            &policy,
            &"c".repeat(64)
        )?
        .candidates
        .is_empty()
    );
    fact["quote_start"] = json!("approved.");
    fact["quote_end"] = json!("Ada");
    assert!(
        quality::parse_graph(
            &json!({"facts":[fact]}).to_string(),
            &chunk,
            &prepared,
            &config.ontology,
            &policy,
            &"c".repeat(64)
        )?
        .candidates
        .is_empty()
    );
    Ok(())
}
#[test]
fn explicit_negative_claim_retains_polarity_and_source_condition() -> Result<()> {
    let (mut chunk, mut prepared, config, policy) = fixture()?;
    chunk.document.text = "Ada does not manage Company if approved.".into();
    chunk.range.end = chunk.document.text.len();
    prepared.content_hash = chunk.document.content_hash();
    prepared.language.content_hash = prepared.content_hash.clone();
    for mention in &mut prepared.language.mentions {
        let start = chunk
            .document
            .text
            .find(&mention.text)
            .expect("fixture mention");
        mention.range = ByteRange {
            start,
            end: start + mention.text.len(),
        };
    }
    let mut fact = relation();
    fact["quote"] = json!(chunk.document.text);
    fact["polarity"] = json!("negative");
    let graph = quality::parse_graph(
        &json!({"facts":[fact]}).to_string(),
        &chunk,
        &prepared,
        &config.ontology,
        &policy,
        &"c".repeat(64),
    )?;
    assert_eq!(graph.candidates.len(), 1);
    assert_eq!(
        serde_json::to_value(&graph.candidates[0].candidate)?["polarity"],
        "negative"
    );
    assert_eq!(
        graph.candidates[0].candidate.qualifiers.conditions,
        ["if approved"]
    );
    Ok(())
}
