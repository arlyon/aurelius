use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::Result;
use chrono::{Datelike, Local, NaiveDate, Utc};
use serde::Serialize;
use swiftide::traits::ChatCompletion;
use tracing::info;
use uuid::Uuid;

use crate::metabolic::facts::{
    delete_all_facts, get_or_create_evicted_facts_table, get_or_create_facts_table,
    is_functional_predicate, load_all_facts, resolve_contradiction, synthesize_neuron, write_facts,
    ContradictionResolution, Fact,
};
use crate::metabolic::social_graph::run_social_graph;

const DECAY_LAMBDA: f64 = 0.001; // per hour
const DECAY_THRESHOLD: f32 = 0.05;
const WEAK_LOWER: f32 = 0.05;
const WEAK_UPPER: f32 = 0.3;

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PulseLine {
    HighEq {
        subject: String,
        predicate: String,
        object: String,
        days_until: i64,
    },
    Neuron {
        subject: String,
        predicate: String,
        object: String,
        confidence: f32,
    },
    Health {
        total_facts: usize,
        decayed_count: usize,
        neurons_merged: usize,
        contradictions_resolved: usize,
    },
    BlockedAction {
        action: String,
        client: String,
        invoice_id: String,
    },
}

pub async fn run_dream(dry_run: bool, yes: bool, completion: &dyn ChatCompletion) -> Result<()> {
    let db = lancedb::connect(&crate::persistence::db_path())
        .execute()
        .await?;
    let facts_table = get_or_create_facts_table(&db).await?;

    let all_facts = load_all_facts(&facts_table).await?;
    info!("Loaded {} facts for dream cycle", all_facts.len());

    if all_facts.is_empty() {
        info!("No facts found — run `aurelius ingest` first");
        return Ok(());
    }

    let now_micros = Utc::now().timestamp_micros();

    // Split previously-derived/neuron facts from extracted facts.
    // Derived facts are transient — we'll re-compute them fresh each run.
    let (_, extracted): (Vec<Fact>, Vec<Fact>) = all_facts
        .into_iter()
        .partition(|f| f.chunk_id == "derived" || f.chunk_id == "neuron");

    // ---- Phase A: Exponential Decay ----
    let mut surviving: Vec<Fact> = Vec::new();
    let mut decayed_count = 0usize;

    for mut fact in extracted {
        if fact.is_core_truth {
            surviving.push(fact);
            continue;
        }
        let hours = (now_micros - fact.created_at).max(0) as f64 / 3_600_000_000.0;
        let new_conf = (fact.confidence as f64 * (-DECAY_LAMBDA * hours).exp()) as f32;
        if new_conf < DECAY_THRESHOLD {
            decayed_count += 1;
        } else {
            fact.confidence = new_conf;
            surviving.push(fact);
        }
    }

    // ---- Phase A: Social Graph ----
    let sg = run_social_graph(&surviving);

    // ---- Phase A: Contradiction Detection ----
    // Group surviving non-core facts by (subject, predicate)
    let mut groups: HashMap<(String, String), Vec<usize>> = HashMap::new();
    for (i, fact) in surviving.iter().enumerate() {
        if !fact.is_core_truth {
            groups
                .entry((fact.subject.clone(), fact.predicate.clone()))
                .or_default()
                .push(i);
        }
    }

    let mut contradiction_pairs: Vec<(usize, usize)> = Vec::new();
    for indices in groups.values() {
        if indices.len() < 2 {
            continue;
        }
        let predicate = &surviving[indices[0]].predicate;
        if !is_functional_predicate(predicate) {
            continue;
        }

        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let a = &surviving[indices[i]];
                let b = &surviving[indices[j]];
                if a.object != b.object {
                    contradiction_pairs.push((indices[i], indices[j]));
                }
            }
        }
    }

    // ---- Phase A: Derive Facts from Social Graph ----
    let mut derived: Vec<Fact> = Vec::new();

    for (a, b) in &sg.spouse_of {
        if !surviving
            .iter()
            .any(|f| f.predicate == "spouse_of" && &f.subject == b && &f.object == a)
        {
            derived.push(Fact {
                id: Uuid::new_v4().to_string(),
                chunk_id: "derived".to_string(),
                subject: b.clone(),
                predicate: "spouse_of".to_string(),
                object: a.clone(),
                confidence: 1.0,
                is_core_truth: false,
                created_at: now_micros,
            });
        }
    }

    for (a, b) in &sg.child_of {
        if !surviving
            .iter()
            .any(|f| f.predicate == "child_of" && &f.subject == a && &f.object == b)
        {
            derived.push(Fact {
                id: Uuid::new_v4().to_string(),
                chunk_id: "derived".to_string(),
                subject: a.clone(),
                predicate: "child_of".to_string(),
                object: b.clone(),
                confidence: 1.0,
                is_core_truth: false,
                created_at: now_micros,
            });
        }
    }

    for person in &sg.hold_communications {
        derived.push(Fact {
            id: Uuid::new_v4().to_string(),
            chunk_id: "derived".to_string(),
            subject: person.clone(),
            predicate: "hold_communications".to_string(),
            object: "active".to_string(),
            confidence: 1.0,
            is_core_truth: false,
            created_at: now_micros,
        });
    }

    for (a, b, trust_mu) in &sg.social_trust {
        derived.push(Fact {
            id: Uuid::new_v4().to_string(),
            chunk_id: "derived".to_string(),
            subject: a.clone(),
            predicate: "social_trust_derived".to_string(),
            object: b.clone(),
            confidence: (*trust_mu as f32 / 1000.0).clamp(0.0, 1.0),
            is_core_truth: false,
            created_at: now_micros,
        });
    }

    info!(
        "Phase A: {} decayed, {} contradictions, {} derived facts",
        decayed_count,
        contradiction_pairs.len(),
        derived.len()
    );

    // ---- Phase B: Contradiction Resolution ----
    let mut contradictions_resolved = 0usize;
    let mut contradiction_loser_ids: HashSet<String> = HashSet::new();
    let mut resolution_details = Vec::new();

    if !contradiction_pairs.is_empty() {
        for (ai, bi) in &contradiction_pairs {
            let fact_a = &surviving[*ai];
            let fact_b = &surviving[*bi];
            match resolve_contradiction(completion, fact_a, fact_b).await {
                Ok(winner) => {
                    let loser_id = if winner == ContradictionResolution::A {
                        resolution_details.push(format!(
                            "Resolved contradiction for {} {}: kept '{}' over '{}'",
                            fact_a.subject, fact_a.predicate, fact_a.object, fact_b.object
                        ));
                        surviving[*bi].id.clone()
                    } else if winner == ContradictionResolution::B {
                        resolution_details.push(format!(
                            "Resolved contradiction for {} {}: kept '{}' over '{}'",
                            fact_b.subject, fact_b.predicate, fact_b.object, fact_a.object
                        ));
                        surviving[*ai].id.clone()
                    } else {
                        continue;
                    };
                    contradiction_loser_ids.insert(loser_id);
                    contradictions_resolved += 1;
                }
                Err(e) => tracing::warn!("Contradiction resolution failed: {}", e),
            }
        }
    }

    // ---- Phase B: Weak Fact Clustering (Neuron Synthesis) ----
    let contradiction_ids: HashSet<String> = contradiction_pairs
        .iter()
        .flat_map(|(ai, bi)| [surviving[*ai].id.clone(), surviving[*bi].id.clone()])
        .collect();

    let mut weak_by_subject: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, fact) in surviving.iter().enumerate() {
        if fact.confidence >= WEAK_LOWER
            && fact.confidence <= WEAK_UPPER
            && !fact.is_core_truth
            && !contradiction_ids.contains(&fact.id)
        {
            weak_by_subject
                .entry(fact.subject.clone())
                .or_default()
                .push(i);
        }
    }

    let mut neuron_facts: Vec<Fact> = Vec::new();
    let mut weak_ids_to_delete: HashSet<String> = HashSet::new();
    let mut neurons_merged = 0usize;
    let mut neuron_details = Vec::new();

    if weak_by_subject.values().any(|v| v.len() >= 2) {
        for (subject, indices) in &weak_by_subject {
            if indices.len() < 2 {
                continue;
            }
            let cluster: Vec<&Fact> = indices.iter().map(|&i| &surviving[i]).collect();
            match synthesize_neuron(completion, subject, &cluster).await {
                Ok(Some((predicate, object, confidence))) => {
                    neuron_facts.push(Fact {
                        id: Uuid::new_v4().to_string(),
                        chunk_id: "neuron".to_string(),
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                        confidence,
                        is_core_truth: false,
                        created_at: now_micros,
                    });
                    neuron_details.push(format!(
                        "Synthesized neuron for {}: {} {} (conf: {:.2})",
                        subject, predicate, object, confidence
                    ));
                    for &i in indices {
                        weak_ids_to_delete.insert(surviving[i].id.clone());
                    }
                    neurons_merged += 1;
                }
                Ok(None) => {}
                Err(e) => tracing::warn!("Neuron synthesis failed for {}: {}", subject, e),
            }
        }
    }

    // ---- Build Final State ----
    let all_delete_ids: HashSet<String> = contradiction_loser_ids
        .iter()
        .chain(weak_ids_to_delete.iter())
        .cloned()
        .collect();

    let (final_extracted, evicted): (Vec<Fact>, Vec<Fact>) = surviving
        .into_iter()
        .partition(|f| !all_delete_ids.contains(&f.id));

    let total_facts = final_extracted.len() + derived.len() + neuron_facts.len();

    // ---- Calendar Events (High-EQ) ----
    let mut high_eq_events: Vec<(String, String, String, i64)> = Vec::new();
    for fact in &final_extracted {
        if fact.predicate == "born_on" || fact.predicate == "life_event" {
            if let Some(days) = parse_days_until(&fact.object) {
                if days >= 0 && days <= 14 {
                    high_eq_events.push((
                        fact.subject.clone(),
                        fact.predicate.clone(),
                        fact.object.clone(),
                        days,
                    ));
                }
            }
        }
    }
    high_eq_events.sort_by_key(|(_, _, _, d)| *d);

    // ---- Build Pulse Lines ----
    let mut pulse: Vec<PulseLine> = Vec::new();

    for (subject, predicate, object, days_until) in &high_eq_events {
        pulse.push(PulseLine::HighEq {
            subject: subject.clone(),
            predicate: predicate.clone(),
            object: object.clone(),
            days_until: *days_until,
        });
    }

    for fact in &neuron_facts {
        pulse.push(PulseLine::Neuron {
            subject: fact.subject.clone(),
            predicate: fact.predicate.clone(),
            object: fact.object.clone(),
            confidence: fact.confidence,
        });
    }

    pulse.push(PulseLine::Health {
        total_facts,
        decayed_count,
        neurons_merged,
        contradictions_resolved,
    });

    for (action, client, invoice_id) in &sg.blocked_actions {
        pulse.push(PulseLine::BlockedAction {
            action: action.clone(),
            client: client.clone(),
            invoice_id: invoice_id.clone(),
        });
    }

    // ---- Summary and Confirmation ----
    if dry_run {
        println!("\n--- Dream Cycle Summary ---");
        println!("  Decayed facts:           {}", decayed_count);
        println!("  Contradictions resolved: {}", contradictions_resolved);
        for detail in &resolution_details {
            println!("    - {}", detail);
        }
        println!("  Neurons synthesized:     {}", neurons_merged);
        for detail in &neuron_details {
            println!("    - {}", detail);
        }
        println!("  Derived facts:           {}", derived.len());
        println!("  Total facts in DB:       {}", total_facts);

        println!("\nDry-run — no changes written");
        for line in &pulse {
            println!("{}", serde_json::to_string(line)?);
        }
        return Ok(());
    }

    if contradiction_pairs.is_empty() {
        println!("feeling refreshed");
    } else {
        println!("\n--- Dream Cycle Summary ---");
        println!("  Decayed facts:           {}", decayed_count);
        println!("  Contradictions resolved: {}", contradictions_resolved);
        for detail in &resolution_details {
            println!("    - {}", detail);
        }
        println!("  Neurons synthesized:     {}", neurons_merged);
        for detail in &neuron_details {
            println!("    - {}", detail);
        }
        println!("  Derived facts:           {}", derived.len());
        println!("  Total facts in DB:       {}", total_facts);

        if !yes {
            print!("\nApply these changes? [y/N]: ");
            std::io::stdout().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            if input.trim().to_lowercase() != "y" {
                println!("Dream cycle aborted.");
                return Ok(());
            }
        }
    }

    // Replace all facts with the computed final state
    delete_all_facts(&facts_table).await?;
    write_facts(&facts_table, &final_extracted).await?;
    write_facts(&facts_table, &derived).await?;
    write_facts(&facts_table, &neuron_facts).await?;

    let evicted_table = get_or_create_evicted_facts_table(&db).await?;
    write_facts(&evicted_table, &evicted).await?;

    // Write LDJSON briefing
    let path = crate::persistence::morning_pulse_path();
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);
    for line in &pulse {
        writeln!(writer, "{}", serde_json::to_string(line)?)?;
    }
    writer.flush()?;
    info!("Morning pulse written to {:?}", path);

    info!(
        "Dream complete: {} total facts, {} decayed, {} neurons merged, {} contradictions resolved",
        total_facts, decayed_count, neurons_merged, contradictions_resolved
    );

    Ok(())
}

/// Parse a date string and return days until the next annual occurrence.
fn parse_days_until(date_str: &str) -> Option<i64> {
    let today = Local::now().date_naive();
    let s = date_str.trim();
    let date = NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(s, "%d/%m/%Y"))
        .ok()?;

    // Next occurrence of this month/day (treat as annual)
    let this_year = NaiveDate::from_ymd_opt(today.year(), date.month(), date.day()).unwrap_or(date);
    let next = if this_year >= today {
        this_year
    } else {
        NaiveDate::from_ymd_opt(today.year() + 1, date.month(), date.day()).unwrap_or(this_year)
    };
    Some((next - today).num_days())
}
