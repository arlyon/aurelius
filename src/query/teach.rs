use anyhow::Result;
use lancedb::connect;
use std::collections::HashMap;
use std::io::{self, Write};
use swiftide::traits::ChatCompletion;

use crate::metabolic::facts::{
    Fact, delete_facts_by_ids, get_or_create_facts_table, load_all_facts, write_facts, Extractor,
    is_functional_predicate,
};

pub async fn run_teach(
    limit: usize,
    prompt: Option<String>,
    completion: &dyn ChatCompletion,
) -> Result<()> {
    let db = connect(&crate::persistence::db_path()).execute().await?;
    let table = get_or_create_facts_table(&db).await?;

    if let Some(prompt_text) = prompt {
        return handle_adhoc_teach(&table, &prompt_text, completion).await;
    }

    let facts = load_all_facts(&table).await?;

    // Group non-core facts by (subject, predicate)
    let mut by_key: HashMap<(String, String), Vec<usize>> = HashMap::new();
    for (i, fact) in facts.iter().enumerate() {
        if !fact.is_core_truth {
            by_key
                .entry((fact.subject.clone(), fact.predicate.clone()))
                .or_default()
                .push(i);
        }
    }

    // Collect contradiction pairs (same key, different object)
    let mut pairs: Vec<(&Fact, &Fact)> = Vec::new();
    for indices in by_key.values() {
        if indices.is_empty() {
            continue;
        }

        let predicate = &facts[indices[0]].predicate;
        if !is_functional_predicate(predicate) {
            continue;
        }

        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let a = &facts[indices[i]];
                let b = &facts[indices[j]];
                if a.object != b.object {
                    pairs.push((a, b));
                }
            }
        }
    }

    // Sort by highest combined confidence first
    pairs.sort_by(|(a1, b1), (a2, b2)| {
        let s1 = a1.confidence + b1.confidence;
        let s2 = a2.confidence + b2.confidence;
        s2.partial_cmp(&s1).unwrap_or(std::cmp::Ordering::Equal)
    });

    pairs.truncate(limit);

    if pairs.is_empty() {
        println!("No contradictions found.");
        return Ok(());
    }

    let total = pairs.len();
    let mut resolved = 0;
    let mut skipped = 0;

    for (n, (fact_a, fact_b)) in pairs.iter().enumerate() {
        println!("\nContradiction #{}/{}:", n + 1, total);
        println!(
            "  A: {} {} {}  (confidence: {:.2})",
            fact_a.subject, fact_a.predicate, fact_a.object, fact_a.confidence
        );
        println!(
            "  B: {} {} {}  (confidence: {:.2})",
            fact_b.subject, fact_b.predicate, fact_b.object, fact_b.confidence
        );

        loop {
            print!("Which is correct? [a/b/both/neither/skip]: ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let choice = input.trim().to_lowercase();

            match choice.as_str() {
                "a" => {
                    let mut winner = (*fact_a).clone();
                    winner.confidence = 1.0;
                    winner.is_core_truth = true;
                    delete_facts_by_ids(&table, &[fact_a.id.clone(), fact_b.id.clone()]).await?;
                    write_facts(&table, &[winner]).await?;
                    resolved += 1;
                    break;
                }
                "b" => {
                    let mut winner = (*fact_b).clone();
                    winner.confidence = 1.0;
                    winner.is_core_truth = true;
                    delete_facts_by_ids(&table, &[fact_a.id.clone(), fact_b.id.clone()]).await?;
                    write_facts(&table, &[winner]).await?;
                    resolved += 1;
                    break;
                }
                "both" => {
                    let mut a = (*fact_a).clone();
                    let mut b = (*fact_b).clone();
                    a.is_core_truth = true;
                    b.is_core_truth = true;
                    delete_facts_by_ids(&table, &[fact_a.id.clone(), fact_b.id.clone()]).await?;
                    write_facts(&table, &[a, b]).await?;
                    resolved += 1;
                    break;
                }
                "neither" => {
                    delete_facts_by_ids(&table, &[fact_a.id.clone(), fact_b.id.clone()]).await?;
                    resolved += 1;
                    break;
                }
                "skip" | "" => {
                    skipped += 1;
                    break;
                }
                _ => {
                    println!("Please enter a, b, both, neither, or skip.");
                }
            }
        }
    }

    println!("\nSummary: {} resolved, {} skipped.", resolved, skipped);
    Ok(())
}

async fn handle_adhoc_teach(
    table: &lancedb::table::Table,
    prompt: &str,
    completion: &dyn ChatCompletion,
) -> Result<()> {
    println!("Extracting facts from prompt...");
    let extractor = Extractor { completion };
    let facts = extractor.extract_facts(prompt, "adhoc").await?;

    if facts.is_empty() {
        println!("No facts extracted from the prompt.");
        return Ok(());
    }

    println!("Extracted {} facts:", facts.len());
    for (i, fact) in facts.iter().enumerate() {
        println!(
            "  {}. {} {} {} (confidence: {:.2})",
            i + 1,
            fact.subject,
            fact.predicate,
            fact.object,
            fact.confidence
        );
    }

    print!("\nSave these facts? [y/N]: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    if input.trim().to_lowercase() == "y" {
        write_facts(table, &facts).await?;
        println!("Saved {} facts.", facts.len());
    } else {
        println!("Discarded facts.");
    }

    Ok(())
}
