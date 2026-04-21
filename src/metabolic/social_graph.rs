#![allow(clippy::clone_on_copy)]

use ascent::ascent;
use chrono::Utc;

use crate::metabolic::facts::Fact;

// Trust scores are stored as milliunits (0–1000) so they satisfy Hash + Eq.
// 1000 milliunits == 1.0 trust.
ascent! {
    struct SocialGraph;

    relation spouse_of(String, String);
    relation parent_of(String, String);
    relation child_of(String, String);
    // (truster, trusted, trust_milliunits)
    relation social_trust(String, String, i32);
    // (person, event_type, days_since) — pre-computed before running
    relation days_since_event(String, String, i64);
    relation hold_communications(String);
    // (invoice_id, client)
    relation invoice(String, String);
    // (invoice_id, client)
    relation overdue(String, String);
    // (action_type, client, invoice_id)
    relation blocked_action(String, String, String);

    // Symmetry: spouse relationship is bidirectional
    spouse_of(b.clone(), a.clone()) <-- spouse_of(a, b);

    // Symmetry: child_of is the inverse of parent_of
    child_of(child.clone(), parent.clone()) <-- parent_of(parent, child);

    // Transitive trust: A trusts C if A trusts B and B trusts C (decayed by 0.6)
    // milliunits: (s1/1000) * (s2/1000) * 0.6 * 1000 = s1 * s2 * 6 / 10_000
    social_trust(a.clone(), c.clone(), *s1 * *s2 * 6 / 10_000) <--
        social_trust(a, b, s1),
        social_trust(b, c, s2),
        if a != c;

    // Hold communications within 30 days of a life event
    hold_communications(person.clone()) <--
        days_since_event(person, _, days),
        if *days < 30;

    // Block invoice reminders for clients under communication hold
    blocked_action(
        "send_invoice_reminder".to_string(),
        client.clone(),
        id.clone()
    ) <--
        invoice(id, client),
        overdue(id, client),
        hold_communications(client);
}

pub struct SocialGraphResults {
    pub hold_communications: Vec<String>,
    pub blocked_actions: Vec<(String, String, String)>,
    pub social_trust: Vec<(String, String, i32)>,
    pub spouse_of: Vec<(String, String)>,
    pub child_of: Vec<(String, String)>,
}

pub fn run_social_graph(facts: &[Fact]) -> SocialGraphResults {
    let mut prog = SocialGraph::default();
    let now_micros = Utc::now().timestamp_micros();

    for fact in facts {
        match fact.predicate.as_str() {
            "spouse_of" => {
                prog.spouse_of
                    .push((fact.subject.clone(), fact.object.clone()));
            }
            "parent_of" => {
                prog.parent_of
                    .push((fact.subject.clone(), fact.object.clone()));
            }
            "social_trust" => {
                let milliunits = (fact.confidence * 1000.0) as i32;
                prog.social_trust
                    .push((fact.subject.clone(), fact.object.clone(), milliunits));
            }
            "life_event" => {
                let days = (now_micros - fact.created_at) / (86_400 * 1_000_000);
                prog.days_since_event
                    .push((fact.subject.clone(), fact.predicate.clone(), days));
            }
            "invoice_due" => {
                prog.invoice.push((fact.id.clone(), fact.subject.clone()));
            }
            _ => {}
        }
    }

    prog.run();

    SocialGraphResults {
        hold_communications: prog.hold_communications.into_iter().map(|(p,)| p).collect(),
        blocked_actions: prog.blocked_action.into_iter().collect(),
        social_trust: prog.social_trust.into_iter().collect(),
        spouse_of: prog.spouse_of.into_iter().collect(),
        child_of: prog.child_of.into_iter().collect(),
    }
}
