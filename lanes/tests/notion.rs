use aurelius_lanes::{
    document::{SourceRecord, SourceRef},
    notion::project,
};
use chrono::DateTime;
use serde_json::{Value, json};

fn record(stream: &str, data: Value) -> SourceRecord {
    let id = format!("{stream}/{}", data["id"].as_str().unwrap());
    SourceRecord {
        organization_id: "fixture-org".into(),
        connection_id: "fixture-notion".into(),
        stream: stream.into(),
        record_id: id.clone(),
        extracted_at: DateTime::from_timestamp(1700000000, 0).unwrap(),
        deleted: false,
        data,
        source: SourceRef {
            stream: stream.into(),
            record_id: id,
            uri: "https://www.notion.so/page-a".into(),
            object_key: format!("raw/notion/1/{stream}/fixture.jsonl"),
            record_index: 3,
        },
    }
}
fn page() -> SourceRecord {
    record(
        "pages",
        json!({"id":"page-a","url":"https://www.notion.so/page-a","properties":{"Name":{"type":"title","title":[{"plain_text":"Delivery plan"}]},"Status":{"type":"status","status":{"name":"Proposed"}}}}),
    )
}
fn block(id: &str, parent: &str, parent_type: &str, sequence: u64, text: &str) -> SourceRecord {
    record(
        "blocks",
        json!({"id":id,"type":"paragraph","parent":{"type":parent_type,parent_type:parent,"sequence_number":sequence},"paragraph":{"rich_text":[{"plain_text":text}]}}),
    )
}
#[test]
fn page_projection_preserves_nested_order_and_exact_source_references() {
    let records = vec![
        block("block-z", "page-a", "page_id", 2, "Second paragraph"),
        page(),
        block(
            "block-a",
            "page-a",
            "page_id",
            1,
            "We must not ship unless approved.",
        ),
        block(
            "block-b",
            "block-a",
            "block_id",
            1,
            "Approval remains pending.",
        ),
    ];
    let output = project(&records).unwrap();
    assert_eq!(output.documents.len(), 1);
    assert!(output.deferred.is_empty());
    let document = &output.documents[0];
    assert_eq!(
        document.sections[1].text,
        "We must not ship unless approved."
    );
    assert_eq!(document.sections[2].text, "Approval remains pending.");
    assert_eq!(document.sections[3].text, "Second paragraph");
    assert_eq!(
        document.sections[1].source.object_key,
        "raw/notion/1/blocks/fixture.jsonl"
    );
    assert_eq!(document.sections[1].source.record_index, 3);
    assert_eq!(
        document.sections[1].source.uri,
        "https://www.notion.so/page-a#blocka"
    );
    let mut reverse = records;
    reverse.reverse();
    assert_eq!(
        document.content_hash().unwrap(),
        project(&reverse).unwrap().documents[0]
            .content_hash()
            .unwrap()
    );
}
#[test]
fn unsupported_attachments_orphans_and_cycles_are_explicitly_deferred() {
    let output = project(&[page(), record("blocks", json!({"id":"attachment-a","type":"file","parent":{"type":"page_id","page_id":"page-a"},"file":{"file":{"url":"https://example.invalid/private-file"}}})), block("orphan-a", "missing-page", "page_id", 1, "Unlinked"), block("cycle-a", "cycle-b", "block_id", 1, "Cycle"), block("cycle-b", "cycle-a", "block_id", 1, "Cycle")]).unwrap();
    assert_eq!(output.documents.len(), 1);
    assert_eq!(output.deferred.len(), 4);
    assert!(!output.documents[0].plaintext().contains("private-file"));
}
#[test]
fn archived_sources_are_not_indexed_and_child_page_stubs_do_not_collide() {
    let mut deleted = block("old-block", "page-a", "page_id", 1, "Removed text");
    deleted.data["in_trash"] = json!(true);
    let stub = record(
        "blocks",
        json!({"id":"page-a","type":"child_page","child_page":{"title":"Delivery plan"},"parent":{"type":"page_id","page_id":"another-page"}}),
    );
    let output = project(&[page(), stub, deleted]).unwrap();
    assert_eq!(output.documents.len(), 1);
    assert!(!output.documents[0].plaintext().contains("Removed text"));
}
#[test]
fn cross_tenant_records_and_credential_bearing_citations_are_rejected() {
    let mut other = block("block-a", "page-a", "page_id", 1, "Other scope");
    other.organization_id = "another-org".into();
    assert!(project(&[page(), other]).is_err());
    let mut unsafe_page = page();
    unsafe_page.data["url"] = json!("https://user:password@example.invalid/page");
    assert!(project(&[unsafe_page]).is_err());
}
#[test]
fn comments_remain_cited_and_users_are_not_free_standing_claims() {
    let output = project(&[page(), record("comments", json!({"id":"comment-a","parent":{"type":"page_id","page_id":"page-a"},"rich_text":[{"plain_text":"This is only a hypothesis."}]})), record("users", json!({"id":"user-a","name":"Synthetic User"}))]).unwrap();
    assert_eq!(output.documents.len(), 1);
    assert!(
        output.documents[0]
            .plaintext()
            .contains("This is only a hypothesis.")
    );
    assert!(!output.documents[0].plaintext().contains("Synthetic User"));
    assert_eq!(output.documents[0].sections[1].source.stream, "comments");
}
