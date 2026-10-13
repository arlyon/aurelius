use aurelius_lanes::{
    config::{Config, Lane},
    processor::{DIMENSIONS, Document, SourceMetadata},
    store::{ObjectTask, Store},
};
use chrono::{Duration, Utc};
use sqlx::{Row, postgres::PgPoolOptions};
use swiftide_core::indexing::{EmbeddedField, TextNode};
use uuid::Uuid;

fn lane() -> Lane {
    serde_json::from_value(serde_json::json!({"name":"slack-demo","organization_id":"org_demo","connection_id":"11111111-1111-4111-8111-111111111111","bucket":"demo","root":"orgs/org_demo/raw/","namespace":"slack","stream":"channel_messages","id_pointers":["/channel_id","/ts"],"text_fields":["/text"],"revision":{"pointer":"/_airbyte_extracted_at","envelope":true,"unit":"milliseconds"},"scope":{"kind":"connection"},"citation":{"kind":"slack","workspace_url":"https://demo.slack.com","channel_pointer":"/channel_id","timestamp_pointer":"/ts"},"mime_type":"text/plain","mode":"content","enabled":true})).unwrap()
}
fn doc(text: &str, n: i64, deleted: bool) -> Document {
    let modified_at = Utc::now() + Duration::seconds(n);
    Document {
        id: "source-1".into(),
        text: text.into(),
        revision: modified_at.to_rfc3339(),
        modified_at,
        source_uri: "https://demo.slack.com/archives/C1/p123456".into(),
        record_index: 0,
        scope_value: "connection".into(),
        deleted,
        metadata: SourceMetadata {
            source_type: "slack".into(),
            author_id: Some("U1".into()),
            author_is_bot: Some(false),
            channel_id: Some("C1".into()),
            thread_ts: Some("123.456".into()),
            source_modified_at: Some(chrono::DateTime::from_timestamp(1700000000, 0).unwrap()),
        },
    }
}
fn nodes() -> Vec<TextNode> {
    (0..3)
        .map(|i| {
            let mut node = TextNode::new(format!("chunk {i}"));
            node.vectors
                .get_or_insert_with(Default::default)
                .insert(EmbeddedField::Chunk, vec![1.0; DIMENSIONS]);
            node
        })
        .collect()
}

#[tokio::test]
async fn pg18_activation_cursor_replay_reversion_revocation_and_rls() -> anyhow::Result<()> {
    let url = std::env::var("TEST_DATABASE_URL")
        .expect("PG18 integration requires TEST_DATABASE_URL; no silent skips");
    let parsed = reqwest::Url::parse(&url)?;
    assert!(
        matches!(parsed.host_str(), Some("localhost" | "127.0.0.1")),
        "integration fixtures require an isolated loopback database"
    );
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .after_connect(|conn, _| {
            Box::pin(async move {
                sqlx::query("SET search_path=lake_ingestion,public")
                    .execute(conn)
                    .await?;
                Ok(())
            })
        })
        .connect(&url)
        .await?;
    let version: i32 = sqlx::query_scalar("SELECT current_setting('server_version_num')::int")
        .fetch_one(&pool)
        .await?;
    assert!(
        version >= 180000,
        "test database must match production PG18"
    );
    sqlx::raw_sql(include_str!("bootstrap.sql"))
        .execute(&pool)
        .await?;
    sqlx::raw_sql(include_str!("../migrations/0002_review_feed.sql"))
        .execute(&pool)
        .await?;
    let store = Store { pool: pool.clone() };
    let mut lane = lane();
    store
        .configure_lanes(&Config {
            lanes: vec![lane.clone()],
        })
        .await?;
    let job: i64 = sqlx::query_scalar(
        "INSERT INTO ingestion_jobs(lane,status) VALUES($1,'running') RETURNING id",
    )
    .bind(&lane.name)
    .fetch_one(&pool)
    .await?;
    let lease = Uuid::new_v4();
    sqlx::query("INSERT INTO ingestion_objects VALUES($1,'key','etag',$2,'running',now()+interval '5 minutes')").bind(job).bind(lease).execute(&pool).await?;
    let task = ObjectTask {
        job_id: job,
        lane: lane.name.clone(),
        key: "key".into(),
        etag: "etag".into(),
        version: None,
        size: 0,
        modified_at: Utc::now(),
        lease,
    };
    let first = doc("first", 0, false);
    assert!(store.observe(&lane, &task, &first).await?);
    store.activate(&lane, &task, &first, nodes()).await?;
    assert!(!store.observe(&lane, &task, &first).await?);
    store.activate(&lane, &task, &first, nodes()).await?;
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT count(*) FROM document_changes")
            .fetch_one(&pool)
            .await?,
        1
    );
    let mut stale = doc("stale upstream snapshot", 99, false);
    stale.metadata.source_modified_at =
        Some(chrono::DateTime::from_timestamp(1600000000, 0).unwrap());
    assert!(!store.observe(&lane, &task, &stale).await?);
    let edited = doc("edited", 1, false);
    assert!(store.observe(&lane, &task, &edited).await?);
    store.activate(&lane, &task, &edited, nodes()).await?;
    let deleted = doc("", 2, true);
    assert!(store.observe(&lane, &task, &deleted).await?);
    store.activate(&lane, &task, &deleted, vec![]).await?;
    let restored = doc("first", 3, false);
    assert_eq!(restored.hash(), first.hash());
    assert!(store.observe(&lane, &task, &restored).await?);
    store.activate(&lane, &task, &restored, nodes()).await?;
    lane.enabled = false;
    store
        .configure_lanes(&Config {
            lanes: vec![lane.clone()],
        })
        .await?;
    store
        .configure_lanes(&Config {
            lanes: vec![lane.clone()],
        })
        .await?;
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT count(*) FROM document_changes")
            .fetch_one(&pool)
            .await?,
        5
    );
    assert!(
        sqlx::query_scalar::<_, bool>("SELECT desired_deleted FROM lake_documents")
            .fetch_one(&pool)
            .await?
    );
    lane.enabled = true;
    store
        .configure_lanes(&Config {
            lanes: vec![lane.clone()],
        })
        .await?;
    assert!(store.observe(&lane, &task, &restored).await?);
    store.activate(&lane, &task, &restored, nodes()).await?;
    let events = sqlx::query(
        "SELECT cursor,deleted,source_modified_at,indexed_at FROM document_changes ORDER BY cursor",
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(events.len(), 6);
    assert!(
        events
            .windows(2)
            .all(|pair| pair[0].get::<i64, _>("cursor") < pair[1].get::<i64, _>("cursor"))
    );
    assert!(events[2].get::<bool, _>("deleted"));
    assert!(events[4].get::<bool, _>("deleted"));
    assert!(!events[5].get::<bool, _>("deleted"));
    assert!(
        events[0].get::<chrono::DateTime<Utc>, _>("source_modified_at")
            < events[0].get::<chrono::DateTime<Utc>, _>("indexed_at")
    );
    let mut tx = pool.begin().await?;
    sqlx::query("SET LOCAL ROLE lake_runtime")
        .execute(&mut *tx)
        .await?;
    sqlx::query("SELECT set_config('lake.organization_id','other_org',true)")
        .execute(&mut *tx)
        .await?;
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT count(*) FROM document_changes")
            .fetch_one(&mut *tx)
            .await?,
        0
    );
    tx.rollback().await?;
    // A tenant lock blocks a later writer before it can allocate a cursor.
    let mut earlier = pool.begin().await?;
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended('org_demo',0))")
        .execute(&mut *earlier)
        .await?;
    let mut later = pool.begin().await?;
    assert!(
        !sqlx::query_scalar::<_, bool>(
            "SELECT pg_try_advisory_xact_lock(hashtextextended('org_demo',0))"
        )
        .fetch_one(&mut *later)
        .await?
    );
    earlier.rollback().await?;
    later.rollback().await?;
    pool.close().await;
    Ok(())
}
