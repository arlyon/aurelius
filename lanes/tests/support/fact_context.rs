use anyhow::{Context, Result, ensure};
use aurelius_lanes::{
    fact_context::{ContextEntity, ContextRequest, FactContextConfig},
    postgres::PostgresSink,
};
use chrono::{Duration, Utc};
use serde_json::json;
use sqlx::{PgPool, Postgres, QueryBuilder};

pub async fn exercise(sink: &PostgresSink, admin: &PgPool) -> Result<()> {
    sqlx::raw_sql(r#"CREATE TABLE "adapter-target"."canonical""facts" (
      id text PRIMARY KEY,organization_id text NOT NULL,scope text NOT NULL,owner_user_id text,
      subject_ref text NOT NULL,predicate text NOT NULL,object_ref text,object_value text,status text NOT NULL,
      evidence jsonb NOT NULL,valid_from timestamp,valid_to timestamp
    );"#).execute(admin).await?;
    let settings = FactContextConfig {
        schema: "adapter-target".into(),
        table: "canonical\"facts".into(),
        per_entity_limit: 16,
    };
    let effective_at = "2026-05-01T12:00:00Z".parse::<chrono::DateTime<Utc>>()?;
    let request = ContextRequest {
        organization_id: "tenant-a".into(),
        entities: vec![ContextEntity {
            id: "campaign:1".into(),
            name: "Mercury".into(),
        }],
        effective_at,
    };
    let denied = sink.fact_context(&settings, &request).await.unwrap_err();
    ensure!(
        denied.to_string().contains("42501"),
        "missing grant was not explicit"
    );
    sqlx::raw_sql(r#"GRANT SELECT ON "adapter-target"."canonical""facts" TO adapter_writer;"#)
        .execute(admin)
        .await?;
    let evidence = json!([{"kind":"app_database","sourceRef":"fixture:source","observedAt":"2026-05-01T00:00:00Z"}]);
    let fixtures = [
        (
            "included",
            "tenant-a",
            "org",
            "active",
            "campaign:1",
            evidence.clone(),
            None,
            None,
        ),
        (
            "other-tenant",
            "tenant-b",
            "org",
            "active",
            "campaign:1",
            evidence.clone(),
            None,
            None,
        ),
        (
            "private-user",
            "tenant-a",
            "user",
            "active",
            "campaign:1",
            evidence.clone(),
            None,
            None,
        ),
        (
            "candidate",
            "tenant-a",
            "org",
            "candidate",
            "campaign:1",
            evidence.clone(),
            None,
            None,
        ),
        (
            "closed",
            "tenant-a",
            "org",
            "closed",
            "campaign:1",
            evidence.clone(),
            None,
            None,
        ),
        (
            "contradicted",
            "tenant-a",
            "org",
            "contradicted",
            "campaign:1",
            evidence.clone(),
            None,
            None,
        ),
        (
            "superseded",
            "tenant-a",
            "org",
            "superseded",
            "campaign:1",
            evidence.clone(),
            None,
            None,
        ),
        (
            "expired",
            "tenant-a",
            "org",
            "active",
            "campaign:1",
            evidence.clone(),
            None,
            Some(effective_at),
        ),
        (
            "future",
            "tenant-a",
            "org",
            "active",
            "campaign:1",
            evidence.clone(),
            Some(effective_at + Duration::seconds(1)),
            None,
        ),
        (
            "starts-now",
            "tenant-a",
            "org",
            "active",
            "campaign:1",
            evidence.clone(),
            Some(effective_at),
            None,
        ),
        (
            "ends-later",
            "tenant-a",
            "org",
            "active",
            "campaign:1",
            evidence.clone(),
            None,
            Some(effective_at + Duration::days(1)),
        ),
        (
            "no-evidence",
            "tenant-a",
            "org",
            "active",
            "campaign:1",
            json!([]),
            None,
            None,
        ),
        (
            "malformed-evidence",
            "tenant-a",
            "org",
            "active",
            "campaign:1",
            json!({"kind":"app_database"}),
            None,
            None,
        ),
        (
            "agent-only",
            "tenant-a",
            "org",
            "active",
            "campaign:1",
            json!([{"kind":"agent_inference","sourceRef":"fixture:agent"}]),
            None,
            None,
        ),
        (
            "no-source-reference",
            "tenant-a",
            "org",
            "active",
            "campaign:1",
            json!([{"kind":"review_confirmation","sourceRef":""}]),
            None,
            None,
        ),
        (
            "another-subject",
            "tenant-a",
            "org",
            "active",
            "campaign:2",
            evidence.clone(),
            None,
            None,
        ),
    ];
    for (id, org, scope, status, subject, evidence, from, to) in fixtures {
        let mut query = QueryBuilder::<Postgres>::new(
            r#"INSERT INTO "adapter-target"."canonical""facts" (id,organization_id,scope,owner_user_id,status,subject_ref,predicate,object_value,evidence,valid_from,valid_to) VALUES($1,$2,$3,'fixture-user',$4,$5,'has_business_goal',$1,$6,$7,$8)"#,
        );
        query
            .build()
            .bind(id)
            .bind(org)
            .bind(scope)
            .bind(status)
            .bind(subject)
            .bind(evidence)
            .bind(from.map(|t| t.naive_utc()))
            .bind(to.map(|t| t.naive_utc()))
            .execute(admin)
            .await?;
    }
    let result = sink.fact_context(&settings, &request).await?;
    let facts = &result.entities[0].facts;
    let mut ids = facts.iter().map(|f| f.id.as_str()).collect::<Vec<_>>();
    ids.sort();
    ensure!(
        ids == ["ends-later", "included", "starts-now"],
        "org/status/privacy/time/evidence filters failed"
    );
    ensure!(!result.entities[0].has_more);
    let config_text = format!(
        "{}\n[postgres]\nurl_env='FACT_CONTEXT_FIXTURE_URL'\nschema='adapter-target'\nartifacts_table='source-docs'\nchunks_table='chunk-rows'\ntenant_setting='projection.scope'\n[fact_context]\nschema='adapter-target'\ntable='canonical\"facts'\nper_entity_limit=16\n[embedding]\nendpoint='http://127.0.0.1:9'\nmodel='fixture'\nrevision='fixture-v1'\nnative_dimensions=3\ndimensions=3\ntoken_limit=512\n[lanes.fixture]\nscope='tenant-a'\nconnection='fixture-connection'\nnormalizer='conversation'\nbucket='fixture'\nraw_prefix='fixture/raw/'\nstreams=['channel_messages']\ncitation_base='https://example.invalid'\n",
        include_str!("../fixtures/business.toml")
    );
    #[cfg(feature = "language")]
    exercise_preparation(
        sink,
        &config_text,
        effective_at,
        &result.canonical_prefix()?,
    )
    .await?;
    let directory = std::env::temp_dir().join(format!(
        "aurelius-fact-context-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&directory)?;
    let config_file = directory.join("config.toml");
    let request_file = directory.join("request.json");
    std::fs::write(&config_file, config_text)?;
    std::fs::write(&request_file, serde_json::to_vec(&request)?)?;
    let mut url = reqwest::Url::parse(&std::env::var("PG_ADAPTER_TEST_URL")?)?;
    url.set_username("adapter_writer")
        .map_err(|_| anyhow::anyhow!("fixture role URL"))?;
    url.set_password(Some("fixture-writer-password"))
        .map_err(|_| anyhow::anyhow!("fixture role URL"))?;
    let process = tokio::process::Command::new(env!("CARGO_BIN_EXE_aurelius-lanes"))
        .arg("fact-context")
        .arg(&config_file)
        .arg(&request_file)
        .env("FACT_CONTEXT_FIXTURE_URL", url.as_str())
        .output()
        .await?;
    std::fs::remove_file(config_file)?;
    std::fs::remove_file(request_file)?;
    std::fs::remove_dir(directory)?;
    ensure!(
        process.status.success(),
        "fact-context CLI failed: {}",
        String::from_utf8_lossy(&process.stderr)
    );
    ensure!(
        String::from_utf8(process.stdout)?.trim() == result.canonical_prefix()?,
        "CLI did not use the canonical reader/prefix"
    );
    let mut same = request.clone();
    same.effective_at += Duration::milliseconds(1);
    ensure!(
        result.canonical_prefix()?
            == sink
                .fact_context(&settings, &same)
                .await?
                .canonical_prefix()?,
        "unchanged context prefix includes a moving clock"
    );
    let capped = sink
        .fact_context(
            &FactContextConfig {
                per_entity_limit: 1,
                ..settings.clone()
            },
            &request,
        )
        .await?;
    ensure!(
        capped.entities[0].facts.len() == 1 && capped.entities[0].has_more,
        "bounded context hides incompleteness"
    );
    let mut batch = request.clone();
    batch.entities.push(ContextEntity {
        id: "campaign:2".into(),
        name: "Mercury".into(),
    });
    let prefix = sink
        .fact_context(&settings, &batch)
        .await?
        .canonical_prefix()?;
    batch.entities.reverse();
    ensure!(
        prefix
            == sink
                .fact_context(&settings, &batch)
                .await?
                .canonical_prefix()?,
        "entity order changed cache prefix"
    );
    let mut foreign = request.clone();
    foreign.organization_id = "tenant-b".into();
    let result = sink.fact_context(&settings, &foreign).await?;
    ensure!(
        result.entities[0].facts.len() == 1 && result.entities[0].facts[0].id == "other-tenant"
    );
    let mut injection = request.clone();
    injection.entities[0].id = "campaign:1') OR true --".into();
    ensure!(
        sink.fact_context(&settings, &injection).await?.entities[0]
            .facts
            .is_empty(),
        "subject value escaped SQL binding"
    );
    let insert_granted:bool=sqlx::query_scalar(r#"SELECT has_table_privilege('adapter_writer','"adapter-target"."canonical""facts"','INSERT')"#).fetch_one(admin).await?;
    ensure!(
        !insert_granted,
        "fact context read accidentally granted writes"
    );
    let mut no_entities = request.clone();
    no_entities.entities.clear();
    ensure!(
        sink.fact_context(&settings, &no_entities)
            .await?
            .entities
            .is_empty()
    );
    let mut duplicate = request.clone();
    duplicate.entities.push(duplicate.entities[0].clone());
    ensure!(sink.fact_context(&settings, &duplicate).await.is_err());
    let mut query = QueryBuilder::<Postgres>::new(
        r#"INSERT INTO "adapter-target"."canonical""facts"(id,organization_id,scope,subject_ref,predicate,object_value,status,evidence) VALUES('oversized','tenant-a','org','campaign:large','has_business_goal',$1,'active',$2)"#,
    );
    query
        .build()
        .bind("x".repeat(2002))
        .bind(evidence)
        .execute(admin)
        .await?;
    let mut oversized = request;
    oversized.entities[0].id = "campaign:large".into();
    let oversized_error = sink.fact_context(&settings, &oversized).await.unwrap_err();
    ensure!(
        oversized_error
            .downcast_ref::<aurelius_lanes::fact_context::ContextReview>()
            .is_some(),
        "oversized context was truncated or misclassified as a provider failure"
    );
    let _: i32 = sqlx::query_scalar("SELECT 1")
        .fetch_one(admin)
        .await
        .context("reader left transaction in an invalid state")?;
    Ok(())
}

#[cfg(feature = "language")]
async fn exercise_preparation(
    sink: &PostgresSink,
    text: &str,
    effective_at: chrono::DateTime<Utc>,
    expected_prefix: &str,
) -> Result<()> {
    use aurelius_lanes::{
        config::{ChatConfig, Config},
        contracts::{Document, DocumentKey},
        fact_preparation::FactPreparer,
        language::LanguageConfig,
        ontology::EntityType,
        references::ReferenceConfig,
    };
    use std::collections::BTreeMap;
    let mut config = Config::parse(text)?;
    config.ontology.entities.insert(
        "campaign".into(),
        EntityType {
            description: "An advertising campaign".into(),
        },
    );
    config.language = Some(LanguageConfig {
        model_dir: std::env::var("AURELIUS_TEST_LANGUAGE_MODEL_DIR")
            .context("reviewed local neural assets required")?,
        max_document_tokens: 2048,
        threads: 2,
    });
    config.chat = Some(ChatConfig {
        endpoint: "http://127.0.0.1:9/chat".into(),
        model: "no-inference-fixture".into(),
        revision: "fixture".into(),
        api_key_env: None,
        gateway: None,
        api: Default::default(),
        references: ReferenceConfig {
            verified_aliases: BTreeMap::from([(
                "tenant-a".into(),
                BTreeMap::from([("Mercury".into(), vec!["campaign:1".into()])]),
            )]),
        },
    });
    let preparer = FactPreparer::with_sink(&config, sink.clone())?;
    let document = Document {
        key: DocumentKey {
            scope: "tenant-a".into(),
            source: "fixture".into(),
            revision: "fixture-revision".into(),
        },
        text: "Mercury is a campaign. It has a monthly budget.".into(),
        context: None,
        metadata: Default::default(),
    };
    let prepared = preparer.prepare(document.clone(), effective_at).await?;
    prepared.language.validate(&document)?;
    ensure!(
        prepared
            .language
            .mentions
            .iter()
            .any(|mention| mention.learned),
        "preparation bypassed neural mention analysis"
    );
    ensure!(
        prepared.canonical_prefix()? == expected_prefix,
        "preparation bypassed scoped canonical retrieval"
    );
    ensure!(
        prepared.alias_hypotheses.len() == 1 && prepared.alias_hypotheses[0].alias == "Mercury",
        "coreference automatically merged a pronoun into a canonical identity"
    );
    let again = preparer
        .prepare(document.clone(), effective_at + Duration::milliseconds(1))
        .await?;
    ensure!(
        again.canonical_prefix()? == prepared.canonical_prefix()?,
        "source clock changed the shared prefix"
    );
    let mut other_tenant = document;
    other_tenant.key.scope = "tenant-b".into();
    ensure!(
        preparer.prepare(other_tenant, effective_at).await.is_err(),
        "preparation crossed configured scope"
    );
    Ok(())
}
