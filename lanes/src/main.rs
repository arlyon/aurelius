use anyhow::{Context, Result, ensure};
use aurelius_lanes::{
    airbyte::Airbyte,
    config::{Config, Mode},
    processor::BundledEmbedder,
    store::{MIGRATIONS, Store, fingerprint},
    worker::Worker,
};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::postgres::PgPoolOptions;
use std::{sync::Arc, time::Duration};
use subtle::ConstantTimeEq;

#[derive(Clone)]
struct App {
    worker: Worker,
    operator_token: String,
    webhook_token: String,
    webhook_org: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Search {
    organization_id: String,
    query: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reconcile {
    organization_id: String,
}
fn same_secret(a: &str, b: &str) -> bool {
    Sha256::digest(a.as_bytes())
        .ct_eq(&Sha256::digest(b.as_bytes()))
        .into()
}
fn authorized(headers: &HeaderMap, token: &str) -> bool {
    headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|v| same_secret(v, token))
}
fn registered(app: &App, organization: &str) -> bool {
    app.worker
        .config
        .lanes
        .iter()
        .any(|l| l.enabled && l.organization_id == organization)
}
async fn health(State(app): State<Arc<App>>) -> StatusCode {
    if sqlx::query("SELECT 1")
        .execute(&app.worker.store.pool)
        .await
        .is_ok()
        && reqwest::Client::new()
            .get("http://127.0.0.1:8081/health")
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
    {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}
async fn webhook(
    State(app): State<Arc<App>>,
    Path(token): Path<String>,
    Json(_body): Json<Value>,
) -> StatusCode {
    if !same_secret(&token, &app.webhook_token) {
        return StatusCode::UNAUTHORIZED;
    }
    // Airbyte webhook formats vary by version. Treat it as a hint and resolve job IDs via its API.
    match app.worker.store.request_reconcile(&app.webhook_org).await {
        Ok(()) => StatusCode::ACCEPTED,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}
async fn reconcile(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(body): Json<Reconcile>,
) -> StatusCode {
    if !authorized(&headers, &app.operator_token) {
        return StatusCode::UNAUTHORIZED;
    }
    if !registered(&app, &body.organization_id) {
        return StatusCode::BAD_REQUEST;
    }
    match app
        .worker
        .store
        .request_reconcile(&body.organization_id)
        .await
    {
        Ok(()) => StatusCode::ACCEPTED,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}
async fn search(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(body): Json<Search>,
) -> std::result::Result<Json<Value>, StatusCode> {
    if !authorized(&headers, &app.operator_token) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    if !registered(&app, &body.organization_id)
        || body.query.trim().is_empty()
        || body.query.len() > 4000
    {
        return Err(StatusCode::BAD_REQUEST);
    }
    let lanes: Vec<_> = app
        .worker
        .config
        .lanes
        .iter()
        .filter(|l| {
            l.enabled && l.mode == Mode::Content && l.organization_id == body.organization_id
        })
        .collect();
    let fingerprints: Vec<String> = lanes
        .iter()
        .map(|l| fingerprint(l))
        .collect::<Result<_>>()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let lane_names = lanes.iter().map(|l| l.name.clone()).collect::<Vec<_>>();
    let vector = BundledEmbedder
        .query(&body.query)
        .await
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;
    let hits = app
        .worker
        .store
        .search(
            &body.organization_id,
            &lane_names,
            &fingerprints,
            &body.query,
            vector,
        )
        .await
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;
    Ok(Json(json!({"hits":hits})))
}
fn crate_limit() -> u64 {
    aurelius_lanes::processor::MAX_OBJECT_BYTES as u64 + 1
}
async fn shutdown() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("register SIGTERM");
    tokio::select! {_=tokio::signal::ctrl_c()=>{},_=terminate.recv()=>{}}
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "aurelius_lanes=info".into()),
        )
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("serve");
    if mode == "validate" {
        let path = args.get(1).context("usage: validate <config> [lane]")?;
        let config = Config::load(std::path::Path::new(path))?;
        if let Some(name) = args.get(2) {
            use std::io::Read;
            let mut bytes = Vec::new();
            std::io::stdin()
                .take(crate_limit())
                .read_to_end(&mut bytes)?;
            let docs = aurelius_lanes::processor::parse_batch(&bytes, config.lane(name)?)?;
            println!("validated_records={}", docs.len());
        } else {
            println!("validated_lanes={}", config.lanes.len());
        }
        return Ok(());
    }
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .acquire_timeout(Duration::from_secs(10))
        .after_connect(|connection, _| {
            Box::pin(async move {
                sqlx::query("SET search_path = lake_ingestion, public")
                    .execute(connection)
                    .await?;
                Ok(())
            })
        })
        .connect(&std::env::var("DATABASE_URL")?)
        .await?;
    let version: i32 = sqlx::query_scalar("SELECT current_setting('server_version_num')::int")
        .fetch_one(&pool)
        .await?;
    ensure!(version >= 180000, "PG18 required; refusing legacy database");
    if mode == "migrate" {
        MIGRATIONS.run(&pool).await?;
        return Ok(());
    }
    let config = Config::load(std::path::Path::new(&std::env::var("LANES_CONFIG")?))?;
    let store = Store { pool };
    store.configure_lanes(&config).await?;
    if mode == "replay" {
        let lane = config.lane(args.get(1).context("usage: replay <lane> <sync-id>")?)?;
        let id = args.get(2).context("sync ID required")?.parse()?;
        store.enqueue(lane, id, true).await?;
        tracing::info!(lane=%lane.name,sync_id=id,"operator archive replay queued");
        return Ok(());
    }
    ensure!(
        matches!(mode, "serve" | "worker"),
        "expected serve, worker, migrate, or replay"
    );
    let worker = Worker::new(
        config,
        store,
        if mode == "serve" {
            Some(Airbyte::from_env()?)
        } else {
            None
        },
    )?;
    if mode == "worker" {
        loop {
            tokio::select! {_=shutdown()=>break,result=worker.index_next()=>{match result{Ok(true)=>{},Ok(false)=>tokio::time::sleep(Duration::from_secs(2)).await,Err(_)=>{tracing::error!(code="worker_failed");tokio::time::sleep(Duration::from_secs(5)).await;}}}}
        }
        return Ok(());
    }
    let operator_token = std::env::var("OPERATOR_TOKEN")?;
    let webhook_token = std::env::var("WEBHOOK_TOKEN")?;
    let webhook_org = std::env::var("WEBHOOK_ORGANIZATION")?;
    ensure!(
        operator_token.len() >= 32 && webhook_token.len() >= 32,
        "authentication secrets must be at least 32 characters"
    );
    let app = Arc::new(App {
        worker,
        operator_token,
        webhook_token,
        webhook_org,
    });
    ensure!(
        registered(&app, &app.webhook_org),
        "webhook organization must have a registered lane"
    );
    let dispatcher = Arc::clone(&app);
    let dispatch = tokio::spawn(async move {
        let mut daily = tokio::time::interval(Duration::from_secs(300));
        loop {
            tokio::select! {
                _=daily.tick()=>{if dispatcher.worker.request_daily().await.is_err(){tracing::error!(code="daily_reconcile_failed");}},
                _=tokio::time::sleep(Duration::from_secs(2))=>{
                    if dispatcher.worker.reconcile_next().await.is_err(){tracing::error!(code="reconcile_dispatch_failed");}
                    if dispatcher.worker.discover_next().await.is_err(){tracing::error!(code="manifest_dispatch_failed");}
                }
            }
        }
    });
    let indexer = app.worker.clone();
    let indexing = tokio::spawn(async move {
        loop {
            match indexer.index_next().await {
                Ok(true) => {}
                Ok(false) => tokio::time::sleep(Duration::from_secs(2)).await,
                Err(_) => {
                    tracing::error!(code = "index_dispatch_failed");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    });
    let router = Router::new()
        .route("/health", get(health))
        .route("/webhooks/airbyte/{token}", post(webhook))
        .route("/reconcile", post(reconcile))
        .route("/search", post(search))
        .layer(DefaultBodyLimit::max(16384))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8090").await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown())
        .await?;
    dispatch.abort();
    indexing.abort();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn secret_comparison() {
        assert!(same_secret("same", "same"));
        assert!(!same_secret("same", "different"));
    }
}
