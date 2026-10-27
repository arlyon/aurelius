use anyhow::{Context, Result, ensure};
use aurelius_lanes::{config::Config, storage};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("config path required")?;
    let config = Config::load(Path::new(&path))?;
    let lane = config.lanes.values().next().context("lane required")?;
    let store = config
        .storage
        .context("storage required")?
        .connect(&lane.bucket)?;
    let key = format!(
        "{}validation/fencing-{}.json",
        lane.output_prefix,
        chrono::Utc::now().timestamp_millis()
    );
    let first = storage::publish(store.as_ref(), &key, b"{\"version\":1}".to_vec(), None).await?;
    ensure!(
        storage::read(store.as_ref(), &first, 100).await? == b"{\"version\":1}",
        "initial read mismatch"
    );
    ensure!(
        storage::publish(store.as_ref(), &key, b"conflict".to_vec(), None)
            .await
            .is_err(),
        "duplicate create was accepted"
    );
    let second =
        storage::overwrite_exclusive(store.as_ref(), &key, b"{\"version\":2}".to_vec()).await?;
    ensure!(
        storage::read(store.as_ref(), &first, 100).await.is_err(),
        "stale read was accepted"
    );
    ensure!(
        storage::read(store.as_ref(), &second, 100).await? == b"{\"version\":2}",
        "updated read mismatch"
    );
    println!("conditional_create=true exclusive_overwrite=true stale_read_rejected=true");
    Ok(())
}
