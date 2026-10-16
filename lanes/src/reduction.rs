use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReductionConfig {
    pub executable: String,
    pub schema: String,
    pub collection: String,
}
impl ReductionConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.executable.is_empty() && !self.schema.is_empty() && !self.collection.is_empty(),
            "reducer configuration required"
        );
        ensure!(
            [&self.executable, &self.schema, &self.collection]
                .iter()
                .all(|s| s.len() <= 1000 && !s.chars().any(char::is_control)),
            "invalid reducer configuration"
        );
        Ok(())
    }
    pub async fn combine(&self, input: &[serde_json::Value]) -> Result<Vec<serde_json::Value>> {
        self.validate()?;
        let bytes = input
            .iter()
            .map(serde_json::to_string)
            .collect::<std::result::Result<Vec<_>, _>>()?
            .join("\n")
            + "\n";
        ensure!(
            bytes.len() <= 128 * 1024 * 1024,
            "reduction input exceeds bound"
        );
        let mut child = tokio::process::Command::new(&self.executable)
            .args([
                "raw",
                "combine",
                "--source",
                &self.schema,
                "--collection",
                &self.collection,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        let mut stdin = child.stdin.take().context("reducer stdin unavailable")?;
        let stdout = child.stdout.take().context("reducer stdout unavailable")?;
        let write = async move {
            stdin.write_all(bytes.as_bytes()).await?;
            stdin.shutdown().await
        };
        let read = async {
            let mut bytes = Vec::new();
            stdout
                .take(128 * 1024 * 1024 + 1)
                .read_to_end(&mut bytes)
                .await?;
            Ok::<_, std::io::Error>(bytes)
        };
        let operation = async {
            let (written, output) = tokio::join!(write, read);
            written?;
            let output = output?;
            ensure!(
                output.len() <= 128 * 1024 * 1024,
                "reduction output exceeds bound"
            );
            ensure!(child.wait().await?.success(), "native reduction failed");
            std::str::from_utf8(&output)?
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| serde_json::from_str(line).map_err(Into::into))
                .collect::<Result<Vec<_>>>()
        };
        tokio::time::timeout(std::time::Duration::from_secs(180), operation).await?
    }
}
