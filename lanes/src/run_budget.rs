//! Durable reserve-before-send accounting for explicitly bounded extraction runs.
use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

const MAX_LEDGER_BYTES: u64 = 32 * 1024 * 1024;
const INPUT_NANO_USD: u64 = 100;
const OUTPUT_NANO_USD: u64 = 500;
#[derive(Debug)]
pub struct RunStopped;
impl std::fmt::Display for RunStopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("bounded extraction run stopped")
    }
}
impl std::error::Error for RunStopped {}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Header {
    version: u32,
    identity: String,
    max_nano_usd: u64,
    max_calls: u64,
    duration_seconds: u64,
    deadline: DateTime<Utc>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Entry {
    Header {
        limits: Header,
    },
    Reserved {
        id: u64,
        nano_usd: u64,
    },
    Settled {
        id: u64,
        nano_usd: u64,
        input_tokens: u64,
        output_tokens: u64,
    },
}
#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    pub calls: u64,
    pub charged_nano_usd: u64,
    pub max_nano_usd: u64,
    pub deadline: DateTime<Utc>,
    pub halted: bool,
}
struct State {
    file: File,
    lock: PathBuf,
    header: Header,
    calls: u64,
    charged: u64,
    pending: BTreeMap<u64, u64>,
    halted: bool,
}
impl Drop for State {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.lock);
    }
}
impl State {
    fn append(&mut self, entry: &Entry) -> Result<()> {
        if self.file.metadata()?.len() >= MAX_LEDGER_BYTES {
            self.halted = true;
            anyhow::bail!("budget ledger exceeds bound");
        }
        let mut bytes = serde_json::to_vec(entry)?;
        bytes.push(b'\n');
        if self
            .file
            .write_all(&bytes)
            .and_then(|()| self.file.sync_all())
            .is_err()
        {
            self.halted = true;
            anyhow::bail!("cannot persist budget accounting");
        }
        Ok(())
    }
    fn check(&self, now: DateTime<Utc>) -> Result<()> {
        if self.halted
            || now >= self.header.deadline
            || self.calls >= self.header.max_calls
            || self.charged >= self.header.max_nano_usd
        {
            return Err(RunStopped.into());
        }
        Ok(())
    }
}
#[derive(Clone)]
pub struct RunBudget(Arc<Mutex<State>>);
impl RunBudget {
    pub fn open(
        path: &Path,
        identity: String,
        max_cents: u64,
        max_calls: u64,
        duration: Duration,
    ) -> Result<Self> {
        Self::open_at(path, identity, max_cents, max_calls, duration, Utc::now())
    }
    fn open_at(
        path: &Path,
        identity: String,
        max_cents: u64,
        max_calls: u64,
        duration: Duration,
        now: DateTime<Utc>,
    ) -> Result<Self> {
        ensure!(
            !identity.is_empty()
                && identity.len() <= 512
                && max_cents > 0
                && max_calls > 0
                && (1..=259200).contains(&duration.as_secs()),
            "invalid bounded run limits"
        );
        let maximum = max_cents
            .checked_mul(10_000_000)
            .context("budget overflow")?;
        let lock = path.with_extension("lock");
        let lock_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&lock)
            .context(
                "budget locked; a crashed run requires operator inspection, never automatic unlock",
            )?;
        lock_file.sync_all()?;
        let opened = (|| -> Result<Self> {
            let mut file = OpenOptions::new()
                .read(true)
                .append(true)
                .create(true)
                .mode(0o600)
                .open(path)?;
            ensure!(
                file.metadata()?.len() <= MAX_LEDGER_BYTES,
                "budget ledger exceeds bound"
            );
            let mut text = String::new();
            file.read_to_string(&mut text)?;
            let mut entries = text.lines().map(serde_json::from_str::<Entry>);
            let header = if text.is_empty() {
                Header {
                    version: 1,
                    identity: identity.clone(),
                    max_nano_usd: maximum,
                    max_calls,
                    duration_seconds: duration.as_secs(),
                    deadline: now + chrono::Duration::from_std(duration)?,
                }
            } else {
                ensure!(
                    text.ends_with('\n'),
                    "incomplete budget ledger; manual inspection required"
                );
                match entries.next().context("missing budget header")?? {
                    Entry::Header { limits } => limits,
                    _ => anyhow::bail!("invalid budget header"),
                }
            };
            ensure!(
                header.version == 1
                    && header.identity == identity
                    && header.max_nano_usd == maximum
                    && header.max_calls == max_calls
                    && header.duration_seconds == duration.as_secs(),
                "run identity or limits changed"
            );
            let mut state = State {
                file,
                lock: lock.clone(),
                header,
                calls: 0,
                charged: 0,
                pending: BTreeMap::new(),
                halted: false,
            };
            if text.is_empty() {
                state.append(&Entry::Header {
                    limits: state.header.clone(),
                })?;
                File::open(path.parent().context("budget directory required")?)?.sync_all()?;
            }
            for entry in entries {
                match entry? {
                    Entry::Reserved { id, nano_usd } => {
                        ensure!(id == state.calls + 1, "invalid request sequence");
                        state.calls = id;
                        state.charged = state
                            .charged
                            .checked_add(nano_usd)
                            .context("budget overflow")?;
                        state.pending.insert(id, nano_usd);
                    }
                    Entry::Settled {
                        id,
                        nano_usd,
                        input_tokens,
                        output_tokens,
                    } => {
                        let reserved = state
                            .pending
                            .remove(&id)
                            .context("unknown or repeated settlement")?;
                        ensure!(
                            nano_usd == cost(input_tokens, output_tokens)? && nano_usd <= reserved,
                            "invalid cost settlement"
                        );
                        state.charged = state.charged - reserved + nano_usd;
                    }
                    Entry::Header { .. } => anyhow::bail!("repeated budget header"),
                }
            }
            ensure!(
                state.calls <= max_calls && state.charged <= maximum,
                "ledger exceeds approved limits"
            );
            // Uncertain requests retain their full reservation and prohibit automatic resume.
            state.halted = !state.pending.is_empty();
            Ok(Self(Arc::new(Mutex::new(state))))
        })();
        if opened.is_err() {
            let _ = std::fs::remove_file(lock);
        }
        opened
    }
    pub fn check(&self) -> Result<()> {
        self.0
            .lock()
            .map_err(|_| anyhow::anyhow!("budget lock poisoned"))?
            .check(Utc::now())
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        let s = self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("budget lock poisoned"))?;
        Ok(Snapshot {
            calls: s.calls,
            charged_nano_usd: s.charged,
            max_nano_usd: s.header.max_nano_usd,
            deadline: s.header.deadline,
            halted: s.halted,
        })
    }
    pub fn reserve(&self, request_bytes: usize, max_output_tokens: u32) -> Result<Permit> {
        self.reserve_at(request_bytes, max_output_tokens, Utc::now())
    }
    fn reserve_at(
        &self,
        request_bytes: usize,
        max_output_tokens: u32,
        now: DateTime<Utc>,
    ) -> Result<Permit> {
        // Byte-level input upper bound plus control-token headroom; cached input is full rate.
        let input = u64::try_from(request_bytes)?
            .checked_add(8192)
            .context("input bound overflow")?;
        let amount = cost(input, u64::from(max_output_tokens))?;
        let mut s = self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("budget lock poisoned"))?;
        s.check(now)?;
        if amount > s.header.max_nano_usd - s.charged {
            return Err(RunStopped.into());
        }
        let id = s.calls + 1;
        s.append(&Entry::Reserved {
            id,
            nano_usd: amount,
        })?;
        s.calls = id;
        s.charged += amount;
        s.pending.insert(id, amount);
        Ok(Permit {
            budget: self.clone(),
            id,
            settled: false,
        })
    }
}
fn cost(input: u64, output: u64) -> Result<u64> {
    input
        .checked_mul(INPUT_NANO_USD)
        .and_then(|n| {
            output
                .checked_mul(OUTPUT_NANO_USD)
                .and_then(|o| n.checked_add(o))
        })
        .context("token cost overflow")
}
pub struct Permit {
    budget: RunBudget,
    id: u64,
    settled: bool,
}
impl Permit {
    pub fn settle(mut self, input: u64, output: u64) -> Result<()> {
        let amount = cost(input, output)?;
        let mut s = self
            .budget
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("budget lock poisoned"))?;
        let reserved = *s
            .pending
            .get(&self.id)
            .context("request reservation missing")?;
        if amount > reserved {
            s.halted = true;
            anyhow::bail!("provider usage exceeded conservative reservation; run halted");
        }
        s.append(&Entry::Settled {
            id: self.id,
            nano_usd: amount,
            input_tokens: input,
            output_tokens: output,
        })?;
        s.pending.remove(&self.id);
        s.charged = s.charged - reserved + amount;
        self.settled = true;
        Ok(())
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        if !self.settled
            && let Ok(mut s) = self.budget.0.lock()
        {
            s.halted = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "budget-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> PathBuf {
            self.0.join("ledger.jsonl")
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn durable_limits_and_cost_settlement() -> Result<()> {
        let t = Temp::new();
        let now = Utc::now();
        let duration = Duration::from_secs(60);
        let b = RunBudget::open_at(&t.path(), "run".into(), 1, 2, duration, now)?;
        assert!(RunBudget::open_at(&t.path(), "run".into(), 1, 2, duration, now).is_err());
        b.reserve_at(100, 100, now)?.settle(10, 10)?;
        assert_eq!(b.snapshot()?.charged_nano_usd, 6000);
        b.reserve_at(100, 100, now)?.settle(10, 10)?;
        assert!(
            b.reserve_at(1, 1, now)
                .err()
                .unwrap()
                .downcast_ref::<RunStopped>()
                .is_some()
        );
        drop(b);
        let reopened = RunBudget::open_at(&t.path(), "run".into(), 1, 2, duration, now)?;
        assert_eq!(reopened.snapshot()?.calls, 2);
        drop(reopened);
        assert!(RunBudget::open_at(&t.path(), "run".into(), 2, 2, duration, now).is_err());
        Ok(())
    }
    #[test]
    fn expiry_unknown_requests_and_malformed_ledger_fail_closed() -> Result<()> {
        let t = Temp::new();
        let now = Utc::now();
        let d = Duration::from_secs(1);
        let b = RunBudget::open_at(&t.path(), "run".into(), 1, 10, d, now)?;
        assert!(
            b.reserve_at(1, 1, now + chrono::Duration::seconds(1))
                .is_err()
        );
        let permit = b.reserve_at(1, 1, now)?;
        let charged = b.snapshot()?.charged_nano_usd;
        drop(permit);
        assert!(b.check().is_err());
        drop(b);
        let b = RunBudget::open_at(&t.path(), "run".into(), 1, 10, d, now)?;
        assert!(b.check().is_err());
        assert_eq!(b.snapshot()?.charged_nano_usd, charged);
        drop(b);
        OpenOptions::new()
            .append(true)
            .open(t.path())?
            .write_all(b"partial")?;
        assert!(RunBudget::open_at(&t.path(), "run".into(), 1, 10, d, now).is_err());
        Ok(())
    }
    #[test]
    fn unaffordable_requests_do_not_consume_or_send() -> Result<()> {
        let t = Temp::new();
        let b = RunBudget::open(&t.path(), "run".into(), 1, 10, Duration::from_secs(60))?;
        assert!(b.reserve(1_000_000, 16384).is_err());
        assert_eq!(b.snapshot()?.calls, 0);
        Ok(())
    }
}
