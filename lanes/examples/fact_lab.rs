use anyhow::{Context, Result, ensure};
use aurelius_lanes::fact_lab::{self, Mode, Options};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::PathBuf,
};
async fn execute() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" {
        println!(
            "fact_lab replay --config FILE --input FILE --responses DIR --output NEW_DIR [--records 9,62] [--fresh]\nfact_lab run --live --max-calls N --config FILE --input FILE --output NEW_DIR [--records 9,62] [--fresh] [--batch-adjacent] [--instructions FILE] [--expected FILE] [--model-dir DIR]\nfact_lab compare --before DIR --after DIR\nCached preparation is the default; use --fresh for new records/model settings. Source/proposals/rejections are private local artifacts; no production writes."
        );
        return Ok(());
    }
    let command = args.first().context("use replay, run or compare")?;
    let mut values = BTreeMap::new();
    let mut switches = BTreeSet::new();
    let mut index = 1;
    while index < args.len() {
        let key = &args[index];
        ensure!(key.starts_with("--"), "expected a named option");
        if matches!(key.as_str(), "--live" | "--fresh" | "--batch-adjacent") {
            ensure!(switches.insert(key.clone()), "duplicate option");
            index += 1;
        } else {
            ensure!(
                matches!(
                    key.as_str(),
                    "--config"
                        | "--input"
                        | "--output"
                        | "--records"
                        | "--responses"
                        | "--max-calls"
                        | "--instructions"
                        | "--before"
                        | "--after"
                        | "--expected"
                        | "--model-dir"
                ),
                "unknown option"
            );
            let value = args.get(index + 1).context("missing option value")?;
            ensure!(
                !value.starts_with("--") && values.insert(key.clone(), value.clone()).is_none(),
                "missing or duplicate option"
            );
            index += 2;
        }
    }
    let path = |key: &str| -> Result<PathBuf> {
        Ok(PathBuf::from(
            values.get(key).context("required path option missing")?,
        ))
    };
    if command == "compare" {
        ensure!(switches.is_empty(), "compare has no live options");
        println!(
            "{}",
            serde_json::to_string_pretty(&fact_lab::compare(
                &path("--before")?,
                &path("--after")?
            )?)?
        );
        return Ok(());
    }
    ensure!(
        matches!(command.as_str(), "replay" | "run"),
        "unknown lab command"
    );
    let mut config = fact_lab::load_config(&path("--config")?)?;
    if let Some(model_dir) = values.get("--model-dir") {
        config
            .language
            .as_mut()
            .context("language configuration required")?
            .model_dir = model_dir.clone();
    }
    if let Some(file) = values.get("--instructions") {
        let mut text = String::new();
        std::fs::File::open(file)?
            .take(16385)
            .read_to_string(&mut text)?;
        ensure!(text.len() <= 16384, "instructions exceed budget");
        config
            .fact_quality
            .as_mut()
            .context("typed quality policy required")?
            .extra_instructions = Some(text);
    }
    let records = fact_lab::load(&path("--input")?)?;
    let mut selected = BTreeSet::new();
    if let Some(ids) = values.get("--records") {
        for id in ids.split(',') {
            let id = if records.iter().any(|r| r.id == id) {
                id.to_owned()
            } else {
                format!(
                    "{:03}",
                    id.parse::<usize>().context("unknown record selector")?
                )
            };
            ensure!(selected.insert(id), "duplicate record selector");
        }
    }
    let mode = if command == "run" {
        ensure!(
            switches.contains("--live"),
            "run requires --live; replay never calls a model"
        );
        Mode::Live
    } else {
        ensure!(
            !switches.contains("--live"),
            "replay cannot enable live calls"
        );
        Mode::Replay(path("--responses")?)
    };
    let output = path("--output")?;
    let options = Options {
        mode,
        output: output.clone(),
        selected,
        fresh: switches.contains("--fresh"),
        batch_adjacent: switches.contains("--batch-adjacent"),
        max_calls: values
            .get("--max-calls")
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or_default(),
    };
    let summary = fact_lab::run(&config, &records, &options).await?;
    if let Some(expected) = values.get("--expected") {
        let mut bytes = Vec::new();
        std::fs::File::open(expected)?
            .take(2 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= 2 * 1024 * 1024,
            "expectations exceed byte budget"
        );
        let expected: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|_| anyhow::anyhow!("invalid expectations JSON"))?;
        use std::{io::Write, os::unix::fs::OpenOptionsExt};
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(output.join("expected-review.json"))?;
        serde_json::to_writer_pretty(
            &mut file,
            &serde_json::json!({"purpose":"manual side-by-side review only; never supplied to model; no automatic precision score","expectations":expected}),
        )?;
        file.write_all(b"\n")?;
    }
    println!("{}", serde_json::to_string_pretty(&summary)?);
    println!("Report: {}", output.join("report.md").display());
    Ok(())
}
#[tokio::main]
async fn main() {
    if execute().await.is_err() {
        eprintln!(
            "fact_lab failed; check arguments, local assets and safe run artifacts. Details withheld to protect source data and credentials."
        );
        std::process::exit(1)
    }
}
