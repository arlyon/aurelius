use ignore::WalkBuilder;
use std::path::PathBuf;
use tracing::{debug, info};

pub fn walk_directory(path: PathBuf) -> Vec<PathBuf> {
    info!("Scanning directory: {:?}", path);
    let walker = WalkBuilder::new(path).hidden(true).git_ignore(true).build();
    let mut files = Vec::new();

    for entry in walker {
        if let Ok(entry) = entry
            && entry.file_type().map(|ft| ft.is_file()).unwrap_or(false)
        {
            let p = entry.into_path();
            debug!("Found file: {:?}", p);
            files.push(p);
        }
    }

    info!("Found {} files in directory", files.len());
    files
}
