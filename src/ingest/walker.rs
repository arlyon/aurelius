use ignore::WalkBuilder;
use std::path::PathBuf;

pub fn walk_directory(path: PathBuf) -> Vec<PathBuf> {
    let walker = WalkBuilder::new(path).hidden(false).git_ignore(true).build();
    let mut files = Vec::new();

    for entry in walker {
        if let Ok(entry) = entry {
            if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                files.push(entry.into_path());
            }
        }
    }

    files
}
