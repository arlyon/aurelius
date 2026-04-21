use directories::ProjectDirs;
use std::path::PathBuf;

pub fn db_path() -> String {
    if let Some(proj_dirs) = ProjectDirs::from("com", "arlyon", "aurelius") {
        let data_dir = proj_dirs.data_dir();
        if !data_dir.exists() {
            std::fs::create_dir_all(data_dir).expect("Failed to create data directory");
        }
        data_dir.to_str().expect("Path is not valid UTF-8").to_string()
    } else {
        "aurelius_db".to_string()
    }
}

pub fn morning_pulse_path() -> PathBuf {
    PathBuf::from(db_path()).join("morning_pulse.ldjson")
}
