//! Document ingestion and text preparation.
//!
//! Text preprocessing (normalization, sanitization) lives here.
//! Format conversion (HTML, PDF) lives in the `deformat` crate.
//! URL resolution lives in the CLI crate (`anno-cli`).

pub mod preprocessor;

pub use preprocessor::{DocumentPreprocessor, PreparedDocument};
