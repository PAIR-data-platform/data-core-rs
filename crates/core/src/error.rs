//! Error type shared by all crates in the workspace.

use thiserror::Error;

/// Error type for every fallible operation in the data engine.
///
/// The I/O, augmentation, and pipeline crates all return this type so that
/// errors can flow through a pipeline (and across the Python boundary)
/// without conversion.
#[derive(Error, Debug)]
pub enum DataCoreError {
    /// An Arrow compute or array construction error.
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// A column exists but has an unexpected data type, or a required field is missing.
    #[error("Schema mismatch: {0}")]
    SchemaMismatch(String),

    /// An episode violates an invariant (for example, it has zero timesteps).
    #[error("Invalid episode: {0}")]
    InvalidEpisode(String),

    /// A dataset-level lookup failed (for example, an out-of-range episode or frame index).
    #[error("Invalid dataset: {0}")]
    InvalidDataset(String),

    /// A named column is not present in the episode.
    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    /// A filesystem error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A JSON (de)serialization error, for example while parsing `info.json`.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// An augmentation could not be applied (bad parameters or incompatible input).
    #[error("Augmentation error: {msg}")]
    Augmentation {
        /// Human-readable description of the failure.
        msg: String,
    },

    /// A pipeline or parallel-execution failure (for example, thread pool construction).
    #[error("Pipeline error: {msg}")]
    Pipeline {
        /// Human-readable description of the failure.
        msg: String,
    },

    /// Any other error, such as Parquet, video decoding, or network failures.
    #[error("{0}")]
    Other(String),
}

/// Convenience alias for `std::result::Result<T, DataCoreError>`.
pub type Result<T> = std::result::Result<T, DataCoreError>;
