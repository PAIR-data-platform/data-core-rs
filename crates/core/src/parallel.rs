//! Rayon configuration for parallel dataset operations.

use serde::{Deserialize, Serialize};

/// Configuration for parallel dataset operations using Rayon.
///
/// This controls how episodes are processed in parallel when using
/// methods like `par_map_episodes()` and `par_filter_episodes()`. Only
/// `par_map_episodes()` uses `chunk_size`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of threads to use for parallel operations.
    ///
    /// - `None`: Use Rayon's default (typically number of logical cores)
    /// - `Some(n)`: Use exactly n threads
    pub num_threads: Option<usize>,

    /// Chunk size for `par_map_episodes()` (ignored by `par_filter_episodes()`).
    ///
    /// Episodes are grouped into chunks of this size before being
    /// distributed to worker threads. Larger chunks reduce overhead
    /// but may cause load imbalance.
    ///
    /// Default is 1 (each episode processed independently).
    pub chunk_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: None, // Use all available cores
            chunk_size: 1,     // Process episodes individually
        }
    }
}

impl ParallelConfig {
    /// Create a new parallel configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of threads to use.
    ///
    /// # Examples
    ///
    /// ```
    /// use data_core::ParallelConfig;
    ///
    /// let config = ParallelConfig::new().with_threads(8);
    /// ```
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Set the chunk size for parallel iteration (values below 1 become 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use data_core::ParallelConfig;
    ///
    /// let config = ParallelConfig::new().with_chunk_size(10);
    /// ```
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }

    /// Configure Rayon's global thread pool with `num_threads`, if set.
    ///
    /// Rayon allows the global pool to be built only once per process, so
    /// this should be called at most once, at program startup. Dataset
    /// methods that take a [`ParallelConfig`] build a scoped pool instead and
    /// do not need this.
    ///
    /// # Errors
    ///
    /// Returns the Rayon error if the global pool was already initialized.
    pub fn apply(&self) -> Result<(), rayon::ThreadPoolBuildError> {
        if let Some(num_threads) = self.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()?;
        }
        Ok(())
    }
}
