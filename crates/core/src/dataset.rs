//! The in-memory [`Dataset`] type and its [`DatasetMetadata`].

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::Schema;
use serde::{Deserialize, Serialize};

use crate::episode::Episode;
use crate::error::{DataCoreError, Result};
use crate::parallel::ParallelConfig;
use crate::schema::SchemaConfig;

/// A robotics dataset: a collection of episodes plus metadata.
///
/// Holds all episodes in memory as Arrow `RecordBatch`es and provides
/// methods for accessing, filtering, and transforming them. Augmentation
/// pipelines (`data-core-pipeline`) operate on this type. For on-demand
/// access to large datasets, see `data_core_io::LazyDataset`.
///
/// Map and filter methods keep `metadata().num_episodes` and
/// `metadata().num_steps` in sync with the episode list.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// All episodes in the dataset.
    episodes: Vec<Episode>,
    /// Dataset-level metadata.
    metadata: DatasetMetadata,
    /// Arrow schema shared by all episodes.
    schema: Arc<Schema>,
    /// Configuration for parallel operations.
    parallel_config: ParallelConfig,
}

/// Dataset-level metadata, typically parsed from LeRobot's `info.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name / identifier.
    pub name: String,
    /// Robot type (e.g., "aloha", "franka", "ur5").
    pub robot_type: Option<String>,
    /// Frames per second of the recorded data.
    pub fps: Option<f64>,
    /// Total number of episodes.
    pub num_episodes: usize,
    /// Total number of steps across all episodes.
    pub num_steps: usize,
    /// Action space dimensionality.
    pub action_dim: usize,
    /// State space dimensionality (if available).
    pub state_dim: Option<usize>,
    /// Camera names used in observations.
    pub camera_names: Vec<String>,
    /// Task descriptions (for multi-task datasets).
    pub tasks: Vec<String>,
    /// Source path or HuggingFace Hub ID.
    pub source: Option<String>,
    /// Arbitrary extra metadata.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for DatasetMetadata {
    fn default() -> Self {
        Self {
            name: "unnamed".to_string(),
            robot_type: None,
            fps: None,
            num_episodes: 0,
            num_steps: 0,
            action_dim: 7,
            state_dim: None,
            camera_names: vec!["cam_0".to_string()],
            tasks: Vec::new(),
            source: None,
            extra: HashMap::new(),
        }
    }
}

impl DatasetMetadata {
    /// Derive a [`SchemaConfig`] from the dataset metadata.
    ///
    /// `observation_state_dim` is set to `state_dim`, which matches LeRobot
    /// datasets where the state is stored as `observation.state`.
    pub fn to_schema_config(&self) -> SchemaConfig {
        SchemaConfig {
            action_dim: self.action_dim,
            state_dim: self.state_dim,
            observation_state_dim: self.state_dim, // often the same
            camera_names: self.camera_names.clone(),
        }
    }
}

impl Dataset {
    /// Create a new dataset from episodes, metadata, and a shared schema.
    ///
    /// No validation is performed; the metadata counts are stored as given.
    /// This currently never returns an error.
    pub fn new(
        episodes: Vec<Episode>,
        metadata: DatasetMetadata,
        schema: Arc<Schema>,
    ) -> Result<Self> {
        Ok(Self {
            episodes,
            metadata,
            schema,
            parallel_config: ParallelConfig::default(),
        })
    }

    /// Create an empty dataset whose schema is built from `metadata` with
    /// [`crate::build_episode_schema`].
    pub fn empty(metadata: DatasetMetadata) -> Self {
        let config = metadata.to_schema_config();
        let schema = Arc::new(crate::schema::build_episode_schema(&config));
        Self {
            episodes: Vec::new(),
            metadata,
            schema,
            parallel_config: ParallelConfig::default(),
        }
    }

    /// Get all episodes.
    pub fn episodes(&self) -> &[Episode] {
        &self.episodes
    }

    /// Get a mutable reference to all episodes.
    ///
    /// Changes made through this reference do not update
    /// `metadata().num_episodes` or `metadata().num_steps`.
    pub fn episodes_mut(&mut self) -> &mut Vec<Episode> {
        &mut self.episodes
    }

    /// Get an episode by its position in the dataset.
    ///
    /// # Errors
    ///
    /// Returns [`DataCoreError::InvalidDataset`] if `index` is out of range.
    pub fn episode(&self, index: usize) -> Result<&Episode> {
        self.episodes.get(index).ok_or_else(|| {
            DataCoreError::InvalidDataset(format!(
                "Episode index {} out of range (dataset has {} episodes)",
                index,
                self.episodes.len()
            ))
        })
    }

    /// Number of episodes in the dataset.
    pub fn num_episodes(&self) -> usize {
        self.episodes.len()
    }

    /// Total number of steps across all episodes.
    pub fn num_steps(&self) -> usize {
        self.episodes.iter().map(|ep| ep.num_steps()).sum()
    }

    /// Get dataset metadata.
    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// Get a mutable reference to dataset metadata.
    pub fn metadata_mut(&mut self) -> &mut DatasetMetadata {
        &mut self.metadata
    }

    /// Get the shared Arrow schema.
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Append an episode and update the metadata counts.
    pub fn add_episode(&mut self, episode: Episode) {
        self.metadata.num_episodes = self.episodes.len() + 1;
        self.metadata.num_steps += episode.num_steps();
        self.episodes.push(episode);
    }

    /// Replace all episodes (e.g., after a global augmentation) and update the
    /// metadata counts.
    pub fn with_episodes(mut self, episodes: Vec<Episode>) -> Self {
        self.metadata.num_episodes = episodes.len();
        self.metadata.num_steps = episodes.iter().map(|ep| ep.num_steps()).sum();
        self.episodes = episodes;
        self
    }

    /// Map a function over all episodes sequentially, producing a new dataset.
    ///
    /// Stops at and returns the first error.
    pub fn map_episodes<F>(&self, f: F) -> Result<Self>
    where
        F: Fn(&Episode) -> Result<Episode>,
    {
        let new_episodes: Result<Vec<Episode>> = self.episodes.iter().map(f).collect();
        let episodes = new_episodes?;
        let mut metadata = self.metadata.clone();
        metadata.num_episodes = episodes.len();
        metadata.num_steps = episodes.iter().map(|ep| ep.num_steps()).sum();
        Ok(Self {
            episodes,
            metadata,
            schema: self.schema.clone(),
            parallel_config: self.parallel_config.clone(),
        })
    }

    /// Keep the episodes for which `predicate` returns true.
    pub fn filter_episodes<F>(&self, predicate: F) -> Self
    where
        F: Fn(&Episode) -> bool,
    {
        let episodes: Vec<Episode> = self
            .episodes
            .iter()
            .filter(|ep| predicate(ep))
            .cloned()
            .collect();
        let mut metadata = self.metadata.clone();
        metadata.num_episodes = episodes.len();
        metadata.num_steps = episodes.iter().map(|ep| ep.num_steps()).sum();
        Self {
            episodes,
            metadata,
            schema: self.schema.clone(),
            parallel_config: self.parallel_config.clone(),
        }
    }

    /// Map a function over all episodes in parallel, producing a new dataset.
    ///
    /// This uses Rayon to process episodes concurrently. The function `f` must be
    /// thread-safe (`Send + Sync`). The order of episodes is preserved. If
    /// [`ParallelConfig::num_threads`] is set, a dedicated pool of that size
    /// is built for this call; otherwise the global Rayon pool is used.
    ///
    /// # Errors
    ///
    /// Returns an error from `f` (which one is unspecified if several
    /// episodes fail), or [`DataCoreError::Pipeline`] if the thread pool
    /// cannot be built.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use data_core::Dataset;
    ///
    /// # fn example(dataset: Dataset) -> data_core::Result<()> {
    /// // Process episodes in parallel
    /// let augmented = dataset.par_map_episodes(|episode| {
    ///     // Apply augmentation to each episode concurrently
    ///     Ok(episode.clone())
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn par_map_episodes<F>(&self, f: F) -> Result<Self>
    where
        F: Fn(&Episode) -> Result<Episode> + Send + Sync,
    {
        use rayon::prelude::*;

        let chunk_size = self.parallel_config.chunk_size;
        let run_parallel = || -> Result<Vec<Episode>> {
            if chunk_size > 1 {
                // Use chunking for better load balancing.
                self.episodes
                    .par_chunks(chunk_size)
                    .map(|chunk| chunk.iter().map(&f).collect::<Result<Vec<_>>>())
                    .collect::<Result<Vec<_>>>()
                    .map(|chunks| chunks.into_iter().flatten().collect())
            } else {
                // Process each episode independently.
                self.episodes.par_iter().map(&f).collect()
            }
        };

        let new_episodes: Result<Vec<Episode>> =
            if let Some(num_threads) = self.parallel_config.num_threads {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .map_err(|e| DataCoreError::Pipeline {
                        msg: format!("Failed to build Rayon thread pool: {e}"),
                    })?;
                pool.install(run_parallel)
            } else {
                run_parallel()
            };

        let episodes = new_episodes?;
        let mut metadata = self.metadata.clone();
        metadata.num_episodes = episodes.len();
        metadata.num_steps = episodes.iter().map(|ep| ep.num_steps()).sum();
        Ok(Self {
            episodes,
            metadata,
            schema: self.schema.clone(),
            parallel_config: self.parallel_config.clone(),
        })
    }

    /// Filter episodes in parallel by a predicate.
    ///
    /// This uses Rayon to evaluate the predicate concurrently across episodes.
    /// Episode order is preserved. If a dedicated pool of
    /// [`ParallelConfig::num_threads`] threads cannot be built, the global
    /// Rayon pool is used instead.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use data_core::Dataset;
    ///
    /// # fn example(dataset: Dataset) -> data_core::Result<()> {
    /// // Filter episodes in parallel
    /// let filtered = dataset.par_filter_episodes(|episode| {
    ///     episode.num_steps() > 100
    /// });
    /// # Ok(())
    /// # }
    /// ```
    pub fn par_filter_episodes<F>(&self, predicate: F) -> Self
    where
        F: Fn(&Episode) -> bool + Send + Sync,
    {
        use rayon::prelude::*;

        let run_parallel = || -> Vec<Episode> {
            self.episodes
                .par_iter()
                .filter(|ep| predicate(ep))
                .cloned()
                .collect()
        };

        let episodes: Vec<Episode> = if let Some(num_threads) = self.parallel_config.num_threads {
            match rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
            {
                Ok(pool) => pool.install(run_parallel),
                Err(_) => run_parallel(),
            }
        } else {
            run_parallel()
        };

        let mut metadata = self.metadata.clone();
        metadata.num_episodes = episodes.len();
        metadata.num_steps = episodes.iter().map(|ep| ep.num_steps()).sum();
        Self {
            episodes,
            metadata,
            schema: self.schema.clone(),
            parallel_config: self.parallel_config.clone(),
        }
    }

    /// Configure parallel processing for this dataset (a `chunk_size` of 0 is
    /// treated as 1).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use data_core::{Dataset, ParallelConfig};
    ///
    /// # let dataset = Dataset::empty(Default::default());
    /// let dataset = dataset.with_parallel_config(
    ///     ParallelConfig::new()
    ///         .with_threads(8)
    ///         .with_chunk_size(10)
    /// );
    /// ```
    pub fn with_parallel_config(mut self, config: ParallelConfig) -> Self {
        let mut config = config;
        config.chunk_size = config.chunk_size.max(1);
        self.parallel_config = config;
        self
    }

    /// Get the current parallel configuration.
    pub fn parallel_config(&self) -> &ParallelConfig {
        &self.parallel_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::build_episode_schema;
    use arrow::record_batch::RecordBatch;
    use arrow_array::{
        BooleanArray, FixedSizeListArray, Float32Array, Int64Array, LargeBinaryArray,
    };
    use arrow_schema::{DataType, Field};

    fn make_test_dataset(num_episodes: usize, steps_per_episode: usize) -> Dataset {
        let config = SchemaConfig::default();
        let schema = Arc::new(build_episode_schema(&config));

        let mut episodes = Vec::new();
        for ep_idx in 0..num_episodes {
            let n = steps_per_episode;
            let step_indices = Int64Array::from((0..n as i64).collect::<Vec<_>>());
            let action_values = Float32Array::from(vec![0.0f32; n * 7]);
            let actions = FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                7,
                Arc::new(action_values),
                None,
            )
            .unwrap();
            let images = LargeBinaryArray::from(vec![b"img" as &[u8]; n]);
            let rewards = Float32Array::from(vec![0.0f32; n]);
            let mut done_vals = vec![false; n];
            done_vals[n - 1] = true;
            let dones = BooleanArray::from(done_vals);
            let ep_indices = Int64Array::from(vec![ep_idx as i64; n]);
            let timestamps = arrow_array::Float64Array::from(
                (0..n).map(|i| i as f64 / 10.0).collect::<Vec<_>>(),
            );
            let task_indices = Int64Array::from(vec![0i64; n]);

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(step_indices),
                    Arc::new(timestamps),
                    Arc::new(actions),
                    Arc::new(images),
                    Arc::new(rewards),
                    Arc::new(dones),
                    Arc::new(ep_indices),
                    Arc::new(task_indices),
                ],
            )
            .unwrap();

            episodes.push(Episode::from_record_batch(batch).unwrap());
        }

        let metadata = DatasetMetadata {
            name: "test".to_string(),
            num_episodes,
            num_steps: num_episodes * steps_per_episode,
            action_dim: 7,
            camera_names: vec!["cam_0".to_string()],
            ..Default::default()
        };

        Dataset::new(episodes, metadata, schema).unwrap()
    }

    #[test]
    fn test_dataset_creation() {
        let ds = make_test_dataset(5, 10);
        assert_eq!(ds.num_episodes(), 5);
        assert_eq!(ds.num_steps(), 50);
    }

    #[test]
    fn test_dataset_episode_access() {
        let ds = make_test_dataset(3, 10);
        assert!(ds.episode(0).is_ok());
        assert!(ds.episode(2).is_ok());
        assert!(ds.episode(3).is_err());
    }

    #[test]
    fn test_dataset_map_episodes() {
        let ds = make_test_dataset(3, 10);
        let mapped = ds.map_episodes(|ep| Ok(ep.clone())).unwrap();
        assert_eq!(mapped.num_episodes(), 3);
        assert_eq!(mapped.num_steps(), 30);
    }

    #[test]
    fn test_dataset_par_map_episodes() {
        let ds = make_test_dataset(3, 10);
        let mapped = ds.par_map_episodes(|ep| Ok(ep.clone())).unwrap();
        assert_eq!(mapped.num_episodes(), 3);
        assert_eq!(mapped.num_steps(), 30);
    }

    #[test]
    fn test_dataset_par_map_episodes_error_propagation() {
        let ds = make_test_dataset(3, 10);
        let result = ds.par_map_episodes(|ep| {
            if ep.metadata().episode_index == 1 {
                Err(DataCoreError::Augmentation {
                    msg: "boom".to_string(),
                })
            } else {
                Ok(ep.clone())
            }
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_dataset_filter_episodes() {
        let ds = make_test_dataset(5, 10);
        let filtered = ds.filter_episodes(|ep| ep.metadata().episode_index < 3);
        assert_eq!(filtered.num_episodes(), 3);
    }

    #[test]
    fn test_dataset_par_filter_episodes() {
        let ds = make_test_dataset(5, 10);
        let filtered = ds.par_filter_episodes(|ep| ep.metadata().episode_index < 3);
        assert_eq!(filtered.num_episodes(), 3);
    }

    #[test]
    fn test_dataset_par_map_edge_cases() {
        let empty = make_test_dataset(0, 1);
        let mapped_empty = empty.par_map_episodes(|ep| Ok(ep.clone())).unwrap();
        assert_eq!(mapped_empty.num_episodes(), 0);

        let single = make_test_dataset(1, 5);
        let mapped_single = single.par_map_episodes(|ep| Ok(ep.clone())).unwrap();
        assert_eq!(mapped_single.num_episodes(), 1);
        assert_eq!(mapped_single.num_steps(), 5);
    }

    #[test]
    fn test_dataset_add_episode() {
        let mut ds = make_test_dataset(2, 10);
        let ds2 = make_test_dataset(1, 5);
        ds.add_episode(ds2.episodes()[0].clone());
        assert_eq!(ds.num_episodes(), 3);
        assert_eq!(ds.num_steps(), 25);
    }
}
