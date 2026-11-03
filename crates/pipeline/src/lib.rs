//! Ordered execution of augmentation steps over an in-memory dataset.
//!
//! A [`Pipeline`] is a list of [`AugmentationStep`]s from
//! `data-core-augmentations`. Steps run in the order they were added, each
//! on the output of the previous one:
//!
//! - a local step is applied to every episode independently (sequentially in
//!   [`Pipeline::run`], or in parallel across episodes with Rayon in
//!   [`Pipeline::run_par`]); episode order is preserved;
//! - a global step is applied once to the whole dataset and may change the
//!   number of episodes.
//!
//! Pipelines operate on an eagerly loaded [`Dataset`] (for example from
//! `data_core_io::LeRobotLoader`). For per-sample augmentation during
//! training, add local augmentations to `data_core_io::LazyDataset` instead.
//!
//! # Example
//!
//! ```
//! use data_core::{Dataset, DatasetMetadata};
//! use data_core_augmentations::action::ActionNoise;
//! use data_core_pipeline::Pipeline;
//!
//! let mut pipeline = Pipeline::new();
//! pipeline.add_local(ActionNoise::new(0.01)).with_threads(2);
//! assert_eq!(pipeline.step_names(), vec!["action_noise"]);
//!
//! let empty = Dataset::empty(DatasetMetadata::default());
//! let out = pipeline.run_par(empty)?;
//! assert_eq!(out.num_episodes(), 0);
//! # Ok::<(), data_core::DataCoreError>(())
//! ```

#![warn(missing_docs)]

use data_core::{Dataset, ParallelConfig, Result};
use data_core_augmentations::AugmentationStep;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::info;

/// Serializable description of one pipeline step (for YAML/JSON configs).
///
/// This crate only defines the format; mapping a `StepConfig` to an
/// augmentation instance is left to the caller (for example through a
/// `data_core_augmentations::AugmentationRegistry`).
///
/// ```
/// use data_core_pipeline::{PipelineConfig, StepType};
///
/// let json = r#"{"pipeline": [{"name": "action_noise", "type": "local", "params": {"std": 0.01}}]}"#;
/// let config: PipelineConfig = serde_json::from_str(json).unwrap();
/// assert_eq!(config.pipeline[0].name, "action_noise");
/// assert!(matches!(config.pipeline[0].step_type, StepType::Local));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepConfig {
    /// Augmentation name, as returned by its `name()`.
    pub name: String,
    /// Whether the step is local or global (serialized as `type`).
    #[serde(rename = "type")]
    pub step_type: StepType,
    /// Free-form parameters for the augmentation (defaults to `null`).
    #[serde(default)]
    pub params: serde_json::Value,
}

/// Kind of a pipeline step (serialized in lowercase).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepType {
    /// Applied to each episode independently.
    Local,
    /// Applied once to the whole dataset.
    Global,
}

/// Serializable pipeline configuration: an ordered list of steps under the
/// `pipeline` key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Steps in execution order.
    pub pipeline: Vec<StepConfig>,
}

/// Timing information for a single pipeline step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepStats {
    /// Step name.
    pub name: String,
    /// Wall-clock duration of the step in milliseconds.
    pub duration_ms: u128,
    /// `"parallel-local"` or `"sequential-global"`.
    pub mode: String,
}

/// Execution statistics for a pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Per-step timings, in execution order.
    pub step_stats: Vec<StepStats>,
    /// Wall-clock duration of the whole run in milliseconds.
    pub total_duration_ms: u128,
    /// Number of episodes in the output dataset.
    pub episodes_processed: usize,
}

/// An ordered list of augmentation steps to apply to a dataset.
pub struct Pipeline {
    steps: Vec<AugmentationStep>,
    parallel_config: Option<ParallelConfig>,
}

impl Pipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            parallel_config: None,
        }
    }

    /// Add a local augmentation step.
    pub fn add_local(
        &mut self,
        aug: impl data_core_augmentations::LocalAugmentation + 'static,
    ) -> &mut Self {
        self.steps
            .push(AugmentationStep::Local(std::sync::Arc::new(aug)));
        self
    }

    /// Add a global augmentation step.
    pub fn add_global(
        &mut self,
        aug: impl data_core_augmentations::GlobalAugmentation + 'static,
    ) -> &mut Self {
        self.steps
            .push(AugmentationStep::Global(std::sync::Arc::new(aug)));
        self
    }

    /// Add a pre-built augmentation step.
    pub fn add_step(&mut self, step: AugmentationStep) -> &mut Self {
        self.steps.push(step);
        self
    }

    /// Configure parallel execution for `run_par` and `run_par_with_stats`.
    ///
    /// The configuration is applied to the input dataset (replacing its own
    /// [`ParallelConfig`]) when a parallel run starts. If no configuration is
    /// set here, the dataset's existing configuration is used.
    pub fn with_parallel_config(&mut self, config: ParallelConfig) -> &mut Self {
        self.parallel_config = Some(config);
        self
    }

    /// Set the number of threads for parallel local steps.
    pub fn with_threads(&mut self, num_threads: usize) -> &mut Self {
        let mut config = self.parallel_config.clone().unwrap_or_default();
        config.num_threads = Some(num_threads);
        self.parallel_config = Some(config);
        self
    }

    /// Set the episode chunk size for parallel local steps (minimum 1).
    pub fn with_chunk_size(&mut self, chunk_size: usize) -> &mut Self {
        let mut config = self.parallel_config.clone().unwrap_or_default();
        config.chunk_size = chunk_size.max(1);
        self.parallel_config = Some(config);
        self
    }

    /// Get the number of steps in the pipeline.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Step names in execution order.
    pub fn step_names(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.name()).collect()
    }

    /// Execute the pipeline sequentially on a dataset.
    ///
    /// # Errors
    ///
    /// Stops at and returns the first error from any step.
    pub fn run(&self, dataset: Dataset) -> Result<Dataset> {
        let mut current = dataset;

        for (i, step) in self.steps.iter().enumerate() {
            info!(
                "Running pipeline step {}/{}: '{}'",
                i + 1,
                self.steps.len(),
                step.name()
            );

            current = match step {
                AugmentationStep::Local(aug) => current.map_episodes(|ep| aug.apply(ep))?,
                AugmentationStep::Global(aug) => aug.apply(&current)?,
            };
        }

        Ok(current)
    }

    /// Execute the pipeline in parallel on a dataset.
    ///
    /// Local augmentations are processed in parallel across episodes using Rayon.
    /// Global augmentations remain sequential (they operate on the entire dataset).
    /// Step order and episode order are the same as in [`Pipeline::run`].
    ///
    /// # Errors
    ///
    /// Returns the first failing step's error (for a local step, the error
    /// of one of its failing episodes).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use data_core_pipeline::Pipeline;
    /// use data_core::{Dataset, ParallelConfig};
    ///
    /// # fn example(dataset: Dataset) -> data_core::Result<()> {
    /// let mut pipeline = Pipeline::new();
    /// // Add augmentations...
    ///
    /// // Configure parallelism
    /// let dataset = dataset.with_parallel_config(
    ///     ParallelConfig::new().with_threads(8)
    /// );
    ///
    /// // Run pipeline in parallel
    /// let augmented = pipeline.run_par(dataset)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_par(&self, dataset: Dataset) -> Result<Dataset> {
        let mut current = if let Some(config) = &self.parallel_config {
            dataset.with_parallel_config(config.clone())
        } else {
            dataset
        };

        for (i, step) in self.steps.iter().enumerate() {
            match step {
                AugmentationStep::Local(aug) => {
                    info!(
                        "Running pipeline step {}/{} (parallel): '{}'",
                        i + 1,
                        self.steps.len(),
                        step.name()
                    );
                    current = current.par_map_episodes(|ep| aug.apply(ep))?;
                }
                AugmentationStep::Global(aug) => {
                    info!(
                        "Running pipeline step {}/{} (sequential, global): '{}'",
                        i + 1,
                        self.steps.len(),
                        step.name()
                    );
                    current = aug.apply(&current)?;
                }
            }
        }

        Ok(current)
    }

    /// Execute the pipeline like [`Pipeline::run_par`] and also return
    /// per-step timing statistics.
    pub fn run_par_with_stats(&self, dataset: Dataset) -> Result<(Dataset, PipelineStats)> {
        let overall_start = Instant::now();
        let mut current = if let Some(config) = &self.parallel_config {
            dataset.with_parallel_config(config.clone())
        } else {
            dataset
        };
        let mut step_stats = Vec::with_capacity(self.steps.len());

        for (i, step) in self.steps.iter().enumerate() {
            let step_start = Instant::now();
            match step {
                AugmentationStep::Local(aug) => {
                    info!(
                        "Running pipeline step {}/{} (parallel): '{}'",
                        i + 1,
                        self.steps.len(),
                        step.name()
                    );
                    current = current.par_map_episodes(|ep| aug.apply(ep))?;
                    step_stats.push(StepStats {
                        name: step.name().to_string(),
                        duration_ms: step_start.elapsed().as_millis(),
                        mode: "parallel-local".to_string(),
                    });
                }
                AugmentationStep::Global(aug) => {
                    info!(
                        "Running pipeline step {}/{} (sequential, global): '{}'",
                        i + 1,
                        self.steps.len(),
                        step.name()
                    );
                    current = aug.apply(&current)?;
                    step_stats.push(StepStats {
                        name: step.name().to_string(),
                        duration_ms: step_start.elapsed().as_millis(),
                        mode: "sequential-global".to_string(),
                    });
                }
            }
        }

        let stats = PipelineStats {
            step_stats,
            total_duration_ms: overall_start.elapsed().as_millis(),
            episodes_processed: current.num_episodes(),
        };
        Ok((current, stats))
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field};
    use arrow::record_batch::RecordBatch;
    use arrow_array::*;
    use data_core::{build_episode_schema, Dataset, DatasetMetadata, Episode, SchemaConfig};
    use data_core_augmentations::{GlobalAugmentation, LocalAugmentation};
    use std::sync::Arc;

    fn make_test_dataset(num_episodes: usize, steps_per_episode: usize) -> Dataset {
        let config = SchemaConfig::default();
        let schema = Arc::new(build_episode_schema(&config));

        let mut episodes = Vec::new();
        for ep_idx in 0..num_episodes {
            let n = steps_per_episode;
            let step_indices = Int64Array::from((0..n as i64).collect::<Vec<_>>());
            let timestamps =
                Float64Array::from((0..n).map(|i| i as f64 / 10.0).collect::<Vec<_>>());
            let action_values = Float32Array::from(vec![1.0f32; n * 7]);
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

    struct DoubleRewardAug;

    impl LocalAugmentation for DoubleRewardAug {
        fn name(&self) -> &str {
            "double_reward"
        }

        fn apply(&self, episode: &Episode) -> data_core::Result<Episode> {
            let rewards = episode.rewards()?;
            let new_rewards: Float32Array = rewards.iter().map(|v| v.map(|x| x * 2.0)).collect();
            episode.with_column("reward", Arc::new(new_rewards))
        }
    }

    struct DuplicateEpisodesAug;

    impl GlobalAugmentation for DuplicateEpisodesAug {
        fn name(&self) -> &str {
            "duplicate_episodes"
        }

        fn apply(&self, dataset: &Dataset) -> data_core::Result<Dataset> {
            let mut new_episodes: Vec<Episode> = dataset.episodes().to_vec();
            new_episodes.extend(dataset.episodes().iter().cloned());
            Ok(dataset.clone().with_episodes(new_episodes))
        }
    }

    #[test]
    fn test_pipeline_local_augmentation() {
        let ds = make_test_dataset(2, 5);
        let mut pipeline = Pipeline::new();
        pipeline.add_local(DoubleRewardAug);

        let result = pipeline.run(ds).unwrap();
        assert_eq!(result.num_episodes(), 2);
    }

    #[test]
    fn test_pipeline_global_augmentation() {
        let ds = make_test_dataset(3, 5);
        let mut pipeline = Pipeline::new();
        pipeline.add_global(DuplicateEpisodesAug);

        let result = pipeline.run(ds).unwrap();
        assert_eq!(result.num_episodes(), 6);
    }

    #[test]
    fn test_pipeline_mixed_steps() {
        let ds = make_test_dataset(2, 5);
        let mut pipeline = Pipeline::new();
        pipeline.add_local(DoubleRewardAug);
        pipeline.add_global(DuplicateEpisodesAug);
        pipeline.add_local(DoubleRewardAug);

        let result = pipeline.run(ds).unwrap();
        assert_eq!(result.num_episodes(), 4); // 2 → 2 (local) → 4 (global) → 4 (local)
        assert_eq!(result.num_steps(), 20); // 4 episodes × 5 steps each
    }

    #[test]
    fn test_empty_pipeline() {
        let ds = make_test_dataset(2, 5);
        let pipeline = Pipeline::new();
        let result = pipeline.run(ds).unwrap();
        assert_eq!(result.num_episodes(), 2);
    }

    #[test]
    fn test_pipeline_step_names() {
        let mut pipeline = Pipeline::new();
        pipeline.add_local(DoubleRewardAug);
        pipeline.add_global(DuplicateEpisodesAug);
        assert_eq!(
            pipeline.step_names(),
            vec!["double_reward", "duplicate_episodes"]
        );
    }

    #[test]
    fn test_pipeline_parallel_matches_sequential() {
        let ds = make_test_dataset(4, 8);
        let mut pipeline = Pipeline::new();
        pipeline.add_local(DoubleRewardAug);
        pipeline.add_global(DuplicateEpisodesAug);
        pipeline.add_local(DoubleRewardAug);

        let seq = pipeline.run(ds.clone()).unwrap();
        let par = pipeline.run_par(ds).unwrap();

        assert_eq!(seq.num_episodes(), par.num_episodes());
        assert_eq!(seq.num_steps(), par.num_steps());
    }

    /// Adds 1 to every reward.
    struct AddOneRewardAug;

    impl LocalAugmentation for AddOneRewardAug {
        fn name(&self) -> &str {
            "add_one_reward"
        }

        fn apply(&self, episode: &Episode) -> data_core::Result<Episode> {
            let rewards = episode.rewards()?;
            let new_rewards: Float32Array = rewards.iter().map(|v| v.map(|x| x + 1.0)).collect();
            episode.with_column("reward", Arc::new(new_rewards))
        }
    }

    /// Keeps only the first episode.
    struct KeepFirstAug;

    impl GlobalAugmentation for KeepFirstAug {
        fn name(&self) -> &str {
            "keep_first"
        }

        fn apply(&self, dataset: &Dataset) -> data_core::Result<Dataset> {
            Ok(dataset
                .clone()
                .with_episodes(dataset.episodes()[..1].to_vec()))
        }
    }

    /// A local step that fails on episode 2.
    struct FailOnEpisodeTwo;

    impl LocalAugmentation for FailOnEpisodeTwo {
        fn name(&self) -> &str {
            "fail_on_two"
        }

        fn apply(&self, episode: &Episode) -> data_core::Result<Episode> {
            if episode.metadata().episode_index == 2 {
                return Err(data_core::DataCoreError::Augmentation {
                    msg: "episode 2".into(),
                });
            }
            Ok(episode.clone())
        }
    }

    fn rewards(ds: &Dataset) -> Vec<Vec<f32>> {
        ds.episodes()
            .iter()
            .map(|ep| ep.rewards().unwrap().values().to_vec())
            .collect()
    }

    fn episode_indices(ds: &Dataset) -> Vec<i64> {
        ds.episodes()
            .iter()
            .map(|ep| ep.metadata().episode_index)
            .collect()
    }

    #[test]
    fn test_steps_run_in_insertion_order() {
        // Rewards start at 0. (0 + 1) * 2 = 2, while (0 * 2) + 1 = 1.
        let mut add_then_double = Pipeline::new();
        add_then_double
            .add_local(AddOneRewardAug)
            .add_local(DoubleRewardAug);
        let mut double_then_add = Pipeline::new();
        double_then_add
            .add_local(DoubleRewardAug)
            .add_local(AddOneRewardAug);

        for parallel in [false, true] {
            let run = |p: &Pipeline| {
                let ds = make_test_dataset(3, 4);
                if parallel {
                    p.run_par(ds).unwrap()
                } else {
                    p.run(ds).unwrap()
                }
            };
            assert_eq!(rewards(&run(&add_then_double)), vec![vec![2.0; 4]; 3]);
            assert_eq!(rewards(&run(&double_then_add)), vec![vec![1.0; 4]; 3]);
        }
    }

    #[test]
    fn test_local_and_global_interleaving() {
        // local(+1) -> global(duplicate) -> local(x2) -> global(keep first)
        let mut pipeline = Pipeline::new();
        pipeline
            .add_local(AddOneRewardAug)
            .add_global(DuplicateEpisodesAug)
            .add_local(DoubleRewardAug)
            .add_global(KeepFirstAug);

        for parallel in [false, true] {
            let ds = make_test_dataset(3, 4);
            let out = if parallel {
                pipeline.run_par(ds).unwrap()
            } else {
                pipeline.run(ds).unwrap()
            };
            assert_eq!(out.num_episodes(), 1);
            assert_eq!(out.metadata().num_episodes, 1);
            assert_eq!(out.metadata().num_steps, 4);
            assert_eq!(rewards(&out), vec![vec![2.0; 4]]);
        }
    }

    #[test]
    fn test_run_par_preserves_episode_order() {
        let ds = make_test_dataset(16, 3);
        let mut pipeline = Pipeline::new();
        pipeline
            .with_threads(4)
            .add_local(AddOneRewardAug)
            .add_global(DuplicateEpisodesAug)
            .add_local(DoubleRewardAug);
        let seq = pipeline.run(ds.clone()).unwrap();
        let par = pipeline.run_par(ds).unwrap();
        let expected: Vec<i64> = (0..16).chain(0..16).collect();
        assert_eq!(episode_indices(&seq), expected);
        assert_eq!(episode_indices(&par), expected);
        assert_eq!(rewards(&seq), rewards(&par));
    }

    #[test]
    fn test_local_step_error_propagates() {
        let mut pipeline = Pipeline::new();
        pipeline.add_local(FailOnEpisodeTwo);
        assert!(pipeline.run(make_test_dataset(4, 2)).is_err());
        assert!(pipeline.run_par(make_test_dataset(4, 2)).is_err());
        assert!(pipeline.run(make_test_dataset(2, 2)).is_ok());
    }

    #[test]
    fn test_pipeline_run_par_with_stats() {
        let ds = make_test_dataset(3, 5);
        let mut pipeline = Pipeline::new();
        pipeline.with_threads(2).with_chunk_size(1);
        pipeline.add_local(DoubleRewardAug);
        pipeline.add_global(DuplicateEpisodesAug);
        let (out, stats) = pipeline.run_par_with_stats(ds).unwrap();
        assert_eq!(out.num_episodes(), 6);
        assert_eq!(stats.step_stats.len(), 2);
        assert_eq!(stats.episodes_processed, 6);
    }
}
