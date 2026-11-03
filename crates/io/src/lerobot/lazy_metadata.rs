//! Extended metadata for lazy dataset loading.
//!
//! This module provides [`LazyDatasetMetadata`], which reads only the metadata
//! files and builds a per-episode index (global frame range, data file, and
//! per-camera video file and timestamps) without decoding any frames:
//!
//! - v2: `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`. Each
//!   episode has its own Parquet file and video files.
//! - v3: `meta/info.json` and `meta/episodes/chunk-*/file-*.parquet`. Many
//!   episodes share each Parquet and video file; per-episode video offsets
//!   come from `videos/{key}/from_timestamp`. For v3 the `episode_index`
//!   column of the files under `data/` is also scanned once to find which
//!   data file holds each episode.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::metadata::LeRobotInfo;
use data_core::{DataCoreError, Result};

/// Extended dataset metadata for lazy loading.
///
/// Unlike `DatasetMetadata`, this includes per-episode index ranges and video paths
/// needed for on-demand frame access without loading all data upfront.
///
/// `episodes` is sorted by `episode_index`. LeRobot datasets number episodes
/// `0..total_episodes`, but lookups by episode index also work when the
/// numbering has gaps.
#[derive(Debug, Clone)]
pub struct LazyDatasetMetadata {
    /// Dataset root directory.
    pub root: PathBuf,
    /// Dataset name / identifier.
    pub name: String,
    /// Robot type (e.g., "aloha", "franka").
    pub robot_type: Option<String>,
    /// Frames per second.
    pub fps: f64,
    /// Total number of episodes.
    pub total_episodes: usize,
    /// Total number of frames across all episodes.
    pub total_frames: usize,
    /// Action space dimensionality.
    pub action_dim: usize,
    /// State space dimensionality.
    pub state_dim: Option<usize>,
    /// Camera names.
    pub camera_names: Vec<String>,
    /// Video feature keys (e.g., "observation.images.top").
    pub video_keys: Vec<String>,
    /// Per-episode metadata.
    pub episodes: Vec<EpisodeInfo>,
    /// Task descriptions indexed by task_index.
    pub tasks: Vec<String>,
    /// Whether this is a v3 dataset (video files stored separately).
    pub is_v3: bool,
    /// Data path template for parquet files.
    pub data_path_template: Option<String>,
    /// Video path template.
    pub video_path_template: Option<String>,
    /// Feature definitions from info.json.
    pub features: HashMap<String, FeatureDefinition>,
    /// Raw info.json for advanced use cases.
    pub info: LeRobotInfo,
}

/// Per-episode metadata from `episodes.jsonl` (v2) or the episodes Parquet
/// files (v3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeInfo {
    /// Episode index.
    pub episode_index: usize,
    /// Task index for this episode.
    pub task_index: usize,
    /// Starting frame index in the global dataset.
    pub dataset_from_index: usize,
    /// Ending frame index (exclusive) in the global dataset.
    pub dataset_to_index: usize,
    /// Number of frames in this episode.
    pub length: usize,
    /// Video timestamps (if applicable).
    #[serde(default)]
    pub video_timestamps: HashMap<String, VideoTimestamps>,
    /// Data chunk index (v3 format).
    #[serde(default)]
    pub data_chunk_index: usize,
    /// Data file index within chunk (v3 format).
    #[serde(default)]
    pub data_file_index: usize,
}

/// Video timestamp and file info for an episode's camera.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VideoTimestamps {
    /// Time (seconds) of the episode's first frame within the video file.
    pub from_timestamp: f64,
    /// Time (seconds) just past the episode's last frame within the video file.
    pub to_timestamp: f64,
    /// Video chunk index (may differ from data chunk index).
    pub chunk_index: usize,
    /// Video file index within chunk (may differ per camera).
    pub file_index: usize,
}

/// Feature definition from info.json (features without a `dtype` are
/// omitted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDefinition {
    /// Data type, e.g. `"float32"` or `"video"`.
    pub dtype: String,
    /// Per-row shape (empty if not given).
    pub shape: Vec<usize>,
    /// Optional per-dimension names.
    #[serde(default)]
    pub names: Option<serde_json::Value>,
}

/// Raw episode entry from episodes.jsonl.
#[derive(Debug, Deserialize)]
struct RawEpisodeEntry {
    episode_index: usize,
    #[serde(default)]
    task_index: usize,
    #[serde(alias = "from")]
    dataset_from_index: Option<usize>,
    #[serde(alias = "to")]
    dataset_to_index: Option<usize>,
    length: Option<usize>,
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

/// Raw task entry from tasks.jsonl.
#[derive(Debug, Deserialize)]
struct RawTaskEntry {
    task_index: usize,
    task: String,
}

impl LazyDatasetMetadata {
    /// Load metadata from a LeRobot dataset directory.
    ///
    /// This reads the metadata files (info.json, episodes.jsonl or the
    /// episodes Parquet files, tasks.jsonl) and does not decode any video.
    /// For v3 datasets it also reads the `episode_index` column of every data
    /// Parquet file to locate episodes.
    ///
    /// If no episode metadata exists, episodes are synthesized from
    /// `total_episodes` and `total_frames` assuming equal lengths, which is
    /// only an estimate.
    ///
    /// # Errors
    ///
    /// Returns an error if `meta/info.json` is missing or malformed, or if an
    /// episodes or tasks file cannot be parsed.
    pub fn load(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let meta_dir = root.join("meta");

        // Load info.json (required)
        let info_path = meta_dir.join("info.json");
        let info = Self::load_info(&info_path)?;

        // Load episodes metadata
        // v3 format: meta/episodes/chunk-XXX/file-XXX.parquet
        // v2 format: meta/episodes.jsonl
        let episodes_parquet_dir = meta_dir.join("episodes");
        let episodes_jsonl_path = meta_dir.join("episodes.jsonl");

        let mut episodes = if episodes_parquet_dir.exists() && episodes_parquet_dir.is_dir() {
            Self::load_episodes_from_parquet(&episodes_parquet_dir, &info)?
        } else if episodes_jsonl_path.exists() {
            Self::load_episodes(&episodes_jsonl_path, &info)?
        } else {
            // Fall back to generating episode info from total_episodes/total_frames
            Self::generate_episode_info(&info)?
        };

        // For v3 datasets, the metadata file_index may be incorrect.
        // Scan actual data files to build correct episode-to-file mapping.
        if info.is_v3() {
            Self::fix_episode_file_indices(&root, &mut episodes)?;
        }

        // Load tasks.jsonl (optional)
        let tasks_path = meta_dir.join("tasks.jsonl");
        let tasks = if tasks_path.exists() {
            Self::load_tasks(&tasks_path)?
        } else {
            Vec::new()
        };

        // Extract feature definitions
        let features = info
            .features
            .as_ref()
            .map(|f| {
                f.iter()
                    .filter_map(|(k, v)| {
                        Some((
                            k.clone(),
                            FeatureDefinition {
                                dtype: v.dtype.clone()?,
                                shape: v.shape.clone().unwrap_or_default(),
                                names: v.names.clone(),
                            },
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Extract video keys
        let video_keys: Vec<String> = info.video_features().into_iter().map(|(k, _)| k).collect();

        let name = root
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string());

        Ok(Self {
            root,
            name,
            robot_type: info.robot_type.clone(),
            fps: info.fps.unwrap_or(10.0),
            total_episodes: info.total_episodes.unwrap_or(episodes.len()),
            total_frames: info
                .total_frames
                .unwrap_or_else(|| episodes.iter().map(|e| e.length).sum()),
            action_dim: info.action_dim(),
            state_dim: info.state_dim(),
            camera_names: info.camera_names(),
            video_keys,
            episodes,
            tasks,
            is_v3: info.is_v3(),
            data_path_template: info.data_path.clone(),
            video_path_template: info.video_path.clone(),
            features,
            info,
        })
    }

    /// Load info.json.
    fn load_info(path: &Path) -> Result<LeRobotInfo> {
        let content = fs::read_to_string(path).map_err(|e| {
            DataCoreError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read {}: {}", path.display(), e),
            ))
        })?;
        serde_json::from_str(&content)
            .map_err(|e| DataCoreError::Other(format!("Failed to parse info.json: {}", e)))
    }

    /// Load episodes.jsonl.
    fn load_episodes(path: &Path, info: &LeRobotInfo) -> Result<Vec<EpisodeInfo>> {
        let file = fs::File::open(path).map_err(|e| {
            DataCoreError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to open {}: {}", path.display(), e),
            ))
        })?;
        let reader = BufReader::new(file);
        // (episode, whether dataset_from_index was present in the file)
        let mut episodes: Vec<(EpisodeInfo, bool)> = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                DataCoreError::Io(std::io::Error::new(
                    e.kind(),
                    format!(
                        "Failed to read line {} of episodes.jsonl: {}",
                        line_num + 1,
                        e
                    ),
                ))
            })?;

            if line.trim().is_empty() {
                continue;
            }

            let raw: RawEpisodeEntry = serde_json::from_str(&line).map_err(|e| {
                DataCoreError::Other(format!(
                    "Failed to parse episodes.jsonl line {}: {}",
                    line_num + 1,
                    e
                ))
            })?;

            // Parse video timestamps from extra fields
            let mut video_timestamps = HashMap::new();
            for key in info.video_features().iter().map(|(k, _)| k) {
                let from_key = format!("videos/{}/from_timestamp", key);
                let to_key = format!("videos/{}/to_timestamp", key);
                if let (Some(from), Some(to)) = (
                    raw.extra.get(&from_key).and_then(|v| v.as_f64()),
                    raw.extra.get(&to_key).and_then(|v| v.as_f64()),
                ) {
                    let chunk_key = format!("videos/{}/chunk_index", key);
                    let file_key = format!("videos/{}/file_index", key);
                    video_timestamps.insert(
                        key.clone(),
                        VideoTimestamps {
                            from_timestamp: from,
                            to_timestamp: to,
                            chunk_index: raw
                                .extra
                                .get(&chunk_key)
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as usize,
                            file_index: raw
                                .extra
                                .get(&file_key)
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as usize,
                        },
                    );
                }
            }

            let length = match raw.length {
                Some(length) => length,
                None => raw
                    .dataset_to_index
                    .unwrap_or(0)
                    .checked_sub(raw.dataset_from_index.unwrap_or(0))
                    .ok_or_else(|| {
                        DataCoreError::Other(format!(
                            "episodes.jsonl line {}: dataset_to_index is smaller than \
                             dataset_from_index",
                            line_num + 1
                        ))
                    })?,
            };

            let from = raw.dataset_from_index.unwrap_or(0);
            episodes.push((
                EpisodeInfo {
                    episode_index: raw.episode_index,
                    task_index: raw.task_index,
                    dataset_from_index: from,
                    dataset_to_index: raw.dataset_to_index.unwrap_or(from + length),
                    length,
                    video_timestamps,
                    data_chunk_index: 0,
                    data_file_index: raw.episode_index, // v2 format: one episode per file
                },
                raw.dataset_from_index.is_some(),
            ));
        }

        episodes.sort_by_key(|(e, _)| e.episode_index);

        // LeRobot v2.x episodes.jsonl files only store `length`. Global frame
        // ranges are then the running sum of the lengths in episode order.
        let mut next_from = 0;
        let episodes = episodes
            .into_iter()
            .map(|(mut ep, has_from)| {
                if !has_from {
                    ep.dataset_from_index = next_from;
                    ep.dataset_to_index = next_from + ep.length;
                }
                next_from = ep.dataset_to_index;
                ep
            })
            .collect();
        Ok(episodes)
    }

    /// Load episodes from parquet files (v3 format).
    /// Episodes are stored in meta/episodes/chunk-XXX/file-XXX.parquet
    fn load_episodes_from_parquet(
        episodes_dir: &Path,
        info: &LeRobotInfo,
    ) -> Result<Vec<EpisodeInfo>> {
        use arrow::array::{Array, ArrayRef, Float64Array, Int64Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let mut episodes = Vec::new();
        let video_keys: Vec<String> = info.video_features().into_iter().map(|(k, _)| k).collect();

        // Find and read all parquet files in the episodes directory
        for chunk_entry in std::fs::read_dir(episodes_dir)? {
            let chunk_entry = chunk_entry?;
            let chunk_path = chunk_entry.path();
            if !chunk_path.is_dir() {
                continue;
            }

            for file_entry in std::fs::read_dir(&chunk_path)? {
                let file_entry = file_entry?;
                let file_path = file_entry.path();
                if file_path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                    continue;
                }

                // Read parquet file using ParquetRecordBatchReaderBuilder
                let file = std::fs::File::open(&file_path)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
                    DataCoreError::Other(format!("Failed to read episodes parquet: {}", e))
                })?;
                let reader = builder.build().map_err(|e| {
                    DataCoreError::Other(format!("Failed to build parquet reader: {}", e))
                })?;

                for batch_result in reader {
                    let batch = batch_result.map_err(DataCoreError::Arrow)?;

                    // Helper to get Int64 column
                    let get_i64_col = |name: &str| -> Option<&Int64Array> {
                        batch
                            .column_by_name(name)
                            .and_then(|c: &ArrayRef| c.as_any().downcast_ref::<Int64Array>())
                    };
                    // Helper to get Float64 column
                    let get_f64_col = |name: &str| -> Option<&Float64Array> {
                        batch
                            .column_by_name(name)
                            .and_then(|c: &ArrayRef| c.as_any().downcast_ref::<Float64Array>())
                    };

                    let episode_index_col = get_i64_col("episode_index");
                    let length_col = get_i64_col("length");
                    let from_index_col = get_i64_col("dataset_from_index");
                    let to_index_col = get_i64_col("dataset_to_index");
                    let data_chunk_col = get_i64_col("data/chunk_index");
                    let data_file_col = get_i64_col("data/file_index");

                    for row in 0..batch.num_rows() {
                        let episode_index = episode_index_col
                            .and_then(|c| c.value(row).try_into().ok())
                            .unwrap_or(0usize);
                        let length = length_col
                            .and_then(|c| c.value(row).try_into().ok())
                            .unwrap_or(0usize);
                        let dataset_from_index = from_index_col
                            .and_then(|c| c.value(row).try_into().ok())
                            .unwrap_or(0usize);
                        let dataset_to_index = to_index_col
                            .and_then(|c| c.value(row).try_into().ok())
                            .unwrap_or(dataset_from_index + length);
                        let data_chunk_index = data_chunk_col
                            .and_then(|c| c.value(row).try_into().ok())
                            .unwrap_or(0usize);
                        let data_file_index = data_file_col
                            .and_then(|c| c.value(row).try_into().ok())
                            .unwrap_or(0usize);

                        // Extract video timestamps and per-camera file indices
                        let mut video_timestamps = HashMap::new();
                        for key in &video_keys {
                            let from_col_name = format!("videos/{}/from_timestamp", key);
                            let to_col_name = format!("videos/{}/to_timestamp", key);
                            let vchunk_col_name = format!("videos/{}/chunk_index", key);
                            let vfile_col_name = format!("videos/{}/file_index", key);

                            if let (Some(from_col), Some(to_col)) =
                                (get_f64_col(&from_col_name), get_f64_col(&to_col_name))
                            {
                                let vid_chunk = get_i64_col(&vchunk_col_name)
                                    .and_then(|c| c.value(row).try_into().ok())
                                    .unwrap_or(data_chunk_index);
                                let vid_file = get_i64_col(&vfile_col_name)
                                    .and_then(|c| c.value(row).try_into().ok())
                                    .unwrap_or(0);
                                video_timestamps.insert(
                                    key.clone(),
                                    VideoTimestamps {
                                        from_timestamp: from_col.value(row),
                                        to_timestamp: to_col.value(row),
                                        chunk_index: vid_chunk,
                                        file_index: vid_file,
                                    },
                                );
                            }
                        }

                        episodes.push(EpisodeInfo {
                            episode_index,
                            task_index: 0, // task_index not always in parquet
                            dataset_from_index,
                            dataset_to_index,
                            length,
                            video_timestamps,
                            data_chunk_index,
                            data_file_index,
                        });
                    }
                }
            }
        }

        episodes.sort_by_key(|e| e.episode_index);
        Ok(episodes)
    }

    /// Fix episode file indices by scanning actual data files.
    /// The metadata's data/file_index may not match the actual file layout.
    fn fix_episode_file_indices(root: &Path, episodes: &mut [EpisodeInfo]) -> Result<()> {
        use arrow::array::{Array, Int64Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let data_dir = root.join("data");
        if !data_dir.exists() {
            return Ok(());
        }

        // Build a map of episode_index -> (chunk_index, file_index)
        let mut episode_to_file: HashMap<usize, (usize, usize)> = HashMap::new();

        // Scan all chunk directories
        for chunk_entry in std::fs::read_dir(&data_dir)? {
            let chunk_entry = chunk_entry?;
            let chunk_path = chunk_entry.path();
            if !chunk_path.is_dir() {
                continue;
            }

            // Extract chunk index from directory name (e.g., "chunk-000")
            let chunk_name = chunk_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            let chunk_index: usize = chunk_name
                .strip_prefix("chunk-")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            // Scan parquet files in this chunk
            for file_entry in std::fs::read_dir(&chunk_path)? {
                let file_entry = file_entry?;
                let file_path = file_entry.path();
                if file_path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                    continue;
                }

                // Extract file index from filename (e.g., "file-000.parquet")
                let file_name = file_path.file_stem().and_then(|n| n.to_str()).unwrap_or("");
                let file_index: usize = file_name
                    .strip_prefix("file-")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);

                // Read just the episode_index column from the parquet file
                let file = std::fs::File::open(&file_path)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
                    DataCoreError::Other(format!("Failed to read data parquet: {}", e))
                })?;

                // Only read the episode_index column for efficiency
                let reader = builder.build().map_err(|e| {
                    DataCoreError::Other(format!("Failed to build parquet reader: {}", e))
                })?;

                for batch_result in reader {
                    let batch = batch_result.map_err(DataCoreError::Arrow)?;

                    if let Some(ep_col) = batch.column_by_name("episode_index") {
                        if let Some(ep_array) = ep_col.as_any().downcast_ref::<Int64Array>() {
                            for i in 0..ep_array.len() {
                                if let Ok(ep_idx) = ep_array.value(i).try_into() {
                                    let ep_idx: usize = ep_idx;
                                    episode_to_file
                                        .entry(ep_idx)
                                        .or_insert((chunk_index, file_index));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update episodes with correct file indices
        for ep in episodes.iter_mut() {
            if let Some(&(chunk_idx, file_idx)) = episode_to_file.get(&ep.episode_index) {
                ep.data_chunk_index = chunk_idx;
                ep.data_file_index = file_idx;
            }
        }

        Ok(())
    }

    /// Load tasks.jsonl.
    fn load_tasks(path: &Path) -> Result<Vec<String>> {
        let file = fs::File::open(path).map_err(|e| {
            DataCoreError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to open {}: {}", path.display(), e),
            ))
        })?;
        let reader = BufReader::new(file);
        let mut tasks: Vec<(usize, String)> = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let raw: RawTaskEntry = serde_json::from_str(&line)
                .map_err(|e| DataCoreError::Other(format!("Failed to parse tasks.jsonl: {}", e)))?;
            tasks.push((raw.task_index, raw.task));
        }

        tasks.sort_by_key(|(idx, _)| *idx);
        Ok(tasks.into_iter().map(|(_, task)| task).collect())
    }

    /// Generate episode info when episodes.jsonl is missing.
    fn generate_episode_info(info: &LeRobotInfo) -> Result<Vec<EpisodeInfo>> {
        let total_episodes = info.total_episodes.unwrap_or(0);
        let total_frames = info.total_frames.unwrap_or(0);

        if total_episodes == 0 {
            return Ok(Vec::new());
        }

        // Estimate frames per episode (uniform distribution)
        let frames_per_episode = total_frames / total_episodes;
        let remainder = total_frames % total_episodes;

        let mut episodes = Vec::with_capacity(total_episodes);
        let mut current_frame = 0;

        for ep_idx in 0..total_episodes {
            let length = frames_per_episode + if ep_idx < remainder { 1 } else { 0 };
            episodes.push(EpisodeInfo {
                episode_index: ep_idx,
                task_index: 0,
                dataset_from_index: current_frame,
                dataset_to_index: current_frame + length,
                length,
                video_timestamps: HashMap::new(),
                data_chunk_index: 0,
                data_file_index: ep_idx, // fallback: one file per episode
            });
            current_frame += length;
        }

        Ok(episodes)
    }

    /// Get the episode whose `episode_index` equals `index`.
    pub fn episode(&self, index: usize) -> Option<&EpisodeInfo> {
        // Fast path: LeRobot numbers episodes 0..n, so the position usually
        // equals the episode index.
        match self.episodes.get(index) {
            Some(ep) if ep.episode_index == index => Some(ep),
            _ => self
                .episodes
                .binary_search_by_key(&index, |ep| ep.episode_index)
                .ok()
                .map(|pos| &self.episodes[pos]),
        }
    }

    /// Find the episode whose `[dataset_from_index, dataset_to_index)` range
    /// contains the global frame index.
    pub fn episode_for_frame(&self, frame_index: usize) -> Option<&EpisodeInfo> {
        // Binary search for efficiency
        self.episodes
            .binary_search_by(|ep| {
                if frame_index < ep.dataset_from_index {
                    std::cmp::Ordering::Greater
                } else if frame_index >= ep.dataset_to_index {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()
            .and_then(|idx| self.episodes.get(idx))
    }

    /// Get the Parquet file path for an episode.
    ///
    /// Uses the `data_path` template from info.json when present, filling the
    /// v3 placeholders (`chunk_index`, `file_index`) from the episode metadata
    /// and the v2 placeholders (`episode_chunk`, `episode_index`) from the
    /// episode index and `chunks_size`. Without a template, the v2 default
    /// layout `data/chunk-{chunk:03}/episode_{index:06}.parquet` is used.
    pub fn get_data_file_path(&self, episode_index: usize) -> PathBuf {
        // Use the episode's stored chunk/file indices
        let ep = self.episode(episode_index);
        let (chunk_index, file_index) = ep
            .map(|e| (e.data_chunk_index, e.data_file_index))
            .unwrap_or((0, episode_index));
        let episode_chunk = episode_index / self.info.chunks_size();

        if let Some(template) = &self.data_path_template {
            let path = fill_path_template(
                template,
                &[
                    ("chunk_index", chunk_index),
                    ("file_index", file_index),
                    ("episode_chunk", episode_chunk),
                    ("episode_index", episode_index),
                    ("total_episodes", self.total_episodes),
                ],
                None,
            );
            self.root.join(path)
        } else {
            self.root
                .join("data")
                .join(format!("chunk-{:03}", episode_chunk))
                .join(format!("episode_{:06}.parquet", episode_index))
        }
    }

    /// Smallest `dataset_from_index` among episodes stored in the given data
    /// chunk (0 if there are none).
    pub fn chunk_start_frame(&self, chunk_index: usize) -> usize {
        self.episodes
            .iter()
            .filter(|ep| ep.data_chunk_index == chunk_index)
            .map(|ep| ep.dataset_from_index)
            .min()
            .unwrap_or(0)
    }

    /// Get the video file path for an episode and camera.
    ///
    /// Uses the `video_path` template from info.json when present. v3
    /// placeholders come from the episode's per-camera video metadata
    /// (falling back to the data chunk and file 0); v2 placeholders come from
    /// the episode index and `chunks_size`. Without a template, the v2
    /// default layout `videos/chunk-{chunk:03}/{key}/episode_{index:06}.mp4`
    /// is used.
    pub fn get_video_file_path(&self, episode_index: usize, video_key: &str) -> PathBuf {
        let episode_chunk = episode_index / self.info.chunks_size();
        if let Some(template) = &self.video_path_template {
            let ep = self.episode(episode_index);
            // Use per-camera video chunk/file indices from episode metadata.
            let (chunk_index, file_index) = ep
                .and_then(|e| e.video_timestamps.get(video_key))
                .map(|vt| (vt.chunk_index, vt.file_index))
                .unwrap_or_else(|| {
                    // Fallback to data chunk index if video-specific indices not available
                    (ep.map(|e| e.data_chunk_index).unwrap_or(0), 0)
                });
            let path = fill_path_template(
                template,
                &[
                    ("chunk_index", chunk_index),
                    ("file_index", file_index),
                    ("episode_chunk", episode_chunk),
                    ("episode_index", episode_index),
                ],
                Some(video_key),
            );
            self.root.join(path)
        } else {
            self.root
                .join("videos")
                .join(format!("chunk-{:03}", episode_chunk))
                .join(video_key)
                .join(format!("episode_{:06}.mp4", episode_index))
        }
    }

    /// Get the task description for a `task_index`.
    pub fn task(&self, index: usize) -> Option<&str> {
        self.tasks.get(index).map(|s| s.as_str())
    }
}

/// Fill a LeRobot path template such as
/// `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`.
///
/// Supports `{name}` and `{name:0Nd}` placeholders for the given integer
/// values, and `{video_key}` when `video_key` is provided. Unknown
/// placeholders are left unchanged.
fn fill_path_template(template: &str, values: &[(&str, usize)], video_key: Option<&str>) -> String {
    let mut out = String::with_capacity(template.len() + 16);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else {
            out.push_str(&rest[open..]);
            return out;
        };
        let placeholder = &after[..close];
        let (name, spec) = placeholder
            .split_once(':')
            .map_or((placeholder, None), |(n, s)| (n, Some(s)));
        let replacement = if name == "video_key" && spec.is_none() {
            video_key.map(str::to_string)
        } else {
            values
                .iter()
                .find(|(key, _)| *key == name)
                .and_then(|&(_, value)| match spec {
                    None => Some(value.to_string()),
                    Some(spec) => {
                        let width = spec.strip_prefix('0')?.strip_suffix('d')?;
                        let width: usize = width.parse().ok()?;
                        Some(format!("{value:0width$}"))
                    }
                })
        };
        match replacement {
            Some(r) => out.push_str(&r),
            None => {
                out.push('{');
                out.push_str(placeholder);
                out.push('}');
            }
        }
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_metadata(dir: &Path) {
        let meta_dir = dir.join("meta");
        fs::create_dir_all(&meta_dir).unwrap();

        // info.json
        let info = serde_json::json!({
            "codebase_version": "v3.0",
            "fps": 10,
            "robot_type": "panda",
            "total_episodes": 3,
            "total_frames": 300,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                "action": {"dtype": "float32", "shape": [7]},
                "observation.images.cam": {"dtype": "video", "shape": [256, 256, 3]}
            }
        });
        fs::write(
            meta_dir.join("info.json"),
            serde_json::to_string_pretty(&info).unwrap(),
        )
        .unwrap();

        // episodes.jsonl
        let mut episodes_file = fs::File::create(meta_dir.join("episodes.jsonl")).unwrap();
        for i in 0..3 {
            let ep = serde_json::json!({
                "episode_index": i,
                "task_index": 0,
                "dataset_from_index": i * 100,
                "dataset_to_index": (i + 1) * 100,
                "length": 100,
                "videos/observation.images.cam/from_timestamp": 0.0,
                "videos/observation.images.cam/to_timestamp": 10.0
            });
            writeln!(episodes_file, "{}", serde_json::to_string(&ep).unwrap()).unwrap();
        }

        // tasks.jsonl
        let mut tasks_file = fs::File::create(meta_dir.join("tasks.jsonl")).unwrap();
        writeln!(
            tasks_file,
            r#"{{"task_index": 0, "task": "Pick up the object"}}"#
        )
        .unwrap();
    }

    #[test]
    fn test_load_metadata() {
        let tmp = TempDir::new().unwrap();
        create_test_metadata(tmp.path());

        let meta = LazyDatasetMetadata::load(tmp.path()).unwrap();
        assert_eq!(meta.fps, 10.0);
        assert_eq!(meta.total_episodes, 3);
        assert_eq!(meta.total_frames, 300);
        assert_eq!(meta.action_dim, 7);
        assert!(meta.is_v3);
        assert_eq!(meta.camera_names, vec!["cam"]);
        assert_eq!(meta.episodes.len(), 3);
        assert_eq!(meta.tasks.len(), 1);
        assert_eq!(meta.tasks[0], "Pick up the object");
    }

    #[test]
    fn test_episode_for_frame() {
        let tmp = TempDir::new().unwrap();
        create_test_metadata(tmp.path());

        let meta = LazyDatasetMetadata::load(tmp.path()).unwrap();

        // Frame 50 should be in episode 0
        let ep = meta.episode_for_frame(50).unwrap();
        assert_eq!(ep.episode_index, 0);

        // Frame 150 should be in episode 1
        let ep = meta.episode_for_frame(150).unwrap();
        assert_eq!(ep.episode_index, 1);

        // Frame 250 should be in episode 2
        let ep = meta.episode_for_frame(250).unwrap();
        assert_eq!(ep.episode_index, 2);

        // Frame 300 should be out of range
        assert!(meta.episode_for_frame(300).is_none());
    }

    #[test]
    fn test_fill_path_template() {
        assert_eq!(
            fill_path_template(
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                &[("chunk_index", 2), ("file_index", 17)],
                Some("observation.images.top"),
            ),
            "videos/observation.images.top/chunk-002/file-017.mp4"
        );
        assert_eq!(
            fill_path_template(
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                &[("episode_chunk", 1), ("episode_index", 1234)],
                None,
            ),
            "data/chunk-001/episode_001234.parquet"
        );
        assert_eq!(
            fill_path_template("a/{unknown}/{episode_index}", &[("episode_index", 5)], None),
            "a/{unknown}/5"
        );
    }

    /// Write a LeRobot v2.1-style metadata directory: episodes.jsonl has only
    /// `episode_index`, `tasks`, and `length`, and info.json has v2 templates.
    fn create_v2_metadata(dir: &Path, lengths: &[usize], chunks_size: usize) {
        let meta_dir = dir.join("meta");
        fs::create_dir_all(&meta_dir).unwrap();
        let info = serde_json::json!({
            "codebase_version": "v2.1",
            "fps": 50,
            "total_episodes": lengths.len(),
            "total_frames": lengths.iter().sum::<usize>(),
            "chunks_size": chunks_size,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "action": {"dtype": "float32", "shape": [14]},
                "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]}
            }
        });
        fs::write(meta_dir.join("info.json"), info.to_string()).unwrap();
        let mut f = fs::File::create(meta_dir.join("episodes.jsonl")).unwrap();
        // Write out of order to check sorting.
        for (i, len) in lengths.iter().enumerate().rev() {
            writeln!(
                f,
                r#"{{"episode_index": {i}, "tasks": ["do it"], "length": {len}}}"#
            )
            .unwrap();
        }
    }

    #[test]
    fn test_v2_episodes_get_cumulative_frame_ranges() {
        // Regression: without dataset_from_index in episodes.jsonl every
        // episode started at global frame 0, so frames past the first episode
        // could not be resolved.
        let tmp = TempDir::new().unwrap();
        create_v2_metadata(tmp.path(), &[5, 3, 4], 1000);
        let meta = LazyDatasetMetadata::load(tmp.path()).unwrap();
        assert!(!meta.is_v3);
        let ranges: Vec<(usize, usize)> = meta
            .episodes
            .iter()
            .map(|e| (e.dataset_from_index, e.dataset_to_index))
            .collect();
        assert_eq!(ranges, vec![(0, 5), (5, 8), (8, 12)]);
        assert_eq!(meta.episode_for_frame(4).unwrap().episode_index, 0);
        assert_eq!(meta.episode_for_frame(5).unwrap().episode_index, 1);
        assert_eq!(meta.episode_for_frame(11).unwrap().episode_index, 2);
        assert!(meta.episode_for_frame(12).is_none());
    }

    #[test]
    fn test_v2_template_paths() {
        // Regression: v2 templates use {episode_chunk} and {episode_index},
        // which were left unsubstituted.
        let tmp = TempDir::new().unwrap();
        create_v2_metadata(tmp.path(), &[2, 2, 2], 2);
        let meta = LazyDatasetMetadata::load(tmp.path()).unwrap();
        assert_eq!(
            meta.get_data_file_path(2),
            tmp.path().join("data/chunk-001/episode_000002.parquet")
        );
        assert_eq!(
            meta.get_video_file_path(1, "observation.images.top"),
            tmp.path()
                .join("videos/chunk-000/observation.images.top/episode_000001.mp4")
        );
    }

    #[test]
    fn test_episode_lookup_with_gaps() {
        let tmp = TempDir::new().unwrap();
        let meta_dir = tmp.path().join("meta");
        fs::create_dir_all(&meta_dir).unwrap();
        fs::write(
            meta_dir.join("info.json"),
            r#"{"codebase_version": "v2.0", "fps": 10}"#,
        )
        .unwrap();
        fs::write(
            meta_dir.join("episodes.jsonl"),
            "{\"episode_index\": 0, \"length\": 4}\n{\"episode_index\": 2, \"length\": 6}\n",
        )
        .unwrap();
        let meta = LazyDatasetMetadata::load(tmp.path()).unwrap();
        assert_eq!(meta.episode(0).unwrap().length, 4);
        assert!(meta.episode(1).is_none());
        assert_eq!(meta.episode(2).unwrap().length, 6);
        assert!(meta.episode(3).is_none());
    }

    #[test]
    fn test_inconsistent_from_to_is_error() {
        // Regression: to < from with no length underflowed (debug panic).
        let tmp = TempDir::new().unwrap();
        let meta_dir = tmp.path().join("meta");
        fs::create_dir_all(&meta_dir).unwrap();
        fs::write(
            meta_dir.join("info.json"),
            r#"{"codebase_version": "v2.0"}"#,
        )
        .unwrap();
        fs::write(
            meta_dir.join("episodes.jsonl"),
            r#"{"episode_index": 0, "dataset_from_index": 10, "dataset_to_index": 4}"#,
        )
        .unwrap();
        assert!(LazyDatasetMetadata::load(tmp.path()).is_err());
    }

    #[test]
    fn test_get_file_paths() {
        let tmp = TempDir::new().unwrap();
        create_test_metadata(tmp.path());

        let meta = LazyDatasetMetadata::load(tmp.path()).unwrap();

        let data_path = meta.get_data_file_path(0);
        assert!(data_path.to_string_lossy().contains("chunk-000"));
        assert!(data_path.to_string_lossy().contains("file-000"));

        let video_path = meta.get_video_file_path(0, "observation.images.cam");
        assert!(video_path
            .to_string_lossy()
            .contains("observation.images.cam"));
        assert!(video_path.to_string_lossy().contains("chunk-000"));
    }
}
