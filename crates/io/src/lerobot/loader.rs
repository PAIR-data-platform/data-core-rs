//! Eager loader that reads a whole LeRobot dataset into a [`Dataset`].

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use arrow_array::*;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use data_core::{DataCoreError, Dataset, Episode, EpisodeMetadata, Result};

use super::metadata::LeRobotInfo;
use super::video::VideoDecoder;

/// Eagerly loads a local LeRobot dataset into an in-memory [`Dataset`].
///
/// Every `.parquet` file under `data/` is read (in sorted path order) and
/// split into episodes by the `episode_index` column. `list<float32>` columns
/// whose rows all have the same length are converted to
/// `FixedSizeList<Float32>`.
///
/// For v3 datasets (see [`LeRobotInfo::is_v3`]) video frames are decoded and
/// added as JPEG columns `observation.image.{camera}`. This loader assumes one
/// video file per chunk, `videos/{key}/chunk-{NNN}/file-000.mp4`, starting at
/// the first frame of `data/chunk-{NNN}/file-000.parquet`; datasets whose
/// videos are split into several files per chunk should be read with
/// [`crate::LazyDataset`], which follows the per-episode video metadata.
/// Missing video files are skipped with a warning. v2 videos are not loaded.
///
/// LeRobot v2 layout:
/// ```text
/// dataset/
/// ├── meta/
/// │   ├── info.json
/// │   ├── episodes.jsonl (optional)
/// │   └── tasks.jsonl (optional)
/// └── data/
///     └── chunk-000/
///         ├── episode_000000.parquet
///         ├── episode_000001.parquet
///         └── ...
/// ```
pub struct LeRobotLoader {
    root: PathBuf,
}

impl LeRobotLoader {
    /// Create a loader for the dataset rooted at `path` (the directory that
    /// contains `meta/` and `data/`).
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { root: path.into() }
    }

    /// Load the full dataset: metadata and all episodes.
    ///
    /// The returned dataset's schema is the canonical schema built from
    /// `info.json`; the episodes keep the columns found on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if `meta/info.json` or `data/` is missing, if a
    /// Parquet file cannot be read or has no `Int64` `episode_index` column,
    /// or if video decoding fails.
    pub fn load(&self) -> Result<Dataset> {
        let info = self.load_info()?;
        let metadata = info.to_dataset_metadata(self.root.to_string_lossy().to_string());
        let schema_config = metadata.to_schema_config();
        let schema = Arc::new(data_core::build_episode_schema(&schema_config));

        let parquet_files = self.discover_parquet_files()?;
        tracing::info!(
            "Found {} parquet files in {}",
            parquet_files.len(),
            self.root.display()
        );

        // For v3 datasets, pre-compute the global start frame for each chunk.
        // Each chunk has ONE video file covering all episodes in the chunk.
        // The video frame number for a given parquet row = global_index - chunk_start_frame.
        let chunk_start_frames: HashMap<usize, i64> = if info.is_v3() {
            self.compute_chunk_start_frames(&parquet_files)?
        } else {
            HashMap::new()
        };

        let mut episodes = Vec::new();
        for parquet_path in &parquet_files {
            let batch = self.read_parquet_file(parquet_path)?;
            // For v3 datasets, decode video frames and inject them as image columns.
            let batch = if info.is_v3() {
                let (chunk_index, _) = Self::parse_chunk_file_indices(parquet_path);
                let chunk_start = chunk_start_frames.get(&chunk_index).copied().unwrap_or(0);
                self.inject_video_frames(batch, parquet_path, &info, chunk_start)?
            } else {
                batch
            };
            let ep_episodes = self.split_into_episodes(batch, &schema)?;
            episodes.extend(ep_episodes);
        }

        tracing::info!(
            "Loaded {} episodes ({} total steps)",
            episodes.len(),
            episodes.iter().map(|e| e.num_steps()).sum::<usize>()
        );

        Dataset::new(episodes, metadata, schema)
    }

    /// Load just the metadata (`meta/info.json`).
    pub fn load_info(&self) -> Result<LeRobotInfo> {
        let info_path = self.root.join("meta").join("info.json");
        if !info_path.exists() {
            return Err(DataCoreError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("info.json not found at {}", info_path.display()),
            )));
        }
        let content = fs::read_to_string(&info_path)?;
        let info: LeRobotInfo = serde_json::from_str(&content)?;
        Ok(info)
    }

    /// Discover all .parquet files under data/ directory.
    fn discover_parquet_files(&self) -> Result<Vec<PathBuf>> {
        let data_dir = self.root.join("data");
        if !data_dir.exists() {
            return Err(DataCoreError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("data/ directory not found at {}", data_dir.display()),
            )));
        }

        let mut files = Vec::new();
        Self::find_parquet_recursive(&data_dir, &mut files)?;
        files.sort();
        Ok(files)
    }

    fn find_parquet_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::find_parquet_recursive(&path, files)?;
            } else if path.extension().is_some_and(|ext| ext == "parquet") {
                files.push(path);
            }
        }
        Ok(())
    }

    /// Normalize variable-length `list<float32>` columns to `FixedSizeList<float32, N>`.
    ///
    /// LeRobot v3 datasets serialize action/state columns as `list<element: float>` even
    /// though every row always has the same length.  Converting to `FixedSizeList` makes
    /// the data compatible with all built-in augmentations.
    fn normalize_list_columns(batch: RecordBatch) -> Result<RecordBatch> {
        use arrow_array::cast::AsArray;

        let mut new_fields: Vec<Field> = Vec::new();
        let mut new_columns: Vec<Arc<dyn arrow_array::Array>> = Vec::new();

        for (i, field) in batch.schema().fields().iter().enumerate() {
            let col = batch.column(i);
            if let DataType::List(inner_field) = field.data_type() {
                if *inner_field.data_type() == DataType::Float32 && !col.is_empty() {
                    let list_arr = col.as_list::<i32>();
                    let first_len = list_arr.value_length(0);
                    let all_same =
                        (1..list_arr.len()).all(|j| list_arr.value_length(j) == first_len);

                    if all_same && first_len > 0 {
                        // Restrict the child values to the rows of this
                        // array; a sliced ListArray shares a larger child.
                        let start = list_arr.value_offsets()[0] as usize;
                        let values = list_arr
                            .values()
                            .slice(start, list_arr.len() * first_len as usize);
                        let inner_f = Arc::new(Field::new("element", DataType::Float32, false));
                        let fixed =
                            FixedSizeListArray::try_new(inner_f.clone(), first_len, values, None)
                                .map_err(DataCoreError::Arrow)?;
                        let new_type = DataType::FixedSizeList(inner_f, first_len);
                        new_fields.push(Field::new(field.name(), new_type, field.is_nullable()));
                        new_columns.push(Arc::new(fixed));
                        continue;
                    }
                }
            }
            new_fields.push(field.as_ref().clone());
            new_columns.push(col.clone());
        }

        let new_schema = Arc::new(Schema::new(new_fields));
        RecordBatch::try_new(new_schema, new_columns).map_err(DataCoreError::Arrow)
    }

    /// Read a single Parquet file into one RecordBatch.
    ///
    /// Row groups are concatenated and uniform-length `list<float32>`
    /// columns are converted to `FixedSizeList<Float32>`. Used by
    /// [`crate::LazyDataset`] to read individual files on demand.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or decoded, or if it
    /// contains no record batches.
    pub fn read_parquet_file(&self, path: &Path) -> Result<RecordBatch> {
        let file = fs::File::open(path).map_err(|e| {
            DataCoreError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to open {}: {}", path.display(), e),
            ))
        })?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| DataCoreError::Other(format!("Parquet read error: {e}")))?;

        let reader = builder
            .build()
            .map_err(|e| DataCoreError::Other(format!("Parquet reader build error: {e}")))?;

        // Collect all batches and concatenate
        let batches: Vec<RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(DataCoreError::Arrow)?;

        if batches.is_empty() {
            return Err(DataCoreError::Other(format!(
                "No data in parquet file: {}",
                path.display()
            )));
        }

        // Concatenate all batches into one, then normalize variable-length lists
        let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)
            .map_err(DataCoreError::Arrow)?;
        Self::normalize_list_columns(batch)
    }

    /// Split a RecordBatch (which may contain multiple episodes) into individual Episodes.
    /// Groups rows by `episode_index` column.
    fn split_into_episodes(
        &self,
        batch: RecordBatch,
        _target_schema: &Arc<Schema>,
    ) -> Result<Vec<Episode>> {
        // Try to find episode_index column
        let ep_idx_col = batch
            .column_by_name("episode_index")
            .ok_or_else(|| DataCoreError::ColumnNotFound("episode_index".to_string()))?;

        let ep_indices = ep_idx_col
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                DataCoreError::SchemaMismatch("episode_index column is not Int64".to_string())
            })?;

        if ep_indices.is_empty() {
            return Ok(Vec::new());
        }

        // Group row ranges by episode_index.
        // LeRobot parquet files are typically one episode per file,
        // but we handle the general case.
        let mut episode_ranges: Vec<(i64, usize, usize)> = Vec::new();
        let mut current_ep = ep_indices.value(0);
        let mut start = 0;

        for i in 1..ep_indices.len() {
            let ep = ep_indices.value(i);
            if ep != current_ep {
                episode_ranges.push((current_ep, start, i));
                current_ep = ep;
                start = i;
            }
        }
        episode_ranges.push((current_ep, start, ep_indices.len()));

        let mut episodes = Vec::new();
        for (ep_index, start, end) in episode_ranges {
            let sliced = batch.slice(start, end - start);
            let metadata = EpisodeMetadata {
                episode_index: ep_index,
                task: None,
                length: end - start,
            };
            episodes.push(Episode::new(sliced, metadata)?);
        }

        Ok(episodes)
    }

    /// For v3 datasets: decode video frames for each camera and inject them as
    /// `LargeBinaryArray` columns named `observation.image.{camera_name}`.
    ///
    /// In LeRobot v3 each chunk has a SINGLE video file per camera that contains all
    /// frames for every episode in that chunk.  The file is always named `file-000.mp4`.
    /// The Parquet `index` column gives each row's global frame number; subtracting
    /// `chunk_start_frame` gives the 0-based position within the video file.
    ///
    /// If a video file is absent (e.g. not downloaded), a warning is logged and the
    /// camera is skipped, so image augmentations do nothing for that camera.
    fn inject_video_frames(
        &self,
        batch: RecordBatch,
        parquet_path: &Path,
        info: &LeRobotInfo,
        chunk_start_frame: i64,
    ) -> Result<RecordBatch> {
        let (chunk_index, _) = Self::parse_chunk_file_indices(parquet_path);

        // Determine the start frame within the video file for this batch.
        let batch_start_frame = Self::get_batch_start_frame(&batch)?;
        let video_start_frame = batch_start_frame - chunk_start_frame;
        let num_rows = batch.num_rows();

        let fps = info.fps.unwrap_or(10.0);

        let mut current_batch = batch;

        for (feature_key, _feature_info) in info.video_features() {
            // Derive the short camera name used by augmentations, stripping either prefix.
            let camera_name = feature_key
                .strip_prefix("observation.images.")
                .or_else(|| feature_key.strip_prefix("observation.image."))
                .unwrap_or(&feature_key)
                .to_string();

            // Each chunk has ONE video file (file-000.mp4) covering all episodes in that chunk.
            let video_path = self.root.join(format!(
                "videos/{}/chunk-{:03}/file-000.mp4",
                feature_key, chunk_index
            ));

            if !video_path.exists() {
                tracing::warn!(
                    "Video file not found (skipping camera '{}'): {}",
                    camera_name,
                    video_path.display()
                );
                continue;
            }

            tracing::debug!(
                "Decoding {} frames starting at frame {} from '{}'",
                num_rows,
                video_start_frame,
                video_path.display()
            );

            let frames = VideoDecoder::decode_range(&video_path, video_start_frame, num_rows, fps)?;

            if frames.len() != num_rows {
                return Err(DataCoreError::Other(format!(
                    "Frame count mismatch for '{}': decoded {} frames but parquet has {} rows \
                     (video_start_frame={}, chunk_start_frame={})",
                    video_path.display(),
                    frames.len(),
                    num_rows,
                    video_start_frame,
                    chunk_start_frame,
                )));
            }

            let col_name = format!("observation.image.{camera_name}");
            let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
            let image_col = Arc::new(LargeBinaryArray::from(refs));

            // Append the new column to the batch.
            let mut fields: Vec<Field> = current_batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect();
            fields.push(Field::new(&col_name, DataType::LargeBinary, false));
            let new_schema = Arc::new(Schema::new(fields));

            let mut columns: Vec<Arc<dyn arrow_array::Array>> = current_batch.columns().to_vec();
            columns.push(image_col);

            current_batch =
                RecordBatch::try_new(new_schema, columns).map_err(DataCoreError::Arrow)?;
        }

        Ok(current_batch)
    }

    /// Get the minimum value of the `index` column in a batch.
    /// This is the global frame number of the first row in the batch.
    fn get_batch_start_frame(batch: &RecordBatch) -> Result<i64> {
        use arrow_array::cast::AsArray;
        let col = batch
            .column_by_name("index")
            .ok_or_else(|| DataCoreError::ColumnNotFound("index".to_string()))?;
        // The `index` column may be Int32 or Int64 depending on the dataset.
        if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
            Ok(arrow::compute::min(arr).unwrap_or(0))
        } else if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
            Ok(arrow::compute::min(arr).unwrap_or(0) as i64)
        } else {
            // Other integer types: cast to Int64. (The previous fallback,
            // `as_primitive::<Int64Type>()`, always panicked here because
            // Int64 was already handled above.)
            if !col.data_type().is_integer() {
                return Err(DataCoreError::SchemaMismatch(format!(
                    "'index' column has type {}, expected an integer type",
                    col.data_type()
                )));
            }
            let cast = arrow::compute::cast(col, &DataType::Int64)?;
            Ok(
                arrow::compute::min(cast.as_primitive::<arrow_array::types::Int64Type>())
                    .unwrap_or(0),
            )
        }
    }

    /// Pre-compute the global start frame for every chunk found in `parquet_files`.
    ///
    /// In v3 format there is one video file per chunk. The "chunk start frame" is the
    /// minimum global `index` value in `file-000.parquet` of that chunk, which
    /// corresponds to video frame 0 in `file-000.mp4`.
    fn compute_chunk_start_frames(&self, parquet_files: &[PathBuf]) -> Result<HashMap<usize, i64>> {
        let mut map: HashMap<usize, i64> = HashMap::new();
        for path in parquet_files {
            let (chunk_index, file_index) = Self::parse_chunk_file_indices(path);
            if file_index == 0 && !map.contains_key(&chunk_index) {
                let batch = self.read_parquet_file(path)?;
                let start = Self::get_batch_start_frame(&batch)?;
                map.insert(chunk_index, start);
            }
        }
        Ok(map)
    }

    /// Parse chunk and file indices from a v3 parquet path.
    ///
    /// Expected format: `{root}/data/chunk-{NNN}/file-{NNN}.parquet`
    /// Falls back to `(0, 0)` if the path doesn't match the pattern.
    fn parse_chunk_file_indices(parquet_path: &Path) -> (usize, usize) {
        let file_index = parquet_path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("file-"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        let chunk_index = parquet_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("chunk-"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        (chunk_index, file_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::{DataType, Field, Schema};
    use std::fs;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn create_test_dataset(dir: &Path, num_episodes: usize, steps_per_episode: usize) {
        // Create directory structure
        let meta_dir = dir.join("meta");
        let data_dir = dir.join("data").join("chunk-000");
        fs::create_dir_all(&meta_dir).unwrap();
        fs::create_dir_all(&data_dir).unwrap();

        // Write info.json
        let info = serde_json::json!({
            "codebase_version": "v2.0",
            "fps": 10,
            "robot_type": "test_robot",
            "total_episodes": num_episodes,
            "total_frames": num_episodes * steps_per_episode,
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": null
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [14],
                    "names": null
                }
            }
        });
        fs::write(
            meta_dir.join("info.json"),
            serde_json::to_string_pretty(&info).unwrap(),
        )
        .unwrap();

        // Write parquet files (one per episode)
        for ep_idx in 0..num_episodes {
            let n = steps_per_episode;
            let schema = Arc::new(Schema::new(vec![
                Field::new("frame_index", DataType::Int64, false),
                Field::new("episode_index", DataType::Int64, false),
                Field::new("timestamp", DataType::Float64, false),
                Field::new(
                    "action",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, false)),
                        7,
                    ),
                    false,
                ),
                Field::new(
                    "observation.state",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, false)),
                        14,
                    ),
                    true,
                ),
            ]));

            let frame_indices = Int64Array::from((0..n as i64).collect::<Vec<_>>());
            let ep_indices = Int64Array::from(vec![ep_idx as i64; n]);
            let timestamps =
                Float64Array::from((0..n).map(|i| i as f64 / 10.0).collect::<Vec<_>>());
            let action_values = Float32Array::from(vec![0.1f32; n * 7]);
            let actions = FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                7,
                Arc::new(action_values),
                None,
            )
            .unwrap();
            let state_values = Float32Array::from(vec![0.5f32; n * 14]);
            let states = FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                14,
                Arc::new(state_values),
                None,
            )
            .unwrap();

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(frame_indices),
                    Arc::new(ep_indices),
                    Arc::new(timestamps),
                    Arc::new(actions),
                    Arc::new(states),
                ],
            )
            .unwrap();

            let path = data_dir.join(format!("episode_{:06}.parquet", ep_idx));
            let file = fs::File::create(&path).unwrap();
            let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }
    }

    #[test]
    fn test_load_info() {
        let tmp = TempDir::new().unwrap();
        create_test_dataset(tmp.path(), 3, 10);

        let loader = LeRobotLoader::new(tmp.path());
        let info = loader.load_info().unwrap();
        assert_eq!(info.fps, Some(10.0));
        assert_eq!(info.robot_type, Some("test_robot".to_string()));
        assert_eq!(info.total_episodes, Some(3));
    }

    #[test]
    fn test_discover_parquet_files() {
        let tmp = TempDir::new().unwrap();
        create_test_dataset(tmp.path(), 5, 10);

        let loader = LeRobotLoader::new(tmp.path());
        let files = loader.discover_parquet_files().unwrap();
        assert_eq!(files.len(), 5);
    }

    #[test]
    fn test_load_full_dataset() {
        let tmp = TempDir::new().unwrap();
        create_test_dataset(tmp.path(), 3, 10);

        let loader = LeRobotLoader::new(tmp.path());
        let dataset = loader.load().unwrap();
        assert_eq!(dataset.num_episodes(), 3);
        assert_eq!(dataset.num_steps(), 30);

        // Check first episode
        let ep = dataset.episode(0).unwrap();
        assert_eq!(ep.num_steps(), 10);
        assert_eq!(ep.metadata().episode_index, 0);
    }

    #[test]
    fn test_batch_start_frame_other_integer_types() {
        // Regression: a non-Int32/Int64 `index` column panicked.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "index",
            DataType::UInt32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(UInt32Array::from(vec![7u32, 5, 9]))])
                .unwrap();
        assert_eq!(LeRobotLoader::get_batch_start_frame(&batch).unwrap(), 5);

        let schema = Arc::new(Schema::new(vec![Field::new(
            "index",
            DataType::Utf8,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["a"]))]).unwrap();
        assert!(LeRobotLoader::get_batch_start_frame(&batch).is_err());
    }

    #[test]
    fn test_split_into_episodes_groups_rows() {
        let loader = LeRobotLoader::new("/nonexistent");
        let schema = Arc::new(Schema::new(vec![Field::new(
            "episode_index",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![4, 4, 5, 5, 5, 4]))],
        )
        .unwrap();
        let eps = loader.split_into_episodes(batch, &schema).unwrap();
        let summary: Vec<(i64, usize)> = eps
            .iter()
            .map(|e| (e.metadata().episode_index, e.num_steps()))
            .collect();
        assert_eq!(summary, vec![(4, 2), (5, 3), (4, 1)]);

        // Regression: an empty batch panicked on `value(0)`.
        let empty = RecordBatch::new_empty(schema.clone());
        assert!(loader
            .split_into_episodes(empty, &schema)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_normalize_list_columns_sliced_input() {
        use arrow_array::types::Float32Type;
        let list = ListArray::from_iter_primitive::<Float32Type, _, _>(vec![
            Some(vec![Some(0.0), Some(1.0)]),
            Some(vec![Some(2.0), Some(3.0)]),
            Some(vec![Some(4.0), Some(5.0)]),
        ]);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "action",
            list.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(list)]).unwrap();
        // Slice off the first row: the list's child array still holds it.
        let normalized = LeRobotLoader::normalize_list_columns(batch.slice(1, 2)).unwrap();
        let fixed = normalized
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(fixed.len(), 2);
        let values = fixed
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(values.values().to_vec(), vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_missing_info_json() {
        let tmp = TempDir::new().unwrap();
        let loader = LeRobotLoader::new(tmp.path());
        assert!(loader.load_info().is_err());
    }
}
