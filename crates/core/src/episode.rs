//! The [`Episode`] type: one trajectory stored as an Arrow `RecordBatch`.
//!
//! Column naming conventions used across the workspace:
//!
//! | Column | Arrow type | Meaning |
//! |---|---|---|
//! | `action` | `FixedSizeList<Float32>` | action vector per step |
//! | `observation.state` | `FixedSizeList<Float32>` | proprioceptive state |
//! | `observation.image.{cam}` | `LargeBinary` | encoded (JPEG) frame |
//! | `observation.image.{cam}.rgb` | `FixedSizeList<UInt8>` | raw HWC RGB pixels |
//! | `observation.image.{cam}.width` / `.height` | `Int64` | size of the `.rgb` frames |
//! | `episode_index`, `reward`, `done`, `timestamp` | scalar | per-step metadata |

use arrow::record_batch::RecordBatch;
use arrow_array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Int64Array, LargeBinaryArray, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::error::{DataCoreError, Result};

/// A single episode (trajectory) in a robotics dataset.
///
/// Backed by an Arrow `RecordBatch` where each row is a timestep.
/// Columns represent actions, observations, states, rewards, etc.
///
/// The Arrow representation enables:
/// - Zero-copy sharing with Python via the C Data Interface
/// - Efficient columnar operations on specific fields
/// - Native Parquet read/write
#[derive(Debug, Clone)]
pub struct Episode {
    /// The underlying Arrow RecordBatch. Each row = one timestep.
    data: RecordBatch,
    /// Episode-level metadata (not per-step).
    metadata: EpisodeMetadata,
}

/// Episode-level metadata that doesn't fit in the per-step RecordBatch.
#[derive(Debug, Clone, Default)]
pub struct EpisodeMetadata {
    /// Unique episode index within the dataset.
    pub episode_index: i64,
    /// Human-readable task description (if available).
    pub task: Option<String>,
    /// Total number of steps in this episode.
    pub length: usize,
}

impl Episode {
    /// Create a new Episode from an Arrow RecordBatch and metadata.
    ///
    /// # Errors
    ///
    /// Returns [`DataCoreError::InvalidEpisode`] if `data` has no rows.
    pub fn new(data: RecordBatch, metadata: EpisodeMetadata) -> Result<Self> {
        if data.num_rows() == 0 {
            return Err(DataCoreError::InvalidEpisode(
                "Episode must have at least one timestep".to_string(),
            ));
        }

        let episode = Self { data, metadata };
        Ok(episode)
    }

    /// Create an Episode from a RecordBatch, inferring metadata from the data.
    ///
    /// The episode index is read from the first row of an `Int64`
    /// `episode_index` column if present, and is 0 otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`DataCoreError::InvalidEpisode`] if `data` has no rows.
    pub fn from_record_batch(data: RecordBatch) -> Result<Self> {
        let length = data.num_rows();

        // Try to extract episode_index from the first row. An empty batch is
        // rejected by `Self::new` below; guard here so that `value(0)` cannot
        // index out of bounds first.
        let episode_index = data
            .column_by_name("episode_index")
            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
            .filter(|arr| !arr.is_empty())
            .map(|arr| arr.value(0))
            .unwrap_or(0);

        let metadata = EpisodeMetadata {
            episode_index,
            task: None,
            length,
        };

        Self::new(data, metadata)
    }

    /// Get the underlying Arrow RecordBatch (zero-copy).
    pub fn data(&self) -> &RecordBatch {
        &self.data
    }

    /// Consume the Episode and return the underlying RecordBatch.
    pub fn into_record_batch(self) -> RecordBatch {
        self.data
    }

    /// Get the episode metadata.
    pub fn metadata(&self) -> &EpisodeMetadata {
        &self.metadata
    }

    /// Number of timesteps in this episode.
    pub fn num_steps(&self) -> usize {
        self.data.num_rows()
    }

    /// Get the Arrow schema of this episode.
    pub fn schema(&self) -> Arc<Schema> {
        self.data.schema()
    }

    /// Get a column by name as a generic Arrow Array.
    ///
    /// # Errors
    ///
    /// Returns [`DataCoreError::ColumnNotFound`] if no column has that name.
    pub fn column(&self, name: &str) -> Result<&Arc<dyn Array>> {
        let idx = self
            .data
            .schema()
            .index_of(name)
            .map_err(|_| DataCoreError::ColumnNotFound(name.to_string()))?;
        Ok(self.data.column(idx))
    }

    /// Get the `action` column as a `FixedSizeListArray` (of `Float32` values).
    pub fn actions(&self) -> Result<&FixedSizeListArray> {
        let col = self.column("action")?;
        col.as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                DataCoreError::SchemaMismatch("'action' column is not a FixedSizeListArray".into())
            })
    }

    /// Get the encoded image column `observation.image.{camera_name}`.
    pub fn image(&self, camera_name: &str) -> Result<&LargeBinaryArray> {
        let col_name = format!("observation.image.{camera_name}");
        let col = self.column(&col_name)?;
        col.as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| {
                DataCoreError::SchemaMismatch(format!("'{col_name}' is not a LargeBinaryArray"))
            })
    }

    /// Get the `reward` column.
    pub fn rewards(&self) -> Result<&Float32Array> {
        let col = self.column("reward")?;
        col.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
            DataCoreError::SchemaMismatch("'reward' column is not a Float32Array".into())
        })
    }

    /// Get the `done` (termination flag) column.
    pub fn dones(&self) -> Result<&BooleanArray> {
        let col = self.column("done")?;
        col.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
            DataCoreError::SchemaMismatch("'done' column is not a BooleanArray".into())
        })
    }

    /// Replace the underlying RecordBatch (e.g., after augmentation).
    ///
    /// Keeps the episode index and task, and updates the length.
    pub fn with_data(self, data: RecordBatch) -> Result<Self> {
        let mut metadata = self.metadata;
        metadata.length = data.num_rows();
        Self::new(data, metadata)
    }

    /// Replace a single column by name, returning a new Episode.
    ///
    /// The new column must have the same data type and length as the existing
    /// column. Use [`Episode::replace_column`] if the type may change (e.g.,
    /// after a crop changes the `FixedSizeList` length).
    pub fn with_column(&self, name: &str, new_col: Arc<dyn Array>) -> Result<Self> {
        let schema = self.data.schema();
        let idx = schema
            .index_of(name)
            .map_err(|_| DataCoreError::ColumnNotFound(name.to_string()))?;

        // Keep the schema when the type is unchanged. When it differs only
        // in list child field names or nullability (Parquet names list
        // children `element`, Arrow builders name them `item`), take the new
        // column's type. Any other mismatch is rejected by
        // RecordBatch::try_new below.
        let old_type = schema.field(idx).data_type();
        let schema = if old_type == new_col.data_type()
            || !same_value_layout(old_type, new_col.data_type())
        {
            schema
        } else {
            let mut fields: Vec<Arc<Field>> = schema.fields().iter().cloned().collect();
            fields[idx] = Arc::new(
                fields[idx]
                    .as_ref()
                    .clone()
                    .with_data_type(new_col.data_type().clone()),
            );
            Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
        };

        let mut columns: Vec<Arc<dyn Array>> = self.data.columns().to_vec();
        columns[idx] = new_col;

        let new_batch = RecordBatch::try_new(schema, columns)?;
        let mut metadata = self.metadata.clone();
        metadata.length = new_batch.num_rows();
        Self::new(new_batch, metadata)
    }

    /// Replace a column by name, allowing the data type to change.
    ///
    /// Rebuilds the schema with the new column's type. Use this when an
    /// augmentation changes the shape (e.g., crop changes FixedSizeList length).
    pub fn replace_column(&self, name: &str, new_col: Arc<dyn Array>) -> Result<Self> {
        let old_schema = self.data.schema();
        let idx = old_schema
            .index_of(name)
            .map_err(|_| DataCoreError::ColumnNotFound(name.to_string()))?;

        let mut fields: Vec<Arc<Field>> = old_schema.fields().iter().cloned().collect();
        fields[idx] = Arc::new(Field::new(name, new_col.data_type().clone(), true));
        let new_schema = Arc::new(Schema::new(fields));

        let mut columns: Vec<Arc<dyn Array>> = self.data.columns().to_vec();
        columns[idx] = new_col;

        let new_batch = RecordBatch::try_new(new_schema, columns)?;
        let mut metadata = self.metadata.clone();
        metadata.length = new_batch.num_rows();
        Self::new(new_batch, metadata)
    }

    /// Append a new nullable column, returning a new Episode.
    ///
    /// The column must have one entry per step. Adding a name that already
    /// exists produces a batch with duplicate field names; check
    /// [`Episode::has_column`] first.
    pub fn add_column(&self, name: &str, col: Arc<dyn Array>) -> Result<Self> {
        let old_schema = self.data.schema();
        let mut fields: Vec<Arc<Field>> = old_schema.fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(name, col.data_type().clone(), true)));
        let new_schema = Arc::new(Schema::new(fields));

        let mut columns: Vec<Arc<dyn Array>> = self.data.columns().to_vec();
        columns.push(col);

        let new_batch = RecordBatch::try_new(new_schema, columns)?;
        let mut metadata = self.metadata.clone();
        metadata.length = new_batch.num_rows();
        Self::new(new_batch, metadata)
    }

    /// Check if a column exists.
    pub fn has_column(&self, name: &str) -> bool {
        self.data.schema().index_of(name).is_ok()
    }

    /// Decode an encoded image column to raw RGB pixels.
    ///
    /// Decodes `observation.image.{camera}` (`LargeBinaryArray` of JPEG or other
    /// `image`-crate-supported bytes) and returns a `FixedSizeListArray<UInt8>`
    /// with H×W×3 bytes per row, together with `(width, height)`.
    ///
    /// Raw pixel columns let augmentations operate on pixels directly without
    /// re-encoding between steps.
    ///
    /// # Errors
    ///
    /// Returns an error if any frame fails to decode or if frames do not all
    /// have the dimensions of the first frame.
    pub fn decode_image_column(
        &self,
        camera_name: &str,
    ) -> Result<(Arc<FixedSizeListArray>, u32, u32)> {
        use rayon::prelude::*;

        let jpeg_col = self.image(camera_name)?;
        let n = jpeg_col.len();

        if n == 0 {
            return Err(DataCoreError::InvalidEpisode("Empty image column".into()));
        }

        // Decode first frame to get dimensions
        let first_bytes = jpeg_col.value(0);
        let first_img = image::load_from_memory(first_bytes)
            .map_err(|e| DataCoreError::Other(format!("Failed to decode image: {e}")))?;
        let width = first_img.width();
        let height = first_img.height();
        let pixel_count = (height * width * 3) as usize;

        // Decode all frames in parallel
        let all_pixels: Vec<Vec<u8>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let bytes = jpeg_col.value(i);
                let img = image::load_from_memory(bytes).map_err(|e| {
                    DataCoreError::Other(format!("Failed to decode image at step {i}: {e}"))
                })?;
                if img.width() != width || img.height() != height {
                    return Err(DataCoreError::InvalidEpisode(format!(
                        "Image at step {i} is {}x{}, but the first frame is {width}x{height}",
                        img.width(),
                        img.height()
                    )));
                }
                Ok(img.to_rgb8().into_raw())
            })
            .collect::<Result<_>>()?;

        // Build FixedSizeListArray<UInt8>: each row is H*W*3 bytes
        let flat: Vec<u8> = all_pixels.into_iter().flatten().collect();
        let values = UInt8Array::from(flat);
        let field = Arc::new(Field::new("item", DataType::UInt8, false));
        let list = FixedSizeListArray::try_new(field, pixel_count as i32, Arc::new(values), None)
            .map_err(DataCoreError::Arrow)?;

        Ok((Arc::new(list), width, height))
    }

    /// Decode encoded image columns to raw RGB pixels, returning a new Episode.
    ///
    /// For each listed camera that has an `observation.image.{cam}` column,
    /// adds (or replaces) `observation.image.{cam}.rgb` and adds
    /// `observation.image.{cam}.width` / `.height` columns if absent. Cameras
    /// without an image column are skipped. The encoded column is kept.
    pub fn decode_images(&self, camera_names: &[&str]) -> Result<Self> {
        let mut current = self.clone();

        for cam in camera_names {
            let col_name = format!("observation.image.{cam}");
            if !self.has_column(&col_name) {
                continue;
            }

            let (rgb_col, width, height) = self.decode_image_column(cam)?;

            let rgb_col_name = format!("observation.image.{cam}.rgb");
            if current.has_column(&rgb_col_name) {
                current = current.with_column(&rgb_col_name, rgb_col)?;
            } else {
                current = current.add_column(&rgb_col_name, rgb_col)?;
            }

            // Store dimensions as metadata columns
            let n = current.num_steps();
            let width_col_name = format!("observation.image.{cam}.width");
            let height_col_name = format!("observation.image.{cam}.height");
            if !current.has_column(&width_col_name) {
                let widths = Arc::new(Int64Array::from(vec![width as i64; n]));
                current = current.add_column(&width_col_name, widths)?;
                let heights = Arc::new(Int64Array::from(vec![height as i64; n]));
                current = current.add_column(&height_col_name, heights)?;
            }
        }

        Ok(current)
    }

    /// Get the raw RGB pixel column `observation.image.{camera_name}.rgb`.
    ///
    /// Each row is H×W×3 bytes in row-major HWC RGB order.
    pub fn image_raw(&self, camera_name: &str) -> Result<&FixedSizeListArray> {
        let col_name = format!("observation.image.{camera_name}.rgb");
        let col = self.column(&col_name)?;
        col.as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                DataCoreError::SchemaMismatch(format!(
                    "'{col_name}' is not a FixedSizeListArray. Call decode_images() first."
                ))
            })
    }

    /// Get `(width, height)` of a raw pixel column, read from the first row of
    /// `observation.image.{camera_name}.width` / `.height`.
    pub fn image_dimensions(&self, camera_name: &str) -> Result<(u32, u32)> {
        let w_col = format!("observation.image.{camera_name}.width");
        let h_col = format!("observation.image.{camera_name}.height");
        let width = self
            .column(&w_col)?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| DataCoreError::ColumnNotFound(w_col.clone()))?
            .value(0) as u32;
        let height = self
            .column(&h_col)?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| DataCoreError::ColumnNotFound(h_col.clone()))?
            .value(0) as u32;
        Ok((width, height))
    }

    /// Convert a raw RGB pixel column to CHW float32 values normalized to [0, 1].
    ///
    /// Input: `FixedSizeListArray<UInt8>` with H×W×3 bytes per row (HWC layout).
    /// Output: `FixedSizeListArray<Float32>` with 3×H×W values per row (CHW layout).
    ///
    /// This is the host-side conversion used by the `arrow_f32` materialization
    /// strategy.
    pub fn image_to_chw_f32(&self, camera_name: &str) -> Result<Arc<FixedSizeListArray>> {
        use rayon::prelude::*;

        let raw_col = self.image_raw(camera_name)?;
        let (width, height) = self.image_dimensions(camera_name)?;
        let pixel_count = (height * width * 3) as usize;
        let n = raw_col.len();

        if raw_col.value_length() as usize != pixel_count {
            return Err(DataCoreError::SchemaMismatch(format!(
                "'observation.image.{camera_name}.rgb' has {} values per row, but \
                 width/height columns give {width}x{height}x3 = {pixel_count}",
                raw_col.value_length()
            )));
        }
        if raw_col
            .values()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .is_none()
        {
            return Err(DataCoreError::SchemaMismatch(format!(
                "'observation.image.{camera_name}.rgb' values are not UInt8"
            )));
        }

        let all_chw: Vec<Vec<f32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = raw_col.value(i);
                let hwc = row
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .expect("Expected UInt8Array in raw pixel column");

                let w = width as usize;
                let h = height as usize;
                let mut chw = vec![0.0f32; pixel_count];

                // HWC → CHW with [0,1] normalization
                #[allow(clippy::erasing_op, clippy::identity_op)]
                // explicit per-channel plane offsets
                for y in 0..h {
                    for x in 0..w {
                        let hwc_idx = (y * w + x) * 3;
                        chw[0 * h * w + y * w + x] = hwc.value(hwc_idx) as f32 / 255.0; // R
                        chw[1 * h * w + y * w + x] = hwc.value(hwc_idx + 1) as f32 / 255.0; // G
                        chw[2 * h * w + y * w + x] = hwc.value(hwc_idx + 2) as f32 / 255.0;
                        // B
                    }
                }
                chw
            })
            .collect();

        let flat: Vec<f32> = all_chw.into_iter().flatten().collect();
        let values = Float32Array::from(flat);
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        let list = FixedSizeListArray::try_new(field, pixel_count as i32, Arc::new(values), None)
            .map_err(DataCoreError::Arrow)?;

        Ok(Arc::new(list))
    }
}

/// Whether two types hold the same values, ignoring the names and
/// nullability of list child fields.
fn same_value_layout(a: &DataType, b: &DataType) -> bool {
    match (a, b) {
        (DataType::FixedSizeList(fa, na), DataType::FixedSizeList(fb, nb)) => {
            na == nb && same_value_layout(fa.data_type(), fb.data_type())
        }
        (DataType::List(fa), DataType::List(fb))
        | (DataType::LargeList(fa), DataType::LargeList(fb)) => {
            same_value_layout(fa.data_type(), fb.data_type())
        }
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        BooleanArray, FixedSizeListArray, Float32Array, Int64Array, LargeBinaryArray,
    };
    use arrow_schema::{DataType, Field};
    use std::sync::Arc;

    fn make_test_episode(num_steps: usize, action_dim: i32) -> Episode {
        let schema = Arc::new(Schema::new(vec![
            Field::new("step_index", DataType::Int64, false),
            Field::new(
                "action",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    action_dim,
                ),
                false,
            ),
            Field::new("observation.image.cam_0", DataType::LargeBinary, true),
            Field::new("reward", DataType::Float32, true),
            Field::new("done", DataType::Boolean, true),
            Field::new("episode_index", DataType::Int64, false),
        ]));

        let step_indices = Int64Array::from((0..num_steps as i64).collect::<Vec<_>>());
        let action_values = Float32Array::from(vec![0.0f32; num_steps * action_dim as usize]);
        let actions = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            action_dim,
            Arc::new(action_values),
            None,
        )
        .unwrap();
        let images = LargeBinaryArray::from(vec![b"fake_jpeg" as &[u8]; num_steps]);
        let rewards = Float32Array::from(vec![0.0f32; num_steps]);
        let dones = {
            let mut vals = vec![false; num_steps];
            if num_steps > 0 {
                vals[num_steps - 1] = true;
            }
            BooleanArray::from(vals)
        };
        let episode_indices = Int64Array::from(vec![0i64; num_steps]);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(step_indices),
                Arc::new(actions),
                Arc::new(images),
                Arc::new(rewards),
                Arc::new(dones),
                Arc::new(episode_indices),
            ],
        )
        .unwrap();

        Episode::from_record_batch(batch).unwrap()
    }

    #[test]
    fn test_episode_creation() {
        let ep = make_test_episode(10, 7);
        assert_eq!(ep.num_steps(), 10);
        assert_eq!(ep.metadata().episode_index, 0);
    }

    #[test]
    fn test_episode_empty_fails() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "step_index",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::new_empty(schema);
        assert!(Episode::from_record_batch(batch).is_err());
    }

    #[test]
    fn test_episode_column_access() {
        let ep = make_test_episode(5, 7);
        assert!(ep.actions().is_ok());
        assert!(ep.image("cam_0").is_ok());
        assert!(ep.rewards().is_ok());
        assert!(ep.dones().is_ok());
    }

    #[test]
    fn test_episode_missing_column() {
        let ep = make_test_episode(5, 7);
        assert!(ep.column("nonexistent").is_err());
    }

    #[test]
    fn test_from_record_batch_empty_with_episode_index_errors() {
        // Regression: an empty batch that has an `episode_index` column used to
        // panic in `Int64Array::value(0)` instead of returning an error.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "episode_index",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::new_empty(schema);
        assert!(matches!(
            Episode::from_record_batch(batch),
            Err(DataCoreError::InvalidEpisode(_))
        ));
    }

    fn encode_jpeg(width: u32, height: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        image::DynamicImage::new_rgb8(width, height)
            .write_to(
                &mut std::io::Cursor::new(&mut buf),
                image::ImageFormat::Jpeg,
            )
            .unwrap();
        buf
    }

    fn make_image_episode(frames: Vec<Vec<u8>>) -> Episode {
        let n = frames.len();
        let schema = Arc::new(Schema::new(vec![
            Field::new("observation.image.cam", DataType::LargeBinary, false),
            Field::new("episode_index", DataType::Int64, false),
        ]));
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(LargeBinaryArray::from(refs)),
                Arc::new(Int64Array::from(vec![0i64; n])),
            ],
        )
        .unwrap();
        Episode::from_record_batch(batch).unwrap()
    }

    #[test]
    fn test_decode_images_and_chw() {
        let ep = make_image_episode(vec![encode_jpeg(4, 2), encode_jpeg(4, 2)]);
        let decoded = ep.decode_images(&["cam", "missing"]).unwrap();
        assert_eq!(decoded.image_dimensions("cam").unwrap(), (4, 2));
        assert_eq!(decoded.image_raw("cam").unwrap().value_length(), 4 * 2 * 3);
        let chw = decoded.image_to_chw_f32("cam").unwrap();
        assert_eq!(chw.len(), 2);
        assert_eq!(chw.value_length(), 3 * 4 * 2);
    }

    #[test]
    fn test_decode_image_column_corrupt_frame_errors() {
        // Regression: a frame that fails to decode used to panic inside the
        // Rayon closure instead of returning an error.
        let ep = make_image_episode(vec![encode_jpeg(4, 4), b"not an image".to_vec()]);
        assert!(ep.decode_image_column("cam").is_err());
    }

    #[test]
    fn test_decode_image_column_size_mismatch_errors() {
        // Regression: frames with different sizes were flattened into one
        // buffer and silently re-split at the first frame's size.
        let ep = make_image_episode(vec![
            encode_jpeg(4, 4),
            encode_jpeg(4, 4),
            encode_jpeg(8, 2),
            encode_jpeg(2, 8),
        ]);
        assert!(ep.decode_image_column("cam").is_err());
    }

    #[test]
    fn test_replace_and_add_column() {
        let ep = make_test_episode(3, 7);
        let wider = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            2,
            Arc::new(Float32Array::from(vec![0.0f32; 6])),
            None,
        )
        .unwrap();
        let ep2 = ep.replace_column("action", Arc::new(wider)).unwrap();
        assert_eq!(ep2.actions().unwrap().value_length(), 2);
        assert!(!ep2.has_column("extra"));
        let ep3 = ep2
            .add_column("extra", Arc::new(Int64Array::from(vec![1i64, 2, 3])))
            .unwrap();
        assert!(ep3.has_column("extra"));
        assert_eq!(ep3.metadata().length, 3);
    }

    #[test]
    fn test_episode_with_column_replacement() {
        let ep = make_test_episode(5, 7);
        let new_rewards = Arc::new(Float32Array::from(vec![1.0f32; 5]));
        let ep2 = ep.with_column("reward", new_rewards).unwrap();
        let rewards = ep2.rewards().unwrap();
        assert_eq!(rewards.value(0), 1.0);
    }

    #[test]
    fn test_with_column_accepts_renamed_list_child() {
        use arrow_array::{FixedSizeListArray, UInt8Array};
        // A Parquet-style column: list child named "element", nullable.
        let element = Arc::new(Field::new("element", DataType::UInt8, true));
        let old = FixedSizeListArray::new(
            Arc::clone(&element),
            2,
            Arc::new(UInt8Array::from(vec![1u8, 2, 3, 4])),
            None,
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "pixels",
            old.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(old)]).unwrap();
        let ep = Episode::new(
            batch,
            EpisodeMetadata {
                episode_index: 0,
                task: None,
                length: 2,
            },
        )
        .unwrap();

        // Arrow builders name the child "item" and make it non-nullable.
        let item = Arc::new(Field::new("item", DataType::UInt8, false));
        let new = FixedSizeListArray::new(
            item,
            2,
            Arc::new(UInt8Array::from(vec![9u8, 9, 9, 9])),
            None,
        );
        let ep2 = ep.with_column("pixels", Arc::new(new)).unwrap();
        assert_eq!(ep2.num_steps(), 2);

        // A different list length is still rejected.
        let wrong =
            FixedSizeListArray::new(element, 4, Arc::new(UInt8Array::from(vec![0u8; 8])), None);
        assert!(ep.with_column("pixels", Arc::new(wrong)).is_err());
    }
}
