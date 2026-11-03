//! Lazy dataset loading for LeRobot datasets.
//!
//! This module provides [`LazyDataset`], which mirrors LeRobot's Python
//! `LeRobotDataset` interface with on-demand frame access and an episode
//! cache.
//!
//! # Caching and forked workers
//!
//! An episode is decoded on first access (Parquet columns plus, for each
//! camera, its video frames) into an immutable Arrow `RecordBatch` and kept
//! in a cache behind an `RwLock`. For multi-process data loading,
//! [`LazyDataset::preload_episodes`] decodes the needed episodes in the
//! parent process and moves them into a frozen, lock-free map. Worker
//! processes forked afterwards read those heap buffers through Linux
//! copy-on-write page sharing: the pages are shared as long as nobody writes
//! to them, and augmentations always write to new buffers. Workers never
//! take the lock and never call into FFmpeg, which is not fork-safe.
//!
//! # Image representations
//!
//! By default each camera is stored as JPEG bytes (`observation.image.{cam}`,
//! re-encoded from the decoded video). With [`LazyDataset::with_raw_pixels`]
//! frames are stored as raw RGB (`observation.image.{cam}.rgb` plus
//! `.width` / `.height` columns). These correspond to the `arrow_uint8_jpeg`
//! and `arrow_uint8` / `arrow_f32` materialization strategies in the paper.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use arrow::record_batch::RecordBatch;
use arrow_array::*;
use arrow_schema;

use data_core::{DataCoreError, Dataset, Episode, EpisodeMetadata, Result};

use super::lazy_metadata::{EpisodeInfo, LazyDatasetMetadata};
use super::loader::LeRobotLoader;
use super::video::VideoDecoder;

use data_core_augmentations::BoxedLocalAug;

/// A single frame (timestep) from the dataset.
///
/// This is the unit returned by random access operations like `dataset[idx]`.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Global frame index in the dataset.
    pub index: usize,
    /// Episode index this frame belongs to.
    pub episode_index: usize,
    /// Frame index within the episode (0-based).
    pub frame_index_in_episode: usize,
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Action vector.
    pub action: Vec<f32>,
    /// State vector (if available).
    pub state: Option<Vec<f32>>,
    /// Image observations keyed by camera name (JPEG bytes). Empty in raw
    /// pixel mode, where frames are not stored as JPEG.
    pub images: HashMap<String, Vec<u8>>,
    /// Reward value (if available).
    pub reward: Option<f32>,
    /// Whether this is a terminal frame.
    pub done: bool,
    /// Task index.
    pub task_index: usize,
    /// Task description (if available).
    pub task: Option<String>,
}

/// A lazy-loading dataset that mirrors LeRobot's Python interface.
///
/// Unlike `Dataset`, this only loads metadata at initialization time.
/// Episodes and frames are loaded on-demand when accessed.
///
/// Frame indices are global (`0..num_frames()`) unless an episode subset is
/// selected with [`LazyDataset::with_episodes`], in which case they run over
/// the concatenation of the selected episodes in the given order.
///
/// # Example
///
/// ```no_run
/// use data_core_io::LazyDataset;
///
/// // Fast initialization - only loads metadata
/// let dataset = LazyDataset::open("/path/to/dataset")?;
///
/// println!("Dataset has {} frames", dataset.num_frames());
///
/// // Load a single frame on-demand
/// let frame = dataset.get_frame(42)?;
/// println!("Frame {} has action {:?}", frame.index, frame.action);
///
/// // Load an episode (cached for future access)
/// let episode = dataset.get_episode(0)?;
/// # Ok::<(), data_core::DataCoreError>(())
/// ```
pub struct LazyDataset {
    /// Extended metadata with per-episode info.
    meta: LazyDatasetMetadata,
    /// Optional episode filter (None = all episodes).
    selected_episodes: Option<Vec<usize>>,
    /// Cache of loaded episodes (episode_index -> Episode).
    episode_cache: RwLock<HashMap<usize, Arc<Episode>>>,
    /// Frozen (lock-free) cache for fork-safe multi-worker access.
    /// Set via `preload_episodes()`. Workers read from this without locks;
    /// once set, episodes that are not in it cannot be loaded.
    frozen_cache: Option<HashMap<usize, Arc<Episode>>>,
    /// Maximum number of episodes to cache (0 = unlimited).
    cache_limit: usize,
    /// When true, decode video to raw RGB pixels instead of JPEG bytes.
    /// Raw pixels enable zero-loss augmentation on the Arrow IR.
    raw_pixels: bool,
    /// Local augmentations applied to each window in `get_frame_windowed`.
    augmentations: Vec<BoxedLocalAug>,
}

impl LazyDataset {
    /// Open a dataset from a local directory.
    ///
    /// This only loads metadata (see [`LazyDatasetMetadata::load`]) and does
    /// not decode any video.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata cannot be read.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let meta = LazyDatasetMetadata::load(path)?;
        Ok(Self {
            meta,
            selected_episodes: None,
            episode_cache: RwLock::new(HashMap::new()),
            frozen_cache: None,
            cache_limit: 0, // unlimited by default
            raw_pixels: false,
            augmentations: Vec::new(),
        })
    }

    /// Enable raw pixel mode: decode video to raw RGB instead of JPEG bytes.
    ///
    /// Raw pixels are stored as `FixedSizeListArray<UInt8>` columns in the Arrow
    /// RecordBatch (H×W×3 bytes per frame). This eliminates JPEG compression
    /// artifacts and enables zero-loss augmentation on the Arrow IR.
    ///
    /// Increases memory usage (about 921 KB of raw RGB versus roughly 27 KB of
    /// JPEG per 480×640 frame). Affects episodes loaded after this call.
    pub fn with_raw_pixels(mut self) -> Self {
        self.raw_pixels = true;
        self
    }

    /// Add a local augmentation to be applied on-the-fly during frame access.
    ///
    /// Augmentations run on a window slice of the cached episode each time
    /// `get_frame_windowed()` is called, producing fresh random augmentations
    /// on every access. The cached raw episode is never modified.
    pub fn add_augmentation(
        &mut self,
        aug: impl data_core_augmentations::LocalAugmentation + 'static,
    ) {
        self.augmentations.push(std::sync::Arc::new(aug));
    }

    /// Add a shared local augmentation (used by the Python bindings).
    pub fn add_augmentation_boxed(&mut self, aug: BoxedLocalAug) {
        self.augmentations.push(aug);
    }

    /// Whether any augmentation has been added.
    pub fn has_augmentations(&self) -> bool {
        !self.augmentations.is_empty()
    }

    /// Get a window of frames around `idx`, with on-the-fly augmentation.
    ///
    /// 1. Resolves `idx` to an episode and frame offset.
    /// 2. Loads the episode from the cache (or decodes it from disk).
    /// 3. Selects one row per entry of `window_offsets`, relative to the
    ///    target frame and clamped to the episode bounds (so windows never
    ///    cross into another episode; out-of-range offsets repeat the first
    ///    or last frame).
    /// 4. Applies the configured augmentations to the window (fresh random
    ///    parameters on each call; the cached episode is not modified).
    /// 5. Returns the window as an `Episode`.
    ///
    /// `window_offsets` are frame offsets relative to the target frame, e.g.,
    /// `[-15, -10, -5, 0]` for a 4-frame window at 50fps with 0.3s spacing.
    ///
    /// Raw pixel columns (`FixedSizeList<UInt8>` with more than 1000 values
    /// per row) are not copied with the other columns. They are gathered
    /// directly from the cached episode right before the first image
    /// augmentation runs, or at the end if raw pixel mode is on and no image
    /// augmentation is configured. In JPEG mode the JPEG column is copied
    /// like any other column.
    ///
    /// # Errors
    ///
    /// Returns an error if `idx` is out of range, if the episode cannot be
    /// loaded, if `window_offsets` is empty, or if an augmentation fails.
    pub fn get_frame_windowed(&self, idx: usize, window_offsets: &[i64]) -> Result<Episode> {
        let (ep_info, frame_in_ep) = self.resolve_frame_index(idx)?;
        let episode = self.get_episode(ep_info.episode_index)?;

        let num_steps = episode.num_steps();

        // Compute window frame indices (clamped to episode bounds)
        let window_indices: Vec<usize> = window_offsets
            .iter()
            .map(|&offset| {
                let target = frame_in_ep as i64 + offset;
                target.clamp(0, (num_steps - 1) as i64) as usize
            })
            .collect();

        let window_len = window_indices.len();

        // Optimized path: skip take() on large image columns.
        // Only take() small columns (state, action, timestamps), then build
        // image columns by reading directly from the source episode at indices.
        let data = episode.data();
        let schema = data.schema();

        let take_indices =
            UInt32Array::from(window_indices.iter().map(|&i| i as u32).collect::<Vec<_>>());

        let mut window_fields = Vec::with_capacity(schema.fields().len());
        let mut window_columns: Vec<Arc<dyn Array>> = Vec::with_capacity(schema.fields().len());

        for (i, field) in schema.fields().iter().enumerate() {
            let col = data.column(i);

            // Skip large FixedSizeList<UInt8> columns (raw pixel data).
            // These will be read directly by augmentations from the source episode.
            if Self::is_large_pixel_column(field) {
                continue;
            }

            let taken = arrow::compute::take(col.as_ref(), &take_indices, None)
                .map_err(DataCoreError::Arrow)?;
            window_fields.push(field.as_ref().clone());
            window_columns.push(taken);
        }

        // Build the lightweight window (no image data yet)
        let window_schema = Arc::new(arrow_schema::Schema::new(window_fields));
        let window_batch =
            RecordBatch::try_new(window_schema, window_columns).map_err(DataCoreError::Arrow)?;

        let mut window_episode = Episode::new(
            window_batch,
            EpisodeMetadata {
                episode_index: ep_info.episode_index as i64,
                task: None,
                length: window_len,
            },
        )?;

        if !self.augmentations.is_empty() {
            // Optimized: apply augmentations with source episode context.
            // First augmentation (typically crop) reads from source at indices,
            // writes smaller output. Subsequent augs operate on the smaller data.
            window_episode =
                self.apply_augmentations_indexed(&episode, &window_indices, window_episode)?;
        } else if self.raw_pixels {
            // No augmentations but raw_pixels mode: add raw pixel columns via take
            window_episode =
                self.add_pixel_columns_via_take(&episode, &window_indices, window_episode)?;
        }

        Ok(window_episode)
    }

    /// Get an augmented window with images already converted to float32 CHW in `[0, 1]`.
    ///
    /// Same as `get_frame_windowed`, plus a float32 CHW `.f32` column (values
    /// in [0, 1]) next to each uint8 HWC `.rgb` column. The `.rgb` columns are
    /// kept. The `.f32` data can be wrapped as a tensor without further
    /// conversion; this is the `arrow_f32` materialization strategy.
    pub fn get_frame_windowed_f32(&self, idx: usize, window_offsets: &[i64]) -> Result<Episode> {
        use rayon::prelude::*;

        let mut window = self.get_frame_windowed(idx, window_offsets)?;
        let data = window.data();
        let schema = data.schema();

        // Find all .rgb columns and convert to .f32 (float32 CHW)
        let rgb_cols: Vec<String> = schema
            .fields()
            .iter()
            .filter(|f| f.name().ends_with(".rgb"))
            .map(|f| f.name().clone())
            .collect();

        for rgb_name in &rgb_cols {
            let cam = rgb_name
                .strip_suffix(".rgb")
                .and_then(|s| s.strip_prefix("observation.image."))
                .unwrap_or("");

            let (width, height) = window.image_dimensions(cam)?;
            let w = width as usize;
            let h = height as usize;
            let pixel_count = h * w * 3;
            let chw_count = 3 * h * w; // same total, different layout

            let raw_col = window
                .column(rgb_name)?
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    DataCoreError::SchemaMismatch(format!("{rgb_name} not FixedSizeListArray"))
                })?;

            let n = raw_col.len();
            let src_values = raw_col.values();
            let src_u8 = src_values
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| DataCoreError::Other("Expected UInt8Array".into()))?;
            let src_buf = src_u8.values();

            // Convert all frames: uint8 HWC → float32 CHW [0,1] in parallel
            let all_f32: Vec<f32> = (0..n)
                .into_par_iter()
                .flat_map(|frame| {
                    let offset = frame * pixel_count;
                    let mut chw = vec![0.0f32; chw_count];
                    for y in 0..h {
                        for x in 0..w {
                            let hwc_idx = offset + (y * w + x) * 3;
                            let base = y * w + x;
                            chw[base] = src_buf[hwc_idx] as f32 / 255.0;
                            chw[h * w + base] = src_buf[hwc_idx + 1] as f32 / 255.0;
                            chw[2 * h * w + base] = src_buf[hwc_idx + 2] as f32 / 255.0;
                        }
                    }
                    chw
                })
                .collect();

            let f32_values = Float32Array::from(all_f32);
            let field = Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Float32,
                false,
            ));
            let f32_col = Arc::new(
                FixedSizeListArray::try_new(field, chw_count as i32, Arc::new(f32_values), None)
                    .map_err(DataCoreError::Arrow)?,
            );

            let f32_name = rgb_name.replace(".rgb", ".f32");
            window = window.add_column(&f32_name, f32_col)?;
        }

        Ok(window)
    }

    /// Check if a schema field is a large pixel column that should skip take().
    fn is_large_pixel_column(field: &arrow_schema::Field) -> bool {
        if let arrow_schema::DataType::FixedSizeList(inner, size) = field.data_type() {
            // Raw pixel columns: FixedSizeList<UInt8> with large element size
            *size > 1000 && *inner.data_type() == arrow_schema::DataType::UInt8
        } else {
            false
        }
    }

    /// Apply augmentations reading image data directly from the source episode.
    ///
    /// The first image augmentation gets pixel data from the source episode at the
    /// given indices (avoiding the full-resolution take()). Subsequent augmentations
    /// operate on the (typically smaller) output of the previous step.
    fn apply_augmentations_indexed(
        &self,
        source_episode: &Episode,
        window_indices: &[usize],
        mut window_episode: Episode,
    ) -> Result<Episode> {
        let mut first_image_aug = true;

        for aug in &self.augmentations {
            if first_image_aug && Self::is_image_augmentation(aug.name()) {
                // First image aug: inject pixel columns from source at indices,
                // then apply augmentation
                window_episode =
                    self.inject_indexed_pixels(source_episode, window_indices, window_episode)?;
                window_episode = aug.apply(&window_episode)?;
                first_image_aug = false;
            } else {
                window_episode = aug.apply(&window_episode)?;
            }
        }

        // If no image augmentation ran but we need pixel columns
        if first_image_aug && self.raw_pixels {
            window_episode =
                self.inject_indexed_pixels(source_episode, window_indices, window_episode)?;
        }

        Ok(window_episode)
    }

    /// Check if an augmentation operates on image columns.
    fn is_image_augmentation(name: &str) -> bool {
        matches!(
            name,
            "random_crop"
                | "color_jitter"
                | "gaussian_noise"
                | "horizontal_flip"
                | "gaussian_blur"
                | "frame_dropout"
                | "fused_image"
        )
    }

    /// Inject raw pixel columns from source episode at specific indices.
    ///
    /// Reads pixels directly from the source episode's FixedSizeListArray
    /// at the given indices. This is faster than Arrow take() because we
    /// copy from the underlying buffer directly.
    fn inject_indexed_pixels(
        &self,
        source: &Episode,
        indices: &[usize],
        mut window: Episode,
    ) -> Result<Episode> {
        let src_data = source.data();
        let src_schema = src_data.schema();

        for (i, field) in src_schema.fields().iter().enumerate() {
            if !Self::is_large_pixel_column(field) {
                continue;
            }

            let col = src_data.column(i);
            let list_col = col
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    DataCoreError::Other("Expected FixedSizeListArray for pixel column".into())
                })?;

            let elem_size = list_col.value_length() as usize;
            let src_values = list_col
                .values()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| DataCoreError::Other("Expected UInt8Array values".into()))?;
            let src_buf = src_values.values();

            // Copy only the indexed frames from the source buffer
            let mut out_buf: Vec<u8> = Vec::with_capacity(indices.len() * elem_size);
            for &idx in indices {
                let start = idx * elem_size;
                out_buf.extend_from_slice(&src_buf[start..start + elem_size]);
            }

            let new_values = UInt8Array::from(out_buf);
            let inner_field = match list_col.data_type() {
                arrow_schema::DataType::FixedSizeList(f, _) => f.clone(),
                _ => unreachable!(),
            };
            let new_col = Arc::new(
                FixedSizeListArray::try_new(
                    inner_field,
                    elem_size as i32,
                    Arc::new(new_values),
                    None,
                )
                .map_err(DataCoreError::Arrow)?,
            );

            let col_name = field.name();
            window = window.add_column(col_name, new_col)?;
        }

        // Also copy dimension columns (width/height) if present
        for field in src_schema.fields() {
            let name = field.name();
            if (name.ends_with(".width") || name.ends_with(".height")) && !window.has_column(name) {
                if let Ok(col) = source.column(name) {
                    let take_indices =
                        UInt32Array::from(indices.iter().map(|&i| i as u32).collect::<Vec<_>>());
                    let taken = arrow::compute::take(col.as_ref(), &take_indices, None)
                        .map_err(DataCoreError::Arrow)?;
                    window = window.add_column(name, taken)?;
                }
            }
        }

        Ok(window)
    }

    /// Fallback: add pixel columns via take (when no augmentations but raw pixels needed).
    fn add_pixel_columns_via_take(
        &self,
        source: &Episode,
        indices: &[usize],
        window: Episode,
    ) -> Result<Episode> {
        // Same as inject_indexed_pixels: copies the pixel data at `indices`.
        self.inject_indexed_pixels(source, indices, window)
    }

    /// Restrict the dataset to the given episode indices, in the given order.
    ///
    /// This does not load any data; it only records the selection. Afterwards
    /// frame indices are relative to the selection, `num_frames` and
    /// `num_episodes` describe the selection, and loading an episode outside
    /// it is an error.
    pub fn with_episodes(mut self, episodes: Vec<usize>) -> Self {
        self.selected_episodes = Some(episodes);
        self
    }

    /// Set the maximum number of episodes to cache in memory.
    ///
    /// When a newly loaded episode would exceed this limit, one cached
    /// episode is evicted first. The evicted episode is an arbitrary entry of
    /// the underlying hash map, not necessarily the oldest or least recently
    /// used one. Set to 0 for unlimited caching (default). The frozen cache
    /// created by [`LazyDataset::preload_episodes`] is not limited.
    pub fn with_cache_limit(mut self, limit: usize) -> Self {
        self.cache_limit = limit;
        self
    }

    /// Set the cache limit on an existing instance; see
    /// [`LazyDataset::with_cache_limit`].
    pub fn set_cache_limit(&mut self, limit: usize) {
        self.cache_limit = limit;
    }

    /// The episode cache limit (0 means unlimited).
    pub fn cache_limit(&self) -> usize {
        self.cache_limit
    }

    /// Whether video is decoded to raw RGB pixels (see
    /// [`LazyDataset::with_raw_pixels`]).
    pub fn raw_pixels(&self) -> bool {
        self.raw_pixels
    }

    /// Locate frame `idx` (in the frame index space of [`LazyDataset::get_frame`]).
    ///
    /// Returns `(episode_index, frame_in_episode, episode_length)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `idx` is out of range.
    pub fn locate_frame(&self, idx: usize) -> Result<(usize, usize, usize)> {
        let (ep, frame_in_ep) = self.resolve_frame_index(idx)?;
        Ok((ep.episode_index, frame_in_ep, ep.length))
    }

    /// The episode selection set by [`LazyDataset::with_episodes`], if any.
    pub fn selected_episodes(&self) -> Option<&[usize]> {
        self.selected_episodes.as_deref()
    }

    /// Clear the lock-protected episode cache. The frozen cache, if any, is
    /// kept.
    pub fn clear_cache(&self) {
        let mut cache = self
            .episode_cache
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.clear();
    }

    /// Pre-load episodes and freeze them into a lock-free cache.
    ///
    /// After calling this, forked DataLoader workers can read cached episodes
    /// without touching the RwLock (a lock held by another thread at fork
    /// time stays locked forever in the child). Heap-allocated pixel data is
    /// shared across fork boundaries through Linux copy-on-write.
    ///
    /// The frozen set replaces any previous one. Afterwards, `get_episode`
    /// for an episode that was not preloaded returns an error instead of
    /// decoding it, because video decoding is not safe in a forked worker.
    ///
    /// # Errors
    ///
    /// Returns the first episode load error. In that case no frozen cache is
    /// installed.
    pub fn preload_episodes(&mut self, episode_indices: &[usize]) -> Result<()> {
        // Temporarily clear frozen cache so get_episode falls through to the
        // RwLock path (safe: this runs in the main process, not a forked worker).
        self.frozen_cache = None;

        let mut frozen = HashMap::new();
        for &idx in episode_indices {
            let ep = self.get_episode(idx)?;
            frozen.insert(idx, ep);
        }
        self.frozen_cache = Some(frozen);
        Ok(())
    }

    /// Dataset metadata.
    pub fn metadata(&self) -> &LazyDatasetMetadata {
        &self.meta
    }

    /// Dataset name.
    pub fn name(&self) -> &str {
        &self.meta.name
    }

    /// Frames per second.
    pub fn fps(&self) -> f64 {
        self.meta.fps
    }

    /// Total number of frames in the dataset (or selected subset). Selected
    /// episodes that do not exist in the metadata count as 0 frames.
    pub fn num_frames(&self) -> usize {
        if let Some(ref selected) = self.selected_episodes {
            selected
                .iter()
                .filter_map(|&idx| self.meta.episode(idx))
                .map(|ep| ep.length)
                .sum()
        } else {
            self.meta.total_frames
        }
    }

    /// Total number of episodes in the dataset (or selected subset).
    pub fn num_episodes(&self) -> usize {
        if let Some(ref selected) = self.selected_episodes {
            selected.len()
        } else {
            self.meta.total_episodes
        }
    }

    /// Camera names (feature keys with the `observation.images.` or
    /// `observation.image.` prefix removed), sorted.
    pub fn camera_names(&self) -> &[String] {
        &self.meta.camera_names
    }

    /// Get a single frame by index (see the type-level note on indexing).
    ///
    /// This will load the containing episode if not already cached.
    pub fn get_frame(&self, idx: usize) -> Result<Frame> {
        // Map global index to episode-relative index
        let (ep_info, frame_in_ep) = self.resolve_frame_index(idx)?;

        // Load the episode (from cache or disk)
        let episode = self.get_episode(ep_info.episode_index)?;

        // Extract the frame data
        self.extract_frame(&episode, ep_info, frame_in_ep)
    }

    /// Get an episode by its `episode_index`, loading it from disk if
    /// necessary.
    ///
    /// Episodes are cached for efficient repeated access.
    ///
    /// # Errors
    ///
    /// Returns an error if the episode is outside the selected subset, is not
    /// in the frozen cache (after `preload_episodes`), does not exist, or
    /// fails to load.
    pub fn get_episode(&self, episode_index: usize) -> Result<Arc<Episode>> {
        // Check if episode is in selected set (if filtering)
        if let Some(ref selected) = self.selected_episodes {
            if !selected.contains(&episode_index) {
                return Err(DataCoreError::InvalidDataset(format!(
                    "Episode {} is not in the selected episode set",
                    episode_index
                )));
            }
        }

        // Check frozen cache first (lock-free, fork-safe)
        if let Some(ref frozen) = self.frozen_cache {
            if let Some(ep) = frozen.get(&episode_index) {
                return Ok(Arc::clone(ep));
            }
            // Frozen cache exists but misses. This may be a forked DataLoader
            // worker, where video decoding (FFmpeg) can deadlock, so never load
            // here. The caller must preload all needed episodes before
            // creating the DataLoader.
            return Err(DataCoreError::Other(format!(
                "Episode {} not preloaded. Call preload_episodes() with all needed \
                 episode indices before creating a multi-worker DataLoader.",
                episode_index
            )));
        }

        // No frozen cache: normal path with RwLock (main process). A poisoned
        // lock only means another thread panicked; the map itself is intact.
        {
            let cache = self
                .episode_cache
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(ep) = cache.get(&episode_index) {
                return Ok(Arc::clone(ep));
            }
        }

        // Load from disk
        let episode = self.load_episode(episode_index)?;
        let episode = Arc::new(episode);

        // Add to cache
        {
            let mut cache = self
                .episode_cache
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            // Evict one arbitrary entry if at the limit (HashMap iteration
            // order, not FIFO or LRU).
            if self.cache_limit > 0 && cache.len() >= self.cache_limit {
                if let Some(&oldest_key) = cache.keys().next() {
                    cache.remove(&oldest_key);
                }
            }

            cache.insert(episode_index, Arc::clone(&episode));
        }

        Ok(episode)
    }

    /// Load all data into memory with [`LeRobotLoader`], returning the full
    /// `Dataset` (filtered to the selected episodes, if any).
    ///
    /// Use this when you need the whole dataset in memory, for example for an
    /// augmentation pipeline. This bypasses the episode cache and the raw
    /// pixel setting; see [`LeRobotLoader::load`] for how images are loaded.
    pub fn load_all(&self) -> Result<Dataset> {
        let loader = LeRobotLoader::new(&self.meta.root);
        let mut dataset = loader.load()?;

        // Filter episodes if selection is active
        if let Some(ref selected) = self.selected_episodes {
            let selected_set: std::collections::HashSet<_> = selected.iter().collect();
            let filtered_episodes: Vec<_> = dataset
                .episodes()
                .iter()
                .filter(|ep| selected_set.contains(&(ep.metadata().episode_index as usize)))
                .cloned()
                .collect();
            dataset = dataset.with_episodes(filtered_episodes);
        }

        Ok(dataset)
    }

    /// Iterate over all frames (or all frames of the selection) in order.
    pub fn frames(&self) -> impl Iterator<Item = Result<Frame>> + '_ {
        FrameIterator::new(self)
    }

    /// Iterate over episode indices: the selection if one is set, otherwise
    /// `0..total_episodes`.
    pub fn episode_indices(&self) -> impl Iterator<Item = usize> + '_ {
        if let Some(ref selected) = self.selected_episodes {
            EpisodeIndexIterator::Selected(selected.iter().copied())
        } else {
            EpisodeIndexIterator::All(0..self.meta.total_episodes)
        }
    }

    // --- Private methods ---

    /// Resolve a global frame index to (episode_info, frame_within_episode).
    fn resolve_frame_index(&self, idx: usize) -> Result<(&EpisodeInfo, usize)> {
        if let Some(ref selected) = self.selected_episodes {
            // When filtering, idx is relative to the selected episodes
            let mut cumulative = 0;
            for &ep_idx in selected {
                let ep = self.meta.episode(ep_idx).ok_or_else(|| {
                    DataCoreError::InvalidDataset(format!("Episode {} not found", ep_idx))
                })?;
                if idx < cumulative + ep.length {
                    return Ok((ep, idx - cumulative));
                }
                cumulative += ep.length;
            }
            Err(DataCoreError::InvalidDataset(format!(
                "Frame index {} out of range (dataset has {} frames)",
                idx, cumulative
            )))
        } else {
            // Global index maps directly to episode ranges
            let ep = self.meta.episode_for_frame(idx).ok_or_else(|| {
                DataCoreError::InvalidDataset(format!(
                    "Frame index {} out of range (dataset has {} frames)",
                    idx, self.meta.total_frames
                ))
            })?;
            let frame_in_ep = idx - ep.dataset_from_index;
            Ok((ep, frame_in_ep))
        }
    }

    /// Load a single episode from disk.
    fn load_episode(&self, episode_index: usize) -> Result<Episode> {
        let ep_info = self.meta.episode(episode_index).ok_or_else(|| {
            DataCoreError::InvalidDataset(format!("Episode {} not found", episode_index))
        })?;

        // Load the parquet file containing this episode
        let parquet_path = self.meta.get_data_file_path(episode_index);
        let loader = LeRobotLoader::new(&self.meta.root);

        // For v3 datasets with chunked files, we need to load the whole chunk
        // and filter to just this episode. For v2 (one episode per file), this
        // loads exactly one episode.
        let batch = loader.read_parquet_file(&parquet_path)?;

        // Filter to just this episode's rows
        let ep_indices = batch
            .column_by_name("episode_index")
            .ok_or_else(|| DataCoreError::ColumnNotFound("episode_index".to_string()))?;

        let ep_indices_i64 = ep_indices
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| DataCoreError::SchemaMismatch("episode_index is not Int64".into()))?;

        // Find row range for this episode
        let mut start = None;
        let mut end = None;
        for (i, val) in ep_indices_i64.iter().enumerate() {
            if let Some(v) = val {
                if v as usize == episode_index {
                    if start.is_none() {
                        start = Some(i);
                    }
                    end = Some(i + 1);
                }
            }
        }

        let (start, end) = match (start, end) {
            (Some(s), Some(e)) => (s, e),
            _ => {
                return Err(DataCoreError::InvalidDataset(format!(
                    "Episode {} not found in parquet file",
                    episode_index
                )));
            }
        };

        let mut episode_batch = batch.slice(start, end - start);

        // Decode video frames for every video feature. v3 episodes start at
        // `from_timestamp` within a shared file; v2 episodes have their own
        // files and start at frame 0.
        if !self.meta.video_keys.is_empty() {
            episode_batch = self.inject_video_frames_for_episode(episode_batch, ep_info)?;
        }

        Episode::from_record_batch(episode_batch)
    }

    /// Decode and inject video frames for a single episode.
    ///
    /// When `raw_pixels` is false: stores JPEG bytes in `LargeBinaryArray`.
    /// When `raw_pixels` is true: stores raw RGB pixels in `FixedSizeListArray<UInt8>`
    /// plus width/height metadata columns. Skips JPEG encoding entirely.
    fn inject_video_frames_for_episode(
        &self,
        batch: RecordBatch,
        ep_info: &EpisodeInfo,
    ) -> Result<RecordBatch> {
        use arrow_schema::{DataType, Field};

        let num_rows = batch.num_rows();
        let fps = self.meta.fps;
        let mut current_batch = batch;

        for video_key in &self.meta.video_keys {
            let camera_name = video_key
                .strip_prefix("observation.images.")
                .or_else(|| video_key.strip_prefix("observation.image."))
                .unwrap_or(video_key)
                .to_string();

            let video_path = self
                .meta
                .get_video_file_path(ep_info.episode_index, video_key);

            if !video_path.exists() {
                tracing::warn!(
                    "Video file not found (skipping camera '{}'): {}",
                    camera_name,
                    video_path.display()
                );
                continue;
            }

            let video_start_frame = ep_info
                .video_timestamps
                .get(video_key)
                .map(|vt| (vt.from_timestamp * fps).round() as i64)
                .unwrap_or(0);

            if self.raw_pixels {
                // Raw RGB path: video → RGB pixels directly (no JPEG)
                let (frames, width, height) =
                    VideoDecoder::decode_range_raw(&video_path, video_start_frame, num_rows, fps)?;

                if frames.len() != num_rows {
                    return Err(DataCoreError::Other(format!(
                        "Frame count mismatch for '{}': decoded {} frames but episode has {} rows",
                        video_path.display(),
                        frames.len(),
                        num_rows,
                    )));
                }

                let pixel_count = (height * width * 3) as i32;

                // Build FixedSizeListArray<UInt8> for raw RGB (heap-allocated).
                // After fork(), workers share these pages via copy-on-write
                // since they only read pixel data (augmentation writes to new buffers).
                let flat: Vec<u8> = frames.into_iter().flatten().collect();
                let values = UInt8Array::from(flat);
                let field = Arc::new(Field::new("item", DataType::UInt8, false));
                let rgb_col = Arc::new(
                    FixedSizeListArray::try_new(field, pixel_count, Arc::new(values), None)
                        .map_err(DataCoreError::Arrow)?,
                );

                // Add rgb column
                let rgb_col_name = format!("observation.image.{}.rgb", camera_name);
                current_batch = Self::append_column(current_batch, &rgb_col_name, rgb_col)?;

                // Add dimension columns
                let w_col_name = format!("observation.image.{}.width", camera_name);
                let h_col_name = format!("observation.image.{}.height", camera_name);
                let widths = Arc::new(Int64Array::from(vec![width as i64; num_rows]));
                let heights = Arc::new(Int64Array::from(vec![height as i64; num_rows]));
                current_batch = Self::append_column(current_batch, &w_col_name, widths)?;
                current_batch = Self::append_column(current_batch, &h_col_name, heights)?;
            } else {
                // JPEG path
                let frames =
                    VideoDecoder::decode_range(&video_path, video_start_frame, num_rows, fps)?;

                if frames.len() != num_rows {
                    return Err(DataCoreError::Other(format!(
                        "Frame count mismatch for '{}': decoded {} frames but episode has {} rows",
                        video_path.display(),
                        frames.len(),
                        num_rows,
                    )));
                }

                let col_name = format!("observation.image.{}", camera_name);
                let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
                let image_col = Arc::new(LargeBinaryArray::from(refs));
                current_batch = Self::append_column(current_batch, &col_name, image_col)?;
            }
        }

        Ok(current_batch)
    }

    /// Helper: append a column to a RecordBatch.
    fn append_column(batch: RecordBatch, name: &str, col: Arc<dyn Array>) -> Result<RecordBatch> {
        use arrow_schema::{Field, Schema};

        let mut fields: Vec<Field> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields.push(Field::new(name, col.data_type().clone(), true));
        let new_schema = Arc::new(Schema::new(fields));

        let mut columns: Vec<Arc<dyn Array>> = batch.columns().to_vec();
        columns.push(col);

        RecordBatch::try_new(new_schema, columns).map_err(DataCoreError::Arrow)
    }

    /// Extract a single frame from a loaded episode.
    fn extract_frame(
        &self,
        episode: &Episode,
        ep_info: &EpisodeInfo,
        frame_idx: usize,
    ) -> Result<Frame> {
        let data = episode.data();

        // Extract action
        let action = self.extract_float_array(data, "action", frame_idx)?;

        // Extract state (optional)
        let state = self
            .extract_float_array(data, "observation.state", frame_idx)
            .ok();

        // Extract images
        let mut images = HashMap::new();
        for camera in &self.meta.camera_names {
            let col_name = format!("observation.image.{}", camera);
            if let Some(col) = data.column_by_name(&col_name) {
                if let Some(arr) = col.as_any().downcast_ref::<LargeBinaryArray>() {
                    if !arr.is_null(frame_idx) {
                        images.insert(camera.clone(), arr.value(frame_idx).to_vec());
                    }
                }
            }
        }

        // Extract reward
        let reward = data
            .column_by_name("reward")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
            .and_then(|a| {
                if a.is_null(frame_idx) {
                    None
                } else {
                    Some(a.value(frame_idx))
                }
            });

        // Extract done
        let done = data
            .column_by_name("done")
            .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
            .map(|a| a.value(frame_idx))
            .unwrap_or(false);

        // Extract timestamp
        let timestamp = data
            .column_by_name("timestamp")
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
            .map(|a| a.value(frame_idx))
            .unwrap_or(frame_idx as f64 / self.meta.fps);

        // Extract task index
        let task_index = data
            .column_by_name("task_index")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            .map(|a| a.value(frame_idx) as usize)
            .unwrap_or(ep_info.task_index);

        let task = self.meta.task(task_index).map(|s| s.to_string());

        Ok(Frame {
            index: ep_info.dataset_from_index + frame_idx,
            episode_index: ep_info.episode_index,
            frame_index_in_episode: frame_idx,
            timestamp,
            action,
            state,
            images,
            reward,
            done,
            task_index,
            task,
        })
    }

    /// Extract a float array column at a specific row.
    fn extract_float_array(
        &self,
        data: &RecordBatch,
        column_name: &str,
        row_idx: usize,
    ) -> Result<Vec<f32>> {
        let col = data
            .column_by_name(column_name)
            .ok_or_else(|| DataCoreError::ColumnNotFound(column_name.to_string()))?;

        // Handle FixedSizeList<float32>
        if let Some(list_arr) = col.as_any().downcast_ref::<FixedSizeListArray>() {
            let values = list_arr.value(row_idx);
            if let Some(f32_arr) = values.as_any().downcast_ref::<Float32Array>() {
                return Ok(f32_arr.values().to_vec());
            }
        }

        // Handle List<float32> (v3 format before normalization)
        if let Some(list_arr) = col.as_any().downcast_ref::<ListArray>() {
            let values = list_arr.value(row_idx);
            if let Some(f32_arr) = values.as_any().downcast_ref::<Float32Array>() {
                return Ok(f32_arr.values().to_vec());
            }
        }

        Err(DataCoreError::SchemaMismatch(format!(
            "Column '{}' is not a float list",
            column_name
        )))
    }
}

// --- Iterators ---

/// Iterator over episode indices.
enum EpisodeIndexIterator<'a> {
    All(std::ops::Range<usize>),
    Selected(std::iter::Copied<std::slice::Iter<'a, usize>>),
}

impl<'a> Iterator for EpisodeIndexIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            EpisodeIndexIterator::All(range) => range.next(),
            EpisodeIndexIterator::Selected(iter) => iter.next(),
        }
    }
}

/// Iterator over frames in a lazy dataset.
struct FrameIterator<'a> {
    dataset: &'a LazyDataset,
    current_frame: usize,
    total_frames: usize,
}

impl<'a> FrameIterator<'a> {
    fn new(dataset: &'a LazyDataset) -> Self {
        Self {
            dataset,
            current_frame: 0,
            total_frames: dataset.num_frames(),
        }
    }
}

impl<'a> Iterator for FrameIterator<'a> {
    type Item = Result<Frame>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_frame >= self.total_frames {
            return None;
        }
        let frame = self.dataset.get_frame(self.current_frame);
        self.current_frame += 1;
        Some(frame)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_frames - self.current_frame;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for FrameIterator<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_dataset(dir: &std::path::Path) {
        let meta_dir = dir.join("meta");
        let data_dir = dir.join("data").join("chunk-000");
        fs::create_dir_all(&meta_dir).unwrap();
        fs::create_dir_all(&data_dir).unwrap();

        // info.json
        let info = serde_json::json!({
            "codebase_version": "v2.0",
            "fps": 10,
            "robot_type": "test",
            "total_episodes": 2,
            "total_frames": 20,
            "features": {
                "action": {"dtype": "float32", "shape": [7]}
            }
        });
        fs::write(
            meta_dir.join("info.json"),
            serde_json::to_string_pretty(&info).unwrap(),
        )
        .unwrap();

        // episodes.jsonl
        let mut episodes_file = fs::File::create(meta_dir.join("episodes.jsonl")).unwrap();
        writeln!(episodes_file, r#"{{"episode_index": 0, "task_index": 0, "dataset_from_index": 0, "dataset_to_index": 10, "length": 10}}"#).unwrap();
        writeln!(episodes_file, r#"{{"episode_index": 1, "task_index": 0, "dataset_from_index": 10, "dataset_to_index": 20, "length": 10}}"#).unwrap();

        // Create minimal parquet files
        use arrow::record_batch::RecordBatch;
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;

        for ep_idx in 0..2 {
            let schema = Arc::new(Schema::new(vec![
                Field::new("index", DataType::Int64, false),
                Field::new("timestamp", DataType::Float64, false),
                Field::new("episode_index", DataType::Int64, false),
                Field::new(
                    "action",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, false)),
                        7,
                    ),
                    false,
                ),
                Field::new("done", DataType::Boolean, false),
            ]));

            let n = 10;
            let indices =
                Int64Array::from((0..n).map(|i| (ep_idx * 10 + i) as i64).collect::<Vec<_>>());
            let timestamps =
                Float64Array::from((0..n).map(|i| i as f64 / 10.0).collect::<Vec<_>>());
            let ep_indices = Int64Array::from(vec![ep_idx as i64; n as usize]);
            let action_values = Float32Array::from(vec![0.0f32; n as usize * 7]);
            let actions = FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                7,
                Arc::new(action_values),
                None,
            )
            .unwrap();
            let mut dones = vec![false; n as usize];
            dones[n as usize - 1] = true;
            let dones = BooleanArray::from(dones);

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(indices),
                    Arc::new(timestamps),
                    Arc::new(ep_indices),
                    Arc::new(actions),
                    Arc::new(dones),
                ],
            )
            .unwrap();

            let path = data_dir.join(format!("episode_{:06}.parquet", ep_idx));
            let file = fs::File::create(&path).unwrap();
            let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }
    }

    #[test]
    fn test_lazy_dataset_open() {
        let tmp = TempDir::new().unwrap();
        create_test_dataset(tmp.path());

        let ds = LazyDataset::open(tmp.path()).unwrap();
        assert_eq!(ds.num_frames(), 20);
        assert_eq!(ds.num_episodes(), 2);
        assert_eq!(ds.fps(), 10.0);
    }

    /// Write a v2 dataset whose action vectors encode their position:
    /// `action[0] = 100 * episode + frame`. When `with_ranges` is false,
    /// episodes.jsonl only has `length`, as in real LeRobot v2.1 datasets.
    fn write_v2_dataset(dir: &std::path::Path, lengths: &[usize], with_ranges: bool) {
        use arrow::record_batch::RecordBatch;
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;

        let meta_dir = dir.join("meta");
        let data_dir = dir.join("data").join("chunk-000");
        fs::create_dir_all(&meta_dir).unwrap();
        fs::create_dir_all(&data_dir).unwrap();
        let total: usize = lengths.iter().sum();
        let info = serde_json::json!({
            "codebase_version": "v2.1",
            "fps": 10,
            "total_episodes": lengths.len(),
            "total_frames": total,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {"action": {"dtype": "float32", "shape": [2]}}
        });
        fs::write(meta_dir.join("info.json"), info.to_string()).unwrap();

        let mut jsonl = fs::File::create(meta_dir.join("episodes.jsonl")).unwrap();
        let mut from = 0;
        for (ep, &n) in lengths.iter().enumerate() {
            if with_ranges {
                writeln!(
                    jsonl,
                    r#"{{"episode_index": {ep}, "length": {n}, "dataset_from_index": {from}, "dataset_to_index": {}}}"#,
                    from + n
                )
                .unwrap();
            } else {
                writeln!(
                    jsonl,
                    r#"{{"episode_index": {ep}, "tasks": ["t"], "length": {n}}}"#
                )
                .unwrap();
            }

            let item = Arc::new(Field::new("item", DataType::Float32, false));
            let schema = Arc::new(Schema::new(vec![
                Field::new("index", DataType::Int64, false),
                Field::new("episode_index", DataType::Int64, false),
                Field::new("action", DataType::FixedSizeList(item.clone(), 2), false),
            ]));
            let actions: Vec<f32> = (0..n).flat_map(|f| [(100 * ep + f) as f32, 0.5]).collect();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(
                        (from..from + n).map(|i| i as i64).collect::<Vec<_>>(),
                    )),
                    Arc::new(Int64Array::from(vec![ep as i64; n])),
                    Arc::new(
                        FixedSizeListArray::try_new(
                            item,
                            2,
                            Arc::new(Float32Array::from(actions)),
                            None,
                        )
                        .unwrap(),
                    ),
                ],
            )
            .unwrap();
            let path = data_dir.join(format!("episode_{:06}.parquet", ep));
            let mut writer =
                ArrowWriter::try_new(fs::File::create(path).unwrap(), schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
            from += n;
        }
    }

    fn action0(frame: &Frame) -> usize {
        frame.action[0] as usize
    }

    #[test]
    fn test_get_frame_global_indexing() {
        let tmp = TempDir::new().unwrap();
        write_v2_dataset(tmp.path(), &[3, 5, 2], true);
        let ds = LazyDataset::open(tmp.path()).unwrap();
        assert_eq!(ds.num_frames(), 10);
        let expected = [0, 1, 2, 100, 101, 102, 103, 104, 200, 201];
        for (idx, &want) in expected.iter().enumerate() {
            let frame = ds.get_frame(idx).unwrap();
            assert_eq!(action0(&frame), want, "frame {idx}");
            assert_eq!(frame.index, idx);
            assert_eq!(frame.episode_index, want / 100);
            assert_eq!(frame.frame_index_in_episode, want % 100);
        }
        assert!(ds.get_frame(10).is_err());
        assert_eq!(ds.frames().count(), 10);
    }

    #[test]
    fn test_v2_episodes_without_ranges_resolve_all_frames() {
        // Regression: real v2.1 metadata (no dataset_from_index) and the v2
        // `data_path` template made every frame after the first episode
        // unreachable.
        let tmp = TempDir::new().unwrap();
        write_v2_dataset(tmp.path(), &[3, 5, 2], false);
        let ds = LazyDataset::open(tmp.path()).unwrap();
        let actions: Vec<usize> = ds.frames().map(|f| action0(&f.unwrap())).collect();
        assert_eq!(actions, vec![0, 1, 2, 100, 101, 102, 103, 104, 200, 201]);
    }

    #[test]
    fn test_with_episodes_indexes_selection_in_order() {
        let tmp = TempDir::new().unwrap();
        write_v2_dataset(tmp.path(), &[3, 5, 2], true);
        let ds = LazyDataset::open(tmp.path())
            .unwrap()
            .with_episodes(vec![2, 0]);
        assert_eq!(ds.num_episodes(), 2);
        assert_eq!(ds.num_frames(), 5);
        assert_eq!(ds.episode_indices().collect::<Vec<_>>(), vec![2, 0]);
        let actions: Vec<usize> = ds.frames().map(|f| action0(&f.unwrap())).collect();
        assert_eq!(actions, vec![200, 201, 0, 1, 2]);
        assert!(ds.get_frame(5).is_err());
        assert!(ds.get_episode(1).is_err());
    }

    #[test]
    fn test_get_frame_windowed_clamps_to_episode() {
        let tmp = TempDir::new().unwrap();
        write_v2_dataset(tmp.path(), &[3, 5, 2], true);
        let ds = LazyDataset::open(tmp.path()).unwrap();
        // Global frame 3 is the first frame of episode 1.
        let window = ds.get_frame_windowed(3, &[-2, -1, 0, 1, 10]).unwrap();
        let values = window
            .actions()
            .unwrap()
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .clone();
        let firsts: Vec<f32> = values.values().iter().step_by(2).copied().collect();
        assert_eq!(firsts, vec![100.0, 100.0, 100.0, 101.0, 104.0]);
        assert!(ds.get_frame_windowed(3, &[]).is_err());
    }

    #[test]
    fn test_cache_limit_bounds_cache_size() {
        let tmp = TempDir::new().unwrap();
        write_v2_dataset(tmp.path(), &[2, 2, 2, 2], true);
        let ds = LazyDataset::open(tmp.path()).unwrap().with_cache_limit(2);
        for ep in 0..4 {
            ds.get_episode(ep).unwrap();
            assert!(ds.episode_cache.read().unwrap().len() <= 2);
        }
        // Repeated access returns the cached Arc.
        let a = ds.get_episode(3).unwrap();
        let b = ds.get_episode(3).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        ds.clear_cache();
        assert!(ds.episode_cache.read().unwrap().is_empty());
    }

    #[test]
    fn test_preload_freezes_cache() {
        let tmp = TempDir::new().unwrap();
        write_v2_dataset(tmp.path(), &[2, 3, 4], true);
        let mut ds = LazyDataset::open(tmp.path()).unwrap();
        ds.preload_episodes(&[0, 2]).unwrap();
        let first = ds.get_episode(2).unwrap();
        assert!(Arc::ptr_eq(&first, &ds.get_episode(2).unwrap()));
        assert_eq!(first.num_steps(), 4);
        // Episode 1 was not preloaded and must not be decoded lazily.
        assert!(ds.get_episode(1).is_err());
        // Preloading again replaces the frozen set.
        ds.preload_episodes(&[1]).unwrap();
        assert!(ds.get_episode(1).is_ok());
        assert!(ds.get_episode(0).is_err());
        // A failed preload leaves no frozen cache behind.
        assert!(ds.preload_episodes(&[7]).is_err());
        assert!(ds.get_episode(0).is_ok());
    }

    #[test]
    fn test_lazy_dataset_with_episodes() {
        let tmp = TempDir::new().unwrap();
        create_test_dataset(tmp.path());

        let ds = LazyDataset::open(tmp.path())
            .unwrap()
            .with_episodes(vec![0]);
        assert_eq!(ds.num_frames(), 10);
        assert_eq!(ds.num_episodes(), 1);
    }
}
