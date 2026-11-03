//! Image augmentations over raw RGB pixel columns (and, for some, JPEG
//! columns).
//!
//! Raw columns are `observation.image.{cam}.rgb` (`FixedSizeList<UInt8>`, HWC,
//! one frame per row) with matching `observation.image.{cam}.width` and
//! `.height` columns. [`FusedImageAugmentation`] is the single-pass
//! combination of crop, color jitter, noise, and flip described in the paper
//! as fused augmentations.

use std::sync::Arc;

use arrow_array::*;
use arrow_schema::{DataType, Field};
use image::{DynamicImage, GenericImageView, ImageFormat};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use data_core::{DataCoreError, Episode, Result};

use crate::{check_magnitude, LocalAugmentation};

// ---------------------------------------------------------------------------
// Helpers for raw pixel columns
// ---------------------------------------------------------------------------

/// Copy the raw pixel bytes of row `idx` into a new `Vec<u8>`.
///
/// # Panics
///
/// Same conditions as [`raw_pixel_slice`].
pub fn raw_pixels_at(col: &FixedSizeListArray, idx: usize) -> Vec<u8> {
    let slice = raw_pixel_slice(col, idx);
    slice.to_vec()
}

/// Zero-copy access to the raw pixel bytes of row `idx`.
///
/// Returns a slice into the underlying Arrow buffer, with no allocation or
/// copy.
///
/// # Panics
///
/// Panics if the list values are not `UInt8` or if `idx` is out of range.
pub fn raw_pixel_slice(col: &FixedSizeListArray, idx: usize) -> &[u8] {
    let values = col.values();
    let u8_arr = values
        .as_any()
        .downcast_ref::<UInt8Array>()
        .expect("Expected UInt8Array in raw pixel column");
    let elem_size = col.value_length() as usize;
    let start = (col.value_offset(idx)) as usize;
    &u8_arr.values().as_ref()[start..start + elem_size]
}

/// Build a `FixedSizeListArray<UInt8>` with `pixel_count` bytes per row from
/// per-frame pixel buffers.
///
/// Each buffer should hold exactly `pixel_count` bytes; the buffers are
/// concatenated and split into rows of that size.
pub fn build_raw_column(
    frames: Vec<Vec<u8>>,
    pixel_count: usize,
) -> Result<Arc<FixedSizeListArray>> {
    let flat: Vec<u8> = frames.into_iter().flatten().collect();
    let values = UInt8Array::from(flat);
    let field = Arc::new(Field::new("item", DataType::UInt8, false));
    let list = FixedSizeListArray::try_new(field, pixel_count as i32, Arc::new(values), None)
        .map_err(DataCoreError::Arrow)?;
    Ok(Arc::new(list))
}

// ---------------------------------------------------------------------------
// RandomCrop
// ---------------------------------------------------------------------------

/// Random crop augmentation for image observations.
///
/// Operates on raw RGB pixel columns (`observation.image.{cam}.rgb`) when available,
/// falling back to JPEG columns. Raw pixel mode avoids JPEG decode/encode overhead.
///
/// When `window_size > 0`, the same crop position is applied to consecutive groups
/// of `window_size` frames (multi-frame consistency for delta_timestamps windows).
///
/// On the raw path the `.width` / `.height` columns are updated to the crop
/// size. On the JPEG path each frame is decoded, cropped, and re-encoded.
/// Cameras with neither column are skipped.
///
/// # Errors
///
/// `apply` fails if a source image is smaller than the crop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomCrop {
    /// Output height in pixels.
    pub height: u32,
    /// Output width in pixels.
    pub width: u32,
    /// Cameras to crop.
    pub camera_names: Vec<String>,
    /// When > 0, apply the same crop to groups of `window_size` consecutive frames.
    /// 0 means independent random crop per frame (default).
    #[serde(default)]
    pub window_size: usize,
}

impl RandomCrop {
    /// Create a crop to `height` x `width` for the given cameras, with an
    /// independent position per frame.
    pub fn new(height: u32, width: u32, camera_names: Vec<String>) -> Self {
        Self {
            height,
            width,
            camera_names,
            window_size: 0,
        }
    }

    /// Share one crop position across each group of `window_size` frames.
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Generate crop positions, respecting window_size for multi-frame consistency.
    fn generate_positions(
        &self,
        n: usize,
        src_width: u32,
        src_height: u32,
    ) -> Result<Vec<(u32, u32)>> {
        if src_width < self.width || src_height < self.height {
            return Err(DataCoreError::Augmentation {
                msg: format!(
                    "Image {}x{} smaller than crop {}x{}",
                    src_width, src_height, self.width, self.height
                ),
            });
        }

        let mut rng = rand::rng();
        let mut positions = Vec::with_capacity(n);

        if self.window_size > 0 {
            // Same crop per window group
            let mut i = 0;
            while i < n {
                let x = rng.random_range(0..=(src_width - self.width));
                let y = rng.random_range(0..=(src_height - self.height));
                let end = (i + self.window_size).min(n);
                for _ in i..end {
                    positions.push((x, y));
                }
                i = end;
            }
        } else {
            // Independent crop per frame
            for _ in 0..n {
                let x = rng.random_range(0..=(src_width - self.width));
                let y = rng.random_range(0..=(src_height - self.height));
                positions.push((x, y));
            }
        }

        Ok(positions)
    }

    /// Crop raw RGB pixels in-place (no decode/encode).
    fn apply_raw(
        &self,
        episode: &Episode,
        camera: &str,
    ) -> Result<(Arc<FixedSizeListArray>, Arc<Int64Array>, Arc<Int64Array>)> {
        let rgb_col_name = format!("observation.image.{camera}.rgb");
        let raw_col = episode
            .column(&rgb_col_name)?
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                DataCoreError::SchemaMismatch(format!("'{rgb_col_name}' not a FixedSizeListArray"))
            })?;

        let (src_width, src_height) = episode.image_dimensions(camera)?;
        let n = raw_col.len();
        let positions = self.generate_positions(n, src_width, src_height)?;

        let crop_w = self.width as usize;
        let crop_h = self.height as usize;
        let src_w = src_width as usize;
        let crop_pixel_count = crop_h * crop_w * 3;

        // Crop in parallel: pure memory copy, no decode
        let cropped: Vec<Vec<u8>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let pixels = raw_pixel_slice(raw_col, i);
                let (cx, cy) = positions[i];
                let cx = cx as usize;
                let cy = cy as usize;

                let mut out = vec![0u8; crop_pixel_count];
                for row in 0..crop_h {
                    let src_offset = ((cy + row) * src_w + cx) * 3;
                    let dst_offset = row * crop_w * 3;
                    out[dst_offset..dst_offset + crop_w * 3]
                        .copy_from_slice(&pixels[src_offset..src_offset + crop_w * 3]);
                }
                out
            })
            .collect();

        let new_col = build_raw_column(cropped, crop_pixel_count)?;
        let new_widths = Arc::new(Int64Array::from(vec![self.width as i64; n]));
        let new_heights = Arc::new(Int64Array::from(vec![self.height as i64; n]));

        Ok((new_col, new_widths, new_heights))
    }

    /// Crop JPEG images (decode, crop, re-encode).
    fn apply_jpeg(&self, episode: &Episode, camera: &str) -> Result<Arc<LargeBinaryArray>> {
        let col_name = format!("observation.image.{camera}");
        let images = episode
            .column(&col_name)?
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| {
                DataCoreError::SchemaMismatch(format!("'{col_name}' not a LargeBinaryArray"))
            })?;

        let n = images.len();

        // Get dimensions from first image
        let first =
            image::load_from_memory(images.value(0)).map_err(|e| DataCoreError::Augmentation {
                msg: format!("Image decode error: {e}"),
            })?;
        let (src_w, src_h) = first.dimensions();
        let positions = self.generate_positions(n, src_w, src_h)?;

        let new_images: Result<Vec<Vec<u8>>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let img = image::load_from_memory(images.value(i)).map_err(|e| {
                    DataCoreError::Augmentation {
                        msg: format!("Image decode error at step {i}: {e}"),
                    }
                })?;
                let (x, y) = positions[i];
                let cropped = img.crop_imm(x, y, self.width, self.height);
                let mut buf = Vec::new();
                cropped
                    .write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Jpeg)
                    .map_err(|e| DataCoreError::Augmentation {
                        msg: format!("Image encode error: {e}"),
                    })?;
                Ok(buf)
            })
            .collect();

        let new_images = new_images?;
        let refs: Vec<&[u8]> = new_images.iter().map(|v| v.as_slice()).collect();
        Ok(Arc::new(LargeBinaryArray::from(refs)))
    }
}

impl LocalAugmentation for RandomCrop {
    fn name(&self) -> &str {
        "random_crop"
    }

    fn apply(&self, episode: &Episode) -> Result<Episode> {
        let mut current = episode.clone();

        for camera in &self.camera_names {
            let rgb_col_name = format!("observation.image.{camera}.rgb");

            if episode.has_column(&rgb_col_name) {
                // Raw pixel path: no JPEG decode/encode
                let (new_col, new_widths, new_heights) = self.apply_raw(episode, camera)?;
                // Use replace_column since crop changes the FixedSizeList length
                current = current.replace_column(&rgb_col_name, new_col)?;
                current = current
                    .with_column(&format!("observation.image.{camera}.width"), new_widths)?;
                current = current
                    .with_column(&format!("observation.image.{camera}.height"), new_heights)?;
            } else {
                // JPEG fallback
                let col_name = format!("observation.image.{camera}");
                if !episode.has_column(&col_name) {
                    continue;
                }
                let new_col = self.apply_jpeg(episode, camera)?;
                current = current.with_column(&col_name, new_col)?;
            }
        }

        Ok(current)
    }
}

// ---------------------------------------------------------------------------
// ColorJitter
// ---------------------------------------------------------------------------

/// Color jitter augmentation: adjusts brightness, contrast, and saturation.
///
/// For each frame, factors are drawn as `1 + U(-x, x)` for each of
/// `brightness`, `contrast`, and `saturation`. Brightness scales the pixel
/// values, contrast scales around 128, and saturation interpolates between
/// the pixel and its luma (0.299 R + 0.587 G + 0.114 B).
///
/// Operates on raw RGB pixel columns when available, falling back to JPEG.
/// The JPEG path applies brightness and contrast only (saturation is
/// ignored there). When `window_size > 0`, the same jitter factors apply to
/// groups of frames.
///
/// # Errors
///
/// `apply` fails if any of the three magnitudes is negative or NaN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorJitter {
    /// Maximum relative brightness change.
    pub brightness: f32,
    /// Maximum relative contrast change.
    pub contrast: f32,
    /// Maximum relative saturation change.
    pub saturation: f32,
    /// Cameras to jitter.
    pub camera_names: Vec<String>,
    /// When > 0, groups of this many consecutive frames share jitter factors.
    #[serde(default)]
    pub window_size: usize,
}

impl ColorJitter {
    /// Create a color jitter with per-frame factors.
    pub fn new(brightness: f32, contrast: f32, saturation: f32, camera_names: Vec<String>) -> Self {
        Self {
            brightness,
            contrast,
            saturation,
            camera_names,
            window_size: 0,
        }
    }

    /// Share jitter factors across each group of `window_size` frames.
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Generate jitter factors for n frames, respecting window_size.
    fn generate_factors(&self, n: usize) -> Vec<(f32, f32, f32)> {
        let mut rng = rand::rng();
        let mut factors = Vec::with_capacity(n);

        if self.window_size > 0 {
            let mut i = 0;
            while i < n {
                let b = 1.0 + rng.random_range(-self.brightness..=self.brightness);
                let c = 1.0 + rng.random_range(-self.contrast..=self.contrast);
                let s = 1.0 + rng.random_range(-self.saturation..=self.saturation);
                let end = (i + self.window_size).min(n);
                for _ in i..end {
                    factors.push((b, c, s));
                }
                i = end;
            }
        } else {
            for _ in 0..n {
                let b = 1.0 + rng.random_range(-self.brightness..=self.brightness);
                let c = 1.0 + rng.random_range(-self.contrast..=self.contrast);
                let s = 1.0 + rng.random_range(-self.saturation..=self.saturation);
                factors.push((b, c, s));
            }
        }

        factors
    }

    /// Apply jitter to raw RGB pixels.
    fn apply_raw(&self, episode: &Episode, camera: &str) -> Result<Arc<FixedSizeListArray>> {
        let rgb_col_name = format!("observation.image.{camera}.rgb");
        let raw_col = episode
            .column(&rgb_col_name)?
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                DataCoreError::SchemaMismatch(format!("'{rgb_col_name}' not a FixedSizeListArray"))
            })?;

        let n = raw_col.len();
        let pixel_count = raw_col.value_length() as usize;
        let factors = self.generate_factors(n);

        let jittered: Vec<Vec<u8>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let pixels = raw_pixel_slice(raw_col, i);
                let (bf, cf, sf) = factors[i];
                let mut out = Vec::with_capacity(pixel_count);

                for px in 0..pixel_count / 3 {
                    let r = pixels[px * 3] as f32;
                    let g = pixels[px * 3 + 1] as f32;
                    let b = pixels[px * 3 + 2] as f32;

                    // Brightness
                    let (mut r, mut g, mut b) = (r * bf, g * bf, b * bf);

                    // Contrast (around 128)
                    r = 128.0 + (r - 128.0) * cf;
                    g = 128.0 + (g - 128.0) * cf;
                    b = 128.0 + (b - 128.0) * cf;

                    // Saturation (lerp toward grayscale)
                    let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                    r = gray + (r - gray) * sf;
                    g = gray + (g - gray) * sf;
                    b = gray + (b - gray) * sf;

                    out.push(r.clamp(0.0, 255.0) as u8);
                    out.push(g.clamp(0.0, 255.0) as u8);
                    out.push(b.clamp(0.0, 255.0) as u8);
                }
                out
            })
            .collect();

        build_raw_column(jittered, pixel_count)
    }

    /// Apply brightness and contrast jitter to JPEG images (decode, jitter, re-encode).
    fn apply_jpeg(&self, episode: &Episode, camera: &str) -> Result<Arc<LargeBinaryArray>> {
        let col_name = format!("observation.image.{camera}");
        let images = episode
            .column(&col_name)?
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| {
                DataCoreError::SchemaMismatch(format!("'{col_name}' not a LargeBinaryArray"))
            })?;

        let n = images.len();
        let factors = self.generate_factors(n);

        let new_images: Result<Vec<Vec<u8>>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let img = image::load_from_memory(images.value(i)).map_err(|e| {
                    DataCoreError::Augmentation {
                        msg: format!("Image decode error at step {i}: {e}"),
                    }
                })?;
                let (bf, cf, _sf) = factors[i];
                let mut rgba = img.to_rgba8();
                for pixel in rgba.pixels_mut() {
                    for c in 0..3 {
                        let val = pixel[c] as f32;
                        let val = val * bf;
                        let val = 128.0 + (val - 128.0) * cf;
                        pixel[c] = val.clamp(0.0, 255.0) as u8;
                    }
                }
                let mut buf = Vec::new();
                DynamicImage::ImageRgba8(rgba)
                    .write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Jpeg)
                    .map_err(|e| DataCoreError::Augmentation {
                        msg: format!("Image encode error: {e}"),
                    })?;
                Ok(buf)
            })
            .collect();

        let new_images = new_images?;
        let refs: Vec<&[u8]> = new_images.iter().map(|v| v.as_slice()).collect();
        Ok(Arc::new(LargeBinaryArray::from(refs)))
    }
}

impl LocalAugmentation for ColorJitter {
    fn name(&self) -> &str {
        "color_jitter"
    }

    fn apply(&self, episode: &Episode) -> Result<Episode> {
        check_magnitude(self.name(), "brightness", self.brightness)?;
        check_magnitude(self.name(), "contrast", self.contrast)?;
        check_magnitude(self.name(), "saturation", self.saturation)?;
        let mut current = episode.clone();

        for camera in &self.camera_names {
            let rgb_col_name = format!("observation.image.{camera}.rgb");

            if episode.has_column(&rgb_col_name) {
                let new_col = self.apply_raw(episode, camera)?;
                current = current.with_column(&rgb_col_name, new_col)?;
            } else {
                let col_name = format!("observation.image.{camera}");
                if !episode.has_column(&col_name) {
                    continue;
                }
                let new_col = self.apply_jpeg(episode, camera)?;
                current = current.with_column(&col_name, new_col)?;
            }
        }

        Ok(current)
    }
}

// ---------------------------------------------------------------------------
// GaussianBlur
// ---------------------------------------------------------------------------

/// Gaussian blur with a fixed `sigma`.
///
/// Operates only on encoded JPEG columns (`observation.image.{cam}`): each
/// frame is decoded, blurred, and re-encoded. Raw `.rgb` columns are not
/// modified. Cameras without a JPEG column are skipped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianBlur {
    /// Standard deviation of the blur kernel, in pixels.
    pub sigma: f32,
    /// Cameras to blur.
    pub camera_names: Vec<String>,
}

impl GaussianBlur {
    /// Create a blur augmentation.
    pub fn new(sigma: f32, camera_names: Vec<String>) -> Self {
        Self {
            sigma,
            camera_names,
        }
    }
}

impl LocalAugmentation for GaussianBlur {
    fn name(&self) -> &str {
        "gaussian_blur"
    }

    fn apply(&self, episode: &Episode) -> Result<Episode> {
        let mut current = episode.clone();

        for camera in &self.camera_names {
            let col_name = format!("observation.image.{camera}");
            let images = match episode.column(&col_name) {
                Ok(col) => col
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .ok_or_else(|| {
                        DataCoreError::SchemaMismatch(format!(
                            "'{col_name}' is not a LargeBinaryArray"
                        ))
                    })?,
                Err(_) => continue,
            };

            let new_images: Result<Vec<Vec<u8>>> = (0..images.len())
                .into_par_iter()
                .map(|i| {
                    let img = image::load_from_memory(images.value(i)).map_err(|e| {
                        DataCoreError::Augmentation {
                            msg: format!("Image decode error at step {i}: {e}"),
                        }
                    })?;
                    let blurred = img.blur(self.sigma);
                    let mut buf = Vec::new();
                    blurred
                        .write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Jpeg)
                        .map_err(|e| DataCoreError::Augmentation {
                            msg: format!("Image encode error: {e}"),
                        })?;
                    Ok(buf)
                })
                .collect();

            let new_images = new_images?;
            let refs: Vec<&[u8]> = new_images.iter().map(|v| v.as_slice()).collect();
            let new_col = Arc::new(LargeBinaryArray::from(refs));
            current = current.with_column(&col_name, new_col)?;
        }

        Ok(current)
    }
}

// ---------------------------------------------------------------------------
// GaussianNoise
// ---------------------------------------------------------------------------

/// Adds independent per-channel noise to raw RGB pixels.
///
/// Despite the name, each channel value is perturbed by `U(-std, std) * 255`
/// (uniform, not Gaussian) and clamped to `[0, 255]`. Only raw `.rgb` columns
/// are modified.
///
/// `window_size` is accepted for API symmetry with the other image
/// augmentations, but in the current implementation each frame's noise seed
/// also mixes in a fresh random draw, so frames within a window group receive
/// different noise patterns.
///
/// # Errors
///
/// `apply` fails if `std` is negative or NaN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianNoise {
    /// Half-width of the uniform noise range, as a fraction of 255.
    pub std: f32,
    /// Cameras to perturb.
    pub camera_names: Vec<String>,
    /// Window grouping; see the type-level note.
    #[serde(default)]
    pub window_size: usize,
}

impl GaussianNoise {
    /// Create a pixel noise augmentation.
    pub fn new(std: f32, camera_names: Vec<String>) -> Self {
        Self {
            std,
            camera_names,
            window_size: 0,
        }
    }

    /// Set the window grouping; see the type-level note.
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }
}

impl LocalAugmentation for GaussianNoise {
    fn name(&self) -> &str {
        "gaussian_noise"
    }

    fn apply(&self, episode: &Episode) -> Result<Episode> {
        check_magnitude(self.name(), "std", self.std)?;
        let mut current = episode.clone();

        for camera in &self.camera_names {
            let rgb_col_name = format!("observation.image.{camera}.rgb");
            if !episode.has_column(&rgb_col_name) {
                continue;
            }

            let raw_col = episode
                .column(&rgb_col_name)?
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    DataCoreError::SchemaMismatch(format!(
                        "'{rgb_col_name}' not a FixedSizeListArray"
                    ))
                })?;

            let n = raw_col.len();
            let pixel_count = raw_col.value_length() as usize;
            let std_val = self.std;
            let window = self.window_size;

            let noisy: Vec<Vec<u8>> = (0..n)
                .into_par_iter()
                .map_init(rand::rng, |rng, i| {
                    let pixels = raw_pixel_slice(raw_col, i);

                    if let Some(group) = i.checked_div(window) {
                        let mut wrng = rand::rngs::StdRng::seed_from_u64(
                            rng.random::<u64>()
                                .wrapping_add(group as u64 * 0x9E3779B97F4A7C15),
                        );
                        pixels
                            .iter()
                            .map(|&v| {
                                let noise = wrng.random_range(-std_val..=std_val)
                                    * (2.0f32).sqrt()
                                    * std::f32::consts::FRAC_1_SQRT_2;
                                (v as f32 + noise * 255.0).clamp(0.0, 255.0) as u8
                            })
                            .collect()
                    } else {
                        pixels
                            .iter()
                            .map(|&v| {
                                let noise = rng.random_range(-std_val..=std_val)
                                    * (2.0f32).sqrt()
                                    * std::f32::consts::FRAC_1_SQRT_2;
                                (v as f32 + noise * 255.0).clamp(0.0, 255.0) as u8
                            })
                            .collect()
                    }
                })
                .collect();

            let new_col = build_raw_column(noisy, pixel_count)?;
            current = current.with_column(&rgb_col_name, new_col)?;
        }

        Ok(current)
    }
}

// ---------------------------------------------------------------------------
// HorizontalFlip
// ---------------------------------------------------------------------------

/// Random horizontal flip with coordinated action mirroring.
///
/// Each frame (or each window group when `window_size > 0`) is flipped with
/// probability `p`. The same decision is used for every camera and for the
/// action column, so images and actions stay consistent. For flipped frames,
/// the action dimensions listed in `action_mirror_dims` are negated;
/// dimensions outside the action length are ignored, and the output action
/// column has no null buffer. Only raw `.rgb` columns are flipped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizontalFlip {
    /// Probability of flipping (0.0 to 1.0).
    pub p: f32,
    /// Cameras to flip.
    pub camera_names: Vec<String>,
    /// Action dimensions to negate when flipping (e.g., `[0, 1, 2, 3, 4, 5, 6]` for one arm).
    #[serde(default)]
    pub action_mirror_dims: Vec<usize>,
    /// When > 0, groups of this many consecutive frames share the flip decision.
    #[serde(default)]
    pub window_size: usize,
}

impl HorizontalFlip {
    /// Create a flip with probability `p` and no action mirroring.
    pub fn new(p: f32, camera_names: Vec<String>) -> Self {
        Self {
            p,
            camera_names,
            action_mirror_dims: Vec::new(),
            window_size: 0,
        }
    }

    /// Negate these action dimensions on flipped frames.
    pub fn with_action_mirror_dims(mut self, dims: Vec<usize>) -> Self {
        self.action_mirror_dims = dims;
        self
    }

    /// Share one flip decision across each group of `window_size` frames.
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }
}

impl LocalAugmentation for HorizontalFlip {
    fn name(&self) -> &str {
        "horizontal_flip"
    }

    fn apply(&self, episode: &Episode) -> Result<Episode> {
        let n = episode.num_steps();
        let mut rng = rand::rng();

        // Decide flip/no-flip per frame (or per window group)
        let flips: Vec<bool> = if self.window_size > 0 {
            let mut v = Vec::with_capacity(n);
            let mut i = 0;
            while i < n {
                let flip = rng.random_range(0.0f32..1.0) < self.p;
                let end = (i + self.window_size).min(n);
                for _ in i..end {
                    v.push(flip);
                }
                i = end;
            }
            v
        } else {
            (0..n)
                .map(|_| rng.random_range(0.0f32..1.0) < self.p)
                .collect()
        };

        let mut current = episode.clone();

        // Flip images
        for camera in &self.camera_names {
            let rgb_col_name = format!("observation.image.{camera}.rgb");
            if !episode.has_column(&rgb_col_name) {
                continue;
            }

            let raw_col = episode
                .column(&rgb_col_name)?
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    DataCoreError::SchemaMismatch(format!(
                        "'{rgb_col_name}' not a FixedSizeListArray"
                    ))
                })?;

            let (width, height) = episode.image_dimensions(camera)?;
            let w = width as usize;
            let h = height as usize;
            let pixel_count = raw_col.value_length() as usize;

            let flipped: Vec<Vec<u8>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let pixels = raw_pixel_slice(raw_col, i);
                    if !flips[i] {
                        return pixels.to_vec();
                    }
                    // Horizontal flip: reverse each row
                    let mut out = vec![0u8; pixel_count];
                    for y in 0..h {
                        for x in 0..w {
                            let src = (y * w + x) * 3;
                            let dst = (y * w + (w - 1 - x)) * 3;
                            out[dst] = pixels[src];
                            out[dst + 1] = pixels[src + 1];
                            out[dst + 2] = pixels[src + 2];
                        }
                    }
                    out
                })
                .collect();

            let new_col = build_raw_column(flipped, pixel_count)?;
            current = current.with_column(&rgb_col_name, new_col)?;
        }

        // Mirror action dimensions
        if !self.action_mirror_dims.is_empty() {
            if let Ok(actions) = episode.actions() {
                let list_size = actions.value_length() as usize;
                let values = actions
                    .values()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        DataCoreError::SchemaMismatch("action values not Float32".into())
                    })?;

                let mirror_dims = &self.action_mirror_dims;
                let flips_ref = &flips;
                let new_values: Vec<f32> = (0..n)
                    .flat_map(|i| {
                        (0..list_size).map(move |d| {
                            let val = values.value(i * list_size + d);
                            if flips_ref[i] && mirror_dims.contains(&d) {
                                -val
                            } else {
                                val
                            }
                        })
                    })
                    .collect();

                let new_arr = Float32Array::from(new_values);
                let inner_field = match actions.data_type() {
                    DataType::FixedSizeList(f, _) => f.clone(),
                    _ => Arc::new(Field::new("item", DataType::Float32, false)),
                };
                let new_actions = FixedSizeListArray::try_new(
                    inner_field,
                    list_size as i32,
                    Arc::new(new_arr),
                    None,
                )
                .map_err(DataCoreError::Arrow)?;

                current = current.with_column("action", Arc::new(new_actions))?;
            }
        }

        Ok(current)
    }
}

// ---------------------------------------------------------------------------
// FusedImageAugmentation
// ---------------------------------------------------------------------------

/// Fused image augmentation: crop, color jitter, noise, and horizontal flip
/// in a single pass over the raw pixels.
///
/// Instead of four separate read, allocate, write, and rebuild-episode
/// cycles, this reads the source pixels once, applies all enabled
/// transforms, and writes once per camera.
///
/// Per-frame random parameters (crop offset, jitter factors, flip decision,
/// noise seed) are drawn once and shared by all cameras. When
/// `window_size > 0`, each group of `window_size` consecutive frames shares
/// the crop offset, jitter factors, and flip decision; the noise pattern is
/// seeded per frame and therefore differs between frames.
///
/// The transforms match [`RandomCrop`], [`ColorJitter`] (raw path),
/// [`GaussianNoise`] (uniform noise), and [`HorizontalFlip`]. Only raw `.rgb`
/// columns are processed; cameras without one are skipped.
///
/// # Errors
///
/// `apply` fails if `camera_names` is empty, if the first camera has no
/// width/height columns, if the cameras do not all have the same
/// dimensions, if the crop is larger than the image, or if jitter is enabled
/// with a negative or NaN magnitude.
///
/// # Example
///
/// ```
/// use data_core_augmentations::image::FusedImageAugmentation;
///
/// let aug = FusedImageAugmentation::new(vec!["top".into(), "wrist".into()])
///     .with_crop(84, 84)
///     .with_jitter(0.2, 0.2, 0.2)
///     .with_flip(0.5, vec![0])
///     .with_window_size(4);
/// assert_eq!(aug.crop_height, Some(84));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedImageAugmentation {
    /// Cameras to process. All must have the same image dimensions.
    pub camera_names: Vec<String>,
    /// When > 0, groups of this many consecutive frames share parameters.
    #[serde(default)]
    pub window_size: usize,

    /// Crop height; cropping is enabled only when both crop fields are set.
    pub crop_height: Option<u32>,
    /// Crop width; cropping is enabled only when both crop fields are set.
    pub crop_width: Option<u32>,

    /// Brightness jitter magnitude. Jitter is enabled if any of the three
    /// magnitudes is positive.
    #[serde(default)]
    pub brightness: f32,
    /// Contrast jitter magnitude.
    #[serde(default)]
    pub contrast: f32,
    /// Saturation jitter magnitude.
    #[serde(default)]
    pub saturation: f32,

    /// Uniform pixel noise magnitude as a fraction of 255 (0 disables noise).
    #[serde(default)]
    pub noise_std: f32,

    /// Flip probability (0 disables flipping).
    #[serde(default)]
    pub flip_p: f32,
    /// Action dimensions to negate on flipped frames.
    #[serde(default)]
    pub action_mirror_dims: Vec<usize>,

    /// Output CHW layout (3×H×W) instead of HWC (H×W×3).
    /// When true, the output is ready for torch tensor wrapping without transpose.
    #[serde(default)]
    pub chw_output: bool,
}

impl FusedImageAugmentation {
    /// Create a fused augmentation with every transform disabled.
    pub fn new(camera_names: Vec<String>) -> Self {
        Self {
            camera_names,
            window_size: 0,
            crop_height: None,
            crop_width: None,
            brightness: 0.0,
            contrast: 0.0,
            saturation: 0.0,
            noise_std: 0.0,
            flip_p: 0.0,
            action_mirror_dims: Vec::new(),
            chw_output: false,
        }
    }

    /// Enable random cropping to `height` x `width`.
    pub fn with_crop(mut self, height: u32, width: u32) -> Self {
        self.crop_height = Some(height);
        self.crop_width = Some(width);
        self
    }

    /// Set the color jitter magnitudes.
    pub fn with_jitter(mut self, brightness: f32, contrast: f32, saturation: f32) -> Self {
        self.brightness = brightness;
        self.contrast = contrast;
        self.saturation = saturation;
        self
    }

    /// Set the pixel noise magnitude.
    pub fn with_noise(mut self, std: f32) -> Self {
        self.noise_std = std;
        self
    }

    /// Set the flip probability and the action dimensions to mirror.
    pub fn with_flip(mut self, p: f32, mirror_dims: Vec<usize>) -> Self {
        self.flip_p = p;
        self.action_mirror_dims = mirror_dims;
        self
    }

    /// Share parameters across each group of `ws` frames.
    pub fn with_window_size(mut self, ws: usize) -> Self {
        self.window_size = ws;
        self
    }
}

impl LocalAugmentation for FusedImageAugmentation {
    fn name(&self) -> &str {
        "fused_image"
    }

    fn apply(&self, episode: &Episode) -> Result<Episode> {
        let n = episode.num_steps();
        let mut rng = rand::rng();

        let has_crop = self.crop_height.is_some() && self.crop_width.is_some();
        let has_jitter = self.brightness > 0.0 || self.contrast > 0.0 || self.saturation > 0.0;
        let has_noise = self.noise_std > 0.0;
        let has_flip = self.flip_p > 0.0;

        if has_jitter {
            check_magnitude(self.name(), "brightness", self.brightness)?;
            check_magnitude(self.name(), "contrast", self.contrast)?;
            check_magnitude(self.name(), "saturation", self.saturation)?;
        }

        // Generate per-frame (or per-window) random params upfront
        struct FrameParams {
            crop_x: u32,
            crop_y: u32,
            brightness: f32,
            contrast: f32,
            saturation: f32,
            flip: bool,
            noise_seed: u64,
        }

        let crop_w = self.crop_width.unwrap_or(0);
        let crop_h = self.crop_height.unwrap_or(0);

        // Source dimensions are needed even without a crop: out_w/out_h fall
        // back to (src_width, src_height) when has_crop is false. The first
        // camera's dimensions are used for all cameras (checked below).
        let (src_width, src_height) = {
            let cam = self
                .camera_names
                .first()
                .ok_or_else(|| DataCoreError::Augmentation {
                    msg: "fused_image: camera_names must not be empty".to_string(),
                })?;
            episode.image_dimensions(cam)?
        };

        if has_crop && (src_width < crop_w || src_height < crop_h) {
            return Err(DataCoreError::Augmentation {
                msg: format!(
                    "Image {}x{} smaller than crop {}x{}",
                    src_width, src_height, crop_w, crop_h
                ),
            });
        }

        let mut params: Vec<FrameParams> = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let p = FrameParams {
                crop_x: if has_crop {
                    rng.random_range(0..=(src_width - crop_w))
                } else {
                    0
                },
                crop_y: if has_crop {
                    rng.random_range(0..=(src_height - crop_h))
                } else {
                    0
                },
                brightness: if has_jitter {
                    1.0 + rng.random_range(-self.brightness..=self.brightness)
                } else {
                    1.0
                },
                contrast: if has_jitter {
                    1.0 + rng.random_range(-self.contrast..=self.contrast)
                } else {
                    1.0
                },
                saturation: if has_jitter {
                    1.0 + rng.random_range(-self.saturation..=self.saturation)
                } else {
                    1.0
                },
                flip: if has_flip {
                    rng.random_range(0.0f32..1.0) < self.flip_p
                } else {
                    false
                },
                noise_seed: if has_noise { rng.random::<u64>() } else { 0 },
            };

            let end = if self.window_size > 0 {
                (i + self.window_size).min(n)
            } else {
                i + 1
            };
            for _ in i..end {
                params.push(FrameParams {
                    crop_x: p.crop_x,
                    crop_y: p.crop_y,
                    brightness: p.brightness,
                    contrast: p.contrast,
                    saturation: p.saturation,
                    flip: p.flip,
                    noise_seed: p.noise_seed,
                });
            }
            i = end;
        }

        let mut current = episode.clone();

        // Process each camera
        for camera in &self.camera_names {
            let rgb_col_name = format!("observation.image.{camera}.rgb");
            if !episode.has_column(&rgb_col_name) {
                continue;
            }

            let raw_col = episode
                .column(&rgb_col_name)?
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    DataCoreError::SchemaMismatch(format!(
                        "'{rgb_col_name}' not a FixedSizeListArray"
                    ))
                })?;

            // All cameras share the first camera's dimensions and crop
            // offsets; a camera with different dimensions would be indexed
            // with the wrong stride (or out of bounds), so reject it.
            let same_dims = raw_col.value_length() as usize
                == (src_width * src_height * 3) as usize
                && episode
                    .image_dimensions(camera)
                    .map(|dims| dims == (src_width, src_height))
                    .unwrap_or(true);
            if !same_dims {
                return Err(DataCoreError::Augmentation {
                    msg: format!(
                        "fused_image: camera '{camera}' has {} bytes per frame, but the \
                         first camera is {src_width}x{src_height}; all cameras must have \
                         the same dimensions",
                        raw_col.value_length()
                    ),
                });
            }

            let src_w = src_width as usize;
            let out_w = if has_crop {
                crop_w as usize
            } else {
                src_width as usize
            };
            let out_h = if has_crop {
                crop_h as usize
            } else {
                src_height as usize
            };
            let out_pixel_count = out_w * out_h * 3;
            let noise_std = self.noise_std;
            let chw = self.chw_output;

            // Single fused pass per frame, parallel across frames
            let processed: Vec<Vec<u8>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let src = raw_pixel_slice(raw_col, i);
                    let p = &params[i];
                    let mut out = vec![0u8; out_pixel_count];

                    let cx = p.crop_x as usize;
                    let cy = p.crop_y as usize;

                    // Optional noise RNG
                    let mut noise_rng = if has_noise {
                        Some(rand::rngs::SmallRng::seed_from_u64(
                            p.noise_seed.wrapping_add(i as u64),
                        ))
                    } else {
                        None
                    };

                    for row in 0..out_h {
                        for col in 0..out_w {
                            // Source pixel (with crop offset)
                            let src_x = if has_crop { cx + col } else { col };
                            let src_y = if has_crop { cy + row } else { row };
                            let si = (src_y * src_w + src_x) * 3;

                            // Destination pixel (with optional flip)
                            let dst_x = if p.flip { out_w - 1 - col } else { col };
                            let di = (row * out_w + dst_x) * 3;

                            let mut r = src[si] as f32;
                            let mut g = src[si + 1] as f32;
                            let mut b = src[si + 2] as f32;

                            // Brightness + contrast
                            if has_jitter {
                                r *= p.brightness;
                                g *= p.brightness;
                                b *= p.brightness;
                                r = 128.0 + (r - 128.0) * p.contrast;
                                g = 128.0 + (g - 128.0) * p.contrast;
                                b = 128.0 + (b - 128.0) * p.contrast;
                                // Saturation
                                let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                                r = gray + (r - gray) * p.saturation;
                                g = gray + (g - gray) * p.saturation;
                                b = gray + (b - gray) * p.saturation;
                            }

                            // Noise
                            if let Some(ref mut nrng) = noise_rng {
                                let scale = noise_std
                                    * 255.0
                                    * std::f32::consts::SQRT_2
                                    * std::f32::consts::FRAC_1_SQRT_2;
                                r += nrng.random_range(-scale..=scale);
                                g += nrng.random_range(-scale..=scale);
                                b += nrng.random_range(-scale..=scale);
                            }

                            let rc = r.clamp(0.0, 255.0) as u8;
                            let gc = g.clamp(0.0, 255.0) as u8;
                            let bc = b.clamp(0.0, 255.0) as u8;

                            if chw {
                                // CHW layout: [C, H, W], channel-first
                                let base = row * out_w + dst_x;
                                out[base] = rc;
                                out[out_h * out_w + base] = gc;
                                out[2 * out_h * out_w + base] = bc;
                            } else {
                                // HWC layout: [H, W, C], the default
                                out[di] = rc;
                                out[di + 1] = gc;
                                out[di + 2] = bc;
                            }
                        }
                    }
                    out
                })
                .collect();

            let new_col = build_raw_column(processed, out_pixel_count)?;
            if has_crop {
                current = current.replace_column(&rgb_col_name, new_col)?;
                current = current.with_column(
                    &format!("observation.image.{camera}.width"),
                    Arc::new(Int64Array::from(vec![crop_w as i64; n])),
                )?;
                current = current.with_column(
                    &format!("observation.image.{camera}.height"),
                    Arc::new(Int64Array::from(vec![crop_h as i64; n])),
                )?;
            } else {
                current = current.with_column(&rgb_col_name, new_col)?;
            }
        }

        // Mirror action dimensions for flipped frames
        if has_flip && !self.action_mirror_dims.is_empty() {
            if let Ok(actions) = episode.actions() {
                let list_size = actions.value_length() as usize;
                let values = actions
                    .values()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        DataCoreError::SchemaMismatch("action values not Float32".into())
                    })?;

                let mirror_dims = &self.action_mirror_dims;
                let params_ref = &params;
                let new_values: Vec<f32> = (0..n)
                    .flat_map(|i| {
                        (0..list_size).map(move |d| {
                            let val = values.value(i * list_size + d);
                            if params_ref[i].flip && mirror_dims.contains(&d) {
                                -val
                            } else {
                                val
                            }
                        })
                    })
                    .collect();

                let new_arr = Float32Array::from(new_values);
                let inner_field = match actions.data_type() {
                    DataType::FixedSizeList(f, _) => f.clone(),
                    _ => Arc::new(Field::new("item", DataType::Float32, false)),
                };
                let new_actions = FixedSizeListArray::try_new(
                    inner_field,
                    list_size as i32,
                    Arc::new(new_arr),
                    None,
                )
                .map_err(DataCoreError::Arrow)?;
                current = current.with_column("action", Arc::new(new_actions))?;
            }
        }

        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::record_batch::RecordBatch;
    use arrow_schema::{DataType, Field, Schema};
    use data_core::EpisodeMetadata;

    fn make_test_episode_jpeg(n: usize) -> Episode {
        let schema = Arc::new(Schema::new(vec![
            Field::new("step_index", DataType::Int64, false),
            Field::new("observation.image.cam_0", DataType::LargeBinary, false),
            Field::new("episode_index", DataType::Int64, false),
        ]));

        let steps = Int64Array::from((0..n as i64).collect::<Vec<_>>());
        let jpeg = {
            let mut buf = Vec::new();
            let img = image::DynamicImage::new_rgb8(8, 8);
            img.write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Jpeg)
                .unwrap();
            buf
        };
        let images = LargeBinaryArray::from(vec![jpeg.as_slice(); n]);
        let ep_indices = Int64Array::from(vec![0i64; n]);

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(steps), Arc::new(images), Arc::new(ep_indices)],
        )
        .unwrap();

        Episode::new(
            batch,
            EpisodeMetadata {
                episode_index: 0,
                task: None,
                length: n,
            },
        )
        .unwrap()
    }

    fn make_test_episode_raw(n: usize, width: u32, height: u32) -> Episode {
        let pixel_count = (width * height * 3) as i32;
        let schema = Arc::new(Schema::new(vec![
            Field::new("step_index", DataType::Int64, false),
            Field::new(
                "observation.image.cam_0.rgb",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, false)),
                    pixel_count,
                ),
                false,
            ),
            Field::new("observation.image.cam_0.width", DataType::Int64, false),
            Field::new("observation.image.cam_0.height", DataType::Int64, false),
            Field::new("episode_index", DataType::Int64, false),
        ]));

        let steps = Int64Array::from((0..n as i64).collect::<Vec<_>>());

        // Create gradient test pixels
        let mut all_pixels = Vec::with_capacity(n * pixel_count as usize);
        for frame in 0..n {
            for y in 0..height {
                for x in 0..width {
                    all_pixels.push(((x * 255 / width) as u8).wrapping_add(frame as u8)); // R
                    all_pixels.push(((y * 255 / height) as u8).wrapping_add(frame as u8)); // G
                    all_pixels.push(128u8); // B
                }
            }
        }
        let values = UInt8Array::from(all_pixels);
        let field = Arc::new(Field::new("item", DataType::UInt8, false));
        let rgb_col =
            FixedSizeListArray::try_new(field, pixel_count, Arc::new(values), None).unwrap();

        let widths = Int64Array::from(vec![width as i64; n]);
        let heights = Int64Array::from(vec![height as i64; n]);
        let ep_indices = Int64Array::from(vec![0i64; n]);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(steps),
                Arc::new(rgb_col),
                Arc::new(widths),
                Arc::new(heights),
                Arc::new(ep_indices),
            ],
        )
        .unwrap();

        Episode::new(
            batch,
            EpisodeMetadata {
                episode_index: 0,
                task: None,
                length: n,
            },
        )
        .unwrap()
    }

    #[test]
    fn test_random_crop_raw() {
        let ep = make_test_episode_raw(4, 16, 16);
        let aug = RandomCrop::new(8, 8, vec!["cam_0".to_string()]);
        let out = aug.apply(&ep).unwrap();
        assert_eq!(out.num_steps(), 4);

        // Check output dimensions
        let (w, h) = out.image_dimensions("cam_0").unwrap();
        assert_eq!((w, h), (8, 8));

        // Check pixel count
        let rgb = out.image_raw("cam_0").unwrap();
        assert_eq!(rgb.value_length(), 8 * 8 * 3);
    }

    #[test]
    fn test_random_crop_window_consistency() {
        let ep = make_test_episode_raw(8, 16, 16);
        let aug = RandomCrop::new(8, 8, vec!["cam_0".to_string()]).with_window_size(4);
        let out = aug.apply(&ep).unwrap();

        // Frames 0-3 should have same crop position, frames 4-7 different
        let rgb = out.image_raw("cam_0").unwrap();
        let f0 = raw_pixels_at(rgb, 0);
        let f3 = raw_pixels_at(rgb, 3);
        let f4 = raw_pixels_at(rgb, 4);

        assert_eq!(f0.len(), 8 * 8 * 3);
        assert_eq!(f3.len(), 8 * 8 * 3);
        assert_eq!(f4.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_random_crop_jpeg_fallback() {
        let ep = make_test_episode_jpeg(4);
        let aug = RandomCrop::new(4, 4, vec!["cam_0".to_string()]);
        let out = aug.apply(&ep).unwrap();
        assert_eq!(out.num_steps(), 4);
    }

    #[test]
    fn test_color_jitter_raw() {
        let ep = make_test_episode_raw(4, 8, 8);
        let aug = ColorJitter::new(0.2, 0.2, 0.2, vec!["cam_0".to_string()]);
        let out = aug.apply(&ep).unwrap();
        assert_eq!(out.num_steps(), 4);

        // Pixels should be modified
        let orig = ep.image_raw("cam_0").unwrap();
        let modified = out.image_raw("cam_0").unwrap();
        let o0 = raw_pixels_at(orig, 0);
        let m0 = raw_pixels_at(modified, 0);
        let any_diff = (0..o0.len()).any(|i| o0[i] != m0[i]);
        assert!(any_diff, "ColorJitter should modify pixels");
    }

    /// Episode with one camera whose frames are all identical (pseudo-random
    /// pixel content) and a 4-dim action column with values 1..=4 per row.
    fn make_identical_frames_episode(n: usize, cams: &[(&str, u32, u32)]) -> Episode {
        let mut fields = vec![Field::new("step_index", DataType::Int64, false)];
        let mut cols: Vec<Arc<dyn Array>> = vec![Arc::new(Int64Array::from(
            (0..n as i64).collect::<Vec<_>>(),
        ))];
        for &(cam, width, height) in cams {
            let pixel_count = (width * height * 3) as usize;
            let frame: Vec<u8> = (0..pixel_count)
                .map(|i| ((i * 37 + 11) % 251) as u8)
                .collect();
            let all: Vec<u8> = (0..n).flat_map(|_| frame.clone()).collect();
            let item = Arc::new(Field::new("item", DataType::UInt8, false));
            fields.push(Field::new(
                format!("observation.image.{cam}.rgb"),
                DataType::FixedSizeList(item.clone(), pixel_count as i32),
                false,
            ));
            cols.push(Arc::new(
                FixedSizeListArray::try_new(
                    item,
                    pixel_count as i32,
                    Arc::new(UInt8Array::from(all)),
                    None,
                )
                .unwrap(),
            ));
            fields.push(Field::new(
                format!("observation.image.{cam}.width"),
                DataType::Int64,
                false,
            ));
            cols.push(Arc::new(Int64Array::from(vec![width as i64; n])));
            fields.push(Field::new(
                format!("observation.image.{cam}.height"),
                DataType::Int64,
                false,
            ));
            cols.push(Arc::new(Int64Array::from(vec![height as i64; n])));
        }
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        fields.push(Field::new(
            "action",
            DataType::FixedSizeList(item.clone(), 4),
            false,
        ));
        let actions: Vec<f32> = (0..n).flat_map(|_| [1.0, 2.0, 3.0, 4.0]).collect();
        cols.push(Arc::new(
            FixedSizeListArray::try_new(item, 4, Arc::new(Float32Array::from(actions)), None)
                .unwrap(),
        ));
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap();
        Episode::from_record_batch(batch).unwrap()
    }

    fn action_values(ep: &Episode) -> Vec<f32> {
        ep.actions()
            .unwrap()
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    fn mirror_hwc(pixels: &[u8], w: usize, h: usize) -> Vec<u8> {
        let mut out = vec![0u8; pixels.len()];
        for y in 0..h {
            for x in 0..w {
                let src = (y * w + x) * 3;
                let dst = (y * w + (w - 1 - x)) * 3;
                out[dst..dst + 3].copy_from_slice(&pixels[src..src + 3]);
            }
        }
        out
    }

    #[test]
    fn test_random_crop_same_position_within_window() {
        // All input frames are identical, so equal outputs within a window
        // mean the crop offset was shared.
        let ep = make_identical_frames_episode(8, &[("cam_0", 16, 12)]);
        for _ in 0..5 {
            let out = RandomCrop::new(4, 4, vec!["cam_0".to_string()])
                .with_window_size(4)
                .apply(&ep)
                .unwrap();
            let rgb = out.image_raw("cam_0").unwrap();
            for i in 1..4 {
                assert_eq!(raw_pixels_at(rgb, 0), raw_pixels_at(rgb, i));
                assert_eq!(raw_pixels_at(rgb, 4), raw_pixels_at(rgb, 4 + i));
            }
        }
    }

    #[test]
    fn test_random_crop_too_large_is_error() {
        let ep = make_identical_frames_episode(2, &[("cam_0", 8, 8)]);
        assert!(RandomCrop::new(16, 4, vec!["cam_0".to_string()])
            .apply(&ep)
            .is_err());
    }

    #[test]
    fn test_color_jitter_same_factors_within_window() {
        let ep = make_identical_frames_episode(6, &[("cam_0", 8, 8)]);
        let out = ColorJitter::new(0.3, 0.3, 0.3, vec!["cam_0".to_string()])
            .with_window_size(3)
            .apply(&ep)
            .unwrap();
        let rgb = out.image_raw("cam_0").unwrap();
        assert_eq!(raw_pixels_at(rgb, 0), raw_pixels_at(rgb, 2));
        assert_eq!(raw_pixels_at(rgb, 3), raw_pixels_at(rgb, 5));
    }

    #[test]
    fn test_negative_magnitudes_are_errors() {
        // Regression: negative magnitudes made `random_range(-x..=x)` panic.
        let ep = make_identical_frames_episode(2, &[("cam_0", 8, 8)]);
        let cams = vec!["cam_0".to_string()];
        assert!(ColorJitter::new(-0.1, 0.0, 0.0, cams.clone())
            .apply(&ep)
            .is_err());
        assert!(GaussianNoise::new(-0.1, cams.clone()).apply(&ep).is_err());
        assert!(FusedImageAugmentation::new(cams.clone())
            .with_jitter(0.2, -0.1, 0.0)
            .apply(&ep)
            .is_err());
        // Zero magnitudes are valid no-ops for the noise augmentation.
        let out = GaussianNoise::new(0.0, cams).apply(&ep).unwrap();
        assert_eq!(
            raw_pixels_at(out.image_raw("cam_0").unwrap(), 0),
            raw_pixels_at(ep.image_raw("cam_0").unwrap(), 0)
        );
    }

    #[test]
    fn test_horizontal_flip_mirrors_pixels_and_actions() {
        let ep = make_identical_frames_episode(3, &[("cam_0", 5, 2)]);
        let src = raw_pixels_at(ep.image_raw("cam_0").unwrap(), 0);

        // p = 1: every frame is flipped and dims 0 and 2 are negated.
        let out = HorizontalFlip::new(1.0, vec!["cam_0".to_string()])
            .with_action_mirror_dims(vec![0, 2, 99])
            .apply(&ep)
            .unwrap();
        let rgb = out.image_raw("cam_0").unwrap();
        for i in 0..3 {
            assert_eq!(raw_pixels_at(rgb, i), mirror_hwc(&src, 5, 2));
        }
        let expected: Vec<f32> = (0..3).flat_map(|_| [-1.0, 2.0, -3.0, 4.0]).collect();
        assert_eq!(action_values(&out), expected);

        // p = 0: nothing changes.
        let out = HorizontalFlip::new(0.0, vec!["cam_0".to_string()])
            .with_action_mirror_dims(vec![0, 2])
            .apply(&ep)
            .unwrap();
        assert_eq!(raw_pixels_at(out.image_raw("cam_0").unwrap(), 1), src);
        assert_eq!(action_values(&out), action_values(&ep));
    }

    #[test]
    fn test_fused_identity_and_chw_layout() {
        let ep = make_identical_frames_episode(2, &[("cam_0", 3, 2)]);
        let src = raw_pixels_at(ep.image_raw("cam_0").unwrap(), 0);

        // No transforms enabled: pixels pass through unchanged.
        let out = FusedImageAugmentation::new(vec!["cam_0".to_string()])
            .apply(&ep)
            .unwrap();
        assert_eq!(raw_pixels_at(out.image_raw("cam_0").unwrap(), 1), src);

        // CHW output is the channel-planar transpose of the HWC input.
        let mut aug = FusedImageAugmentation::new(vec!["cam_0".to_string()]);
        aug.chw_output = true;
        let out = aug.apply(&ep).unwrap();
        let chw = raw_pixels_at(out.image_raw("cam_0").unwrap(), 0);
        let (w, h) = (3, 2);
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    assert_eq!(chw[c * h * w + y * w + x], src[(y * w + x) * 3 + c]);
                }
            }
        }
    }

    #[test]
    fn test_fused_window_consistency_and_flip_mirroring() {
        let ep = make_identical_frames_episode(8, &[("top", 12, 10), ("wrist", 12, 10)]);
        let aug = FusedImageAugmentation::new(vec!["top".to_string(), "wrist".to_string()])
            .with_crop(6, 6)
            .with_jitter(0.3, 0.3, 0.3)
            .with_flip(0.5, vec![1])
            .with_window_size(4);
        for _ in 0..5 {
            let out = aug.apply(&ep).unwrap();
            assert_eq!(out.image_dimensions("top").unwrap(), (6, 6));
            for cam in ["top", "wrist"] {
                let rgb = out.image_raw(cam).unwrap();
                for i in 1..4 {
                    assert_eq!(raw_pixels_at(rgb, 0), raw_pixels_at(rgb, i));
                    assert_eq!(raw_pixels_at(rgb, 4), raw_pixels_at(rgb, 4 + i));
                }
            }
            // Both cameras have identical content, so they must receive the
            // same crop, jitter, and flip.
            assert_eq!(
                raw_pixels_at(out.image_raw("top").unwrap(), 0),
                raw_pixels_at(out.image_raw("wrist").unwrap(), 0)
            );
            // Action dim 1 is negated on all frames of a group or on none.
            let acts = action_values(&out);
            for group in [0usize, 4] {
                let first = acts[group * 4 + 1];
                assert!(first == 2.0 || first == -2.0);
                for i in group..group + 4 {
                    assert_eq!(acts[i * 4 + 1], first);
                    assert_eq!(acts[i * 4], 1.0);
                }
            }
        }
    }

    #[test]
    fn test_fused_empty_camera_list_is_error() {
        // Regression: an empty camera list panicked on `camera_names[0]`.
        let ep = make_identical_frames_episode(2, &[("cam_0", 4, 4)]);
        assert!(FusedImageAugmentation::new(Vec::new())
            .with_flip(1.0, vec![0])
            .apply(&ep)
            .is_err());
    }

    #[test]
    fn test_fused_mismatched_camera_sizes_is_error() {
        // Regression: the second camera was indexed with the first camera's
        // dimensions, which panicked (smaller image) or produced garbage
        // (larger image).
        let ep = make_identical_frames_episode(2, &[("big", 16, 16), ("small", 8, 8)]);
        let cams = vec!["big".to_string(), "small".to_string()];
        assert!(FusedImageAugmentation::new(cams.clone())
            .with_crop(8, 8)
            .apply(&ep)
            .is_err());
        let ep = make_identical_frames_episode(2, &[("small", 8, 8), ("big", 16, 16)]);
        assert!(
            FusedImageAugmentation::new(vec!["small".into(), "big".into()])
                .apply(&ep)
                .is_err()
        );
    }

    #[test]
    fn test_gaussian_blur_runs() {
        let ep = make_test_episode_jpeg(4);
        let aug = GaussianBlur::new(1.0, vec!["cam_0".to_string()]);
        let out = aug.apply(&ep).unwrap();
        assert_eq!(out.num_steps(), ep.num_steps());
    }
}
