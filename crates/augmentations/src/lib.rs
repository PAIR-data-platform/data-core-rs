//! Augmentation traits and built-in augmentations for the data engine.
//!
//! Augmentations come in two kinds, matching the two categories in the paper:
//!
//! - [`LocalAugmentation`]: a one-to-one transform of a single
//!   [`Episode`] (or of a window of frames cut from one). Examples are image
//!   cropping, color jitter, pixel noise, horizontal flips, and action noise.
//!   Local augmentations can run in parallel across episodes, and
//!   `data_core_io::LazyDataset` applies them on the fly to each sampled window.
//! - [`GlobalAugmentation`]: a many-to-many transform of a whole [`Dataset`]
//!   that may add, remove, or merge episodes.
//!
//! Built-in local augmentations live in [`image`] (pixel transforms, including
//! [`image::FusedImageAugmentation`], which applies crop, jitter, noise, and
//! flip in a single pass) and [`action`] (action and state transforms).
//!
//! Image augmentations operate on raw RGB columns
//! (`observation.image.{cam}.rgb`, `FixedSizeList<UInt8>`, HWC) when present.
//! Some also accept encoded JPEG columns (`observation.image.{cam}`); see each
//! type's documentation.
//!
//! Randomness comes from the thread-local RNG; the built-in augmentations do
//! not take a seed. Several of them accept a `window_size`: when it is greater
//! than zero, consecutive groups of `window_size` frames share the same random
//! parameters, which keeps the augmentation consistent across a temporal
//! window (for example, the frames returned for one `delta_timestamps` sample).
//!
//! These traits are executed in order by `data_core_pipeline::Pipeline`.

#![warn(missing_docs)]

use data_core::{DataCoreError, Dataset, Episode, Result};
use std::sync::Arc;

/// A local augmentation transforms a single episode into a single episode (1:1).
///
/// Examples: image cropping, action noise, color jitter.
/// These are applied independently to each episode in the dataset.
pub trait LocalAugmentation: Send + Sync {
    /// Human-readable name of this augmentation.
    ///
    /// `data_core_io::LazyDataset` uses the names of the built-in image
    /// augmentations to decide when raw pixel columns must be injected, so
    /// custom augmentations should not reuse those names.
    fn name(&self) -> &str;

    /// Apply the augmentation to a single episode, returning a new episode.
    ///
    /// The input episode is not modified; Arrow buffers that are not changed
    /// may be shared with the output.
    fn apply(&self, episode: &Episode) -> Result<Episode>;
}

/// A global augmentation transforms an entire dataset into a new dataset.
///
/// Examples: MimicGen (creates new trajectories by querying across episodes),
/// dataset balancing, cross-episode stitching.
/// These can add, remove, or merge episodes.
pub trait GlobalAugmentation: Send + Sync {
    /// Human-readable name of this augmentation.
    fn name(&self) -> &str;

    /// Apply the augmentation to the entire dataset.
    fn apply(&self, dataset: &Dataset) -> Result<Dataset>;
}

/// Shared, type-erased local augmentation.
pub type BoxedLocalAug = Arc<dyn LocalAugmentation>;

/// Shared, type-erased global augmentation.
pub type BoxedGlobalAug = Arc<dyn GlobalAugmentation>;

/// A step in an augmentation pipeline: either local or global.
#[derive(Clone)]
pub enum AugmentationStep {
    /// Applied to each episode independently.
    Local(BoxedLocalAug),
    /// Applied once to the whole dataset.
    Global(BoxedGlobalAug),
}

impl AugmentationStep {
    /// Name of the wrapped augmentation.
    pub fn name(&self) -> &str {
        match self {
            AugmentationStep::Local(aug) => aug.name(),
            AugmentationStep::Global(aug) => aug.name(),
        }
    }
}

/// Name-keyed registry of local and global augmentations.
///
/// Registering an augmentation under a name that is already present replaces
/// the earlier entry.
#[derive(Default)]
pub struct AugmentationRegistry {
    local: std::collections::HashMap<String, BoxedLocalAug>,
    global: std::collections::HashMap<String, BoxedGlobalAug>,
}

impl AugmentationRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a local augmentation under its [`LocalAugmentation::name`].
    pub fn register_local(&mut self, aug: BoxedLocalAug) {
        self.local.insert(aug.name().to_string(), aug);
    }

    /// Register a global augmentation under its [`GlobalAugmentation::name`].
    pub fn register_global(&mut self, aug: BoxedGlobalAug) {
        self.global.insert(aug.name().to_string(), aug);
    }

    /// Look up a local augmentation by name.
    pub fn get_local(&self, name: &str) -> Option<&BoxedLocalAug> {
        self.local.get(name)
    }

    /// Look up a global augmentation by name.
    pub fn get_global(&self, name: &str) -> Option<&BoxedGlobalAug> {
        self.global.get(name)
    }

    /// Names of all registered local augmentations (unordered).
    pub fn local_names(&self) -> Vec<&str> {
        self.local.keys().map(|s| s.as_str()).collect()
    }

    /// Names of all registered global augmentations (unordered).
    pub fn global_names(&self) -> Vec<&str> {
        self.global.keys().map(|s| s.as_str()).collect()
    }
}

pub mod action;
pub mod image;

/// Return an augmentation error unless `value` is a non-negative number.
///
/// Used for magnitudes that are passed to `random_range(-value..=value)`,
/// which panics on an empty or NaN range.
pub(crate) fn check_magnitude(aug: &str, param: &str, value: f32) -> Result<()> {
    if value >= 0.0 {
        Ok(())
    } else {
        Err(DataCoreError::Augmentation {
            msg: format!("{aug}: '{param}' must be a non-negative number, got {value}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data_core::Episode;

    /// A no-op local augmentation for testing.
    struct IdentityAug;

    impl LocalAugmentation for IdentityAug {
        fn name(&self) -> &str {
            "identity"
        }

        fn apply(&self, episode: &Episode) -> Result<Episode> {
            Ok(episode.clone())
        }
    }

    #[test]
    fn test_registry() {
        let mut registry = AugmentationRegistry::new();
        registry.register_local(Arc::new(IdentityAug));
        assert!(registry.get_local("identity").is_some());
        assert!(registry.get_local("nonexistent").is_none());
        assert_eq!(registry.local_names(), vec!["identity"]);
    }

    #[test]
    fn test_augmentation_step_name() {
        let step = AugmentationStep::Local(Arc::new(IdentityAug));
        assert_eq!(step.name(), "identity");
    }
}
