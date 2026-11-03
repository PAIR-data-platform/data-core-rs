//! I/O layer for the `data-core` data engine: reading and writing LeRobot
//! datasets.
//!
//! - [`LazyDataset`]: opens a local LeRobot v2 or v3 dataset by reading only
//!   its metadata, then decodes episodes on demand (Parquet columns plus
//!   video frames) into Arrow `RecordBatch`es held in an episode cache.
//!   [`LazyDataset::preload_episodes`] freezes that cache so forked
//!   DataLoader workers share it through copy-on-write. Windows of frames
//!   can be augmented on the fly with `data_core_augmentations`.
//! - [`LazyDatasetMetadata`]: the per-episode index behind `LazyDataset`.
//! - [`LeRobotLoader`]: eagerly reads a whole dataset into a
//!   [`data_core::Dataset`] for offline pipelines.
//! - [`LeRobotWriter`]: writes a `Dataset` back to the v2 layout.
//! - [`LeRobotDownloader`]: fetches a dataset repository from the
//!   HuggingFace Hub.
//! - [`lerobot::video::VideoDecoder`]: FFmpeg-based range decoding used by
//!   the readers.
//!
//! The types themselves come from `data-core`; the Python bindings in
//! `data-core-python` wrap `LazyDataset` for PyTorch training.
//!
//! # Example
//!
//! ```no_run
//! use data_core_io::LazyDataset;
//!
//! let mut dataset = LazyDataset::open("/path/to/lerobot/dataset")?.with_raw_pixels();
//! let episodes: Vec<usize> = dataset.episode_indices().collect();
//! dataset.preload_episodes(&episodes)?;
//! // Four-frame window ending at frame 100, clamped to its episode.
//! let window = dataset.get_frame_windowed(100, &[-3, -2, -1, 0])?;
//! assert_eq!(window.num_steps(), 4);
//! # Ok::<(), data_core::DataCoreError>(())
//! ```

#![warn(missing_docs)]

pub mod lerobot;

pub use lerobot::downloader::LeRobotDownloader;
pub use lerobot::writer::LeRobotWriter;
pub use lerobot::{Frame, LazyDataset, LazyDatasetMetadata};
pub use lerobot::{LeRobotInfo, LeRobotLoader};
