//! Readers and writers for the LeRobot dataset format (v2 and v3).

pub mod downloader;
pub mod lazy_dataset;
pub mod lazy_metadata;
pub mod loader;
pub mod metadata;
pub mod video;
pub mod writer;

pub use lazy_dataset::{Frame, LazyDataset};
pub use lazy_metadata::LazyDatasetMetadata;
pub use loader::LeRobotLoader;
pub use metadata::LeRobotInfo;
