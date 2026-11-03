//! Parsing of LeRobot's `meta/info.json`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use data_core::DatasetMetadata;

/// Default number of episodes (v2) or files (v3) per chunk directory, used
/// when `info.json` has no `chunks_size`.
pub const DEFAULT_CHUNKS_SIZE: usize = 1000;

/// Parsed representation of LeRobot's `meta/info.json`.
///
/// All fields are optional so that partial or older files still parse.
/// Unknown keys are kept in [`LeRobotInfo::extra`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeRobotInfo {
    /// Format version string, e.g. `"v2.1"` or `"v3.0"`.
    pub codebase_version: Option<String>,
    /// Recording frame rate.
    pub fps: Option<f64>,
    /// Robot type, e.g. `"aloha"`.
    pub robot_type: Option<String>,
    /// Number of episodes in the dataset.
    pub total_episodes: Option<usize>,
    /// Number of frames across all episodes.
    pub total_frames: Option<usize>,
    /// Feature (column) definitions keyed by column name.
    pub features: Option<HashMap<String, FeatureInfo>>,

    /// Path template for Parquet data files, relative to the dataset root.
    ///
    /// v3: `"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"`.
    /// v2: `"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"`.
    pub data_path: Option<String>,

    /// Path template for video files, relative to the dataset root.
    ///
    /// v3: `"videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"`.
    /// v2: `"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"`.
    pub video_path: Option<String>,

    /// All other keys (for example `chunks_size`, `splits`).
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Info about a single feature (column) in the dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureInfo {
    /// Data type, e.g. `"float32"`, `"int64"`, `"video"`, or `"image"`.
    pub dtype: Option<String>,
    /// Per-row shape, e.g. `[14]` or `[480, 640, 3]`.
    pub shape: Option<Vec<usize>>,
    /// Optional per-dimension names (format varies between datasets).
    pub names: Option<serde_json::Value>,

    /// All other keys (for example `info` for video features).
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl LeRobotInfo {
    /// Action dimensionality: the first entry of `features.action.shape`, or 7
    /// if it is missing.
    pub fn action_dim(&self) -> usize {
        self.features
            .as_ref()
            .and_then(|f| f.get("action"))
            .and_then(|a| a.shape.as_ref())
            .and_then(|s| s.first().copied())
            .unwrap_or(7)
    }

    /// State dimensionality: the first entry of the `observation.state` (or
    /// `state`) shape, if present.
    pub fn state_dim(&self) -> Option<usize> {
        self.features
            .as_ref()
            .and_then(|f| f.get("observation.state").or_else(|| f.get("state")))
            .and_then(|a| a.shape.as_ref())
            .and_then(|s| s.first().copied())
    }

    /// Extract camera names from features that match `observation.images.*` or `observation.image.*`.
    pub fn camera_names(&self) -> Vec<String> {
        let Some(features) = &self.features else {
            return vec![];
        };
        let mut cameras = Vec::new();
        for key in features.keys() {
            if let Some(cam) = key.strip_prefix("observation.images.") {
                cameras.push(cam.to_string());
            } else if let Some(cam) = key.strip_prefix("observation.image.") {
                cameras.push(cam.to_string());
            }
        }
        cameras.sort();
        cameras
    }

    /// Returns true for the LeRobot v3 layout (many episodes per Parquet and
    /// video file, located through the episodes metadata).
    ///
    /// The `codebase_version` field decides when present. Without it, the
    /// dataset is treated as v3 if either path template uses the v3
    /// placeholders `{chunk_index}` or `{file_index}`.
    pub fn is_v3(&self) -> bool {
        match self.codebase_version.as_deref() {
            Some(version) => version.starts_with("v3"),
            None => [&self.data_path, &self.video_path]
                .into_iter()
                .flatten()
                .any(|t| t.contains("{chunk_index") || t.contains("{file_index")),
        }
    }

    /// Number of episodes (v2) or files (v3) per chunk directory, from
    /// `chunks_size`, defaulting to [`DEFAULT_CHUNKS_SIZE`].
    pub fn chunks_size(&self) -> usize {
        self.extra
            .get("chunks_size")
            .and_then(|v| v.as_u64())
            .filter(|&n| n > 0)
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_CHUNKS_SIZE)
    }

    /// Returns all features whose `dtype` is `"video"`, keyed by feature name.
    pub fn video_features(&self) -> Vec<(String, &FeatureInfo)> {
        let Some(features) = &self.features else {
            return vec![];
        };
        let mut result: Vec<(String, &FeatureInfo)> = features
            .iter()
            .filter(|(_, fi)| fi.dtype.as_deref() == Some("video"))
            .map(|(k, v)| (k.clone(), v))
            .collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Convert to the core [`DatasetMetadata`] type, using `source` as both
    /// the name and the source. `tasks` is left empty.
    pub fn to_dataset_metadata(&self, source: String) -> DatasetMetadata {
        DatasetMetadata {
            name: source.clone(),
            robot_type: self.robot_type.clone(),
            fps: self.fps,
            num_episodes: self.total_episodes.unwrap_or(0),
            num_steps: self.total_frames.unwrap_or(0),
            action_dim: self.action_dim(),
            state_dim: self.state_dim(),
            camera_names: self.camera_names(),
            tasks: Vec::new(),
            source: Some(source),
            extra: self.extra.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_info_json() {
        let json = r#"{
            "codebase_version": "v2.0",
            "fps": 10,
            "robot_type": "aloha",
            "total_episodes": 50,
            "total_frames": 5000,
            "features": {
                "action": {"dtype": "float32", "shape": [14], "names": null},
                "observation.state": {"dtype": "float32", "shape": [14], "names": null},
                "observation.images.top": {"dtype": "video", "shape": [480, 640, 3], "names": null},
                "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3], "names": null}
            }
        }"#;

        let info: LeRobotInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.fps, Some(10.0));
        assert_eq!(info.robot_type, Some("aloha".to_string()));
        assert_eq!(info.action_dim(), 14);
        assert_eq!(info.state_dim(), Some(14));
        assert_eq!(info.camera_names(), vec!["top", "wrist"]);
        assert!(!info.is_v3());
        assert_eq!(info.video_features().len(), 2);
    }

    #[test]
    fn test_v3_detection() {
        let json = r#"{
            "codebase_version": "v3.0",
            "fps": 10,
            "robot_type": "panda",
            "total_episodes": 379,
            "total_frames": 101469,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                "action": {"dtype": "float32", "shape": [7], "names": null},
                "observation.images.image": {"dtype": "video", "shape": [256, 256, 3], "names": null},
                "observation.images.wrist_image": {"dtype": "video", "shape": [256, 256, 3], "names": null}
            }
        }"#;

        let info: LeRobotInfo = serde_json::from_str(json).unwrap();
        assert!(info.is_v3());
        assert_eq!(
            info.video_path.as_deref(),
            Some("videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        );
        assert_eq!(info.camera_names(), vec!["image", "wrist_image"]);
        let video_feats = info.video_features();
        assert_eq!(video_feats.len(), 2);
        assert!(video_feats
            .iter()
            .any(|(k, _)| k == "observation.images.image"));
    }

    #[test]
    fn test_v2_with_video_path_is_not_v3() {
        // Regression: real LeRobot v2.x datasets have a `video_path` template
        // and were classified as v3 because of it.
        let json = r#"{
            "codebase_version": "v2.1",
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        }"#;
        let info: LeRobotInfo = serde_json::from_str(json).unwrap();
        assert!(!info.is_v3());
        assert_eq!(info.chunks_size(), 1000);

        // Without a version string, the template placeholders decide.
        let json = r#"{"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}"#;
        let info: LeRobotInfo = serde_json::from_str(json).unwrap();
        assert!(info.is_v3());
        assert_eq!(info.chunks_size(), DEFAULT_CHUNKS_SIZE);
    }

    #[test]
    fn test_to_dataset_metadata() {
        let json = r#"{
            "fps": 30,
            "robot_type": "franka",
            "total_episodes": 100,
            "total_frames": 10000,
            "features": {
                "action": {"dtype": "float32", "shape": [7], "names": null}
            }
        }"#;

        let info: LeRobotInfo = serde_json::from_str(json).unwrap();
        let meta = info.to_dataset_metadata("test_source".to_string());
        assert_eq!(meta.fps, Some(30.0));
        assert_eq!(meta.robot_type, Some("franka".to_string()));
        assert_eq!(meta.action_dim, 7);
        assert_eq!(meta.num_episodes, 100);
    }

    #[test]
    fn test_defaults_when_features_missing() {
        let json = r#"{"fps": 10}"#;
        let info: LeRobotInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.action_dim(), 7); // default
        assert_eq!(info.state_dim(), None);
        assert!(info.camera_names().is_empty());
    }
}
