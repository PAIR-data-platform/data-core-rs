//! Integration tests for the `data-engine` binary.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Create a fresh, empty directory under Cargo's per-target temp dir.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write the metadata of a tiny LeRobot v2.1 dataset (three episodes of 4, 5,
/// and 6 frames, two cameras). `data-engine info` reads only `meta/`.
fn write_v2_dataset(root: &Path) {
    let meta = root.join("meta");
    fs::create_dir_all(&meta).unwrap();
    fs::write(
        meta.join("info.json"),
        r#"{
            "codebase_version": "v2.1",
            "robot_type": "koch",
            "fps": 30,
            "total_episodes": 3,
            "total_frames": 15,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "action": {"dtype": "float32", "shape": [6], "names": null},
                "observation.state": {"dtype": "float32", "shape": [6], "names": null},
                "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
                "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]}
            }
        }"#,
    )
    .unwrap();
    fs::write(
        meta.join("episodes.jsonl"),
        "{\"episode_index\": 0, \"tasks\": [\"pick\"], \"length\": 4}\n\
         {\"episode_index\": 1, \"tasks\": [\"pick\"], \"length\": 5}\n\
         {\"episode_index\": 2, \"tasks\": [\"pick\"], \"length\": 6}\n",
    )
    .unwrap();
    fs::write(
        meta.join("tasks.jsonl"),
        "{\"task_index\": 0, \"task\": \"pick\"}\n",
    )
    .unwrap();
}

#[test]
fn info_prints_v2_summary() {
    let root = fresh_dir("cli_info_v2").join("tiny_koch");
    write_v2_dataset(&root);

    let output = Command::new(env!("CARGO_BIN_EXE_data-engine"))
        .arg("info")
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let expected = format!(
        "Dataset:    tiny_koch\n\
         Root:       {}\n\
         Format:     LeRobot v2\n\
         Robot type: koch\n\
         FPS:        30\n\
         Episodes:   3\n\
         Frames:     15\n\
         Action dim: 6\n\
         State dim:  6\n\
         Cameras:    front, wrist\n",
        root.display()
    );
    assert_eq!(stdout, expected);
}

#[test]
fn info_fails_without_metadata() {
    let root = fresh_dir("cli_info_missing");

    let output = Command::new(env!("CARGO_BIN_EXE_data-engine"))
        .arg("info")
        .arg(&root)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("info.json"), "stderr: {stderr}");
}
