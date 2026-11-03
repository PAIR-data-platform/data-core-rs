//! Simple terminal benchmark for Libero-10 pipeline execution.
//!
//! Output format is intentionally minimal:
//! - number of episodes
//! - number of frames
//! - total sequential pipeline time
//! - total parallel pipeline time

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use data_core::{Dataset, DatasetMetadata};
use data_core_augmentations::action::ActionNoise;
use data_core_augmentations::image::RandomCrop;
use data_core_io::LazyDataset;
use data_core_pipeline::Pipeline;
use ffmpeg_next as ffmpeg;

fn get_libero10_path() -> Option<PathBuf> {
    // An explicit root wins (LIBERO10_PATH is accepted as an older alias).
    for var in ["LIBERO10_ROOT", "LIBERO10_PATH"] {
        if let Ok(explicit) = std::env::var(var) {
            let path = PathBuf::from(explicit);
            if path.join("meta/info.json").exists() {
                return Some(path);
            }
        }
    }

    // Otherwise the newest snapshot of lerobot/libero_10 in the Hugging Face
    // hub cache: $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub.
    let hub_cache = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache/huggingface/hub")))?;
    let snapshots = hub_cache.join("datasets--lerobot--libero_10/snapshots");
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.join("meta/info.json").exists())
        .map(|path| {
            let modified = path
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            (modified, path)
        })
        .collect();
    candidates.sort();
    candidates.pop().map(|(_, path)| path)
}

fn load_dataset_from_lazy(path: &PathBuf) -> Result<Dataset, String> {
    let lazy = LazyDataset::open(path).map_err(|e| format!("failed to open lazy dataset: {e}"))?;
    let total_episodes = lazy.num_episodes();

    if total_episodes == 0 {
        return Err("dataset has 0 episodes".to_string());
    }

    let mut episodes = Vec::with_capacity(total_episodes);
    for idx in 0..total_episodes {
        let episode = lazy
            .get_episode(idx)
            .map_err(|e| format!("failed to load episode {idx}: {e}"))?;
        episodes.push((*episode).clone());
    }

    let total_frames: usize = episodes.iter().map(|ep| ep.num_steps()).sum();
    let schema = episodes[0].schema();
    let meta = lazy.metadata();
    let dataset_meta = DatasetMetadata {
        name: meta.name.clone(),
        robot_type: meta.robot_type.clone(),
        fps: Some(meta.fps),
        num_episodes: episodes.len(),
        num_steps: total_frames,
        action_dim: meta.action_dim,
        state_dim: meta.state_dim,
        camera_names: meta.camera_names.clone(),
        tasks: meta.tasks.clone(),
        source: Some(path.to_string_lossy().to_string()),
        extra: HashMap::new(),
    };

    Dataset::new(episodes, dataset_meta, schema)
        .map_err(|e| format!("failed to build dataset: {e}"))
}

fn format_duration(d: Duration) -> String {
    format!("{:.3}s", d.as_secs_f64())
}

fn benchmark_camera_names(dataset: &Dataset) -> Result<Vec<String>, String> {
    let schema = dataset.schema();
    let metadata_cameras = &dataset.metadata().camera_names;
    let mut cameras: Vec<String> = metadata_cameras
        .iter()
        .filter(|camera| {
            let col_name = format!("observation.image.{camera}");
            schema.field_with_name(&col_name).is_ok()
        })
        .cloned()
        .collect();

    if cameras.is_empty() {
        cameras = schema
            .fields()
            .iter()
            .filter_map(|field| {
                field
                    .name()
                    .strip_prefix("observation.image.")
                    .map(str::to_string)
            })
            .collect();
    }

    if cameras.is_empty() {
        return Err(format!(
            "no usable image columns found (metadata cameras: {:?})",
            metadata_cameras
        ));
    }

    // Use a single camera to keep benchmark work stable and comparable.
    cameras.truncate(1);
    Ok(cameras)
}

fn silence_decoder_logs() {
    // Keep benchmark terminal output clean and focused on timing numbers.
    unsafe {
        ffmpeg::ffi::av_log_set_level(ffmpeg::ffi::AV_LOG_QUIET);
    }
}

fn main() {
    silence_decoder_logs();

    let Some(path) = get_libero10_path() else {
        eprintln!("lerobot/libero_10 not found. Set LIBERO10_ROOT to a dataset root containing meta/info.json, or run: hf download lerobot/libero_10 --repo-type dataset");
        std::process::exit(1);
    };

    let dataset = match load_dataset_from_lazy(&path) {
        Ok(ds) => ds,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    let episodes = dataset.num_episodes();
    let frames = dataset.num_steps();
    let camera_names = match benchmark_camera_names(&dataset) {
        Ok(cameras) => cameras,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    let mut seq_pipeline = Pipeline::new();
    seq_pipeline.add_local(RandomCrop::new(64, 64, camera_names.clone()));
    seq_pipeline.add_local(ActionNoise::new(0.02));

    let mut par_pipeline = Pipeline::new();
    par_pipeline.add_local(RandomCrop::new(64, 64, camera_names));
    par_pipeline.add_local(ActionNoise::new(0.02));

    let seq_start = Instant::now();
    let _ = seq_pipeline
        .run(dataset.clone())
        .expect("sequential pipeline failed");
    let seq_elapsed = seq_start.elapsed();

    let par_start = Instant::now();
    let _ = par_pipeline
        .run_par(dataset.clone())
        .expect("parallel pipeline failed");
    let par_elapsed = par_start.elapsed();

    println!("episodes: {episodes}");
    println!("frames: {frames}");
    println!("sequential_time: {}", format_duration(seq_elapsed));
    println!("parallel_time: {}", format_duration(par_elapsed));
}
