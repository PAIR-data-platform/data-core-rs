//! `data-engine`: command-line tool for inspecting LeRobot datasets.
//!
//! Usage: `data-engine info <path>` prints a summary of a local LeRobot v2 or
//! v3 dataset. Only metadata is read (for v3 also the `episode_index` column
//! of the data files); no video is decoded.

use std::fmt::Write as _;

use anyhow::Result;
use clap::{Parser, Subcommand};
use data_core_io::{LazyDataset, LazyDatasetMetadata};

#[derive(Parser)]
#[command(name = "data-engine")]
#[command(about = "Robotics data engine: inspect LeRobot datasets")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Display information about a local LeRobot dataset
    Info {
        /// Path to the dataset root (the directory containing meta/)
        path: String,
    },
}

/// Render the `info` summary for a dataset.
fn format_info(meta: &LazyDatasetMetadata) -> String {
    let state_dim = meta
        .state_dim
        .map_or_else(|| "none".to_string(), |dim| dim.to_string());
    let mut out = String::new();
    // Writing to a String cannot fail, so the fmt::Result is ignored.
    let _ = write!(
        out,
        "Dataset:    {}\n\
         Root:       {}\n\
         Format:     LeRobot {}\n\
         Robot type: {}\n\
         FPS:        {}\n\
         Episodes:   {}\n\
         Frames:     {}\n\
         Action dim: {}\n\
         State dim:  {}\n\
         Cameras:    {}\n",
        meta.name,
        meta.root.display(),
        if meta.is_v3 { "v3" } else { "v2" },
        meta.robot_type.as_deref().unwrap_or("unknown"),
        meta.fps,
        meta.total_episodes,
        meta.total_frames,
        meta.action_dim,
        state_dim,
        meta.camera_names.join(", "),
    );
    out
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { path } => {
            let dataset = LazyDataset::open(&path)?;
            print!("{}", format_info(dataset.metadata()));
        }
    }

    Ok(())
}
