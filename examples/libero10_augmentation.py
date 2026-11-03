"""Image and state augmentation pipeline on LIBERO-10.

Loads the LeRobot v3 dataset `lerobot/libero_10` with `data_core.Dataset`,
runs a pipeline of built-in image and action/state augmentations plus one
custom Python augmentation, prints a before/after spot check, and saves the
augmented dataset. `examples/visualize_augmented.py` plots the result.

`Dataset.from_local` decodes every video frame into memory (as JPEG bytes), so
the full dataset (379 episodes, about 101k frames per camera) takes a few
minutes and about 5 GB of RAM (measured: 4.3 minutes, 5.2 GB peak RSS on a
32-core machine). The augmented copy written to --output-dir is about 1.4 GB.

Dataset location, in order of precedence:
  1. --dataset-root PATH
  2. the LIBERO10_ROOT environment variable
  3. the Hugging Face cache snapshot of lerobot/libero_10 at the pinned revision
     ($HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub),
     downloaded there on first use (about 0.65 GB)

Usage:
    python examples/libero10_augmentation.py
    python examples/libero10_augmentation.py --dataset-root /path/to/libero_10 \
        --output-dir outputs/augmented_libero10

Requirements: data_core, pyarrow, numpy, huggingface_hub (for the download),
Pillow (optional, for the pixel-level spot check).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

import data_core

REPO_ID = "lerobot/libero_10"
# Pinned dataset revision (Hugging Face commit hash).
REVISION = "551d7d86f25edd0ffeda8b60053c15438cfd1d6a"


# ---------------------------------------------------------------------------
# 1. Locate or download the dataset
# ---------------------------------------------------------------------------

def hf_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home).expanduser() / "hub"


def resolve_dataset_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("LIBERO10_ROOT")
    if root:
        path = Path(root).expanduser()
        if not (path / "meta" / "info.json").exists():
            raise FileNotFoundError(f"{path} is not a LeRobot dataset (no meta/info.json)")
        return path

    snapshot = hf_hub_cache() / f"datasets--{REPO_ID.replace('/', '--')}" / "snapshots" / REVISION
    if (snapshot / "meta" / "info.json").exists() and any(snapshot.glob("videos/*/*/*.mp4")):
        print(f"Using cached dataset at {snapshot}")
        return snapshot

    from huggingface_hub import snapshot_download

    print(f"Downloading {REPO_ID}@{REVISION[:8]} into the Hugging Face cache ...")
    return Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION))


# ---------------------------------------------------------------------------
# 2. Inspect the dataset
# ---------------------------------------------------------------------------

def inspect_dataset(ds: data_core.Dataset) -> None:
    print("\nDataset info")
    print(f"  Name       : {ds.name}")
    print(f"  Robot type : {ds.robot_type}")
    print(f"  FPS        : {ds.fps}")
    print(f"  Episodes   : {ds.num_episodes}")
    print(f"  Total steps: {ds.num_steps}")

    ep0 = ds.episode(0)
    batch = ep0.to_pyarrow()
    print(f"\n  Episode 0 schema ({ep0.num_steps} steps):")
    for field in batch.schema:
        print(f"    {field.name}: {field.type}")


# ---------------------------------------------------------------------------
# 3. Custom Python augmentation: decaying exploration noise on actions
# ---------------------------------------------------------------------------

class ExplorationNoise:
    """Add zero-mean Gaussian noise to `action` that decays over the episode.

    The standard deviation falls linearly from `sigma_start` at the first step
    to `0.1 * sigma_start` at the last step, so the trajectory stays close to
    the demonstration near task completion.
    """

    def __init__(self, sigma_start: float = 0.05, seed: int = 0):
        self.sigma_start = sigma_start
        self._rng = np.random.default_rng(seed)

    def run(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        if "action" not in batch.schema.names:
            return batch

        action_col = batch.column("action")
        if pa.types.is_fixed_size_list(action_col.type):
            action_dim = action_col.type.list_size
        elif pa.types.is_list(action_col.type):
            action_dim = len(action_col[0].as_py())
        else:
            return batch

        n_steps = batch.num_rows
        arr = np.asarray(action_col.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
        arr = arr.reshape(n_steps, action_dim).copy()

        t = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
        sigma = self.sigma_start * (1.0 - 0.9 * t)
        arr += self._rng.standard_normal((n_steps, action_dim)).astype(np.float32) * sigma[:, None]

        new_action = pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1), type=pa.float32()), action_dim)
        idx = batch.schema.get_field_index("action")
        return batch.set_column(idx, pa.field("action", new_action.type), new_action)


# ---------------------------------------------------------------------------
# 4. Build the pipeline
# ---------------------------------------------------------------------------

def build_pipeline(camera_names: list[str], seed: int) -> data_core.Pipeline:
    """Pipeline stages, in order:

    1. color_jitter       random brightness / contrast / saturation per frame
    2. gaussian_blur      mild spatial smoothing of camera images
    3. action_noise       small uniform jitter on actions (built-in)
    4. state_scaling      scale observation.state by a constant (identity here)
    5. ExplorationNoise   custom Python augmentation defined above

    Image stages are skipped when the dataset has no cameras.
    """
    pipeline = data_core.Pipeline()

    if camera_names:
        pipeline.add_builtin("color_jitter", json.dumps({
            "brightness": 0.15,
            "contrast": 0.10,
            "saturation": 0.10,
            "camera_names": camera_names,
        }))
        pipeline.add_builtin("gaussian_blur", json.dumps({
            "sigma": 0.8,
            "camera_names": camera_names,
        }))

    pipeline.add_builtin("action_noise", json.dumps({"std": 0.02}))
    pipeline.add_builtin("state_scaling", json.dumps({
        "scale": 1.0,  # identity; replace with your normalization factor
        "column": "observation.state",
    }))
    pipeline.add_local(ExplorationNoise(sigma_start=0.05, seed=seed))
    return pipeline


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def image_delta(orig_bytes: bytes, aug_bytes: bytes) -> str:
    try:
        from PIL import Image
    except ImportError:
        return f"orig={len(orig_bytes)} bytes, aug={len(aug_bytes)} bytes (install Pillow for a pixel diff)"
    orig = np.asarray(Image.open(io.BytesIO(orig_bytes)), dtype=np.float32)
    aug = np.asarray(Image.open(io.BytesIO(aug_bytes)), dtype=np.float32)
    delta = np.abs(orig - aug)
    return f"mean={delta.mean():.2f} max={delta.max():.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an image and state augmentation pipeline on lerobot/libero_10.",
    )
    parser.add_argument("--dataset-root", default=None,
                        help="Local LeRobot v3 dataset directory. Default: $LIBERO10_ROOT, "
                             "else the Hugging Face cache snapshot (downloaded if missing).")
    parser.add_argument("--output-dir", default="outputs/augmented_libero10",
                        help="Where to save the augmented dataset (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the custom Python augmentation.")
    args = parser.parse_args()

    local_path = resolve_dataset_root(args.dataset_root)

    print(f"\nLoading dataset from {local_path} ...")
    t0 = time.perf_counter()
    ds = data_core.Dataset.from_local(str(local_path))
    print(f"Loaded in {time.perf_counter() - t0:.2f}s")

    inspect_dataset(ds)

    # Camera names come from meta/info.json. For LeRobot v3 the loader decodes
    # the videos and adds one `observation.image.<camera>` column per camera.
    camera_names = ds.camera_names()
    print(f"\n  Cameras: {camera_names or '(none)'}")

    pipeline = build_pipeline(camera_names, args.seed)
    print(f"  Pipeline: {pipeline}")

    print("\nRunning augmentation pipeline ...")
    t0 = time.perf_counter()
    augmented = pipeline.run(ds)
    print(f"Augmented {augmented.num_episodes} episodes ({augmented.num_steps} steps) "
          f"in {time.perf_counter() - t0:.2f}s")

    orig_batch = ds.episode(0).to_pyarrow()
    aug_batch = augmented.episode(0).to_pyarrow()

    if "action" in orig_batch.schema.names:
        orig_actions = np.array(orig_batch.column("action").to_pylist())
        aug_actions = np.array(aug_batch.column("action").to_pylist())
        delta = np.abs(aug_actions - orig_actions)
        print(f"\n  Action delta (episode 0): mean={delta.mean():.5f} max={delta.max():.5f}")

    if camera_names:
        col = f"observation.image.{camera_names[0]}"
        if col in orig_batch.schema.names and col in aug_batch.schema.names:
            diff = image_delta(orig_batch.column(col)[0].as_py(), aug_batch.column(col)[0].as_py())
            print(f"  Image delta  (episode 0, camera '{camera_names[0]}', frame 0): {diff}")
        else:
            print(f"  Image column '{col}' not found; are the video files present?")

    output_dir = Path(args.output_dir)
    print(f"\nSaving augmented dataset to {output_dir}/ ...")
    augmented.save_lerobot(str(output_dir))
    print(f"Saved {augmented.num_episodes} episodes")


if __name__ == "__main__":
    main()
