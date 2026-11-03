#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Port locht131/mimicgen_100_lerobot from LeRobot v2.1 to v3.0.

Reads the v2.1 dataset (per-episode parquet files and MP4 videos, 84x84 agent
and wrist cameras), decodes the videos with PyAV and writes a v3.0 dataset with
LeRobotDataset.create / add_frame / save_episode.

Source location: --local-dir if given (with --download, the source is
downloaded there first); otherwise the Hugging Face cache snapshot of
locht131/mimicgen_100_lerobot at the pinned revision, downloaded on first use
(about 0.6 GB in 3606 files; set HF_TOKEN if anonymous downloads hit the Hub
rate limit). Output: --output-dir, else $HF_LEROBOT_HOME/<repo-id>.

Requires LeRobot, PyAV, pandas, huggingface_hub and tqdm.

Usage:

  # Convert the first 5 episodes and validate the result:
  python examples/port_mimicgen.py \\
      --repo-id local/mimicgen_100_lerobot_v3 \\
      --output-dir outputs/mimicgen_100_lerobot_v3 \\
      --num-episodes 5 \\
      --validate

  # Full conversion and push to the Hub:
  python examples/port_mimicgen.py \\
      --repo-id your_user/mimicgen_100_lerobot_v3 \\
      --push-to-hub
"""

import argparse
import json
import logging
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

MIMICGEN_HUB_ID = "locht131/mimicgen_100_lerobot"
MIMICGEN_REVISION = "61db41c96dd7876601fccbdb3313a30b1127e797"  # pinned source revision
MIMICGEN_FPS = 20
MIMICGEN_ROBOT_TYPE = "franka"
MIMICGEN_CHUNK_SIZE = 1000  # episodes per chunk in the v2.1 source

MIMICGEN_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (84, 84, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (84, 84, 3),
        "names": ["height", "width", "rgb"],
    },
    # Shape is (8,) in the source dataset even though only 7 motor names are listed.
    # The 8th dimension appears to be an additional gripper state channel.
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper2"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
    },
}


def read_video_frames(video_path: Path) -> list[np.ndarray]:
    """Decode all frames from an MP4 file into a list of RGB numpy arrays."""
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def generate_lerobot_frames(local_dir: Path, ep_idx: int, task: str, chunk: int):
    """Yield per-frame dicts for one episode, reading parquet data and decoding videos."""
    parquet_path = local_dir / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
    image_path = local_dir / f"videos/chunk-{chunk:03d}/observation.images.image/episode_{ep_idx:06d}.mp4"
    wrist_path = local_dir / f"videos/chunk-{chunk:03d}/observation.images.wrist_image/episode_{ep_idx:06d}.mp4"

    df = pd.read_parquet(parquet_path)
    image_frames = read_video_frames(image_path)
    wrist_frames = read_video_frames(wrist_path)

    if len(image_frames) != len(df) or len(wrist_frames) != len(df):
        logging.warning(
            f"Episode {ep_idx}: parquet rows={len(df)}, image frames={len(image_frames)}, "
            f"wrist frames={len(wrist_frames)}; lengths differ, using the minimum"
        )

    n_frames = min(len(df), len(image_frames), len(wrist_frames))
    for i in range(n_frames):
        row = df.iloc[i]
        yield {
            "observation.state": np.array(row["observation.state"], dtype=np.float32),
            "action": np.array(row["action"], dtype=np.float32),
            "observation.images.image": image_frames[i],
            "observation.images.wrist_image": wrist_frames[i],
            "task": task,
        }


def port_mimicgen(
    local_dir: Path,
    repo_id: str,
    output_dir: Path | None = None,
    push_to_hub: bool = False,
    num_episodes: int | None = None,
):
    episodes_path = local_dir / "meta/episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(
            f"{episodes_path} not found: {local_dir} is not a complete LeRobot v2.1 copy of {MIMICGEN_HUB_ID}."
        )
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    if num_episodes is not None:
        episodes = episodes[:num_episodes]

    dest = str(output_dir) if output_dir else repo_id
    logging.info(f"Porting {len(episodes)} episodes from {local_dir} → {dest}")

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type=MIMICGEN_ROBOT_TYPE,
        fps=MIMICGEN_FPS,
        features=MIMICGEN_FEATURES,
    )

    start_time = time.time()
    for ep in tqdm(episodes, desc="episodes"):
        ep_idx = ep["episode_index"]
        task = ep["tasks"][0]
        chunk = ep_idx // MIMICGEN_CHUNK_SIZE

        for frame in generate_lerobot_frames(local_dir, ep_idx, task, chunk):
            lerobot_dataset.add_frame(frame)

        lerobot_dataset.save_episode()

    elapsed = time.time() - start_time
    logging.info(f"Porting complete in {elapsed:.1f}s. Finalizing...")

    lerobot_dataset.finalize()

    if push_to_hub:
        lerobot_dataset.push_to_hub(private=False)
        logging.info(f"Pushed to hub: {repo_id}")


def validate_dataset(repo_id: str, root: Path | None = None):
    """Verify that all parquet and video files exist and episode count is non-zero."""
    logging.info(f"Validating {repo_id}...")
    kwargs = {"root": root} if root is not None else {}
    meta = LeRobotDatasetMetadata(repo_id, **kwargs)

    if meta.total_episodes == 0:
        raise ValueError("Validation failed: total_episodes is 0.")

    for ep_idx in tqdm(range(meta.total_episodes), desc="validate"):
        data_path = meta.root / meta.get_data_file_path(ep_idx)
        if not data_path.exists():
            raise ValueError(f"Missing parquet for episode {ep_idx}: {data_path}")

        for vid_key in meta.video_keys:
            vid_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
            if not vid_path.exists():
                raise ValueError(f"Missing video for episode {ep_idx}, key {vid_key}: {vid_path}")

    logging.info(f"Validation passed: {meta.total_episodes} episodes, {meta.total_frames} frames.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Port mimicgen_100_lerobot from v2.1 to v3.0.")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Local directory containing the v2.1 dataset (or where --download puts it). "
        "Default: the Hugging Face cache snapshot of the source, downloaded if missing.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Output HuggingFace repo ID (e.g. your_user/mimicgen_100_lerobot_v3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local directory to write the v3.0 dataset. If omitted, defaults to $HF_LEROBOT_HOME/repo_id.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the converted v3.0 dataset to the HuggingFace Hub.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"Download the v2.1 source dataset from {MIMICGEN_HUB_ID} into --local-dir first.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Limit conversion to the first N episodes (useful for testing).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run a file-existence validation pass after conversion.",
    )

    args = parser.parse_args()

    if args.local_dir is None:
        logging.info(f"Using the Hugging Face cache copy of {MIMICGEN_HUB_ID} (downloading if missing)...")
        local_dir = Path(snapshot_download(MIMICGEN_HUB_ID, repo_type="dataset", revision=MIMICGEN_REVISION))
    else:
        local_dir = args.local_dir
        if args.download:
            logging.info(f"Downloading {MIMICGEN_HUB_ID} to {local_dir}...")
            snapshot_download(
                MIMICGEN_HUB_ID,
                repo_type="dataset",
                revision=MIMICGEN_REVISION,
                local_dir=local_dir,
            )

    port_mimicgen(
        local_dir=local_dir,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        num_episodes=args.num_episodes,
    )

    if args.validate:
        validate_dataset(repo_id=args.repo_id, root=args.output_dir)


if __name__ == "__main__":
    main()
