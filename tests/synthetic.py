"""Synthetic LeRobot v2 datasets for the Python tests.

Nothing here needs network access or a real dataset. Only the ``"jpeg"``
image mode needs Pillow; tests that use it call
``pytest.importorskip("PIL")`` first.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 10
TASK = "pick up the cube"


def frame_pixels(global_index: int, height: int, width: int) -> np.ndarray:
    """Deterministic HWC uint8 image for one frame."""
    rng = np.random.default_rng(global_index)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def encode_jpeg(pixels: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def create_test_dataset(
    path: str,
    num_episodes: int = 3,
    steps_per_episode: int | Sequence[int] = 10,
    action_dim: int = 7,
    state_dim: int | None = None,
    camera: str | None = None,
    image_mode: str = "jpeg",
    image_hw: tuple[int, int] = (24, 32),
) -> None:
    """Write a minimal LeRobot v2 dataset to ``path``.

    Args:
        path: Output directory (created if needed).
        num_episodes: Number of episodes when ``steps_per_episode`` is an int.
        steps_per_episode: Length of every episode, or a list of lengths.
        action_dim: Size of the ``action`` vector. Every action entry equals
            the frame's global index, so tests can tell frames apart.
        state_dim: If set, add ``observation.state`` where every entry
            equals the episode index.
        camera: If set, add one camera with this name.
        image_mode: ``"jpeg"`` stores JPEG bytes in
            ``observation.image.{camera}`` (needs Pillow). ``"rgb"`` stores
            raw HWC pixels in ``observation.image.{camera}.rgb`` plus
            ``.width`` and ``.height`` columns, the layout data-core uses for
            decoded frames.
        image_hw: Image ``(height, width)``.
    """
    if isinstance(steps_per_episode, int):
        episode_lengths = [steps_per_episode] * num_episodes
    else:
        episode_lengths = list(steps_per_episode)
        num_episodes = len(episode_lengths)
    if image_mode not in ("jpeg", "rgb"):
        raise ValueError(f"unknown image_mode {image_mode!r}")

    meta_dir = os.path.join(path, "meta")
    data_dir = os.path.join(path, "data", "chunk-000")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    height, width = image_hw
    features: dict[str, dict] = {
        "action": {"dtype": "float32", "shape": [action_dim], "names": None},
    }
    if state_dim is not None:
        features["observation.state"] = {"dtype": "float32", "shape": [state_dim]}
    if camera is not None:
        features[f"observation.image.{camera}"] = {
            "dtype": "image",
            "shape": [height, width, 3],
        }

    info = {
        "codebase_version": "v2.0",
        "fps": FPS,
        "robot_type": "test_robot",
        "total_episodes": num_episodes,
        "total_frames": sum(episode_lengths),
        "features": features,
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f)

    offset = 0
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep_idx, n in enumerate(episode_lengths):
            ep = {
                "episode_index": ep_idx,
                "task_index": 0,
                "dataset_from_index": offset,
                "dataset_to_index": offset + n,
                "length": n,
            }
            f.write(json.dumps(ep) + "\n")
            offset += n

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")

    offset = 0
    for ep_idx, n in enumerate(episode_lengths):
        global_idx = np.arange(offset, offset + n)
        columns: dict[str, pa.Array] = {
            "frame_index": pa.array(range(n), type=pa.int64()),
            "episode_index": pa.array([ep_idx] * n, type=pa.int64()),
            "index": pa.array(global_idx, type=pa.int64()),
            "timestamp": pa.array([i / FPS for i in range(n)], type=pa.float64()),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(np.repeat(global_idx, action_dim).astype(np.float32)),
                action_dim,
            ),
        }
        if state_dim is not None:
            columns["observation.state"] = pa.FixedSizeListArray.from_arrays(
                pa.array(np.full(n * state_dim, ep_idx, dtype=np.float32)),
                state_dim,
            )
        if camera is not None:
            frames = [frame_pixels(int(g), height, width) for g in global_idx]
            if image_mode == "jpeg":
                columns[f"observation.image.{camera}"] = pa.array(
                    [encode_jpeg(px) for px in frames], type=pa.large_binary()
                )
            else:
                pixel_count = height * width * 3
                columns[f"observation.image.{camera}.rgb"] = pa.FixedSizeListArray.from_arrays(
                    pa.array(np.concatenate([px.ravel() for px in frames]), type=pa.uint8()),
                    pixel_count,
                )
                columns[f"observation.image.{camera}.width"] = pa.array([width] * n, pa.int64())
                columns[f"observation.image.{camera}.height"] = pa.array([height] * n, pa.int64())

        pq.write_table(pa.table(columns), os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet"))
        offset += n
