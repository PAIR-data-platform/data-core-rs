#!/usr/bin/env python
"""Convert a raw MimicGen HDF5 file into a LeRobot v3 dataset.

Keeps only the camera images (stored as videos), the robot state and the
actions. Simulator states are not converted; use
scripts/extract_mimicgen_init_states.py for those.

Requires h5py, numpy, tqdm and LeRobot (stock LeRobot is enough).

Example (raw file download: see data/README.md):
    python scripts/convert_mimicgen_hdf5_to_lerobot_v3.py \
        --input-hdf5 data/mimicgen_raw/core/coffee_d0.hdf5 \
        --repo-id local/mimicgen_coffee_d0 \
        --output-dir outputs/mimicgen_coffee_d0_v3 \
        --validate
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

DEFAULT_CAMERA_NAME_MAP = {
    "agentview_image": "image",
    "robot0_eye_in_hand_image": "wrist_image",
}


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    def _demo_idx(key: str) -> int:
        m = re.search(r"(\d+)$", key)
        return int(m.group(1)) if m else 0

    return sorted(data_group.keys(), key=_demo_idx)


def _parse_env_args(file_handle: h5py.File) -> dict[str, Any]:
    raw = file_handle["data"].attrs.get("env_args", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _infer_image_keys(first_demo_obs: h5py.Group) -> list[str]:
    return sorted([k for k in first_demo_obs.keys() if k.endswith("_image")])


def _camera_output_name(src_key: str, keep_source_names: bool) -> str:
    if keep_source_names:
        return src_key
    if src_key in DEFAULT_CAMERA_NAME_MAP:
        return DEFAULT_CAMERA_NAME_MAP[src_key]
    return src_key.replace("_image", "")


def _build_state_array(demo_group: h5py.Group, mode: str) -> np.ndarray:
    if mode == "raw_states":
        return demo_group["states"][:].astype(np.float32)

    obs = demo_group["obs"]
    if mode == "joint_pos_gripper":
        joint_pos = obs["robot0_joint_pos"][:].astype(np.float32)
        gripper = obs["robot0_gripper_qpos"][:, :1].astype(np.float32)
        return np.concatenate([joint_pos, gripper], axis=1)

    # mimicgen_eef mode (default): 3 + 4 + 1 = 8
    eef_pos = obs["robot0_eef_pos"][:].astype(np.float32)
    eef_quat = obs["robot0_eef_quat"][:].astype(np.float32)
    gripper = obs["robot0_gripper_qpos"][:, :1].astype(np.float32)
    return np.concatenate([eef_pos, eef_quat, gripper], axis=1)


def _build_features(
    first_demo: h5py.Group,
    image_keys: list[str],
    keep_source_names: bool,
    state_mode: str,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}

    obs = first_demo["obs"]
    for src_key in image_keys:
        camera_name = _camera_output_name(src_key, keep_source_names)
        video_key = f"observation.images.{camera_name}"
        shape = tuple(obs[src_key].shape[1:])
        features[video_key] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "rgb"],
        }

    state_shape = (_build_state_array(first_demo, state_mode).shape[1],)
    action_shape = (first_demo["actions"].shape[1],)
    features["observation.state"] = {"dtype": "float32", "shape": state_shape}
    features["action"] = {"dtype": "float32", "shape": action_shape}
    return features


def convert(
    input_hdf5: Path,
    repo_id: str,
    output_dir: Path | None,
    fps: int | None,
    robot_type: str,
    limit_episodes: int | None,
    image_keys: list[str] | None,
    keep_source_camera_names: bool,
    state_mode: str,
    dry_run: bool,
) -> None:
    with h5py.File(input_hdf5, "r") as f:
        data_group = f["data"]
        demo_keys = _sorted_demo_keys(data_group)
        if limit_episodes is not None:
            demo_keys = demo_keys[:limit_episodes]
        if not demo_keys:
            raise ValueError("No episodes found under 'data/' in the HDF5 file.")

        first_demo = data_group[demo_keys[0]]
        first_obs = first_demo["obs"]
        selected_image_keys = image_keys if image_keys else _infer_image_keys(first_obs)
        for key in selected_image_keys:
            if key not in first_obs:
                raise KeyError(f"Image key '{key}' not found in first demo obs keys: {list(first_obs.keys())}")

        env_args = _parse_env_args(f)
        inferred_fps = env_args.get("env_kwargs", {}).get("control_freq", 20)
        resolved_fps = fps if fps is not None else int(inferred_fps)

        features = _build_features(
            first_demo=first_demo,
            image_keys=selected_image_keys,
            keep_source_names=keep_source_camera_names,
            state_mode=state_mode,
        )

        logging.info("Input: %s", input_hdf5)
        logging.info("Episodes to convert: %d", len(demo_keys))
        logging.info("Image keys: %s", selected_image_keys)
        logging.info("State mode: %s", state_mode)
        logging.info("FPS: %d", resolved_fps)
        logging.info("Features: %s", list(features.keys()))

        if dry_run:
            return

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output_dir,
            robot_type=robot_type,
            fps=resolved_fps,
            features=features,
        )

        task_name = env_args.get("env_name", "mimicgen_task")

        for demo_key in tqdm(demo_keys, desc="episodes"):
            demo = data_group[demo_key]
            obs = demo["obs"]
            action_array = demo["actions"][:].astype(np.float32)
            state_array = _build_state_array(demo, mode=state_mode)

            image_arrays: dict[str, np.ndarray] = {}
            for src_key in selected_image_keys:
                camera_name = _camera_output_name(src_key, keep_source_camera_names)
                out_key = f"observation.images.{camera_name}"
                image_arrays[out_key] = obs[src_key][:]

            frame_counts = [action_array.shape[0], state_array.shape[0]]
            frame_counts.extend(arr.shape[0] for arr in image_arrays.values())
            n_frames = min(frame_counts)

            for t in range(n_frames):
                frame: dict[str, Any] = {
                    "observation.state": state_array[t],
                    "action": action_array[t],
                    "task": task_name,
                }
                for out_key, arr in image_arrays.items():
                    frame[out_key] = arr[t]
                dataset.add_frame(frame)

            dataset.save_episode()

        dataset.finalize()


def validate_dataset(repo_id: str, root: Path | None) -> None:
    kwargs = {"root": root} if root is not None else {}
    meta = LeRobotDatasetMetadata(repo_id, **kwargs)
    if meta.total_episodes == 0:
        raise ValueError("Validation failed: dataset has 0 episodes.")
    logging.info(
        "Validation OK: episodes=%d frames=%d video_keys=%s",
        meta.total_episodes,
        meta.total_frames,
        list(meta.video_keys),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MimicGen HDF5 to LeRobot v3.")
    parser.add_argument("--input-hdf5", type=Path, required=True, help="Path to MimicGen HDF5 file.")
    parser.add_argument("--repo-id", type=str, required=True, help="Output dataset repo id.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local output root. If omitted, LeRobot default cache/root behavior is used.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Override FPS. Defaults to env control_freq.")
    parser.add_argument("--robot-type", type=str, default="franka", help="Robot type metadata.")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Convert only first N episodes.")
    parser.add_argument(
        "--image-keys",
        type=str,
        default=None,
        help="Comma-separated HDF5 obs image keys (default: infer *_image keys).",
    )
    parser.add_argument(
        "--keep-source-camera-names",
        action="store_true",
        help="Keep camera output names identical to source keys (instead of compatibility remap).",
    )
    parser.add_argument(
        "--state-mode",
        choices=["mimicgen_eef", "joint_pos_gripper", "raw_states"],
        default="mimicgen_eef",
        help=(
            "How to build observation.state: "
            "mimicgen_eef=robot0_eef_pos+robot0_eef_quat+robot0_gripper_qpos[:1] (8d), "
            "joint_pos_gripper=robot0_joint_pos+robot0_gripper_qpos[:1] (8d), "
            "raw_states=demo/states."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect schema/features without writing output.")
    parser.add_argument("--validate", action="store_true", help="Validate produced dataset metadata after conversion.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    image_keys = [k.strip() for k in args.image_keys.split(",") if k.strip()] if args.image_keys else None
    if not args.input_hdf5.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {args.input_hdf5}")

    convert(
        input_hdf5=args.input_hdf5,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        fps=args.fps,
        robot_type=args.robot_type,
        limit_episodes=args.limit_episodes,
        image_keys=image_keys,
        keep_source_camera_names=args.keep_source_camera_names,
        state_mode=args.state_mode,
        dry_run=args.dry_run,
    )

    if args.validate and not args.dry_run:
        validate_dataset(repo_id=args.repo_id, root=args.output_dir)


if __name__ == "__main__":
    main()
