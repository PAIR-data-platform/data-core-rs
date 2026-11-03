"""Extract MimicGen simulator init states from a raw HDF5 file.

The converted LeRobot dataset stores observations, actions and videos, but not
the flattened MuJoCo state or the per-demo model XML needed to reset the
simulator to a recorded initial state. This helper writes them to a torch file
(states, model_files, env_args, episode_indices) that the MimicGen environment
(lerobot.envs.mimicgen) loads through its init_states_path option.

The rollout scripts call it on data/mimicgen_raw/core/coffee_d0.hdf5 (see
data/README.md for the download) to produce init_states_coffee_d0.pt.

Requires h5py, numpy and torch.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    def demo_index(key: str) -> int:
        match = re.search(r"(\d+)$", key)
        return int(match.group(1)) if match else 0

    return sorted((k for k in data_group.keys() if k.startswith("demo")), key=demo_index)


def decode_env_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        return {}
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MimicGen init_states.pt from raw HDF5.")
    parser.add_argument("--input-hdf5", type=Path, required=True, help="Raw MimicGen HDF5 file.")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt file.")
    parser.add_argument("--episodes", default=None, help="Optional range/list, e.g. 0:100 or 0,3,9.")
    args = parser.parse_args()

    if not args.input_hdf5.exists():
        raise FileNotFoundError(args.input_hdf5)

    episode_filter: set[int] | None = None
    if args.episodes:
        text = args.episodes.strip()
        if ":" in text:
            start_s, end_s = text.split(":", 1)
            episode_filter = set(range(int(start_s), int(end_s)))
        else:
            episode_filter = {int(v) for v in text.split(",") if v.strip()}

    with h5py.File(args.input_hdf5, "r") as f:
        data = f["data"]
        env_args = decode_env_args(data.attrs.get("env_args", "{}"))
        demo_keys = sorted_demo_keys(data)
        if episode_filter is not None:
            demo_keys = [key for idx, key in enumerate(demo_keys) if idx in episode_filter]
        if not demo_keys:
            raise ValueError("No matching demos found.")

        states = []
        model_files = []
        episode_indices = []
        for idx, demo_key in enumerate(sorted_demo_keys(data)):
            if episode_filter is not None and idx not in episode_filter:
                continue
            demo = data[demo_key]
            states.append(np.asarray(demo["states"][0]))
            model_files.append(demo.attrs.get("model_file", ""))
            episode_indices.append(idx)

    payload = {
        "states": torch.tensor(np.stack(states)),
        "model_files": model_files,
        "env_args": env_args,
        "episode_indices": episode_indices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Wrote {len(states)} init states to {args.output}")


if __name__ == "__main__":
    main()
