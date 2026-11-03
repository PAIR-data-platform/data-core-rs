"""Throughput benchmark for data_core.LazyDataset.

Measures frames/sec under four access patterns: sequential, random with a warm
episode cache, random with a cold cache, and full-episode iteration. LazyDataset
decodes whole episodes and caches them, so the warm numbers are amortized.
Single process, no DataLoader. Results go to --out as JSON.

Dataset location: --root, else a local copy of --dataset found in
$HF_LEROBOT_HOME/<repo_id>, the Hugging Face hub cache ($HF_HUB_CACHE, else
$HF_HOME/hub, else ~/.cache/huggingface/hub) or $HF_LEROBOT_HOME/hub
(HF_LEROBOT_HOME defaults to $HF_HOME/lerobot). Download the default dataset
with: hf download lerobot/aloha_static_fork_pick_up --repo-type dataset

Requires data_core and numpy.

Run: python scripts/benchmark_lazy_dataset.py [--n-samples 200] [--runs 3]
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np


def find_dataset_root(repo_id: str) -> str:
    """Find a local copy of a Hub dataset in the usual LeRobot / Hugging Face cache locations."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", hf_home / "lerobot")).expanduser()
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub")).expanduser()
    slug = "datasets--" + repo_id.replace("/", "--")

    candidates = [lerobot_home / repo_id]
    for base in (hub_cache / slug / "snapshots", lerobot_home / "hub" / slug / "snapshots"):
        if base.is_dir():
            candidates.extend(sorted(base.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True))
    # Prefer LeRobot v3.0 copies; fall back to v2.x.
    for prefix in ("v3", "v2"):
        for path in candidates:
            info = path / "meta" / "info.json"
            if info.exists() and json.loads(info.read_text()).get("codebase_version", "").startswith(prefix):
                return str(path)
    raise FileNotFoundError(
        f"No local LeRobot v2/v3 copy of {repo_id} found. Download it with "
        f"'hf download {repo_id} --repo-type dataset' or pass --root."
    )


def bench_sequential(ds, n_samples: int) -> float:
    """Sequential access within episodes (amortized decode cost)."""
    ds.clear_cache()
    indices = list(range(min(n_samples, ds.num_frames)))

    # Warmup: first episode
    for i in range(5):
        ds.get_frame(i)

    t0 = time.time()
    for idx in indices:
        ds.get_frame(idx)
    elapsed = time.time() - t0
    return len(indices) / elapsed


def bench_random_warm(ds, n_samples: int) -> float:
    """Random access with warm cache (10 episodes pre-loaded, sample within them)."""
    ds.clear_cache()
    # Only warm 10 episodes to keep timing reasonable
    n_warm = min(10, ds.num_episodes)
    for ep in range(n_warm):
        ds.get_episode(ep)

    # Sample only from warmed episodes
    max_frame = n_warm * (ds.num_frames // ds.num_episodes)
    indices = random.sample(range(max_frame), min(n_samples, max_frame))

    t0 = time.time()
    for idx in indices:
        ds.get_frame(idx)
    elapsed = time.time() - t0
    return len(indices) / elapsed


def bench_random_cold(ds, n_samples: int) -> float:
    """Random access, cold cache (worst case: each frame may trigger episode decode)."""
    ds.clear_cache()
    indices = random.sample(range(ds.num_frames), min(n_samples, ds.num_frames))

    # No warmup: measure cold start
    t0 = time.time()
    for idx in indices[:n_samples]:
        ds.get_frame(idx)
    elapsed = time.time() - t0
    return n_samples / elapsed


def bench_episode_iteration(ds, n_episodes: int) -> float:
    """Iterate full episodes sequentially (realistic epoch scan)."""
    ds.clear_cache()
    n_episodes = min(n_episodes, ds.num_episodes)
    total_frames = 0

    t0 = time.time()
    for ep in range(n_episodes):
        episode = ds.get_episode(ep)
        total_frames += episode.num_steps
    elapsed = time.time() - t0
    return total_frames / elapsed


def run_config(ds, name: str, bench_fn, param: int, runs: int) -> dict:
    rates = []
    for r in range(runs):
        random.seed(42 + r)
        rate = bench_fn(ds, param)
        rates.append(rate)
        print(f"    run {r+1}/{runs}: {rate:.2f} frames/sec")

    mean = float(np.mean(rates))
    std = float(np.std(rates))
    print(f"  => {mean:.2f} ± {std:.2f} frames/sec\n")
    return {"pattern": name, "rates": rates, "mean": mean, "std": std}


def main():
    import data_core

    parser = argparse.ArgumentParser(description="data_core.LazyDataset throughput benchmark")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Frames per run for the sequential and warm random patterns.")
    parser.add_argument("--runs", type=int, default=3, help="Repetitions per access pattern.")
    parser.add_argument("--dataset", type=str, default="lerobot/aloha_static_fork_pick_up",
                        help="Hub repo ID, used to find a local copy and as the label in the output.")
    parser.add_argument("--root", type=str, default=None, help="Local dataset directory (skips the cache lookup).")
    parser.add_argument("--out", type=str, default="results/lazy_dataset_benchmark.json")
    args = parser.parse_args()

    root = args.root or find_dataset_root(args.dataset)
    ds = data_core.LazyDataset.open(root)
    print(f"Dataset: {args.dataset}")
    print(f"  episodes={ds.num_episodes}, frames={ds.num_frames}, fps={ds.fps}")
    print(f"  cameras={ds.camera_names()}\n")

    results = {
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "runs": args.runs,
        "configs": [],
    }

    configs = [
        ("sequential", bench_sequential, args.n_samples),
        ("random_warm", bench_random_warm, args.n_samples),
        ("random_cold_20", bench_random_cold, 20),
        ("episode_iter_10ep", bench_episode_iteration, 10),
    ]

    for name, fn, param in configs:
        print(f"[{name}]")
        r = run_config(ds, name, fn, param, args.runs)
        results["configs"].append(r)

    # Summary
    print("=" * 60)
    print(f"{'Pattern':<22} | {'frames/sec':>12} {'±':>4} {'std':>6}")
    print("-" * 60)
    for c in results["configs"]:
        print(f"{c['pattern']:<22} | {c['mean']:>12.2f}   ± {c['std']:>5.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
