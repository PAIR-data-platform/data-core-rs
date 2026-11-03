#!/usr/bin/env python3
"""Dataset loading benchmark: data_core.LazyDataset vs LeRobotDataset.

Times opening the dataset, sequential frame access, random frame access and
(data_core only) cold episode loading, single process, on lerobot/libero_10 by
default, and prints a comparison table. Each timed pass follows a warm-up
pass, so the data_core access numbers come from LazyDataset's episode cache,
while LeRobotDataset decodes video frames on every access; the two columns
are therefore not a like-for-like decode comparison.

Dataset location, in order of precedence: --dataset PATH, the LIBERO10_ROOT
environment variable, then a snapshot of lerobot/libero_10 in the Hugging Face
hub cache ($HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub).
Download it with: hf download lerobot/libero_10 --repo-type dataset

Usage:
    maturin develop --release          # build the extension first
    python benches/loading_benchmark.py
    python benches/loading_benchmark.py --dataset /path/to/libero_10 --rust-only

Requirements: data_core and/or lerobot, numpy; tabulate (optional) for the table.
"""

import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# IMPORTANT: Import LeRobot BEFORE data_core to avoid library conflicts
# Try to import LeRobot
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    print("Warning: lerobot not found. Install with 'pip install lerobot'.")

# Try to import our Rust bindings
try:
    import data_core
    HAS_DATA_CORE = True
except ImportError:
    HAS_DATA_CORE = False
    print("Warning: data_core not found. Run 'maturin develop --release' first.")


def find_libero10_dataset() -> Optional[Path]:
    """Find lerobot/libero_10 via LIBERO10_ROOT or the Hugging Face hub cache."""
    if os.environ.get("LIBERO10_ROOT"):
        return Path(os.environ["LIBERO10_ROOT"]).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub")).expanduser()
    snapshots = hub_cache / "datasets--lerobot--libero_10" / "snapshots"
    if snapshots.is_dir():
        for path in sorted(snapshots.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
            if (path / "meta" / "info.json").exists():
                return path
    return None


def benchmark(fn: Callable[[], Any], warmup: int = 1, iterations: int = 5) -> dict:
    """Run a benchmark function and return timing statistics."""
    # Warmup
    for _ in range(warmup):
        fn()
        gc.collect()
    
    # Timed runs
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "iterations": iterations,
    }


class RustBenchmarks:
    """Benchmarks for our Rust data_core implementation."""
    
    def __init__(self, path: Path):
        self.path = path
        self.dataset = None
    
    def open_dataset(self):
        """Open dataset (metadata only)."""
        self.dataset = data_core.LazyDataset.open(str(self.path))
        return self.dataset
    
    def bench_open(self) -> dict:
        """Benchmark: Open dataset (metadata only)."""
        return benchmark(self.open_dataset, warmup=1, iterations=10)
    
    def bench_sequential_access(self, n_frames: int = 100) -> dict:
        """Benchmark: Sequential frame access."""
        if self.dataset is None:
            self.open_dataset()
        
        def access_frames():
            for i in range(n_frames):
                frame = self.dataset[i]
                _ = frame.action
            return n_frames
        
        result = benchmark(access_frames, warmup=1, iterations=3)
        result["frames"] = n_frames
        result["frames_per_sec"] = n_frames / (result["mean_ms"] / 1000)
        return result
    
    def bench_random_access(self, n_samples: int = 100) -> dict:
        """Benchmark: Random frame access."""
        if self.dataset is None:
            self.open_dataset()
        
        num_frames = len(self.dataset)
        random.seed(42)
        indices = [random.randint(0, num_frames - 1) for _ in range(n_samples)]
        
        def access_frames():
            for idx in indices:
                frame = self.dataset[idx]
                _ = frame.action
            return n_samples
        
        result = benchmark(access_frames, warmup=1, iterations=3)
        result["samples"] = n_samples
        result["samples_per_sec"] = n_samples / (result["mean_ms"] / 1000)
        return result
    
    def bench_episode_load(self, n_episodes: int = 5) -> dict:
        """Benchmark: Episode loading (cold cache)."""
        if self.dataset is None:
            self.open_dataset()
        
        n_episodes = min(n_episodes, self.dataset.num_episodes)
        
        def load_episodes():
            self.dataset.clear_cache()
            for i in range(n_episodes):
                self.dataset.get_episode(i)
            return n_episodes
        
        result = benchmark(load_episodes, warmup=0, iterations=3)
        result["episodes"] = n_episodes
        return result


class LeRobotBenchmarks:
    """Benchmarks for LeRobot's Python implementation."""
    
    def __init__(self, repo_id: str = "lerobot/libero_10", root: Optional[Path] = None):
        self.repo_id = repo_id
        self.root = root
        self.dataset = None
    
    def open_dataset(self):
        """Open dataset."""
        if self.root:
            self.dataset = LeRobotDataset(self.repo_id, root=self.root)
        else:
            self.dataset = LeRobotDataset(self.repo_id)
        return self.dataset
    
    def bench_open(self) -> dict:
        """Benchmark: Open dataset."""
        return benchmark(self.open_dataset, warmup=1, iterations=5)
    
    def bench_sequential_access(self, n_frames: int = 100) -> dict:
        """Benchmark: Sequential frame access."""
        if self.dataset is None:
            self.open_dataset()
        
        def access_frames():
            for i in range(n_frames):
                frame = self.dataset[i]
                _ = frame["action"]
            return n_frames
        
        result = benchmark(access_frames, warmup=1, iterations=3)
        result["frames"] = n_frames
        result["frames_per_sec"] = n_frames / (result["mean_ms"] / 1000)
        return result
    
    def bench_random_access(self, n_samples: int = 100) -> dict:
        """Benchmark: Random frame access."""
        if self.dataset is None:
            self.open_dataset()
        
        num_frames = len(self.dataset)
        random.seed(42)  # Same seed as Rust benchmarks
        indices = [random.randint(0, num_frames - 1) for _ in range(n_samples)]
        
        def access_frames():
            for idx in indices:
                frame = self.dataset[idx]
                _ = frame["action"]
            return n_samples
        
        result = benchmark(access_frames, warmup=1, iterations=3)
        result["samples"] = n_samples
        result["samples_per_sec"] = n_samples / (result["mean_ms"] / 1000)
        return result


def print_results_table(rust_results: dict, lerobot_results: dict):
    """Print a comparison table of benchmark results."""
    try:
        from tabulate import tabulate
    except ImportError:
        print("\nInstall tabulate for nicer output: pip install tabulate")
        print("\n--- Results ---")
        print(f"Rust: {rust_results}")
        print(f"LeRobot: {lerobot_results}")
        return
    
    headers = ["Benchmark", "Rust (ms)", "LeRobot (ms)", "Speedup"]
    rows = []
    
    for key in rust_results:
        rust_val = rust_results[key].get("mean_ms", 0)
        lerobot_val = lerobot_results.get(key, {}).get("mean_ms", 0)
        
        if rust_val > 0 and lerobot_val > 0:
            speedup = f"{lerobot_val / rust_val:.2f}x"
        else:
            speedup = "N/A"
        
        rows.append([
            key,
            f"{rust_val:.2f} ± {rust_results[key].get('std_ms', 0):.2f}",
            f"{lerobot_val:.2f} ± {lerobot_results.get(key, {}).get('std_ms', 0):.2f}",
            speedup,
        ])
    
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS: Rust data_core vs LeRobot Python")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()
    
    # Print throughput comparison if available
    if "sequential" in rust_results and "frames_per_sec" in rust_results["sequential"]:
        print("Sequential Access Throughput:")
        print(f"  Rust:    {rust_results['sequential']['frames_per_sec']:.1f} frames/sec")
        if "sequential" in lerobot_results:
            print(f"  LeRobot: {lerobot_results['sequential']['frames_per_sec']:.1f} frames/sec")
    
    if "random" in rust_results and "samples_per_sec" in rust_results["random"]:
        print("Random Access Throughput:")
        print(f"  Rust:    {rust_results['random']['samples_per_sec']:.1f} samples/sec")
        if "random" in lerobot_results:
            print(f"  LeRobot: {lerobot_results['random']['samples_per_sec']:.1f} samples/sec")


def main():
    parser = argparse.ArgumentParser(description="Dataset Loading Benchmark")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Local dataset directory (default: $LIBERO10_ROOT, else the hub cache snapshot of lerobot/libero_10)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/libero_10",
        help="HuggingFace repo ID for LeRobot (default: lerobot/libero_10)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames for sequential access benchmark",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples for random access benchmark",
    )
    parser.add_argument(
        "--rust-only",
        action="store_true",
        help="Only run Rust benchmarks",
    )
    parser.add_argument(
        "--lerobot-only",
        action="store_true",
        help="Only run LeRobot benchmarks",
    )
    args = parser.parse_args()
    
    # Find dataset
    dataset_path = args.dataset or find_libero10_dataset()
    
    if dataset_path is None:
        print("Error: could not find lerobot/libero_10.")
        print("Pass --dataset PATH, set LIBERO10_ROOT, or download it with:")
        print("  hf download lerobot/libero_10 --repo-type dataset")
        sys.exit(1)
    
    print(f"Using dataset at: {dataset_path}")
    
    rust_results = {}
    lerobot_results = {}
    
    # Run Rust benchmarks
    if HAS_DATA_CORE and not args.lerobot_only:
        print("\n--- Running Rust data_core benchmarks ---")
        rust = RustBenchmarks(dataset_path)
        
        print("  Benchmarking: open (metadata only)...")
        rust_results["open"] = rust.bench_open()
        
        print(f"  Benchmarking: sequential access ({args.frames} frames)...")
        rust_results["sequential"] = rust.bench_sequential_access(args.frames)
        
        print(f"  Benchmarking: random access ({args.samples} samples)...")
        rust_results["random"] = rust.bench_random_access(args.samples)
        
        print("  Benchmarking: episode loading (cold)...")
        rust_results["episode_cold"] = rust.bench_episode_load(5)
    
    # Run LeRobot benchmarks
    if HAS_LEROBOT and not args.rust_only:
        print("\n--- Running LeRobot Python benchmarks ---")
        lerobot = LeRobotBenchmarks(args.repo_id, root=dataset_path)
        
        print("  Benchmarking: open...")
        lerobot_results["open"] = lerobot.bench_open()
        
        print(f"  Benchmarking: sequential access ({args.frames} frames)...")
        lerobot_results["sequential"] = lerobot.bench_sequential_access(args.frames)
        
        print(f"  Benchmarking: random access ({args.samples} samples)...")
        lerobot_results["random"] = lerobot.bench_random_access(args.samples)
    
    # Print results
    if rust_results or lerobot_results:
        print_results_table(rust_results, lerobot_results)
    else:
        print("\nNo benchmarks were run. Install data_core and/or lerobot.")


if __name__ == "__main__":
    main()
