"""Plot frames, actions and states from a dataset saved by libero10_augmentation.py.

Reads the per-episode parquet files that `Dataset.save_lerobot` writes
(`<input>/data/chunk-*/episode_NNNNNN.parquet`, with JPEG frames in
`observation.image.<camera>` columns) and produces:
  1. a grid of evenly spaced camera frames from one episode
  2. the action trajectories of that episode, one panel per dimension
  3. the observation.state trajectories of that episode
  4. optionally (--heatmap), a heatmap of the actions of up to 50 episodes

Usage:
    python examples/libero10_augmentation.py          # writes outputs/augmented_libero10
    python examples/visualize_augmented.py [--episode N] [--heatmap]

Requirements: matplotlib, numpy, pyarrow, Pillow (data_core is not needed).
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

IMAGE_PREFIX = "observation.image."


def find_episode_file(data_dir: Path, episode_idx: int) -> Path:
    matches = sorted(data_dir.rglob(f"episode_{episode_idx:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No parquet file for episode {episode_idx} under {data_dir}. "
            "Run examples/libero10_augmentation.py first or pass --input."
        )
    return matches[0]


def decode_jpeg(raw: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(raw)))


def pick_frame_indices(n_frames_total: int, n_show: int) -> list[int]:
    """Return n_show evenly spaced frame indices across the episode."""
    if n_show >= n_frames_total:
        return list(range(n_frames_total))
    if n_show <= 1:
        return [0]
    return [int(round(i * (n_frames_total - 1) / (n_show - 1))) for i in range(n_show)]


def finish(fig, plt, out: Path | None) -> None:
    fig.tight_layout()
    if out is None:
        plt.show()
    else:
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


def plot_image_grid(plt, table, episode_idx: int, n_frames: int, out: Path | None) -> None:
    cams = sorted(c for c in table.schema.names if c.startswith(IMAGE_PREFIX))
    if not cams:
        print("  No observation.image.* columns; skipping the frame grid.")
        return

    frame_indices = pick_frame_indices(table.num_rows, n_frames)
    fig, axes = plt.subplots(len(cams), len(frame_indices),
                             figsize=(2.5 * len(frame_indices), 2.8 * len(cams)), squeeze=False)
    fig.suptitle(f"Episode {episode_idx}: camera frames", fontsize=14)
    timestamps = table.column("timestamp").to_pylist() if "timestamp" in table.schema.names else None

    for row, cam_col in enumerate(cams):
        col_data = table.column(cam_col)
        for col, fi in enumerate(frame_indices):
            ax = axes[row, col]
            ax.imshow(decode_jpeg(col_data[fi].as_py()))
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(cam_col[len(IMAGE_PREFIX):], fontsize=9)
            ax.set_title(f"t={timestamps[fi]:.2f}s" if timestamps else f"frame {fi}", fontsize=7)
    finish(fig, plt, out)


def plot_series(plt, table, column: str, label: str, episode_idx: int, out: Path | None) -> None:
    if column not in table.schema.names:
        print(f"  No '{column}' column; skipping the {label} plot.")
        return
    values = np.array(table.column(column).to_pylist(), dtype=np.float32)
    if "timestamp" in table.schema.names:
        x = np.array(table.column("timestamp").to_pylist(), dtype=np.float32)
        xlabel = "time (s)"
    else:
        x = np.arange(len(values))
        xlabel = "step"
    n_dim = values.shape[1]

    fig, axes = plt.subplots(n_dim, 1, figsize=(12, 1.8 * n_dim), sharex=True, squeeze=False)
    fig.suptitle(f"Episode {episode_idx}: {column} ({n_dim} dims)", fontsize=13)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_dim))
    for i in range(n_dim):
        ax = axes[i, 0]
        ax.plot(x, values[:, i], color=colors[i], linewidth=1.2)
        ax.set_ylabel(f"{label}[{i}]", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel(xlabel, fontsize=9)
    finish(fig, plt, out)


def plot_action_heatmap(plt, files: list[Path], out: Path | None, max_eps: int = 50) -> None:
    """Stack the action trajectories of several episodes into one heatmap per dimension."""
    files = files[:max_eps]
    print(f"  Building action heatmap from {len(files)} episodes ...")
    episodes = []
    for f in files:
        t = pq.read_table(f, columns=["action"])
        episodes.append(np.array(t.column("action").to_pylist(), dtype=np.float32))
    if not episodes:
        print("  No action data for the heatmap.")
        return

    target_len = 200  # resample every episode to this length for display
    mat = np.stack([ep[np.linspace(0, len(ep) - 1, target_len).astype(int)] for ep in episodes])
    n_dim = mat.shape[2]

    fig, axes = plt.subplots(1, n_dim, figsize=(3.5 * n_dim, 5), squeeze=False)
    fig.suptitle(f"Actions of {len(files)} episodes (resampled to {target_len} steps)", fontsize=13)
    for i in range(n_dim):
        ax = axes[0, i]
        data = mat[:, :, i]
        vmax = float(np.abs(data).max()) or 1.0
        im = ax.imshow(data, aspect="auto", origin="upper", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[0, target_len, len(files), 0])
        ax.set_title(f"action[{i}]", fontsize=9)
        ax.set_xlabel("resampled step", fontsize=8)
        if i == 0:
            ax.set_ylabel("episode", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    finish(fig, plt, out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot frames, actions and states from a dataset saved by libero10_augmentation.py.",
    )
    parser.add_argument("--input", default="outputs/augmented_libero10",
                        help="Dataset directory written by Dataset.save_lerobot (default: %(default)s).")
    parser.add_argument("--episode", type=int, default=0, metavar="N",
                        help="Episode index to plot (default: %(default)s).")
    parser.add_argument("--output", default="outputs/augmented_libero10_vis",
                        help="Directory for the PNG files (default: %(default)s).")
    parser.add_argument("--frames", type=int, default=8,
                        help="Number of frames in the camera grid (default: %(default)s).")
    parser.add_argument("--heatmap", action="store_true",
                        help="Also plot an action heatmap across up to 50 episodes.")
    parser.add_argument("--show", action="store_true",
                        help="Show the figures in a window instead of saving PNG files.")
    args = parser.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = Path(args.input) / "data"
    output_dir = Path(args.output)
    if not args.show:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    def out(name: str) -> Path | None:
        return None if args.show else output_dir / name

    path = find_episode_file(data_dir, args.episode)
    print(f"Episode {args.episode}: {path}")
    table = pq.read_table(path)
    print(f"  {table.num_rows} steps; columns: {table.schema.names}")

    stem = f"episode_{args.episode:06d}"
    plot_image_grid(plt, table, args.episode, args.frames, out(f"{stem}_frames.png"))
    plot_series(plt, table, "action", "action", args.episode, out(f"{stem}_actions.png"))
    plot_series(plt, table, "observation.state", "state", args.episode, out(f"{stem}_state.png"))

    if args.heatmap:
        plot_action_heatmap(plt, sorted(data_dir.rglob("episode_*.parquet")), out("action_heatmap.png"))

    print("Done.")


if __name__ == "__main__":
    main()
