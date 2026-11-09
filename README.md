# data-core

[![CI](https://github.com/PAIR-data-platform/data-core-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/PAIR-data-platform/data-core-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10 to 3.13](https://img.shields.io/badge/python-3.10%20to%203.13-blue.svg)
[![Paper](https://img.shields.io/badge/paper-CoRL%202026-b31b1b.svg)](https://openreview.net/forum?id=Vv1WXMkSTD)
[![Website](https://img.shields.io/badge/website-project%20page-0a7d5a.svg)](https://pair-data-platform.github.io/data-core-rs/)

A Rust data engine for robot learning datasets, with Python bindings. It reads
[LeRobot](https://github.com/huggingface/lerobot) datasets (v2 and v3), decodes
video with FFmpeg into Apache Arrow buffers, applies augmentations in Rust, and
hands samples to PyTorch.

This repository accompanies the paper

> **An Empirical Study of Data-Loading Bottlenecks (and Non-Bottlenecks) in Robotic Imitation Learning**
> Mrinal Jain, Arpit Gupta, Animesh Garg. Conference on Robot Learning (CoRL), 2026.
> [OpenReview](https://openreview.net/forum?id=Vv1WXMkSTD) · [PDF](https://openreview.net/pdf?id=Vv1WXMkSTD) · [Project page](https://pair-data-platform.github.io/data-core-rs/)

It contains the engine and the scripts that launch the training, benchmark and
evaluation runs reported in the paper.

## What to expect

The paper profiles end-to-end policy training in LeRobot. In the ACT
configurations it measures, the data-path ratio r = data_s / updt_s (the time
the trainer waits for a batch over the time of the forward and backward pass)
is 1.1 to 3.5%. In the configuration where it measured end-to-end wall-clock
(MimicGen Coffee, 84x84, batch 64, three seeds), torchcodec took 326 ± 4 s and
this engine's copy-on-write backend 324 ± 1 s, within run-to-run noise, even
though the engine is up to 3.9x faster in an isolated single-process
benchmark. The exception is a decoder without indexed seeking (PyAV on AV1
video), where the data path dominates.

Use data-core where its other properties matter: one shared decoded copy of the
dataset for many DataLoader workers, decoding that finishes before workers
fork, online augmentation in Rust, offline dataset augmentation, and
measuring how much of your own training step the data path takes. Profile your
setup before expecting a speed-up. [FEATURES.md](FEATURES.md) lists the
features and the cases each one fits.

## Features

- **Lazy loading**: open a dataset from its metadata alone and decode episodes on demand, with an optional cache limit.
- **Copy-on-write shared cache**: decode episodes once in the parent process and share them with forked DataLoader workers, with no per-worker copies and no FFmpeg in the workers.
- **Augmentations in Rust**: image, action and state augmentations, including a fused single-pass image augmentation that, with `window_size` set, keeps crop, jitter and flip consistent across the frames of a sample.
- **Custom augmentations in Python**: per-episode and whole-dataset augmentation classes over PyArrow record batches.
- **PyTorch frontend**: frame, chunk and episode sampling modes with ready-made DataLoaders.
- **Typed Python API**: the package ships type information (`py.typed` and a stub for the Rust module) for editors and type checkers.

## Repository layout

| Path | Contents |
|---|---|
| `crates/core` | Arrow-backed `Episode` and `Dataset` types |
| `crates/io` | LeRobot readers and writer (Parquet, FFmpeg video), lazy loading, Hub downloads |
| `crates/augmentations` | Augmentation traits and built-in augmentations |
| `crates/pipeline` | Pipeline builder and executors |
| `crates/python` | PyO3 bindings, built into the `data_core._native` extension module |
| `src/main.rs` | `data-engine` CLI: `data-engine info <path>` prints a dataset summary |
| `python/data_core` | The Python package: re-exports the Rust module (`data_core._native`, typed by `_native.pyi`) and adds `data_core.torch`, a PyTorch `Dataset` with LeRobot-style samples |
| `examples/` | Examples for the Python API |
| `benches/` | A Rust pipeline benchmark and a Python loading benchmark |
| `scripts/` | Training, benchmark, feature-cache and rollout-evaluation scripts |
| `lerobot-scripts/` | `lerobot-train` launchers and a GPU-utilization logger |
| `tests/` | Python tests (Rust tests live next to the code and in `tests/cli.rs`) |

## Installation

### Requirements

- Linux (tested on Ubuntu 22.04 and 24.04)
- Rust, stable toolchain (CI uses 1.94)
- Python 3.10 to 3.13
- FFmpeg development libraries (tested with FFmpeg 4.4 and 6.1), `pkg-config`, `clang` and `libclang`

On Ubuntu or Debian:

```bash
sudo apt-get install -y \
  libavcodec-dev libavformat-dev libavutil-dev \
  libavfilter-dev libavdevice-dev libswscale-dev libswresample-dev \
  pkg-config clang libclang-dev
```

ffmpeg-next, the FFmpeg binding used here, supports FFmpeg up to 7.1; only
4.4 and 6.1 are tested.

### Build the Python module

```bash
git clone https://github.com/PAIR-data-platform/data-core-rs.git
cd data-core-rs
python3 -m venv .venv && source .venv/bin/activate   # needs the python3-venv package on Ubuntu
pip install "maturin[patchelf]"
maturin develop --release
python -c "import data_core; print('OK')"
```

If the build fails because bindgen cannot find `limits.h`, clang's own headers
are missing (usually because `clang` and `libclang-dev` from the list above are
not installed). Install them, or point bindgen at your GCC headers. Adjust the
version (`13`) to match `ls /usr/lib/gcc/x86_64-linux-gnu/`, and keep the
variable exported for later `cargo test`, `cargo run` and `pip install -e`
builds:

```bash
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/13/include"
```

If linking fails on a cluster with a non-default toolchain, try:

```bash
RUSTFLAGS="-C linker=cc" AR="" maturin develop --release
```

The `Dockerfile` builds an image with CUDA, FFmpeg, Rust and the module
installed (`docker build -t data-core .`).

### Optional dependencies

```bash
pip install -e ".[dev]"     # pytest and Pillow, for tests/
pip install -e ".[torch]"   # PyTorch, for torch_dataloader and data_core.torch
pip install -e ".[bench]"   # LeRobot, PyTorch, torchcodec and plotting, for scripts/, benches/ and examples/
```

The experiments used LeRobot 0.4.4; the `bench` extra allows 0.4.x. The
MimicGen rollout scripts additionally need `robosuite`, `mimicgen`, `h5py`,
`scipy` and `imageio`, and the GPU logger needs `nvidia-ml-py`. For type
checking code that uses PyArrow types, install `pyarrow-stubs`.

## Quick start

### Lazy loading

```python
from data_core import LazyDataset

ds = LazyDataset.open("/path/to/lerobot/dataset")   # reads metadata only
print(len(ds), ds.num_episodes, ds.camera_names())

frame = ds[42]                       # decodes the containing episode on demand
print(frame.action, frame.camera_names())

subset = ds.with_episodes([0, 1, 5])  # frame indices now count these episodes only
subset.preload_episodes([0, 1, 5])    # decode once, share with forked workers
```

`preload_episodes` holds every frame of the chosen episodes in RAM: about
4 GB per episode of `lerobot/aloha_static_coffee` (1100 frames, 4 cameras at
480x640) in raw-pixel mode.

### PyTorch DataLoader

This needs PyTorch (`pip install torch`, or the `torch` extra).

```python
from data_core import LazyDataset

lazy = LazyDataset.open("/path/to/lerobot/dataset").with_episodes([0, 1])

# Random frames
loader = lazy.torch_dataloader(mode="random", batch_size=64, shuffle=True)

# Contiguous chunks of frames
loader = lazy.torch_dataloader(mode="chunk", chunk_size=8, batch_size=16, shuffle=True)

# Whole episodes, padded or truncated to a fixed length
loader = lazy.torch_dataloader(
    mode="episode", batch_size=4, shuffle=True,
    truncate_to_length=128, pad_to_length=128,
)
```

`lazy.to_torch_dataset(...)` returns the underlying map-style dataset if you
want to build the DataLoader yourself. Episode mode returns float32 images, so
a batch of long high-resolution episodes is large: 4 episodes of 128 frames
from 4 cameras at 480x640 take about 7.5 GB.

### Augmentation pipelines

```python
from data_core import Dataset, Pipeline, ParallelConfig

ds = Dataset.from_local("/path/to/lerobot/dataset")   # loads the full dataset

pipeline = Pipeline()
# "cam_high" is an Aloha camera: use names from ds.camera_names(). run()
# raises ValueError for a camera the dataset does not have.
pipeline.add_builtin("color_jitter", '{"brightness": 0.2, "camera_names": ["cam_high"]}')
pipeline.add_builtin("action_noise", '{"std": 0.01}')

augmented = pipeline.run(ds)

# Or run per-episode steps in parallel
ds = ds.with_parallel_config(ParallelConfig().set_num_threads(8))
augmented = pipeline.run_parallel(ds)

augmented.save_lerobot("/path/to/output")
```

`Dataset.from_local` decodes the whole dataset into memory in one thread: on
`lerobot/aloha_static_coffee` (50 episodes) that takes about 16 minutes and
8 GB of RAM. Start with a small dataset; for frame-level access to a large one,
use `LazyDataset` instead.

Built-in augmentations: `color_jitter`, `random_crop`, `gaussian_blur`,
`gaussian_noise`, `horizontal_flip`, `frame_dropout`, `fused_image`,
`action_noise`, `state_noise`, `state_scaling`, `temporal_subsample`. Image
augmentations need `camera_names` (the names from `camera_names()`). Their
parameters and defaults are listed in
[FEATURES.md](FEATURES.md#built-in-augmentations).

### Custom augmentations

A local augmentation maps one episode to one episode. A global augmentation
maps the list of all episodes to a new list, so it can add or remove episodes.
Both are plain Python classes with a `run` method over PyArrow record batches.

```python
import pyarrow as pa
from data_core import Dataset, Pipeline

class DropLastSteps:
    def __init__(self, n: int):
        self.n = n

    def run(self, episode: pa.RecordBatch) -> pa.RecordBatch:
        return episode.slice(0, max(1, len(episode) - self.n))

class DuplicateEpisodes:
    def run(self, episodes: list[pa.RecordBatch], metadata: dict) -> list[pa.RecordBatch]:
        return episodes + episodes

pipeline = Pipeline()
pipeline.add_local(DropLastSteps(5))
pipeline.add_global(DuplicateEpisodes())
augmented = pipeline.run(Dataset.from_local("/path/to/lerobot/dataset"))
```

The `metadata` dict carries `name`, `robot_type`, `fps`, `action_dim`,
`state_dim`, `camera_names`, `tasks`, `num_episodes`, `num_steps` and
`source`.

Longer examples are in `examples/`:

| Example | Shows |
|---|---|
| `custom_augmentation_local.py` | Per-episode augmentation classes (action smoothing, truncation, noise), run on synthetic episodes |
| `custom_augmentation_global.py` | Dataset-level augmentation classes (duplication, trajectory stitching, balancing), run on synthetic episodes |
| `end_to_end_example.py` | Local and global augmentation classes chained on synthetic episodes |
| `libero10_augmentation.py` | An image and state augmentation pipeline on `lerobot/libero_10` |
| `visualize_augmented.py` | Plots frames, actions and states from an augmented dataset |
| `train_act_lazy.py` | ACT training on the `LazyDataset` PyTorch frontend |
| `port_mimicgen.py` | Converting a MimicGen dataset from LeRobot v2.1 to v3.0 |

### PyTorch `Dataset` with LeRobot-style samples

`data_core.torch` wraps `LazyDataset` in a `torch.utils.data.Dataset` that
returns the same keys as `LeRobotDataset` (`observation.images.*`,
`observation.state`, `action`, `action_is_pad`, ...), with a DataLoader helper
that preloads the copy-on-write cache and groups samples by episode. It needs
PyTorch.

```python
from data_core.torch import AugmentedDataset, make_dataloader

# Keys and timestamps are for a 50 fps dataset with a camera named "top";
# use your dataset's camera names and multiples of 1 / fps (other values raise
# ValueError).
ds = AugmentedDataset(
    "/path/to/lerobot/dataset",
    delta_timestamps={"observation.images.top": [0.0], "action": [i / 50 for i in range(100)]},
)
# With workers, the chosen episodes (default: all) are preloaded into RAM.
loader = make_dataloader(ds, batch_size=8, num_workers=4, preload_episodes=[0, 1])
```

### Command line

```bash
cargo run --release -- info /path/to/lerobot/dataset
```

## Tests and benchmarks

```bash
cargo test --workspace
pytest tests/          # after pip install -e ".[dev]"; the PyTorch tests also need torch

cargo bench --bench loading_benchmark        # Rust pipeline timings, sequential vs parallel (needs lerobot/libero_10)
python benches/loading_benchmark.py          # Python, next to LeRobotDataset
python scripts/benchmark_lazy_dataset.py     # LazyDataset access patterns (needs lerobot/aloha_static_fork_pick_up)
```

`benchmark_lazy_dataset.py` repeats each pattern `--runs` times (default 3);
the cold random-access pattern decodes a full 4-camera episode per sample and
dominates the run time, so pass `--runs 1` for a quick check.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup.

## Experiment scripts

These are the scripts behind the runs in the paper. Each script's header says
what it runs, which table it corresponds to, and what it needs. Run them from
the repository root; they write to `outputs/` and `results/`, which are
git-ignored.

Datasets, pinned revisions and download commands are in
[data/README.md](data/README.md). The scripts read the datasets from the
Hugging Face cache by default; set `COFFEE_ROOT`, `ALOHA_ROOT`,
`MIMICGEN_RAW_DIR` or `LIBERO10_ROOT` to use copies elsewhere.

### LeRobot fork

The paper's `lerobot-train` runs on the data-core backends use a fork of LeRobot
0.4.4, [PAIR-data-platform/lerobot](https://github.com/PAIR-data-platform/lerobot).
Its `main` branch adds the dataset flags `--dataset.use_data_core`,
`--dataset.preload_episodes` and `--dataset.data_core_aug_preset`, and a
MimicGen environment for closed-loop evaluation. The branch
`arpit/arrow-jpeg-on-main` is `main` plus the Arrow backend
(`--dataset.use_data_core_arrow`), so checking it out covers every launcher.
Install the fork in the same environment as `data_core`. Clone it next to
`data-core-rs` rather than inside it:

```bash
cd ..   # from the data-core-rs checkout
git clone https://github.com/PAIR-data-platform/lerobot.git
git -C lerobot checkout arpit/arrow-jpeg-on-main   # needed only for the Arrow backend
pip install -e "./lerobot[mimicgen]"
# MimicGen task registrations for the rollout scripts
git clone --branch v0.1.1 https://github.com/NVlabs/mimicgen.git
pip install -e ./mimicgen   # pins mujoco==2.3.2, replacing the newer mujoco from the extra
cd data-core-rs
```

Scripts marked "fork" below need it; the others run on stock LeRobot or use
`data_core` directly. The SmolVLA configuration additionally needs LeRobot's
SmolVLA and LIBERO extras (`pip install -e "./lerobot[smolvla,libero]"`, or
`pip install "lerobot[smolvla,libero]==0.4.4"` on stock LeRobot).

### `lerobot-train` launchers (`lerobot-scripts/`)

| Script | Run | Paper | Needs |
|---|---|---|---|
| `act_coffee_torchcodec.sh` | ACT on MimicGen Coffee, torchcodec | Tables 1, 3, 4 | stock LeRobot |
| `act_coffee_torchcodec_aug.sh` | Same, with LeRobot image transforms | Table 4 | stock LeRobot |
| `act_coffee_cow.sh` | Same, data-core copy-on-write cache | Tables 1, 3, 4 | fork |
| `act_coffee_cow_aug.sh` | Same, with Rust `fused_image` augmentation | Table 4 | fork |
| `act_coffee_arrow_jpeg.sh` | Same, data-core Arrow backend with a JPEG cache | Table 3 | fork (Arrow branch) |
| `act_aloha_torchcodec.sh` | ACT on Aloha at 480x640, torchcodec | Table 1 | stock LeRobot |
| `act_aloha_small.sh` | Smaller ACT on Aloha; Table 1 is `act_aloha_small.sh torchcodec 16` (Arrow JPEG is also available but not in the paper) | Table 1 | stock LeRobot (torchcodec), fork (Arrow branch) |
| `smolvla_libero_10.yaml` | SmolVLA trained on `HuggingFaceVLA/libero`, evaluated on LIBERO-10 (the torchcodec row; add `--dataset.image_transforms.enable=true` for the +aug row) | Table 5 | stock LeRobot (rust rows: fork) |
| `nvml_gpu_usage_wrapper.sh` | Wraps any command and logs GPU utilization through NVML | Table 5 | `nvidia-ml-py` |

`scripts/run_wallclock_seeds.sh` runs the torchcodec and copy-on-write Coffee
launchers for seeds 42 to 44 and prints each run's training wall-clock
(Table 3; needs the fork). The PyAV row of Table 1 is the torchcodec
Coffee launcher with the decoder switched:
`lerobot-scripts/act_coffee_torchcodec.sh --dataset.video_backend pyav`.

The data-path ratio r in Tables 1 and 4 is computed from the `data_s` and
`updt_s` values that `lerobot-train` logs every 100 steps: drop the windows
logged at steps 100 and 200 as warm-up, then divide the mean `data_s` by the
mean `updt_s`.

### Standalone scripts (`scripts/`)

| Script | Purpose | Needs |
|---|---|---|
| `train_act_lerobot.py` | ACT training loop on `LeRobotDataset` | LeRobot |
| `train_act_data_core.py` | The same loop on `data_core.LazyDataset`. With DataLoader workers it preloads every episode into RAM unless `--preload-episodes N` is given | LeRobot (policy only) |
| `benchmark_lazy_dataset.py` | `LazyDataset` throughput for sequential, warm and cold random access | `data_core` |
| `convert_mimicgen_hdf5_to_lerobot_v3.py` | Convert a MimicGen HDF5 file to LeRobot v3 | LeRobot, `h5py` |
| `run_wallclock_seeds.sh` | Torchcodec and copy-on-write Coffee launchers for seeds 42 to 44; prints each run's training wall-clock (Table 3) | fork |
| `dataset_paths.sh` | Default dataset locations and pinned revisions, sourced by the shell scripts | none |
| `run_act_baseline_rollouts.sh` | Stock ACT on MimicGen Coffee (all 1000 episodes), 10k steps, with 25 closed-loop rollouts from randomly drawn initial states | fork (MimicGen env) |

The rollout options (`--eval-*`) of the two training loops also need the
fork's MimicGen environment.

### Feature cache (`scripts/`, Table 6)

These scripts precompute frozen ResNet-18 features once and train the ACT
action head from the cache instead of running the backbone every step.

| Script | Purpose | Needs |
|---|---|---|
| `build_feature_cache.py` | Run a frozen ResNet over a dataset and write one feature shard per episode | LeRobot |
| `train_act_feature_cache.py` | Train the action head from cached features | PyTorch |
| `train_act_frozen_online.py` | Baseline: the same model with features recomputed every step | LeRobot |
| `run_feature_cache_seeds.sh` | Cached training on Coffee (map features) and Aloha (pooled features), seeds 42 to 44 | LeRobot |
| `run_feature_cache_coffee.sh` | Build the Coffee cache, train for 10k steps, evaluate with rollouts | LeRobot; rollouts need the fork |
| `run_feature_cache_rollouts.sh` | 25-rollout evaluation of that policy | fork (MimicGen env) |
| `eval_feature_act_mimicgen.py`, `extract_mimicgen_init_states.py` | Rollout evaluation and deterministic initial states | evaluation: fork (MimicGen env), `scipy`; init states: `h5py` |

Memory: with its default settings (4 workers, 4 cameras at 480x640), the Aloha
cache build in `run_feature_cache_seeds.sh` needs more than 12 GB of RAM.

## Citation

```bibtex
@inproceedings{jain2026dataloading,
  title     = {An Empirical Study of Data-Loading Bottlenecks (and Non-Bottlenecks) in Robotic Imitation Learning},
  author    = {Jain, Mrinal and Gupta, Arpit and Garg, Animesh},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Vv1WXMkSTD}
}
```

## License

MIT. See [LICENSE](LICENSE). `examples/port_mimicgen.py` is adapted from LeRobot
and keeps its Apache-2.0 license header.
