# What data-core provides

data-core is a Rust data engine for LeRobot datasets with Python bindings. This
page describes its main features and where each one lives in the code. For
installation and usage, see the [README](README.md).

## Overview

| Area | Feature | Code |
|---|---|---|
| Representation | Arrow-backed episodes | `crates/core` |
| Caching | Copy-on-write shared cache | `crates/io/src/lerobot/lazy_dataset.rs` |
| Caching | Lazy loading with a bounded cache | `crates/io/src/lerobot/lazy_dataset.rs` |
| Transfer | Three paths from cache to GPU | `crates/python/src/lib.rs` |
| Augmentation | Fused single-pass image augmentation | `crates/augmentations/src/image.rs` |
| Augmentation | Window-consistent augmentations | `crates/augmentations` |
| Augmentation | Custom augmentations in Python | `crates/pipeline`, `crates/python` |
| Input and output | LeRobot v2 and v3 readers, Parquet writer | `crates/io/src/lerobot` |
| Training | PyTorch frontend | `crates/python`, `python/data_core/torch` |
| Experiments | Launchers, timing and evaluation scripts | `scripts/`, `lerobot-scripts/` |

## Use cases

| If you want to | Use | Why it fits |
|---|---|---|
| Train with many DataLoader workers without per-worker copies of the decoded data | Copy-on-write shared cache | One decoded copy is shared by all workers, where per-worker caches multiply memory by the worker count. The preloaded episodes must still fit in RAM |
| Avoid FFmpeg crashes or deadlocks in forked workers | Copy-on-write shared cache | All decoding finishes in the parent before the fork |
| Explore or sample from a dataset larger than RAM in one process | Lazy loading with a cache limit | Opening reads metadata only, and cached episodes are evicted when the limit is reached |
| Fit more episodes in a preloaded cache | `arrow_uint8_jpeg` path | The cache holds compressed frames and decodes them on access |
| Reduce host-to-GPU transfer volume | `arrow_uint8` path | Frames cross as integers and can be normalized on the GPU |
| Add online image augmentation in Rust | `fused_image` | One pass per camera, with no intermediate buffers |
| Augment multi-frame samples for chunked policies such as ACT | `window_size` on image augmentations | All frames of a sample share the same crop, jitter and flip |
| Use horizontal flips on a bimanual or symmetric setup | `horizontal_flip` with action mirroring on raw pixels | Actions are mirrored along with the images |
| Prototype a new augmentation without writing Rust | Local augmentations in Python | A class with one `run` method over a PyArrow record batch |
| Rebalance, duplicate or stitch trajectories | Global augmentations in Python | They see every episode and can change how many there are |
| Generate an augmented dataset offline | `Pipeline` with `Dataset.save_lerobot` | The result is written as Parquet in a LeRobot v2-style layout with JPEG image columns; LeRobot itself expects v3 and does not read it directly |
| Bring MimicGen demonstrations into LeRobot | The HDF5-to-v3 converter | It writes LeRobot v3 directly |
| Try the engine in an existing LeRobot training loop | `data_core.torch` | It returns LeRobot-style sample dicts |
| Train sequence models on whole trajectories | Episode sampling mode | Episodes are padded or truncated to a fixed length |
| Measure how much of a training step your data path takes | Launchers, wall-clock script and NVML logger | They run the same model on each backend and record step time and GPU utilization |
| Iterate quickly on an action head with a frozen vision encoder | Feature-cache scripts | Visual features are computed once and reused every epoch |

### When it will not help

In the paper's ACT configurations the data-path ratio r = data_s / updt_s
(time the trainer waits for a batch over the time of the forward and backward
pass) is 1.1 to 3.5%. In the configuration where the paper measured end-to-end
wall-clock (MimicGen Coffee, 84x84, batch 64, three seeds), torchcodec took
326 ± 4 s and the data-core copy-on-write backend 324 ± 1 s, within run-to-run
noise. Use data-core for the memory, augmentation and measurement cases above,
and expect a speed-up only where your own profile shows the data path is a
large share of the step.

## Data representation and caching

### Arrow-backed episodes

Each episode is decoded once into an immutable Apache Arrow record batch that
holds its actions, states and image columns. Reading a frame is then a lookup
into that buffer rather than a video seek. The requested frames are copied into
a Python `bytes` object once per sample, without serialization or
inter-process transfer.

### Copy-on-write shared cache

`preload_episodes` decodes the chosen episodes in the parent process and
freezes them into a lock-free, heap-allocated cache. DataLoader workers forked
afterwards read the same physical pages through Linux copy-on-write. No episode
is decoded twice, nothing is pickled between processes, and the decoded cache
is not duplicated per worker. Because decoding finishes before the fork,
workers never touch FFmpeg. After a preload, reading an episode that was not
preloaded is an error.

### Lazy loading with a bounded cache

`LazyDataset.open` reads only the dataset metadata. Episodes are decoded on
first access and kept in a cache. When a cache limit is set and reached, an
arbitrary cached episode is evicted, so a dataset larger than RAM can still be
read. Use this in a single process; for forked DataLoader workers, preload
instead.

## Getting frames to the GPU

The engine can hand frames to PyTorch in three forms, the materialization
strategies of the paper, which trade host memory against transfer volume and
CPU work:

| Strategy | Cache holds | Work per access | Trade-off |
|---|---|---|---|
| `arrow_f32` | Raw pixels | Gather the window; transpose, cast and normalize in Rust (Rayon) | Least GPU work, four times the transfer volume |
| `arrow_uint8` | Raw pixels | Gather the window and copy it to Python; the caller normalizes, for example on the GPU as in `scripts/train_act_data_core.py` | Least transfer volume |
| `arrow_uint8_jpeg` | JPEG bytes | Parallel JPEG decode in Rust (Rayon, `image` crate) | Smallest cache, repeated CPU decode |

## Augmentation

### Fused single-pass image augmentation

`fused_image` applies crop, color jitter, noise and horizontal flip in one
read and one write per camera. Running the four as separate steps would
allocate three intermediate buffers and rebuild the Arrow column three more
times. It can write CHW output directly, ready to wrap as a tensor.

### Window-consistent augmentations

Policies such as ACT read several frames per sample. The image augmentations
take a `window_size` so that every frame of one sample gets the same crop,
jitter and flip. Horizontal flip can also mirror chosen action dimensions, which
keeps actions consistent with the flipped images.

### Built-in augmentations

`Pipeline.add_builtin(name, params_json)` and
`LazyDataset.add_augmentation(name, params_json)` take these names. Parameters
are given as a JSON object; missing ones take the defaults below and unknown
keys are ignored. Image augmentations require `camera_names`, using the names
returned by `camera_names()` (for example `"cam_high"`); a name the dataset
does not have raises ValueError (in `add_augmentation`, or when the pipeline
runs). They accept `window_size` where listed.

| Name | Parameters (default) |
|---|---|
| `action_noise` | `std` (0.01). Adds uniform noise in [-std, std] to the actions |
| `state_noise` | `std` (0.01), `column` (`"observation.state"`). Uniform noise on the state |
| `state_scaling` | `scale` (1.0), `column` (`"observation.state"`) |
| `temporal_subsample` | `step` (2). Keeps every `step`-th frame |
| `color_jitter` | `camera_names`, `brightness`, `contrast`, `saturation` (0.1 each), `window_size` (0) |
| `random_crop` | `camera_names`, `height`, `width` (224 each), `window_size` (0) |
| `gaussian_blur` | `camera_names`, `sigma` (1.0) |
| `gaussian_noise` | `camera_names`, `std` (0.02, as a fraction of 255; the noise is uniform in [-std, std]), `window_size` (0; accepted, but each frame still gets its own noise) |
| `horizontal_flip` | `camera_names`, `p` (0.5), `action_mirror_dims` (none), `window_size` (0) |
| `frame_dropout` | `camera_names`, `drop_prob` (0.1) |
| `fused_image` | `camera_names`, `crop_height` and `crop_width` (no crop), `brightness`, `contrast`, `saturation`, `noise_std`, `flip_p` (0 each, meaning off), `action_mirror_dims`, `chw_output` (false), `window_size` (0) |

`gaussian_noise`, `horizontal_flip`, `frame_dropout` and `fused_image` modify
only raw-pixel columns (`LazyDataset.with_raw_pixels`) and leave JPEG images
unchanged; `horizontal_flip` still mirrors the actions. `gaussian_blur`
modifies only JPEG columns. `Dataset.from_local` loads JPEG images.

### Custom augmentations in Python

An augmentation is a plain Python class with a `run` method over PyArrow record
batches.

- **Local** augmentations map one episode to one episode.
- **Global** augmentations map the list of all episodes to a new list, so they
  can add, remove or combine episodes.

A `Pipeline` chains built-in and custom steps and can run the per-episode steps
in parallel with Rayon.

## Input, output and training

### LeRobot v2 and v3

The readers handle both LeRobot layouts: Parquet tables plus FFmpeg-decoded
video. `LazyDataset` decodes video for v2 and v3; `Dataset.from_local` decodes
it only for v3. `Dataset.save_lerobot` writes a dataset as Parquet
in a LeRobot v2-style layout with JPEG image columns and minimal metadata
(no `features` or episode table), which `Dataset.from_local` reads back;
LeRobot does not read it, and `LazyDataset` opens it only with estimated
(equal-length) episode boundaries and no cameras.
`scripts/convert_mimicgen_hdf5_to_lerobot_v3.py` converts MimicGen HDF5 files
to LeRobot v3.

### PyTorch frontend

`LazyDataset.torch_dataloader` offers three sampling modes: random frames,
contiguous chunks, and whole episodes padded or truncated to a fixed length.
`data_core.torch` returns LeRobot-style sample dicts, so it can replace
`LeRobotDataset` in an existing training loop. It needs PyTorch.

## Experiment scripts

The repository includes the scripts used for the runs in the paper.

- **Training launchers:** ACT on MimicGen Coffee and Aloha for each backend
  (torchcodec, copy-on-write cache, Arrow with a JPEG cache), with and without
  augmentation, and a SmolVLA configuration for LIBERO-10. The data-core
  backends in `lerobot-train` need the LeRobot fork linked in the README.
- **Timing:** a script that repeats the Coffee runs across three seeds and
  reports training wall-clock, and a wrapper that logs GPU utilization through
  NVML.
- **Feature cache:** scripts that precompute frozen ResNet features once, train
  the ACT action head from the cache, and evaluate the policy with closed-loop
  MimicGen rollouts.
- **Benchmarks:** a Rust pipeline benchmark and a Python loading benchmark, the latter
  compared against `LeRobotDataset`.
