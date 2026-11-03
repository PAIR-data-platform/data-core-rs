# Data

Everything in this directory except this file is git-ignored. The scripts read
datasets from the Hugging Face cache by default; only the raw MimicGen file
lives here (`data/mimicgen_raw/`).

## Datasets

| Dataset | Revision | Size | Used by | Override |
|---|---|---|---|---|
| `mineral/lerobot_mimicgen_coffee_d0` (MimicGen Coffee D0, LeRobot v3, 1000 episodes, 2 cameras at 84x84, 20 fps) | `cbf90909acf57ce67b91b3a38abca40a911af4b2` | 0.28 GB | `lerobot-scripts/act_coffee_*.sh`, `scripts/run_wallclock_seeds.sh`, `scripts/run_feature_cache_*.sh`, `scripts/run_act_baseline_rollouts.sh`, default of `scripts/train_act_data_core.py` and `examples/train_act_lazy.py` | `COFFEE_ROOT` |
| `lerobot/aloha_static_coffee` (LeRobot v3, 50 episodes, 4 cameras at 480x640, 50 fps) | `b144896feb1f37398a862927b22cd3abdf005a6b` | 1.6 GB | `lerobot-scripts/act_aloha_*.sh`, `scripts/run_feature_cache_seeds.sh` | `ALOHA_ROOT` |
| `lerobot/libero_10` (LeRobot v3, 379 episodes, 2 cameras at 256x256, 10 fps) | `551d7d86f25edd0ffeda8b60053c15438cfd1d6a` (pinned in the example) | 0.65 GB | `examples/libero10_augmentation.py`, `benches/loading_benchmark.py`, `benches/loading_benchmark.rs` | `LIBERO10_ROOT` |
| `lerobot/aloha_static_fork_pick_up` (LeRobot v3, 100 episodes, 4 cameras at 480x640) | latest | 1.5 GB | `scripts/benchmark_lazy_dataset.py`, default of `scripts/train_act_lerobot.py` | `benchmark_lazy_dataset.py --root`; `train_act_lerobot.py --dataset-root` |
| `HuggingFaceVLA/libero` (all four LIBERO suites, 40 tasks; SmolVLA trains on it and evaluates on LIBERO-10) | latest | 35 GB | `lerobot-scripts/smolvla_libero_10.yaml` (downloaded by LeRobot) | `--dataset.root` |
| `locht131/mimicgen_100_lerobot` (LeRobot v2.1 source) | `61db41c96dd7876601fccbdb3313a30b1127e797` (pinned in the example) | 0.6 GB | `examples/port_mimicgen.py` | `--local-dir` |
| `amandlek/mimicgen_datasets`, file `core/coffee_d0.hdf5` (raw MimicGen Coffee D0) | latest | 2.0 GB | init states for the MimicGen rollouts (`scripts/run_feature_cache_coffee.sh`, `scripts/run_act_baseline_rollouts.sh`), `scripts/convert_mimicgen_hdf5_to_lerobot_v3.py` | `MIMICGEN_RAW_DIR` (rollout scripts); `--input-hdf5` (converter) |

The ACT experiments in the paper use `mineral/lerobot_mimicgen_coffee_d0`
(episodes 0-99 of 1000) and `lerobot/aloha_static_coffee` at the revisions
above. The isolated throughput measurements (Table 2) used
`lerobot/aloha_static_fork_pick_up`. The SmolVLA experiment trains on
`HuggingFaceVLA/libero` and evaluates on LIBERO-10.

## Download

The scripts expect the two ACT datasets as Hugging Face cache snapshots at the
pinned revisions:

```bash
hf download mineral/lerobot_mimicgen_coffee_d0 --repo-type dataset \
    --revision cbf90909acf57ce67b91b3a38abca40a911af4b2
hf download lerobot/aloha_static_coffee --repo-type dataset \
    --revision b144896feb1f37398a862927b22cd3abdf005a6b
hf download lerobot/libero_10 --repo-type dataset \
    --revision 551d7d86f25edd0ffeda8b60053c15438cfd1d6a
hf download lerobot/aloha_static_fork_pick_up --repo-type dataset
```

The raw MimicGen file goes into this directory:

```bash
hf download amandlek/mimicgen_datasets core/coffee_d0.hdf5 --repo-type dataset \
    --local-dir data/mimicgen_raw
```

The rollout scripts then create `data/mimicgen_raw/init_states_coffee_d0.pt`
with `scripts/extract_mimicgen_init_states.py` on first use.

`hf` is the Hugging Face CLI that ships with `huggingface_hub`
(`huggingface-cli download` takes the same arguments in older versions).

## Locations and overrides

Snapshots land in the Hugging Face hub cache: `$HF_HUB_CACHE` if set, else
`$HF_HOME/hub` if `HF_HOME` is set, else `~/.cache/huggingface/hub`. A snapshot
directory looks like
`<hub cache>/datasets--mineral--lerobot_mimicgen_coffee_d0/snapshots/<revision>/`.
The shell scripts build their defaults from these variables in
`scripts/dataset_paths.sh`.

To use a copy somewhere else, export the variable from the table before
running a script, for example:

```bash
export COFFEE_ROOT=/data/lerobot_mimicgen_coffee_d0
export ALOHA_ROOT=/data/aloha_static_coffee
export MIMICGEN_RAW_DIR=/data/mimicgen_raw   # holds core/coffee_d0.hdf5
bash scripts/run_wallclock_seeds.sh
```

The directory must be a LeRobot dataset root (it contains `meta/info.json`).
The shell scripts stop with a download hint if it does not.
