#!/usr/bin/env bash
# Feature-cache pipeline on MimicGen Coffee D0:
#   1. build a spatial-map feature cache (frozen ImageNet ResNet-18, native
#      84x84 frames) for episodes 0-99 with scripts/build_feature_cache.py
#   2. train FeatureACT on episodes 0-79 (validation on 80-99) for 10k steps at
#      batch 64 with mean/std normalization of state and action
#   3. extract the deterministic MimicGen init states if missing
#   4. quick closed-loop checks on init state 80 and on init states 80-84
#
# Paper: produces the 10k-step cached-feature policy whose 25-rollout result is
# the "cached map (coffee, 10k steps)" row of Table 6; run
# run_feature_cache_rollouts.sh afterwards for that evaluation.
#
# Requires a CUDA GPU and LeRobot (the cache is built through LeRobotDataset
# with torchcodec). Step 3 needs h5py and $MIMICGEN_RAW_DIR/core/coffee_d0.hdf5
# if the init states are not extracted yet (see data/README.md). Step 4 needs
# robosuite, mimicgen, scipy and the MimicGen environment lerobot.envs.mimicgen
# (provided by the data-core LeRobot fork). Steps 1 and 2 run on stock
# LeRobot.
#
# Environment:
#   PYTHON            Python interpreter (default: python)
#   COFFEE_ROOT       dataset directory (default: Hugging Face cache snapshot,
#                     see scripts/dataset_paths.sh)
#   MIMICGEN_RAW_DIR  raw MimicGen directory (default: data/mimicgen_raw)
#
# Outputs (under outputs/):
#   feature_cache_coffee_100ep_resnet18_map/                 feature cache
#   feature_cache_coffee_80train_20val_map_norm_train_10k/   model.pt, summary.json
#   feature_act_mimicgen_eval_coffee_norm10k_ep80_exec100/   1-rollout check
#   feature_act_mimicgen_eval_coffee_norm10k_80_85_exec100/  5-rollout check
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/dataset_paths.sh
source scripts/dataset_paths.sh
require_coffee

PYTHON="${PYTHON:-python}"
CACHE_DIR="outputs/feature_cache_coffee_100ep_resnet18_map"
INIT_STATES="${MIMICGEN_RAW_DIR}/init_states_coffee_d0.pt"
TRAIN_OUT="outputs/feature_cache_coffee_80train_20val_map_norm_train_10k"

if [[ ! -f "${CACHE_DIR}/manifest.json" ]]; then
  "${PYTHON}" scripts/build_feature_cache.py \
    --repo-id local/mimicgen_coffee_d0 \
    --root "${COFFEE_ROOT}" \
    --episodes 0:100 \
    --output-dir "${CACHE_DIR}" \
    --video-backend torchcodec \
    --batch-size 128 \
    --num-workers 4 \
    --image-size 84 \
    --feature-kind map \
    --device cuda
fi

"${PYTHON}" scripts/train_act_feature_cache.py \
  --cache-dir "${CACHE_DIR}" \
  --episodes 0:80 \
  --val-cache-dir "${CACHE_DIR}" \
  --val-episodes 80:100 \
  --batch-size 64 \
  --num-workers 4 \
  --steps 10000 \
  --chunk-size 100 \
  --dim-model 512 \
  --n-heads 8 \
  --dim-feedforward 3200 \
  --n-encoder-layers 4 \
  --n-decoder-layers 1 \
  --log-freq 500 \
  --val-freq 2500 \
  --normalize-state \
  --normalize-action \
  --device cuda \
  --output-dir "${TRAIN_OUT}"

if [[ ! -f "${INIT_STATES}" ]]; then
  "${PYTHON}" scripts/extract_mimicgen_init_states.py \
    --input-hdf5 "${MIMICGEN_RAW_DIR}/core/coffee_d0.hdf5" \
    --output "${INIT_STATES}"
fi

"${PYTHON}" scripts/eval_feature_act_mimicgen.py \
  --checkpoint "${TRAIN_OUT}/model.pt" \
  --cache-dir "${CACHE_DIR}" \
  --init-states-path "${INIT_STATES}" \
  --episode-indices 80:81 \
  --max-steps 400 \
  --exec-horizon 100 \
  --device cuda \
  --progress-freq 100 \
  --output-dir outputs/feature_act_mimicgen_eval_coffee_norm10k_ep80_exec100

"${PYTHON}" scripts/eval_feature_act_mimicgen.py \
  --checkpoint "${TRAIN_OUT}/model.pt" \
  --cache-dir "${CACHE_DIR}" \
  --init-states-path "${INIT_STATES}" \
  --episode-indices 80:85 \
  --max-steps 400 \
  --exec-horizon 100 \
  --device cuda \
  --progress-freq 100 \
  --output-dir outputs/feature_act_mimicgen_eval_coffee_norm10k_80_85_exec100
