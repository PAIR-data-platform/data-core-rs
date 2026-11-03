#!/usr/bin/env bash
# Feature-cache training across seeds 42, 43 and 44 (2000 steps each):
#   - MimicGen Coffee D0: spatial-map features (native 84x84 frames),
#     train on episodes 0-79, validate on 80-99, batch 64
#   - Aloha static coffee: pooled features (frames resized to 224x224),
#     train on episodes 0-39, validate on 40-49, batch 8
# Each feature cache (frozen ImageNet ResNet-18) is built first if missing.
# Runs whose summary.json already exists are skipped.
#
# Paper: the "3-seed held-out L1 (coffee map / aloha pooled)" row of Table 6.
#
# Requires a CUDA GPU and LeRobot (the caches are built through LeRobotDataset
# with torchcodec); stock LeRobot is enough.
#
# Environment:
#   PYTHON       Python interpreter (default: python)
#   COFFEE_ROOT  Coffee dataset directory (default: Hugging Face cache snapshot,
#                see scripts/dataset_paths.sh)
#   ALOHA_ROOT   Aloha dataset directory (default: Hugging Face cache snapshot)
#
# Outputs (under outputs/): the two feature caches and one directory per run,
# feature_cache_coffee_80train_20val_map_seed<seed>/ and
# feature_cache_aloha_coffee_40train_10val_pooled_seed<seed>/, each with
# summary.json (held-out L1 in best_val_l1 / last_val_l1) and model.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/dataset_paths.sh
source scripts/dataset_paths.sh
require_coffee
require_aloha

PYTHON="${PYTHON:-python}"
COFFEE_CACHE="outputs/feature_cache_coffee_100ep_resnet18_map"
ALOHA_POOLED_CACHE="outputs/feature_cache_aloha_coffee_50ep_resnet18_pooled"

if [[ ! -f "${COFFEE_CACHE}/manifest.json" ]]; then
  "${PYTHON}" scripts/build_feature_cache.py \
    --repo-id local/mimicgen_coffee_d0 \
    --root "${COFFEE_ROOT}" \
    --episodes 0:100 \
    --output-dir "${COFFEE_CACHE}" \
    --video-backend torchcodec \
    --batch-size 128 \
    --num-workers 4 \
    --image-size 84 \
    --feature-kind map \
    --device cuda
fi

if [[ ! -f "${ALOHA_POOLED_CACHE}/manifest.json" ]]; then
  "${PYTHON}" scripts/build_feature_cache.py \
    --repo-id lerobot/aloha_static_coffee \
    --root "${ALOHA_ROOT}" \
    --episodes 0:50 \
    --output-dir "${ALOHA_POOLED_CACHE}" \
    --video-backend torchcodec \
    --batch-size 64 \
    --num-workers 4 \
    --image-size 224 \
    --feature-kind pooled \
    --device cuda
fi

for seed in 42 43 44; do
  out="outputs/feature_cache_coffee_80train_20val_map_seed${seed}"
  if [[ -f "${out}/summary.json" ]]; then
    echo "Skipping existing ${out}"
  else
    "${PYTHON}" scripts/train_act_feature_cache.py \
      --cache-dir "${COFFEE_CACHE}" \
      --episodes 0:80 \
      --val-cache-dir "${COFFEE_CACHE}" \
      --val-episodes 80:100 \
      --batch-size 64 \
      --num-workers 4 \
      --steps 2000 \
      --chunk-size 100 \
      --dim-model 512 \
      --n-heads 8 \
      --dim-feedforward 3200 \
      --n-encoder-layers 4 \
      --n-decoder-layers 1 \
      --log-freq 100 \
      --val-freq 500 \
      --seed "${seed}" \
      --output-dir "${out}"
  fi
done

for seed in 42 43 44; do
  out="outputs/feature_cache_aloha_coffee_40train_10val_pooled_seed${seed}"
  if [[ -f "${out}/summary.json" ]]; then
    echo "Skipping existing ${out}"
  else
    "${PYTHON}" scripts/train_act_feature_cache.py \
      --cache-dir "${ALOHA_POOLED_CACHE}" \
      --episodes 0:40 \
      --val-cache-dir "${ALOHA_POOLED_CACHE}" \
      --val-episodes 40:50 \
      --batch-size 8 \
      --num-workers 4 \
      --steps 2000 \
      --chunk-size 100 \
      --dim-model 512 \
      --n-heads 8 \
      --dim-feedforward 3200 \
      --n-encoder-layers 4 \
      --n-decoder-layers 1 \
      --log-freq 100 \
      --val-freq 500 \
      --seed "${seed}" \
      --output-dir "${out}"
  fi
done
