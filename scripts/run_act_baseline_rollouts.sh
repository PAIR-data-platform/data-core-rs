#!/usr/bin/env bash
# Stock LeRobot ACT on MimicGen Coffee D0 with closed-loop rollout evaluation.
# Trains the ACT architecture from Appendix A (trainable ResNet-18, CVAE,
# 7 decoder layers, chunk size 100) on all 1000 episodes for 10k steps at
# batch 64 (seed 42), and evaluates 25 MimicGen rollouts (400-step horizon) at
# steps 5000 and 10000; the environment draws each initial state at random from
# the recorded MimicGen initial states. This is the
# uncached baseline for the cached-feature policy evaluated by
# run_feature_cache_rollouts.sh (same training budget).
#
# Paper: the "MimicGen rollout success, stock LeRobot ACT (coffee, 10k steps)"
# row of Table 6 and the stock baseline in Appendix C.
#
# Uses scripts/train_act_lerobot.py, not lerobot-train. Requires a CUDA GPU,
# robosuite and mimicgen, and the MimicGen environment lerobot.envs.mimicgen,
# which is provided by the data-core LeRobot fork. The init states are
# extracted from $MIMICGEN_RAW_DIR/core/coffee_d0.hdf5 if missing (see
# data/README.md for the download).
#
# Environment:
#   PYTHON            Python interpreter (default: python)
#   COFFEE_ROOT       dataset directory (default: Hugging Face cache snapshot,
#                     see scripts/dataset_paths.sh)
#   MIMICGEN_RAW_DIR  raw MimicGen directory (default: data/mimicgen_raw)
#
# Output: outputs/act_baseline_mimicgen_10k/ (checkpoints, train_summary.json,
# run.log with the eval success rates).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/dataset_paths.sh
source scripts/dataset_paths.sh
require_coffee

PYTHON="${PYTHON:-python}"
INIT_STATES="${MIMICGEN_RAW_DIR}/init_states_coffee_d0.pt"
OUT=outputs/act_baseline_mimicgen_10k

"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'"

if [[ ! -f "$INIT_STATES" ]]; then
  "$PYTHON" scripts/extract_mimicgen_init_states.py \
    --input-hdf5 "${MIMICGEN_RAW_DIR}/core/coffee_d0.hdf5" \
    --output "$INIT_STATES"
fi

mkdir -p "$OUT"
"$PYTHON" scripts/train_act_lerobot.py \
  --dataset local/mimicgen_coffee_d0 \
  --dataset-root "$COFFEE_ROOT" \
  --backend torchcodec \
  --steps 10000 --batch-size 64 --seed 42 \
  --chunk-size 100 --n-decoder-layers 7 \
  --save-freq 5000 \
  --output-dir "$OUT" \
  --eval-freq 5000 --eval-n-episodes 25 --eval-n-envs 1 \
  --eval-env-task Coffee_D0 \
  --eval-init-states-path "$INIT_STATES" \
  2>&1 | tee "$OUT/run.log"
