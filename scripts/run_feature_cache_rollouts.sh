#!/usr/bin/env bash
# Closed-loop evaluation of the cached-feature Coffee policy (FeatureACT on
# spatial-map features) trained by run_feature_cache_coffee.sh: 25 MimicGen
# rollouts from the deterministic initial states of episodes 80-104
# (400-step horizon, actions executed in chunks of 100). The policy was trained
# on episodes 0-79, so these are held-out initial states.
# run_act_baseline_rollouts.sh uses the same number of rollouts and horizon,
# but draws its initial states at random from all recorded ones.
#
# Paper: the "MimicGen rollout success, cached map (coffee, 10k steps)" row of
# Table 6 and Appendix C.
#
# Requires a CUDA GPU, robosuite and mimicgen, the MimicGen environment
# lerobot.envs.mimicgen (provided by the data-core LeRobot fork), and
# the outputs of run_feature_cache_coffee.sh (checkpoint, feature cache and
# init states).
#
# Environment:
#   PYTHON            Python interpreter (default: python)
#   MIMICGEN_RAW_DIR  raw MimicGen directory (default: data/mimicgen_raw)
#
# Output: outputs/feature_cache_rollouts_25/summary.json and
# outputs/feature_cache_rollouts_25.log.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/dataset_paths.sh
source scripts/dataset_paths.sh

PYTHON="${PYTHON:-python}"
CHECKPOINT=outputs/feature_cache_coffee_80train_20val_map_norm_train_10k/model.pt
INIT_STATES="${MIMICGEN_RAW_DIR}/init_states_coffee_d0.pt"

"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'"

for f in "$CHECKPOINT" "$INIT_STATES"; do
  if [[ ! -f "$f" ]]; then
    echo "error: $f not found; run scripts/run_feature_cache_coffee.sh first" >&2
    exit 1
  fi
done

mkdir -p outputs
"$PYTHON" scripts/eval_feature_act_mimicgen.py \
  --checkpoint "$CHECKPOINT" \
  --task Coffee_D0 \
  --init-states-path "$INIT_STATES" \
  --episode-indices 80:105 \
  --n-rollouts 25 \
  --exec-horizon 100 \
  --output-dir outputs/feature_cache_rollouts_25 \
  2>&1 | tee outputs/feature_cache_rollouts_25.log
