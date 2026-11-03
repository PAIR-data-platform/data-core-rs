#!/usr/bin/env bash
# Training wall-clock across seeds 42, 43 and 44 for the Coffee 100-episode,
# batch-64 configuration with the torchcodec backend
# (lerobot-scripts/act_coffee_torchcodec.sh) and the data-core CoW backend
# (lerobot-scripts/act_coffee_cow.sh), 2000 steps each. Afterwards it prints
# the training-loop wall-clock of every run: the time between the first and
# the last per-step log line, which excludes the one-time episode preload.
# Runs whose output directory already exists are skipped.
#
# Paper: the torchcodec and CoW raw-pixel rows of Table 3.
#
# Requires a CUDA GPU and the data-core LeRobot fork (the CoW launcher
# uses --dataset.use_data_core; the torchcodec runs alone work on stock LeRobot).
#
# Environment:
#   PYTHON       Python interpreter (default: python)
#   COFFEE_ROOT  dataset directory (default: Hugging Face cache snapshot,
#                see scripts/dataset_paths.sh)
#
# Extra arguments are passed through to both launchers, e.g. for a quick
# check: scripts/run_wallclock_seeds.sh --steps=30 --log_freq=10 --save_checkpoint=false
#
# Outputs (under outputs/): wallclock_<backend>_seed<seed>/ (lerobot-train
# output) and wallclock_<backend>_seed<seed>.log.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/dataset_paths.sh
source scripts/dataset_paths.sh
require_coffee
export COFFEE_ROOT

PYTHON="${PYTHON:-python}"
"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'"

mkdir -p outputs
for backend in torchcodec cow; do
  for seed in 42 43 44; do
    out="outputs/wallclock_${backend}_seed${seed}"
    if [[ -d "$out" ]]; then
      echo "skip $out (exists)"
      continue
    fi
    echo "=== $backend seed $seed ==="
    SEED="$seed" OUTPUT_DIR="$out" "lerobot-scripts/act_coffee_${backend}.sh" \
      --wandb.enable=false "$@" \
      2>&1 | tee "${out}.log"
  done
done

"$PYTHON" - <<'EOF'
import glob
import re
from datetime import datetime

TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*step:\S+.*updt_s:")
for log in sorted(glob.glob("outputs/wallclock_*.log")):
    with open(log) as f:
        times = [datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") for ln in f if (m := TS.search(ln))]
    if len(times) > 1:
        print(f"{log}: {(times[-1] - times[0]).total_seconds():.1f} s")
EOF
