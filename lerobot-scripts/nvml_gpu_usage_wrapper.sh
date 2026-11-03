#!/usr/bin/env bash
# Run a training command while a background process logs NVML GPU metrics
# (utilization, memory, temperature, power, clocks) to CSV every 20 ms.
#
# Paper: the NVML GPU-utilization measurements (20 ms sampling, Section 4),
# e.g. the SmolVLA duty cycle in Table 5 and Figures 3 and 4.
#
# Works with stock LeRobot or any other command. Requires the pynvml module
# (pip install nvidia-ml-py). Run with no arguments for usage.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: nvml_gpu_usage_wrapper.sh [command...]

Runs the provided training command while logging high-frequency GPU metrics via NVML.
The CSV log is written to <output_dir>_nvml/, where <output_dir> is taken from the
command's --output_dir or --output-dir argument (default: ./outputs).
If the output directory comes from a config file, also pass it on the command line
so that this script can see it.

Environment variables:
  NVML_INTERVAL_MS  Sampling interval in milliseconds (default: 20)
  NVML_GPU_IDS      Comma-separated GPU indices or UUIDs to log (default: CUDA_VISIBLE_DEVICES or all)
  PYTHON_BIN        Python executable to use (default: python3)
  NVML_LOG_FILE     Optional log filename override. If absolute, it must be inside <output_dir>_nvml.

Examples:
  ./lerobot-scripts/nvml_gpu_usage_wrapper.sh lerobot-train \
      --config_path lerobot-scripts/smolvla_libero_10.yaml --output_dir ./outputs/smolvla_libero_10
  NVML_INTERVAL_MS=50 ./lerobot-scripts/nvml_gpu_usage_wrapper.sh \
      python scripts/train_act_data_core.py --output-dir ./outputs/data_core_act --preload-episodes 100
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

output_dir=""
expect_output_dir="false"

for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then
    expect_output_dir="false"
    break
  fi

  if [[ "$expect_output_dir" == "true" ]]; then
    if [[ "$arg" != -* ]]; then
      output_dir="$arg"
    fi
    expect_output_dir="false"
    continue
  fi

  case "$arg" in
    --output_dir|--output-dir)
      expect_output_dir="true"
      ;;
    --output_dir=*|--output-dir=*)
      output_dir="${arg#*=}"
      ;;
  esac
done

if [[ -z "$output_dir" ]]; then
  output_dir="./outputs"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="${output_dir}_nvml"
mkdir -p "$log_dir"

if [[ -n "${NVML_LOG_FILE:-}" ]]; then
  if [[ "$NVML_LOG_FILE" = /* ]]; then
    case "$NVML_LOG_FILE" in
      "${log_dir%/}"/*) log_file="$NVML_LOG_FILE" ;;
      *)
        echo "NVML_LOG_FILE must be inside ${log_dir}." >&2
        exit 2
        ;;
    esac
  else
    log_file="${log_dir}/${NVML_LOG_FILE}"
  fi
else
  log_file="${log_dir}/nvml_${timestamp}.csv"
fi

export NVML_LOG_FILE="$log_file"
export NVML_INTERVAL_MS="${NVML_INTERVAL_MS:-20}"
export NVML_GPU_IDS="${NVML_GPU_IDS:-}"
python_bin="${PYTHON_BIN:-python3}"

if ! "$python_bin" -c "import pynvml" 2>/dev/null; then
  echo "error: $python_bin cannot import pynvml; install it with 'pip install nvidia-ml-py' or set PYTHON_BIN." >&2
  exit 2
fi

# shellcheck disable=SC2317  # invoked through the trap below
cleanup() {
  if [[ -n "${logger_pid:-}" ]] && kill -0 "$logger_pid" 2>/dev/null; then
    kill "$logger_pid"
    wait "$logger_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$python_bin" - <<'PY' &
import os
import time
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetHandleByUUID,
    nvmlDeviceGetIndex,
    nvmlDeviceGetUUID,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetClockInfo,
    NVML_TEMPERATURE_GPU,
    NVML_CLOCK_SM,
    NVML_CLOCK_MEM,
)


def parse_gpu_list(value: str):
    tokens = []
    for item in value.split(","):
        token = item.strip()
        if token:
            tokens.append(token)
    return tokens


log_file = os.environ["NVML_LOG_FILE"]
interval_ms = int(os.environ.get("NVML_INTERVAL_MS", "20"))
gpu_ids = os.environ.get("NVML_GPU_IDS", "").strip()
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

nvmlInit()
try:
    if gpu_ids:
        requested = parse_gpu_list(gpu_ids)
    elif cuda_visible:
        requested = parse_gpu_list(cuda_visible)
    else:
        requested = [str(i) for i in range(nvmlDeviceGetCount())]

    handles = []
    for token in requested:
        if token.isdigit():
            handle = nvmlDeviceGetHandleByIndex(int(token))
            gpu_index = int(token)
            gpu_uuid = nvmlDeviceGetUUID(handle)
        else:
            handle = nvmlDeviceGetHandleByUUID(token)
            gpu_index = nvmlDeviceGetIndex(handle)
            gpu_uuid = token
        handles.append((gpu_index, gpu_uuid, handle))

    with open(log_file, "w", buffering=1) as f:
        f.write(
            "timestamp,gpu_index,gpu_uuid,util_gpu,util_mem,mem_used,mem_total,temp,power_w,sm_clock,mem_clock\n"
        )
        while True:
            ts = time.time()
            for gpu_index, gpu_uuid, handle in handles:
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                power_w = nvmlDeviceGetPowerUsage(handle) / 1000.0
                sm_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
                mem_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)
                f.write(
                    f"{ts},{gpu_index},{gpu_uuid},{util.gpu},{util.memory},"
                    f"{mem.used},{mem.total},{temp},{power_w},{sm_clock},{mem_clock}\n"
                )
            time.sleep(interval_ms / 1000.0)
finally:
    nvmlShutdown()
PY
logger_pid=$!

set +e
"$@"
status=$?
set -e
exit "$status"
