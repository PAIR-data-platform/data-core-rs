# shellcheck shell=bash
# Dataset locations shared by scripts/run_*.sh and lerobot-scripts/act_*.sh.
# Source this file; do not execute it.
#
# Each root defaults to the Hugging Face cache snapshot of the dataset at the
# revision used for the paper. The cache directory is $HF_HUB_CACHE if set,
# else $HF_HOME/hub if HF_HOME is set, else ~/.cache/huggingface/hub.
# Download with (see data/README.md):
#   hf download mineral/lerobot_mimicgen_coffee_d0 --repo-type dataset --revision "$COFFEE_REVISION"
#   hf download lerobot/aloha_static_coffee --repo-type dataset --revision "$ALOHA_REVISION"
#
# Override any of these by exporting it before running a script:
#   COFFEE_ROOT       local copy of mineral/lerobot_mimicgen_coffee_d0
#   ALOHA_ROOT        local copy of lerobot/aloha_static_coffee
#   MIMICGEN_RAW_DIR  directory holding core/coffee_d0.hdf5 and the extracted
#                     init_states_coffee_d0.pt (default: data/mimicgen_raw)

COFFEE_REPO_ID="mineral/lerobot_mimicgen_coffee_d0"
COFFEE_REVISION="cbf90909acf57ce67b91b3a38abca40a911af4b2"
ALOHA_REPO_ID="lerobot/aloha_static_coffee"
ALOHA_REVISION="b144896feb1f37398a862927b22cd3abdf005a6b"

HF_HUB_CACHE_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

COFFEE_ROOT="${COFFEE_ROOT:-$HF_HUB_CACHE_DIR/datasets--mineral--lerobot_mimicgen_coffee_d0/snapshots/$COFFEE_REVISION}"
ALOHA_ROOT="${ALOHA_ROOT:-$HF_HUB_CACHE_DIR/datasets--lerobot--aloha_static_coffee/snapshots/$ALOHA_REVISION}"
MIMICGEN_RAW_DIR="${MIMICGEN_RAW_DIR:-data/mimicgen_raw}"

# require_dataset ROOT REPO_ID REVISION VAR_NAME
# Exit with a download hint if ROOT does not look like a LeRobot dataset.
require_dataset() {
  local root="$1" repo_id="$2" revision="$3" var_name="$4"
  if [[ ! -f "$root/meta/info.json" ]]; then
    {
      echo "error: dataset $repo_id not found at $root"
      echo "  download it with:"
      echo "    hf download $repo_id --repo-type dataset --revision $revision"
      echo "  or point $var_name at a local copy."
    } >&2
    exit 1
  fi
}

require_coffee() { require_dataset "$COFFEE_ROOT" "$COFFEE_REPO_ID" "$COFFEE_REVISION" COFFEE_ROOT; }
require_aloha() { require_dataset "$ALOHA_ROOT" "$ALOHA_REPO_ID" "$ALOHA_REVISION" ALOHA_ROOT; }
