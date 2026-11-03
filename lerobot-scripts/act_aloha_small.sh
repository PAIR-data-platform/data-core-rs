#!/usr/bin/env bash
# Smaller ACT on lerobot/aloha_static_coffee (50 episodes, 480x640, 4 cameras,
# 2000 steps). The transformer is reduced to dim_model=256, dim_feedforward=1024,
# 2 encoder / 4 decoder / 2 VAE-encoder layers; all other settings as in
# Appendix A of the paper. A cheaper model lowers updt_s, so data_s becomes a
# larger fraction of the step time, and larger batches fit in 16 GB of VRAM.
#
# Paper: the "aloha 480x640, bs=16 (smaller ACT)" configuration in Table 1:
# run `act_aloha_small.sh torchcodec 16` (the default batch size is 8). The
# arrow_jpeg backend is not reported in the paper.
#
# Usage:
#   lerobot-scripts/act_aloha_small.sh <backend> [batch_size] [extra lerobot-train args...]
#   backend: torchcodec  runs on stock LeRobot
#            arrow_jpeg  requires the data-core LeRobot fork on its Arrow
#                        branch, arpit/arrow-jpeg-on-main
#                        (--dataset.use_data_core_arrow)
#   batch_size defaults to 8.
#
# Environment:
#   ALOHA_ROOT  dataset directory (default: Hugging Face cache snapshot at the
#               pinned revision; see scripts/dataset_paths.sh and data/README.md)
#   OUTPUT_DIR  output directory (default: ./outputs/aloha_small_<backend>_bs<batch_size>)
#   SEED        random seed (default: 42)
set -euo pipefail
# shellcheck source=scripts/dataset_paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/dataset_paths.sh"
require_aloha

BACKEND="${1:?usage: act_aloha_small.sh <torchcodec|arrow_jpeg> [batch_size] [extra args...]}"
BATCH="${2:-8}"
shift $(( $# < 2 ? $# : 2 ))

case "$BACKEND" in
    torchcodec)
        BACKEND_ARGS=(--dataset.video_backend torchcodec)
        ;;
    arrow_jpeg)
        BACKEND_ARGS=(--dataset.use_data_core_arrow true --dataset.data_core_arrow_raw_pixels false)
        ;;
    *)
        echo "Unknown backend: $BACKEND (expected torchcodec or arrow_jpeg)" >&2
        exit 1
        ;;
esac

lerobot-train \
    --policy.type act \
    --policy.push_to_hub false \
    --dataset.repo_id lerobot/aloha_static_coffee \
    --dataset.root "$ALOHA_ROOT" \
    "${BACKEND_ARGS[@]}" \
    --steps 2000 \
    --batch_size "$BATCH" \
    --num_workers 4 \
    --log_freq 100 \
    --optimizer.grad_clip_norm 10.0 \
    --policy.chunk_size 100 \
    --policy.n_obs_steps 1 \
    --policy.optimizer_lr 1e-5 \
    --policy.optimizer_lr_backbone 1e-5 \
    --policy.optimizer_weight_decay 1e-4 \
    --policy.kl_weight 10.0 \
    --policy.dim_model 256 \
    --policy.n_heads 8 \
    --policy.dim_feedforward 1024 \
    --policy.n_encoder_layers 2 \
    --policy.n_decoder_layers 4 \
    --policy.latent_dim 32 \
    --policy.n_vae_encoder_layers 2 \
    --policy.vision_backbone resnet18 \
    --policy.pretrained_backbone_weights ResNet18_Weights.IMAGENET1K_V1 \
    --output_dir "${OUTPUT_DIR:-./outputs/aloha_small_${BACKEND}_bs${BATCH}}" \
    --seed "${SEED:-42}" \
    "$@"
