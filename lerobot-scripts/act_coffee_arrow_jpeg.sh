#!/usr/bin/env bash
# ACT on MimicGen Coffee D0 with the data-core Arrow backend in JPEG mode
# (frames cached as JPEG in Arrow and decoded in parallel in Rust with the
# `image` crate).
# Same configuration as act_coffee_torchcodec.sh (100 episodes, 84x84,
# 2 cameras, batch 64, 2000 steps) for a direct comparison.
#
# Paper: the arrow_uint8_jpeg row of Table 3 (one seed).
#
# Requires the data-core LeRobot fork on its Arrow branch,
# arpit/arrow-jpeg-on-main (--dataset.use_data_core_arrow and
# --dataset.data_core_arrow_raw_pixels are not available in stock LeRobot).
#
# Environment:
#   COFFEE_ROOT  dataset directory (default: Hugging Face cache snapshot at the
#                pinned revision; see scripts/dataset_paths.sh and data/README.md)
#   OUTPUT_DIR   output directory (default: ./outputs/coffee_arrow_jpeg_100ep_bs64)
#   SEED         random seed (default: 42)
# Extra arguments are passed through to lerobot-train, e.g. --steps=5 --wandb.enable=false.
set -euo pipefail
# shellcheck source=scripts/dataset_paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/dataset_paths.sh"
require_coffee

EPISODES_JSON="[$(seq -s ', ' 0 99)]"  # first 100 episodes: [0, 1, ..., 99]
lerobot-train \
    --policy.type act \
    --policy.push_to_hub false \
    --dataset.repo_id local/mimicgen_coffee_d0 \
    --dataset.root "$COFFEE_ROOT" \
    --dataset.use_data_core_arrow true \
    --dataset.data_core_arrow_raw_pixels false \
    --dataset.episodes "$EPISODES_JSON" \
    --steps 2000 \
    --batch_size 64 \
    --num_workers 4 \
    --log_freq 100 \
    --optimizer.grad_clip_norm 10.0 \
    --policy.chunk_size 100 \
    --policy.n_obs_steps 1 \
    --policy.optimizer_lr 1e-5 \
    --policy.optimizer_lr_backbone 1e-5 \
    --policy.optimizer_weight_decay 1e-4 \
    --policy.kl_weight 10.0 \
    --policy.dim_model 512 \
    --policy.n_heads 8 \
    --policy.dim_feedforward 3200 \
    --policy.n_encoder_layers 4 \
    --policy.n_decoder_layers 7 \
    --policy.latent_dim 32 \
    --policy.n_vae_encoder_layers 4 \
    --policy.vision_backbone resnet18 \
    --policy.pretrained_backbone_weights ResNet18_Weights.IMAGENET1K_V1 \
    --output_dir "${OUTPUT_DIR:-./outputs/coffee_arrow_jpeg_100ep_bs64}" \
    --seed "${SEED:-42}" \
    "$@"
