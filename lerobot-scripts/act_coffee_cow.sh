#!/usr/bin/env bash
# ACT on MimicGen Coffee D0 with the data-core CoW backend: 100 episodes are
# decoded once in the parent process into an Arrow raw-pixel cache that the
# forked DataLoader workers share copy-on-write. No augmentation. 84x84,
# 2 cameras, batch 64, 2000 steps; ACT settings as in Appendix A of the paper.
#
# Paper: the "coffee 84x84, bs=64 (CoW)" configuration in Table 1, the CoW
# raw-pixel row in Table 3 (seeds 42-44 via scripts/run_wallclock_seeds.sh) and
# the CoW no-aug cell of Table 4.
#
# Requires the data-core LeRobot fork (--dataset.use_data_core and
# --dataset.preload_episodes are not available in stock LeRobot).
#
# Environment:
#   COFFEE_ROOT  dataset directory (default: Hugging Face cache snapshot at the
#                pinned revision; see scripts/dataset_paths.sh and data/README.md)
#   OUTPUT_DIR   output directory (default: ./outputs/coffee_cow_100ep_bs64)
#   SEED         random seed (default: 42)
# Extra arguments are passed through to lerobot-train, e.g. --steps=5 --wandb.enable=false.
set -euo pipefail
# shellcheck source=scripts/dataset_paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/dataset_paths.sh"
require_coffee

lerobot-train \
    --policy.type act \
    --policy.push_to_hub false \
    --dataset.repo_id local/mimicgen_coffee_d0 \
    --dataset.root "$COFFEE_ROOT" \
    --dataset.use_data_core true \
    --dataset.preload_episodes 100 \
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
    --output_dir "${OUTPUT_DIR:-./outputs/coffee_cow_100ep_bs64}" \
    --seed "${SEED:-42}" \
    "$@"
