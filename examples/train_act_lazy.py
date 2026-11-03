"""Train an ACT policy on the data_core.LazyDataset PyTorch frontend.

Uses `LazyDataset.to_torch_dataset(mode="chunk")` to read contiguous action
chunks and feeds them to LeRobot's ACTPolicy. Image and state features are read
from the dataset's meta/info.json, and inputs are normalized with the mean/std
in meta/stats.json. The training loop is deliberately minimal (no evaluation);
scripts/train_act_data_core.py is the fuller version used for benchmarking.

The dataset must be a local LeRobot v3 directory. The default is $COFFEE_ROOT
if set, else the Hugging Face cache snapshot of mineral/lerobot_mimicgen_coffee_d0
at the pinned revision ($HF_HUB_CACHE, else $HF_HOME/hub, else
~/.cache/huggingface/hub). See data/README.md for the download.

Requires data_core, torch and LeRobot (for ACTPolicy). W&B logging is optional.

Usage (smoke test):
    python examples/train_act_lazy.py --steps 10 --batch-size 4 --num-workers 0

Usage (longer run):
    python examples/train_act_lazy.py --dataset-path /path/to/dataset \
        --output-dir outputs/act_lazy --batch-size 32 --steps 120000 \
        --num-workers 8 --wandb
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import data_core
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


COFFEE_REVISION = "cbf90909acf57ce67b91b3a38abca40a911af4b2"


def default_dataset_path() -> str:
    """$COFFEE_ROOT, else the Hugging Face cache snapshot of the Coffee D0 dataset."""
    if os.environ.get("COFFEE_ROOT"):
        return os.environ["COFFEE_ROOT"]
    hub_cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub"
    )
    return os.path.join(hub_cache, "datasets--mineral--lerobot_mimicgen_coffee_d0", "snapshots", COFFEE_REVISION)


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

def load_stats(dataset_path: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    """Load mean/std tensors from meta/stats.json."""
    eps = 1e-8
    with open(Path(dataset_path) / "meta" / "stats.json") as f:
        raw = json.load(f)

    result = {}
    for key, s in raw.items():
        mean = torch.tensor(s["mean"], dtype=torch.float32)
        std = torch.tensor(s["std"], dtype=torch.float32).clamp(min=eps)
        result[key] = {"mean": mean, "std": std}
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ACTChunkDataset(Dataset):
    """Wraps LazyDataset chunk mode → ACT training format.

    ACT trains on a single-frame observation + chunk_size future actions.
    We use chunk mode to get contiguous sequences, then take:
      - first frame of each chunk → observation (images, state)
      - all chunk_size frames      → action sequence
    """

    def __init__(self, lazy_ds, chunk_size: int, stats: dict):
        self.inner = lazy_ds.to_torch_dataset(mode="chunk", chunk_size=chunk_size)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.inner[idx]

        batch: dict[str, torch.Tensor] = {}

        # --- observations: first frame only ---
        for cam_name, img_seq in sample["images"].items():
            key = f"observation.images.{cam_name}"
            mean = self.stats[key]["mean"]  # [3, 1, 1]
            std = self.stats[key]["std"]    # [3, 1, 1]
            batch[key] = (img_seq[0] - mean) / std  # [3, H, W]

        state_mean = self.stats["observation.state"]["mean"]
        state_std = self.stats["observation.state"]["std"]
        batch["observation.state"] = (sample["state"][0] - state_mean) / state_std  # [state_dim]

        # --- actions: full chunk ---
        action_mean = self.stats["action"]["mean"]
        action_std = self.stats["action"]["std"]
        batch["action"] = (sample["action"] - action_mean) / action_std  # [chunk_size, action_dim]

        # valid_mask is always all-True in chunk mode (no partial chunks)
        batch["action_is_pad"] = ~sample["valid_mask"]  # [chunk_size] bool

        return batch


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def policy_features(dataset_path: str | Path) -> tuple[dict, dict]:
    """Build ACT input/output features from meta/info.json."""
    with open(Path(dataset_path) / "meta" / "info.json") as f:
        features = json.load(f)["features"]
    inputs = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=tuple(ft["shape"]))
        for key, ft in features.items()
        if key.startswith("observation.images.") and ft["dtype"] in ("video", "image")
    }
    inputs["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=tuple(features["observation.state"]["shape"])
    )
    outputs = {"action": PolicyFeature(type=FeatureType.ACTION, shape=tuple(features["action"]["shape"]))}
    return inputs, outputs


def build_policy(args, input_features: dict, output_features: dict, device: torch.device) -> ACTPolicy:
    config = ACTConfig(
        device=device.type,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": "MEAN_STD",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        },
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        dropout=0.1,
        optimizer_lr=1e-5,
        optimizer_lr_backbone=1e-5,
        optimizer_weight_decay=1e-4,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        feedforward_activation="relu",
        n_encoder_layers=4,
        n_decoder_layers=1,
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=4,
        kl_weight=10.0,
        temporal_ensemble_coeff=None,
        pre_norm=False,
        replace_final_stride_with_dilation=False,
    )
    return ACTPolicy(config)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(policy: ACTPolicy, optimizer, step: int, output_dir: str):
    ckpt_dir = Path(output_dir) / "checkpoints" / f"{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    with open(ckpt_dir / "step.json", "w") as f:
        json.dump({"step": step}, f)
    print(f"  [ckpt] saved to {ckpt_dir}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # WandB
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.job_name,
            config=vars(args),
        )

    # Dataset
    print(f"Opening dataset: {args.dataset_path}")
    lazy = data_core.LazyDataset.open(args.dataset_path)
    print(f"  {lazy.num_episodes} episodes, {len(lazy)} frames")

    stats = load_stats(args.dataset_path)

    dataset = ACTChunkDataset(lazy, chunk_size=args.chunk_size, stats=stats)
    print(f"  {len(dataset)} chunks (chunk_size={args.chunk_size})")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Policy
    input_features, output_features = policy_features(args.dataset_path)
    policy = build_policy(args, input_features, output_features, device).to(device)
    policy.train()
    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy: {total_params / 1e6:.1f}M trainable params")

    # Optimizer: get_optim_params() already sets lr_backbone for the backbone group
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=policy.config.optimizer_lr,
        weight_decay=policy.config.optimizer_weight_decay,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    step = 0
    t0 = time.time()
    running_loss = 0.0
    running_l1 = 0.0
    running_kld = 0.0

    print(f"\nTraining for {args.steps} steps...")
    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            running_l1 += loss_dict.get("l1_loss", 0.0)
            running_kld += loss_dict.get("kld_loss", 0.0)

            if (step + 1) % args.log_freq == 0:
                elapsed = time.time() - t0
                steps_per_sec = args.log_freq / elapsed
                avg_loss = running_loss / args.log_freq
                avg_l1 = running_l1 / args.log_freq
                avg_kld = running_kld / args.log_freq
                eta_h = (args.steps - step - 1) / steps_per_sec / 3600

                print(
                    f"step {step+1:>7d}/{args.steps}"
                    f"  loss={avg_loss:.4f}"
                    f"  l1={avg_l1:.4f}"
                    f"  kld={avg_kld:.4f}"
                    f"  {steps_per_sec:.1f} steps/s"
                    f"  ETA {eta_h:.1f}h"
                )

                if args.wandb:
                    import wandb
                    wandb.log(
                        {
                            "loss": avg_loss,
                            "l1_loss": avg_l1,
                            "kld_loss": avg_kld,
                            "steps_per_sec": steps_per_sec,
                        },
                        step=step + 1,
                    )

                running_loss = running_l1 = running_kld = 0.0
                t0 = time.time()

            if (step + 1) % args.save_freq == 0:
                save_checkpoint(policy, optimizer, step + 1, args.output_dir)

            step += 1

    # Final checkpoint
    save_checkpoint(policy, optimizer, step, args.output_dir)
    print("Training complete.")

    if args.wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    # Underscore spellings (e.g. --dataset_path) are accepted as aliases.
    p = argparse.ArgumentParser(description="Train ACT on the data_core.LazyDataset PyTorch frontend.")
    p.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", default=default_dataset_path(),
                   help="Local LeRobot v3 dataset directory. Default: $COFFEE_ROOT, else the Hugging Face "
                        "cache snapshot of mineral/lerobot_mimicgen_coffee_d0.")
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", default="outputs/act_lazy")
    p.add_argument("--job-name", "--job_name", dest="job_name", default="act_lazy", help="W&B run name.")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=120000)
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=8)
    p.add_argument("--chunk-size", "--chunk_size", dest="chunk_size", type=int, default=20)
    p.add_argument("--log-freq", "--log_freq", dest="log_freq", type=int, default=100)
    p.add_argument("--save-freq", "--save_freq", dest="save_freq", type=int, default=20000)
    p.add_argument("--device", default="cuda", help="Falls back to cpu if CUDA is unavailable.")
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    p.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default="data-core")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
