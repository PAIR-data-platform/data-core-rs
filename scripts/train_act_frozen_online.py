"""Train the FeatureACT head while recomputing frozen visual features online.

Baseline for the feature cache: the same FeatureACT model as
scripts/train_act_feature_cache.py, but instead of reading cached feature
shards it decodes frames through LeRobotDataset and runs the frozen ResNet on
every step. Comparing the two isolates the effect of caching the features.
Writes summary.json and model.pt to --output-dir.

Paper: the "cache vs frozen-online" rows of Table 6. There is no wrapper
script; run it with the same episodes, feature kind, image size and model
settings as the cached run it is compared against, for example:
  python scripts/train_act_frozen_online.py --repo-id lerobot/aloha_static_coffee \
    --root "$ALOHA_ROOT" --episodes 0:40 --image-size 224 --feature-kind map

Requires torch, torchvision and LeRobot (stock LeRobot is enough).
"""

from __future__ import annotations

import argparse
import json
import random
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from build_feature_cache import (
    add_lerobot_to_path,
    build_backbone,
    image_to_float_nchw,
    imagenet_normalize,
    maybe_resize,
    parse_episodes,
)
from train_act_feature_cache import FeatureACT, FeatureACTConfig, collect_gpu_metrics


def build_delta_timestamps(dataset: Any, chunk_size: int) -> dict[str, list[float]]:
    fps = max(float(dataset.fps or 20), 1.0)
    dt = 1.0 / fps
    return {"action": [i * dt for i in range(chunk_size)]}


def get_last_step(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1] if x.ndim == 3 else x


def compute_features(
    batch: dict[str, Any],
    camera_keys: list[str],
    backbone: torch.nn.Module,
    device: torch.device,
    image_size: int | None,
    feature_kind: str,
    imagenet_norm: bool,
) -> torch.Tensor:
    features = []
    for cam in camera_keys:
        images = image_to_float_nchw(batch[cam]).to(device, non_blocking=True)
        images = maybe_resize(images, image_size)
        if imagenet_norm:
            images = imagenet_normalize(images)
        feature_map = backbone(images)["feature_map"]
        if feature_kind == "pooled":
            feat = feature_map.mean(dim=(-2, -1))
        elif feature_kind == "map":
            feat = feature_map
        else:
            raise ValueError(f"Unknown feature kind: {feature_kind}")
        features.append(feat)
    return torch.stack(features, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FeatureACT with online frozen visual features.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--episodes", default=None, help="JSON list, comma list, or start:end range.")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/frozen_online_act")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--image-size", type=int, default=224, help="Square resize before encoding; 0 = no resize.")
    parser.add_argument("--feature-kind", choices=["map", "pooled"], default="map")
    parser.add_argument("--vision-backbone", default="resnet18")
    parser.add_argument("--pretrained-backbone-weights", default="ResNet18_Weights.IMAGENET1K_V1")
    parser.add_argument("--no-imagenet-normalize", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--dim-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=3200)
    parser.add_argument("--n-encoder-layers", type=int, default=4)
    parser.add_argument("--n-decoder-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pre-norm", action="store_true")
    parser.add_argument("--lerobot-src", default=os.environ.get("LEROBOT_SRC"))
    args = parser.parse_args()

    add_lerobot_to_path(args.lerobot_src)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    episodes = parse_episodes(args.episodes)
    image_size = None if args.image_size <= 0 else args.image_size

    base_dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        episodes=episodes,
        video_backend=args.video_backend,
    )
    delta_timestamps = build_delta_timestamps(base_dataset, args.chunk_size)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )
    camera_keys = list(dataset.meta.camera_keys)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    backbone, _ = build_backbone(args.vision_backbone, args.pretrained_backbone_weights)
    backbone = backbone.to(device).eval()
    for param in backbone.parameters():
        param.requires_grad_(False)

    iterator = iter(loader)
    probe_batch = next(iterator)
    with torch.no_grad():
        probe_features = compute_features(
            probe_batch,
            camera_keys,
            backbone,
            device,
            image_size,
            args.feature_kind,
            not args.no_imagenet_normalize,
        )

    state_probe = get_last_step(probe_batch["observation.state"])
    cfg = FeatureACTConfig(
        feature_kind=args.feature_kind,
        feature_shape=tuple(probe_features.shape[2:]),
        state_dim=int(state_probe.shape[-1]),
        action_dim=int(probe_batch["action"].shape[-1]),
        n_cameras=len(camera_keys),
        chunk_size=args.chunk_size,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        dropout=args.dropout,
        pre_norm=args.pre_norm,
    )
    model = FeatureACT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset: {args.repo_id}")
    print(f"Samples: {len(dataset)}")
    print(f"Cameras: {camera_keys}")
    print(f"Feature kind/shape: {cfg.feature_kind} {cfg.feature_shape}")
    print(f"State dim: {cfg.state_dim}; action dim: {cfg.action_dim}")

    train_start = time.time()
    last_log = train_start
    running_loss = 0.0
    last_train_l1 = None
    best_train_l1 = None
    step = 0

    # Reuse the probe batch as the first batch so no decoded work is wasted.
    pending_batch = probe_batch

    while step < args.steps:
        data_start = time.time()
        if pending_batch is not None:
            batch = pending_batch
            pending_batch = None
        else:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
        data_s = time.time() - data_start

        update_start = time.time()
        with torch.no_grad():
            features = compute_features(
                batch,
                camera_keys,
                backbone,
                device,
                image_size,
                args.feature_kind,
                not args.no_imagenet_normalize,
            )

        state = get_last_step(batch["observation.state"]).to(device, non_blocking=True).float()
        action = batch["action"].to(device, non_blocking=True).float()
        action_is_pad = batch["action_is_pad"].to(device, non_blocking=True).bool()

        model_batch = {
            "features": features.float(),
            "observation.state": state,
            "action": action,
            "action_is_pad": action_is_pad,
        }
        pred = model(model_batch)
        l1_unreduced = F.l1_loss(pred, action, reduction="none")
        mask = ~action_is_pad.unsqueeze(-1)
        denom = (mask.sum() * pred.shape[-1]).clamp(min=1)
        loss = (l1_unreduced * mask).sum() / denom

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_s = time.time() - update_start

        step += 1
        running_loss += float(loss.item())

        if step % args.log_freq == 0 or step == args.steps:
            now = time.time()
            window = args.log_freq if step % args.log_freq == 0 else max(1, step % args.log_freq)
            last_train_l1 = running_loss / window
            best_train_l1 = last_train_l1 if best_train_l1 is None else min(best_train_l1, last_train_l1)
            steps_s = window / max(now - last_log, 1e-6)
            print(
                f"step {step:5d} loss={last_train_l1:.4f} "
                f"steps/s={steps_s:.2f} data_s={data_s:.4f} updt_s={update_s:.4f}"
            )
            running_loss = 0.0
            last_log = now

    total_s = time.time() - train_start
    summary = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "elapsed_s": total_s,
        "train_wall_s_excluding_validation": total_s,
        "samples_per_sec": args.steps * args.batch_size / max(total_s, 1e-6),
        "feature_kind": cfg.feature_kind,
        "feature_shape": list(cfg.feature_shape),
        "num_samples": len(dataset),
        "last_train_l1": last_train_l1,
        "best_train_l1": best_train_l1,
        "repo_id": args.repo_id,
        "root": args.root,
        "episodes": episodes,
        "video_backend": args.video_backend,
    }
    summary.update(collect_gpu_metrics(device))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "args": vars(args)}, output_dir / "model.pt")
    print(f"Done in {total_s:.1f}s; wrote {output_dir}")


if __name__ == "__main__":
    main()
