"""Build a per-frame visual feature cache for a LeRobot dataset.

Decodes frames through the stock LeRobotDataset (torchcodec by default), runs a
frozen, ImageNet-pretrained torchvision ResNet (FrozenBatchNorm, eval mode)
over every camera image, and writes one shard per episode to
<output-dir>/episodes/episode_NNNNNN.pt plus <output-dir>/manifest.json
(format "data-core-feature-cache-v1"). Each shard holds the features
(the layer4 map, or its global average with --feature-kind pooled), states,
actions and frame indices. scripts/train_act_feature_cache.py trains from it.

Paper: the cache construction described in Appendix C (feature caching).

Requires torch, torchvision and LeRobot (stock LeRobot is enough). If LeRobot
is not installed, point --lerobot-src (or LEROBOT_SRC) at a lerobot/src checkout.

Example (Coffee D0, see scripts/run_feature_cache_coffee.sh):
  python scripts/build_feature_cache.py --repo-id local/mimicgen_coffee_d0 \
    --root "$COFFEE_ROOT" --episodes 0:100 --image-size 84 --feature-kind map \
    --output-dir outputs/feature_cache_coffee_100ep_resnet18_map
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


def add_lerobot_to_path(path: str | None) -> None:
    if path:
        sys.path.insert(0, str(Path(path).expanduser()))


def parse_episodes(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    text = value.strip()
    if text.startswith("["):
        return [int(v) for v in json.loads(text)]
    if ":" in text:
        start_s, end_s = text.split(":", 1)
        return list(range(int(start_s), int(end_s)))
    return [int(v) for v in text.split(",") if v.strip()]


def build_backbone(name: str, weights: str | None) -> tuple[torch.nn.Module, int]:
    weights_arg = None if weights in (None, "", "None", "none") else weights
    model = getattr(torchvision.models, name)(
        weights=weights_arg,
        norm_layer=FrozenBatchNorm2d,
    )
    out_channels = model.fc.in_features
    return IntermediateLayerGetter(model, return_layers={"layer4": "feature_map"}), out_channels


def image_to_float_nchw(x: torch.Tensor) -> torch.Tensor:
    """Normalize common LeRobot image layouts to float NCHW in [0, 1]."""
    if x.ndim == 5:
        # Delta timestamp path: use the most recent observation.
        x = x[:, -1]
    if x.ndim != 4:
        raise ValueError(f"Expected image batch with 4 or 5 dims, got {tuple(x.shape)}")
    if x.shape[1] not in (1, 3):
        x = x.permute(0, 3, 1, 2).contiguous()
    x = x.float()
    if x.max().item() > 2.0:
        x = x / 255.0
    return x


def maybe_resize(x: torch.Tensor, image_size: int | None) -> torch.Tensor:
    if image_size is None:
        return x
    if x.shape[-2:] == (image_size, image_size):
        return x
    return F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def scalar_list(batch: dict[str, Any], key: str, batch_size: int) -> list[int]:
    value = batch.get(key)
    if value is None:
        return [0] * batch_size
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.view(-1).tolist()]
    return [int(v) for v in value]


def append_rows(
    store: dict[int, dict[str, Any]],
    batch: dict[str, Any],
    features: dict[str, torch.Tensor],
) -> list[int]:
    first_feature = next(iter(features.values()))
    batch_size = first_feature.shape[0]
    episode_indices = scalar_list(batch, "episode_index", batch_size)
    frame_indices = scalar_list(batch, "frame_index", batch_size)
    global_indices = scalar_list(batch, "index", batch_size)

    state = batch.get("observation.state")
    action = batch.get("action")

    if isinstance(state, torch.Tensor) and state.ndim > 2:
        state = state[:, -1]
    if isinstance(action, torch.Tensor) and action.ndim > 2:
        action = action[:, -1]

    for row in range(batch_size):
        ep_idx = episode_indices[row]
        ep = store[ep_idx]
        ep["episode_index"] = ep_idx
        ep["frame_index"].append(frame_indices[row])
        ep["index"].append(global_indices[row])
        if isinstance(state, torch.Tensor):
            ep["state"].append(state[row].detach().cpu())
        if isinstance(action, torch.Tensor):
            ep["action"].append(action[row].detach().cpu())
        for cam, feat in features.items():
            ep["features"][cam].append(feat[row].detach().cpu())

    return episode_indices


def finalize_episode(ep: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "episode_index": int(ep["episode_index"]),
        "frame_index": torch.tensor(ep["frame_index"], dtype=torch.long),
        "index": torch.tensor(ep["index"], dtype=torch.long),
        "features": {},
    }
    for cam in camera_keys:
        values = ep["features"].get(cam, [])
        if values:
            out["features"][cam] = torch.stack(values, dim=0).contiguous()
    if ep["state"]:
        out["state"] = torch.stack(ep["state"], dim=0).float().contiguous()
    if ep["action"]:
        out["action"] = torch.stack(ep["action"], dim=0).float().contiguous()
    return out


def write_episode_shard(
    ep_idx: int,
    ep: dict[str, Any],
    camera_keys: list[str],
    shards_dir: Path,
    output_dir: Path,
) -> str:
    shard = finalize_episode(ep, camera_keys)
    shard_path = shards_dir / f"episode_{ep_idx:06d}.pt"
    torch.save(shard, shard_path)
    return str(shard_path.relative_to(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen visual feature cache for LeRobot.")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo ID, e.g. local/mimicgen_coffee_d0.")
    parser.add_argument("--root", default=None, help="Local dataset directory (default: LeRobot's cache).")
    parser.add_argument("--episodes", default=None, help="JSON list, comma list, or start:end range.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-backend", default="torchcodec", help="LeRobot video backend.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224, help="Square resize before encoding; 0 = no resize.")
    parser.add_argument("--feature-kind", choices=["map", "pooled"], default="map")
    parser.add_argument("--vision-backbone", default="resnet18")
    parser.add_argument("--pretrained-backbone-weights", default="ResNet18_Weights.IMAGENET1K_V1")
    parser.add_argument("--no-imagenet-normalize", action="store_true",
                        help="Skip ImageNet mean/std normalization of the input images.")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                        help="Storage dtype of the cached features.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--lerobot-src",
        default=os.environ.get("LEROBOT_SRC"),
        help="Path to lerobot/src when the package is not installed.",
    )
    args = parser.parse_args()

    add_lerobot_to_path(args.lerobot_src)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    image_size = None if args.image_size <= 0 else args.image_size
    episodes = parse_episodes(args.episodes)
    output_dir = Path(args.output_dir).expanduser()
    shards_dir = output_dir / "episodes"
    shards_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        episodes=episodes,
        video_backend=args.video_backend,
    )
    camera_keys = list(dataset.meta.camera_keys)
    if not camera_keys:
        raise ValueError("Dataset has no camera keys.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    backbone, feature_channels = build_backbone(args.vision_backbone, args.pretrained_backbone_weights)
    backbone = backbone.to(device).eval()
    output_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    store: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "episode_index": None,
            "frame_index": [],
            "index": [],
            "state": [],
            "action": [],
            "features": defaultdict(list),
        }
    )

    t0 = time.time()
    encoded_frames = 0
    feature_shape: list[int] | None = None
    shard_paths: list[str] = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            features_by_cam: dict[str, torch.Tensor] = {}
            for cam in camera_keys:
                if cam not in batch:
                    continue
                images = image_to_float_nchw(batch[cam]).to(device, non_blocking=True)
                images = maybe_resize(images, image_size)
                if not args.no_imagenet_normalize:
                    images = imagenet_normalize(images)
                feature_map = backbone(images)["feature_map"]
                if args.feature_kind == "pooled":
                    feat = feature_map.mean(dim=(-2, -1))
                else:
                    feat = feature_map
                feat = feat.to(dtype=output_dtype).cpu()
                features_by_cam[cam] = feat
                if feature_shape is None:
                    feature_shape = list(feat.shape[1:])

            episode_indices = append_rows(store, batch, features_by_cam)
            encoded_frames += next(iter(features_by_cam.values())).shape[0]

            # The LeRobot DataLoader is sequential here, so once a batch has
            # advanced to episode K, any stored episode < K is complete and can
            # be flushed. This keeps Aloha map-feature builds bounded in RAM.
            max_seen_ep = max(episode_indices)
            completed = [ep_idx for ep_idx in store if ep_idx < max_seen_ep]
            for ep_idx in sorted(completed):
                shard_paths.append(
                    write_episode_shard(ep_idx, store.pop(ep_idx), camera_keys, shards_dir, output_dir)
                )

            if (batch_idx + 1) % 20 == 0:
                elapsed = max(time.time() - t0, 1e-6)
                print(
                    f"batch={batch_idx + 1} frames={encoded_frames} "
                    f"rate={encoded_frames / elapsed:.1f} frames/s"
                )

    for ep_idx in sorted(store):
        shard_paths.append(
            write_episode_shard(ep_idx, store[ep_idx], camera_keys, shards_dir, output_dir)
        )

    elapsed = time.time() - t0
    cache_size_bytes = sum((output_dir / p).stat().st_size for p in shard_paths)
    manifest = {
        "format": "data-core-feature-cache-v1",
        "repo_id": args.repo_id,
        "root": args.root,
        "episodes": episodes,
        "video_backend": args.video_backend,
        "camera_keys": camera_keys,
        "num_frames": encoded_frames,
        "num_episodes": len(shard_paths),
        "fps": float(dataset.fps),
        "feature_kind": args.feature_kind,
        "feature_shape": feature_shape,
        "feature_channels": feature_channels,
        "feature_dtype": args.dtype,
        "vision_backbone": args.vision_backbone,
        "pretrained_backbone_weights": args.pretrained_backbone_weights,
        "image_size": image_size,
        "imagenet_normalize": not args.no_imagenet_normalize,
        "shards": shard_paths,
        "cache_size_bytes": cache_size_bytes,
        "build_time_s": elapsed,
        "frames_per_sec": encoded_frames / max(elapsed, 1e-6),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(shard_paths)} episode shards to {output_dir}")
    print(f"Encoded {encoded_frames} frames in {elapsed:.1f}s ({manifest['frames_per_sec']:.1f} frames/s)")


if __name__ == "__main__":
    main()
