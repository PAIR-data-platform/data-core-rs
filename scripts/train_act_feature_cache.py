"""Train an ACT-style action head (FeatureACT) from cached visual features.

Consumes a cache written by scripts/build_feature_cache.py, so the visual
backbone never runs in the training loop. Logs data_s / updt_s per step, the
train L1 and (with --val-episodes) the held-out L1, and writes summary.json
and model.pt (weights, config and normalization statistics) to --output-dir.
scripts/eval_feature_act_mimicgen.py rolls out the resulting checkpoint.

Paper: the feature-caching results (Section 4.6, Table 6, Appendix C). See
scripts/run_feature_cache_seeds.sh and scripts/run_feature_cache_coffee.sh for
the configurations used. FeatureACT is a plain transformer encoder-decoder
over the cached features and the state; it has no CVAE.

Requires torch only (LeRobot is not imported).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


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


def episode_index_from_path(path: Path) -> int | None:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return None


class FeatureCacheDataset(Dataset):
    """Map-style dataset over per-episode feature-cache shards."""

    def __init__(
        self,
        cache_dir: str | Path,
        chunk_size: int,
        boundary: str = "pad",
        episode_filter: list[int] | None = None,
        load_to_memory: bool = True,
    ):
        self.cache_dir = Path(cache_dir).expanduser()
        self.chunk_size = chunk_size
        self.boundary = boundary
        self.load_to_memory = load_to_memory
        self.episode_filter = set(episode_filter) if episode_filter is not None else None

        with (self.cache_dir / "manifest.json").open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.camera_keys: list[str] = list(self.manifest["camera_keys"])
        self.feature_kind: str = self.manifest["feature_kind"]
        all_shard_paths = [self.cache_dir / p for p in self.manifest["shards"]]
        self.shard_paths = []
        self.episode_indices = []
        for shard_path in all_shard_paths:
            episode_index = episode_index_from_path(shard_path)
            if self.episode_filter is not None and episode_index not in self.episode_filter:
                continue
            self.shard_paths.append(shard_path)
            self.episode_indices.append(episode_index)

        if not self.shard_paths:
            detail = f" for episode filter {sorted(self.episode_filter)}" if self.episode_filter is not None else ""
            raise ValueError(f"Feature cache has no matching episode shards{detail}.")

        self.episodes = [torch.load(p, map_location="cpu") for p in self.shard_paths] if load_to_memory else None

        self.samples: list[tuple[int, int]] = []
        for ep_pos, shard_path in enumerate(self.shard_paths):
            ep = self.episodes[ep_pos] if self.episodes is not None else torch.load(shard_path, map_location="cpu")
            if self.episode_indices[ep_pos] is None:
                self.episode_indices[ep_pos] = int(ep["episode_index"])
            n = int(ep["action"].shape[0])
            if boundary == "trim":
                end = max(0, n - chunk_size + 1)
            elif boundary == "pad":
                end = n
            else:
                raise ValueError(f"Unknown boundary mode: {boundary}")
            self.samples.extend((ep_pos, i) for i in range(end))

        if not self.samples:
            raise ValueError("Feature cache produced zero trainable samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def _episode(self, ep_pos: int) -> dict[str, Any]:
        if self.episodes is not None:
            return self.episodes[ep_pos]
        return torch.load(self.shard_paths[ep_pos], map_location="cpu")

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ep_pos, frame = self.samples[idx]
        ep = self._episode(ep_pos)

        features = torch.stack([ep["features"][cam][frame] for cam in self.camera_keys], dim=0)
        state = ep["state"][frame].float()

        actions = ep["action"]
        n = int(actions.shape[0])
        action_rows = []
        pad = []
        for offset in range(self.chunk_size):
            target = frame + offset
            if target < n:
                action_rows.append(actions[target])
                pad.append(False)
            else:
                action_rows.append(actions[n - 1])
                pad.append(True)

        return {
            "features": features.float(),
            "observation.state": state,
            "action": torch.stack(action_rows, dim=0).float(),
            "action_is_pad": torch.tensor(pad, dtype=torch.bool),
            "episode_index": torch.tensor(int(ep["episode_index"]), dtype=torch.long),
            "frame_index": ep["frame_index"][frame].long(),
            "index": ep["index"][frame].long(),
        }


@dataclass
class FeatureACTConfig:
    feature_kind: str
    feature_shape: tuple[int, ...]
    state_dim: int
    action_dim: int
    n_cameras: int
    chunk_size: int
    dim_model: int
    n_heads: int
    dim_feedforward: int
    n_encoder_layers: int
    n_decoder_layers: int
    dropout: float
    pre_norm: bool


class FeatureACT(nn.Module):
    """Small ACT-style transformer that consumes cached visual features."""

    def __init__(self, cfg: FeatureACTConfig):
        super().__init__()
        self.cfg = cfg

        self.state_proj = nn.Linear(cfg.state_dim, cfg.dim_model)
        self.camera_embed = nn.Embedding(cfg.n_cameras, cfg.dim_model)

        if cfg.feature_kind == "pooled":
            in_dim = cfg.feature_shape[0]
            self.feature_proj = nn.Linear(in_dim, cfg.dim_model)
            self.spatial_embed = None
            feature_tokens_per_camera = 1
        elif cfg.feature_kind == "map":
            channels, height, width = cfg.feature_shape
            self.feature_proj = nn.Conv2d(channels, cfg.dim_model, kernel_size=1)
            self.spatial_embed = nn.Parameter(torch.zeros(height * width, cfg.dim_model))
            nn.init.normal_(self.spatial_embed, std=0.02)
            feature_tokens_per_camera = height * width
        else:
            raise ValueError(f"Unknown feature kind: {cfg.feature_kind}")

        self.state_pos = nn.Parameter(torch.zeros(1, cfg.dim_model))
        nn.init.normal_(self.state_pos, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.dim_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="relu",
            batch_first=False,
            norm_first=cfg.pre_norm,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.dim_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="relu",
            batch_first=False,
            norm_first=cfg.pre_norm,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_decoder_layers)

        self.decoder_query = nn.Embedding(cfg.chunk_size, cfg.dim_model)
        self.action_head = nn.Linear(cfg.dim_model, cfg.action_dim)

        self.feature_tokens_per_camera = feature_tokens_per_camera

    def _feature_tokens(self, features: Tensor) -> Tensor:
        # pooled: [B, cameras, C] -> [cameras, B, D]
        if self.cfg.feature_kind == "pooled":
            tokens = self.feature_proj(features)
            cam = self.camera_embed.weight.view(1, self.cfg.n_cameras, self.cfg.dim_model)
            tokens = tokens + cam
            return tokens.permute(1, 0, 2).contiguous()

        # map: [B, cameras, C, H, W] -> [cameras*H*W, B, D]
        bsz, n_cams, channels, height, width = features.shape
        x = features.reshape(bsz * n_cams, channels, height, width)
        x = self.feature_proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B*cams, H*W, D]
        x = x + self.spatial_embed.view(1, height * width, self.cfg.dim_model)
        x = x.view(bsz, n_cams, height * width, self.cfg.dim_model)
        cam = self.camera_embed.weight.view(1, n_cams, 1, self.cfg.dim_model)
        x = x + cam
        x = x.reshape(bsz, n_cams * height * width, self.cfg.dim_model)
        return x.permute(1, 0, 2).contiguous()

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch["observation.state"]
        features = batch["features"]
        bsz = state.shape[0]

        state_tok = self.state_proj(state).unsqueeze(0) + self.state_pos.view(1, 1, self.cfg.dim_model)
        feature_tokens = self._feature_tokens(features)
        memory_in = torch.cat([state_tok, feature_tokens], dim=0)
        memory = self.encoder(memory_in)

        queries = self.decoder_query.weight.unsqueeze(1).expand(-1, bsz, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded.transpose(0, 1))


def collect_gpu_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}

    idx = device.index if device.index is not None else torch.cuda.current_device()
    metrics = {
        "gpu/memory_allocated_mb": float(torch.cuda.memory_allocated(device) / (1024**2)),
        "gpu/memory_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024**2)),
        "gpu/max_memory_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={idx}",
                "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) == 6:
            metrics.update(
                {
                    "gpu/utilization_pct": float(parts[0]),
                    "gpu/memory_utilization_pct": float(parts[1]),
                    "gpu/temperature_c": float(parts[2]),
                    "gpu/power_w": float(parts[3]),
                    "gpu/memory_used_mb": float(parts[4]),
                    "gpu/memory_total_mb": float(parts[5]),
                }
            )
    except Exception:
        pass
    return metrics


def move_batch_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def masked_l1_loss(pred: Tensor, action: Tensor, action_is_pad: Tensor) -> Tensor:
    l1_unreduced = F.l1_loss(pred, action, reduction="none")
    mask = ~action_is_pad.unsqueeze(-1)
    denom = (mask.sum() * pred.shape[-1]).clamp(min=1)
    return (l1_unreduced * mask).sum() / denom


def _stats_for_tensor(x: Tensor, eps: float) -> dict[str, Any]:
    x = x.float()
    return {
        "mode": "mean_std",
        "mean": x.mean(dim=0).tolist(),
        "std": x.std(dim=0, unbiased=False).clamp_min(eps).tolist(),
        "min": x.amin(dim=0).tolist(),
        "max": x.amax(dim=0).tolist(),
        "eps": eps,
    }


def compute_normalization_stats(
    dataset: FeatureCacheDataset,
    normalize_state: bool,
    normalize_action: bool,
    eps: float,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "enabled": bool(normalize_state or normalize_action),
        "state_enabled": bool(normalize_state),
        "action_enabled": bool(normalize_action),
    }
    if not stats["enabled"]:
        return stats

    states = []
    actions = []
    for ep_pos in range(len(dataset.shard_paths)):
        ep = dataset._episode(ep_pos)
        if normalize_state:
            states.append(ep["state"].float())
        if normalize_action:
            actions.append(ep["action"].float())

    if normalize_state:
        stats["state"] = _stats_for_tensor(torch.cat(states, dim=0), eps)
    if normalize_action:
        stats["action"] = _stats_for_tensor(torch.cat(actions, dim=0), eps)
    return stats


def normalization_to_device(stats: dict[str, Any] | None, device: torch.device) -> dict[str, dict[str, Tensor]]:
    out: dict[str, dict[str, Tensor]] = {}
    if not stats or not stats.get("enabled"):
        return out
    for key in ("state", "action"):
        if key not in stats:
            continue
        out[key] = {
            "mean": torch.tensor(stats[key]["mean"], dtype=torch.float32, device=device),
            "std": torch.tensor(stats[key]["std"], dtype=torch.float32, device=device),
        }
    return out


def normalize_batch(batch: dict[str, Tensor], stats: dict[str, dict[str, Tensor]]) -> dict[str, Tensor]:
    if not stats:
        return batch
    batch = dict(batch)
    if "state" in stats:
        batch["observation.state"] = (batch["observation.state"] - stats["state"]["mean"]) / stats["state"]["std"]
    if "action" in stats:
        batch["action"] = (batch["action"] - stats["action"]["mean"]) / stats["action"]["std"]
    return batch


def unnormalize_action(action: Tensor, stats: dict[str, dict[str, Tensor]]) -> Tensor:
    if "action" not in stats:
        return action
    return action * stats["action"]["std"] + stats["action"]["mean"]


def evaluate(
    model: FeatureACT,
    loader: DataLoader,
    device: torch.device,
    normalization: dict[str, dict[str, Tensor]],
    max_batches: int,
) -> tuple[float, float, int]:
    model.eval()
    total_abs_raw = 0.0
    total_abs_normalized = 0.0
    total_count = 0.0
    batches = 0
    with torch.no_grad():
        for raw_batch in loader:
            raw_batch = move_batch_to_device(raw_batch, device)
            batch = normalize_batch(raw_batch, normalization)
            pred = model(batch)
            mask = ~batch["action_is_pad"].unsqueeze(-1)
            norm_l1 = F.l1_loss(pred, batch["action"], reduction="none")
            raw_pred = unnormalize_action(pred, normalization)
            raw_l1 = F.l1_loss(raw_pred, raw_batch["action"], reduction="none")
            total_abs_normalized += float((norm_l1 * mask).sum().item())
            total_abs_raw += float((raw_l1 * mask).sum().item())
            total_count += float((mask.sum() * pred.shape[-1]).item())
            batches += 1
            if max_batches > 0 and batches >= max_batches:
                break
    model.train()
    denom = max(total_count, 1.0)
    return total_abs_raw / denom, total_abs_normalized / denom, batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ACT-style head from cached visual features.")
    parser.add_argument("--cache-dir", required=True, help="Feature cache written by build_feature_cache.py.")
    parser.add_argument("--episodes", default=None, help="Train episode filter: JSON list, comma list, or start:end range.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--boundary", choices=["pad", "trim"], default="pad",
                        help="pad: sample every frame and pad action chunks past the episode end; "
                             "trim: only sample frames with a full action chunk.")
    parser.add_argument("--load-to-memory", action=argparse.BooleanOptionalAction, default=True,
                        help="Load all episode shards into RAM once (default) instead of on every access.")
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/feature_cache_act")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--val-cache-dir", default=None, help="Validation cache directory. Defaults to --cache-dir.")
    parser.add_argument("--val-episodes", default=None, help="Validation episode filter: JSON list, comma list, or start:end range.")
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-freq", type=int, default=0, help="Evaluate every N steps; 0 means final validation only.")
    parser.add_argument("--val-max-batches", type=int, default=0, help="Limit validation batches; 0 means full validation split.")
    parser.add_argument("--normalize-state", action="store_true", help="Apply train-split mean/std normalization to state.")
    parser.add_argument("--normalize-action", action="store_true", help="Apply train-split mean/std normalization to action targets.")
    parser.add_argument("--normalization-eps", type=float, default=1e-6)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--dim-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=3200)
    parser.add_argument("--n-encoder-layers", type=int, default=4)
    parser.add_argument("--n-decoder-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pre-norm", action="store_true", help="Use pre-LayerNorm transformer layers.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_episodes = parse_episodes(args.episodes)
    val_episodes = parse_episodes(args.val_episodes)

    dataset = FeatureCacheDataset(
        args.cache_dir,
        chunk_size=args.chunk_size,
        boundary=args.boundary,
        episode_filter=train_episodes,
        load_to_memory=args.load_to_memory,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    val_dataset = None
    if args.val_cache_dir is not None or val_episodes is not None:
        val_cache_dir = args.val_cache_dir or args.cache_dir
        val_dataset = FeatureCacheDataset(
            val_cache_dir,
            chunk_size=args.chunk_size,
            boundary=args.boundary,
            episode_filter=val_episodes,
            load_to_memory=args.load_to_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )

    sample = dataset[0]
    cfg = FeatureACTConfig(
        feature_kind=dataset.feature_kind,
        feature_shape=tuple(sample["features"].shape[1:]),
        state_dim=int(sample["observation.state"].shape[-1]),
        action_dim=int(sample["action"].shape[-1]),
        n_cameras=len(dataset.camera_keys),
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
    normalization_stats = compute_normalization_stats(
        dataset,
        normalize_state=args.normalize_state,
        normalize_action=args.normalize_action,
        eps=args.normalization_eps,
    )
    normalization = normalization_to_device(normalization_stats, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Cache: {args.cache_dir}")
    print(f"Train samples: {len(dataset)}; episodes: {dataset.episode_indices}")
    if val_dataset is not None:
        print(f"Val samples: {len(val_dataset)}; episodes: {val_dataset.episode_indices}")
    print(f"Feature kind/shape: {cfg.feature_kind} {cfg.feature_shape}")
    print(f"State dim: {cfg.state_dim}; action dim: {cfg.action_dim}; cameras: {cfg.n_cameras}")
    print(
        "Normalization: "
        f"state={bool(normalization_stats.get('state_enabled'))} "
        f"action={bool(normalization_stats.get('action_enabled'))}"
    )

    iterator = iter(loader)
    train_start = time.time()
    last_log = train_start
    last_log_step = 0
    running_loss = 0.0
    running_l1_normalized = 0.0
    running_l1_raw = 0.0
    last_train_l1 = None
    last_train_l1_normalized = None
    best_train_l1 = None
    best_train_l1_normalized = None
    best_val_l1 = None
    best_val_l1_normalized = None
    last_val_l1 = None
    last_val_l1_normalized = None
    validation_wall_s = 0.0
    validation_runs = 0
    validation_batches_total = 0
    step = 0

    while step < args.steps:
        data_start = time.time()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        data_s = time.time() - data_start

        raw_batch = move_batch_to_device(batch, device)
        batch = normalize_batch(raw_batch, normalization)

        update_start = time.time()
        pred = model(batch)
        loss = masked_l1_loss(pred, batch["action"], batch["action_is_pad"])
        raw_pred = unnormalize_action(pred, normalization)
        raw_l1 = masked_l1_loss(raw_pred, raw_batch["action"], raw_batch["action_is_pad"])

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
        running_l1_normalized += float(loss.item())
        running_l1_raw += float(raw_l1.item())

        should_log = step % args.log_freq == 0 or step == args.steps
        should_val = val_loader is not None and (
            (args.val_freq > 0 and step % args.val_freq == 0) or step == args.steps
        )

        if should_log or should_val:
            now = time.time()
            window = max(1, step - last_log_step) if should_log else max(1, step % args.log_freq)
            elapsed = now - train_start
            steps_s = window / max(now - last_log, 1e-6)
            log = {
                "step": step,
                "loss": running_loss / window,
                "l1_loss": running_l1_raw / window,
                "l1_loss_normalized": running_l1_normalized / window,
                "steps_per_sec": steps_s,
                "samples_per_sec": steps_s * args.batch_size,
                "elapsed_s": elapsed,
                "data_s": data_s,
                "updt_s": update_s,
            }
            log.update(collect_gpu_metrics(device))
            val_text = ""
            if should_val:
                val_start = time.time()
                last_val_l1, last_val_l1_normalized, val_batches = evaluate(
                    model,
                    val_loader,
                    device,
                    normalization,
                    args.val_max_batches,
                )
                val_s = time.time() - val_start
                validation_wall_s += val_s
                validation_runs += 1
                validation_batches_total += val_batches
                best_val_l1 = last_val_l1 if best_val_l1 is None else min(best_val_l1, last_val_l1)
                best_val_l1_normalized = (
                    last_val_l1_normalized
                    if best_val_l1_normalized is None
                    else min(best_val_l1_normalized, last_val_l1_normalized)
                )
                val_text = (
                    f" val_l1={last_val_l1:.4f} val_l1_norm={last_val_l1_normalized:.4f} "
                    f"val_s={val_s:.1f} val_batches={val_batches}"
                )
            if should_log:
                last_train_l1 = log["l1_loss"]
                last_train_l1_normalized = log["l1_loss_normalized"]
                best_train_l1 = last_train_l1 if best_train_l1 is None else min(best_train_l1, last_train_l1)
                best_train_l1_normalized = (
                    last_train_l1_normalized
                    if best_train_l1_normalized is None
                    else min(best_train_l1_normalized, last_train_l1_normalized)
                )
                print(
                    f"step {step:5d} loss={log['loss']:.4f} raw_l1={log['l1_loss']:.4f} "
                    f"steps/s={log['steps_per_sec']:.2f} data_s={data_s:.4f} updt_s={update_s:.4f}"
                    f"{val_text}"
                )
                running_loss = 0.0
                running_l1_normalized = 0.0
                running_l1_raw = 0.0
                last_log = time.time()
                last_log_step = step
            elif val_text:
                print(f"step {step:5d}{val_text}")

    total_s = time.time() - train_start
    summary = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "elapsed_s": total_s,
        "train_wall_s_excluding_validation": total_s - validation_wall_s,
        "samples_per_sec": args.steps * args.batch_size / max(total_s, 1e-6),
        "feature_kind": cfg.feature_kind,
        "feature_shape": list(cfg.feature_shape),
        "num_samples": len(dataset),
        "train_episodes": train_episodes,
        "train_episode_indices": dataset.episode_indices,
        "val_num_samples": len(val_dataset) if val_dataset is not None else None,
        "val_episodes": val_episodes,
        "val_episode_indices": val_dataset.episode_indices if val_dataset is not None else None,
        "last_train_l1": last_train_l1,
        "last_train_l1_normalized": last_train_l1_normalized,
        "best_train_l1": best_train_l1,
        "best_train_l1_normalized": best_train_l1_normalized,
        "last_val_l1": last_val_l1,
        "last_val_l1_normalized": last_val_l1_normalized,
        "best_val_l1": best_val_l1,
        "best_val_l1_normalized": best_val_l1_normalized,
        "validation_wall_s": validation_wall_s,
        "validation_runs": validation_runs,
        "validation_batches": validation_batches_total,
        "normalization": normalization_stats,
        "cache_manifest": dataset.manifest,
    }
    summary.update(collect_gpu_metrics(device))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "args": vars(args),
            "normalization": normalization_stats,
        },
        output_dir / "model.pt",
    )
    print(f"Done in {total_s:.1f}s; wrote {output_dir}")


if __name__ == "__main__":
    main()
