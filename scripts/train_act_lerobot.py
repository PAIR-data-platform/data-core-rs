"""Train ACT on a stock LeRobotDataset with optional W&B and GPU metric logging.

A self-contained ACT training loop (ACTPolicy + LeRobotDataset) that logs loss,
throughput and periodic GPU statistics, and can run closed-loop MimicGen
rollouts during training. scripts/train_act_data_core.py runs the same loop on
data_core.LazyDataset.

--dataset is a Hub repo ID (downloaded by LeRobot), a local dataset directory,
or any repo ID together with --dataset-root pointing at a local copy.

Requires torch and LeRobot. Rollout evaluation (--eval-freq > 0) also needs
robosuite, mimicgen and lerobot.envs.mimicgen, which is provided by the
data-core LeRobot fork. scripts/run_act_baseline_rollouts.sh shows the
configuration used for the stock ACT baseline in the paper.

Examples:
  # Hub dataset
  python scripts/train_act_lerobot.py \
    --dataset lerobot/aloha_static_fork_pick_up \
    --steps 2000 --batch-size 16 --wandb

  # Local copy of MimicGen Coffee D0 (see data/README.md)
  python scripts/train_act_lerobot.py \
    --dataset local/mimicgen_coffee_d0 --dataset-root "$COFFEE_ROOT" \
    --num-episodes 10 --steps 10 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def resolve_dataset_candidates(dataset: str, dataset_root: str | None) -> list[tuple[str, str | None]]:
    dataset_path = Path(dataset).expanduser()
    candidates: list[tuple[str, str | None]] = []

    if dataset_root:
        candidates.append((dataset, str(Path(dataset_root).expanduser())))

    if dataset_path.exists():
        if (dataset_path / "meta").exists():
            candidates.append((f"local/{dataset_path.name}", str(dataset_path)))
        candidates.append((dataset_path.name, str(dataset_path.parent)))

    candidates.append((dataset, None))
    return candidates


def build_delta_timestamps(base_ds: LeRobotDataset, chunk_size: int, n_obs_steps: int) -> dict[str, list[float]]:
    fps = max(float(base_ds.fps or 20), 1.0)
    dt = 1.0 / fps
    obs_window = [-(n_obs_steps - 1 - i) * dt for i in range(n_obs_steps)]

    image_keys = list(getattr(base_ds.meta, "camera_keys", []) or [])
    if not image_keys:
        image_keys = [k for k in base_ds.features if k.startswith("observation.images.")]

    delta_timestamps: dict[str, list[float]] = {k: obs_window for k in image_keys}
    if "observation.state" in base_ds.features:
        delta_timestamps["observation.state"] = obs_window
    if "action" in base_ds.features:
        delta_timestamps["action"] = [i * dt for i in range(chunk_size)]
    return delta_timestamps


def load_dataset(
    dataset: str, dataset_root: str | None, backend: str, chunk_size: int, n_obs_steps: int,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    last_error = None
    for repo_id, root in resolve_dataset_candidates(dataset, dataset_root):
        try:
            base_ds = LeRobotDataset(repo_id, root=root, video_backend=backend, episodes=episodes)
            delta_timestamps = build_delta_timestamps(
                base_ds, chunk_size=chunk_size, n_obs_steps=n_obs_steps
            )
            return LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps, video_backend=backend, episodes=episodes)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load dataset {dataset!r}. Last error: {last_error}")


def create_eval_env(args):
    import json
    from functools import partial
    import gymnasium as gym
    from lerobot.envs.mimicgen import create_mimicgen_envs

    cam_mapping = json.loads(args.eval_camera_name_mapping)
    env_cls = partial(gym.vector.SyncVectorEnv, autoreset_mode="Disabled")
    gym_kwargs: dict[str, Any] = {"camera_name_mapping": cam_mapping}
    if args.eval_save_video:
        gym_kwargs["render_height"] = args.eval_render_height
        gym_kwargs["render_width"] = args.eval_render_width
    return create_mimicgen_envs(
        task=args.eval_env_task,
        n_envs=args.eval_n_envs,
        camera_name=args.eval_camera_name,
        init_states_path=args.eval_init_states_path,
        env_cls=env_cls,
        episode_length=args.eval_episode_length,
        gym_kwargs=gym_kwargs,
    )


def run_eval(
    policy,
    device,
    eval_envs,
    n_episodes: int,
    episode_length: int,
    step: int = 0,
    save_video: bool = False,
    video_dir: Path | None = None,
    video_fps: int = 20,
):
    """Run n_episodes rollouts; returns eval/success_rate and eval/mean_reward.

    When save_video=True, one .mp4 is written per rollout batch (env 0) to video_dir.
    Note: only correct for n_obs_steps=1.
    """
    import imageio
    from lerobot.envs.utils import preprocess_observation

    all_successes: list[bool] = []
    all_rewards: list[float] = []

    try:
        for task_name, task_envs in eval_envs.items():
            for vec_env in task_envs.values():
                n_envs = vec_env.num_envs
                batches_needed = (n_episodes + n_envs - 1) // n_envs

                for batch_idx in range(batches_needed):
                    policy.reset()
                    obs, _ = vec_env.reset()
                    done = np.zeros(n_envs, dtype=bool)
                    ep_successes = np.zeros(n_envs, dtype=bool)
                    ep_rewards = np.zeros(n_envs, dtype=float)
                    ep_frames: list[np.ndarray] = []

                    for _ in range(episode_length):
                        if np.all(done):
                            break
                        if save_video:
                            frames = vec_env.call("render")
                            ep_frames.append(frames[0])

                        obs_tensors = preprocess_observation(obs)
                        obs_tensors = {k: v.to(device) for k, v in obs_tensors.items()}
                        with torch.no_grad():
                            action = policy.select_action(obs_tensors)
                        obs, reward, terminated, truncated, info = vec_env.step(action.cpu().numpy())

                        is_success = info.get("is_success", np.zeros(n_envs, dtype=bool))
                        for i in range(n_envs):
                            if not done[i]:
                                ep_rewards[i] += float(reward[i])
                                if terminated[i] or truncated[i]:
                                    ep_successes[i] = bool(is_success[i])
                        done = done | terminated | truncated

                    if save_video and ep_frames and video_dir is not None:
                        ep_idx = batch_idx * n_envs
                        video_path = Path(video_dir) / f"step_{step:06d}_{task_name}_ep{ep_idx:03d}.mp4"
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        imageio.mimsave(str(video_path), ep_frames, fps=video_fps)

                    all_successes.extend(ep_successes.tolist())
                    all_rewards.extend(ep_rewards.tolist())
    finally:
        policy.train()

    return {
        "eval/success_rate": float(np.mean(all_successes[:n_episodes])),
        "eval/mean_reward": float(np.mean(all_rewards[:n_episodes])),
    }


def get_last_step(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1] if x.dim() == 3 else x


def image_to_nchw_last_step(x: torch.Tensor) -> torch.Tensor:
    # [B, T, C, H, W] or [B, T, H, W, C] -> [B, C, H, W]
    if x.dim() == 5:
        x = x[:, -1]
    # [B, H, W, C] -> [B, C, H, W]
    if x.dim() == 4 and x.shape[1] not in (1, 3):
        x = x.permute(0, 3, 1, 2).contiguous()
    return x


def infer_policy_features(sample: dict[str, Any]) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    input_features: dict[str, PolicyFeature] = {}
    output_features: dict[str, PolicyFeature] = {}

    for k, v in sample.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith("observation.images.") and not k.endswith("_is_pad"):
            img = image_to_nchw_last_step(v)
            # ACT config convention in this repo uses HWC
            h, w = int(img.shape[-2]), int(img.shape[-1])
            input_features[k] = PolicyFeature(type=FeatureType.VISUAL, shape=(h, w, 3))
        elif k == "observation.state":
            state = get_last_step(v)
            input_features[k] = PolicyFeature(type=FeatureType.STATE, shape=(int(state.shape[-1]),))
        elif k == "action":
            action = get_last_step(v)
            output_features[k] = PolicyFeature(type=FeatureType.ACTION, shape=(int(action.shape[-1]),))

    if "observation.state" not in input_features:
        raise ValueError("Dataset batch is missing 'observation.state'.")
    if "action" not in output_features:
        raise ValueError("Dataset batch is missing 'action'.")
    if not any(k.startswith("observation.images.") for k in input_features):
        raise ValueError("Dataset batch has no observation.images.* keys.")

    return input_features, output_features


def build_policy(input_features: dict[str, PolicyFeature], output_features: dict[str, PolicyFeature], args) -> ACTPolicy:
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        n_obs_steps=args.n_obs_steps,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        dropout=args.dropout,
        optimizer_lr=args.lr,
        optimizer_lr_backbone=args.lr_backbone,
        optimizer_weight_decay=args.weight_decay,
        vision_backbone=args.vision_backbone,
        pretrained_backbone_weights=args.pretrained_backbone_weights,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        feedforward_activation="relu",
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        use_vae=True,
        latent_dim=args.latent_dim,
        n_vae_encoder_layers=args.n_vae_encoder_layers,
        kl_weight=args.kl_weight,
        temporal_ensemble_coeff=None,
        pre_norm=False,
        replace_final_stride_with_dilation=False,
    )
    return ACTPolicy(cfg)


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


def maybe_init_wandb(args):
    if not args.wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
    )


def main():
    p = argparse.ArgumentParser(
        description="Train ACT on a stock LeRobotDataset with optional W&B and GPU metric logging.",
    )
    p.add_argument("--dataset", default="lerobot/aloha_static_fork_pick_up",
                   help="Hub repo ID or local dataset directory (default: %(default)s).")
    p.add_argument("--dataset-root", default=None, help="Local directory holding the dataset given by --dataset.")
    p.add_argument("--backend", default="pyav", choices=["pyav", "torchcodec", "video_reader"],
                   help="LeRobot video backend (default: %(default)s). With torchcodec the DataLoader "
                        "runs in the main process (--num-workers is forced to 0).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-episodes", type=int, default=None,
                   help="Limit to first N episodes for benchmarking. Default: all.")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--n-obs-steps", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=20)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--save-freq", type=int, default=20000)
    p.add_argument("--output-dir", default="outputs/lerobot_act_train")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda")

    # ACT hyperparameters (shared with train_act_data_core.py)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--grad-clip-norm", type=float, default=10.0)
    p.add_argument("--vision-backbone", default="resnet18")
    p.add_argument("--pretrained-backbone-weights", default="ResNet18_Weights.IMAGENET1K_V1")
    p.add_argument("--dim-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--dim-feedforward", type=int, default=3200)
    p.add_argument("--n-encoder-layers", type=int, default=4)
    p.add_argument("--n-decoder-layers", type=int, default=1)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--n-vae-encoder-layers", type=int, default=4)

    # Eval
    p.add_argument("--eval-freq", type=int, default=0,
                   help="Steps between evals; 0 = disabled.")
    p.add_argument("--eval-n-episodes", type=int, default=10)
    p.add_argument("--eval-n-envs", type=int, default=1)
    p.add_argument("--eval-env-task", type=str, default=None,
                   help="MimicGen task name(s), comma-separated. Required when --eval-freq > 0.")
    p.add_argument("--eval-init-states-path", type=str, default=None,
                   help="Path to init_states.pt; each eval reset draws one of these recorded initial states.")
    p.add_argument("--eval-camera-name", type=str, default="agentview,robot0_eye_in_hand")
    p.add_argument("--eval-camera-name-mapping", type=str,
                   default='{"agentview":"image","robot0_eye_in_hand":"wrist_image"}',
                   help="JSON: robosuite camera name -> LeRobot obs key.")
    p.add_argument("--eval-episode-length", type=int, default=400)
    p.add_argument("--eval-save-video", action="store_true",
                   help="Save one MP4 per rollout batch (env 0) to --eval-video-dir.")
    p.add_argument("--eval-video-dir", type=str, default=None,
                   help="Directory for eval videos. Defaults to <output-dir>/eval_videos.")
    p.add_argument("--eval-render-height", type=int, default=256)
    p.add_argument("--eval-render-width", type=int, default=256)

    # W&B
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    p.add_argument("--wandb-project", default="data-core")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.backend == "torchcodec" and args.num_workers > 0:
        print("WARNING: torchcodec with DataLoader workers was unstable with this script; forcing num_workers=0")
        args.num_workers = 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run = maybe_init_wandb(args)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    episodes_filter = list(range(args.num_episodes)) if args.num_episodes else None
    ds = load_dataset(
        args.dataset,
        args.dataset_root,
        args.backend,
        args.chunk_size,
        args.n_obs_steps,
        episodes=episodes_filter,
    )
    print(f"Dataset loaded: episodes={ds.num_episodes}, frames={len(ds)}, fps={ds.fps}")
    print(f"Features: {list(ds.features.keys())}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    probe_batch = next(iter(loader))
    input_features, output_features = infer_policy_features(probe_batch)
    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")

    policy = build_policy(input_features, output_features, args).to(device)
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=policy.config.optimizer_lr,
        weight_decay=policy.config.optimizer_weight_decay,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    eval_envs = None
    if args.eval_freq > 0:
        if not args.eval_env_task:
            raise ValueError("--eval-env-task is required when --eval-freq > 0")
        eval_envs = create_eval_env(args)

    step = 0
    losses: list[float] = []
    running_loss = running_l1 = running_kld = 0.0
    train_start = time.time()
    t0 = train_start
    it = iter(loader)

    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        gpu_batch: dict[str, Any] = {}
        for key, val in batch.items():
            if not isinstance(val, torch.Tensor):
                gpu_batch[key] = val
                continue
            v = val.to(device, non_blocking=True)
            if key.startswith("observation.images.") and not key.endswith("_is_pad"):
                v = image_to_nchw_last_step(v).float() / 255.0
            elif key == "observation.state":
                v = get_last_step(v).float()
            elif key == "action":
                v = v.float()
            gpu_batch[key] = v

        if "action_is_pad" not in gpu_batch and "action" in gpu_batch:
            bsz, horizon = gpu_batch["action"].shape[:2]
            gpu_batch["action_is_pad"] = torch.zeros((bsz, horizon), dtype=torch.bool, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = policy.forward(gpu_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=args.grad_clip_norm)
        optimizer.step()

        loss_item = float(loss.item())
        losses.append(loss_item)
        running_loss += loss_item
        running_l1 += float(loss_dict.get("l1_loss", 0.0))
        running_kld += float(loss_dict.get("kld_loss", 0.0))

        step += 1
        if step % args.log_freq == 0:
            elapsed = time.time() - t0
            steps_per_sec = args.log_freq / max(elapsed, 1e-6)
            avg_loss = running_loss / args.log_freq
            avg_l1 = running_l1 / args.log_freq
            avg_kld = running_kld / args.log_freq

            gpu_metrics = collect_gpu_metrics(device)
            log_payload = {
                "train/loss": avg_loss,
                "train/l1_loss": avg_l1,
                "train/kld_loss": avg_kld,
                "train/steps_per_sec": steps_per_sec,
            }
            log_payload.update(gpu_metrics)

            print(
                f"step {step}/{args.steps} "
                f"loss={avg_loss:.4f} l1={avg_l1:.4f} kld={avg_kld:.4f} "
                f"steps/s={steps_per_sec:.2f}"
            )

            if run:
                run.log(log_payload, step=step)

            running_loss = running_l1 = running_kld = 0.0
            t0 = time.time()

        if step % args.save_freq == 0:
            ckpt_dir = Path(args.output_dir) / "checkpoints" / f"{step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        if eval_envs is not None and step % args.eval_freq == 0:
            video_dir = args.eval_video_dir or str(Path(args.output_dir) / "eval_videos")
            eval_metrics = run_eval(
                policy, device, eval_envs, args.eval_n_episodes, args.eval_episode_length,
                step=step,
                save_video=args.eval_save_video,
                video_dir=Path(video_dir) if args.eval_save_video else None,
            )
            print(
                f"step {step}/{args.steps} eval: "
                f"success={eval_metrics['eval/success_rate']:.3f} "
                f"reward={eval_metrics['eval/mean_reward']:.3f}"
            )
            if run:
                run.log(eval_metrics, step=step)

    if eval_envs is not None:
        from lerobot.envs.utils import close_envs
        close_envs(eval_envs)

    total_elapsed = time.time() - train_start
    summary = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "elapsed_s": round(total_elapsed, 2),
        "samples_per_sec": round((args.steps * args.batch_size) / max(total_elapsed, 1e-6), 2),
    }
    summary.update(collect_gpu_metrics(device))

    out_json = Path(args.output_dir) / "train_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_json}")

    if run:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
