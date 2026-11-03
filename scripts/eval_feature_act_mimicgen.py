"""Closed-loop MimicGen rollouts of a FeatureACT checkpoint.

Feature-cache training never sees raw images after the cache is built, but at
rollout time the policy receives live camera observations. This script runs
the same frozen backbone and preprocessing as the cache (read from the cache
manifest) on the simulator images, feeds the features to the FeatureACT head
trained by scripts/train_act_feature_cache.py, and reports task success.
Writes summary.json (success rate and per-rollout results) to --output-dir.

Paper: the cached-policy rollout result in Table 6 and Appendix C, via
scripts/run_feature_cache_rollouts.sh.

Requires torch, torchvision, scipy, robosuite, mimicgen and the MimicGen
environment lerobot.envs.mimicgen, which is provided by the data-core LeRobot
fork. If the fork is not installed, point --lerobot-src (or LEROBOT_SRC) at a
lerobot/src checkout of it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from build_feature_cache import (
    add_lerobot_to_path,
    build_backbone,
    image_to_float_nchw,
    imagenet_normalize,
    maybe_resize,
    parse_episodes,
)
from train_act_feature_cache import FeatureACT, FeatureACTConfig


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_path(path: str | Path | None, base: Path) -> Path | None:
    if path in (None, ""):
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base / p
    return p


def load_manifest(checkpoint: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    cache_dir = resolve_path(args.cache_dir, Path.cwd())
    if cache_dir is None:
        ckpt_args = checkpoint.get("args", {})
        cache_dir = resolve_path(ckpt_args.get("cache_dir"), Path.cwd())
    if cache_dir is None:
        return None
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return load_json(manifest_path)


def parse_camera_keys(value: str | None, manifest: dict[str, Any] | None) -> list[str]:
    if value:
        return [v.strip() for v in value.split(",") if v.strip()]
    if manifest is not None and manifest.get("camera_keys"):
        return list(manifest["camera_keys"])
    return ["observation.images.image", "observation.images.wrist_image"]


def default_camera_mapping(camera_keys: list[str]) -> dict[str, str]:
    names = [key.rsplit(".", 1)[-1] for key in camera_keys]
    if len(names) < 2:
        names = names + ["image2"]
    return {
        "agentview": names[0],
        "robot0_eye_in_hand": names[1],
    }


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[FeatureACT, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = dict(checkpoint["config"])
    cfg_dict["feature_shape"] = tuple(cfg_dict["feature_shape"])
    cfg = FeatureACTConfig(**cfg_dict)
    model = FeatureACT(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg_dict, checkpoint


def normalization_to_device(stats: dict[str, Any] | None, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
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


def normalize_state(state: torch.Tensor, normalization: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    if "state" not in normalization:
        return state
    return (state - normalization["state"]["mean"]) / normalization["state"]["std"]


def unnormalize_action(action: torch.Tensor, normalization: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    if "action" not in normalization:
        return action
    return action * normalization["action"]["std"] + normalization["action"]["mean"]


def obs_to_policy_batch(
    obs: dict[str, Any],
    camera_keys: list[str],
    backbone: torch.nn.Module,
    device: torch.device,
    image_size: int | None,
    feature_kind: str,
    imagenet_norm: bool,
    state_override: np.ndarray | None = None,
    normalization: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    from lerobot.envs.utils import preprocess_observation

    obs_proc = preprocess_observation(obs)
    expected = [*camera_keys] if state_override is not None else [*camera_keys, "observation.state"]
    missing = [key for key in expected if key not in obs_proc]
    if missing:
        raise KeyError(f"Environment observation missing policy keys: {missing}. Got {sorted(obs_proc)}")

    features = []
    with torch.inference_mode():
        for cam in camera_keys:
            images = image_to_float_nchw(obs_proc[cam]).to(device, non_blocking=True)
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

    state = (
        torch.as_tensor(state_override, dtype=torch.float32, device=device).view(1, -1)
        if state_override is not None
        else obs_proc["observation.state"].to(device, non_blocking=True).float()
    )
    state = normalize_state(state, normalization or {})

    return {
        "features": torch.stack(features, dim=1).float(),
        "observation.state": state,
    }


def state_from_raw_obs(raw_obs: dict[str, Any], state_mode: str) -> np.ndarray:
    eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32)
    gripper_qpos = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)

    if state_mode == "mimicgen_eef_quat_gripper1":
        return np.concatenate([eef_pos, eef_quat, gripper_qpos[:1]]).astype(np.float32)
    if state_mode == "mimicgen_eef_axis_gripper2":
        axis_angle = Rotation.from_quat(eef_quat).as_rotvec().astype(np.float32)
        return np.concatenate([eef_pos, axis_angle, gripper_qpos]).astype(np.float32)
    if state_mode == "env_preprocess":
        raise ValueError("env_preprocess should not call state_from_raw_obs.")
    raise ValueError(f"Unknown state mode: {state_mode}")


def set_mimicgen_episode_state(env: Any, episode_idx: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reset the private robosuite env to one recorded MimicGen init state."""
    if env._init_states is None:
        raise ValueError("--episode-indices requires --init-states-path.")
    if episode_idx >= len(env._init_states):
        raise IndexError(f"Episode {episode_idx} requested, but only {len(env._init_states)} init states exist.")

    env._elapsed_steps = 0
    env._env.reset()
    if env._model_files is not None and episode_idx < len(env._model_files) and env._model_files[episode_idx]:
        xml = env._model_files[episode_idx]
        if hasattr(env._env, "edit_model_xml"):
            xml = env._env.edit_model_xml(xml)
        else:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(xml)
        env._env.reset_from_xml_string(xml)
        env._env.sim.reset()

    state_vec = env._init_states[episode_idx].numpy()
    env._env.sim.set_state_from_flattened(state_vec)
    env._env.sim.forward()
    raw_obs = env._env._get_observations()

    for _ in range(env.num_steps_wait):
        raw_obs, _, _, _ = env._env.step([0, 0, 0, 0, 0, 0, -1])

    obs = env._format_raw_obs(raw_obs)
    return obs, {"is_success": False, "init_state_idx": episode_idx}


def run_rollout(
    *,
    env: Any,
    model: FeatureACT,
    backbone: torch.nn.Module,
    camera_keys: list[str],
    device: torch.device,
    image_size: int | None,
    feature_kind: str,
    imagenet_norm: bool,
    max_steps: int,
    seed: int,
    episode_idx: int | None,
    exec_horizon: int,
    clip_actions: bool,
    progress_freq: int,
    state_mode: str,
    normalization: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    if episode_idx is None:
        obs, info = env.reset(seed=seed)
        init_state_idx = info.get("init_state_idx")
    else:
        obs, info = set_mimicgen_episode_state(env, episode_idx)
        init_state_idx = episode_idx

    queued_actions: list[np.ndarray] = []
    total_reward = 0.0
    inference_s = 0.0
    success = False
    final_info: dict[str, Any] = {}
    steps = 0

    with torch.inference_mode():
        for step in range(max_steps):
            if not queued_actions:
                t0 = time.time()
                state_override = None
                if state_mode != "env_preprocess":
                    state_override = state_from_raw_obs(env._env._get_observations(), state_mode)
                batch = obs_to_policy_batch(
                    obs,
                    camera_keys,
                    backbone,
                    device,
                    image_size,
                    feature_kind,
                    imagenet_norm,
                    state_override=state_override,
                    normalization=normalization,
                )
                pred_t = unnormalize_action(model(batch)[0], normalization)
                pred = pred_t.detach().cpu().numpy()
                inference_s += time.time() - t0
                horizon = max(1, min(exec_horizon, pred.shape[0]))
                queued_actions.extend(pred[:horizon])

            action = queued_actions.pop(0).astype(np.float32)
            if clip_actions:
                action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps = step + 1
            success = bool(info.get("is_success", False))
            final_info = dict(info)

            if terminated or truncated or success:
                break
            if progress_freq > 0 and steps % progress_freq == 0:
                print(f"  step {steps:04d}/{max_steps} success={success}", flush=True)

    return {
        "seed": seed,
        "episode_index": episode_idx,
        "init_state_idx": init_state_idx,
        "success": success,
        "steps": steps,
        "total_reward": total_reward,
        "inference_s": inference_s,
        "policy_hz": steps / max(inference_s, 1e-9),
        "final_info": jsonable(final_info),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a FeatureACT checkpoint in MimicGen.")
    parser.add_argument("--checkpoint", required=True, help="model.pt written by train_act_feature_cache.py.")
    parser.add_argument("--cache-dir", default=None,
                        help="Feature cache whose manifest.json describes the backbone and preprocessing. "
                             "Default: the --cache-dir recorded in the checkpoint.")
    parser.add_argument("--task", default="Coffee_D0", help="MimicGen environment name.")
    parser.add_argument("--init-states-path", default=None,
                        help="init_states.pt from extract_mimicgen_init_states.py (needed for --episode-indices).")
    parser.add_argument("--episode-indices", default=None,
                        help="Roll out from these recorded init states, e.g. 80:105 or 80,81. "
                             "When given, --n-rollouts is ignored.")
    parser.add_argument("--n-rollouts", type=int, default=5,
                        help="Number of rollouts from environment resets (without --episode-indices).")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed; rollout i uses seed + i.")
    parser.add_argument("--max-steps", type=int, default=400, help="Rollout horizon in environment steps.")
    parser.add_argument("--exec-horizon", type=int, default=1,
                        help="Number of predicted actions executed before the policy is queried again.")
    parser.add_argument("--progress-freq", type=int, default=0, help="Print progress every N steps; 0 = off.")
    parser.add_argument(
        "--state-mode",
        choices=["mimicgen_eef_quat_gripper1", "mimicgen_eef_axis_gripper2", "env_preprocess"],
        default="mimicgen_eef_quat_gripper1",
        help="How to construct observation.state at rollout time.",
    )
    parser.add_argument("--num-steps-wait", type=int, default=10,
                        help="No-op steps after each reset before the policy acts.")
    parser.add_argument("--clip-actions", action=argparse.BooleanOptionalAction, default=True,
                        help="Clip actions to the environment action space.")
    parser.add_argument("--output-dir", default="outputs/feature_act_mimicgen_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lerobot-src", default=os.environ.get("LEROBOT_SRC"),
                        help="Path to lerobot/src when LeRobot is not installed (default: $LEROBOT_SRC).")
    parser.add_argument("--disable-numba-jit", action=argparse.BooleanOptionalAction, default=True,
                        help="Set NUMBA_DISABLE_JIT=1 before importing robosuite.")

    parser.add_argument("--camera-keys", default=None,
                        help="Comma-separated policy image keys. Default: the cache manifest's camera keys.")
    parser.add_argument("--camera-name", default="agentview,robot0_eye_in_hand", help="robosuite camera names.")
    parser.add_argument("--camera-name-mapping", default=None, help="JSON mapping from env camera to LeRobot image key.")
    parser.add_argument("--observation-height", type=int, default=84)
    parser.add_argument("--observation-width", type=int, default=84)

    parser.add_argument("--image-size", type=int, default=None, help="Override manifest image size; 0 = no resize.")
    parser.add_argument("--feature-kind", choices=["map", "pooled"], default=None, help="Override manifest value.")
    parser.add_argument("--vision-backbone", default=None, help="Override manifest value.")
    parser.add_argument("--pretrained-backbone-weights", default=None, help="Override manifest value.")
    parser.add_argument("--no-imagenet-normalize", action="store_true",
                        help="Skip ImageNet normalization even if the manifest enables it.")
    args = parser.parse_args()

    add_lerobot_to_path(args.lerobot_src)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).expanduser()
    model, cfg, checkpoint = load_model(checkpoint_path, device)
    normalization_stats = checkpoint.get("normalization", {})
    normalization = normalization_to_device(normalization_stats, device)
    manifest = load_manifest(checkpoint, args)

    camera_keys = parse_camera_keys(args.camera_keys, manifest)
    if len(camera_keys) != cfg["n_cameras"]:
        raise ValueError(f"Model expects {cfg['n_cameras']} cameras, but got {len(camera_keys)} camera keys.")

    feature_kind = args.feature_kind or (manifest or {}).get("feature_kind") or cfg["feature_kind"]
    image_size_value = args.image_size
    if image_size_value is None:
        image_size_value = (manifest or {}).get("image_size", args.observation_height)
    image_size = None if image_size_value is None or int(image_size_value) <= 0 else int(image_size_value)
    vision_backbone = args.vision_backbone or (manifest or {}).get("vision_backbone", "resnet18")
    weights = args.pretrained_backbone_weights or (manifest or {}).get(
        "pretrained_backbone_weights", "ResNet18_Weights.IMAGENET1K_V1"
    )
    imagenet_norm = not args.no_imagenet_normalize and bool((manifest or {}).get("imagenet_normalize", True))

    backbone, _ = build_backbone(vision_backbone, weights)
    backbone = backbone.to(device).eval()
    for param in backbone.parameters():
        param.requires_grad_(False)

    mapping = json.loads(args.camera_name_mapping) if args.camera_name_mapping else default_camera_mapping(camera_keys)
    episode_indices = parse_episodes(args.episode_indices)
    if episode_indices is not None:
        rollout_episode_indices: list[int | None] = episode_indices
    else:
        rollout_episode_indices = [None] * args.n_rollouts

    if args.disable_numba_jit:
        # robosuite's numba cache can fail from some site-packages installs.
        # Disable JIT before importing the MimicGen wrapper so eval is portable.
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    from lerobot.envs.mimicgen import MimicGenEnv

    env = MimicGenEnv(
        env_name=args.task,
        init_states_path=args.init_states_path,
        camera_name=args.camera_name,
        camera_name_mapping=mapping,
        observation_height=args.observation_height,
        observation_width=args.observation_width,
        num_steps_wait=args.num_steps_wait,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Task: {args.task}")
    print(f"Device: {device}")
    print(f"Camera keys: {camera_keys}")
    print(f"Camera mapping: {mapping}")
    print(f"Feature: {feature_kind} {cfg['feature_shape']}; image_size={image_size}")
    print(
        "Normalization: "
        f"state={'state' in normalization} action={'action' in normalization}"
    )
    print(f"Rollouts: {len(rollout_episode_indices)}; exec_horizon={args.exec_horizon}")

    t0 = time.time()
    rollouts = []
    try:
        for rollout_idx, episode_idx in enumerate(rollout_episode_indices):
            seed = args.seed + rollout_idx
            result = run_rollout(
                env=env,
                model=model,
                backbone=backbone,
                camera_keys=camera_keys,
                device=device,
                image_size=image_size,
                feature_kind=feature_kind,
                imagenet_norm=imagenet_norm,
                max_steps=args.max_steps,
                seed=seed,
                episode_idx=episode_idx,
                exec_horizon=args.exec_horizon,
                clip_actions=args.clip_actions,
                progress_freq=args.progress_freq,
                state_mode=args.state_mode,
                normalization=normalization,
            )
            rollouts.append(result)
            status = "success" if result["success"] else "fail"
            print(
                f"rollout {rollout_idx:03d} {status} "
                f"steps={result['steps']} init={result['init_state_idx']} "
                f"policy_hz={result['policy_hz']:.1f}"
            )
    finally:
        env.close()

    elapsed_s = time.time() - t0
    success_rate = sum(1 for r in rollouts if r["success"]) / max(len(rollouts), 1)
    summary = {
        "checkpoint": str(checkpoint_path),
        "task": args.task,
        "success_rate": success_rate,
        "num_rollouts": len(rollouts),
        "elapsed_s": elapsed_s,
        "mean_steps": float(np.mean([r["steps"] for r in rollouts])) if rollouts else 0.0,
        "mean_policy_hz": float(np.mean([r["policy_hz"] for r in rollouts])) if rollouts else 0.0,
        "camera_keys": camera_keys,
        "camera_name": args.camera_name,
        "camera_name_mapping": mapping,
        "feature_kind": feature_kind,
        "feature_shape": list(cfg["feature_shape"]),
        "image_size": image_size,
        "vision_backbone": vision_backbone,
        "pretrained_backbone_weights": weights,
        "imagenet_normalize": imagenet_norm,
        "max_steps": args.max_steps,
        "exec_horizon": args.exec_horizon,
        "progress_freq": args.progress_freq,
        "state_mode": args.state_mode,
        "normalization": normalization_stats,
        "clip_actions": args.clip_actions,
        "num_steps_wait": args.num_steps_wait,
        "init_states_path": args.init_states_path,
        "episode_indices": episode_indices,
        "rollouts": rollouts,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, indent=2)

    print(f"Success rate: {success_rate:.3f} ({sum(r['success'] for r in rollouts)}/{len(rollouts)})")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
