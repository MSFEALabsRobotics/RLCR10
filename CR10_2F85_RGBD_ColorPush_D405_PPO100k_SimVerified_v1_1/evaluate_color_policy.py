#!/usr/bin/env python3
"""Evaluate the saved color-push policy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cr10_2f85_lab.color_scene import build_color_task_scene
from cr10_2f85_lab.config import load_config
from cr10_2f85_lab.rl import COLOR_TARGET_ENV_ID, ColorTargetPushEnv
from cr10_2f85_lab.runtime import prepare_locked_scene
from cr10_2f85_lab.source_lock import LockError


DEFAULT_MODEL = ROOT / "artifacts/pack5_v02_color_ppo_100k_seed7.zip"
DEFAULT_NORMALIZER = (
    ROOT / "artifacts/pack5_v02_color_ppo_100k_seed7_vecnormalize.pkl"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reevaluate the inherited compact-feature PPO policy"
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--normalizer", default=str(DEFAULT_NORMALIZER))
    parser.add_argument(
        "--output",
        default="artifacts/d405_policy_reevaluation_seed7_metadata.json",
    )
    parser.add_argument("--dobot", default=None)
    parser.add_argument("--robotiq", default=None)
    parser.add_argument("--menagerie", default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:
        print("Stable-Baselines3 is not installed:", exc)
        print("Install it with: python3 -m pip install -e '.[rl]'")
        return 2

    model_path = Path(args.model).expanduser().resolve()
    normalizer_path = Path(args.normalizer).expanduser().resolve()
    if not model_path.is_file() or not normalizer_path.is_file():
        print("[FAIL] model or VecNormalize file is missing")
        return 1

    try:
        base_scene = prepare_locked_scene(
            ROOT,
            dobot=args.dobot,
            robotiq=args.robotiq,
            menagerie=args.menagerie,
        )
    except LockError as exc:
        print("[FAIL] SOURCE LOCK:", exc)
        return 1

    cfg = load_config()
    scene = build_color_task_scene(base_scene, cfg)
    statistics_env = DummyVecEnv([lambda: ColorTargetPushEnv(scene)])
    normalizer = VecNormalize.load(str(normalizer_path), statistics_env)
    normalizer.training = False
    normalizer.norm_reward = False
    model = PPO.load(str(model_path), device=args.device)
    env = ColorTargetPushEnv(scene)

    layouts = (("red", -1), ("red", 1), ("blue", -1), ("blue", 1))
    episodes = []
    successes = 0
    for index in range(args.episodes):
        target, sign = layouts[index % len(layouts)]
        observation, _ = env.reset(
            seed=args.seed + index,
            options={"target_color": target, "red_lateral_sign": sign},
        )
        actions = []
        terminated = truncated = False
        info = None
        while not (terminated or truncated):
            normalized = normalizer.normalize_obs(observation.copy())
            action, _ = model.predict(normalized, deterministic=True)
            action_array = np.asarray(action, dtype=np.float32).reshape(2)
            actions.append(action_array.tolist())
            observation, _, terminated, truncated, info = env.step(action_array)
        if info is None:
            raise RuntimeError("evaluation episode ended without info")
        success = bool(info["is_success"])
        successes += int(success)
        action_values = np.asarray(actions, dtype=float)
        episodes.append(
            {
                "episode": index + 1,
                "target_color": target,
                "red_lateral_sign": sign,
                "success": success,
                "episode_steps": int(info["episode_step"]),
                "target_progress_m": float(info["target_progress_m"]),
                "wrong_cube_displacement_m": float(
                    info["wrong_cube_displacement_m"]
                ),
                "safety_fault": info["safety_fault"],
                "action_min_mean_max_by_component": [
                    [
                        float(np.min(action_values[:, component])),
                        float(np.mean(action_values[:, component])),
                        float(np.max(action_values[:, component])),
                    ]
                    for component in range(2)
                ],
            }
        )

    env.close()
    normalizer.close()
    p5 = cfg["pack5"]
    report = {
        "environment_id": COLOR_TARGET_ENV_ID,
        "evaluation_purpose": "D405 candidate compatibility reevaluation of inherited PPO100k policy",
        "model": str(model_path),
        "normalizer": str(normalizer_path),
        "camera_model": p5["camera_model"],
        "camera_name": p5["camera_name"],
        "intrinsics_status": p5["intrinsics_status"],
        "camera_matrix_px": [
            [p5["focal_length_px"][0], 0.0, p5["principal_point_px"][0]],
            [0.0, p5["focal_length_px"][1], p5["principal_point_px"][1]],
            [0.0, 0.0, 1.0],
        ],
        "seed": int(args.seed),
        "evaluation_episodes": int(args.episodes),
        "evaluation_successes": int(successes),
        "episodes": episodes,
        "versions": {
            "stable_baselines3": importlib.metadata.version("stable-baselines3"),
            "gymnasium": importlib.metadata.version("gymnasium"),
            "mujoco": importlib.metadata.version("mujoco"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "hardware_motion_commanded": False,
        "claim_boundary": "Simulation compatibility reevaluation only. Nominal D405 intrinsics and the wrist transform remain uncalibrated for physical hardware.",
    }
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation success: {successes}/{args.episodes}")
    print(f"Report: {output}")
    print("Simulation only. No physical motion was commanded.")
    return 0 if successes == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
