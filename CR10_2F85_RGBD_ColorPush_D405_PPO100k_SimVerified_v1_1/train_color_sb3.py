#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cr10_2f85_lab.color_scene import build_color_task_scene
from cr10_2f85_lab.config import load_config
from cr10_2f85_lab.rl import COLOR_TARGET_ENV_ID, ColorTargetPushEnv
from cr10_2f85_lab.runtime import prepare_locked_scene
from cr10_2f85_lab.source_lock import LockError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the compact-feature RGB-D color-target PPO baseline"
    )
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/pack5_v02_color_ppo")
    display = parser.add_mutually_exclusive_group()
    display.add_argument(
        "--headless",
        action="store_true",
        help="train without a MuJoCo viewer (default and fastest)",
    )
    display.add_argument(
        "--viewer",
        action="store_true",
        help="show the live training environment; slower than headless mode",
    )
    parser.add_argument(
        "--viewer-delay-s",
        type=float,
        default=0.02,
        help="wall-clock pause after each visible policy action (default: 0.02)",
    )
    parser.add_argument("--dobot", default=None)
    parser.add_argument("--robotiq", default=None)
    parser.add_argument("--menagerie", default=None)
    args = parser.parse_args()
    if args.timesteps <= 0 or args.eval_episodes <= 0:
        parser.error("--timesteps and --eval-episodes must be positive")
    if args.viewer_delay_s < 0.0:
        parser.error("--viewer-delay-s must be nonnegative")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:
        print("Stable-Baselines3 is not installed:", exc)
        print("Install it with: python3 -m pip install -e '.[rl]'")
        return 2

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
    color_scene = build_color_task_scene(base_scene, load_config())

    checker = ColorTargetPushEnv(color_scene)
    check_env(checker, warn=True, skip_render_check=True)
    checker.close()

    def make_env() -> Monitor:
        return Monitor(ColorTargetPushEnv(color_scene))

    vector_env = DummyVecEnv([make_env])
    normalized_env = VecNormalize(
        vector_env,
        training=True,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    model = PPO(
        "MlpPolicy",
        normalized_env,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    class LiveViewerCallback(BaseCallback):
        """Synchronize a passive MuJoCo viewer after each training action."""

        def __init__(self, raw_env: ColorTargetPushEnv, delay_s: float):
            super().__init__(verbose=0)
            viewer_module = importlib.import_module("mujoco.viewer")
            self.live_viewer = viewer_module.launch_passive(
                raw_env.model,
                raw_env.data,
            )
            self.delay_s = float(delay_s)

        def _on_step(self) -> bool:
            if self.live_viewer is None:
                return True
            if not self.live_viewer.is_running():
                self.live_viewer.close()
                self.live_viewer = None
                print("Live viewer closed; training continues headless.")
                return True
            self.live_viewer.sync()
            if self.delay_s > 0.0:
                time.sleep(self.delay_s)
            return True

        def _on_training_end(self) -> None:
            if self.live_viewer is not None:
                self.live_viewer.close()
                self.live_viewer = None

    callback = None
    if args.viewer:
        monitored_env = vector_env.envs[0]
        raw_training_env = monitored_env.unwrapped
        if not isinstance(raw_training_env, ColorTargetPushEnv):
            raise RuntimeError("could not resolve live ColorTargetPushEnv")
        callback = LiveViewerCallback(raw_training_env, args.viewer_delay_s)
        print(
            "Training display: VIEWER "
            f"(delay={args.viewer_delay_s:.3f} s/action; slower wall time)"
        )
    else:
        print("Training display: HEADLESS (fastest)")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    output_base = (ROOT / args.output).resolve()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_base))
    normalization_path = output_base.with_name(output_base.name + "_vecnormalize.pkl")
    normalized_env.save(str(normalization_path))
    normalized_env.training = False
    normalized_env.norm_reward = False

    evaluation_layouts = (
        ("red", -1),
        ("red", 1),
        ("blue", -1),
        ("blue", 1),
    )
    successes = 0
    episodes = []
    eval_env = ColorTargetPushEnv(color_scene)
    for episode in range(args.eval_episodes):
        target, sign = evaluation_layouts[episode % len(evaluation_layouts)]
        observation, _ = eval_env.reset(
            seed=args.seed + episode,
            options={"target_color": target, "red_lateral_sign": sign},
        )
        final_info = None
        actions: list[list[float]] = []
        terminated = truncated = False
        while not (terminated or truncated):
            normalized_observation = normalized_env.normalize_obs(observation.copy())
            action, _ = model.predict(normalized_observation, deterministic=True)
            action_array = np.asarray(action, dtype=np.float32).reshape(2)
            actions.append(action_array.tolist())
            observation, _, terminated, truncated, final_info = eval_env.step(
                action_array
            )
        assert final_info is not None
        success = bool(final_info["is_success"])
        successes += int(success)
        action_values = np.asarray(actions, dtype=float)
        episodes.append(
            {
                "episode": episode + 1,
                "target_color": target,
                "red_lateral_sign": sign,
                "success": success,
                "episode_steps": int(final_info["episode_step"]),
                "target_progress_m": float(final_info["target_progress_m"]),
                "wrong_cube_displacement_m": float(
                    final_info["wrong_cube_displacement_m"]
                ),
                "safety_fault": final_info["safety_fault"],
                "action_min_mean_max_by_component": [
                    [
                        float(np.min(action_values[:, index])),
                        float(np.mean(action_values[:, index])),
                        float(np.max(action_values[:, index])),
                    ]
                    for index in range(2)
                ],
            }
        )
    eval_env.close()

    metadata = {
        "environment_id": COLOR_TARGET_ENV_ID,
        "algorithm": "PPO",
        "policy": "MlpPolicy over compact RGB-D-derived features",
        "perception": (
            "128x72 RGB HSV masks plus aligned metric-depth back-projection; "
            "raw images are not sent to PPO"
        ),
        "action": (
            "bounded [TCP forward, TCP lateral] increment with 2 mm maximum "
            "Euclidean step and fixed closed gripper"
        ),
        "seed": int(args.seed),
        "training_display_mode": "viewer" if args.viewer else "headless",
        "viewer_delay_s": float(args.viewer_delay_s) if args.viewer else 0.0,
        "training_timesteps": int(args.timesteps),
        "evaluation_episodes": int(args.eval_episodes),
        "evaluation_successes": int(successes),
        "episodes": episodes,
        "versions": {
            "stable_baselines3": importlib.metadata.version("stable-baselines3"),
            "gymnasium": importlib.metadata.version("gymnasium"),
            "mujoco": importlib.metadata.version("mujoco"),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
        },
        "hardware_motion_commanded": False,
        "claim_boundary": (
            "Simulation baseline only. This does not establish policy quality, "
            "physical color/depth robustness, hardware safety, or sim-to-real transfer."
        ),
    }
    metadata_path = output_base.with_name(output_base.name + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    normalized_env.close()

    print(f"Saved model: {output_base}.zip")
    print(f"Saved observation normalization: {normalization_path}")
    print(f"Saved run metadata: {metadata_path}")
    print(f"Evaluation success: {successes}/{args.eval_episodes}")
    print("Simulation only. No physical motion was commanded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
