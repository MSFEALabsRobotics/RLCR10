#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cr10_2f85_lab.color_scene import build_color_task_scene
from cr10_2f85_lab.config import load_config
from cr10_2f85_lab.rl import ColorTargetPushEnv
from cr10_2f85_lab.runtime import prepare_locked_scene
from cr10_2f85_lab.source_lock import LockError


DEFAULT_MODEL = ROOT / "artifacts/pack5_v02_color_ppo_100k_seed7.zip"
DEFAULT_NORMALIZER = (
    ROOT / "artifacts/pack5_v02_color_ppo_100k_seed7_vecnormalize.pkl"
)


def _depth_image(depth_m: np.ndarray) -> Image.Image:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    if np.any(valid):
        lo = float(np.percentile(depth[valid], 1.0))
        hi = float(np.percentile(depth[valid], 99.0))
        if hi <= lo:
            hi = lo + 1e-6
        normalized[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.where(valid, 255.0 * (1.0 - normalized), 0.0).astype(np.uint8)
    rgb[..., 1] = np.where(
        valid, 255.0 * (1.0 - np.abs(2.0 * normalized - 1.0)), 0.0
    ).astype(np.uint8)
    rgb[..., 2] = np.where(valid, 255.0 * normalized, 0.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _compose_frame(
    overview: np.ndarray,
    wrist_rgb: np.ndarray,
    wrist_depth_m: np.ndarray,
    *,
    target: str,
    step: int,
    info: dict,
    action: np.ndarray,
) -> Image.Image:
    overview_image = Image.fromarray(np.asarray(overview, dtype=np.uint8), mode="RGB")
    wrist = Image.fromarray(np.asarray(wrist_rgb, dtype=np.uint8), mode="RGB")
    depth = _depth_image(wrist_depth_m)
    side_width = 512
    panel_height = 288
    wrist = wrist.resize((side_width, panel_height), Image.Resampling.NEAREST)
    depth = depth.resize((side_width, panel_height), Image.Resampling.NEAREST)
    canvas = Image.new(
        "RGB",
        (overview_image.width + side_width, max(overview_image.height, 576)),
        "black",
    )
    canvas.paste(overview_image, (0, 36))
    canvas.paste(wrist, (overview_image.width, 0))
    canvas.paste(depth, (overview_image.width, panel_height))
    draw = ImageDraw.Draw(canvas)
    status = (
        f"target={target}  step={step}  "
        f"progress={1000.0 * float(info['target_progress_m']):.1f} mm  "
        f"action=[{float(action[0]):+.2f}, {float(action[1]):+.2f}]  "
        f"success={bool(info['is_success'])}"
    )
    draw.rectangle((0, 0, overview_image.width, 36), fill=(18, 18, 18))
    draw.text((12, 10), status, fill=(255, 255, 255))
    draw.text((overview_image.width + 10, 8), "Wrist RGB (policy input)", fill="white")
    draw.text(
        (overview_image.width + 10, panel_height + 8),
        "Aligned metric depth (visualized)",
        fill="white",
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the frozen 100k RGB-D color-target PPO policy"
    )
    parser.add_argument("--target", choices=("red", "blue"), default="red")
    parser.add_argument(
        "--red-lateral-sign",
        type=int,
        choices=(-1, 1),
        default=-1,
        help="-1 puts red on one side; +1 swaps the red/blue sides",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--gif", default=None, help="optional output GIF path")
    parser.add_argument("--frame-ms", type=int, default=100)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--normalizer", default=str(DEFAULT_NORMALIZER))
    parser.add_argument("--dobot", default=None)
    parser.add_argument("--robotiq", default=None)
    parser.add_argument("--menagerie", default=None)
    args = parser.parse_args()
    if args.frame_ms <= 0:
        parser.error("--frame-ms must be positive")

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
        print("[FAIL] frozen model or VecNormalize file is missing")
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
    scene = build_color_task_scene(base_scene, load_config())

    env = ColorTargetPushEnv(scene)
    observation, info = env.reset(
        seed=args.seed,
        options={
            "target_color": args.target,
            "red_lateral_sign": args.red_lateral_sign,
        },
    )
    statistics_env = DummyVecEnv([lambda: ColorTargetPushEnv(scene)])
    normalizer = VecNormalize.load(str(normalizer_path), statistics_env)
    normalizer.training = False
    normalizer.norm_reward = False
    model = PPO.load(str(model_path), device="cpu")

    viewer = None
    if args.viewer:
        mujoco_viewer = importlib.import_module("mujoco.viewer")
        viewer = mujoco_viewer.launch_passive(env.model, env.data)

    renderer = None
    camera = None
    frames: list[Image.Image] = []
    if args.gif:
        renderer = mujoco.Renderer(env.model, height=540, width=960)
        camera = mujoco.MjvCamera()
        target_points = [
            env.data.site_xpos[env.tcp_sid],
            env.data.xpos[env.cubes["red"].body],
            env.data.xpos[env.cubes["blue"].body],
        ]
        camera.lookat[:] = np.mean(np.asarray(target_points), axis=0)
        camera.distance = 1.05
        camera.azimuth = 135.0
        camera.elevation = -28.0

    terminated = truncated = False
    step = 0
    last_action = np.zeros(2, dtype=np.float32)
    while not (terminated or truncated):
        normalized = normalizer.normalize_obs(observation.copy())
        action, _ = model.predict(normalized, deterministic=True)
        last_action = np.asarray(action, dtype=np.float32).reshape(2)
        observation, _, terminated, truncated, info = env.step(last_action)
        step += 1

        if viewer is not None:
            if not viewer.is_running():
                print("Viewer was closed; ending replay.")
                break
            viewer.sync()
            time.sleep(args.frame_ms / 1000.0)

        if renderer is not None and camera is not None:
            renderer.update_scene(env.data, camera=camera)
            overview = renderer.render().copy()
            frames.append(
                _compose_frame(
                    overview,
                    env.last_rgb,
                    env.last_depth_m,
                    target=args.target,
                    step=step,
                    info=info,
                    action=last_action,
                )
            )

    result = {
        "target_color": args.target,
        "red_lateral_sign": args.red_lateral_sign,
        "seed": args.seed,
        "episode_steps": int(info["episode_step"]),
        "success": bool(info["is_success"]),
        "target_progress_m": float(info["target_progress_m"]),
        "wrong_cube_displacement_m": float(info["wrong_cube_displacement_m"]),
        "safety_fault": info["safety_fault"],
        "hardware_motion_commanded": False,
    }
    print(json.dumps(result, indent=2))

    if args.gif and frames:
        gif_path = Path(args.gif).expanduser().resolve()
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=args.frame_ms,
            loop=0,
            optimize=False,
        )
        print(f"Saved replay GIF: {gif_path}")

    if viewer is not None:
        viewer.close()
    if renderer is not None:
        renderer.close()
    normalizer.close()
    env.close()
    print("Simulation only. No physical motion was commanded.")
    return 0 if bool(result["success"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
