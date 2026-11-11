#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cr10_2f85_lab.color_scene import build_color_task_scene
from cr10_2f85_lab.config import load_config
from cr10_2f85_lab.rl import ColorTargetPushEnv, scripted_color_action
from cr10_2f85_lab.runtime import prepare_locked_scene
from cr10_2f85_lab.source_lock import LockError


def _depth_visual(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    visual = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return visual
    low = float(np.min(depth[valid]))
    high = float(np.max(depth[valid]))
    if high <= low:
        visual[valid] = 255
    else:
        normalized = (depth[valid] - low) / (high - low)
        visual[valid] = np.asarray(255.0 * (1.0 - normalized), dtype=np.uint8)
    return visual


def _save_frame(
    output: Path,
    label: str,
    rgb: np.ndarray,
    depth: np.ndarray,
) -> dict[str, str]:
    rgb_path = output / f"{label}_rgb_128x72.png"
    depth_path = output / f"{label}_depth_m.npy"
    depth_visual_path = output / f"{label}_depth_visual_128x72.png"
    rgb_large_path = output / f"{label}_rgb_10x.png"
    depth_large_path = output / f"{label}_depth_visual_10x.png"
    rgb_image = Image.fromarray(rgb)
    depth_image = Image.fromarray(_depth_visual(depth))
    rgb_image.save(rgb_path)
    depth_image.save(depth_visual_path)
    rgb_image.resize((1280, 720), Image.Resampling.NEAREST).save(rgb_large_path)
    depth_image.resize((1280, 720), Image.Resampling.NEAREST).save(depth_large_path)
    np.save(depth_path, depth.astype(np.float32))
    return {
        "rgb_policy_png": str(rgb_path),
        "rgb_10x_png": str(rgb_large_path),
        "depth_metres_npy": str(depth_path),
        "depth_visual_policy_png": str(depth_visual_path),
        "depth_visual_10x_png": str(depth_large_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save Pack 5 v0.2 red/blue RGB-D feature-task previews"
    )
    parser.add_argument("--output", default="artifacts/color_task_preview")
    parser.add_argument("--dobot", default=None)
    parser.add_argument("--robotiq", default=None)
    parser.add_argument("--menagerie", default=None)
    args = parser.parse_args()

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
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    snapshots: list[dict] = []
    for target, sign in (("red", -1), ("blue", 1)):
        env = ColorTargetPushEnv(color_scene)
        observation, reset_info = env.reset(
            seed=41,
            options={"target_color": target, "red_lateral_sign": sign},
        )
        reset_files = _save_frame(
            output,
            f"target_{target}_reset",
            env.last_rgb,
            env.last_depth_m,
        )
        final_info = None
        for _ in range(env.max_episode_steps):
            action = scripted_color_action(
                observation,
                max_step_m=float(env.ac["max_xy_step_m"]),
            )
            observation, _, terminated, truncated, final_info = env.step(action)
            if terminated or truncated:
                break
        if final_info is None or not bool(final_info["is_success"]):
            env.close()
            raise RuntimeError(f"{target} preview reference did not succeed")
        success_files = _save_frame(
            output,
            f"target_{target}_success",
            env.last_rgb,
            env.last_depth_m,
        )
        snapshots.append(
            {
                "target_color": target,
                "red_lateral_sign": sign,
                "reset": {
                    **reset_files,
                    "perception": reset_info["perception"],
                },
                "success": {
                    **success_files,
                    "episode_steps": int(final_info["episode_step"]),
                    "target_progress_m": float(final_info["target_progress_m"]),
                    "wrong_cube_displacement_m": float(
                        final_info["wrong_cube_displacement_m"]
                    ),
                    "safety_fault": final_info["safety_fault"],
                },
            }
        )
        env.close()

    report = {
        "task": "requested red-or-blue tabletop push",
        "perception": "HSV color masks plus aligned metric depth back-projection",
        "policy_resolution_px": [128, 72],
        "snapshots": snapshots,
        "hardware_motion_commanded": False,
    }
    report_path = output / "color_task_preview_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[PASS] saved red/blue RGB-D task previews")
    print("Report:", report_path)
    print("Simulation only. No physical motion was commanded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
