#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cr10_2f85_lab.camera import CameraIntrinsics
from cr10_2f85_lab.config import load_config
from cr10_2f85_lab.rl import FixedGripPushEnv
from cr10_2f85_lab.runtime import prepare_locked_scene
from cr10_2f85_lab.source_lock import LockError


def _render_rgb_depth(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    renderer.update_scene(data, camera=camera_id)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_id)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_id)
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    depth[segmentation[:, :, 0] < 0] = 0.0
    return rgb, depth


def _depth_visual(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    visual = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return visual
    low = float(np.min(depth[valid]))
    high = float(np.max(depth[valid]))
    if high <= low:
        visual[valid] = 255
        return visual
    normalized = (depth[valid] - low) / (high - low)
    visual[valid] = np.asarray(255.0 * (1.0 - normalized), dtype=np.uint8)
    return visual


def _save_snapshot(
    output: Path,
    label: str,
    rgb: np.ndarray,
    depth: np.ndarray,
) -> dict[str, object]:
    rgb_path = output / f"d405_{label}_rgb.png"
    depth_path = output / f"d405_{label}_depth_m.npy"
    depth_visual_path = output / f"d405_{label}_depth_visual.png"
    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    Image.fromarray(_depth_visual(depth)).save(depth_visual_path)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise RuntimeError(f"{label} depth render contains no valid geometry")
    return {
        "label": label,
        "rgb_png": str(rgb_path),
        "depth_npy_metres": str(depth_path),
        "depth_visual_png": str(depth_visual_path),
        "valid_depth_fraction": float(np.mean(valid)),
        "depth_min_m": float(np.min(depth[valid])),
        "depth_max_m": float(np.max(depth[valid])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save native-resolution Pack 5 D405 RGB-D previews"
    )
    parser.add_argument("--output", default="artifacts/d405_preview")
    parser.add_argument("--dobot", default=None)
    parser.add_argument("--robotiq", default=None)
    parser.add_argument("--menagerie", default=None)
    args = parser.parse_args()

    try:
        scene = prepare_locked_scene(
            ROOT,
            dobot=args.dobot,
            robotiq=args.robotiq,
            menagerie=args.menagerie,
        )
    except LockError as exc:
        print("[FAIL] SOURCE LOCK:", exc)
        return 1

    cfg = load_config()
    p5 = cfg["pack5"]
    intrinsics = CameraIntrinsics.from_config(p5)
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    env = FixedGripPushEnv(scene)
    env.reset(seed=31)
    camera_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_CAMERA, str(p5["camera_name"])
    )
    if camera_id < 0:
        raise RuntimeError("Pack 5 D405 camera is missing")

    try:
        renderer = mujoco.Renderer(
            env.model, height=intrinsics.height, width=intrinsics.width
        )
    except Exception as exc:
        env.close()
        print("[FAIL] MuJoCo could not create an OpenGL render context:", exc)
        print("Run this command from the Ubuntu desktop session. For a configured headless GPU, try: MUJOCO_GL=egl python3 camera_preview.py")
        return 2

    snapshots: list[dict[str, object]] = []
    try:
        rgb, depth = _render_rgb_depth(renderer, env.data, camera_id)
        snapshots.append(_save_snapshot(output, "reset", rgb, depth))

        final_info = None
        for _ in range(env.max_episode_steps):
            _, _, terminated, truncated, final_info = env.step(
                np.asarray([1.0], dtype=np.float32)
            )
            if terminated or truncated:
                break
        if final_info is None or not bool(final_info["is_success"]):
            raise RuntimeError("reference push did not reach the Pack 4 goal")

        rgb, depth = _render_rgb_depth(renderer, env.data, camera_id)
        snapshots.append(_save_snapshot(output, "success", rgb, depth))
    finally:
        renderer.close()
        env.close()

    report = {
        "camera_model": p5["camera_model"],
        "intrinsics_status": p5["intrinsics_status"],
        "stream_convention": p5["stream_convention"],
        "resolution_px": [intrinsics.width, intrinsics.height],
        "camera_matrix_px": intrinsics.matrix().tolist(),
        "depth_unit": "m",
        "invalid_depth_value": 0.0,
        "background_conversion": (
            "pixels with no segmented MuJoCo object are remapped from the "
            "far plane to 0"
        ),
        "camera_mount_status": p5["mount_status"],
        "snapshots": snapshots,
        "hardware_motion_commanded": False,
    }
    report_path = output / "d405_preview_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[PASS] saved provisional wrist D405 RGB-D previews")
    for snapshot in snapshots:
        print(" ", snapshot["label"], snapshot["rgb_png"])
        print(" ", snapshot["label"], snapshot["depth_visual_png"])
        print(" ", snapshot["label"], snapshot["depth_npy_metres"])
    print("Report:", report_path)
    print("Simulation only. No physical motion was commanded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
