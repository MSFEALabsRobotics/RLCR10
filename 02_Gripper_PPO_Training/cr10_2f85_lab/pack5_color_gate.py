from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

from .rl import (
    COLOR_TARGET_ENV_ID,
    ColorTargetPushEnv,
    scripted_color_action,
)


ROOT = Path(__file__).resolve().parents[1]


def _same_state(
    left: tuple[np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    return all(np.array_equal(a, b) for a, b in zip(left, right))


def _scripted_episode(
    scene: Path,
    *,
    target_color: str,
    red_lateral_sign: int,
    diagnostic_cpu_rgbd: bool,
) -> dict:
    env = ColorTargetPushEnv(scene, diagnostic_cpu_rgbd=diagnostic_cpu_rgbd)
    observation, reset_info = env.reset(
        seed=41,
        options={
            "target_color": target_color,
            "red_lateral_sign": red_lateral_sign,
        },
    )
    total_reward = 0.0
    accepted_actions = 0
    maximum_wrong_motion = 0.0
    final_info: dict | None = None
    for _ in range(env.max_episode_steps):
        action = scripted_color_action(
            observation,
            max_step_m=float(env.ac["max_xy_step_m"]),
        )
        observation, reward, terminated, truncated, info = env.step(action)
        if not env.observation_space.contains(observation):
            raise AssertionError("scripted RGB-D observation leaves declared space")
        if not np.isfinite(reward):
            raise AssertionError("scripted RGB-D reward is non-finite")
        accepted_actions += int(info["action_accepted"])
        total_reward += float(reward)
        maximum_wrong_motion = max(
            maximum_wrong_motion,
            float(info["wrong_cube_displacement_m"]),
        )
        final_info = info
        if terminated or truncated:
            break
    contract = env.contract()
    env.close()

    if final_info is None:
        raise AssertionError("scripted color-target episode did not execute")
    if not bool(final_info["is_success"]):
        raise AssertionError(
            f"scripted {target_color} episode failed: "
            f"progress={final_info['target_progress_m']:.6f} "
            f"fault={final_info['safety_fault']}"
        )
    if final_info["safety_fault"] is not None:
        raise AssertionError("successful color-target episode has a safety fault")
    if maximum_wrong_motion > float(contract["maximum_wrong_cube_displacement_m"]):
        raise AssertionError("scripted task moved the wrong cube beyond its gate")
    if accepted_actions <= 0:
        raise AssertionError("scripted task admitted no planar actions")
    return {
        "target_color": target_color,
        "red_lateral_sign": int(red_lateral_sign),
        "episode_steps": int(final_info["episode_step"]),
        "accepted_actions": int(accepted_actions),
        "target_progress_m": float(final_info["target_progress_m"]),
        "maximum_wrong_cube_displacement_m": float(maximum_wrong_motion),
        "total_reward": float(total_reward),
        "success": True,
        "safety_fault": None,
        "reset_color_mask_pixels": reset_info["reset_diagnostics"]["color_mask_pixels"],
    }


def run(
    scene: Path,
    pack5_result: dict,
    *,
    diagnostic_cpu_rgbd: bool = False,
) -> dict:
    """Verify actual RGB+depth perception and both color-conditioned pushes."""

    checker = ColorTargetPushEnv(scene, diagnostic_cpu_rgbd=diagnostic_cpu_rgbd)
    check_env(checker, skip_render_check=True)
    checker.close()

    registered = gym.make(
        COLOR_TARGET_ENV_ID,
        scene=scene,
        diagnostic_cpu_rgbd=diagnostic_cpu_rgbd,
    )
    registered_observation, registered_info = registered.reset(
        seed=3,
        options={"target_color": "red", "red_lateral_sign": -1},
    )
    if not registered.observation_space.contains(registered_observation):
        raise AssertionError("registered color-target reset is invalid")
    if registered_info["observation_schema_version"] != (
        "cr10_2f85_pack5_rgbd_color_features_v1"
    ):
        raise AssertionError("registered color-task observation schema changed")
    registered.close()

    env = ColorTargetPushEnv(scene, diagnostic_cpu_rgbd=diagnostic_cpu_rgbd)
    options = {"target_color": "red", "red_lateral_sign": -1}
    observation_a, info_a = env.reset(seed=17, options=options)
    rgb_a = env.last_rgb.copy()
    depth_a = env.last_depth_m.copy()
    observation_b, info_b = env.reset(seed=17, options=options)
    if not np.array_equal(observation_a, observation_b):
        raise AssertionError("same-seed RGB-D feature reset is not deterministic")
    if not np.array_equal(rgb_a, env.last_rgb):
        raise AssertionError("same-seed low-resolution RGB reset changed")
    if not np.array_equal(depth_a, env.last_depth_m):
        raise AssertionError("same-seed low-resolution depth reset changed")
    if not env.observation_space.contains(observation_b):
        raise AssertionError("RGB-D feature reset leaves observation space")

    for color in ("red", "blue"):
        detection = info_b["perception"][color]
        if detection is None:
            raise AssertionError(f"{color} cube was not perceived")
        if int(detection["mask_pixels"]) < int(env.ct["minimum_color_mask_pixels"]):
            raise AssertionError(f"{color} color mask is below the pixel gate")
        if not (
            float(env.ct["minimum_valid_depth_m"])
            <= float(detection["median_depth_m"])
            <= float(env.ct["maximum_valid_depth_m"])
        ):
            raise AssertionError(f"{color} depth is outside declared bounds")
        error = float(info_b["reset_diagnostics"]["lateral_perception_error_m"][color])
        if error > float(env.ct["maximum_reset_lateral_perception_error_m"]):
            raise AssertionError(f"{color} lateral RGB-D error exceeds gate")

    state_before = env.state_arrays()
    rejected_observation, rejected_reward, terminated, truncated, rejection = env.step(
        np.asarray([1.01, 0.0], dtype=np.float32)
    )
    state_after = env.state_arrays()
    if not _same_state(state_before, state_after):
        raise AssertionError("out-of-range planar action wrote MuJoCo state")
    if rejection["action_code"] != "ACTION_OUT_OF_RANGE":
        raise AssertionError("unexpected planar rejection code")
    if int(rejection["simulation_substeps"]) != 0:
        raise AssertionError("rejected planar action advanced simulation")
    if terminated or truncated:
        raise AssertionError("one rejected planar action ended the episode")
    if rejected_reward != float(env.ct["invalid_action_penalty"]):
        raise AssertionError("invalid planar action penalty changed")
    if not env.observation_space.contains(rejected_observation):
        raise AssertionError("rejected action returned invalid observation")

    red_observation, _ = env.reset(
        seed=23,
        options={"target_color": "red", "red_lateral_sign": -1},
    )
    red_rgb = env.last_rgb.copy()
    red_depth = env.last_depth_m.copy()
    blue_observation, _ = env.reset(
        seed=23,
        options={"target_color": "blue", "red_lateral_sign": -1},
    )
    if not np.array_equal(red_rgb, env.last_rgb) or not np.array_equal(
        red_depth, env.last_depth_m
    ):
        raise AssertionError("target command changed the physical RGB-D scene")
    if not (
        red_observation[10] == 1.0
        and red_observation[11] == 0.0
        and blue_observation[10] == 0.0
        and blue_observation[11] == 1.0
    ):
        raise AssertionError("requested-color one-hot command is incorrect")
    contract = env.contract()
    env.close()

    episodes = [
        _scripted_episode(
            scene,
            target_color=target,
            red_lateral_sign=sign,
            diagnostic_cpu_rgbd=diagnostic_cpu_rgbd,
        )
        for target, sign in (
            ("red", -1),
            ("red", 1),
            ("blue", -1),
            ("blue", 1),
        )
    ]

    result = {
        "verification_level": (
            "CR10_2F85_RGBD_COLOR_TARGET_PLANAR_PUSH_ENV"
            if not diagnostic_cpu_rgbd
            else "CR10_2F85_COLOR_TARGET_CPU_DIAGNOSTIC_ONLY"
        ),
        "pack5_camera_regression_level": pack5_result["verification_level"],
        "gymnasium": {
            "version": importlib.metadata.version("gymnasium"),
            "environment_id": COLOR_TARGET_ENV_ID,
            "check_env_passed": True,
            "gym_make_passed": True,
        },
        "contract": contract,
        "diagnostic_cpu_rgbd": bool(diagnostic_cpu_rgbd),
        "deterministic_reset": {
            "same_seed_observation_exactly_equal": True,
            "same_seed_rgb_exactly_equal": True,
            "same_seed_depth_exactly_equal": True,
        },
        "rgbd_perception": {
            "resolution_px": info_b["rgbd_resolution_px"],
            "red": info_b["perception"]["red"],
            "blue": info_b["perception"]["blue"],
            "lateral_error_m": info_b["reset_diagnostics"][
                "lateral_perception_error_m"
            ],
            "target_command_changes_scene": False,
        },
        "fail_closed_invalid_action": {
            "rejection_code": rejection["action_code"],
            "simulation_substeps": int(rejection["simulation_substeps"]),
            "wrote_mujoco_state": False,
        },
        "scripted_reference_episodes": episodes,
        "all_target_and_side_combinations_succeeded": True,
        "hardware_motion_commanded": False,
        "claims": {
            "validated": (
                [
                    "actual low-resolution MuJoCo RGB and aligned metric depth feed the observation extractor",
                ]
                if not diagnostic_cpu_rgbd
                else [
                    "CPU ray depth plus geom-ID synthetic colors exercise mechanics and feature plumbing only",
                ]
            )
            + [
                "HSV masks distinguish configured red and blue cubes",
                "metric depth and scaled D405 intrinsics produce wrist-relative visible-surface points",
                "the requested-color one-hot command selects the task without changing the scene",
                "bounded planar actions have a 2 mm maximum Euclidean increment and a fixed gripper command",
                "same-seed RGB, depth, features, and reset layout are deterministic",
                "scripted observation-only references push red and blue from both side assignments",
                "wrong-cube motion, cube/table support, and robot/table contact are gated",
            ],
            "not_validated": [
                "learned-policy performance",
                "raw-image or end-to-end visual RL",
                "physical HSV thresholds or illumination robustness",
                "physical wrist-camera hand-eye calibration",
                "RealSense depth noise, holes, latency, or distortion",
                "physical force control, ROS 2 execution, or sim-to-real transfer",
            ],
        },
    }
    output_name = (
        "CR10_2F85_RGBD_COLOR_TARGET_PLANAR_PUSH_ENV.json"
        if not diagnostic_cpu_rgbd
        else "CR10_2F85_COLOR_TARGET_CPU_DIAGNOSTIC_ONLY.json"
    )
    output = ROOT / "verification" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
