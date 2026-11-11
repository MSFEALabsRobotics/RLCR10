from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import warnings

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

from .rl import (
    ENV_ID,
    TRAINING_ENV_ID,
    FixedGripPushEnv,
    TabletopPushEnv,
)


ROOT = Path(__file__).resolve().parents[1]


def _same_state(
    before: tuple[np.ndarray, np.ndarray, np.ndarray],
    after: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    return all(np.array_equal(left, right) for left, right in zip(before, after))


def run(scene: Path, pack3_result: dict) -> dict:
    """Verify the Gymnasium contract without training or changing Pack 1--3."""

    checker_env = TabletopPushEnv(scene)
    training_checker_env = FixedGripPushEnv(scene)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*Box observation space minimum value is -infinity.*"
        )
        warnings.filterwarnings(
            "ignore", message=".*Box observation space maximum value is infinity.*"
        )
        check_env(checker_env, skip_render_check=True)
        check_env(training_checker_env, skip_render_check=True)
    checker_env.close()
    training_checker_env.close()

    registered_env = gym.make(ENV_ID, scene=scene)
    registered_observation, registered_info = registered_env.reset(seed=5)
    if not registered_env.observation_space.contains(registered_observation):
        raise AssertionError("registered Pack 4 environment reset is invalid")
    if registered_info["observation_schema_version"] != (
        "cr10_2f85_pack4_state_observation_v1"
    ):
        raise AssertionError("registered Pack 4 observation schema changed")
    registered_env.close()

    registered_training_env = gym.make(TRAINING_ENV_ID, scene=scene)
    training_observation, training_info = registered_training_env.reset(seed=5)
    if not registered_training_env.observation_space.contains(
        training_observation
    ):
        raise AssertionError("registered fixed-grip training reset is invalid")
    if registered_training_env.action_space.shape != (1,):
        raise AssertionError("fixed-grip training action shape must be (1,)")
    if training_info["fixed_gripper_normalized"] != 1.0:
        raise AssertionError("fixed-grip training reset is not closed")
    registered_training_env.close()

    env = TabletopPushEnv(scene)
    observation_a, reset_info_a = env.reset(seed=7)
    observation_b, reset_info_b = env.reset(seed=7)
    if not np.array_equal(observation_a, observation_b):
        raise AssertionError("Pack 4 deterministic reset observation changed")
    if not env.observation_space.contains(observation_b):
        raise AssertionError("Pack 4 reset observation leaves declared space")
    if reset_info_a["is_success"] or reset_info_b["is_success"]:
        raise AssertionError("Pack 4 task starts already solved")

    state_before = env.state_arrays()
    rejected_obs, rejected_reward, terminated, truncated, rejected_info = env.step(
        np.asarray([-1.0, 1.0], dtype=np.float32)
    )
    state_after = env.state_arrays()
    if not _same_state(state_before, state_after):
        raise AssertionError("rejected Pack 4 boundary action wrote MuJoCo state")
    if rejected_info["action_accepted"]:
        raise AssertionError("Pack 4 segment-underflow action was accepted")
    if rejected_info["action_code"] != "SEGMENT_BOUND":
        raise AssertionError(
            "unexpected Pack 4 rejection code: "
            + str(rejected_info["action_code"])
        )
    if int(rejected_info["simulation_substeps"]) != 0:
        raise AssertionError("rejected Pack 4 action advanced simulation")
    if terminated or truncated:
        raise AssertionError("one rejected action unexpectedly ended the episode")
    if rejected_reward != float(env.p4["invalid_action_penalty"]):
        raise AssertionError("Pack 4 invalid-action reward contract changed")
    if not env.observation_space.contains(rejected_obs):
        raise AssertionError("Pack 4 rejected-action observation is invalid")

    shaping_observation, _ = env.reset(seed=9)
    shaping_observation, shaping_reward, terminated, truncated, shaping_info = (
        env.step(np.asarray([1.0, 1.0], dtype=np.float32))
    )
    if terminated or truncated or not shaping_info["action_accepted"]:
        raise AssertionError("Pack 4 first approach action was not admissible")
    if not env.observation_space.contains(shaping_observation):
        raise AssertionError("Pack 4 approach observation is invalid")
    if float(shaping_info["reward_terms"]["tcp_progress"]) <= 0.0:
        raise AssertionError("Pack 4 approach action has no positive shaping signal")
    if not np.isfinite(shaping_reward):
        raise AssertionError("Pack 4 approach reward is non-finite")

    observation, _ = env.reset(seed=11)
    if not env.observation_space.contains(observation):
        raise AssertionError("Pack 4 scripted reset observation is invalid")
    total_reward = 0.0
    total_substeps = 0
    peak_gripper_cube_force = 0.0
    peak_cube_table_force = 0.0
    final_info: dict | None = None
    accepted_actions = 0

    for _ in range(env.max_episode_steps):
        observation, reward, terminated, truncated, info = env.step(
            np.asarray([1.0, 1.0], dtype=np.float32)
        )
        if not env.observation_space.contains(observation):
            raise AssertionError("Pack 4 rollout observation is invalid")
        if not np.isfinite(reward):
            raise AssertionError("Pack 4 rollout produced non-finite reward")
        if info["action_accepted"]:
            accepted_actions += 1
        total_reward += float(reward)
        total_substeps += int(info["simulation_substeps"])
        peak_gripper_cube_force = max(
            peak_gripper_cube_force,
            float(info.get("gripper_cube_peak_force_norm_N", 0.0)),
        )
        peak_cube_table_force = max(
            peak_cube_table_force,
            float(info.get("cube_table_peak_force_norm_N", 0.0)),
        )
        final_info = info
        if terminated or truncated:
            break

    if final_info is None:
        raise AssertionError("Pack 4 scripted rollout did not execute")
    if not bool(final_info["is_success"]):
        raise AssertionError(
            "Pack 4 maximum-forward reference policy did not reach the task goal: "
            f"progress={final_info['cube_progress_m']:.6f} m "
            f"fault={final_info['safety_fault']}"
        )
    if final_info["safety_fault"] is not None:
        raise AssertionError("Pack 4 reference policy ended with a safety fault")
    if not bool(final_info["cube_table_contact"]):
        raise AssertionError("Pack 4 successful cube lost table support")
    if final_info["robot_table_contact_geoms"]:
        raise AssertionError("Pack 4 reference policy contacted the table")
    if accepted_actions <= 0 or total_substeps <= 0:
        raise AssertionError("Pack 4 reference policy did not advance simulation")

    base_contract = env.contract()
    env.close()

    training_env = FixedGripPushEnv(scene)
    training_env.reset(seed=13)
    if training_env.action_space.shape != (1,):
        raise AssertionError("fixed-grip policy action shape changed")

    training_state_before = training_env.state_arrays()
    _, _, training_terminated, training_truncated, training_rejection = (
        training_env.step(np.asarray([1.01], dtype=np.float32))
    )
    training_state_after = training_env.state_arrays()
    if not _same_state(training_state_before, training_state_after):
        raise AssertionError("forward-only invalid action wrote MuJoCo state")
    if training_rejection["action_code"] != "POLICY_ACTION_OUT_OF_RANGE":
        raise AssertionError("forward-only out-of-range action was not rejected")
    if int(training_rejection["simulation_substeps"]) != 0:
        raise AssertionError("forward-only invalid action advanced simulation")
    if training_terminated or training_truncated:
        raise AssertionError("one forward-only rejection ended the episode")

    training_env.reset(seed=15)
    _, _, training_terminated, training_truncated, minimum_mapping = (
        training_env.step(np.asarray([-1.0], dtype=np.float32))
    )
    if not minimum_mapping["action_accepted"]:
        raise AssertionError("zero-forward policy command was not admitted")
    if minimum_mapping.get("normalized_action") != [0.0, 1.0]:
        raise AssertionError("policy -1 did not map exactly to Pack 3 [0,+1]")
    if minimum_mapping.get("mapped_pack3_push_normalized") != 0.0:
        raise AssertionError("minimum policy action did not map to zero push")
    if float(minimum_mapping["commanded_segment_m"]) != 0.0:
        raise AssertionError("zero-forward command changed the segment target")
    if training_terminated or training_truncated:
        raise AssertionError("one zero-forward action ended the episode")

    training_env.reset(seed=16)
    neutral_total_reward = 0.0
    neutral_cube_reward = 0.0
    neutral_contact_actions = 0
    neutral_final_info: dict | None = None
    for _ in range(training_env.max_episode_steps):
        _, reward, terminated, truncated, info = training_env.step(
            np.asarray([0.0], dtype=np.float32)
        )
        neutral_total_reward += float(reward)
        neutral_cube_reward += float(info["reward_terms"]["progress"])
        neutral_contact_actions += int(info["gripper_cube_contact"])
        neutral_final_info = info
        if info.get("normalized_action") != [0.5, 1.0]:
            raise AssertionError("policy 0 did not map exactly to Pack 3 [0.5,+1]")
        if terminated or truncated:
            break
    if neutral_final_info is None:
        raise AssertionError("neutral forward-only diagnostic did not execute")
    if neutral_final_info["safety_fault"] is not None:
        raise AssertionError("neutral forward-only diagnostic hit a safety fault")
    if neutral_contact_actions <= 0:
        raise AssertionError("neutral policy did not expose gripper/cube contact")
    if float(neutral_final_info["cube_progress_m"]) <= 0.0:
        raise AssertionError("neutral policy exposed no positive cube progress")
    if neutral_cube_reward <= 0.0:
        raise AssertionError("neutral policy exposed no positive cube-task reward")

    training_env.reset(seed=17)
    training_total_reward = 0.0
    training_accepted_actions = 0
    training_final_info: dict | None = None
    for _ in range(training_env.max_episode_steps):
        _, reward, terminated, truncated, info = training_env.step(
            np.asarray([1.0], dtype=np.float32)
        )
        training_total_reward += float(reward)
        training_accepted_actions += int(info["action_accepted"])
        training_final_info = info
        if info.get("normalized_action") != [1.0, 1.0]:
            raise AssertionError(
                "maximum policy action did not map exactly to Pack 3 [1,+1]"
            )
        if float(info["fixed_gripper_command_0_to_255"]) != float(
            training_env.vc["close_command"]
        ):
            raise AssertionError("forward-only adapter changed the closed command")
        if terminated or truncated:
            break

    if training_final_info is None or not bool(
        training_final_info["is_success"]
    ):
        raise AssertionError(
            "forward-only reference did not reach the cube goal: "
            f"progress={training_final_info['cube_progress_m'] if training_final_info else None}"
        )
    if training_final_info["safety_fault"] is not None:
        raise AssertionError("forward-only reference ended with a safety fault")
    training_contract = training_env.contract()
    training_env.close()

    result = {
        "verification_level": (
            "CR10_2F85_GYMNASIUM_FORWARD_ONLY_FIXED_GRIP_PUSH_BASELINE"
        ),
        "base_environment_level": "CR10_2F85_GYMNASIUM_TABLETOP_PUSH_ENV",
        "pack3_regression_level": pack3_result["verification_level"],
        "gymnasium": {
            "version": importlib.metadata.version("gymnasium"),
            "check_env_passed": True,
            "base_environment_id": ENV_ID,
            "training_environment_id": TRAINING_ENV_ID,
            "gym_make_passed": True,
        },
        "contract": training_contract,
        "base_contract": base_contract,
        "deterministic_reset": {
            "same_seed_observation_exactly_equal": True,
            "randomization_enabled": False,
            "rest_gate": reset_info_b["reset_diagnostics"],
        },
        "fail_closed_boundary_action": {
            "rejection_code": rejected_info["action_code"],
            "reward": float(rejected_reward),
            "simulation_substeps": int(rejected_info["simulation_substeps"]),
            "wrote_mujoco_state": False,
        },
        "precontact_approach_signal": {
            "action": [1.0, 1.0],
            "tcp_progress_reward": float(
                shaping_info["reward_terms"]["tcp_progress"]
            ),
            "total_reward": float(shaping_reward),
            "force_used_in_reward": False,
        },
        "base_reference_policy": {
            "policy": "constant normalized action [1, 1]",
            "accepted_actions": int(accepted_actions),
            "episode_steps": int(final_info["episode_step"]),
            "simulation_substeps": int(total_substeps),
            "cube_progress_m": float(final_info["cube_progress_m"]),
            "goal_displacement_m": float(final_info["cube_goal_displacement_m"]),
            "total_reward": float(total_reward),
            "gripper_cube_peak_force_norm_N": float(
                peak_gripper_cube_force
            ),
            "cube_table_peak_force_norm_N": float(peak_cube_table_force),
            "success": True,
            "safety_fault": None,
            "robot_table_contact_geoms": [],
        },
        "forward_only_fixed_grip_adapter": {
            "policy_action": "[a] in [-1,1]",
            "mapped_pack3_action": "[a] -> [(a+1)/2,+1]",
            "mapped_pack3_push_range": [0.0, 1.0],
            "physical_increment_range_m": [
                0.0,
                float(training_env.ac["max_xy_step_m"]),
            ],
            "fixed_gripper_command_0_to_255": float(
                training_final_info["fixed_gripper_command_0_to_255"]
            ),
            "invalid_action_rejection_code": training_rejection["action_code"],
            "invalid_action_wrote_state": False,
            "minimum_policy_action_mapping": minimum_mapping["normalized_action"],
        },
        "neutral_policy_diagnostic": {
            "policy": "constant policy action [0]",
            "mapped_pack3_action": [0.5, 1.0],
            "episode_steps": int(neutral_final_info["episode_step"]),
            "gripper_contact_actions": int(neutral_contact_actions),
            "cube_progress_m": float(neutral_final_info["cube_progress_m"]),
            "cube_progress_reward": float(neutral_cube_reward),
            "total_reward": float(neutral_total_reward),
            "success": bool(neutral_final_info["is_success"]),
            "safety_fault": None,
        },
        "reference_policy": {
            "policy": "constant fixed-grip policy action [1]",
            "accepted_actions": int(training_accepted_actions),
            "episode_steps": int(training_final_info["episode_step"]),
            "cube_progress_m": float(training_final_info["cube_progress_m"]),
            "goal_displacement_m": float(
                training_final_info["cube_goal_displacement_m"]
            ),
            "total_reward": float(training_total_reward),
            "success": True,
            "safety_fault": None,
            "robot_table_contact_geoms": [],
        },
        "hardware_motion_commanded": False,
        "claims": {
            "validated": [
                "Gymnasium reset/step API and declared spaces",
                "deterministic state-based supported-cube reset",
                "unchanged Pack 3 bounded action admission",
                "fail-closed boundary rejection without MuJoCo state writes",
                "finite progress-based reward with no force threshold",
                "scripted reachability of the explicit tabletop-push goal",
                "one-component training adapter maps every admitted policy action to the verified closed gripper command",
                "affine policy mapping removes backward motion while preserving the Pack 3 2 mm maximum",
                "neutral policy action exposes gripper contact, positive cube progress, and positive cube-task reward",
            ],
            "not_validated": [
                "learned-policy performance",
                "camera observations or vision policy",
                "domain randomization",
                "physical force calibration or force control",
                "ROS 2 hardware execution",
                "sim-to-real transfer",
            ],
        },
    }
    output = ROOT / (
        "verification/"
        "CR10_2F85_GYMNASIUM_FORWARD_ONLY_FIXED_GRIP_PUSH_BASELINE.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
