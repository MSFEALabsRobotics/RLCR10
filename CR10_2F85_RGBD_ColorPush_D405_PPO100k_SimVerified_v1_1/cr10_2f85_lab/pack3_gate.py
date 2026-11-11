from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from .config import load_config
from .control.bounded_push_action import BoundedPushActionInterface
from .control.joint_layout import JointLayout
from .telemetry import validate_telemetry_summary


ROOT = Path(__file__).resolve().parents[1]


def _make_interface(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: dict,
    origin: np.ndarray,
    direction: np.ndarray,
    rotation: np.ndarray,
    initial_segment_m: float,
) -> BoundedPushActionInterface:
    return BoundedPushActionInterface(
        model,
        data,
        site_name=cfg["model"]["control_site_name"],
        segment_origin_world_m=origin,
        segment_direction_world=direction,
        fixed_tcp_rotation_world=rotation,
        segment_length_m=float(cfg["known_cube_push"]["push_distance_m"]),
        initial_segment_m=initial_segment_m,
        action_config=cfg["action"],
        model_config=cfg["model"],
    )


def run(scene: Path, supported_push_result: dict) -> dict:
    """Verify Pack 3 contracts while preserving the Pack 2 dynamics run."""

    cfg = load_config()
    telemetry_summary = supported_push_result.get("simulation_telemetry")
    if not isinstance(telemetry_summary, dict):
        raise AssertionError("Pack 2 supported push did not provide Pack 3 telemetry")
    telemetry_gate = validate_telemetry_summary(telemetry_summary)

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    layout = JointLayout.from_model(model)
    q_pose = np.deg2rad(np.asarray(cfg["model"]["push_pose_deg"], dtype=float))
    mujoco.mj_resetData(model, data)
    layout.set_q(data, q_pose)
    layout.set_qdot_zero(data)
    table_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pack2_table"
    )
    if table_id < 0:
        raise AssertionError("Pack 2 table missing in Pack 3 action gate")
    model.body_pos[table_id] = np.array([8.0, 8.0, 8.0])
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        cfg["model"]["control_site_name"],
    )
    if site_id < 0:
        raise AssertionError("nominal TCP site missing in Pack 3 action gate")
    origin = data.site_xpos[site_id].copy()
    rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    direction = rotation[:, 2].copy()
    direction /= np.linalg.norm(direction)

    interface = _make_interface(
        model, data, cfg, origin, direction, rotation, initial_segment_m=0.0
    )
    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()
    ctrl_before = data.ctrl.copy()

    accepted = interface.plan_normalized([1.0, 1.0])
    if not accepted.accepted:
        raise AssertionError(
            "valid bounded action was rejected: "
            f"{accepted.code}: {accepted.detail}"
        )
    expected_step = float(cfg["action"]["max_xy_step_m"])
    if abs(float(accepted.target_segment_m) - expected_step) > 1e-12:
        raise AssertionError("accepted action did not map to one configured TCP step")

    open_only = interface.plan_normalized([0.0, -1.0])
    close_only = interface.plan_normalized([0.0, 1.0])
    for label, decision in [("open", open_only), ("close", close_only)]:
        if not decision.accepted:
            raise AssertionError(
                f"valid gripper {label} action was rejected: {decision.code}"
            )

    rejection_cases = {
        "segment_underflow": (interface.plan_normalized([-1.0, 0.0]), "SEGMENT_BOUND"),
        "normalized_overflow": (
            interface.plan_normalized([1.000001, 0.0]),
            "ACTION_OUT_OF_RANGE",
        ),
        "nonfinite": (interface.plan_normalized([float("nan"), 0.0]), "NONFINITE_ACTION"),
        "wrong_shape": (interface.plan_normalized([0.0]), "INVALID_ACTION_SHAPE"),
    }
    segment_length = float(cfg["known_cube_push"]["push_distance_m"])
    at_end = _make_interface(
        model,
        data,
        cfg,
        origin,
        direction,
        rotation,
        initial_segment_m=segment_length,
    )
    rejection_cases["segment_overflow"] = (
        at_end.plan_normalized([1.0, 0.0]),
        "SEGMENT_BOUND",
    )

    for label, (decision, expected_code) in rejection_cases.items():
        if decision.accepted or decision.code != expected_code:
            raise AssertionError(
                f"{label} rejection contract failed: "
                f"accepted={decision.accepted} code={decision.code}"
            )

    interface.commit(accepted)
    if abs(interface.segment_m - expected_step) > 1e-12:
        raise AssertionError("committed action did not update the segment coordinate")
    if not (
        np.array_equal(data.qpos, qpos_before)
        and np.array_equal(data.qvel, qvel_before)
        and np.array_equal(data.ctrl, ctrl_before)
    ):
        raise AssertionError("Pack 3 action admission unexpectedly wrote MuJoCo state")

    result = {
        "verification_level": "CR10_2F85_SIM_TELEMETRY_BOUNDED_PUSH_ACTION",
        "pack2_regression_level": supported_push_result["verification_level"],
        "telemetry": telemetry_gate,
        "action_contract": interface.contract(),
        "action_gate": {
            "valid_step": accepted.as_dict(),
            "open_mapping_command": float(open_only.gripper_command),
            "closed_mapping_command": float(close_only.gripper_command),
            "rejection_codes": {
                key: decision.code
                for key, (decision, _) in rejection_cases.items()
            },
            "planner_wrote_mujoco_state": False,
        },
        "hardware_motion_commanded": False,
        "claims": {
            "validated": [
                "read-only access to finite MuJoCo gripper actuator outputs",
                "read-only cube/table and gripper/cube contact wrenches",
                "explicit contact/world-frame conversion contract",
                "normalized two-component simulation action bounds",
                "exact 0--80 mm Pack 2 segment admission",
                "IK, Jacobian, joint-margin, and joint-target-delta admission gates",
                "fail-closed rejection of malformed, nonfinite, and out-of-bound actions",
            ],
            "not_validated": [
                "physical fingertip force or torque",
                "physical force/torque sensor behavior",
                "physical gripper current-to-force calibration",
                "direct force control",
                "real CR10 workspace or collision safety",
                "real motor velocity tracking",
                "hardware motion",
                "RL policy performance",
            ],
        },
    }
    out = ROOT / "verification/CR10_2F85_SIM_TELEMETRY_BOUNDED_PUSH_ACTION.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
