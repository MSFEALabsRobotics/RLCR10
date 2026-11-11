from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .config import load_config
from .control.joint_layout import JointLayout
from .control.mujoco_ik import DampedLeastSquaresIK
from .telemetry import SimulationTelemetry, TelemetryAccumulator

ROOT = Path(__file__).resolve().parents[1]


def _id(model: mujoco.MjModel, obj, name: str) -> int:
    value = mujoco.mj_name2id(model, obj, name)
    if value < 0:
        raise AssertionError(f"required element missing: {name}")
    return value


def _descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    body = int(body_id)
    while body > 0:
        if body == ancestor_id:
            return True
        body = int(model.body_parentid[body])
    return body == ancestor_id


def _finite(data: mujoco.MjData) -> bool:
    return bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))


def run(scene: Path) -> dict:
    cfg = load_config()
    mc, vc, ac = cfg["model"], cfg["verification"], cfg["action"]
    cg, tg = cfg["known_cube_push"], cfg["table_gravity_push"]

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    layout = JointLayout.from_model(model)
    model.opt.gravity[:] = np.asarray(tg["gravity_m_s2"], dtype=float)

    arm_aids = [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm_servo_{i}") for i in range(1, 7)]
    gripper_aid = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator")
    tcp_sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, mc["control_site_name"])
    mount_bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_mount_frame")
    cube_bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, "known_push_cube")
    cube_gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "known_push_cube_geom")
    cube_jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "known_push_cube_free")
    table_bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, "pack2_table")
    table_gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "pack2_table_geom")
    cube_qadr = int(model.jnt_qposadr[cube_jid])
    cube_dadr = int(model.jnt_dofadr[cube_jid])
    telemetry = TelemetryAccumulator(SimulationTelemetry(model, data))

    q_pose = np.deg2rad(np.asarray(mc["push_pose_deg"], dtype=float))
    mujoco.mj_resetData(model, data)
    layout.set_q(data, q_pose)
    layout.set_qdot_zero(data)
    for aid, value in zip(arm_aids, q_pose):
        data.ctrl[aid] = float(value)
    data.ctrl[gripper_aid] = float(vc["close_command"])
    mujoco.mj_forward(model, data)

    def hold(q_target: np.ndarray) -> None:
        for aid, value in zip(arm_aids, q_target):
            data.ctrl[aid] = float(value)
        data.ctrl[gripper_aid] = float(vc["close_command"])

    for _ in range(int(math.ceil(float(vc["gripper_settle_s"]) / model.opt.timestep))):
        hold(q_pose)
        mujoco.mj_step(model, data)
        if not _finite(data):
            raise AssertionError("non-finite state while establishing Pack 2 pose")

    tcp_start = data.site_xpos[tcp_sid].copy()
    fixed_R = data.site_xmat[tcp_sid].reshape(3, 3).copy()
    forward = fixed_R[:, 2].copy()
    forward /= np.linalg.norm(forward)
    if abs(float(forward[2])) > 0.03:
        raise AssertionError(f"Pack 2 push axis is not horizontal: {forward.round(6).tolist()}")

    cube_half = float(cg["cube_size_m"]) / 2.0
    cube_command = tcp_start + forward * (cube_half + float(cg["initial_gap_m"]))
    table_length = float(tg["table_size_m"][0])
    table_height = float(tg["table_size_m"][2])
    table_axis = forward.copy()
    table_axis[2] = 0.0
    table_axis /= np.linalg.norm(table_axis)
    back_support = float(tg["table_back_support_m"])
    table_center = cube_command + table_axis * (table_length / 2.0 - back_support)
    table_center[2] = cube_command[2] - cube_half - table_height / 2.0
    table_yaw = math.atan2(float(table_axis[1]), float(table_axis[0]))
    model.body_pos[table_bid] = table_center
    model.body_quat[table_bid] = np.array([
        math.cos(table_yaw / 2.0), 0.0, 0.0, math.sin(table_yaw / 2.0)
    ])
    data.qpos[cube_qadr:cube_qadr + 3] = cube_command
    data.qpos[cube_qadr + 3:cube_qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[cube_dadr:cube_dadr + 6] = 0.0
    mujoco.mj_forward(model, data)

    counts = {"table_total": 0, "table_during_push": 0, "gripper_cube": 0}
    gripper_geoms: dict[str, int] = {}
    max_penetration = 0.0
    phase = "settle"

    def inspect_contacts() -> None:
        nonlocal max_penetration
        table_seen = False
        gripper_seen: set[str] = set()
        robot_table_seen: set[str] = set()
        for index in range(data.ncon):
            contact = data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if {g1, g2} == {cube_gid, table_gid}:
                table_seen = True
                max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
            elif table_gid in (g1, g2):
                other = g2 if g1 == table_gid else g1
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom_{other}"
                robot_table_seen.add(name)
            if cube_gid not in (g1, g2):
                continue
            other = g2 if g1 == cube_gid else g1
            if _descendant(model, int(model.geom_bodyid[other]), mount_bid):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom_{other}"
                gripper_seen.add(name)
        if table_seen:
            counts["table_total"] += 1
            if phase == "push":
                counts["table_during_push"] += 1
        if gripper_seen:
            counts["gripper_cube"] += 1
            for name in gripper_seen:
                gripper_geoms[name] = gripper_geoms.get(name, 0) + 1
        if robot_table_seen:
            raise AssertionError(
                "robot/table intersection detected during "
                f"{phase}: geoms={sorted(robot_table_seen)}"
            )
        telemetry.observe(phase)

    inspect_contacts()

    settle_steps = int(math.ceil(float(tg["cube_settle_s"]) / model.opt.timestep))
    for _ in range(settle_steps):
        hold(q_pose)
        mujoco.mj_step(model, data)
        if not _finite(data):
            raise AssertionError("non-finite state while cube settled on table")
        inspect_contacts()

    table_top = float(table_center[2] + table_height / 2.0)
    expected_cube_z = table_top + cube_half
    rest_height_error = abs(float(data.xpos[cube_bid][2]) - expected_cube_z)
    rest_speed = float(np.linalg.norm(data.qvel[cube_dadr:cube_dadr + 3]))
    if counts["table_total"] < int(tg["min_table_contact_substeps"]):
        raise AssertionError(f"cube did not establish sustained table support: substeps={counts['table_total']}")
    if rest_height_error > float(tg["max_rest_height_error_m"]):
        raise AssertionError(f"cube rest-height error too high: {rest_height_error:.6f} m")
    if rest_speed > float(tg["max_rest_speed_m_s"]):
        raise AssertionError(f"cube did not settle on table: speed={rest_speed:.6f} m/s")
    if max_penetration > float(tg["max_contact_penetration_m"]):
        raise AssertionError(f"cube/table penetration too high: {max_penetration:.6f} m")
    if counts["gripper_cube"]:
        raise AssertionError("cube contacted gripper during gravity settling before commanded push")

    cube_start = data.xpos[cube_bid].copy()
    ik = DampedLeastSquaresIK(
        model,
        site_name=mc["control_site_name"],
        rot_scale=float(mc["rotation_characteristic_length_m"]),
        damping=float(mc["ik_damping"]),
        max_iterations=int(mc["ik_max_iterations"]),
        max_joint_step=float(mc["ik_max_joint_step_rad"]),
    )
    cart_target = data.site_xpos[tcp_sid].copy()
    remaining = float(cg["push_distance_m"])
    accepted = 0
    phase = "push"

    while remaining > 1e-12:
        step = min(remaining, float(ac["max_xy_step_m"]))
        cart_target += forward * step
        result = ik.solve(
            cart_target,
            fixed_R,
            layout.get_q(data).copy(),
            pos_tol=float(ac["max_ik_pos_error_m"]),
            rot_tol=float(ac["max_ik_rot_error_rad"]),
        )
        if not result.success:
            raise AssertionError(
                f"IK failed during supported push: pos={result.pos_error_m:.6f} m "
                f"rot={result.rot_error_rad:.6f} rad"
            )
        q_target = result.q.copy()
        accepted += 1

        tolerance = float(cg["waypoint_tracking_tolerance_m"])
        max_substeps = int(math.ceil(float(cg["waypoint_timeout_s"]) / model.opt.timestep))
        required_streak = int(cg["waypoint_settle_substeps"])
        streak = 0
        reached = False
        for _ in range(max_substeps):
            hold(q_target)
            mujoco.mj_step(model, data)
            if not _finite(data):
                raise AssertionError("non-finite state during supported push")
            inspect_contacts()
            if np.linalg.norm(data.site_xpos[tcp_sid] - cart_target) <= tolerance:
                streak += 1
                if streak >= required_streak:
                    reached = True
                    break
            else:
                streak = 0
        if not reached:
            actual_tcp = data.site_xpos[tcp_sid].copy()
            tcp_error = actual_tcp - cart_target
            q_error = layout.get_q(data) - q_target
            cube_now = data.xpos[cube_bid].copy()
            raise AssertionError(
                "arm did not reach supported-push waypoint: "
                f"waypoint={accepted} target={cart_target.round(6).tolist()} "
                f"actual={actual_tcp.round(6).tolist()} "
                f"error={np.linalg.norm(tcp_error):.6f} m "
                f"error_along={np.dot(tcp_error, forward):.6f} m "
                f"joint_error_norm={np.linalg.norm(q_error):.6f} rad "
                f"joint_error_max={np.max(np.abs(q_error)):.6f} rad "
                f"cube_along={np.dot(cube_now-cube_start, forward):.6f} m "
                f"table_push_substeps={counts['table_during_push']} "
                f"gripper_contact_substeps={counts['gripper_cube']} "
                f"contact_geoms={dict(sorted(gripper_geoms.items()))}"
            )
        remaining -= step

    cube_end = data.xpos[cube_bid].copy()
    cube_delta = cube_end - cube_start
    cube_along = float(np.dot(cube_delta, forward))
    vertical_drift = abs(float(cube_end[2] - cube_start[2]))
    cube_up = data.xmat[cube_bid].reshape(3, 3)[:, 2]
    tilt_deg = math.degrees(math.acos(float(np.clip(np.dot(cube_up, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))))

    if counts["gripper_cube"] < int(cg["min_contact_substeps"]):
        raise AssertionError(f"no sustained gripper/cube contact during table push: {counts['gripper_cube']}")
    if counts["table_during_push"] < int(tg["min_supported_push_contact_substeps"]):
        raise AssertionError("cube lost table support during the push")
    if cube_along < float(cg["min_cube_displacement_m"]):
        raise AssertionError(f"cube did not move enough along table: {cube_along:.6f} m")
    if vertical_drift > float(tg["max_cube_vertical_drift_m"]):
        raise AssertionError(f"cube vertical drift too high: {vertical_drift:.6f} m")
    if tilt_deg > float(tg["max_cube_tilt_deg"]):
        raise AssertionError(f"cube tipped too far during planar push: {tilt_deg:.3f} deg")
    if max_penetration > float(tg["max_contact_penetration_m"]):
        raise AssertionError(f"maximum cube/table penetration too high: {max_penetration:.6f} m")

    result = {
        "verification_level": "CR10_2F85_TABLE_GRAVITY_SUPPORTED_PUSH",
        "gravity_m_s2": [float(x) for x in model.opt.gravity],
        "table_center_world_m": table_center.tolist(),
        "table_size_m": [float(x) for x in tg["table_size_m"]],
        "table_friction_simulation_value": [float(x) for x in tg["table_friction"]],
        "table_push_axis_world": table_axis.tolist(),
        "table_back_support_m": back_support,
        "cube_start_world_m_after_settle": cube_start.tolist(),
        "rest_height_error_m": rest_height_error,
        "rest_speed_m_s": rest_speed,
        "max_cube_table_penetration_m": max_penetration,
        "accepted_cartesian_waypoints": accepted,
        "cube_motion_along_push_m": cube_along,
        "cube_vertical_drift_m": vertical_drift,
        "cube_final_tilt_deg": tilt_deg,
        "table_contact_substeps_total": counts["table_total"],
        "table_contact_substeps_during_push": counts["table_during_push"],
        "gripper_cube_contact_substeps": counts["gripper_cube"],
        "gripper_contact_geoms": dict(sorted(gripper_geoms.items())),
        "robot_table_contact_substeps": 0,
        "simulation_telemetry": telemetry.summary(),
        "hardware_motion_commanded": False,
        "claim_boundary": tg["claim"],
    }
    out = ROOT / "verification/CR10_2F85_TABLE_GRAVITY_SUPPORTED_PUSH.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
