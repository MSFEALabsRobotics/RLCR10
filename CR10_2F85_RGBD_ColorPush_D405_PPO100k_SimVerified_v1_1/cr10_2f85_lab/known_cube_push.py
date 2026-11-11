from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import mujoco

from .config import load_config
from .control.joint_layout import JointLayout
from .control.mujoco_ik import DampedLeastSquaresIK

ROOT = Path(__file__).resolve().parents[1]


def _id(model: mujoco.MjModel, obj, name: str) -> int:
    i = mujoco.mj_name2id(model, obj, name)
    if i < 0:
        raise AssertionError(f"required element missing: {name}")
    return i


def _body_is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    b = int(body_id)
    while b > 0:
        if b == ancestor_id:
            return True
        b = int(model.body_parentid[b])
    return b == ancestor_id


def _set_arm(model, data, layout, arm_ids, q):
    layout.set_q(data, q)
    layout.set_qdot_zero(data)
    for aid, value in zip(arm_ids, q):
        data.ctrl[aid] = float(value)


def _hold(data, arm_ids, q, gripper_aid, gripper_cmd):
    for aid, value in zip(arm_ids, q):
        data.ctrl[aid] = float(value)
    data.ctrl[gripper_aid] = float(gripper_cmd)


def _finite(data):
    return bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))


def run(scene: Path) -> dict:
    cfg = load_config()
    mc = cfg["model"]
    cg = cfg["known_cube_push"]
    vc = cfg["verification"]

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    layout = JointLayout.from_model(model)

    if bool(cg["zero_gravity"]):
        model.opt.gravity[:] = 0.0

    arm_ids = [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm_servo_{i}") for i in range(1, 7)]
    gripper_aid = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator")
    tcp_sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, mc["control_site_name"])
    mount_bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_mount_frame")
    cube_bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, "known_push_cube")
    cube_gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "known_push_cube_geom")
    cube_jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "known_push_cube_free")

    cube_qadr = int(model.jnt_qposadr[cube_jid])
    cube_dadr = int(model.jnt_dofadr[cube_jid])
    if int(model.jnt_type[cube_jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise AssertionError("known_push_cube_free is not a free joint")

    q_pose = np.deg2rad(np.asarray(mc["push_pose_deg"], dtype=float))
    mujoco.mj_resetData(model, data)
    _set_arm(model, data, layout, arm_ids, q_pose)
    data.ctrl[gripper_aid] = float(vc["close_command"])
    mujoco.mj_forward(model, data)

    nclose = int(math.ceil(float(vc["gripper_settle_s"]) / float(model.opt.timestep)))
    for _ in range(nclose):
        _hold(data, arm_ids, q_pose, gripper_aid, float(vc["close_command"]))
        mujoco.mj_step(model, data)
        if not _finite(data):
            raise AssertionError("non-finite state while closing 2F-85")

    tcp_start = data.site_xpos[tcp_sid].copy()
    fixed_R = data.site_xmat[tcp_sid].reshape(3, 3).copy()
    forward = fixed_R[:, 2].copy()
    forward /= np.linalg.norm(forward)

    if abs(float(forward[2])) > 0.03:
        raise AssertionError(
            f"configured push pose does not make TCP +Z horizontal: axis={forward.round(6).tolist()}"
        )

    cube_half = float(cg["cube_size_m"]) / 2.0
    tcp_clearance = float(cg["initial_gap_m"])
    cube_start_commanded = tcp_start + forward * (cube_half + tcp_clearance)

    data.qpos[cube_qadr:cube_qadr + 3] = cube_start_commanded
    data.qpos[cube_qadr + 3:cube_qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[cube_dadr:cube_dadr + 6] = 0.0
    mujoco.mj_forward(model, data)

    cube_start = data.xpos[cube_bid].copy()

    initial_pairs = []
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if cube_gid not in (g1, g2):
            continue
        other = g2 if g1 == cube_gid else g1
        if _body_is_descendant(model, int(model.geom_bodyid[other]), mount_bid):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom_{other}"
            initial_pairs.append(name)
    if initial_pairs:
        raise AssertionError(
            "known cube starts in gripper contact even though its near face is "
            f"{tcp_clearance:.3f} m beyond the nominal TCP plane: {sorted(set(initial_pairs))}"
        )

    ik = DampedLeastSquaresIK(
        model,
        site_name=mc["control_site_name"],
        rot_scale=float(mc["rotation_characteristic_length_m"]),
        damping=float(mc["ik_damping"]),
        max_iterations=int(mc["ik_max_iterations"]),
        max_joint_step=float(mc["ik_max_joint_step_rad"]),
    )
    cart_target = tcp_start.copy()

    counters = {"gripper_cube_contact_substeps": 0}
    contact_geoms = {}

    def inspect_contacts():
        seen = set()
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            if cube_gid not in (g1, g2):
                continue
            other = g2 if g1 == cube_gid else g1
            if not _body_is_descendant(model, int(model.geom_bodyid[other]), mount_bid):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom_{other}"
            seen.add(name)
        if seen:
            counters["gripper_cube_contact_substeps"] += 1
            for name in seen:
                contact_geoms[name] = contact_geoms.get(name, 0) + 1

    remaining = float(cg["push_distance_m"])
    accepted = 0
    min_sigma = float("inf")
    max_ik_pos = 0.0
    max_ik_rot = 0.0

    while remaining > 1e-12:
        step = min(remaining, float(cfg["action"]["max_xy_step_m"]))
        cart_target += forward * step

        seed_q = layout.get_q(data).copy()
        result = ik.solve(
            cart_target, fixed_R, seed_q,
            pos_tol=float(cfg["action"]["max_ik_pos_error_m"]),
            rot_tol=float(cfg["action"]["max_ik_rot_error_rad"]),
        )
        max_ik_pos = max(max_ik_pos, float(result.pos_error_m))
        max_ik_rot = max(max_ik_rot, float(result.rot_error_rad))
        min_sigma = min(min_sigma, float(result.scaled_sigma_min))
        if not result.success:
            raise AssertionError(
                f"IK failed for known scripted push: pos={result.pos_error_m:.6f} m "
                f"rot={result.rot_error_rad:.6f} rad sigma={result.scaled_sigma_min:.6f}"
            )
        q_target = result.q.copy()
        accepted += 1

        tol = float(cg["waypoint_tracking_tolerance_m"])
        timeout_s = float(cg["waypoint_timeout_s"])
        need_streak = int(cg["waypoint_settle_substeps"])
        max_substeps = int(math.ceil(timeout_s / float(model.opt.timestep)))
        streak = 0
        reached = False
        for sub in range(max_substeps):
            _hold(data, arm_ids, q_target, gripper_aid, float(vc["close_command"]))
            mujoco.mj_step(model, data)
            if not _finite(data):
                raise AssertionError("non-finite state during known-cube push")
            inspect_contacts()
            actual_err = float(np.linalg.norm(data.site_xpos[tcp_sid] - cart_target))
            if actual_err <= tol:
                streak += 1
                if streak >= need_streak:
                    reached = True
                    break
            else:
                streak = 0
        if not reached:
            actual = data.site_xpos[tcp_sid].copy()
            raise AssertionError(
                "arm dynamics did not reach scripted Cartesian waypoint before timeout: "
                f"target={cart_target.round(6).tolist()} actual={actual.round(6).tolist()} "
                f"error={np.linalg.norm(actual-cart_target):.6f} m timeout={timeout_s:.2f} s"
            )
        remaining -= step

    cube_end = data.xpos[cube_bid].copy()
    tcp_end = data.site_xpos[tcp_sid].copy()
    cube_delta = cube_end - cube_start
    cube_along_motion = float(np.dot(cube_delta, forward))
    cube_total_motion = float(np.linalg.norm(cube_delta))
    tcp_along_motion = float(np.dot(tcp_end - tcp_start, forward))
    tcp_tracking_error = float(np.linalg.norm(tcp_end - cart_target))

    if counters["gripper_cube_contact_substeps"] < int(cg["min_contact_substeps"]):
        raise AssertionError(
            "no sustained source-locked 2F-85/cube contact: "
            f"substeps={counters['gripper_cube_contact_substeps']}; "
            f"tcp_start={tcp_start.round(6).tolist()}; "
            f"forward={forward.round(6).tolist()}; "
            f"cube_start={cube_start.round(6).tolist()}; "
            f"tcp_along={tcp_along_motion:.6f} m; cube_along={cube_along_motion:.6f} m; "
            f"contact_geoms={sorted(contact_geoms)}"
        )
    if cube_along_motion < float(cg["min_cube_displacement_m"]):
        raise AssertionError(
            f"cube did not move enough along commanded push: {cube_along_motion:.6f} m"
        )
    if tcp_tracking_error > float(cg["max_final_tcp_tracking_error_m"]):
        raise AssertionError(
            f"final TCP tracking error too high: {tcp_tracking_error:.6f} m"
        )

    result = {
        "push_pose_deg": [float(x) for x in mc["push_pose_deg"]],
        "tcp_start_xyz_m": tcp_start.tolist(),
        "tcp_forward_axis_world": forward.tolist(),
        "cube_start_xyz_m": cube_start.tolist(),
        "cube_near_face_clearance_from_nominal_tcp_m": tcp_clearance,
        "commanded_push_distance_m": float(cg["push_distance_m"]),
        "accepted_cartesian_waypoints": accepted,
        "gripper_cube_contact_substeps": counters["gripper_cube_contact_substeps"],
        "contact_geoms": dict(sorted(contact_geoms.items())),
        "cube_motion_along_push_m": cube_along_motion,
        "cube_total_motion_m": cube_total_motion,
        "tcp_motion_along_push_m": tcp_along_motion,
        "final_tcp_tracking_error_m": tcp_tracking_error,
        "min_scaled_sigma": min_sigma,
        "max_ik_position_error_m": max_ik_pos,
        "max_ik_rotation_error_rad": max_ik_rot,
        "zero_gravity": bool(cg["zero_gravity"]),
        "placement_method": "explicit_nominal_tcp_plus_forward_axis_offset",
        "waypoint_tracking_tolerance_m": float(cg["waypoint_tracking_tolerance_m"]),
        "waypoint_timeout_s": float(cg["waypoint_timeout_s"]),
    }
    out = ROOT / "verification/CR10_2F85_KNOWN_CUBE_PUSH_CONTACT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
