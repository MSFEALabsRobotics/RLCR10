from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import mujoco

from .config import load_config

ROOT = Path(__file__).resolve().parents[1]

def _id(model, obj, name: str) -> int:
    i = mujoco.mj_name2id(model, obj, name)
    if i < 0:
        raise AssertionError(f"required element missing: {name}")
    return i

def _qadr(model, joint: str) -> int:
    jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    return int(model.jnt_qposadr[jid])

def _set_arm_home(model, data, home_rad: np.ndarray) -> None:
    for i, q in enumerate(home_rad, 1):
        data.qpos[_qadr(model, f"joint{i}")] = float(q)
        aid = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm_servo_{i}")
        data.ctrl[aid] = float(q)

def _step(model, data, seconds: float) -> None:
    n = int(math.ceil(seconds / float(model.opt.timestep)))
    for _ in range(n):
        mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            raise AssertionError("non-finite simulation state")

def run(scene: Path) -> dict:
    cfg = load_config(); v = cfg["verification"]; mc = cfg["model"]
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    assert model.nu == 7, f"expected 7 actuators, got {model.nu}"

    for name in [f"joint{i}" for i in range(1,7)]:
        _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    for name in [f"arm_servo_{i}" for i in range(1,7)]:
        _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    gid = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator")
    sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, mc["control_site_name"])
    pinch_sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, "menagerie_pinch_site")
    mount_bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_mount_frame")
    ldriver = _qadr(model, "left_driver_joint")
    rdriver = _qadr(model, "right_driver_joint")

    home = np.deg2rad(np.asarray(mc["home_deg"], dtype=float))
    mujoco.mj_resetData(model, data); _set_arm_home(model, data, home); data.ctrl[gid] = float(v["open_command"]); mujoco.mj_forward(model, data)

    tcp_local = np.asarray(mc["nominal_tcp_xyz_m"], dtype=float)
    mount_p = data.xpos[mount_bid].copy()
    mount_R = data.xmat[mount_bid].reshape(3, 3).copy()
    tcp_world_expected = mount_p + mount_R @ tcp_local
    tcp_world = data.site_xpos[sid].copy()
    tcp_local_err = float(np.linalg.norm(tcp_world - tcp_world_expected))
    if tcp_local_err > float(v["tcp_local_position_tolerance_m"]):
        raise AssertionError(f"nominal TCP local transform mismatch: {tcp_local_err:.3e} m")
    tcp_offset_norm = float(np.linalg.norm(tcp_world - mount_p))

    _step(model, data, float(v["gripper_settle_s"]))
    open_l = float(data.qpos[ldriver]); open_r = float(data.qpos[rdriver])
    if max(abs(open_l), abs(open_r)) > float(v["open_driver_max_rad"]):
        raise AssertionError(f"open command did not reach open state: left={open_l:.4f}, right={open_r:.4f}")

    data.ctrl[gid] = float(v["close_command"])
    _step(model, data, float(v["gripper_settle_s"]))
    close_l = float(data.qpos[ldriver]); close_r = float(data.qpos[rdriver])
    if min(close_l, close_r) < float(v["close_driver_min_rad"]):
        raise AssertionError(f"close command did not close both drivers: left={close_l:.4f}, right={close_r:.4f}")
    sym = abs(close_l - close_r)
    if sym > float(v["driver_symmetry_max_error_rad"]):
        raise AssertionError(f"driver coupling symmetry error too large: {sym:.6f} rad")

    mujoco.mj_forward(model, data)
    nominal_to_pinch = float(np.linalg.norm(data.site_xpos[sid] - data.site_xpos[pinch_sid]))

    mujoco.mj_resetData(model, data); _set_arm_home(model, data, home); data.ctrl[gid] = float(v["open_command"]); mujoco.mj_forward(model, data)
    p_home = data.site_xpos[sid].copy()
    qprobe = home.copy(); qprobe[0] += float(v["joint1_probe_rad"])
    for i, q in enumerate(qprobe, 1):
        data.qpos[_qadr(model, f"joint{i}")] = float(q)
    mujoco.mj_forward(model, data)
    p_probe = data.site_xpos[sid].copy()
    fk_motion = float(np.linalg.norm(p_probe - p_home))
    if fk_motion < float(v["fk_motion_min_m"]):
        raise AssertionError(f"nominal TCP site did not move with CR10 FK: {fk_motion:.6g} m")

    result = {
        "verification_level": "CR10_2F85_SIM_MODEL_FOUNDATION_NOMINAL_TCP",
        "model": {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu)},
        "arm_home_deg": mc["home_deg"],
        "nominal_tcp": {
            "xyz_m_in_mount_frame": mc["nominal_tcp_xyz_m"],
            "rpy_rad_in_mount_frame": mc["nominal_tcp_rpy_rad"],
            "world_home_m": [float(x) for x in p_home],
            "offset_norm_m": tcp_offset_norm,
            "local_transform_error_m": tcp_local_err,
            "status": mc["tcp_status"],
            "source": mc["tcp_source"],
            "menagerie_pinch_separation_m_diagnostic": nominal_to_pinch,
        },
        "gripper": {
            "open_driver_rad": [open_l, open_r],
            "closed_driver_rad": [close_l, close_r],
            "closed_driver_symmetry_error_rad": sym,
            "actuator_ctrlrange": [float(x) for x in model.actuator_ctrlrange[gid]],
        },
        "fk_probe_motion_m": fk_motion,
        "mount_assumption": mc["mount_assumption"],
        "mount_status": mc["mount_status"],
        "dynamics_status": mc["dynamics_status"],
        "hardware_motion_commanded": False,
        "claims": {
            "validated": [
                "exact source locks",
                "CR10+2F85 MuJoCo compile",
                "six named CR10 joints",
                "one Menagerie 2F-85 actuator",
                "2F-85 open/close mechanism response",
                "left/right driver coupling",
                "nominal Robotiq 2F-85 TCP convention [0,0,0.174] m encoded in mount frame",
                "nominal TCP attached to CR10 kinematic chain",
            ],
            "not_validated": [
                "physical CR10-to-2F85 mount/index orientation",
                "physical lab TCP calibration/verification",
                "identified CR10 dynamics",
                "physical grasp/contact fidelity",
                "hardware gripper communication",
                "hardware arm motion",
                "RL task",
            ],
        },
    }
    out = ROOT / "verification/CR10_2F85_SIM_MODEL_FOUNDATION_NOMINAL_TCP.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
