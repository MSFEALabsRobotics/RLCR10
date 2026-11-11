#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cr10_2f85_lab.config import load_config
from cr10_2f85_lab.source_lock import validate_sources, import_locked_sources, LockError
from cr10_2f85_lab.build_model import build
from cr10_2f85_lab.verify_sim import run as run_foundation
from cr10_2f85_lab.known_cube_push import run as run_push
from cr10_2f85_lab.table_gravity_push import run as run_table_gravity
from cr10_2f85_lab.pack3_gate import run as run_pack3
from cr10_2f85_lab.pack4_gate import run as run_pack4
from cr10_2f85_lab.pack5_gate import run as run_pack5
from cr10_2f85_lab.color_scene import build_color_task_scene
from cr10_2f85_lab.pack5_color_gate import run as run_pack5_color


def main() -> int:
    ap = argparse.ArgumentParser(
        description=("CR10 + Robotiq 2F-85 Pack 5 RGB-D color-target verifier")
    )
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--dobot", default=None)
    ap.add_argument("--robotiq", default=None)
    ap.add_argument("--menagerie", default=None)
    ap.add_argument(
        "--skip-camera-render",
        action="store_true",
        help=(
            "Run only the CPU camera-geometry gate. This is for headless "
            "build systems and does not establish the Pack 5 RGB-D render level."
        ),
    )
    args = ap.parse_args()
    if not args.sim:
        print(
            "Refusing to run without --sim. This package has no hardware-motion mode."
        )
        return 2

    cfg = load_config()
    s = cfg["source_lock"]
    paths = {
        "dobot": Path(args.dobot or s["dobot_repo"]).expanduser(),
        "robotiq": Path(args.robotiq or s["robotiq_repo"]).expanduser(),
        "menagerie": Path(args.menagerie or s["menagerie_repo"]).expanduser(),
    }

    print("=== CR10 + ROBOTIQ 2F-85 LAB PACK 5 v0.2 CANDIDATE ===")
    print("Simulation only. No physical motion can be commanded by this package.\n")
    try:
        manifest = validate_sources(**paths)
    except LockError as exc:
        print("[FAIL] SOURCE LOCK:", exc)
        return 1

    print("[PASS] source lock")
    print("  Dobot     ", manifest["dobot"]["commit"])
    print("  Robotiq   ", manifest["robotiq"]["commit"])
    print("  Menagerie ", manifest["menagerie"]["commit"])

    import_locked_sources(ROOT, **paths, manifest=manifest)
    print("[PASS] imported locked source files")
    scene = build()
    print("[PASS] combined CR10 + 2F-85 + one free cube model compiled")

    foundation = run_foundation(scene)
    print("[PASS] frozen arm+gripper foundation regression")
    print(
        "  nq/nv/nu:",
        foundation["model"]["nq"],
        foundation["model"]["nv"],
        foundation["model"]["nu"],
    )
    print("  nominal TCP xyz(m):", foundation["nominal_tcp"]["xyz_m_in_mount_frame"])

    push = run_push(scene)
    print("[PASS] deterministic known-cube push contact")
    print("  cube start xyz(m):", [round(x, 6) for x in push["cube_start_xyz_m"]])
    print("  push axis world:  ", [round(x, 6) for x in push["tcp_forward_axis_world"]])
    print("  TCP moved(m):     ", round(push["tcp_motion_along_push_m"], 6))
    print("  cube moved(m):    ", round(push["cube_motion_along_push_m"], 6))
    print("  contact substeps: ", push["gripper_cube_contact_substeps"])
    print("  contact geoms:    ", push["contact_geoms"])

    supported = run_table_gravity(scene)
    print("[PASS] Earth-gravity table support + supported cube push")
    print("  rest height error(m):", round(supported["rest_height_error_m"], 7))
    print("  rest speed(m/s):     ", round(supported["rest_speed_m_s"], 7))
    print("  cube moved(m):       ", round(supported["cube_motion_along_push_m"], 6))
    print("  table contacts:      ", supported["table_contact_substeps_total"])
    print("  gripper contacts:    ", supported["gripper_cube_contact_substeps"])
    print("  contact geoms:       ", supported["gripper_contact_geoms"])
    print("  robot/table contacts:", supported["robot_table_contact_substeps"])

    pack3 = run_pack3(scene, supported)
    telemetry = pack3["telemetry"]
    print("[PASS] read-only simulated gripper/contact telemetry")
    print(
        "  gripper actuator peak(abs, model units):",
        round(telemetry["gripper_actuator_force_scalar_peak_abs_model_units"], 6),
    )
    print(
        "  driver torque peaks(abs, N*m):",
        [
            round(telemetry["left_driver_joint_torque_peak_abs_Nm"], 6),
            round(telemetry["right_driver_joint_torque_peak_abs_Nm"], 6),
        ],
    )
    print(
        "  gripper/cube contact peak(norm, N):",
        round(telemetry["gripper_cube_peak_force_norm_N"], 6),
    )
    print(
        "  cube/table contact peak(norm, N):",
        round(telemetry["cube_table_peak_force_norm_N"], 6),
    )
    print("[PASS] bounded two-component simulation task-action contract")
    print("  action: [push_increment, gripper_position] in [-1, 1]^2")
    print("  push increment: <= 2.0 mm on the exact 0--80 mm Pack 2 segment")
    print(
        "  planner wrote MuJoCo state:",
        pack3["action_gate"]["planner_wrote_mujoco_state"],
    )

    pack4 = run_pack4(scene, pack3)
    reference = pack4["reference_policy"]
    rejection = pack4["fail_closed_boundary_action"]
    print("[PASS] deterministic state-based Gymnasium tabletop-push environment")
    print("  Gymnasium version:       ", pack4["gymnasium"]["version"])
    print("  check_env passed:        ", pack4["gymnasium"]["check_env_passed"])
    print("  observation shape:       ", pack4["contract"]["observation_shape"])
    print(
        "  deterministic reset:     ",
        pack4["deterministic_reset"]["same_seed_observation_exactly_equal"],
    )
    print("  rejected action code:    ", rejection["rejection_code"])
    print("  rejection wrote state:   ", rejection["wrote_mujoco_state"])
    print("  reference episode steps: ", reference["episode_steps"])
    print("  cube progress(m):        ", round(reference["cube_progress_m"], 6))
    print("  task goal(m):            ", round(reference["goal_displacement_m"], 6))
    print("  reference success:       ", reference["success"])
    print("[PASS] forward-only fixed-grip PPO training adapter")
    print("  training environment:    ", pack4["gymnasium"]["training_environment_id"])
    print("  policy action shape:     ", pack4["contract"]["action"]["shape"])
    adapter = pack4["forward_only_fixed_grip_adapter"]
    neutral = pack4["neutral_policy_diagnostic"]
    print("  mapped Pack 3 action:    ", adapter["mapped_pack3_action"])
    print("  physical push range(m):  ", adapter["physical_increment_range_m"])
    print("  fixed gripper command:   ", adapter["fixed_gripper_command_0_to_255"])
    print("  neutral-policy cube(m):  ", round(neutral["cube_progress_m"], 6))
    print("  neutral contact actions: ", neutral["gripper_contact_actions"])

    print(
        "NOTE: reported forces are MuJoCo solver values, not calibrated physical sensor readings."
    )
    print(
        "NOTE: the gripper command remains position-like; this pack does not implement force control."
    )
    print(
        "NOTE: the training adapter maps policy action [-1,+1] to forward push [0,+1] and fixes the gripper closed."
    )
    print(
        "NOTE: rewards, task thresholds, dynamics, contacts, and the signed Pack 3 interface are unchanged."
    )
    print(
        "NOTE: the Pack 4 gate checks the environment and a scripted reference; it does not claim learned-policy performance."
    )

    try:
        pack5 = run_pack5(
            scene,
            pack4,
            require_opengl_render=not args.skip_camera_render,
        )
    except RuntimeError as exc:
        print("[FAIL] Pack 5 native RGB-D render gate:", exc)
        return 1
    camera = pack5["camera"]
    depth = pack5["cpu_depth_gate"]
    rigid = pack5["rigid_attachment"]
    print("[PASS] provisional wrist D405 camera geometry")
    print("  stream convention:       ", camera["stream_convention"])
    print("  native resolution(px):   ", camera["native_resolution_px"])
    print(
        "  initial cube pixel:      ",
        [round(value, 3) for value in camera["initial_cube_center_pixel"]],
    )
    print(
        "  initial cube depth(m):   ", round(camera["initial_cube_center_depth_m"], 6)
    )
    print("  trajectory frames:       ", depth["trajectory_frames_checked"])
    print("  minimum cube pixels:     ", depth["minimum_cube_pixels"])
    print("  minimum table pixels:    ", depth["minimum_table_pixels"])
    print("  camera translation drift:", rigid["maximum_position_drift_m"])
    if pack5["opengl_rgbd_render"]["validated"]:
        snapshot = pack5["opengl_rgbd_render"]["snapshot"]
        print("[PASS] native MuJoCo RGB + metric-depth render")
        print("  RGB shape:               ", snapshot["rgb_shape"])
        print("  depth shape:             ", snapshot["depth_shape"])
        print("  rendered cube pixels:    ", snapshot["cube_pixels"])
        print("  rendered table pixels:   ", snapshot["table_pixels"])
        print("  rendered CR10 arm pixels:", snapshot["cr10_arm_pixels"])
        print("  cube center raster match:", snapshot["projected_center_pixel_is_cube"])
    else:
        print("[SKIP] OpenGL RGB-D render was explicitly skipped")

    if pack5["opengl_rgbd_render"]["validated"]:
        color_scene = build_color_task_scene(scene, cfg)
        print("[PASS] derived separate red+blue task scene")
        print("  frozen one-cube regression scene unchanged: True")
        try:
            color_task = run_pack5_color(color_scene, pack5)
        except (RuntimeError, AssertionError) as exc:
            print("[FAIL] Pack 5 v0.2 RGB-D color-target gate:", exc)
            return 1
        print("[PASS] RGB-D red/blue perception and bounded planar task")
        print("  environment:             ", color_task["gymnasium"]["environment_id"])
        print(
            "  observation shape:       ", color_task["contract"]["observation_shape"]
        )
        print(
            "  policy RGB-D resolution: ",
            color_task["rgbd_perception"]["resolution_px"],
        )
        print(
            "  raw images sent to PPO:  ",
            color_task["contract"]["perception"]["policy_receives_raw_images"],
        )
        print(
            "  planar action shape:     ",
            color_task["contract"]["action"]["normalized_action"]["shape"],
        )
        print(
            "  maximum TCP step(m):     ",
            color_task["contract"]["action"]["maximum_euclidean_increment_m"],
        )
        print(
            "  scripted combinations:   ",
            len(color_task["scripted_reference_episodes"]),
        )
        print(
            "  scripted successes:      ",
            sum(
                int(item["success"])
                for item in color_task["scripted_reference_episodes"]
            ),
        )
        verification_level = color_task["verification_level"]
        report_name = "CR10_2F85_RGBD_COLOR_TARGET_PLANAR_PUSH_ENV.json"
    else:
        verification_level = pack5["verification_level"]
        report_name = "CR10_2F85_D405_WRIST_CAMERA_GEOMETRY_ONLY.json"

    print(f"\n=== VERIFIED LEVEL: {verification_level} ===")
    print("Report:", ROOT / "verification" / report_name)
    if not pack5["opengl_rgbd_render"]["validated"]:
        print(
            "IMPORTANT: this run did not validate actual RGB/depth rendering; rerun without --skip-camera-render on the workstation."
        )
        print(
            "IMPORTANT: the v0.2 color-target environment was not run because it consumes actual RGB and depth."
        )
    else:
        print(
            "NOTE: PPO receives compact RGB-D-derived features; this is not end-to-end raw-image RL."
        )
        print(
            "NOTE: the gripper command remains fixed and position-like; this pack does not implement force control."
        )
    print(
        "IMPORTANT: the camera wrist transform is provisional, not a physical calibration."
    )
    print(
        "IMPORTANT: nominal 2F-85 TCP = [0,0,0.174] m; physical lab mount/index orientation remains unverified."
    )
    print("No physical motion was commanded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
