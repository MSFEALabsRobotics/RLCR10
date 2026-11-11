#!/usr/bin/env python3
"""D405 -> frozen PPO -> CR10 closed loop, using calibrated deployment.json.

The existing bridge's physical gates are mandatory. No camera-axis guesses,
no automatic robot enable, no physical motion without independently verified
hand-eye, tool frame, collision geometry, and workspace.

Usage:
  python3 cr10_d405_continuous_ppo_guarded.py --target red --steps 20
  python3 cr10_d405_continuous_ppo_guarded.py --target red --steps 20 --execute
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from arm_bridge.core import (
    SafetyFault, OBS_LOW, OBS_HIGH, axes, camera_transform, checked_static_preflight,
    observation, proposed_command, check_wrong_cube_clearance,
)
from arm_bridge.perception import D405, find_colors
from arm_bridge.policy import Policy
from arm_bridge.ros_arm import ROSArm

ROOT = Path(__file__).resolve().parent

def alarm_check(arm):
    reply = arm.call('GetErrorID').robot_return.strip()
    if not (reply.startswith('{') and reply.endswith('}')):
        raise SafetyFault('Missing/invalid GetErrorID response')
    errors = json.loads(reply[1:-1])
    if not isinstance(errors, list) or len(errors) != 7 or any(errors):
        raise SafetyFault(f'Controller reports alarms: {reply}')

def read_detections(cfg, camera, T):
    frame = camera.read()
    det = find_colors(frame, cfg['camera'], T, rs_module=camera.rs)
    for color in ('red', 'blue'):
        if det[color] is None:
            raise SafetyFault('RGBD_COLOR_TARGET_LOST: ' + color)
        if not np.all(np.isfinite(det[color]['point_base_m'])):
            raise SafetyFault('Nonfinite 3D detection: ' + color)
    return frame, det

def main():
    ap = argparse.ArgumentParser(description='Guarded continuous RGB-D PPO for the real CR10')
    ap.add_argument('--target', choices=['red', 'blue'], default='red')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--config', default=str(ROOT / 'config/deployment.json'))
    ap.add_argument('--execute', action='store_true', help='Real motion, only with physical calibration approvals')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if not 1 <= args.steps <= min(155, int(cfg['task']['max_steps'])):
        ap.error('steps out of guarded range')
    print('MODE:', 'PHYSICAL_MOTION' if args.execute else 'CAMERA_AND_PPO_SHADOW_NO_MOTION', flush=True)
    # No physical movement can occur before all mandatory calibration fields are verified.
    try:
        checked_static_preflight(cfg, live=args.execute)
    except (SafetyFault, ValueError, KeyError, TypeError) as exc:
        print('[STOP/REJECT] Calibration/safety preflight:', str(exc), flush=True)
        return 2
    arm = None
    policy = None
    moved = False
    try:
        arm = ROSArm(cfg)
        arm.ensure_idle()
        alarm_check(arm)
        first_pose = arm.pose()  # Calibrated tool TCP, not bare flange.
        origin = first_pose[:3] * 0.001
        fixed_rpy = first_pose[3:].copy()
        f, lat = axes(cfg)
        policy = Policy(ROOT)
        last = {}
        initial_box = None
        with D405(cfg['camera']) as camera:
            # Hold a single camera session; a new observation follows every completed move.
            for i in range(args.steps):
                arm.ensure_idle()
                alarm_check(arm)
                pose = arm.pose()
                ang_error = (pose[3:] - fixed_rpy + 180) % 360 - 180
                if np.max(np.abs(ang_error)) > 0.5:
                    raise SafetyFault('WRIST_ORIENTATION_CHANGED')
                tcp = pose[:3] * .001
                T = camera_transform(cfg, pose)
                frame, det = read_detections(cfg, camera, T)
                for color in ('red', 'blue'):
                    point = det[color]['point_base_m']
                    if color in last:
                        dist = float(np.linalg.norm(point - last[color]))
                        if dist > float(cfg['task']['max_detection_jump_m']):
                            raise SafetyFault(f'{color}: RGBD_DETECTION_JUMP {dist:.3f}m')
                    last[color] = point.copy()
                if initial_box is None:
                    initial_box = {k: v['point_base_m'].copy() for k, v in det.items()}
                other = 'blue' if args.target == 'red' else 'red'
                progress = float(f @ (det[args.target]['point_base_m'] - initial_box[args.target]))
                wrong_forward = float(f @ (det[other]['point_base_m'] - initial_box[other]))
                wrong_lateral = float(lat @ (det[other]['point_base_m'] - initial_box[other]))
                wrong = float(np.hypot(wrong_forward, wrong_lateral))
                if wrong > float(cfg['task']['max_wrong_displacement_m']):
                    raise SafetyFault(f'WRONG_BOX_MOVED {wrong:.3f}m')
                if progress >= float(cfg['task']['goal_displacement_m']):
                    print(f'GOAL_OBSERVED {progress:.4f} m; no further commands', flush=True)
                    return 0
                command = np.array([f @ (tcp - origin), lat @ (tcp - origin)])
                obs, raw = observation(cfg, command, det, tcp, args.target)
                if np.any(raw < OBS_LOW - 1e-5) or np.any(raw > OBS_HIGH + 1e-5):
                    raise SafetyFault('OBSERVATION_OUTSIDE_TRAINING_BOUNDS')
                action = policy.predict(raw)
                next_command, target_tcp = proposed_command(cfg, action, command, origin)
                check_wrong_cube_clearance(cfg, det, origin, target_tcp, args.target)
                target_pose = np.r_[target_tcp * 1000.0, fixed_rpy]
                print(json.dumps({
                    'step': i, 'frame': frame.frame_number,
                    'target': args.target, 'progress_m': round(progress, 5),
                    'action2': action.tolist(),
                    'proposed_tcp_mm': np.round(target_tcp * 1000, 3).tolist(),
                    'motion': args.execute,
                }), flush=True)
                if not args.execute:
                    continue
                # The pre-existing bridge performs IK, joint-limit and motion tracking checks.
                arm.ensure_idle()
                alarm_check(arm)
                arm.move_and_wait(target_pose)
                moved = True
                updated = arm.pose()
                xyz_error = float(np.linalg.norm((updated[:3] - target_pose[:3]) / 1000.0))
                if xyz_error > float(cfg['robot']['max_tcp_tracking_error_m']):
                    raise SafetyFault(f'TRACKING_ERROR {xyz_error:.4f}m')
        print('MAX_STEPS_REACHED; policy stopped', flush=True)
        return 0
    except (SafetyFault, RuntimeError, KeyboardInterrupt, ValueError) as exc:
        print('[STOP/REJECT]', str(exc), flush=True)
        if args.execute and arm is not None:
            arm.stop_best_effort()
        return 2
    finally:
        if policy is not None:
            policy.close()
        if arm is not None:
            arm.close()

if __name__ == '__main__':
    raise SystemExit(main())
