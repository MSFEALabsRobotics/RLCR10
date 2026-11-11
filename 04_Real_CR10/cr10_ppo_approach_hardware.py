import argparse
import json
import time
from pathlib import Path
import numpy as np
from arm_bridge.core import SafetyFault, OBS_LOW, OBS_HIGH
from arm_bridge.perception import D405, find_colors
from arm_bridge.policy import Policy
from arm_bridge.ros_arm import ROSArm

ROOT = Path(__file__).resolve().parent
STEP_MM = 0.5
MAX_TRAVEL_MM = 10.0
MIN_DEPTH_M = 0.16
CAMERA_FORWARD_CORRECTION_M = 0.023


def angle_error(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def alarms(arm):
    text = arm.call('GetErrorID').robot_return.strip()
    if not (text.startswith('{') and text.endswith('}')):
        raise SafetyFault('Cannot interpret Dobot alarms')
    values = json.loads(text[1:-1])
    if not isinstance(values, list) or len(values) != 7 or any(values):
        raise SafetyFault('Active Dobot alarm: ' + text)


def pose_check(arm, reference, home):
    arm.ensure_idle()
    alarms(arm)
    pose = arm.raw_flange_pose_read_only()
    if not np.isfinite(pose).all():
        raise SafetyFault('Nonfinite flange pose')
    if np.linalg.norm(pose[:3] - reference[:3]) > 20.0:
        raise SafetyFault('Arm is more than 20 mm from measured reference pose')
    if max(abs(angle_error(pose[3:], reference[3:]))) > 0.5:
        raise SafetyFault('Wrist rotated from measured reference')
    if abs(pose[2] - reference[2]) > 1.0:
        raise SafetyFault('Z changed from measured reference')
    if np.linalg.norm(pose[:3] - home[:3]) > MAX_TRAVEL_MM + 0.7:
        raise SafetyFault('Approach exceeded 10 mm trial envelope')
    return pose


def read_policy(camera, cfg, policy, color, progress_mm):
    frame = camera.read()
    found = find_colors(frame, cfg['camera'], rs_module=camera.rs)
    if any(found[c] is None for c in ('red', 'blue')):
        raise SafetyFault('Red or blue target not detected')
    values = [float(progress_mm[1]) / 1000.0, float(progress_mm[0]) / -1000.0]
    for c in ('red', 'blue'):
        item = found[c]
        optical = np.asarray(item['point_optical_m'], dtype=float)
        if optical.shape != (3,) or not np.isfinite(optical).all():
            raise SafetyFault('Invalid D405 point')
        if not (MIN_DEPTH_M <= float(item['depth_m']) <= 0.50):
            raise SafetyFault('Target outside conservative depth range')
        u, v = item['uv_policy_px']
        width, height = cfg['camera']['policy_resolution_px']
        if not (8 < u < width - 8 and 8 < v < height - 8):
            raise SafetyFault('Target nearing edge of image')
        values.extend([float(optical[2]) - CAMERA_FORWARD_CORRECTION_M,
                       -float(optical[0]), float(item['depth_m']), 1.0])
    values.extend([float(color == 'red'), float(color == 'blue')])
    obs = np.asarray(values, dtype=np.float32)
    if np.any(obs < OBS_LOW - 1e-6) or np.any(obs > OBS_HIGH + 1e-6):
        raise SafetyFault('Observation outside PPO training range')
    action = policy.predict(obs)
    scaled = np.asarray(action, dtype=float) / max(1.0, float(np.linalg.norm(action))) * STEP_MM
    base_delta = np.array([-scaled[1], scaled[0], 0.0], dtype=float)
    return frame.frame_number, found, obs, action, base_delta


def move_and_confirm(arm, delta, reference, home):
    before = pose_check(arm, reference, home)
    target = before[:3] + delta
    if np.linalg.norm(target - home[:3]) > MAX_TRAVEL_MM + 1e-6:
        raise SafetyFault('Requested step exceeds 10 mm trial envelope')
    if np.linalg.norm(delta) > STEP_MM + 1e-6 or abs(delta[2]) > 1e-9:
        raise SafetyFault('Step exceeds 0.5 mm horizontal limit')
    arm.call('RelMovLUser', {'a': float(delta[0]), 'b': float(delta[1]),
                            'c': 0.0, 'd': 0.0, 'e': 0.0, 'f': 0.0,
                            'param_value': []}, timeout_s=5.0)
    deadline = time.monotonic() + 10.0
    settled = 0
    while time.monotonic() < deadline:
        mode = arm.robot_mode()
        if mode not in (5, 7):
            raise SafetyFault(f'Robot mode changed during movement: {mode}')
        actual = arm.raw_flange_pose_read_only()
        if abs(actual[2] - reference[2]) > 1.0 or max(abs(angle_error(actual[3:], reference[3:]))) > 0.5:
            raise SafetyFault('Unexpected Z or wrist movement')
        if np.linalg.norm(actual[:3] - target) < 0.25 and mode == 5:
            settled += 1
            if settled >= 2:
                break
        else:
            settled = 0
        time.sleep(0.1)
    else:
        raise SafetyFault('Movement tracking timeout')
    return pose_check(arm, reference, home)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['red', 'blue'], required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--return-home', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.steps <= 20:
        parser.error('Steps must be from 1 to 20')
    cfg = json.loads((ROOT / 'config/deployment.json').read_text())
    measurements = json.loads((ROOT / 'cr10_motion_measurements.json').read_text())
    reference = np.asarray(measurements['home']['flange_pose_mm_deg'], dtype=float)
    arm = None
    policy = None
    history = []
    try:
        policy = Policy(ROOT)
        if args.execute:
            arm = ROSArm(cfg)
            home = pose_check(arm, reference, reference)
            if input('Clear 10 mm arm path? Type MOVE to start: ').strip() != 'MOVE':
                print('CANCELLED')
                return
            pose_check(arm, reference, home)
            arm.call('SpeedFactor', {'ratio': 1})
        else:
            home = reference.copy()
        with D405(cfg['camera']) as camera:
            for step in range(args.steps):
                current = pose_check(arm, reference, home) if args.execute else home
                progress = current[:3] - home[:3]
                frame, found, obs, action, delta = read_policy(camera, cfg, policy, args.target, progress)
                print(json.dumps({'step': step, 'frame': frame, 'target': args.target,
                                  'red_depth_m': round(found['red']['depth_m'], 4),
                                  'blue_depth_m': round(found['blue']['depth_m'], 4),
                                  'ppo_action': np.round(action, 4).tolist(),
                                  'base_delta_mm': np.round(delta, 4).tolist(),
                                  'motion': args.execute}), flush=True)
                if args.execute:
                    next_pose = move_and_confirm(arm, delta, reference, home)
                    history.append(delta.copy())
                    print('actual_base_xyz_mm:', np.round(next_pose[:3], 3).tolist(), flush=True)
        if args.execute and args.return_home:
            if input('Reverse the same verified short path? Type HOME: ').strip() == 'HOME':
                for delta in reversed(history):
                    move_and_confirm(arm, -delta, reference, home)
                final = pose_check(arm, reference, home)
                if np.linalg.norm(final[:3] - home[:3]) > 1.0:
                    raise SafetyFault('Return-to-home tracking mismatch')
                print('RETURNED TO TRIAL HOME', flush=True)
        print('DONE: short PPO approach test only; fixed 100 mm push not executed', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        print('STOP:', exc, flush=True)
        if arm is not None:
            arm.stop_best_effort()
        raise SystemExit(2)
    finally:
        if arm is not None:
            arm.close()
        if policy is not None:
            policy.close()


if __name__ == '__main__':
    main()
