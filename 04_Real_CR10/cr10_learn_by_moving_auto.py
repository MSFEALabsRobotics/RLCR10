import json
import time
from pathlib import Path

import numpy as np

from arm_bridge.perception import D405
from arm_bridge.ros_arm import ROSArm
from cr10_learn_by_moving import snapshot
from cr10_live_d405_ppo_one_supervised_step import assert_no_alarms

ROOT = Path(__file__).resolve().parent
MOVEMENT_MM = 10.0
CHUNK_MM = 2.0


def ang_diff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def checked_pose(arm, origin=None):
    arm.ensure_idle()
    assert_no_alarms(arm)
    p = arm.raw_flange_pose_read_only()
    if not np.all(np.isfinite(p)):
        raise RuntimeError('Invalid robot pose')
    if origin is not None:
        if abs(p[2] - origin[2]) > 0.8:
            raise RuntimeError('Unexpected vertical movement')
        if np.max(np.abs(ang_diff(p[3:], origin[3:]))) > 0.35:
            raise RuntimeError('Unexpected wrist rotation')
    return p


def move_small(arm, delta, origin, expected_start):
    direction = np.asarray(delta, float)
    if abs(direction[2]) > 1e-9 or np.linalg.norm(direction) > MOVEMENT_MM + 1e-9:
        raise RuntimeError('Only short horizontal movements permitted')
    count = int(round(np.linalg.norm(direction) / CHUNK_MM))
    if count == 0 or count * CHUNK_MM > MOVEMENT_MM + 1e-9:
        raise RuntimeError('Invalid movement')
    increment = direction / count
    start = checked_pose(arm, origin)
    if np.linalg.norm(start[:3] - expected_start) > 1.0:
        raise RuntimeError('Robot is not at the expected calibration waypoint')
    for index in range(count):
        before = checked_pose(arm, origin)
        goal = before[:3] + increment
        arm.call('RelMovLUser', {
            'a': float(increment[0]), 'b': float(increment[1]),
            'c': 0.0, 'd': 0.0, 'e': 0.0, 'f': 0.0,
            'param_value': []
        }, timeout_s=5.0)
        deadline = time.monotonic() + 20.0
        stable = 0
        while time.monotonic() < deadline:
            p = arm.raw_flange_pose_read_only()
            mode = arm.robot_mode()
            if mode not in (5, 7):
                raise RuntimeError(f'Robot mode changed to {mode}')
            if abs(p[2] - origin[2]) > 0.8 or np.max(np.abs(ang_diff(p[3:], origin[3:]))) > 0.35:
                raise RuntimeError('Vertical position or wrist rotation changed')
            if np.linalg.norm(p[:3] - goal) < 0.5 and mode == 5:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            time.sleep(0.12)
        else:
            raise RuntimeError('Movement did not settle at target')
        checked_pose(arm, origin)
        print(f'  Completed {index + 1}/{count}', flush=True)
    end = checked_pose(arm, origin)
    if np.linalg.norm(end[:3] - start[:3] - direction) > 1.0:
        raise RuntimeError('Actual movement differs from commanded movement')
    return end


def fit(home, forward, sideways):
    poses = np.array([s['flange_pose_mm_deg'] for s in (home, forward, sideways)])
    df = (poses[1, :3] - poses[0, :3]) / 1000
    dl = (poses[2, :3] - poses[1, :3]) / 1000
    df[2] = 0.0
    dl[2] = 0.0
    if min(np.linalg.norm(df), np.linalg.norm(dl)) < 0.008:
        raise RuntimeError('Movement too small for measurement')
    f = df / np.linalg.norm(df)
    side = dl - np.dot(dl, f) * f
    if np.linalg.norm(side) < 0.008:
        raise RuntimeError('Movements are not independent')
    l = side / np.linalg.norm(side)
    measurements = {
        'home': home,
        'forward': forward,
        'sideways': sideways,
        'candidate_forward_axis_base': f.tolist(),
        'candidate_lateral_axis_base': l.tolist(),
        'approved_for_motion': False,
        'camera_rotation_candidate': None,
        'rotation_residual_m': None
    }
    optical_deltas = []
    robot_deltas = []
    for color in ('red', 'blue'):
        if color not in home['optical_points_m']:
            continue
        base_point = np.asarray(home['optical_points_m'][color], float)
        for sample in (forward, sideways):
            if color not in sample['optical_points_m']:
                continue
            optical_deltas.append(np.asarray(sample['optical_points_m'][color], float) - base_point)
            robot_deltas.append(-(np.asarray(sample['flange_pose_mm_deg'][:3]) - poses[0, :3]) / 1000)
    if len(optical_deltas) >= 2 and np.linalg.matrix_rank(optical_deltas, tol=0.002) >= 2:
        optical_deltas = np.asarray(optical_deltas)
        robot_deltas = np.asarray(robot_deltas)
        u, _, vt = np.linalg.svd(optical_deltas.T @ robot_deltas)
        r = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
        measurements['camera_rotation_candidate'] = r.tolist()
        measurements['rotation_residual_m'] = float(np.sqrt(np.mean(np.sum(((r @ optical_deltas.T).T - robot_deltas) ** 2, axis=1))))
    return measurements


def main():
    cfg = json.loads((ROOT / 'config/deployment.json').read_text())
    arm = None
    moving = False
    try:
        arm = ROSArm(cfg)
        current = checked_pose(arm)
        print('Robot mode: 5 (idle)', flush=True)
        print('Current flange:', np.round(current, 2).tolist(), flush=True)
        print('This current pose is your temporary HOME for the calibration.', flush=True)
        print('Planned BASE-frame motions: +10 mm Y, then -10 mm X; Z and wrist fixed.', flush=True)
        print('The script will reverse both movements and return to the recorded start.', flush=True)
        print('Check plate, camera, cables, and full swept arm for clearance; keep people clear.', flush=True)
        if input('Type MOVE to allow these four short calibration motions: ').strip() != 'MOVE':
            print('Cancelled. No movement sent.', flush=True)
            return
        if np.linalg.norm(checked_pose(arm)[:3] - current[:3]) > 0.5:
            raise RuntimeError('Robot moved after confirmation')
        with D405(cfg['camera']) as camera:
            home = snapshot(arm, camera, cfg)
            if not all(k in home['optical_points_m'] for k in ('red', 'blue')):
                raise RuntimeError('Both boxes must be visible at HOME')
            if min(home['optical_points_m'][c][2] for c in ('red', 'blue')) < 0.16:
                raise RuntimeError('Camera too close to a box for this uncalibrated movement')
            print('HOME:', np.round(home['flange_pose_mm_deg'], 2).tolist(), flush=True)
            arm.call('SpeedFactor', {'ratio': 2})
            moving = True
            print('Forward calibration movement (+Y)', flush=True)
            move_small(arm, [0, MOVEMENT_MM, 0], current, current[:3])
            forward = snapshot(arm, camera, cfg)
            print('FORWARD:', np.round(forward['flange_pose_mm_deg'], 2).tolist(), flush=True)
            if not all(k in forward['optical_points_m'] for k in ('red', 'blue')):
                raise RuntimeError('A box was lost after forward movement')
            print('Sideways calibration movement (-X)', flush=True)
            move_small(arm, [-MOVEMENT_MM, 0, 0], current, current[:3] + [0, MOVEMENT_MM, 0])
            sideways = snapshot(arm, camera, cfg)
            print('SIDEWAYS:', np.round(sideways['flange_pose_mm_deg'], 2).tolist(), flush=True)
            if not all(k in sideways['optical_points_m'] for k in ('red', 'blue')):
                raise RuntimeError('A box was lost after sideways movement')
            result = fit(home, forward, sideways)
            path = ROOT / 'cr10_motion_measurements.json'
            path.write_text(json.dumps(result, indent=2) + '\n')
            print('Candidate forward axis:', np.round(result['candidate_forward_axis_base'], 4).tolist(), flush=True)
            print('Candidate lateral axis:', np.round(result['candidate_lateral_axis_base'], 4).tolist(), flush=True)
            print('Camera rotation fit residual (m):', result['rotation_residual_m'], flush=True)
            print('Saved:', path, flush=True)
            print('Returning to HOME via reverse path', flush=True)
            move_small(arm, [MOVEMENT_MM, 0, 0], current, current[:3] + [-MOVEMENT_MM, MOVEMENT_MM, 0])
            move_small(arm, [0, -MOVEMENT_MM, 0], current, current[:3] + [0, MOVEMENT_MM, 0])
            end = checked_pose(arm, current)
            if np.linalg.norm(end[:3] - current[:3]) > 1.5:
                raise RuntimeError('Return-to-start error exceeds 1.5 mm')
            moving = False
            print('DONE. Robot returned to start. No PPO motion commanded.', flush=True)
            print('Results are measurements, not authorization for pushing.', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        print('STOP:', str(exc), flush=True)
        if moving and arm is not None:
            arm.stop_best_effort()
            print('Robot position may be away from HOME. Inspect before any further movement.', flush=True)
        raise SystemExit(2)
    finally:
        if arm is not None:
            arm.close()


if __name__ == '__main__':
    main()
