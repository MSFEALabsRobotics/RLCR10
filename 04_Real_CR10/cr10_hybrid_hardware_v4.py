import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from arm_bridge.core import SafetyFault, OBS_LOW, OBS_HIGH
from arm_bridge.perception import D405, find_colors
from arm_bridge.policy import Policy
from arm_bridge.ros_arm import ROSArm, parse_payload

ROOT = Path(__file__).resolve().parent
STEP_MM = 0.5
PATH_STEP_MM = 2.0
MAX_PPO_FORWARD_MM = 190.0
MAX_PPO_LATERAL_MM = 125.0
STROKE_MM = 100.0
HANDOVER_GAP_MM = 45.0
ALIGN_TOL_MM = 10.0
EDGE_PX = 8


def read_json(name):
    return json.loads((ROOT / name).read_text())


def rotation_error(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def alarms(arm):
    data = arm.call('GetErrorID').robot_return.strip()
    if not data.startswith('{') or not data.endswith('}'):
        raise SafetyFault('Cannot read robot alarms')
    values = json.loads(data[1:-1])
    if not isinstance(values, list) or len(values) != 7 or any(values):
        raise SafetyFault('Active alarm: ' + data)


def pose(arm, home, tolerance_mm=None):
    arm.ensure_idle()
    alarms(arm)
    actual = arm.raw_flange_pose_read_only()
    if max(abs(rotation_error(actual[3:], home[3:]))) > 0.5:
        raise SafetyFault('Wrist orientation differs from recorded HOME')
    if abs(actual[2] - home[2]) > 1.0:
        raise SafetyFault('Height differs from recorded HOME')
    if tolerance_mm is not None and np.linalg.norm(actual[:3] - home[:3]) > tolerance_mm:
        raise SafetyFault('Robot is not at recorded HOME')
    return actual


def camera_geometry(measurements, capture):
    r = np.asarray(measurements['camera_rotation_candidate'], float)
    f = np.asarray(measurements['candidate_forward_axis_base'], float)
    l = np.asarray(measurements['candidate_lateral_axis_base'], float)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)) or not np.allclose(r.T @ r, np.eye(3), atol=.01):
        raise SafetyFault('Invalid measured camera rotation')
    if not np.allclose(f, [0, 1, 0], atol=.01) or not np.allclose(l, [-1, 0, 0], atol=.01):
        raise SafetyFault('Measured robot axes differ from tested axes')
    red = np.asarray(capture['detections']['red']['optical_point_m'], float)
    gap = float(capture['geometry']['approx_home_plate_to_box_face_m'])
    forward_offset = gap - f @ (r @ red)
    if not 0.03 < forward_offset < 0.15:
        raise SafetyFault('Estimated camera-to-plate forward offset implausible')
    return r, f, l, float(forward_offset)


def vision(camera, cfg, color, progress_mm, r, f, l, forward_offset):
    frame = camera.read()
    found = find_colors(frame, cfg['camera'], rs_module=camera.rs)
    values = [float(np.dot(f, progress_mm)) / 1000.0, float(np.dot(l, progress_mm)) / 1000.0]
    info = {}
    width, height = cfg['camera']['policy_resolution_px']
    for name in ('red', 'blue'):
        item = found.get(name)
        if item is None:
            info[name] = {'visible': False}
            values += [0.0, 0.0, 0.0, 0.0]
            continue
        p = np.asarray(item['point_optical_m'], float)
        u, v = np.asarray(item['uv_policy_px'], float)
        depth = float(item['depth_m'])
        valid = (p.shape == (3,) and np.isfinite(p).all() and
                 EDGE_PX < u < width - EDGE_PX and EDGE_PX < v < height - EDGE_PX and
                 .16 <= depth <= .5)
        if not valid:
            info[name] = {'visible': False, 'uv': [float(u), float(v)]}
            values += [0.0, 0.0, 0.0, 0.0]
            continue
        relative = r @ p
        forward_mm = 1000.0 * (f @ relative + forward_offset)
        lateral_mm = 1000.0 * (l @ relative)
        info[name] = {'visible': True, 'gap_est_mm': round(float(forward_mm), 1),
                      'lateral_est_mm': round(float(lateral_mm), 1),
                      'depth_mm': round(depth * 1000, 1), 'uv': [float(u), float(v)]}
        values += [float(forward_mm) / 1000.0, float(lateral_mm) / 1000.0, depth, 1.0]
    values += [float(color == 'red'), float(color == 'blue')]
    obs = np.asarray(values, np.float32)
    if not np.isfinite(obs).all():
        raise SafetyFault('Invalid D405 observation')
    obs = np.clip(obs, OBS_LOW, OBS_HIGH)
    if not info[color].get('visible'):
        raise SafetyFault('Selected box no longer reliably visible')
    return obs, info, frame.frame_number


def move(arm, current, delta_mm, home, limit_mm):
    delta = np.asarray(delta_mm, float)
    if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > limit_mm + 1e-6:
        raise SafetyFault('Movement exceeds step limit')
    current = pose(arm, home)
    expected = current[:3] + delta
    arm.call('RelMovLUser', {'a': float(delta[0]), 'b': float(delta[1]), 'c': float(delta[2]),
                            'd': 0.0, 'e': 0.0, 'f': 0.0, 'param_value': []}, timeout_s=5.0)
    until = time.monotonic() + 12.0
    stable = 0
    while time.monotonic() < until:
        mode = arm.robot_mode()
        if mode not in (5, 7):
            raise SafetyFault('Robot mode changed: ' + str(mode))
        actual = arm.raw_flange_pose_read_only()
        if max(abs(rotation_error(actual[3:], home[3:]))) > 0.5 or abs(actual[2] - expected[2]) > 1.0:
            raise SafetyFault('Unexpected wrist or height motion')
        if mode == 5 and np.linalg.norm(actual[:3] - expected) < .3:
            stable += 1
            if stable == 2:
                return actual
        else:
            stable = 0
        time.sleep(.1)
    raise SafetyFault('Motion tracking timeout')


def path_segments(start, end, length=PATH_STEP_MM):
    length_mm = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
    n = max(1, int(math.ceil(length_mm / length)))
    return [np.asarray(start) + (np.asarray(end) - np.asarray(start)) * (k / n) for k in range(1, n + 1)]


def preflight_ik(arm, start, targets, home, limits):
    if limits is None:
        raise SafetyFault('Full physical push requires verified robot joint limits')
    near = arm.angles()
    lowhigh = np.asarray(limits, float)
    if lowhigh.shape != (6, 2) or not np.isfinite(lowhigh).all():
        raise SafetyFault('Invalid joint limits')
    last = np.asarray(start, float)
    for q in targets:
        if np.linalg.norm(q - last) > PATH_STEP_MM + .02:
            raise SafetyFault('Waypoints exceed 2 mm')
        data = dict(zip(('x','y','z','rx','ry','rz'), [float(x) for x in q] + [float(x) for x in home[3:]]))
        data.update({'use_joint_near': '1', 'joint_near': '{' + ','.join(f'{x:.5f}' for x in near) + '}', 'user': '0', 'tool': '0'})
        joint = parse_payload(arm.call('InverseKin', data, timeout_s=4.0).robot_return, 6)
        if np.any(joint <= lowhigh[:, 0] + 5.0) or np.any(joint >= lowhigh[:, 1] - 5.0):
            raise SafetyFault('IK joint limit margin failed')
        if max(abs(rotation_error(joint, near))) > 2.0:
            raise SafetyFault('IK joint step exceeds 2 degrees')
        near = joint
        last = q


def run_cycle(color, arm, camera, policy, cfg, measurements, capture, execute, steps, dry_push):
    home = np.asarray(measurements['home']['flange_pose_mm_deg'], float)
    r, f, l, offset = camera_geometry(measurements, capture)
    print('Estimated camera-forward correction (mm):', round(1000.0 * offset, 1), 'from approximate 280 mm plate gap')
    print('This estimate is not an independently measured contact transform')
    current = pose(arm, home, tolerance_mm=5.0)
    path = [current[:3].copy()]
    if execute and input('Clear PPO approach region? Type MOVE: ').strip() != 'MOVE':
        print('Cancelled')
        return
    for i in range(steps):
        current = pose(arm, home)
        progress_mm = current[:3] - home[:3]
        obs, info, frame = vision(camera, cfg, color, progress_mm, r, f, l, offset)
        target = info[color]
        gap = target['gap_est_mm']
        side = target['lateral_est_mm']
        if 0.0 < gap <= HANDOVER_GAP_MM and abs(side) <= ALIGN_TOL_MM:
            print('HANDOVER ESTIMATE:', json.dumps({'gap_mm': gap, 'side_mm': side}))
            break
        if gap <= 0:
            raise SafetyFault('Estimated contact passed; stop')
        if abs(float(f @ progress_mm)) >= MAX_PPO_FORWARD_MM - STEP_MM or abs(float(l @ progress_mm)) >= MAX_PPO_LATERAL_MM - STEP_MM:
            print('STOP: trained PPO workspace boundary reached')
            break
        action = policy.predict(obs)
        scaled = np.asarray(action, float) / max(1.0, float(np.linalg.norm(action))) * STEP_MM
        delta = f * scaled[0] + l * scaled[1]
        next_progress = progress_mm + delta
        if not 0.0 <= f @ next_progress <= MAX_PPO_FORWARD_MM or abs(l @ next_progress) > MAX_PPO_LATERAL_MM:
            print('STOP: next PPO action outside trained workspace')
            break
        print(json.dumps({'step': i, 'frame': frame, 'target': color, 'box': target,
                          'ppo': np.round(action, 4).tolist(), 'robot_delta_mm': np.round(delta, 4).tolist()}), flush=True)
        if not execute:
            print('PREVIEW ONLY; no robot movement')
            return
        current = move(arm, current, delta, home, STEP_MM)
        path.append(current[:3].copy())
    else:
        print('PPO step limit reached')
    if execute and not dry_push:
        print('FIXED PUSH NOT EXECUTED: estimated contact location alone is insufficient for a guarded 100 mm stroke')
    if dry_push:
        current = pose(arm, home)
        obs, info, _ = vision(camera, cfg, color, current[:3] - home[:3], r, f, l, offset)
        gap = info[color]['gap_est_mm']
        side = info[color]['lateral_est_mm']
        contact = current[:3] + f * max(0., gap)
        end = contact + f * STROKE_MM
        plan = {'estimated_gap_mm': gap, 'estimated_lateral_mm': side,
                'contact_flange_xyz_mm': np.round(contact, 2).tolist(),
                'push_end_flange_xyz_mm': np.round(end, 2).tolist(),
                'stroke_mm': STROKE_MM, 'physical_motion': False}
        print('CANDIDATE PUSH PLAN:', json.dumps(plan), flush=True)
    if execute and len(path) > 1:
        if input('Reverse verified PPO steps to HOME? Type HOME: ').strip() == 'HOME':
            for previous in reversed(path[:-1]):
                now = pose(arm, home)
                delta = previous - now[:3]
                move(arm, now, delta, home, STEP_MM + .3)
            print('HOME', np.round(pose(arm, home)[:3], 3).tolist(), flush=True)
        else:
            print('Robot remains at its current pose')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=['red', 'blue'], required=True)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--plan-push', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.steps <= 380:
        ap.error('--steps must be 1 to 380')
    cfg = read_json('config/deployment.json')
    measurement = read_json('cr10_motion_measurements.json')
    capture = read_json('cr10_current_geometry_capture.json')
    arm = None
    policy = None
    try:
        arm = ROSArm(cfg)
        policy = Policy(ROOT)
        with D405(cfg['camera']) as camera:
            run_cycle(args.target, arm, camera, policy, cfg, measurement, capture,
                      args.execute, args.steps, args.plan_push)
    except (Exception, KeyboardInterrupt) as e:
        print('STOP:', e, flush=True)
        if arm is not None:
            arm.stop_best_effort()
        raise SystemExit(2)
    finally:
        if policy is not None:
            policy.close()
        if arm is not None:
            arm.close()


if __name__ == '__main__':
    main()
