import argparse
import json
import re
import signal
import time
from pathlib import Path

import numpy as np

from arm_bridge.core import SafetyFault
from arm_bridge.perception import D405, find_colors
from arm_bridge.ros_arm import ROSArm, parse_payload

ROOT = Path(__file__).resolve().parent
REFERENCE = np.array([166.3857, 620.9493, -58.2358, 88.48, 3.5859, -173.5134])
JOINT_REFERENCE = np.array([91.0986, -42.3739, -128.0878, -8.2381, -95.4285, 356.4184])
POSITION_TOL = 0.05
ANGLE_TOL = 0.05
STEP_MM = 1.0
OFFSET_MM = 10.0
RADIUS_MM = 10.2
PATH_BUDGET_MM = 42.0
JOINT_STEP_DEG = 0.25
JOINT_REGION_DEG = 2.0


def check_reply(reply, label):
    text = str(getattr(reply, 'robot_return', '')).strip()
    match = re.match(r'^\s*(-?\d+)\s*,', text)
    if match and int(match.group(1)) != 0:
        raise SafetyFault(f'{label}: {text}')


def check_state(pose, joints):
    if pose.shape != (6,) or joints.shape != (6,) or not np.all(np.isfinite(np.r_[pose, joints])):
        raise SafetyFault('Invalid robot feedback')
    if np.linalg.norm(pose[:3] - REFERENCE[:3]) > RADIUS_MM:
        raise SafetyFault('Outside the fixed reference region; no repositioning')
    if abs(pose[2] - REFERENCE[2]) > POSITION_TOL:
        raise SafetyFault('Flange height changed')
    angles = (pose[3:] - REFERENCE[3:] + 180) % 360 - 180
    if np.max(np.abs(angles)) > ANGLE_TOL:
        raise SafetyFault('Wrist orientation changed')
    if np.max(np.abs(joints - JOINT_REFERENCE)) > JOINT_REGION_DEG:
        raise SafetyFault('Joint posture outside the reference region')


def read_state(arm, idle=True):
    mode = arm.robot_mode()
    if mode not in ((5,) if idle else (5, 7)):
        raise SafetyFault(f'Unexpected robot mode: {mode}')
    reply = arm.call('GetErrorID')
    check_reply(reply, 'GetErrorID')
    text = reply.robot_return.strip()
    if not (text.startswith('{') and text.endswith('}')):
        raise SafetyFault('Malformed alarm reply')
    errors = json.loads(text[1:-1])
    if not isinstance(errors, list) or len(errors) != 7 or any(errors):
        raise SafetyFault(f'Robot alarms: {text}')
    pose = arm.raw_flange_pose_read_only()
    joints = arm.angles()
    check_state(pose, joints)
    return pose, joints, mode


def require_services(arm):
    required = {
        'GetPose': ('user', 'tool'), 'GetAngle': (), 'RobotMode': (),
        'GetErrorID': (), 'Stop': (), 'SpeedFactor': ('ratio',),
        'InverseKin': ('x', 'y', 'z', 'rx', 'ry', 'rz', 'user', 'tool',
                       'use_joint_near', 'joint_near'),
        'MovL': ('a', 'b', 'c', 'd', 'e', 'f', 'mode', 'param_value'),
    }
    for label, fields in required.items():
        srv = getattr(arm.srv, label, None)
        if srv is None or any(not hasattr(srv.Request(), key) for key in fields):
            raise SafetyFault(f'ROS interface mismatch: {label}')
        if label not in arm.clients:
            client = arm.node.create_client(srv, f'{arm.prefix}/{label}')
            if not client.wait_for_service(timeout_sec=2):
                raise SafetyFault(f'Service unavailable: {label}')
            arm.clients[label] = client


def check_ik(arm, before, joints, target):
    check_state(target, joints)
    if np.linalg.norm(target[:3] - before[:3]) > STEP_MM + 1e-8:
        raise SafetyFault('Requested step exceeds 1 mm')
    args = dict(zip(('x', 'y', 'z', 'rx', 'ry', 'rz'), map(float, target)))
    args.update(user='0', tool='0', use_joint_near='1',
                joint_near='{' + ','.join(f'{v:.6f}' for v in joints) + '}')
    predicted = parse_payload(arm.call('InverseKin', args).robot_return, 6)
    if np.max(np.abs(predicted - joints)) > JOINT_STEP_DEG:
        raise SafetyFault('IK joint change exceeds 0.25 degree')
    check_state(target, predicted)
    return predicted


def move_step(arm, before, joints, target):
    check_ik(arm, before, joints, target)
    now, now_joints, _ = read_state(arm)
    if np.linalg.norm(now[:3] - before[:3]) > 0.01 or np.max(np.abs(now_joints - joints)) > 0.02:
        raise SafetyFault('Robot moved after planning')
    fields = dict(zip(('a', 'b', 'c', 'd', 'e', 'f'), map(float, target)))
    fields.update(mode=False, param_value=['user=0', 'tool=0', 'speed=5', 'a=1', 'cp=0'])
    check_reply(arm.call('MovL', fields, timeout_s=3), 'MovL')
    delta = target[:3] - before[:3]
    length = float(np.linalg.norm(delta))
    direction = delta / length
    deadline, stable = time.monotonic() + 30, 0
    while time.monotonic() < deadline:
        pose, actual_joints, mode = read_state(arm, idle=False)
        if np.max(np.abs(actual_joints - joints)) > JOINT_STEP_DEG + 0.01:
            raise SafetyFault('Measured joint change too large')
        travelled = pose[:3] - before[:3]
        along = float(travelled @ direction)
        across = float(np.linalg.norm(travelled - along * direction))
        if along < -POSITION_TOL or along > length + POSITION_TOL or across > POSITION_TOL:
            raise SafetyFault('Motion departed from the short planned segment')
        done = mode == 5 and np.linalg.norm(pose[:3] - target[:3]) <= POSITION_TOL
        stable = stable + 1 if done else 0
        if stable >= 3:
            return pose
        time.sleep(0.08)
    raise SafetyFault('Motion timeout; no further motion will be requested')


def next_target(pose, waypoint):
    remaining = waypoint[:3] - pose[:3]
    if np.linalg.norm(remaining) <= POSITION_TOL:
        return None
    target = waypoint.copy()
    vertical = waypoint[2] - pose[2]
    allowance = np.sqrt(max(0.0, STEP_MM**2 - vertical**2))
    xy = remaining[:2]
    target[:2] = pose[:2] + xy * min(1.0, allowance / max(float(np.linalg.norm(xy)), 1e-12))
    return target


def snapshot(arm, camera, cfg, label):
    before, _, _ = read_state(arm)
    full = dict(cfg['camera'])
    full['policy_resolution_px'] = list(full['resolution_px'])
    full['min_mask_pixels'] = int(cfg['camera']['min_mask_pixels']) * 100
    samples = []
    for _ in range(5):
        camera.read()
    for _ in range(21):
        frame = camera.read()
        detections = find_colors(frame, full, rs_module=camera.rs)
        item = {'frame': int(frame.frame_number), 'objects': {}}
        for color in ('red', 'blue'):
            detection = detections.get(color)
            if detection is None:
                raise SafetyFault(f'{color} detection missing during {label}')
            point = np.asarray(detection['point_optical_m'], float)
            if point.shape != (3,) or not np.all(np.isfinite(point)) or point[2] < 0.15:
                raise SafetyFault('Both boxes must remain visible, at least 150 mm from the camera')
            item['objects'][color] = point.tolist()
        samples.append(item)
    after, _, _ = read_state(arm)
    if np.linalg.norm(after[:3] - before[:3]) > POSITION_TOL:
        raise SafetyFault('Robot moved during camera measurement')
    objects, spread = {}, {}
    for color in ('red', 'blue'):
        points = np.array([sample['objects'][color] for sample in samples])
        objects[color] = np.median(points, axis=0).tolist()
        spread[color] = (np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)).tolist()
    result = {'label': label, 'flange_pose_mm_deg': after.tolist(),
              'optical_points_m': objects, 'spread_p95_minus_p05_m': spread, 'frames': samples}
    print(json.dumps({key: value for key, value in result.items() if key != 'frames'}), flush=True)
    return result


def estimate(samples):
    origin = samples[0]
    source, destination = [], []
    for point in (samples[1], samples[3]):
        displacement = (np.array(point['flange_pose_mm_deg'][:3]) - origin['flange_pose_mm_deg'][:3]) / 1000
        for color in ('red', 'blue'):
            source.append(np.array(point['optical_points_m'][color]) - origin['optical_points_m'][color])
            destination.append(-displacement)
    x, y = np.array(source), np.array(destination)
    result = {'translation_determined': False, 'approved_for_motion': False}
    singular = np.linalg.svd(x, compute_uv=False)
    result['measured_change_singular_values_m'] = singular.tolist()
    if singular[1] <= 1e-6:
        result['rotation_candidate'] = None
        return result
    u, _, vt = np.linalg.svd(x.T @ y)
    rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    result['rotation_candidate'] = rotation.tolist()
    result['fit_residual_m'] = float(np.sqrt(np.mean(np.sum((x @ rotation.T - y)**2, axis=1))))
    result['return_camera_discrepancy_m'] = {
        sample['label']: {color: float(np.linalg.norm(np.array(sample['optical_points_m'][color]) - origin['optical_points_m'][color]))
                          for color in ('red', 'blue')}
        for sample in (samples[2], samples[4])
    }
    return result


def run(arm, camera, cfg, execute, report, save, confirm=input):
    start, joints, _ = read_state(arm)
    if np.linalg.norm(start[:3] - REFERENCE[:3]) > POSITION_TOL or np.max(np.abs(joints - JOINT_REFERENCE)) > 0.03:
        raise SafetyFault('Start pose differs from the supplied pose; no automatic repositioning')
    waypoints = []
    for label, xy in (('X_PLUS_10', (OFFSET_MM, 0)), ('RETURN_AFTER_X', (0, 0)),
                      ('Y_PLUS_10', (0, OFFSET_MM)), ('RETURN_AFTER_Y', (0, 0))):
        point = start.copy()
        point[:2] += xy
        waypoints.append((label, point))
    for _, waypoint in waypoints:
        if np.linalg.norm(waypoint[:3] - REFERENCE[:3]) + POSITION_TOL > RADIUS_MM:
            raise SafetyFault('Planned waypoint exceeds the fixed region')
    report['planned_waypoints_mm_deg'] = {name: point.tolist() for name, point in waypoints}
    print('NO-CONTACT CAMERA MEASUREMENT: +X 10 mm, return, +Y 10 mm, return.', flush=True)
    print('Z and wrist orientation held; <=1 mm per command; <=42 mm total path.', flush=True)
    report['samples'].append(snapshot(arm, camera, cfg, 'START'))
    save()
    if not execute:
        report['status'] = 'read_only_complete'
        save()
        print('READ-ONLY preview complete. No motion commands sent.', flush=True)
        return
    print('Verify the entire planned sweep is clear for arm, plate, mount and cable.', flush=True)
    print('Keep boxes stationary, people clear, controller protections active and E-stop ready.', flush=True)
    if confirm('Type CLEAR for this no-contact measurement, or Enter to cancel: ').strip() != 'CLEAR':
        report['status'] = 'cancelled'
        save()
        print('Cancelled.', flush=True)
        return
    fresh, fresh_joints, _ = read_state(arm)
    if np.linalg.norm(fresh[:3] - start[:3]) > POSITION_TOL or np.max(np.abs(fresh_joints - joints)) > 0.02:
        raise SafetyFault('Robot moved during confirmation')
    report['samples'][0] = snapshot(arm, camera, cfg, 'START')
    check_reply(arm.call('SpeedFactor', {'ratio': 1}), 'SpeedFactor')
    path, commands = 0.0, 0
    for label, waypoint in waypoints:
        while True:
            pose, current_joints, _ = read_state(arm)
            target = next_target(pose, waypoint)
            if target is None:
                break
            length = float(np.linalg.norm(target[:3] - pose[:3]))
            if path + length > PATH_BUDGET_MM or commands >= 48:
                raise SafetyFault('Calibration movement budget exhausted')
            path += length
            commands += 1
            report['commands'].append({'target_mm_deg': target.tolist(), 'commanded_path_mm': path, 'status': 'pending'})
            save()
            actual = move_step(arm, pose, current_joints, target)
            report['commands'][-1].update(status='completed', measured_mm_deg=actual.tolist())
            save()
        report['samples'].append(snapshot(arm, camera, cfg, label))
        save()
    report['estimate'] = estimate(report['samples'])
    report['status'] = 'completed'
    save()
    print(json.dumps({'result': 'MEASUREMENTS_RECORDED', **report['estimate']}), flush=True)
    print('Returned to start. Deployment config unchanged. This does not determine camera translation or plate TCP.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'config/deployment.json').read_text())
    path = ROOT / f'cr10_translation_measurements_{time.time_ns()}.json'
    report = {'reference_mm_deg': REFERENCE.tolist(), 'execute': args.execute,
              'status': 'started', 'approved_for_motion': False, 'samples': [], 'commands': []}

    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')

    def interrupted(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    arm = None
    try:
        arm = ROSArm(cfg)
        require_services(arm)
        with D405(cfg['camera']) as camera:
            run(arm, camera, cfg, args.execute, report, save)
    except (Exception, KeyboardInterrupt) as error:
        if arm is not None and args.execute:
            arm.stop_best_effort()
        report.update(status='stopped', error=f'{type(error).__name__}: {error}')
        save()
        print(f'[STOP] {report["error"]}; no automatic return.', flush=True)
        raise SystemExit(1)
    finally:
        if arm is not None:
            arm.close()
        print(f'Measurements: {path}', flush=True)


if __name__ == '__main__':
    main()
