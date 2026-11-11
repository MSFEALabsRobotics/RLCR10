import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from arm_bridge.ros_arm import ROSArm

ROOT = Path(__file__).resolve().parent
TRACE = ROOT / 'cr10_pose0_trace.json'
MAX_STEP_MM = 8.0
LEGACY_START = np.array([191.737, 661.957, -27.159, 91.33, 3.48, -177.69], dtype=float)
LEGACY_PPO = np.array([
    [-3.789, 2.171], [-3.640, 2.368], [-3.659, 2.418],
    [-3.657, 2.492], [-3.542, 2.677], [-3.473, 2.814],
    [-3.449, 2.918], [-3.394, 3.042], [-3.245, 3.287]
], dtype=float)


def angles_delta(a, b):
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0


def write_trace(data):
    temp = TRACE.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temp, TRACE)


def record(home):
    home = np.asarray(home, dtype=float)
    if home.shape != (6,) or not np.isfinite(home).all():
        raise RuntimeError('Invalid session START')
    write_trace({'pose0': home.tolist(), 'poses': [home.tolist()]})
    print('POSE 0 RECORDED:', np.round(home, 3).tolist(), flush=True)


def log(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise RuntimeError('Invalid robot position for return trace')
    if not TRACE.exists():
        raise RuntimeError('Pose-0 trace does not exist')
    data = json.loads(TRACE.read_text())
    previous = np.asarray(data['poses'][-1], dtype=float)
    if np.max(np.abs(angles_delta(pose[3:], previous[3:]))) > 0.7:
        raise RuntimeError('Wrist changed since previous recorded move')
    if np.linalg.norm(pose[:3] - previous[:3]) > 10.0:
        raise RuntimeError('Unexpected jump in recorded robot trajectory')
    data['poses'].append(pose.tolist())
    write_trace(data)


def install(name):
    src = ROOT / name
    if not src.is_file():
        raise RuntimeError(f'File not found: {src}')
    content = src.read_text()
    if 'import cr10_pose0' in content:
        raise RuntimeError('Controller already instrumented')
    expected = (
        'from arm_bridge.ros_arm import ROSArm, parse_payload',
        '    home = current.copy()',
        '            if consecutive >= 2:\n                return now'
    )
    for line in expected:
        if content.count(line) != 1:
            raise RuntimeError('Controller layout differs; nothing changed: ' + line[:55])
    content = content.replace(expected[0], expected[0] + '\nimport cr10_pose0', 1)
    content = content.replace(expected[1], expected[1] + '\n    cr10_pose0.record(home)', 1)
    content = content.replace(expected[2], '            if consecutive >= 2:\n                cr10_pose0.log(now)\n                return now', 1)
    out = src.with_name(src.stem + '_pose0.py')
    import ast
    ast.parse(content)
    out.write_text(content)
    print('CREATED:', out.name)
    print('This version logs every completed robot move for a reusable return command.')


def legacy_path():
    p = LEGACY_START[:3].copy()
    route = [p.copy()]
    for delta in LEGACY_PPO:
        p = p + np.array([delta[0], delta[1], 0.0])
        route.append(p.copy())
    p = p + np.array([-77.5, 0.0, 0.0])
    route.append(p.copy())
    p = p + np.array([0.0, 97.423, 0.0])
    route.append(p.copy())
    return LEGACY_START, np.asarray(route)


def candidate_paths():
    paths = []
    if TRACE.exists():
        try:
            d = json.loads(TRACE.read_text())
            pose0 = np.asarray(d['pose0'], dtype=float)
            poses = np.asarray(d['poses'], dtype=float)
            if pose0.shape == (6,) and poses.ndim == 2 and poses.shape[1] == 6 and len(poses) >= 1:
                paths.append(('recorded session', pose0, poses[:, :3]))
        except (ValueError, TypeError, KeyError):
            pass
    home, route = legacy_path()
    paths.append(('previous blue-run path', home, route))
    return paths


def locate(route, xyz):
    xyz = np.asarray(xyz, dtype=float)
    if len(route) == 1:
        return float(np.linalg.norm(xyz-route[0])), 0, route[0].copy()
    best = (float('inf'), -1, None)
    for i, (a, b) in enumerate(zip(route[:-1], route[1:])):
        d = b-a
        denom = float(d@d)
        t = 0.0 if denom < 1e-12 else float(np.clip((xyz-a)@d / denom, 0, 1))
        projected = a + d*t
        error = float(np.linalg.norm(projected-xyz))
        if error < best[0] - 1e-6 or (abs(error-best[0]) <= 1e-6 and i > best[1]):
            best = (error, i, projected)
    return best


def get_pose(arm):
    a = np.asarray(arm.raw_flange_pose_read_only(), dtype=float)
    if a.shape != (6,) or not np.isfinite(a).all():
        raise RuntimeError('Robot pose invalid')
    return a


def check_idle(arm):
    mode = arm.robot_mode()
    if mode != 5:
        raise RuntimeError(f'RobotMode {mode}, expected 5')
    raw = arm.call('GetErrorID').robot_return.strip()
    if not raw.startswith('{') or not raw.endswith('}'):
        raise RuntimeError('Could not read robot alarms')
    alarms = json.loads(raw[1:-1])
    if not isinstance(alarms, list) or len(alarms) != 7 or any(alarms):
        raise RuntimeError(f'Active robot alarms: {raw}')


def move_to(arm, target, home):
    start = get_pose(arm)
    target = np.asarray(target, dtype=float)
    count = max(1, int(math.ceil(np.linalg.norm(target - start[:3]) / MAX_STEP_MM)))
    for k in range(1, count+1):
        q = start[:3] + (target - start[:3]) * (k / count)
        check_idle(arm)
        before = get_pose(arm)
        if np.max(np.abs(angles_delta(before[3:], home[3:]))) > 0.7:
            raise RuntimeError('Wrist orientation changed')
        delta = q - before[:3]
        if np.linalg.norm(delta) > MAX_STEP_MM + 0.5:
            raise RuntimeError('Return step exceeds expected distance')
        fields = dict(zip(('a','b','c','d','e','f'), [float(x) for x in delta] + [0.0, 0.0, 0.0]))
        fields['param_value'] = []
        arm.call('RelMovLUser', fields, timeout_s=5.0)
        end = time.monotonic() + 20.0
        stable = 0
        while time.monotonic() < end:
            mode = arm.robot_mode()
            if mode not in (5,7):
                raise RuntimeError(f'Unexpected robot mode {mode}')
            measured = get_pose(arm)
            if np.max(np.abs(angles_delta(measured[3:], home[3:]))) > 0.7:
                raise RuntimeError('Wrist moved unexpectedly')
            if mode == 5 and np.linalg.norm(measured[:3]-q) < 0.5:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            time.sleep(0.08)
        else:
            raise RuntimeError('Robot motion tracking timed out')


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--install', metavar='CONTROLLER_PY')
    action.add_argument('--return', dest='return_pose0', action='store_true')
    action.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.install:
        install(args.install)
        return
    cfg = json.loads((ROOT/'config/deployment.json').read_text())
    arm = ROSArm(cfg)
    try:
        check_idle(arm)
        current = get_pose(arm)
        options = []
        for name, home, route in candidate_paths():
            if np.max(np.abs(angles_delta(current[3:], home[3:]))) > 0.7:
                continue
            error, i, projected = locate(route, current[:3])
            options.append((error, name, home, route, i))
        print('CURRENT:', np.round(current, 3).tolist(), flush=True)
        if not options:
            raise RuntimeError('No saved return path matches this wrist orientation')
        options.sort(key=lambda x: x[0])
        error, name, home, route, i = options[0]
        print('SAVED POSE 0:', np.round(home, 3).tolist(), flush=True)
        print('RECORDED PATH:', name, 'distance off path (mm):', round(error, 3), flush=True)
        if error > 3.0:
            raise RuntimeError('Current robot position is not on a recorded return route; no blind HOME move')
        if args.status:
            print('RETURN READY: pose is on recorded path')
            return
        if np.linalg.norm(current[:3]-home[:3]) < 1.0:
            print('ALREADY AT POSE 0')
            return
        print('The plate must be free and the recorded return path must still be unobstructed.', flush=True)
        if input('Type RETURN to retrace recorded path: ').strip() != 'RETURN':
            print('CANCELLED')
            return
        for j in range(i, -1, -1):
            move_to(arm, route[j], home)
            print(f'RETURN {i-j+1}/{i+1}:', np.round(get_pose(arm)[:3], 2).tolist(), flush=True)
        final = get_pose(arm)
        if np.linalg.norm(final[:3]-home[:3]) > 1.0:
            raise RuntimeError('Return ended outside pose-0 tolerance')
        print('RETURNED TO POSE 0:', np.round(final, 3).tolist(), flush=True)
    except (Exception, KeyboardInterrupt) as e:
        print('STOP:', e, flush=True)
        try:
            arm.stop_best_effort()
        except Exception:
            pass
        raise SystemExit(2)
    finally:
        try:
            arm.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
