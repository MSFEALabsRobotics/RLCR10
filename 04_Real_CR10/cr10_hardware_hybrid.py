import argparse
import json
from pathlib import Path
import numpy as np
from arm_bridge.core import SafetyFault, OBS_LOW, OBS_HIGH, axes, camera_transform, checked_static_preflight, observation, proposed_command, check_wrong_cube_clearance
from arm_bridge.perception import D405, find_colors
from arm_bridge.policy import Policy
from arm_bridge.ros_arm import ROSArm, parse_payload

ROOT = Path(__file__).resolve().parent


def alarms(arm):
    response = arm.call('GetErrorID').robot_return.strip()
    if not (response.startswith('{') and response.endswith('}')):
        raise SafetyFault('Robot alarm response invalid')
    values = json.loads(response[1:-1])
    if not isinstance(values, list) or len(values) != 7 or any(values):
        raise SafetyFault('Active robot alarms: ' + response)


def measured(h, key, positive=True):
    value = h.get(key)
    if value is None:
        raise SafetyFault(f'Missing measured setting: {key}')
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise SafetyFault(f'Invalid setting: {key}')
    return value


def prepare(cfg, h, execute):
    checked_static_preflight(cfg, live=execute)
    for key in ('handover_gap_m', 'lateral_tolerance_m', 'vertical_tolerance_m', 'camera_edge_margin_px', 'max_contact_gap_m', 'lift_after_push_m'):
        measured(h, key)
    measured(h, 'plate_front_offset_m', False)
    measured(h, 'box_surface_to_rear_face_m', False)
    measured(h, 'box_surface_to_contact_z_m', False)
    measured(h, 'plate_tcp_to_contact_z_m', False)
    if abs(measured(h, 'push_distance_m') - .10) > 1e-9:
        raise SafetyFault('Fixed push distance must be 0.10 m')
    if not (0 < measured(h, 'segment_m') <= .002):
        raise SafetyFault('Segment length must be at most 2 mm')
    w, ht = cfg['camera']['policy_resolution_px']
    if not (0 < h['camera_edge_margin_px'] < min(w, ht) / 3):
        raise SafetyFault('Invalid camera margin')
    if h['handover_gap_m'] > h['max_contact_gap_m']:
        raise SafetyFault('Handover gap exceeds max contact gap')
    if execute and not all(h.get(k) is True for k in ('contact_geometry_verified', 'lift_and_return_clearance_verified', 'full_stroke_verified')):
        raise SafetyFault('Physical contact, stroke, and elevated return path are not validated')


def home_check(arm, home):
    current = arm.pose()
    angle = (current[3:] - home[3:] + 180) % 360 - 180
    if np.linalg.norm(current[:3] - home[:3]) > 2 or np.max(np.abs(angle)) > .5:
        raise SafetyFault('Not at session HOME')


def image_safe(det, cfg, h):
    u, v = det['uv_policy_px']
    w, ht = cfg['camera']['policy_resolution_px']
    m = h['camera_edge_margin_px']
    return m < u < w - m and m < v < ht - m


def capture(camera, arm, cfg):
    pose = arm.pose()
    T = camera_transform(cfg, pose)
    frame = camera.read()
    found = find_colors(frame, cfg['camera'], T, rs_module=camera.rs)
    if any(found[c] is None for c in ('red', 'blue')):
        raise SafetyFault('Red or blue box not detected')
    return pose, found


def line(a, b, step):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    count = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
    return [a + (b - a) * (i / count) for i in range(1, count + 1)]


def check_path(cfg, h, points, home_xyz, det, selected):
    f, l = axes(cfg)
    s = cfg['safety']
    minimum = np.asarray(s['workspace_min_base_m'], float)
    maximum = np.asarray(s['workspace_max_base_m'], float)
    box_w = measured(s, 'box_lateral_width_m')
    plate_w = measured(s, 'pusher_lateral_width_m')
    margin = measured(s, 'extra_object_clearance_m', False)
    other = det['blue' if selected == 'red' else 'red']['point_base_m']
    separation = (plate_w + box_w) / 2 + margin
    for p in points:
        if np.any(p < minimum) or np.any(p > maximum):
            raise SafetyFault('Full path violates robot-base XYZ workspace')
        if np.linalg.norm(p - home_xyz) > float(s['max_total_tcp_travel_m']):
            raise SafetyFault('Full path exceeds maximum travel from HOME')
        relative = p - home_xyz
        dist_f = float(f @ relative)
        dist_l = float(l @ relative)
        if not (cfg['task']['forward_bounds_m'][0] <= dist_f <= cfg['task']['forward_bounds_m'][1]):
            raise SafetyFault('Full path exceeds forward bounds')
        if not (cfg['task']['lateral_bounds_m'][0] <= dist_l <= cfg['task']['lateral_bounds_m'][1]):
            raise SafetyFault('Full path exceeds lateral bounds')
        if abs(float(l @ (other - p))) < separation and abs(float(f @ (other - p))) < separation:
            if abs(p[2] - other[2]) < measured(h, 'lift_after_push_m') + box_w:
                raise SafetyFault('Other box too close to plate sweep')


def precheck_ik(arm, cfg, points, rpy):
    joints = arm.angles()
    r = cfg['robot']
    limits = np.asarray(r['joint_limits_deg'], float)
    margin = float(r['joint_margin_deg'])
    for xyz in points:
        goal = np.r_[xyz * 1000, rpy]
        fields = dict(zip(('x', 'y', 'z', 'rx', 'ry', 'rz'), map(float, goal)))
        fields.update({'use_joint_near': '1', 'joint_near': '{' + ','.join(f'{x:.5f}' for x in joints) + '}', 'user': str(r['user_frame']), 'tool': str(r['tool_frame'])})
        candidate = parse_payload(arm.call('InverseKin', fields).robot_return, 6)
        if np.any(candidate < limits[:, 0] + margin) or np.any(candidate > limits[:, 1] - margin):
            raise SafetyFault('Full path IK violates joint limits')
        if np.max(np.abs(candidate - joints)) > float(r['max_joint_delta_deg']):
            raise SafetyFault('Full path IK exceeds joint step limit')
        joints = candidate


def move(arm, cfg, xyz, rpy):
    arm.ensure_idle()
    alarms(arm)
    arm.move_and_wait(np.r_[np.asarray(xyz, float) * 1000, rpy])


def plan(cfg, h, home, trace, tcp, det, selected, gap):
    f, _ = axes(cfg)
    z = np.array([0., 0., 1.])
    step = float(h['segment_m'])
    contact = tcp + f * gap
    finish = contact + f * .10
    lift = z * float(h['lift_after_push_m'])
    paths = {
        'CONTACT': line(tcp, contact, step),
        'PUSH': line(contact, finish, step),
        'LIFT': line(finish, finish + lift, step),
        'RETRACT': line(finish + lift, tcp + lift, step),
    }
    anchor = tcp + lift
    back = []
    for p in reversed(trace[:-1]):
        target = p + lift
        back.extend(line(anchor, target, step))
        anchor = target
    back.extend(line(anchor, home[:3] * .001, step))
    paths['HOME'] = back
    all_points = [p for part in paths.values() for p in part]
    check_path(cfg, h, all_points, home[:3] * .001, det, selected)
    return paths


def cycle(color, arm, camera, policy, cfg, h, home, execute):
    f, l = axes(cfg)
    origin = home[:3] * .001
    rpy = home[3:].copy()
    trace = [origin.copy()]
    last = {}
    for index in range(int(cfg['task']['max_steps'])):
        arm.ensure_idle()
        alarms(arm)
        pose, det = capture(camera, arm, cfg)
        tcp = pose[:3] * .001
        if np.max(np.abs((pose[3:] - rpy + 180) % 360 - 180)) > .5:
            raise SafetyFault('Wrist orientation changed')
        if not image_safe(det[color], cfg, h):
            raise SafetyFault('Target approaching image edge before safe handover')
        for name in ('red', 'blue'):
            point = det[name]['point_base_m']
            if name in last and np.linalg.norm(point - last[name]) > float(cfg['task']['max_detection_jump_m']):
                raise SafetyFault('Unstable RGB-D detection: ' + name)
            last[name] = point.copy()
        distance = float(f @ (det[color]['point_base_m'] - tcp))
        side = float(l @ (det[color]['point_base_m'] - tcp))
        gap = distance - measured(h, 'plate_front_offset_m', False) - measured(h, 'box_surface_to_rear_face_m', False)
        print(f'PPO {index} {color}: gap={gap:.3f}m side={side:.3f}m', flush=True)
        if gap <= h['handover_gap_m']:
            box_z = float(det[color]['point_base_m'][2]) + float(h['box_surface_to_contact_z_m'])
            tool_z = float(tcp[2]) + float(h['plate_tcp_to_contact_z_m'])
            if abs(box_z - tool_z) > float(h['vertical_tolerance_m']):
                raise SafetyFault('Plate not vertically aligned to box')
            if not 0 < gap <= h['max_contact_gap_m'] or abs(side) > h['lateral_tolerance_m']:
                raise SafetyFault('Unsafe handover geometry: stop')
            paths = plan(cfg, h, home, trace, tcp, det, color, gap)
            path = [p for part in paths.values() for p in part]
            print(f'HANDOVER: RL OFF; push=100mm; waypoints={len(path)}', flush=True)
            if not execute:
                print('PREVIEW: No robot commands', flush=True)
                return
            precheck_ik(arm, cfg, path, rpy)
            if input('Type PUSH ' + color.upper() + ' to execute full trajectory: ').strip() != 'PUSH ' + color.upper():
                raise SafetyFault('Operator cancelled at handover; robot remains stationary')
            live_pose = arm.pose()
            if np.linalg.norm(live_pose[:3] - pose[:3]) > 1.0 or np.max(np.abs((live_pose[3:] - pose[3:] + 180) % 360 - 180)) > .5:
                raise SafetyFault('Arm moved after trajectory preflight')
            for phase, waypoints in paths.items():
                print(phase, flush=True)
                for xyz in waypoints:
                    move(arm, cfg, xyz, rpy)
            home_check(arm, home)
            print(f'DONE {color}; HOME', flush=True)
            return
        command = np.array([f @ (tcp - origin), l @ (tcp - origin)])
        _, raw = observation(cfg, command, det, tcp, color)
        if np.any(raw < OBS_LOW - 1e-5) or np.any(raw > OBS_HIGH + 1e-5):
            raise SafetyFault('PPO observation exceeds training bounds')
        action = policy.predict(raw)
        _, next_tcp = proposed_command(cfg, action, command, origin)
        check_wrong_cube_clearance(cfg, det, origin, next_tcp, color)
        print('ACTION', np.round(action, 3).tolist(), flush=True)
        if not execute:
            print('PREVIEW: First PPO prediction only; no robot commands', flush=True)
            return
        move(arm, cfg, next_tcp, rpy)
        trace.append(next_tcp.copy())
    raise SafetyFault('PPO step limit reached without a safe handover')


def status(cfg, h):
    fields = {
        'task.forward_axis_base': cfg['task'].get('forward_axis_base'),
        'task.lateral_axis_base': cfg['task'].get('lateral_axis_base'),
        'robot.tool_frame': cfg['robot'].get('tool_frame'),
        'robot.joint_limits_deg': cfg['robot'].get('joint_limits_deg'),
        'calibration.baseline_tcp_pose_mm_deg': cfg['calibration'].get('baseline_tcp_pose_mm_deg'),
        'calibration.T_base_camera_at_baseline': cfg['calibration'].get('T_base_camera_at_baseline'),
        'calibration.holdout_rmse_m': cfg['calibration'].get('holdout_rmse_m'),
        'safety.workspace_min_base_m': cfg['safety'].get('workspace_min_base_m'),
        'safety.workspace_max_base_m': cfg['safety'].get('workspace_max_base_m'),
        'safety.box_lateral_width_m': cfg['safety'].get('box_lateral_width_m'),
    }
    fields.update({'hybrid.' + k: h.get(k) for k in ('handover_gap_m', 'lateral_tolerance_m', 'vertical_tolerance_m', 'plate_front_offset_m', 'box_surface_to_rear_face_m', 'box_surface_to_contact_z_m', 'plate_tcp_to_contact_z_m', 'max_contact_gap_m', 'lift_after_push_m')})
    print('MISSING MEASUREMENTS:')
    for name, value in fields.items():
        if value is None:
            print(' ', name)
    print('UNVERIFIED PHYSICAL CHECKS:')
    flags = {k: v for k, v in cfg['safety'].items() if isinstance(v, bool)}
    flags['calibration.approved'] = cfg['calibration']['approved']
    flags.update({'hybrid.' + k: h.get(k) for k in ('contact_geometry_verified', 'lift_and_return_clearance_verified', 'full_stroke_verified')})
    for name, value in flags.items():
        if value is not True:
            print(' ', name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['red', 'blue'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'config/deployment.json').read_text())
    h = json.loads((ROOT / 'config/hardware_hybrid.json').read_text())
    if args.status:
        status(cfg, h)
        return 0
    arm = None
    policy = None
    try:
        prepare(cfg, h, args.execute)
        arm = ROSArm(cfg)
        arm.ensure_idle()
        alarms(arm)
        home = arm.pose().copy()
        camera_transform(cfg, home)
        policy = Policy(ROOT)
        print('SESSION HOME:', np.round(home, 3).tolist(), flush=True)
        with D405(cfg['camera']) as camera:
            while True:
                color = args.target or input('Color [red/blue/quit]: ').strip().lower()
                if color in ('quit', 'exit', 'q'):
                    break
                if color not in ('red', 'blue'):
                    print('Choose red or blue', flush=True)
                    continue
                home_check(arm, home)
                if args.execute and input('Type START ' + color.upper() + ': ').strip() != 'START ' + color.upper():
                    print('Cancelled', flush=True)
                    continue
                cycle(color, arm, camera, policy, cfg, h, home, args.execute)
                if args.target or not args.execute:
                    break
    except (SafetyFault, RuntimeError, ValueError, KeyError, TypeError, KeyboardInterrupt) as exc:
        print('STOP:', str(exc), flush=True)
        if args.execute and arm is not None:
            arm.stop_best_effort()
        return 2
    finally:
        if policy is not None:
            policy.close()
        if arm is not None:
            arm.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
