import argparse
import json
import math
from pathlib import Path
import numpy as np
from arm_bridge.core import SafetyFault, OBS_LOW, OBS_HIGH
from arm_bridge.perception import D405, find_colors
from arm_bridge.policy import Policy
from arm_bridge.ros_arm import ROSArm, parse_payload

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / 'config' / 'cr10_real_hybrid.json'
MEASUREMENTS = ROOT / 'cr10_motion_measurements.json'


def vector(x, name):
    if x is None:
        raise SafetyFault('Missing: ' + name)
    v = np.asarray(x, float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise SafetyFault('Invalid: ' + name)
    return v


def number(x, name):
    if x is None or not np.isfinite(float(x)):
        raise SafetyFault('Missing or invalid: ' + name)
    return float(x)


def angle_error(a, b):
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def points_between(a, b, step=0.002):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, math.ceil(float(np.linalg.norm(b - a)) / step))
    return [a + (b - a) * k / n for k in range(1, n + 1)]


def geometry(cfg, setup, measured):
    f = vector(measured['candidate_forward_axis_base'], 'forward axis')
    l = vector(measured['candidate_lateral_axis_base'], 'lateral axis')
    f, l = f / np.linalg.norm(f), l / np.linalg.norm(l)
    if abs(f @ l) > .01 or abs(f[2]) > .03 or abs(l[2]) > .03:
        raise SafetyFault('Measured axes are not horizontal and orthogonal')
    r = measured.get('camera_rotation_candidate')
    if r is None or np.asarray(r).shape != (3, 3):
        raise SafetyFault('Camera rotation not measured')
    r = np.asarray(r, float)
    if np.linalg.norm(r.T @ r - np.eye(3)) > .01 or np.linalg.det(r) < .99:
        raise SafetyFault('Camera rotation is invalid')
    return f, l, r


def verify(cfg, setup, measured):
    f, l, r = geometry(cfg, setup, measured)
    if not setup.get('physical_geometry_verified'):
        raise SafetyFault('Physical camera and plate geometry have not been verified')
    if not setup.get('full_path_clearance_verified'):
        raise SafetyFault('The complete 100 mm push and return path have not been verified')
    if not setup.get('robot_protection_verified'):
        raise SafetyFault('Robot protection and operating area have not been verified')
    if not setup.get('independent_camera_validation_passed'):
        raise SafetyFault('An independent camera-to-plate validation is missing')
    fit = number(measured.get('rotation_residual_m'), 'camera motion fit residual')
    if fit > .002:
        raise SafetyFault(f'Camera motion fit residual {fit * 1000:.2f} mm exceeds 2 mm')
    origin = vector(setup.get('camera_origin_from_flange_base_m'), 'camera optical-center offset')
    plate = vector(setup.get('plate_contact_from_flange_base_m'), 'plate contact offset')
    minimum = vector(setup.get('workspace_min_base_m'), 'workspace minimum')
    maximum = vector(setup.get('workspace_max_base_m'), 'workspace maximum')
    if np.any(maximum <= minimum):
        raise SafetyFault('Workspace min/max invalid')
    limits = setup.get('verified_joint_limits_deg')
    if limits is None or np.asarray(limits).shape != (6, 2):
        raise SafetyFault('Six real robot joint limits are required')
    limits = np.asarray(limits, float)
    if not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise SafetyFault('Joint limits invalid')
    width = number(setup.get('box_width_m'), 'box width')
    if width <= 0 or width > .2:
        raise SafetyFault('Box width invalid')
    gap = number(setup.get('handover_gap_m'), 'handover gap')
    if not .006 <= gap <= .06:
        raise SafetyFault('Handover gap outside 6–60 mm')
    vertical = number(setup.get('vertical_tolerance_m'), 'vertical tolerance')
    if vertical <= 0 or vertical > .025:
        raise SafetyFault('Vertical tolerance invalid')
    side = number(setup.get('lateral_tolerance_m'), 'lateral tolerance')
    if side <= 0 or side > .025:
        raise SafetyFault('Lateral tolerance invalid')
    if float(cfg['task']['max_step_m']) > .002:
        raise SafetyFault('PPO action step must not exceed 2 mm')
    return f, l, r, origin, plate, minimum, maximum, limits


def alarms(arm):
    data = arm.call('GetErrorID').robot_return.strip()
    if not (data.startswith('{') and data.endswith('}')):
        raise SafetyFault('Cannot verify robot alarms')
    values = json.loads(data[1:-1])
    if not isinstance(values, list) or any(values):
        raise SafetyFault('Robot alarm active')


def capture(camera, cfg, r, camera_base, pose_base):
    frame = camera.read()
    items = find_colors(frame, cfg['camera'], rs_module=camera.rs)
    if any(items[c] is None for c in ('red', 'blue')):
        raise SafetyFault('Red or blue detection lost')
    result = {}
    for color, item in items.items():
        point = pose_base + camera_base + r @ item['point_optical_m']
        result[color] = dict(item, point_base_m=point)
    return result


def observation(cfg, home, flange, plate, boxes, f, l, color):
    tcp = flange + plate
    origin = home + plate
    raw = [float(f @ (tcp - origin)), float(l @ (tcp - origin))]
    for name in ('red', 'blue'):
        p = boxes[name]['point_base_m'] - tcp
        raw.extend([float(f @ p), float(l @ p), float(boxes[name]['depth_m']), 1.])
    raw.extend([float(color == 'red'), float(color == 'blue')])
    raw = np.asarray(raw, np.float32)
    if np.any(raw < OBS_LOW - 1e-5) or np.any(raw > OBS_HIGH + 1e-5):
        raise SafetyFault('Camera observation outside PPO training range')
    return raw


def verify_target(cfg, setup, home, flange, boxes, f, l, plate, color):
    tcp = flange + plate
    target = boxes[color]['point_base_m']
    remaining = float(f @ (target - tcp))
    side = float(l @ (target - tcp))
    dz = float(target[2] - tcp[2])
    return remaining, side, dz


def check_waypoints(points, setup, minimum, maximum, flange, boxes, color, f, l, plate):
    other = boxes['blue' if color == 'red' else 'red']['point_base_m']
    margin = (.09 + float(setup['box_width_m'])) / 2 + .01
    for p in points:
        tcp = p + plate
        if not np.isfinite(p).all() or np.any(p < minimum) or np.any(p > maximum):
            raise SafetyFault('Flange trajectory leaves verified workspace')
        if np.any(tcp < minimum) or np.any(tcp > maximum):
            raise SafetyFault('Plate-contact trajectory leaves verified workspace')
        if abs(float(l @ (tcp - other))) < margin and abs(float(f @ (tcp - other))) < margin:
            if abs(tcp[2] - other[2]) < .08:
                raise SafetyFault('Other box intersects plate sweep')


def precheck_path(arm, cfg, points, rpy):
    joints = arm.angles()
    joint_limits = np.asarray(cfg['robot']['joint_limits_deg'], float)
    margin = float(cfg['robot']['joint_margin_deg'])
    for point in points:
        pose = np.r_[point * 1000, rpy]
        fields = dict(zip(('x', 'y', 'z', 'rx', 'ry', 'rz'), map(float, pose)))
        fields.update({'use_joint_near': '1', 'joint_near': '{' + ','.join(f'{v:.5f}' for v in joints) + '}', 'user': '0', 'tool': '0'})
        candidate = parse_payload(arm.call('InverseKin', fields).robot_return, 6)
        if np.any(candidate <= joint_limits[:, 0] + margin) or np.any(candidate >= joint_limits[:, 1] - margin):
            raise SafetyFault('Planned waypoint violates joint limits')
        if np.max(np.abs(candidate - joints)) > float(cfg['robot']['max_joint_delta_deg']):
            raise SafetyFault('Planned joint step exceeds limit')
        joints = candidate


def move(arm, point, rpy):
    arm.ensure_idle()
    alarms(arm)
    arm.move_and_wait(np.r_[point * 1000, rpy])


def run_one(color, cfg, setup, measured, arm, camera, policy, g, home):
    f, l, r, cam_offset, plate, low, high, _ = g
    origin = home[:3] * .001
    orientation = home[3:].copy()
    trace = [origin.copy()]
    last = None
    initial = None
    max_steps = min(int(cfg['task']['max_steps']), 155)
    for i in range(max_steps):
        arm.ensure_idle()
        alarms(arm)
        pose = arm.pose()
        if np.max(np.abs(angle_error(pose[3:], orientation))) > .5:
            raise SafetyFault('Wrist orientation changed')
        flange = pose[:3] * .001
        boxes = capture(camera, cfg, r, cam_offset, flange)
        if last is not None and any(np.linalg.norm(boxes[c]['point_base_m'] - last[c]) > .025 for c in ('red', 'blue')):
            raise SafetyFault('RGB-D point jumped')
        last = {c: boxes[c]['point_base_m'].copy() for c in ('red', 'blue')}
        if initial is None:
            initial = {c: boxes[c]['point_base_m'].copy() for c in ('red', 'blue')}
        if any(np.linalg.norm(boxes[c]['point_base_m'] - initial[c]) > .008 for c in ('red', 'blue')):
            raise SafetyFault('Box moved or visual registration drifted')
        u, v = boxes[color]['uv_policy_px']
        w, h = cfg['camera']['policy_resolution_px']
        if not (8 < u < w - 8 and 8 < v < h - 8):
            raise SafetyFault('Target near camera image edge; stop before vision loss')
        remaining, lateral, dz = verify_target(cfg, setup, origin, flange, boxes, f, l, plate, color)
        print(f'{color} step={i} gap={remaining*1000:.1f}mm side={lateral*1000:.1f}mm height={dz*1000:.1f}mm', flush=True)
        if remaining <= float(setup['handover_gap_m']):
            if not 0 < remaining <= float(setup['handover_gap_m']):
                raise SafetyFault('Box is behind or in contact with plate')
            if abs(lateral) > float(setup['lateral_tolerance_m']):
                raise SafetyFault('Plate is not aligned laterally')
            if abs(dz) > float(setup['vertical_tolerance_m']):
                raise SafetyFault('Plate is not aligned vertically')
            contact = flange + f * remaining
            end = contact + f * .1
            approach = points_between(flange, contact)
            stroke = points_between(contact, end)
            retreat = list(reversed([flange.copy()] + approach[:-1] + stroke[:-1]))
            retreat += list(reversed(trace[:-1]))
            points = approach + stroke + retreat
            check_waypoints(points, setup, low, high, flange, boxes, color, f, l, plate)
            if np.linalg.norm(retreat[-1] - origin) > .001:
                raise SafetyFault('Return does not end at session HOME')
            precheck_path(arm, cfg, points, orientation)
            print(f'PPO OFF. Push 100 mm; return HOME. {len(points)} guarded waypoints.', flush=True)
            if input('Confirm clear full arm path and type PUSH ' + color.upper() + ': ').strip() != 'PUSH ' + color.upper():
                raise SafetyFault('Push cancelled; robot remains at handover pose')
            now = arm.pose()
            if np.linalg.norm(now[:3] - pose[:3]) > .5 or np.max(np.abs(angle_error(now[3:], pose[3:]))) > .5:
                raise SafetyFault('Robot pose changed after confirmation')
            for j, p in enumerate(points):
                move(arm, p, orientation)
                if j == len(approach) - 1:
                    print('CONTACT', flush=True)
                if j == len(approach) + len(stroke) - 1:
                    print('PUSH COMPLETE', flush=True)
            actual = arm.pose()
            if np.linalg.norm(actual[:3] - home[:3]) > 2:
                raise SafetyFault('Return-to-HOME position error')
            print('HOME. Ready for next color.', flush=True)
            return
        raw = observation(cfg, origin, flange, plate, boxes, f, l, color)
        action = policy.predict(raw)
        delta = action / max(1., np.linalg.norm(action)) * min(.002, float(cfg['task']['max_step_m']))
        next_flange = flange + f * float(delta[0]) + l * float(delta[1])
        check_waypoints([next_flange], setup, low, high, flange, boxes, color, f, l, plate)
        rel = next_flange - origin
        uf, ul = float(f @ rel), float(l @ rel)
        if not (0 <= uf <= .19 and -.125 <= ul <= .125):
            raise SafetyFault('PPO approach exceeds trained workspace')
        move(arm, next_flange, orientation)
        trace.append(next_flange)
    raise SafetyFault('PPO did not reach safe handover')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    p.add_argument('--status', action='store_true')
    args = p.parse_args()
    cfg = json.loads((ROOT / 'config/deployment.json').read_text())
    setup = json.loads(SETTINGS.read_text())
    measured = json.loads(MEASUREMENTS.read_text())
    if args.status:
        for name in ('camera_origin_from_flange_base_m', 'plate_contact_from_flange_base_m', 'workspace_min_base_m', 'workspace_max_base_m', 'verified_joint_limits_deg', 'box_width_m', 'handover_gap_m', 'lateral_tolerance_m', 'vertical_tolerance_m'):
            print(name, setup.get(name))
        print('Camera motion residual (mm):', round(float(measured['rotation_residual_m']) * 1000, 2))
        for name in ('physical_geometry_verified', 'independent_camera_validation_passed', 'full_path_clearance_verified', 'robot_protection_verified'):
            print(name, setup.get(name))
        return
    if not args.execute:
        raise SystemExit('Use --status to inspect, or --execute for guarded hardware motion')
    g = verify(cfg, setup, measured)
    cfg['robot']['tool_frame'] = 0
    cfg['robot']['joint_limits_deg'] = setup['verified_joint_limits_deg']
    cfg['robot']['linear_speed_mm_s'] = min(5, int(cfg['robot']['linear_speed_mm_s']))
    cfg['robot']['acceleration_percent'] = min(5, int(cfg['robot']['acceleration_percent']))
    arm = None
    policy = None
    try:
        arm = ROSArm(cfg)
        arm.ensure_idle()
        alarms(arm)
        home = arm.pose().copy()
        print('HOME:', np.round(home, 2).tolist(), flush=True)
        arm.call('SpeedFactor', {'ratio': 2})
        policy = Policy(ROOT)
        with D405(cfg['camera']) as camera:
            while True:
                color = input('red / blue / quit: ').strip().lower()
                if color in ('quit', 'q', 'exit'):
                    break
                if color not in ('red', 'blue'):
                    continue
                actual = arm.pose()
                if np.linalg.norm(actual[:3] - home[:3]) > 2 or np.max(np.abs(angle_error(actual[3:], home[3:]))) > .5:
                    raise SafetyFault('Robot must be at saved HOME')
                if input('Type START ' + color.upper() + ': ').strip() != 'START ' + color.upper():
                    continue
                run_one(color, cfg, setup, measured, arm, camera, policy, g, home)
    except (Exception, KeyboardInterrupt) as exc:
        print('STOP:', exc, flush=True)
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
