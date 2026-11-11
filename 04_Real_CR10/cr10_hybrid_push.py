import argparse
import json
from pathlib import Path
import numpy as np
from arm_bridge.core import SafetyFault, OBS_LOW, OBS_HIGH, axes, camera_transform, checked_static_preflight, observation, proposed_command, check_wrong_cube_clearance
from arm_bridge.perception import D405, find_colors
from arm_bridge.policy import Policy
from arm_bridge.ros_arm import ROSArm

ROOT = Path(__file__).resolve().parent


def number(cfg, key, lower=0):
    value = cfg.get(key)
    if value is None or not np.isfinite(float(value)) or float(value) <= lower:
        raise SafetyFault(f'Measured hybrid setting required: {key}')
    return float(value)


def parameters(cfg, hybrid, execute):
    checked_static_preflight(cfg, live=execute)
    number(hybrid, 'handover_forward_m')
    number(hybrid, 'handover_lateral_m')
    number(hybrid, 'plate_front_offset_m', -1)
    number(hybrid, 'box_reference_to_rear_face_m', -1)
    number(hybrid, 'maximum_contact_approach_m')
    number(hybrid, 'image_edge_margin_px')
    if float(hybrid.get('push_distance_m', 0)) != 0.10:
        raise SafetyFault('Push stroke must be 0.10 m')
    if float(hybrid.get('motion_step_m', 1)) > 0.002 or float(hybrid.get('motion_step_m', 0)) <= 0:
        raise SafetyFault('Motion segments must be <= 2 mm')
    home = np.asarray(hybrid.get('home_pose_mm_deg'), dtype=float) if hybrid.get('home_pose_mm_deg') is not None else None
    if home is None or home.shape != (6,) or not np.all(np.isfinite(home)):
        raise SafetyFault('Measured home_pose_mm_deg required')
    if execute and not (hybrid.get('push_geometry_verified') and hybrid.get('stroke_clearance_verified')):
        raise SafetyFault('Physical contact geometry and full 10 cm stroke must be verified')
    return home


def alarms(arm):
    reply = arm.call('GetErrorID').robot_return.strip()
    if not (reply.startswith('{') and reply.endswith('}')):
        raise SafetyFault('Invalid controller alarm response')
    data = json.loads(reply[1:-1])
    if not isinstance(data, list) or len(data) != 7 or any(data):
        raise SafetyFault('Active robot alarms')


def pose_ok(actual, home):
    if np.linalg.norm(actual[:3] - home[:3]) > 3:
        raise SafetyFault('Start position is not verified HOME')
    angle = (actual[3:] - home[3:] + 180) % 360 - 180
    if np.max(np.abs(angle)) > 0.5:
        raise SafetyFault('Start orientation differs from HOME')


def segments(start, stop, step):
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    n = max(1, int(np.ceil(np.linalg.norm(stop - start) / step)))
    return [start + (stop - start) * (i / n) for i in range(1, n + 1)]


def validate_path(cfg, detections, home_xyz, path, target):
    forward, lateral = axes(cfg)
    min_xyz = np.asarray(cfg['safety']['workspace_min_base_m'], dtype=float)
    max_xyz = np.asarray(cfg['safety']['workspace_max_base_m'], dtype=float)
    other = 'blue' if target == 'red' else 'red'
    selected_point = detections[target]['point_base_m']
    other_point = detections[other]['point_base_m']
    bw = float(cfg['safety']['box_lateral_width_m'])
    pw = float(cfg['safety']['pusher_lateral_width_m'])
    clearance = float(cfg['safety']['extra_object_clearance_m'])
    side_separation = abs(float(lateral @ (selected_point - other_point)))
    along_separation = float(forward @ (other_point - selected_point))
    stroke = 0.10
    if side_separation < bw + clearance and -bw <= along_separation <= stroke + bw:
        raise SafetyFault('Other box is in selected box 10 cm swept path')
    for p in path:
        if np.any(p < min_xyz) or np.any(p > max_xyz):
            raise SafetyFault('Path exceeds measured TCP workspace')
        if np.linalg.norm(p - home_xyz) > float(cfg['safety']['max_total_tcp_travel_m']):
            raise SafetyFault('Path exceeds maximum travel from HOME')
        f = float(forward @ (p - home_xyz))
        l = float(lateral @ (p - home_xyz))
        a, b = cfg['task']['forward_bounds_m']
        c, d = cfg['task']['lateral_bounds_m']
        if not (a - 1e-6 <= f <= b + 1e-6 and c - 1e-6 <= l <= d + 1e-6):
            raise SafetyFault('Path exceeds PPO work area')
        if abs(float(lateral @ (other_point - p))) < (bw + pw) / 2 + clearance:
            if abs(float(forward @ (other_point - p))) < (bw + pw) / 2 + clearance:
                raise SafetyFault('Other box too close to planned plate path')


def capture(cfg, camera, arm):
    pose = arm.pose()
    transform = camera_transform(cfg, pose)
    frame = camera.read()
    found = find_colors(frame, cfg['camera'], transform, rs_module=camera.rs)
    if any(found[name] is None for name in ('red', 'blue')):
        raise SafetyFault('Box not visible: stop before losing vision')
    return pose, frame, found


def within_image(frame, det, cfg, hybrid):
    width, height = cfg['camera']['policy_resolution_px']
    u, v = det['uv_policy_px']
    margin = float(hybrid['image_edge_margin_px'])
    return margin < u < width - margin and margin < v < height - margin


def move(arm, cfg, xyz, orientation):
    arm.ensure_idle()
    alarms(arm)
    arm.move_and_wait(np.r_[np.asarray(xyz) * 1000, orientation])


def run_cycle(color, camera, arm, policy, cfg, hybrid, home, execute):
    forward, lateral = axes(cfg)
    step = float(hybrid['motion_step_m'])
    home_xyz = home[:3] / 1000
    orientation = home[3:]
    approach_trace = [home_xyz.copy()]
    initial_boxes = None
    last = None
    handover = None
    for i in range(int(cfg['task']['max_steps'])):
        arm.ensure_idle()
        alarms(arm)
        pose, frame, det = capture(cfg, camera, arm)
        if not within_image(frame, det[color], cfg, hybrid):
            raise SafetyFault('Selected box near image edge: no safe handover')
        points = {name: det[name]['point_base_m'].copy() for name in ('red', 'blue')}
        if initial_boxes is None:
            initial_boxes = points
        if last is not None:
            for name in points:
                if np.linalg.norm(points[name] - last[name]) > float(cfg['task']['max_detection_jump_m']):
                    raise SafetyFault('Box detection jumped')
        last = points
        other = 'blue' if color == 'red' else 'red'
        if np.linalg.norm(points[other][:2] - initial_boxes[other][:2]) > float(cfg['task']['max_wrong_displacement_m']):
            raise SafetyFault('Other box moved')
        tcp = pose[:3] / 1000
        delta = points[color] - tcp
        distance = float(forward @ delta)
        side = float(lateral @ delta)
        if distance <= float(hybrid['handover_forward_m']):
            if distance <= 0 or abs(side) > float(hybrid['handover_lateral_m']):
                raise SafetyFault('Not aligned for fixed push at handover')
            handover = (tcp.copy(), det, distance, side)
            print(f'HANDOVER {color}: forward={distance:.3f}m lateral={side:.3f}m', flush=True)
            break
        command = np.array([forward @ (tcp - home_xyz), lateral @ (tcp - home_xyz)])
        _, raw = observation(cfg, command, det, tcp, color)
        if np.any(raw < OBS_LOW - 1e-5) or np.any(raw > OBS_HIGH + 1e-5):
            raise SafetyFault('PPO observation outside training range')
        action = policy.predict(raw)
        _, next_tcp = proposed_command(cfg, action, command, home_xyz)
        check_wrong_cube_clearance(cfg, det, home_xyz, next_tcp, color)
        print(f'PPO {i+1}: {color} action={np.round(action, 3).tolist()}', flush=True)
        if not execute:
            print('PREVIEW: no robot movement; handover not reached', flush=True)
            return
        move(arm, cfg, next_tcp, orientation)
        approach_trace.append(np.asarray(next_tcp).copy())
    if handover is None:
        raise SafetyFault('PPO maximum steps reached without handover')
    tcp, det, distance, side = handover
    contact_distance = distance - float(hybrid['plate_front_offset_m']) - float(hybrid['box_reference_to_rear_face_m'])
    if not 0 < contact_distance <= float(hybrid['maximum_contact_approach_m']):
        raise SafetyFault('Contact position not physically valid')
    if abs(side) > float(hybrid['handover_lateral_m']):
        raise SafetyFault('Plate is not aligned with box')
    contact = tcp + contact_distance * forward
    endpoint = contact + 0.10 * forward
    contact_path = segments(tcp, contact, step)
    push_path = segments(contact, endpoint, step)
    retreat_path = segments(endpoint, contact, step)
    return_path = segments(contact, tcp, step) + list(reversed(approach_trace[:-1]))
    full_path = contact_path + push_path + retreat_path + return_path
    validate_path(cfg, det, home_xyz, full_path, color)
    print(f'PLAN {color}: contact={np.round(contact, 4).tolist()} push_end={np.round(endpoint, 4).tolist()} length=10cm', flush=True)
    if not execute:
        print('PREVIEW: no robot movement', flush=True)
        return
    if input(f'Type PUSH {color.upper()} to execute checked 10cm stroke: ').strip() != f'PUSH {color.upper()}':
        print('CANCELLED at handover', flush=True)
        return
    for name, path in (('CONTACT', contact_path), ('PUSH', push_path), ('RETRACT', retreat_path), ('HOME', return_path)):
        print(name, flush=True)
        for point in path:
            move(arm, cfg, point, orientation)
    pose_ok(arm.pose(), home)
    print(f'DONE {color}; HOME', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['red', 'blue'])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'config/deployment.json').read_text())
    hybrid = json.loads((ROOT / 'config/hybrid_push.json').read_text())
    home = parameters(cfg, hybrid, args.execute)
    arm = None
    policy = None
    try:
        arm = ROSArm(cfg)
        arm.ensure_idle()
        alarms(arm)
        pose_ok(arm.pose(), home)
        policy = Policy(ROOT)
        with D405(cfg['camera']) as camera:
            while True:
                color = args.target or input('Color [red/blue/quit]: ').strip().lower()
                if color in ('quit', 'q', 'exit'):
                    break
                if color not in ('red', 'blue'):
                    print('Choose red or blue', flush=True)
                    continue
                pose_ok(arm.pose(), home)
                run_cycle(color, camera, arm, policy, cfg, hybrid, home, args.execute)
                if args.target or not args.execute:
                    break
    except (SafetyFault, RuntimeError, ValueError, KeyboardInterrupt) as exc:
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
