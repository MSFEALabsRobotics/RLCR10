import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from arm_bridge.core import OBS_LOW, OBS_HIGH, SafetyFault
from arm_bridge.perception import D405, find_colors
from arm_bridge.policy import Policy
from arm_bridge.ros_arm import ROSArm, parse_payload

ROOT = Path(__file__).resolve().parent
F = np.array([0.0, 1.0, 0.0])
L = np.array([-1.0, 0.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])
PPO_STEP_MM = 0.5
MOVE_STEP_MM = 2.0
STROKE_MM = 100.0
RETRACT_MM = 20.0
HANDOVER_GAP_MM = 100.0
HANDOVER_DEPTH_M = 0.13
LATERAL_TOL_MM = 10.0
EDGE_PX = 8.0
CAMERA_FORWARD_MM = 80.0
CAMERA_LATERAL_MM = 0.0
MAX_FIXED_FORWARD_MM = 450.0
MAX_FIXED_LATERAL_MM = 160.0


def read(name):
    return json.loads((ROOT / name).read_text())


def angles_delta(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def check_alarms(arm):
    text = arm.call('GetErrorID').robot_return.strip()
    if not (text.startswith('{') and text.endswith('}')):
        raise SafetyFault('Cannot interpret CR10 alarms')
    errors = json.loads(text[1:-1])
    if not isinstance(errors, list) or len(errors) != 7 or any(errors):
        raise SafetyFault('Active CR10 alarm: ' + text)


def pose(arm, home, require_home=False):
    arm.ensure_idle()
    check_alarms(arm)
    actual = arm.raw_flange_pose_read_only()
    if not np.all(np.isfinite(actual)):
        raise SafetyFault('Invalid robot pose')
    if np.max(np.abs(angles_delta(actual[3:], home[3:]))) > 0.5:
        raise SafetyFault('Wrist orientation changed')
    if require_home and np.linalg.norm(actual[:3] - home[:3]) > 3.0:
        raise SafetyFault('Robot not at recorded HOME')
    return actual


def orientation(measurements):
    r = np.asarray(measurements['camera_rotation_candidate'], dtype=float)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)):
        raise SafetyFault('Invalid camera rotation')
    if np.linalg.norm(r.T @ r - np.eye(3)) > .02 or np.linalg.det(r) < .98:
        raise SafetyFault('Camera rotation inconsistent')
    if not np.allclose(measurements['candidate_forward_axis_base'], F, atol=.01):
        raise SafetyFault('Forward axis changed')
    if not np.allclose(measurements['candidate_lateral_axis_base'], L, atol=.01):
        raise SafetyFault('Lateral axis changed')
    return r


def see(camera, cfg, r, current, home, color, forward_offset_mm, lateral_offset_mm):
    frame = camera.read()
    found = find_colors(frame, cfg['camera'], rs_module=camera.rs)
    forward_progress = float(F @ (current[:3] - home[:3]))
    lateral_progress = float(L @ (current[:3] - home[:3]))
    obs = [forward_progress / 1000.0, lateral_progress / 1000.0]
    state = {}
    width, height = cfg['camera']['policy_resolution_px']
    for c in ('red', 'blue'):
        det = found.get(c)
        if det is None:
            state[c] = {'visible': False}
            obs.extend([0.0, 0.0, 0.0, 0.0])
            continue
        point = np.asarray(det['point_optical_m'], dtype=float)
        u, v = [float(x) for x in det['uv_policy_px']]
        depth = float(det['depth_m'])
        good = (point.shape == (3,) and np.isfinite(point).all() and
                EDGE_PX < u < width - EDGE_PX and EDGE_PX < v < height - EDGE_PX and
                0.09 <= depth <= 0.5)
        if not good:
            state[c] = {'visible': False, 'pixel': [u, v], 'reason': 'edge or depth'}
            obs.extend([0.0, 0.0, 0.0, 0.0])
            continue
        relative_mm = (r @ point) * 1000.0 + F * forward_offset_mm + L * lateral_offset_mm
        gap = float(F @ relative_mm)
        lateral = float(L @ relative_mm)
        state[c] = {'visible': True, 'gap_mm': gap, 'lateral_mm': lateral,
                    'depth_mm': depth * 1000.0, 'pixel': [u, v],
                    'box_base_mm': (current[:3] + relative_mm).tolist()}
        obs.extend([gap / 1000.0, lateral / 1000.0, depth, 1.0])
    obs.extend([float(color == 'red'), float(color == 'blue')])
    obs = np.asarray(obs, dtype=np.float32)
    if not np.isfinite(obs).all():
        raise SafetyFault('Non-finite PPO observation')
    obs = np.clip(obs, OBS_LOW, OBS_HIGH)
    return frame.frame_number, obs, state


def segments(a, b, step=MOVE_STEP_MM):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    count = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
    return [a + (b - a) * (i / count) for i in range(1, count + 1)]


def plan_path(current, home, gap_mm, lateral_mm):
    start = np.asarray(current[:3], dtype=float)
    end_home = np.asarray(home[:3], dtype=float)
    if not np.isfinite([gap_mm, lateral_mm]).all() or not 0.0 < gap_mm <= 400.0:
        raise SafetyFault('Estimated box gap is not valid for contact')
    align = start + L * lateral_mm
    contact = align + F * gap_mm
    pushed = contact + F * STROKE_MM
    retracted = pushed - F * RETRACT_MM
    back = align.copy()
    phases = [
        ('align', segments(start, align)),
        ('contact', segments(align, contact)),
        ('push', segments(contact, pushed)),
        ('retract', segments(pushed, retracted)),
        ('back', segments(retracted, back)),
        ('unalign', segments(back, start)),
        ('home', segments(start, end_home))
    ]
    if abs(float(np.linalg.norm(pushed - contact)) - STROKE_MM) > 1e-5:
        raise SafetyFault('Push length calculation failed')
    if np.linalg.norm(phases[-1][1][-1] - end_home) > 1e-5:
        raise SafetyFault('Return-to-HOME calculation failed')
    for _,points in phases:
        for point in points:
            relative = point - end_home
            if not 0.0 <= F @ relative <= MAX_FIXED_FORWARD_MM or abs(L @ relative) > MAX_FIXED_LATERAL_MM:
                raise SafetyFault('Planned waypoint exits guarded task envelope')
            if abs(Z @ relative) > 2.0:
                raise SafetyFault('Unexpected vertical movement in fixed path')
    return phases


def ik_check(arm, home, start, phases):
    previous = np.asarray(start[:3], dtype=float)
    near = arm.angles()
    for label, points in phases:
        for point in points[::5] + ([points[-1]] if points else []):
            if np.linalg.norm(point - previous) > 12.0:
                raise SafetyFault('Path IK sampling discontinuity')
            values = [float(v) for v in np.r_[point, home[3:]]]
            args = dict(zip(('x', 'y', 'z', 'rx', 'ry', 'rz'), values))
            args.update({'use_joint_near': '1',
                         'joint_near': '{' + ','.join(f'{v:.5f}' for v in near) + '}',
                         'user': '0', 'tool': '0'})
            candidate = parse_payload(arm.call('InverseKin', args, timeout_s=5.0).robot_return, 6)
            if np.max(np.abs(angles_delta(candidate, near))) > 6.0:
                raise SafetyFault('Large joint change along planned path')
            near = candidate
            previous = point


def move_small(arm, home, delta, limit=MOVE_STEP_MM):
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise SafetyFault('Invalid movement')
    if np.linalg.norm(delta) > limit + 1e-5:
        raise SafetyFault('Movement exceeds step limit')
    before = pose(arm, home)
    expected = before[:3] + delta
    args = dict(zip(('a', 'b', 'c', 'd', 'e', 'f'), [float(x) for x in delta] + [0.0] * 3))
    args['param_value'] = []
    arm.call('RelMovLUser', args, timeout_s=5.0)
    end = time.monotonic() + 12.0
    consecutive = 0
    while time.monotonic() < end:
        mode = arm.robot_mode()
        if mode not in (5, 7):
            raise SafetyFault(f'Robot mode changed: {mode}')
        now = arm.raw_flange_pose_read_only()
        if np.max(np.abs(angles_delta(now[3:], home[3:]))) > 0.5:
            raise SafetyFault('Unexpected wrist rotation')
        if np.linalg.norm(now[:3] - expected) < 0.35 and mode == 5:
            consecutive += 1
            if consecutive >= 2:
                return now
        else:
            consecutive = 0
        time.sleep(0.08)
    raise SafetyFault('Robot tracking timeout')


def run(color, arm, camera, policy, cfg, measurements, capture, execute, max_steps, approach_only):
    recorded_home = np.asarray(measurements['home']['flange_pose_mm_deg'], dtype=float)
    r = orientation(measurements)
    offset = CAMERA_FORWARD_MM
    side_offset = CAMERA_LATERAL_MM
    print('User-approved camera offsets (mm):', offset, side_offset)
    print('Camera motion-fit residual (mm):', round(float(measurements['rotation_residual_m'])*1000.0, 2))
    current = pose(arm, recorded_home)
    start_offset = float(np.linalg.norm(current[:3] - recorded_home[:3]))
    print('Current XYZ (mm):', np.round(current[:3], 3).tolist())
    print('Old HOME XYZ (mm):', np.round(recorded_home[:3], 3).tolist())
    print('Offset from old HOME (mm):', round(start_offset, 3))
    if start_offset > 20.0:
        raise SafetyFault('Current pose is more than 20 mm from calibrated HOME; cannot rebase automatically')
    home = current.copy()
    print('Session HOME saved from current robot pose')
    if execute and input('Clear PPO approach path? Type MOVE: ').strip() != 'MOVE':
        print('Cancelled')
        return
    samples = []
    path_history = []
    handed_over = False
    handover_reason = None
    last_valid_boxes = []
    for step in range(max_steps):
        current = pose(arm, home)
        progress = current[:3] - home[:3]
        if not 0 <= F @ progress <= 190.0 or abs(L @ progress) > 125.0:
            raise SafetyFault('PPO left trained workspace')
        frame, obs, boxes = see(camera, cfg, r, current, home, color, offset, side_offset)
        selected = boxes[color]
        if not selected.get('visible'):
            if len(last_valid_boxes) >= 3:
                saved_box = np.median(np.asarray(last_valid_boxes, dtype=float), axis=0)
                gap = float(F @ (saved_box - current[:3]))
                side = float(L @ (saved_box - current[:3]))
                if gap <= 0 or gap > 400:
                    print('STOP: Last-known box location is outside acceptable planning range')
                    break
                med_gap, med_side = gap, side
                handed_over = True
                handover_reason = 'target lost, using last valid RGB-D location'
                print('PPO HANDOVER:', json.dumps({'reason': handover_reason,
                    'estimated_remaining_gap_mm': round(gap, 1),
                    'estimated_lateral_mm': round(side, 1),
                    'fixed_forward_total_mm': round(gap + STROKE_MM, 1)}), flush=True)
            else:
                print('STOP: No reliable previous position for target')
            break
        last_valid_boxes.append(selected['box_base_mm'])
        last_valid_boxes = last_valid_boxes[-5:]
        gap, side = selected['gap_mm'], selected['lateral_mm']
        samples.append([gap, side])
        samples = samples[-5:]
        med_gap, med_side = np.median(samples, axis=0)
        if len(samples) >= 3 and np.ptp(np.asarray(samples)[:, 0]) > 20:
            raise SafetyFault('Depth estimate unstable')
        nearing_depth_limit = selected['depth_mm'] <= HANDOVER_DEPTH_M * 1000.0
        if (med_gap <= HANDOVER_GAP_MM or nearing_depth_limit) and abs(med_side) <= LATERAL_TOL_MM:
            handed_over = True
            handover_reason = 'handover threshold reached'
            print('PPO HANDOVER:',json.dumps({'gap_mm':round(float(med_gap),1), 'side_mm':round(float(med_side),1), 'depth_mm':round(selected['depth_mm'],1)}))
            break
        if nearing_depth_limit and abs(med_side) > LATERAL_TOL_MM:
            handed_over = True
            handover_reason = 'depth threshold reached'
            print('HANDOVER WITH FIXED ALIGNMENT:',json.dumps({'gap_mm':round(float(med_gap),1), 'side_mm':round(float(med_side),1)}))
            break
        if med_gap <= 0:
            raise SafetyFault('Estimated box contact already passed')
        action = np.asarray(policy.predict(obs), dtype=float)
        delta_policy = action / max(1.0, float(np.linalg.norm(action))) * PPO_STEP_MM
        delta = F * delta_policy[0] + L * delta_policy[1]
        following = progress + delta
        if not 0 <= F @ following <= 190.0 or abs(L @ following) > 125.0:
            print('STOP: PPO trained workspace boundary')
            break
        print(json.dumps({'step':step, 'frame':frame, 'color':color,
                          'gap_est_mm':round(gap,1), 'side_est_mm':round(side,1),
                          'action':np.round(action,3).tolist(),
                          'robot_delta_mm':np.round(delta,3).tolist()}), flush=True)
        if not execute:
            print('PREVIEW: robot motion disabled')
            return
        current = move_small(arm, home, delta, PPO_STEP_MM)
        path_history.append(delta.copy())
    else:
        print('PPO step limit reached')
    if handed_over:
        plan = plan_path(current, home, float(med_gap), float(med_side))
        print('FIXED PLAN:', json.dumps({name: len(points) for name, points in plan}))
        print('DISTANCE_TO_ESTIMATED_CONTACT_MM:', round(float(med_gap), 1))
        print('PUSH_STROKE_MM:', STROKE_MM)
        print('TOTAL_FORWARD_FIXED_MM:', round(float(med_gap + STROKE_MM), 1))
        print('LATERAL_ALIGNMENT_MM:', round(float(med_side), 1))
        if execute and not approach_only:
            ik_passed = False
            try:
                ik_check(arm, home, current, plan)
                ik_passed = True
                print('KINEMATICS PREFLIGHT: passed sampled waypoints')
            except SafetyFault as exc:
                print('KINEMATICS PREFLIGHT: failed:', str(exc))
            saf = cfg.get('safety', {})
            required = ('full_robot_collision_model_validated',
                        'pusher_plate_box_clearance_validated',
                        'operator_has_verified_robot_protection',
                        'real_motion_authorized')
            path_verified = all(saf.get(k) is True for k in required)
            camera_verified = cfg.get('calibration', {}).get('approved') is True
            if ik_passed and path_verified and camera_verified:
                if input('Verified full trajectory; type PUSH to execute: ').strip() == 'PUSH':
                    for name, points in plan:
                        for q in points:
                            actual = pose(arm, home)
                            move_small(arm, home, q - actual[:3], MOVE_STEP_MM + .4)
                        print('FINISHED:', name, flush=True)
                    ending = pose(arm, home, True)
                    print('PUSH AND HOME COMPLETED:', np.round(ending[:3], 2).tolist())
                    return
            else:
                print('FIXED PATH NOT EXECUTED: camera registration and full collision path are not validated.')
    else:
        print('NO HANDOVER: fixed phase not planned')
    if execute and path_history:
        if input('Reverse the short PPO approach? Type HOME: ').strip() == 'HOME':
            for d in reversed(path_history):
                move_small(arm, home, -d, PPO_STEP_MM)
            print('RETURNED HOME')
        else:
            print('Robot remains at last approach pose')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', choices=('red', 'blue'))
    p.add_argument('--interactive', action='store_true')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--approach-only', action='store_true')
    args = p.parse_args()
    if not 1 <= args.steps <= 380:
        p.error('--steps must be 1..380')
    if args.interactive == (args.target is not None):
        p.error('Provide --target red/blue OR --interactive')
    cfg = read('config/deployment.json')
    measured = read('cr10_motion_measurements.json')
    captured = read('cr10_current_geometry_capture.json')
    arm = None
    policy = None
    try:
        arm = ROSArm(cfg)
        policy = Policy(ROOT)
        with D405(cfg['camera']) as camera:
            if args.interactive:
                while True:
                    choice = input('Color [red/blue/quit]: ').strip().lower()
                    if choice in ('quit', 'q', 'exit'):
                        break
                    if choice not in ('red', 'blue'):
                        print('Enter red or blue')
                        continue
                    run(choice, arm, camera, policy, cfg, measured, captured,
                        args.execute, args.steps, args.approach_only)
            else:
                run(args.target, arm, camera, policy, cfg, measured, captured,
                    args.execute, args.steps, args.approach_only)
    except (Exception, KeyboardInterrupt) as exc:
        print('STOP:',str(exc),flush=True)
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
