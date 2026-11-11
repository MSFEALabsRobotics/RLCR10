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
LIFT_MM = 170.0
HANDOVER_GAP_MM = 110.0
HANDOVER_DEPTH_M = 0.185
LATERAL_TOL_MM = 10.0
EDGE_PX = 8.0


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


def camera_offset_estimate(capture, r):
    optical = np.asarray(capture['detections']['red']['optical_point_m'], dtype=float)
    approximate_plate_gap_m = float(capture['geometry']['approx_home_plate_to_box_face_m'])
    candidate = 1000.0 * (approximate_plate_gap_m - F @ (r @ optical))
    if not 30.0 <= candidate <= 150.0:
        raise SafetyFault('Camera-to-plate estimate implausible')
    return candidate


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
                0.16 <= depth <= 0.5)
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
    if np.any(obs < OBS_LOW - 1e-5) or np.any(obs > OBS_HIGH + 1e-5):
        raise SafetyFault('Observation outside the trained policy range')
    return frame.frame_number, obs, state


def segments(a, b, step=MOVE_STEP_MM):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    count = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
    return [a + (b - a) * (i / count) for i in range(1, count + 1)]


def plan_path(current, home, gap_mm, lateral_mm):
    start = np.asarray(current[:3], dtype=float)
    end_home = np.asarray(home[:3], dtype=float)
    align = start + L * lateral_mm
    contact = align + F * gap_mm
    pushed = contact + F * STROKE_MM
    lifted = pushed + Z * LIFT_MM
    reverse_forward = np.array([lifted[0], end_home[1], lifted[2]], dtype=float)
    above_home = end_home + Z * LIFT_MM
    phases = [
        ('align', segments(start, align)),
        ('contact', segments(align, contact)),
        ('push', segments(contact, pushed)),
        ('lift', segments(pushed, lifted)),
        ('retreat', segments(lifted, reverse_forward)),
        ('cross', segments(reverse_forward, above_home)),
        ('home', segments(above_home, end_home))
    ]
    if abs(float(np.linalg.norm(pushed - contact)) - STROKE_MM) > 1e-5:
        raise SafetyFault('Push length calculation failed')
    if np.linalg.norm(phases[-1][1][-1] - end_home) > 1e-5:
        raise SafetyFault('Return-to-HOME calculation failed')
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


def verify_push_geometry(path, home, configuration, target, camera_fit_mm):
    if camera_fit_mm > 2.0:
        raise SafetyFault(f'Camera fit {camera_fit_mm:.2f} mm exceeds 2 mm for contact')
    if not configuration.get('independent_camera_validation_passed'):
        raise SafetyFault('Independent camera/plate alignment is not verified')
    if not configuration.get('full_path_clearance_verified'):
        raise SafetyFault('Full push and return path is not verified')
    if not configuration.get('robot_protection_verified'):
        raise SafetyFault('Robot protection is not verified')
    if not configuration.get('physical_geometry_verified'):
        raise SafetyFault('Physical contact geometry is not verified')
    minimum = np.asarray(configuration.get('workspace_min_base_m'), dtype=float)
    maximum = np.asarray(configuration.get('workspace_max_base_m'), dtype=float)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise SafetyFault('Verified workspace bounds are missing')
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise SafetyFault('Invalid workspace bounds')
    for _, points in path:
        for p in points:
            if np.any(p * .001 < minimum) or np.any(p * .001 > maximum):
                raise SafetyFault('Planned path exits verified workspace')


def run(color, arm, camera, policy, cfg, measurements, capture, execute, max_steps, approach_only):
    home = np.asarray(measurements['home']['flange_pose_mm_deg'], dtype=float)
    r = orientation(measurements)
    offset_estimate_mm = camera_offset_estimate(capture, r)
    settings = read('cr10_real_hybrid_settings.json') if (ROOT / 'cr10_real_hybrid_settings.json').exists() else {}
    offset = settings.get('verified_forward_camera_to_plate_mm')
    side_offset = settings.get('verified_lateral_camera_to_plate_mm')
    validated = offset is not None and side_offset is not None
    if not validated:
        offset = offset_estimate_mm
        side_offset = 0.0
    print('Camera forward offset used (mm):', round(offset, 2))
    print('Camera lateral offset used (mm):', round(side_offset, 2))
    print('Geometry:', 'MEASURED' if validated else 'PROVISIONAL; push inhibited')
    if execute and not approach_only:
        if not validated:
            raise SafetyFault('Physical 100 mm push blocked: camera-to-plate contact offsets are not established')
        if float(measurements['rotation_residual_m']) * 1000.0 > 2.0:
            raise SafetyFault('Physical 100 mm push blocked: the camera fit exceeds 2 mm')
        if not all(settings.get(k, False) for k in ('independent_camera_validation_passed', 'full_path_clearance_verified', 'robot_protection_verified', 'physical_geometry_verified')):
            raise SafetyFault('Physical 100 mm push blocked: the full path and camera/plate geometry are not validated')
    current = pose(arm, home, True)
    if execute and input('Clear PPO approach path? Type MOVE: ').strip() != 'MOVE':
        print('Cancelled')
        return
    samples = []
    path_history = []
    handed_over = False
    for step in range(max_steps):
        current = pose(arm, home)
        progress = current[:3] - home[:3]
        if not 0 <= F @ progress <= 190.0 or abs(L @ progress) > 125.0:
            raise SafetyFault('PPO left trained workspace')
        frame, obs, boxes = see(camera, cfg, r, current, home, color, offset, side_offset)
        selected = boxes[color]
        if not selected.get('visible'):
            print('STOP: Target no longer visible; no blind push')
            break
        gap, side = selected['gap_mm'], selected['lateral_mm']
        samples.append([gap, side])
        samples = samples[-5:]
        med_gap, med_side = np.median(samples, axis=0)
        if len(samples) >= 3 and np.ptp(np.asarray(samples)[:, 0]) > 20:
            raise SafetyFault('Depth estimate unstable')
        nearing_depth_limit = selected['depth_mm'] <= HANDOVER_DEPTH_M * 1000.0
        if (med_gap <= HANDOVER_GAP_MM or nearing_depth_limit) and abs(med_side) <= LATERAL_TOL_MM:
            handed_over = True
            print('PPO HANDOVER:',json.dumps({'gap_mm':round(float(med_gap),1), 'side_mm':round(float(med_side),1), 'depth_mm':round(selected['depth_mm'],1)}))
            break
        if nearing_depth_limit and abs(med_side) > LATERAL_TOL_MM:
            print('STOP: insufficient lateral alignment before D405 depth limit')
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
        if not validated and np.linalg.norm(following) > 10.0:
            print('STOP: 10 mm uncalibrated approach envelope')
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
        print('FIXED PLAN:',json.dumps({name:len(points) for name,points in plan}))
        print('PUSH:',STROKE_MM,'mm; LIFT:',LIFT_MM,'mm; RETURN HOME')
        if execute and not approach_only:
            if not validated:
                raise SafetyFault('Camera-to-plate offsets are estimated, not validated: no automatic contact')
            verify_push_geometry(plan, home, settings, color,
                                 float(measurements['rotation_residual_m']) * 1000.0)
            ik_check(arm, home, current, plan)
            if input('Complete path is clear? Type PUSH: ').strip() != 'PUSH':
                print('PUSH cancelled')
                return
            for name, points in plan:
                for q in points:
                    actual = pose(arm, home)
                    move_small(arm, home, q - actual[:3])
                print('FINISHED:',name,flush=True)
            ending = pose(arm, home, True)
            print('PUSH AND HOME COMPLETED:',np.round(ending[:3],2).tolist())
            return
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
