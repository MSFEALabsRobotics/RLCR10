import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from arm_bridge.core import SafetyFault
from arm_bridge.perception import D405
from arm_bridge.policy import Policy
from arm_bridge.ros_arm import ROSArm, parse_payload
from cr10_ppo_approach_hardware_v2 import read_policy, VisionStop, alarms, angle_error

ROOT = Path(__file__).resolve().parent
CONTACT_FILE = ROOT / 'cr10_taught_contacts.json'
MEASUREMENT_FILE = ROOT / 'cr10_motion_measurements.json'
STEP_MM = 0.5
MOVE_MM = 2.0
PPO_LIMIT_MM = 40.0
HANDOVER_DEPTH_M = 0.185
PUSH_MM = 100.0
LIFT_MM = 160.0


def load(path):
    return json.loads(Path(path).read_text())


def home_pose():
    return np.asarray(load(MEASUREMENT_FILE)['home']['flange_pose_mm_deg'], dtype=float)


def check_current(arm, reference, tolerance_mm=2.0):
    arm.ensure_idle()
    alarms(arm)
    p = arm.raw_flange_pose_read_only()
    if np.linalg.norm(p[:3] - reference[:3]) > tolerance_mm:
        raise SafetyFault('Robot not at expected pose')
    if np.max(np.abs(angle_error(p[3:], reference[3:]))) > 0.5:
        raise SafetyFault('Wrist orientation changed')
    return p


def segments(start, end, step_mm=MOVE_MM):
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    length = float(np.linalg.norm(b-a))
    n = max(1, int(math.ceil(length / step_mm)))
    return [a + (b-a)*i/n for i in range(1, n+1)]


def contact_check(home, contact):
    c = np.asarray(contact, dtype=float)
    if c.shape != (6,) or not np.isfinite(c).all():
        raise SafetyFault('Invalid contact pose')
    if np.max(np.abs(angle_error(c[3:], home[3:]))) > 0.5:
        raise SafetyFault('Contact must have HOME wrist orientation')
    diff = c[:3]-home[:3]
    if not 50 < diff[1] < 320 or abs(diff[0]) > 125 or abs(diff[2]) > 2:
        raise SafetyFault('Contact pose outside intended horizontal task region')
    return c


def path_from_handover(current, home, contact):
    p = np.asarray(current, float)[:3]
    h = np.asarray(home, float)[:3]
    c = np.asarray(contact, float)[:3]
    if abs(p[2] - c[2]) > 2:
        raise SafetyFault('Cannot join contact path with different Z')
    lateral = np.array([c[0], p[1], p[2]], dtype=float)
    contact_xyz = np.array([c[0], c[1], p[2]], dtype=float)
    end = contact_xyz + np.array([0, PUSH_MM, 0], float)
    lifted = end + np.array([0, 0, LIFT_MM], float)
    retreat_y = np.array([end[0], h[1], lifted[2]], float)
    retreat_xy = np.array([h[0], h[1], lifted[2]], float)
    elevated_home = np.array([h[0], h[1], h[2] + LIFT_MM], float)
    pieces = [
        ('align', segments(p, lateral)),
        ('contact', segments(lateral, contact_xyz)),
        ('push', segments(contact_xyz, end)),
        ('lift', segments(end, lifted)),
        ('retreat', segments(lifted, retreat_y)),
        ('cross', segments(retreat_y, retreat_xy)),
        ('lower', segments(retreat_xy, elevated_home) + segments(elevated_home, h))
    ]
    last = h
    for _, pts in pieces:
        for q in pts:
            if not np.isfinite(q).all():
                raise SafetyFault('Nonfinite waypoint')
        if pts:
            last = pts[-1]
    if np.linalg.norm(last - h) > 1e-6 or abs(np.linalg.norm(end-contact_xyz) - PUSH_MM) > 1e-6:
        raise SafetyFault('Path endpoint mismatch')
    return pieces


def ik_check(arm, current, waypoints):
    near = arm.angles()
    prev = np.asarray(current, float)[:3]
    home_rpy = np.asarray(current, float)[3:]
    for q in waypoints:
        if np.linalg.norm(q - prev) > MOVE_MM + 0.02:
            raise SafetyFault('Path waypoint exceeds 2 mm')
        args = dict(zip(('x', 'y', 'z', 'rx', 'ry', 'rz'), [float(x) for x in q] + [float(x) for x in home_rpy]))
        args.update({'use_joint_near':'1', 'joint_near':'{' + ','.join(f'{v:.5f}' for v in near) + '}', 'user':'0', 'tool':'0'})
        candidate = parse_payload(arm.call('InverseKin', args, timeout_s=4.0).robot_return, 6)
        jump = np.max(np.abs(angle_error(candidate, near)))
        if jump > 2.0:
            raise SafetyFault(f'Inverse-kinematics joint change too large: {jump:.2f} degrees')
        near = candidate
        prev = q


def move_small(arm, start, delta, home):
    d = np.asarray(delta, float)
    if d.shape != (3,) or not np.isfinite(d).all() or np.linalg.norm(d) > MOVE_MM + 1e-6:
        raise SafetyFault('Movement larger than 2 mm')
    if np.max(np.abs(angle_error(start[3:], home[3:]))) > .5:
        raise SafetyFault('Wrist orientation mismatch')
    arm.ensure_idle()
    alarms(arm)
    expected = start[:3] + d
    arm.call('RelMovLUser', dict(zip(('a','b','c','d','e','f'), [float(d[0]),float(d[1]),float(d[2]),0.0,0.0,0.0])) | {'param_value': []}, timeout_s=5.0)
    deadline = time.monotonic() + 12
    consecutive = 0
    while time.monotonic() < deadline:
        mode = arm.robot_mode()
        if mode not in (5, 7):
            raise SafetyFault(f'Unsafe robot mode: {mode}')
        actual = arm.raw_flange_pose_read_only()
        if np.max(np.abs(angle_error(actual[3:],home[3:]))) > .5:
            raise SafetyFault('Wrist orientation changed')
        if np.linalg.norm(actual[:3] - expected) < 0.3 and mode == 5:
            consecutive += 1
            if consecutive >= 2:
                return actual
        else:
            consecutive = 0
        time.sleep(0.1)
    raise SafetyFault('Robot did not reach commanded position')


def verify_and_move_path(arm, current, home, pieces):
    waypoints = [x for _, pts in pieces for x in pts]
    ik_check(arm, current, waypoints)
    print('Preflight IK: PASS', flush=True)
    print('Fixed plan:', json.dumps({name: len(pts) for name,pts in pieces}), flush=True)
    if input('Entire arm, plate, camera, cable and box path clear? Type PUSH: ').strip() != 'PUSH':
        print('PUSH CANCELLED', flush=True)
        return
    pose = check_current(arm, current, tolerance_mm=0.5)
    for name, pts in pieces:
        for point in pts:
            delta = point - pose[:3]
            pose = move_small(arm, pose, delta, home)
        print('PHASE COMPLETE:', name, flush=True)
    check_current(arm, home, tolerance_mm=1.0)
    print('HOME REACHED', flush=True)


def get_contact(color):
    if not CONTACT_FILE.exists():
        raise SafetyFault('No contact pose saved. Use --teach-contact ' + color)
    data = load(CONTACT_FILE)
    if color not in data:
        raise SafetyFault('No contact pose saved for ' + color + '. Use --teach-contact ' + color)
    return contact_check(home_pose(), data[color]['flange_pose_mm_deg'])


def teach(color, arm):
    h = home_pose()
    pose = arm.raw_flange_pose_read_only()
    pose = contact_check(h, pose)
    print('Current contact candidate:', np.round(pose, 3).tolist())
    print('This must be the plate CONTACT pose behind the stationary', color, 'box.')
    if input('Type SAVE to record the manually positioned contact: ').strip() != 'SAVE':
        print('CANCELLED')
        return
    data = load(CONTACT_FILE) if CONTACT_FILE.exists() else {}
    data[color] = {'flange_pose_mm_deg':pose.tolist(), 'home_pose_mm_deg':h.tolist()}
    CONTACT_FILE.write_text(json.dumps(data,indent=2)+'\n')
    print('Saved', color, 'contact pose; no robot motion commanded')


def one_cycle(color, arm, cfg, policy, camera, home, contact, *, execute, max_ppo_mm):
    start = check_current(arm, home)
    pose = start.copy()
    traveled = 0.0
    last_valid = False
    stop_reason = None
    for step in range(int(math.ceil(max_ppo_mm / STEP_MM))):
        progress = pose[:3] - home[:3]
        try:
            frame, found, obs, action, delta, visibility = read_policy(camera, cfg, policy, color, progress)
            target = found.get(color)
            if target is None or not visibility[color]['visible']:
                raise VisionStop('Target not reliably visible')
            last_valid = True
            depth = float(target['depth_m'])
            if depth <= HANDOVER_DEPTH_M:
                stop_reason = f'camera depth {depth:.3f} m'
                break
            print(json.dumps({'step':step, 'target':color, 'depth_m':round(depth,4), 'ppo':np.round(action,3).tolist(), 'delta_mm':np.round(delta,3).tolist()}), flush=True)
            if not execute:
                print('PREVIEW: No motion; run --execute for physical approach')
                return
            next_xyz = pose[:3] + delta
            if np.linalg.norm(next_xyz - home[:3]) > max_ppo_mm + 1.0:
                stop_reason = 'PPO travel limit'
                break
            pose = move_small(arm, pose, delta, home)
            traveled += float(np.linalg.norm(delta))
        except VisionStop as exc:
            stop_reason = str(exc)
            break
    if not last_valid:
        raise SafetyFault('No valid target observation for handover')
    if stop_reason is None:
        stop_reason = 'PPO travel limit'
    print('PPO OFF. Handover:', stop_reason, 'PPO travel (mm):',round(traveled,2), flush=True)
    parts = path_from_handover(pose, home, contact)
    print('Contact pose:',np.round(contact[:3],2).tolist(), 'push: 100 mm +Y; lift: 160 mm +Z; return HOME', flush=True)
    if execute:
        verify_and_move_path(arm, pose, home, parts)
    return pose, parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teach-contact', choices=['red','blue'])
    parser.add_argument('--target', choices=['red','blue'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--max-ppo-mm',type=float, default=10.0)
    parser.add_argument('--status',action='store_true')
    args = parser.parse_args()
    h = home_pose()
    if args.status:
        print('HOME:',np.round(h,2).tolist())
        print('RED contact saved:',CONTACT_FILE.exists() and 'red' in load(CONTACT_FILE))
        print('BLUE contact saved:',CONTACT_FILE.exists() and 'blue' in load(CONTACT_FILE))
        print('PPO maximum per step:', STEP_MM, 'mm; fixed push:', PUSH_MM,'mm')
        return
    if not 0.5 <= args.max_ppo_mm <= PPO_LIMIT_MM:
        parser.error('--max-ppo-mm must be 0.5 to 40')
    if args.teach_contact and args.target:
        parser.error('Choose either --teach-contact or --target')
    if not args.teach_contact and not args.target:
        parser.error('Specify --teach-contact or --target')
    cfg = load(ROOT / 'config/deployment.json')
    arm = ROSArm(cfg)
    policy = None
    try:
        if args.teach_contact:
            teach(args.teach_contact, arm)
            return
        contact = get_contact(args.target)
        check_current(arm, h)
        policy = Policy(ROOT)
        with D405(cfg['camera']) as camera:
            print('TARGET:',args.target,'PHYSICAL MOTION:',args.execute, flush=True)
            if args.execute and input('Clear PPO approach path? Type MOVE: ').strip() != 'MOVE':
                print('CANCELLED')
                return
            one_cycle(args.target, arm, cfg, policy, camera, h, contact, execute=args.execute, max_ppo_mm=args.max_ppo_mm)
    except (Exception, KeyboardInterrupt) as exc:
        print('STOP:',exc,flush=True)
        arm.stop_best_effort()
        raise SystemExit(2)
    finally:
        if policy is not None:
            policy.close()
        arm.close()


if __name__ == '__main__':
    main()
