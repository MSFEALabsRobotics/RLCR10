#!/usr/bin/env python3
"""Four-action, supervised, NO-CONTACT CR10/D405/PPO communication trial.

Copy beside run.py in CR10_D405_PLATE_PPO_OFFLINE_READY_v0_3.
    python3 cr10_ppo_four_step_clearance_test.py --execute

This is NOT the calibrated pushing controller. Inputs retain the previous
single-step diagnostic's nominal camera convention; output axes are BASE X/Y
TEST axes. Neither is a calibrated deployment mapping. Poor actions therefore
must be confined to an operator-verified clear region. No pushing/contact is
permitted. The original deployment.json and its approval gates are untouched.

Four commands maximum, <=0.5 mm each, <=2 mm commanded path, all commanded
positions within 2 mm of the exact flange pose supplied on 2026-11-10.
Fixed flange orientation; explicit User=0/Tool=0; nearest-IK checks; small
joint-change bounds; pose/joint/alarm feedback; one command at a time.
Software bounds and best-effort Stop are NOT safety-rated protection.
The operator must verify clearance of the arm, plate, mount and cable, keep
people clear, use the controller's protective settings, and keep E-stop ready.
Do not increase limits or reuse after moving/remounting the setup.

Physical operation has NOT been tested by the author. Only offline mock checks.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import time
from pathlib import Path

import numpy as np

from arm_bridge.core import SafetyFault
from arm_bridge.perception import D405, find_colors
from arm_bridge.policy import Policy
from arm_bridge.ros_arm import ROSArm, parse_payload

ROOT = Path(__file__).resolve().parent
REFERENCE_POSE = np.array([258.8247, 481.4442, -51.2842,
                           89.9245, 3.4817, 176.2381])
REFERENCE_JOINTS = np.array([80.8196, -39.8883, -138.4963,
                             -1.7611, -95.3618, 356.4096])
STEPS = 4
STEP_MM = 0.5
TOTAL_MM = 2.0
ANGLE_TOL_DEG = 0.05
JOINT_STEP_DEG = 0.10
JOINT_REFERENCE_DEG = 0.25
POSITION_TOL_MM = 0.05
MOVE_TIMEOUT_S = 25.0


def check_reply(reply, label):
    text = str(getattr(reply, 'robot_return', '')).strip()
    code = re.match(r'^\s*(-?\d+)\s*,', text)
    if code and int(code.group(1)) != 0:
        raise SafetyFault(f'{label}: controller rejected command: {text}')


def alarms(arm):
    reply = arm.call('GetErrorID').robot_return.strip()
    if not (reply.startswith('{') and reply.endswith('}')):
        raise SafetyFault('Missing/malformed alarm response')
    errors = json.loads(reply[1:-1])
    if not isinstance(errors, list) or len(errors) != 7 or any(errors):
        raise SafetyFault(f'Controller alarm: {reply}')


def check_state(pose, joints):
    pose, joints = np.asarray(pose, float), np.asarray(joints, float)
    if pose.shape != (6,) or joints.shape != (6,) or not np.all(np.isfinite(np.r_[pose, joints])):
        raise SafetyFault('Invalid robot feedback')
    if np.linalg.norm(pose[:3] - REFERENCE_POSE[:3]) > TOTAL_MM + 1e-8:
        raise SafetyFault('Outside fixed 2 mm reference region; no automatic repositioning')
    if abs(pose[2] - REFERENCE_POSE[2]) > POSITION_TOL_MM:
        raise SafetyFault('Flange height changed')
    angles = (pose[3:] - REFERENCE_POSE[3:] + 180.0) % 360.0 - 180.0
    if np.max(np.abs(angles)) > ANGLE_TOL_DEG:
        raise SafetyFault('Flange orientation changed')
    if np.max(np.abs(joints - REFERENCE_JOINTS)) > JOINT_REFERENCE_DEG:
        raise SafetyFault('Joint posture differs from recorded reference')


def read_state(arm, idle=True):
    mode = arm.robot_mode()
    if (idle and mode != 5) or (not idle and mode not in (5, 7)):
        raise SafetyFault(f'Unexpected robot mode {mode}')
    alarms(arm)
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
        if srv is None or any(not hasattr(srv.Request(), f) for f in fields):
            raise SafetyFault(f'Installed ROS interface mismatch: {label}')
        if label not in arm.clients:
            client = arm.node.create_client(srv, f'{arm.prefix}/{label}')
            if not client.wait_for_service(timeout_sec=2.0):
                raise SafetyFault(f'Required service unavailable: {label}')
            arm.clients[label] = client


def camera_observation(camera, cfg, command_m):
    samples = []
    for _ in range(3):
        frame = camera.read()
        detections = find_colors(frame, cfg['camera'], rs_module=camera.rs)
        raw = [float(command_m[0]), float(command_m[1])]
        for color in ('red', 'blue'):
            det = detections[color]
            if det is None:
                raise SafetyFault(f'{color} detection lost')
            xyz = np.asarray(det['point_optical_m'], float)
            forward = float(xyz[2] - 0.023)  # SAME provisional diagnostic convention.
            if forward < 0.15:
                raise SafetyFault('Boxes too near: NO-CONTACT test requires >=150 mm')
            raw.extend([forward, -float(xyz[0]), float(det['depth_m']), 1.0])
        raw.extend([1.0, 0.0])
        samples.append(np.asarray(raw, np.float32))
    values = np.stack(samples)
    if not np.all(np.isfinite(values)):
        raise SafetyFault('Nonfinite camera observation')
    if np.any(np.ptp(values[:, [2, 3, 6, 7]], axis=0) > 0.04):
        raise SafetyFault('RGB-D positions unstable across 3 frames (40 mm gate unchanged)')
    return samples[-1], frame.frame_number


def plan_step(arm, pose, joints, action, path_mm):
    action = np.asarray(action, float)
    if action.shape != (2,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.000001):
        raise SafetyFault('Invalid policy action')
    delta = np.clip(action, -1, 1) / max(1.0, float(np.linalg.norm(action))) * STEP_MM
    length = float(np.linalg.norm(delta))
    if path_mm + length > TOTAL_MM + 1e-9:
        raise SafetyFault('2 mm total commanded-path limit')
    target = pose.copy()
    target[:2] += delta
    check_state(target, joints)
    req = dict(zip(('x', 'y', 'z', 'rx', 'ry', 'rz'), map(float, target)))
    req.update(user='0', tool='0', use_joint_near='1',
               joint_near='{' + ','.join(f'{v:.6f}' for v in joints) + '}')
    ik = parse_payload(arm.call('InverseKin', req).robot_return, 6)
    if np.max(np.abs(ik - joints)) > JOINT_STEP_DEG:
        raise SafetyFault('IK requires >0.10 degree joint change; no motion')
    check_state(target, ik)
    return target, length


def move_and_verify(arm, before, before_joints, target):
    current, current_joints, _ = read_state(arm)
    if np.linalg.norm(current[:3] - before[:3]) > POSITION_TOL_MM or np.max(np.abs(current_joints - before_joints)) > 0.02:
        raise SafetyFault('Robot moved after planning')
    fields = dict(zip(('a', 'b', 'c', 'd', 'e', 'f'), map(float, target)))
    fields.update(mode=False, param_value=['user=0', 'tool=0', 'speed=5', 'a=1', 'cp=0'])
    check_reply(arm.call('MovL', fields, timeout_s=3.0), 'MovL')
    deadline, stable = time.monotonic() + MOVE_TIMEOUT_S, 0
    while time.monotonic() < deadline:
        actual, joints, mode = read_state(arm, idle=False)
        if np.max(np.abs(joints - before_joints)) > JOINT_STEP_DEG + 0.01:
            raise SafetyFault('Measured joint change too large')
        if np.linalg.norm(actual[:3] - before[:3]) > STEP_MM + POSITION_TOL_MM:
            raise SafetyFault('Measured step too large')
        delta = target[:3] - before[:3]
        length = float(np.linalg.norm(delta))
        if length > 1e-9:
            direction = delta / length
            actual_delta = actual[:3] - before[:3]
            forward = float(actual_delta @ direction)
            lateral = float(np.linalg.norm(actual_delta - forward * direction))
            if forward < -POSITION_TOL_MM or forward > length + POSITION_TOL_MM or lateral > POSITION_TOL_MM:
                raise SafetyFault('Movement departed from short planned segment')
        angular = (actual[3:] - target[3:] + 180) % 360 - 180
        done = np.linalg.norm(actual[:3] - target[:3]) <= POSITION_TOL_MM and np.max(np.abs(angular)) <= ANGLE_TOL_DEG and mode == 5
        stable = stable + 1 if done else 0
        if stable >= 3:
            return actual
        time.sleep(0.08)
    raise SafetyFault('Motion timeout: queued motion status uncertain')


def trial(arm, camera, policy, cfg, execute, confirm=input):
    start, _, _ = read_state(arm)
    raw, _ = camera_observation(camera, cfg, [0.0, 0.0])
    policy.predict(raw)  # Check observation/normalizer/model before any motion.
    if execute:
        print('FOUR-STEP NO-CONTACT TEST. Uncalibrated BASE X/Y test axes.', flush=True)
        print('Max 0.5 mm/command, 2 mm commanded path, fixed reference region.', flush=True)
        print('Verify arm/plate/camera/cable clearance, boxes >=15 cm away, people clear,', flush=True)
        print('controller protective limits active and E-stop in hand. NO touching/pushing.', flush=True)
        if confirm('Type CLEAR to confirm this clear-space test, or Enter to cancel: ').strip() != 'CLEAR':
            print('CANCELLED: no motion commands.', flush=True)
            return
        current, _, _ = read_state(arm)
        if np.linalg.norm(current[:3] - start[:3]) > POSITION_TOL_MM:
            raise SafetyFault('Pose changed during confirmation')
        check_reply(arm.call('SpeedFactor', {'ratio': 1}), 'SpeedFactor')
        # A new observation is acquired after confirmation and after every move.
    path_mm = 0.0
    previous = None
    for step in range(STEPS):
        pose, joints, _ = read_state(arm)
        command_m = (pose[:2] - start[:2]) * 0.001
        raw, frame = camera_observation(camera, cfg, command_m)
        if previous is not None and np.any(np.abs(raw[[2, 3, 6, 7]] - previous[[2, 3, 6, 7]]) > 0.04):
            raise SafetyFault('Detection jumped between successive actions')
        previous = raw.copy()
        action = policy.predict(raw)
        target, length = plan_step(arm, pose, joints, action, path_mm)
        print(json.dumps({'step': step + 1, 'camera_frame': frame, 'raw_obs12': raw.tolist(),
                          'action2': action.tolist(), 'base_xy_test_delta_mm': (target[:2] - pose[:2]).tolist(),
                          'execute': execute}), flush=True)
        if execute and length > POSITION_TOL_MM:
            actual = move_and_verify(arm, pose, joints, target)
            print('Measured XYZ delta (mm):', np.round(actual[:3] - pose[:3], 4).tolist(), flush=True)
        path_mm += length
    print('DONE: four-action diagnostic ended. This does not establish calibrated pushing.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'config/deployment.json').read_text())
    if cfg['robot']['user_frame'] != 0:
        raise SafetyFault('This diagnostic requires robot-base user frame 0')
    arm = policy = None
    def interrupted(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        policy = Policy(ROOT)
        arm = ROSArm(cfg)
        require_services(arm)
        with D405(cfg['camera']) as camera:
            trial(arm, camera, policy, cfg, args.execute)
        return 0
    except BaseException as exc:
        print('[STOP/REJECT]', type(exc).__name__, str(exc), flush=True)
        if args.execute and arm is not None:
            arm.stop_best_effort()
        return 2
    finally:
        if policy is not None:
            policy.close()
        if arm is not None:
            arm.close()


if __name__ == '__main__':
    raise SystemExit(main())
