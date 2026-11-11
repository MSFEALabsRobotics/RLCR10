#!/usr/bin/env python3
"""Live D405 -> original frozen PPO -> ONE supervised diagnostic CR10 step.

This is NOT a calibrated pushing controller. Robot-base XY are TEST axes,
NOT the policy's physical forward/lateral axes. This script intentionally
refuses to run when boxes are close, robot pose differs from the previously
recorded starting pose, either color is lost, or the robot is not enabled/idle.

Usage: copy into CR10_D405_PLATE_PPO_OFFLINE_READY_v0_3 and run there.
"""
import json
import time
from pathlib import Path

import numpy as np

from arm_bridge.core import SafetyFault
from arm_bridge.perception import D405, find_colors
from arm_bridge.policy import Policy
from arm_bridge.ros_arm import ROSArm

ROOT = Path(__file__).resolve().parent
CAMERA_BEHIND_MM = 23.0
STEP_MM = 0.5
# The user's physically established starting pose. Used as a tight guard,
# not as an autonomous movement target.
REFERENCE_FLANGE = np.array([257.7098, 482.4614, -51.2743,
                             89.9215, 3.4816, 176.2373], dtype=float)


def assert_no_alarms(arm):
    response = arm.call('GetErrorID').robot_return.strip()
    if not (response.startswith('{') and response.endswith('}')):
        raise SafetyFault('Empty or malformed GetErrorID response')
    try:
        alarms = json.loads(response[1:-1])
    except (ValueError, TypeError) as exc:
        raise SafetyFault('Cannot parse GetErrorID') from exc
    if not isinstance(alarms, list) or len(alarms) != 7 or any(alarms):
        raise SafetyFault(f'Active alarm: {response}')


def inspect_flange(arm):
    arm.ensure_idle()
    assert_no_alarms(arm)
    p = arm.raw_flange_pose_read_only()
    if not np.all(np.isfinite(p)):
        raise SafetyFault('Invalid robot pose')
    # Tight reference pose check limits unexpected motion from a different arm configuration.
    if np.linalg.norm(p[:3] - REFERENCE_FLANGE[:3]) > 20.0:
        raise SafetyFault('Robot is >20mm from saved candidate approach pose')
    angular = (p[3:] - REFERENCE_FLANGE[3:] + 180.0) % 360.0 - 180.0
    if np.max(np.abs(angular)) > 2.0:
        raise SafetyFault('Flange orientation differs >2 degrees from reference')
    clearance = float(p[2] + 134.5)  # approximate only, based on floor measurements
    if not (65.0 <= clearance <= 125.0):
        raise SafetyFault('Nominal flange/table clearance outside reference range')
    return p


def main():
    cfg = json.loads((ROOT / 'config/deployment.json').read_text())
    print('LIVE D405 -> FROZEN PPO -> 0.5 mm TEST ONLY', flush=True)
    print('UNVERIFIED: PPO axes vs robot-base XY. THIS IS NOT A PUSH.', flush=True)
    policy = Policy(ROOT)
    arm = None
    try:
        # No robot service is ever called before a valid PPO observation exists.
        with D405(cfg['camera']) as camera:
            samples = []
            for _ in range(3):
                frame = camera.read()
                detections = find_colors(frame, cfg['camera'], rs_module=camera.rs)
                if detections['red'] is None or detections['blue'] is None:
                    raise SafetyFault('Red/blue box lost in RGB-D')
                obs = [0.0, 0.0]
                forward_distances = []
                for color in ('red', 'blue'):
                    det = detections[color]
                    xyz = np.asarray(det['point_optical_m'], dtype=float)
                    forward = float(xyz[2] - CAMERA_BEHIND_MM * 0.001)
                    forward_distances.append(forward)
                    # Keep original nominal camera -> observation convention, uncalibrated.
                    obs.extend([forward, -float(xyz[0]), float(det['depth_m']), 1.0])
                obs.extend([1.0, 0.0]) # red target only for this diagnostic
                raw = np.asarray(obs, dtype=np.float32)
                # Avoid test near objects; cannot yet establish wrong-box sweep geometry.
                if min(forward_distances) < 0.15:
                    raise SafetyFault('Boxes too near for uncalibrated diagnostic (min 150mm)')
                action = policy.predict(raw)
                samples.append((frame.frame_number, raw, action))
            # Require reasonable stationary detections before a motion command.
            observations = np.stack([s[1] for s in samples])
            for col in (2, 3, 6, 7):
                if float(np.ptp(observations[:, col])) > 0.04:
                    raise SafetyFault('RGB-D positions unstable across 3 frames')
            selected = samples[-1]
            action = np.asarray(selected[2], dtype=float)
            delta_mm = action / max(1.0, float(np.linalg.norm(action))) * STEP_MM
            if np.linalg.norm(delta_mm) > STEP_MM + 1e-9:
                raise SafetyFault('Unexpected oversized step')

        arm = ROSArm(cfg)
        before = inspect_flange(arm)
        print('Camera frame:', selected[0], flush=True)
        print('Raw observation:', selected[1].tolist(), flush=True)
        print('PPO action:', action.tolist(), flush=True)
        print('CR10 current flange:', before.tolist(), flush=True)
        print('ONE proposed TEST displacement in robot BASE X/Y (mm):', delta_mm.tolist(), flush=True)
        print('This is NOT a calibrated push direction. Boxes must be well clear.', flush=True)
        choice = input('If mount/cable/arm clearance is checked, type MOVE for exactly ONE 0.5mm test: ').strip()
        if choice != 'MOVE':
            print('CANCELLED. Zero robot commands.', flush=True)
            return
        now = inspect_flange(arm)
        if float(np.linalg.norm(now[:3] - before[:3])) > 0.25:
            raise SafetyFault('Arm moved since confirmation')
        arm.call('SpeedFactor', {'ratio': 1})
        response = arm.call('RelMovLUser', {
            'a': float(delta_mm[0]), 'b': float(delta_mm[1]),
            'c': 0.0, 'd': 0.0, 'e': 0.0, 'f': 0.0,
            'param_value': [],
        }, timeout_s=5.0)
        print('ROS command accepted, res:', response.res, flush=True)
        time.sleep(3.0)
        after = arm.raw_flange_pose_read_only()
        print('Measured base XYZ displacement (mm):',
              np.round(after[:3] - before[:3], 4).tolist(), flush=True)
        print('Robot mode:', arm.robot_mode(), flush=True)
        assert_no_alarms(arm)
        if float(np.linalg.norm(after[:3] - before[:3] - np.r_[delta_mm, 0.0])) > 0.5:
            raise SafetyFault('Commanded vs measured displacement mismatch')
        print('PASS: Single real-camera-derived PPO action executed. NOT autonomous pushing.', flush=True)
    except (SafetyFault, RuntimeError, KeyboardInterrupt) as exc:
        print('[STOP/REJECT]', str(exc), flush=True)
        raise SystemExit(2)
    finally:
        if arm is not None:
            arm.close()
        policy.close()

if __name__ == '__main__':
    main()
