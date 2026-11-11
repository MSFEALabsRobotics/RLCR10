#!/usr/bin/env python3
"""One supervised PPO-derived 1 mm Cartesian jog for CR10 connection testing.

Not a camera-guided push and NOT a general-purpose motion controller.
Run from the CR10_D405_PLATE_PPO_OFFLINE_READY_v0_3 package directory.
"""
import json
import time
from pathlib import Path

import numpy as np
from arm_bridge.policy import Policy
from arm_bridge.ros_arm import ROSArm

root = Path.cwd()
cfg = json.loads((root / 'config' / 'deployment.json').read_text())
robot = ROSArm(cfg)
policy = None

def no_alarms():
    value = robot.call('GetErrorID').robot_return.strip()
    if not (value.startswith('{') and value.endswith('}')):
        raise RuntimeError(f'Unexpected alarm format: {value}')
    alarms = json.loads(value[1:-1])
    if not isinstance(alarms, list) or len(alarms) != 7 or any(alarms):
        raise RuntimeError(f'Active robot alarm: {value}')

try:
    robot.ensure_idle()
    no_alarms()
    before = robot.raw_flange_pose_read_only()
    joints = robot.angles()
    if not np.all(np.isfinite(np.r_[before, joints])):
        raise RuntimeError('Invalid robot feedback')

    # Measured table 729 mm, measured base datum 863.5 mm above floor.
    # This is a nominal clearance check, not full-body collision avoidance.
    nominal_clearance = float(before[2] + 134.5)
    if not (60.0 < nominal_clearance < 250.0):
        raise RuntimeError(f'Nominal flange clearance {nominal_clearance:.1f} mm outside 60..250 mm')

    policy = Policy(root)
    # The SAME synthetic red-left observation used in policy-check.
    obs = np.array([0, 0, .08, -.10, .17, 1,
                    .08, .10, .18, 1, 1, 0], dtype=np.float32)
    action = policy.predict(obs)
    step = action / max(1.0, float(np.linalg.norm(action)))  # 1 mm maximum
    print('Robot mode: 5, no alarms; current flange:', before.tolist(), flush=True)
    print('Nominal flange-to-table clearance (mm):', nominal_clearance, flush=True)
    print('PPO action [forward, lateral]:', action.tolist(), flush=True)
    print('Proposed one-step robot-BASE XY translation (mm):', step.tolist(), flush=True)
    print('NOTICE: robot-base XY are TEST axes, NOT calibrated PPO push axes.', flush=True)
    if input('Clear swept area and confirm ONE real 1 mm move: type MOVE: ').strip() != 'MOVE':
        print('Cancelled; NO motion sent.', flush=True)
    else:
        robot.ensure_idle()
        no_alarms()
        now = robot.raw_flange_pose_read_only()
        if np.linalg.norm(now[:3] - before[:3]) > 0.5:
            raise RuntimeError('Robot position changed while waiting; no motion sent')
        robot.call('SpeedFactor', {'ratio': 1})
        response = robot.call('RelMovLUser', {
            'a': float(step[0]), 'b': float(step[1]), 'c': 0.0,
            'd': 0.0, 'e': 0.0, 'f': 0.0, 'param_value': [],
        }, timeout_s=5.0)
        print('RelMovLUser service accepted; res:', response.res, flush=True)
        # Allow the motion to finish; only one command is ever sent.
        time.sleep(4.0)
        after = robot.raw_flange_pose_read_only()
        print('Measured Cartesian displacement (mm):',
              np.round(after[:3] - before[:3], 4).tolist(), flush=True)
        print('Robot mode:', robot.robot_mode(), flush=True)
        no_alarms()
        print('Finished; no other motion requested.', flush=True)
finally:
    if policy is not None:
        policy.close()
    robot.close()
