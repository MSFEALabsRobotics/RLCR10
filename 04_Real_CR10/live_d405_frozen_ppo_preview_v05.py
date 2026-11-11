#!/usr/bin/env python3
"""Live D405 -> frozen PPO prediction, nominal camera frame; NEVER sends robot commands.

Nominal only: this is not a calibrated 12D task observation. The D405 optical
frame is assumed to face parallel to the pushing direction at zero pitch, with
its optical center 23 mm behind the flange. Lateral sign must be calibrated.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description='Live D405 -> frozen PPO, NO robot commands')
    parser.add_argument('--target', choices=('red','blue'), default='red')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--lens-behind-flange-mm', type=float, default=23.0)
    parser.add_argument('--lateral-sign', type=int, choices=(-1, 1), default=-1,
                        help='Nominal only: camera optical X -> policy lateral sign; must be calibrated')
    args = parser.parse_args()
    if not 1 <= args.steps <= 80:
        parser.error('steps must be 1..80')
    if not 0 <= args.lens_behind_flange_mm <= 50:
        parser.error('lens offset outside expected v0.5 prototype range 0..50mm')

    root = Path(__file__).resolve().parent
    try:
        from arm_bridge.perception import D405, find_colors
        from arm_bridge.policy import Policy
        from arm_bridge.core import SafetyFault
    except ImportError as e:
        print(f'[STOP] Run this file inside CR10_D405_PLATE_PPO_OFFLINE_READY_v0_3: {e}', file=sys.stderr)
        return 2

    cfg = json.loads((root / 'config/deployment.json').read_text())
    print('MODE: LIVE_D405_TO_FROZEN_PPO_PREVIEW_ONLY, NO_ROBOT_COMMANDS', flush=True)
    print('WARNING: NOMINAL (UNCALIBRATED) frame. Not valid for physical control.', flush=True)
    print('WARNING: lateral sign, plate TCP, camera mounting alignment and camera-to-robot transform NOT verified.', flush=True)
    ppo = None
    try:
        ppo = Policy(root)
        with D405(cfg['camera']) as camera:
            for step in range(args.steps):
                frame = camera.read()
                det = find_colors(frame, cfg['camera'], rs_module=camera.rs)
                if any(det[c] is None for c in ('red','blue')):
                    print(json.dumps({'step':step, 'frame':frame.frame_number,
                                      'status':'REJECTED_ONE_OR_BOTH_COLORS_NOT_DETECTED',
                                      'red_pixels':None if det['red'] is None else det['red']['pixels'],
                                      'blue_pixels':None if det['blue'] is None else det['blue']['pixels'],
                                      'robot_motion':False}), flush=True)
                    continue
                obs = [0., 0.]
                for name in ('red', 'blue'):
                    point = np.asarray(det[name]['point_optical_m'], dtype=float)
                    depth = float(det[name]['depth_m'])
                    # Optical Z points forward. Nominal TCP is near the flange's front plane;
                    # lens is behind it. This is only a camera-space SURROGATE observation.
                    nominal_forward = float(point[2] - args.lens_behind_flange_mm * 0.001)
                    nominal_lateral = float(args.lateral_sign * point[0])
                    obs.extend([nominal_forward, nominal_lateral, depth, 1.])
                obs += [float(args.target == 'red'), float(args.target == 'blue')]
                raw = np.asarray(obs, dtype=np.float32)
                try:
                    action = ppo.predict(raw)
                    delta_mm = action / max(1., float(np.linalg.norm(action))) * 2.
                    status = 'NOMINAL_PPO_PREDICTION_ONLY'
                    result = {'action2':action.tolist(), 'proposed_delta_planar_mm_UNCALIBRATED':delta_mm.tolist()}
                except SafetyFault as e:
                    status = 'REJECTED_OBSERVATION_OUT_OF_TRAINING_BOUNDS'
                    result = {'reject_reason':str(e)}
                print(json.dumps({'step':step, 'frame':frame.frame_number,
                                  'status':status, 'raw_obs12_nominal':raw.tolist(),
                                  'red_depth_m':det['red']['depth_m'],
                                  'blue_depth_m':det['blue']['depth_m'],
                                  'red_mask_pixels':det['red']['pixels'],
                                  'blue_mask_pixels':det['blue']['pixels'],
                                  'robot_motion':False, **result}), flush=True)
        return 0
    except (RuntimeError, KeyboardInterrupt) as e:
        print(f'[STOP] {e}', file=sys.stderr)
        return 2
    finally:
        if ppo is not None:
            ppo.close()


if __name__ == '__main__':
    raise SystemExit(main())
