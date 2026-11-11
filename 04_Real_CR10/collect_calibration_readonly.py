#!/usr/bin/env python3
"""Collect and evaluate CR10/D405 physical point correspondences; never command motion."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

from arm_bridge.perception import D405, find_colors
from arm_bridge.ros_arm import ROSArm
from arm_bridge.core import SafetyFault
from calibrate_from_pairs import rigid_fit

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'config/deployment.json'
DEFAULT_SAMPLES = ROOT / 'calibration_samples.json'


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def six_pose(arm):
    return np.asarray(arm.raw_flange_pose_read_only(), dtype=float)


def assert_stationary(p0, p1, message):
    if np.linalg.norm(p0[:3] - p1[:3]) > 0.5:
        raise SafetyFault(f'{message}: flange translation exceeds 0.5 mm')
    error = (p0[3:] - p1[3:] + 180) % 360 - 180
    if np.max(np.abs(error)) > 0.2:
        raise SafetyFault(f'{message}: flange orientation changed more than 0.2 degrees')


def read_optical_point(camera, cam_cfg, color, pixel):
    frame = camera.read()
    if color:
        detection = find_colors(frame, cam_cfg, rs_module=camera.rs).get(color)
        if detection is None:
            raise SafetyFault(f'{color} marker not detected in RGB-D frame')
        return np.asarray(detection['point_optical_m'], dtype=float), {
            'source': f'color_mask_{color}',
            'uv_policy_px': detection['uv_policy_px'],
            'depth_m': detection['depth_m'],
            'frame_number': frame.frame_number,
        }
    u, v = pixel
    h, w = frame.depth_m.shape
    if not (0 <= u < w and 0 <= v < h):
        raise SafetyFault(f'pixel ({u},{v}) outside {w}x{h} image')
    # Use depth at exactly the selected pixel. Do not silently shift to another surface.
    depth = float(frame.depth_m[v, u])
    if not (cam_cfg['min_valid_depth_m'] <= depth <= cam_cfg['max_valid_depth_m']):
        raise SafetyFault(f'invalid depth at pixel ({u},{v}): {depth}')
    point = camera.rs.rs2_deproject_pixel_to_point(
        frame.intrinsics, [float(u), float(v)], depth
    )
    return np.asarray(point, dtype=float), {
        'source': 'pixel', 'uv_full_px': [u, v],
        'depth_m': depth, 'frame_number': frame.frame_number,
    }


def capture(args):
    data = read_json(args.output) if args.output.exists() else {'schema': 1, 'samples': []}
    if data.get('schema') != 1 or not isinstance(data.get('samples'), list):
        raise ValueError('unknown calibration samples schema')
    if any(rec['name'] == args.name for rec in data['samples']):
        raise ValueError(f'duplicate sample name {args.name!r}; use a unique name')
    cfg = read_json(args.config)
    arm = None
    try:
        arm = ROSArm(cfg)
        before = six_pose(arm)
        with D405(cfg['camera']) as camera:
            points, details = [], []
            for _ in range(args.frames):
                point, info = read_optical_point(camera, cfg['camera'], args.color, args.pixel)
                if not np.all(np.isfinite(point)):
                    raise SafetyFault('nonfinite D405 optical point')
                points.append(point)
                details.append(info)
        after = six_pose(arm)
        assert_stationary(before, after, 'During RGB-D capture')
        if data['samples']:
            assert_stationary(
                np.asarray(data['samples'][0]['flange_pose_mm_deg'], dtype=float),
                before, 'Camera moved relative to saved calibration baseline'
            )
        points = np.asarray(points)
        mean = np.median(points, axis=0)
        jitter = float(np.sqrt(np.mean(np.sum((points - mean)**2, axis=1))))
        if jitter > 0.005:
            raise SafetyFault(f'RGB-D position jitter {jitter*1000:.1f}mm >5mm; no sample saved')
        rec = {
            'name': args.name,
            'set': args.set,
            'camera_optical_m': np.round(mean, 8).tolist(),
            'robot_base_m': [float(x) for x in args.base_m],
            'flange_pose_mm_deg': before.tolist(),
            'camera_jitter_rmse_m': jitter,
            'measurement_details': details,
            'captured_utc': datetime.now(timezone.utc).isoformat(),
        }
        data['samples'].append(rec)
        save_json(args.output, data)
        print(json.dumps({
            'saved_to': str(args.output), 'name': args.name,
            'set': args.set, 'camera_optical_m': rec['camera_optical_m'],
            'robot_base_m': rec['robot_base_m'],
            'jitter_mm': round(jitter * 1000, 3),
            'read_only': True,
        }, indent=2))
    finally:
        if arm is not None:
            arm.close()


def fit(args):
    data = read_json(args.samples)
    records = data['samples']
    train = [x for x in records if x['set'] == 'fit']
    hold = [x for x in records if x['set'] == 'holdout']
    if len(train) < 4:
        raise ValueError('need >=4 physical fit points, non-collinear')
    if len(hold) < 2:
        raise ValueError('need >=2 independent physical holdout points')
    reference = np.asarray(records[0]['flange_pose_mm_deg'], dtype=float)
    for rec in records:
        assert_stationary(reference, np.asarray(rec['flange_pose_mm_deg'], float),
                          f'camera baseline inconsistent for {rec["name"]}')
    src = np.asarray([x['camera_optical_m'] for x in train], float)
    dst = np.asarray([x['robot_base_m'] for x in train], float)
    T, fit_rmse = rigid_fit(src, dst)
    src_h = np.asarray([x['camera_optical_m'] for x in hold], float)
    dst_h = np.asarray([x['robot_base_m'] for x in hold], float)
    err = np.linalg.norm((T[:3, :3] @ src_h.T).T + T[:3, 3] - dst_h, axis=1)
    holdout_rmse = float(np.sqrt(np.mean(err**2)))
    report = {
        'candidate_T_base_camera_at_baseline': np.round(T, 9).tolist(),
        'fit_rmse_m': fit_rmse,
        'holdout_rmse_m': holdout_rmse,
        'holdout_error_m_by_name': {x['name']: float(e) for x, e in zip(hold, err)},
        'baseline_raw_flange_pose_mm_deg_NOT_pusher_TCP': reference.tolist(),
        'numeric_holdout_within_2mm': bool(holdout_rmse <= 0.002),
        'calibration_approved': False,
        'robot_motion_authorized': False,
        'note': 'Candidate only. Physical geometry, independently confirmed base coordinates, pusher TCP, axes, joint limits, workspace and collision protection still require validation.',
    }
    save_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report['numeric_holdout_within_2mm'] else 2


def status(args):
    cfg = read_json(args.config)
    requirements = {
        'task.forward_axis_base': cfg['task']['forward_axis_base'] is not None,
        'task.lateral_axis_base': cfg['task']['lateral_axis_base'] is not None,
        'calibration.baseline_tcp_pose_mm_deg': cfg['calibration']['baseline_tcp_pose_mm_deg'] is not None,
        'calibration.T_base_camera_at_baseline': cfg['calibration']['T_base_camera_at_baseline'] is not None,
        'calibration.approved': cfg['calibration']['approved'] is True,
        'calibration.holdout_rmse_m': cfg['calibration']['holdout_rmse_m'] is not None,
        'robot.tool_frame': cfg['robot']['tool_frame'] is not None,
        'robot.joint_limits_deg': cfg['robot']['joint_limits_deg'] is not None,
        'safety.workspace_min_base_m': cfg['safety']['workspace_min_base_m'] is not None,
        'safety.workspace_max_base_m': cfg['safety']['workspace_max_base_m'] is not None,
        'safety.box_lateral_width_m': cfg['safety']['box_lateral_width_m'] is not None,
        'safety.full_robot_collision_model_validated': cfg['safety']['full_robot_collision_model_validated'] is True,
        'safety.pusher_plate_box_clearance_validated': cfg['safety']['pusher_plate_box_clearance_validated'] is True,
        'safety.operator_has_verified_robot_protection': cfg['safety']['operator_has_verified_robot_protection'] is True,
        'safety.real_motion_authorized': cfg['safety']['real_motion_authorized'] is True,
    }
    print('READ-ONLY deployment gate audit (no robot or camera access):')
    for key, ok in requirements.items():
        print(('OK     ' if ok else 'MISSING') + '  ' + key)
    print(f'{sum(requirements.values())}/{len(requirements)} fields populated/affirmed')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    s = subs.add_parser('status', help='list missing physical deployment fields; no hardware access')
    s.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    s.set_defaults(function=status)
    c = subs.add_parser('capture', help='record one known physical 3D point; robot read-only')
    c.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    c.add_argument('--output', type=Path, default=DEFAULT_SAMPLES)
    c.add_argument('--name', required=True)
    c.add_argument('--set', choices=('fit', 'holdout'), default='fit')
    c.add_argument('--base-m', type=float, nargs=3, metavar=('X', 'Y', 'Z'), required=True,
                   help='independently measured same surface point in robot base coordinates [m]')
    selector = c.add_mutually_exclusive_group(required=True)
    selector.add_argument('--color', choices=('red', 'blue'), help='median RGB-D visible colored surface')
    selector.add_argument('--pixel', nargs=2, type=int, metavar=('U', 'V'), help='color-image pixel [full resolution]')
    c.add_argument('--frames', type=int, default=5)
    c.set_defaults(function=capture)
    f = subs.add_parser('fit', help='estimate unapproved transform and independent holdout error')
    f.add_argument('--samples', type=Path, default=DEFAULT_SAMPLES)
    f.add_argument('--output', type=Path, default=ROOT / 'candidate_calibration_report.json')
    f.set_defaults(function=fit)
    args = p.parse_args()
    if args.command == 'capture' and args.frames not in range(3, 21):
        p.error('--frames must be 3..20')
    try:
        return args.function(args)
    except (ValueError, KeyError, RuntimeError, SafetyFault, OSError) as err:
        print(f'[STOP/REJECT] {err}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
