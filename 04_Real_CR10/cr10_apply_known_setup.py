import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    deployment = ROOT / 'config' / 'deployment.json'
    hybrid = ROOT / 'config' / 'cr10_real_hybrid.json'
    measurements = ROOT / 'cr10_motion_measurements.json'
    for required in (deployment, hybrid, measurements):
        if not required.exists():
            raise SystemExit('Missing: ' + str(required))
    cfg = load(deployment)
    h = load(hybrid)
    m = load(measurements)
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    for path in (deployment, hybrid):
        shutil.copy2(path, path.with_suffix('.backup_' + now + '.json'))
    f = np.asarray(m['candidate_forward_axis_base'], float)
    l = np.asarray(m['candidate_lateral_axis_base'], float)
    if f.shape != (3,) or l.shape != (3,) or not np.isfinite(f).all() or not np.isfinite(l).all() or abs(float(f @ l)) > 0.01:
        raise SystemExit('Invalid measured directions')
    if not (np.allclose(f, [0, 1, 0], atol=0.005) and np.allclose(l, [-1, 0, 0], atol=0.005)):
        raise SystemExit('Measured directions differ from the verified motion test')
    cfg['task']['forward_axis_base'] = f.tolist()
    cfg['task']['lateral_axis_base'] = l.tolist()
    cfg['safety']['box_lateral_width_m'] = 0.10
    cfg['safety']['pusher_lateral_width_m'] = 0.09
    h.update({
        'box_width_m': 0.10,
        'box_height_m': 0.14,
        'boxes_per_stack': 2,
        'box_count': 4,
        'stack_colors': ['red', 'blue'],
        'approx_home_plate_to_box_face_m': 0.28,
        'fixed_push_m': 0.10,
        'plate_width_m': 0.09,
        'plate_height_m': 0.09,
        'plate_thickness_m': 0.0045,
        'camera_body_projects_beyond_plate_m_approx': 0.08,
        'handover_gap_m': 0.04,
        'lateral_tolerance_m': 0.01,
        'vertical_tolerance_m': 0.015,
    })
    save(deployment, cfg)
    save(hybrid, h)
    print('Saved known box/plate geometry and measured task axes.')
    print('Stacks: red=2 boxes, blue=2 boxes; each stack 100 x 100 x 140 mm.')
    print('Approximate starting gap: 280 mm; requested fixed push: 100 mm.')
    print('Handover 40 mm and tolerances 10/15 mm are provisional control choices.')
    residual = m.get('rotation_residual_m')
    if residual is not None:
        print('Camera motion-fit residual:', round(float(residual) * 1000, 2), 'mm')
    if args.capture:
        from arm_bridge.ros_arm import ROSArm
        from arm_bridge.perception import D405, find_colors
        arm = ROSArm(cfg)
        try:
            pose = arm.raw_flange_pose_read_only()
            mode = arm.robot_mode()
            joints = arm.angles()
            with D405(cfg['camera']) as camera:
                frame = camera.read()
                detections = find_colors(frame, cfg['camera'], rs_module=camera.rs)
            record = {
                'robot_mode': int(mode),
                'flange_pose_mm_deg': pose.tolist(),
                'joints_deg': joints.tolist(),
                'detections': {c: None if d is None else {
                    'optical_point_m': np.asarray(d['point_optical_m']).tolist(),
                    'depth_m': float(d['depth_m']),
                    'pixels': int(d['pixels']),
                } for c, d in detections.items()},
                'geometry': {k: h[k] for k in (
                    'box_width_m', 'box_height_m', 'approx_home_plate_to_box_face_m',
                    'fixed_push_m', 'camera_body_projects_beyond_plate_m_approx')},
            }
            path = ROOT / 'cr10_current_geometry_capture.json'
            save(path, record)
            print('Read-only capture:', path)
            for color, det in record['detections'].items():
                print(color, 'not detected' if det is None else 'depth=' + str(round(det['depth_m'], 4)) + ' m')
        finally:
            arm.close()
    print('Remaining unknown: 3D optical-center and plate-contact offsets; independent validation; verified full-arm path.')
    print('No robot motion commanded. Motion approval flags were not changed.')


if __name__ == '__main__':
    main()
