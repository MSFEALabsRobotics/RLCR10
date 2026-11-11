import json
from pathlib import Path
import numpy as np
from arm_bridge.ros_arm import ROSArm
from arm_bridge.perception import D405, find_colors

ROOT = Path(__file__).resolve().parent


def snapshot(arm, camera, config):
    if arm.robot_mode() not in (5, 6): raise RuntimeError('Stop robot motion before capturing')
    before = arm.raw_flange_pose_read_only()
    detections = []
    for _ in range(9):
        frame = camera.read()
        found = find_colors(frame, config['camera'], rs_module=camera.rs)
        detections.append(found)
    after = arm.raw_flange_pose_read_only()
    if np.linalg.norm(after[:3] - before[:3]) > 0.5:
        raise RuntimeError('Robot moved during image measurement')
    if np.max(np.abs((after[3:] - before[3:] + 180) % 360 - 180)) > 0.2:
        raise RuntimeError('Robot wrist rotated during measurement')
    objects = {}
    for color in ('red', 'blue'):
        points = [d[color]['point_optical_m'] for d in detections if d[color] is not None]
        if len(points) >= 6:
            objects[color] = np.median(np.asarray(points), axis=0).tolist()
    return {'flange_pose_mm_deg': ((before + after) / 2).tolist(), 'optical_points_m': objects}


def main():
    cfg = json.loads((ROOT / 'config/deployment.json').read_text())
    print('Read-only: this program never commands robot motion.', flush=True)
    print('Keep the boxes stationary, wrist orientation fixed, and plate clear of objects.', flush=True)
    arm = ROSArm(cfg)
    try:
        with D405(cfg['camera']) as camera:
            input('HOME: with the box visible, press Enter to save robot pose + image. ')
            home = snapshot(arm, camera, cfg)
            print('HOME', np.round(home['flange_pose_mm_deg'], 2).tolist(), home['optical_points_m'], flush=True)
            input('Jog robot FORWARD toward the box in free space, without contact. Stop and press Enter. ')
            forward = snapshot(arm, camera, cfg)
            print('FORWARD', np.round(forward['flange_pose_mm_deg'], 2).tolist(), forward['optical_points_m'], flush=True)
            input('Now jog the robot SIDEWAYS, without rotating the wrist. Stop and press Enter. ')
            sideways = snapshot(arm, camera, cfg)
            print('SIDEWAYS', np.round(sideways['flange_pose_mm_deg'], 2).tolist(), sideways['optical_points_m'], flush=True)
    finally:
        arm.close()
    poses = np.asarray([s['flange_pose_mm_deg'] for s in (home, forward, sideways)], float)
    if np.max(np.abs((poses[:, 3:] - poses[0, 3:] + 180) % 360 - 180)) > 0.5:
        raise RuntimeError('Wrist rotation changed; captures cannot be used together')
    df = (poses[1, :3] - poses[0, :3]) / 1000.0
    dl = (poses[2, :3] - poses[1, :3]) / 1000.0
    if min(np.linalg.norm(df), np.linalg.norm(dl)) < 0.008:
        raise RuntimeError('Need two distinct movements of at least 8 mm to estimate the axes')
    if max(abs(df[2]), abs(dl[2])) > 0.002:
        raise RuntimeError('Please keep both motions horizontal')
    df[2] = 0.0
    dl[2] = 0.0
    f = df / np.linalg.norm(df)
    side = dl - np.dot(dl, f) * f
    if np.linalg.norm(side) < 0.008:
        raise RuntimeError('The sideways movement must be different from forward')
    l = side / np.linalg.norm(side)
    out = {
        'home': home, 'forward': forward, 'sideways': sideways,
        'candidate_forward_axis_base': f.tolist(),
        'candidate_lateral_axis_base': l.tolist(),
        'camera_rotation_candidate': None,
        'rotation_residual_m': None,
        'camera_translation': 'not determined from these measurements',
        'approved_for_motion': False,
    }
    optic_changes, base_changes = [], []
    for color in ('red', 'blue'):
        p0 = home['optical_points_m'].get(color)
        if p0 is None:
            continue
        for sample in (forward, sideways):
            if color in sample['optical_points_m']:
                optic_changes.append(np.asarray(sample['optical_points_m'][color]) - np.asarray(p0))
                base_changes.append(-(np.asarray(sample['flange_pose_mm_deg'][:3]) - poses[0, :3]) / 1000)
    if len(optic_changes) >= 2 and np.linalg.matrix_rank(optic_changes, tol=0.001) >= 2:
        x = np.asarray(optic_changes)
        y = np.asarray(base_changes)
        u, _, vt = np.linalg.svd(x.T @ y)
        r = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
        residual = float(np.sqrt(np.mean(np.sum(((r @ x.T).T - y)**2, axis=1))))
        out['camera_rotation_candidate'] = r.tolist()
        out['rotation_residual_m'] = residual
    path = ROOT / 'cr10_motion_measurements.json'
    path.write_text(json.dumps(out, indent=2) + '\n')
    print('\nFORWARD AXIS (candidate):', np.round(f, 4).tolist())
    print('SIDEWAYS AXIS (candidate):', np.round(l, 4).tolist())
    print('Camera rotation estimation residual (m):', out['rotation_residual_m'])
    print('Measurements saved to:', path)
    print('No robot commands sent. Camera position and plate-contact offset still need verification.')


if __name__ == '__main__':
    main()
