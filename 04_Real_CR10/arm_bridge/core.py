"""Exact plate PPO 12D layout and independent Cartesian gating."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

NAMES = (
    'tcp_command_forward_m', 'tcp_command_lateral_m',
    'red_surface_minus_tcp_forward_m', 'red_surface_minus_tcp_lateral_m',
    'red_surface_depth_m', 'red_visible',
    'blue_surface_minus_tcp_forward_m', 'blue_surface_minus_tcp_lateral_m',
    'blue_surface_depth_m', 'blue_visible', 'requested_red', 'requested_blue',
)
OBS_LOW = np.array([0, -.125, -.30, -.30, 0, 0, -.30, -.30, 0, 0, 0, 0], dtype=np.float32)
OBS_HIGH = np.array([.19, .125, .30, .30, .5, 1, .30, .30, .5, 1, 1, 1], dtype=np.float32)

class SafetyFault(RuntimeError):
    pass


def vec3(x, key):
    if x is None:
        raise SafetyFault(f'{key} is not calibrated')
    v = np.asarray(x, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise SafetyFault(f'{key} must be three finite numbers')
    return v


def axes(cfg):
    f = vec3(cfg['task']['forward_axis_base'], 'forward_axis_base')
    l = vec3(cfg['task']['lateral_axis_base'], 'lateral_axis_base')
    f = f / np.linalg.norm(f)
    l = l / np.linalg.norm(l)
    if abs(float(f @ l)) > 0.01 or abs(f[2]) > 0.03 or abs(l[2]) > 0.03:
        raise SafetyFault('task axes must be orthogonal and approximately horizontal')
    return f, l


def camera_transform(cfg, current_tcp_pose_mm_deg):
    cal = cfg['calibration']
    if not cal['approved']:
        raise SafetyFault('hand-eye calibration not approved')
    T = np.asarray(cal['T_base_camera_at_baseline'], dtype=float)
    base = np.asarray(cal['baseline_tcp_pose_mm_deg'], dtype=float)
    pose = np.asarray(current_tcp_pose_mm_deg, dtype=float)
    if T.shape != (4, 4) or base.shape != (6,) or pose.shape != (6,):
        raise SafetyFault('invalid camera transform or robot pose')
    if not np.all(np.isfinite(T)) or not np.all(np.isfinite(pose)):
        raise SafetyFault('nonfinite camera or robot pose')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8):
        raise SafetyFault('camera transform last row must be [0,0,0,1]')
    R = T[:3, :3]
    if np.linalg.norm(R.T @ R - np.eye(3)) > 1e-2 or np.linalg.det(R) < .99:
        raise SafetyFault('camera transform rotation must be orthonormal and right-handed')
    angle_error = np.mod(pose[3:] - base[3:] + 180.0, 360.0) - 180.0
    if np.max(np.abs(angle_error)) > float(cal['max_orientation_change_deg']):
        raise SafetyFault('physical wrist orientation changed; camera transform now invalid')
    delta_m = (pose[:3] - base[:3]) * .001
    if np.linalg.norm(delta_m) > float(cal['max_translation_deviation_from_baseline_m']):
        raise SafetyFault('robot moved outside approved camera calibration range')
    out = T.copy()
    out[:3, 3] += delta_m
    return out


def observation(cfg, command, detections_base, tcp_m, target):
    f, l = axes(cfg)
    if target not in ('red', 'blue'):
        raise SafetyFault('target must be red or blue')
    command = np.asarray(command, float)
    tcp_m = vec3(tcp_m, 'tcp position')
    if command.shape != (2,) or not np.all(np.isfinite(command)):
        raise SafetyFault('invalid commanded planar position')
    obs = [command[0], command[1]]
    for color in ('red', 'blue'):
        det = detections_base[color]
        if det is None:
            if cfg['task']['require_both_detections']:
                raise SafetyFault(f'{color} cube lost / invalid depth')
            obs.extend([0,0,0,0]); continue
        delta = det['point_base_m'] - tcp_m
        depth = float(det['depth_m'])
        obs.extend([float(f @ delta), float(l @ delta), depth, 1.0])
    obs.extend([float(target == 'red'), float(target == 'blue')])
    raw = np.asarray(obs, dtype=np.float32)
    if raw.shape != (12,) or not np.all(np.isfinite(raw)):
        raise SafetyFault('invalid 12-feature observation')
    # Same clipping as original simulation. Do not silently allow invalid positions:
    # additional geometrical gates are applied separately.
    return np.clip(raw, OBS_LOW, OBS_HIGH), raw


def proposed_command(cfg, action, command, origin_tcp_m):
    f, l = axes(cfg)
    action = np.asarray(action, dtype=float)
    command = np.asarray(command, dtype=float)
    if action.shape != (2,) or command.shape != (2,) or not np.all(np.isfinite(action)):
        raise SafetyFault('nonfinite or misshaped policy action')
    if np.any(action < -1 - 1e-6) or np.any(action > 1 + 1e-6):
        raise SafetyFault('PPO action exceeds [-1,1]')
    action = np.clip(action, -1.0, 1.0)
    delta = (action / max(1.0, float(np.linalg.norm(action)))) * cfg['task']['max_step_m']
    if np.linalg.norm(delta) > 0.002 + 1e-12:
        raise SafetyFault('2 mm hard cap violated')
    cmd = command + delta
    for i, key in enumerate(('forward_bounds_m','lateral_bounds_m')):
        low, high = cfg['task'][key]
        if not low <= cmd[i] <= high:
            raise SafetyFault(f'PLANAR_BOUND {key}: {cmd[i]:.4f} m')
    target = vec3(origin_tcp_m, 'origin_tcp_m') + cmd[0] * f + cmd[1] * l
    s = cfg['safety']
    lower, upper = vec3(s['workspace_min_base_m'],'workspace_min_base_m'), vec3(s['workspace_max_base_m'],'workspace_max_base_m')
    if np.any(target < lower) or np.any(target > upper):
        raise SafetyFault('ABSOLUTE_TCP_WORKSPACE_BOUND')
    if np.linalg.norm(target - origin_tcp_m) > float(s['max_total_tcp_travel_m']):
        raise SafetyFault('TOTAL_TCP_TRAVEL_BOUND')
    return cmd, target


def checked_static_preflight(cfg, live=False):
    axes(cfg)
    if not cfg['calibration']['approved']:
        raise SafetyFault('camera-to-base transform unapproved')
    _ = camera_transform(cfg, cfg['calibration']['baseline_tcp_pose_mm_deg'])
    if live:
        cal = cfg['calibration']
        err = cal.get('holdout_rmse_m')
        if err is None or not np.isfinite(float(err)) or not (0 <= float(err) <= 0.002):
            raise SafetyFault('independent hand-eye hold-out RMSE must be verified at <=2mm')
        s = cfg['safety']; r = cfg['robot']
        checks = ['full_robot_collision_model_validated',
                  'pusher_plate_box_clearance_validated',
                  'operator_has_verified_robot_protection', 'real_motion_authorized']
        missing = [key for key in checks if s.get(key) is not True]
        if missing:
            raise SafetyFault('LIVE MOTION BLOCKED: ' + ', '.join(missing))
        if r['tool_frame'] is None or r['joint_limits_deg'] is None:
            raise SafetyFault('real tool frame or physical joint limits are not configured')
        limits = np.asarray(r['joint_limits_deg'], float)
        if limits.shape != (6,2) or np.any(limits[:,0] >= limits[:,1]):
            raise SafetyFault('invalid six-joint limits')
        if not (0 < float(r['linear_speed_mm_s']) <= 10 and 0 < float(r['acceleration_percent']) <= 10):
            raise SafetyFault('live speed and acceleration gates violated')
        if float(cfg['task']['max_step_m']) > 0.002:
            raise SafetyFault('live per-step displacement exceeds 2mm')
        _ = vec3(s['workspace_min_base_m'], 'workspace_min_base_m')
        _ = vec3(s['workspace_max_base_m'], 'workspace_max_base_m')


def check_wrong_cube_clearance(cfg, detections, tcp_origin_m, target_tcp_m, selected):
    s = cfg['safety']
    box_w = s['box_lateral_width_m']
    if box_w is None or float(box_w) <= 0:
        raise SafetyFault('physical box width not measured')
    f,l = axes(cfg)
    other = 'blue' if selected == 'red' else 'red'
    point = detections[other]['point_base_m']
    origin = vec3(tcp_origin_m, 'origin_tcp_m')
    # Conservative planar footprint: visible surface point assumed at box center laterally.
    # Reject if the undesired box falls inside the plate's swept lateral strip.
    lateral_distance = abs(float(l @ (point - target_tcp_m)))
    clearance = .5 * (float(s['pusher_lateral_width_m'])+float(box_w)) + float(s['extra_object_clearance_m'])
    wrong_forward = float(f @ (point - origin))
    if wrong_forward >= -0.03 and wrong_forward <= float(cfg['task']['forward_bounds_m'][1]) + float(box_w):
        if lateral_distance < clearance:
            raise SafetyFault(f'WRONG_BOX_IN_PLATE_SWEEP: lateral separation={lateral_distance:.3f}m, required={clearance:.3f}m')
