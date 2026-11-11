#!/usr/bin/env python3
"""Staged hardware integration. No movement in camera, robot-status or dry-run stages."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

from arm_bridge.core import (SafetyFault, axes, camera_transform, checked_static_preflight,
                             observation, proposed_command, check_wrong_cube_clearance)

ROOT=Path(__file__).resolve().parent

def logline(obj):
    def cv(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,(np.generic,)):return x.item()
        raise TypeError(type(x).__name__)
    print(json.dumps(obj,default=cv,sort_keys=True),flush=True)


def camera_only(cfg,steps,point):
    from arm_bridge.perception import D405, find_colors
    with D405(cfg['camera']) as camera:
        for i in range(steps):
            frame=camera.read()
            det=find_colors(frame,cfg['camera'],rs_module=camera.rs)
            logline({'step':i,'mode':'CAMERA_ONLY_NO_ROBOT_COMMANDS',
                     'resolution':list(frame.depth_m.shape[::-1]),
                     'frame':frame.frame_number,
                     'detections':{k:None if v is None else {'uv':v['uv_policy_px'],
                                  'depth_m':v['depth_m'],'point_optical_m':v['point_optical_m'],
                                  'mask_pixels':v['pixels']} for k,v in det.items()}})
            if point is not None:
                u,v=point
                if not (0<=u<frame.rgb.shape[1] and 0<=v<frame.rgb.shape[0]):
                    raise SafetyFault('pixel is outside RGB frame')
                d=float(frame.depth_m[v,u])
                if not (cfg['camera']['min_valid_depth_m']<=d<=cfg['camera']['max_valid_depth_m']):
                    raise SafetyFault('invalid depth at chosen pixel')
                xyz=camera.rs.rs2_deproject_pixel_to_point(frame.intrinsics,[float(u),float(v)],d)
                logline({'pixel':point,'point_camera_optical_m':xyz,'depth_m':d})
                break


def robot_status(cfg):
    from arm_bridge.ros_arm import ROSArm
    arm=ROSArm(cfg)
    try:
        logline({'mode':'ROBOT_STATUS_READ_ONLY', 'raw_tool0_pose_mm_deg_NOT_PLATE_TCP':arm.raw_flange_pose_read_only(),
                 'joints_deg':arm.angles(),'robot_mode':arm.robot_mode()})
    finally:
        arm.close()


def run_policy(cfg, stage, steps, selected):
    from arm_bridge.perception import D405, find_colors
    from arm_bridge.ros_arm import ROSArm
    from arm_bridge.policy import Policy
    execute = False  # Physical motion deliberately unavailable in the v0.2 preview CLI.
    if stage != 'dry-run':
        raise SafetyFault('This package has no physical-motion mode')
    checked_static_preflight(cfg,live=False)
    arm=ROSArm(cfg)
    ppo=None
    try:
        # Pose is read from the calibrated pusher TCP, not the obsolete 2F-85 model.
        initial=arm.pose()
        arm.ensure_idle()
        initial_m=initial[:3]*.001
        f,l=axes(cfg)
        initial_cmd=np.zeros(2,float)
        ppo=Policy(ROOT)
        command=initial_cmd.copy()
        initial_box=None
        previous_detection={}
        with D405(cfg['camera']) as camera:
            for step in range(steps):
                pose=arm.pose()
                arm.ensure_idle()
                tcp_m=pose[:3]*.001
                T=camera_transform(cfg,pose)
                frame=camera.read()
                det=find_colors(frame,cfg['camera'],T,rs_module=camera.rs)
                if any(det.get(c) is None for c in ('red','blue')):
                    raise SafetyFault('one or both boxes not detected with valid RGB-D')
                for color in ('red','blue'):
                    xyz=det[color]['point_base_m']
                    if not np.all(np.isfinite(xyz)):
                        raise SafetyFault(f'{color}: nonfinite 3D point')
                    if color in previous_detection:
                        if np.linalg.norm(xyz-previous_detection[color]) > cfg['task']['max_detection_jump_m']:
                            raise SafetyFault(f'{color} box perceived jump is too large')
                    previous_detection[color]=xyz.copy()
                if initial_box is None:
                    initial_box={k:v['point_base_m'].copy() for k,v in det.items()}
                progress=float(f @ (det[selected]['point_base_m']-initial_box[selected]))
                other='blue' if selected=='red' else 'red'
                wrong=float(np.linalg.norm([f@(det[other]['point_base_m']-initial_box[other]),
                                            l@(det[other]['point_base_m']-initial_box[other])]))
                if wrong > cfg['task']['max_wrong_displacement_m']:
                    raise SafetyFault(f'WRONG_BOX_MOVED_{wrong*1000:.1f}_mm')
                if progress>=cfg['task']['goal_displacement_m']:
                    logline({'step':step,'event':'TARGET_GOAL_OBSERVED','progress_m':progress,'wrong_motion_m':wrong})
                    break
                obs,raw=observation(cfg,command,det,tcp_m,selected)
                action=ppo.predict(obs)
                attempt={'step':step,'target_color':selected,'obs12':obs,'raw_obs12':raw,
                         'action2':action,'target_progress_m':progress,'wrong_motion_m':wrong,
                         'camera_frame':frame.frame_number}
                try:
                    next_cmd,target_tcp=proposed_command(cfg,action,command,initial_m)
                    check_wrong_cube_clearance(cfg,det,initial_m,target_tcp,selected)
                    attempt.update({'new_planar_m':next_cmd,'target_tcp_base_m':target_tcp,'gates':'PASSED'})
                except SafetyFault as gate:
                    attempt['gates']=f'REJECTED {gate}'
                    logline(attempt)
                    continue
                attempt['mode']='DRY_RUN_NO_MOVEMENT'
                logline(attempt)
            else:
                logline({'event':'MAX_STEPS','steps':steps,'movement_enabled':execute})
    except (SafetyFault,KeyboardInterrupt):
        raise
    finally:
        if ppo is not None:ppo.close()
        arm.close()


def policy_check(obs_file=None):
    """CPU inference with the user's checkpoint. Does not access the camera or robot."""
    from arm_bridge.policy import Policy
    from arm_bridge.core import OBS_LOW, OBS_HIGH
    if obs_file is None:
        # Explicitly SYNTHETIC, useful only for checking weights and normalization.
        scenarios = {
            'red_left': [0,0,.08,-.10,.17,1,.08,.10,.18,1,1,0],
            'red_right': [0,0,.08,.10,.17,1,.08,-.10,.18,1,1,0],
            'blue_left': [0,0,.08,-.10,.17,1,.08,.10,.18,1,0,1],
            'blue_right': [0,0,.08,.10,.17,1,.08,-.10,.18,1,0,1],
        }
    else:
        payload=json.loads(Path(obs_file).read_text())
        scenarios={'user_supplied_raw_obs':payload['obs12'] if isinstance(payload,dict) else payload}
    ppo=Policy(ROOT)
    try:
        for label,raw in scenarios.items():
            o=np.asarray(raw,dtype=np.float32)
            if o.shape!=(12,) or not np.all(np.isfinite(o)) or np.any(o<OBS_LOW) or np.any(o>OBS_HIGH):
                raise SafetyFault(f'{label}: invalid raw 12D observation')
            act=ppo.predict(o)
            logline({'stage':'OFFLINE_POLICY_ONLY_NO_ROBOT','scenario':label,
                     'raw_obs12':o,'action2':act,
                     'displacement_mm_if_2mm_limit':act / max(1.0,np.linalg.norm(act))*2.0})
    finally:
        ppo.close()


def main():
    a=argparse.ArgumentParser(description='D405 -> 12D PPO -> guarded Dobot ROS2')
    a.add_argument('--stage',choices=('checkpoint','policy-check','camera','robot-status','dry-run'),default='checkpoint')
    a.add_argument('--target',choices=('red','blue'),default='red')
    a.add_argument('--config',default=str(ROOT/'config/deployment.json'))
    a.add_argument('--steps',type=int,default=10)
    a.add_argument('--pixel',type=int,nargs=2,metavar=('U','V'))
    a.add_argument('--obs-file',help='12D raw observation JSON file for offline policy-check')
    args=a.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if not 1<=args.steps<=80:
        a.error('steps must be between 1 and 80')
    try:
        if args.stage=='checkpoint':
            from checkpoint_audit import audit
            logline(audit())
        elif args.stage=='policy-check':policy_check(args.obs_file)
        elif args.stage=='camera':camera_only(cfg,args.steps,args.pixel)
        elif args.stage=='robot-status':robot_status(cfg)
        else:run_policy(cfg,args.stage,min(args.steps,cfg['task']['max_steps']),args.target)
        return 0
    except (SafetyFault,KeyboardInterrupt,RuntimeError) as e:
        print(f'[STOP/REJECT] {e}',file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
