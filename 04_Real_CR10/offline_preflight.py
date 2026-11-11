#!/usr/bin/env python3
"""Offline verification of the trained plate PPO hardware bridge.

Default run touches neither RealSense, ROS 2 nor the physical CR10. It exercises
actual image processing from generated RGB-D, 12D construction, saved PPO
inference (unless skipped), independent Cartesian gates and mocked feedback.

Optional --sim-parity loads the USER'S already-patched MuJoCo project and checks
that camera frames produce the same 12D features through the hardware bridge.
"""
from __future__ import annotations
import argparse
import json
import sys
from types import SimpleNamespace
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
from checkpoint_audit import audit
from arm_bridge.core import (SafetyFault, axes, camera_transform, observation,
                             proposed_command, check_wrong_cube_clearance,
                             checked_static_preflight, OBS_LOW, OBS_HIGH, NAMES)
from arm_bridge.perception import RGBDFrame, D405, find_colors, resize_aligned_frame
from arm_bridge.ros_arm import parse_payload, ROSArm
from types import SimpleNamespace
from calibrate_from_pairs import rigid_fit


def demo_config():
    """Independent test-only coordinate system. Not persisted or approved for hardware."""
    c = json.loads((ROOT / 'config/deployment.json').read_text())
    c['task']['forward_axis_base'] = [1.0, 0.0, 0.0]
    c['task']['lateral_axis_base'] = [0.0, 1.0, 0.0]
    c['safety']['workspace_min_base_m'] = [-0.05, -0.25, 0.80]
    c['safety']['workspace_max_base_m'] = [0.25, 0.25, 1.00]
    c['safety']['box_lateral_width_m'] = 0.10  # synthetic box ONLY
    # Do not approve the *real* configuration. These are in-memory fake values.
    c['calibration']['approved'] = True
    c['calibration']['T_base_camera_at_baseline'] = np.eye(4).tolist()
    c['calibration']['baseline_tcp_pose_mm_deg'] = [0.,0.,900.,0.,0.,0.]
    return c


def mock_frame(red_u=64, blue_u=116, *, depth_m=.19, missing=None, bad_depth=False, old=False, number=1):
    rgb = np.full((72, 128, 3), 33, dtype=np.uint8)
    depth = np.full((72,128), np.float32(depth_m), dtype=np.float32)
    for color, u, val in [('red', red_u, (210, 46, 30)),('blue', blue_u, (26, 63, 217))]:
        if color != missing:
            lo,hi = max(0,u-5),min(128,u+6)
            rgb[31:42,lo:hi] = val
            if bad_depth and color == 'red':
                depth[31:42,lo:hi] = 0.
    return RGBDFrame(rgb, depth, 80., 80., 64., 36.,
                     time.monotonic()-(3. if old else 0.), number)


def fake_transform(tcp):
    """RealSense optical (right, down, forward) -> fake base (lateral, -up, forward)."""
    T = np.eye(4)
    T[:3,:3] = np.array([[0.,0.,1.], [1.,0.,0.], [0.,1.,0.]])
    # Right-handed fake optical-to-base map: Xbase=Zopt, Ybase=Xopt, Zbase=Yopt.
    T[:3,3] = np.asarray(tcp,float)
    return T


def projected_fake_frame(T,world_points,number):
    """Render simple pinhole RGB-D patches for FIXED objects and a MOVING camera."""
    rgb=np.full((72,128,3),33,dtype=np.uint8)
    depth=np.zeros((72,128),dtype=np.float32)
    colors={'red':(210,46,30),'blue':(26,63,217)}
    for name,world in world_points.items():
        optical=T[:3,:3].T @ (world-T[:3,3])
        if optical[2]<=.07:raise AssertionError('mock object behind/too close to camera')
        u=int(np.rint(64.+80.*optical[0]/optical[2]))
        v=int(np.rint(36.+80.*optical[1]/optical[2]))
        x0,x1=max(0,u-5),min(128,u+6)
        y0,y1=max(0,v-5),min(72,v+6)
        if (x1-x0)*(y1-y0)<12:raise AssertionError('mock object left camera FOV')
        rgb[y0:y1,x0:x1]=colors[name]
        depth[y0:y1,x0:x1]=optical[2]
    return RGBDFrame(rgb,depth,80.,80.,64.,36.,time.monotonic(),number)


def one_synthetic_loop(policy, report, steps):
    cfg = demo_config()
    tcp = np.array([0.,0.,.9])
    origin = tcp.copy()
    command=np.zeros(2)
    first_obs=None
    actions=[]
    # Fixed scene in synthetic base coordinates, unlike constant fake pixels.
    fixed_boxes={'red':np.array([.19,0.,.9]),
                 'blue':np.array([.19,.1235,.9])}
    for i in range(steps):
        T=fake_transform(tcp)
        frame=projected_fake_frame(T,fixed_boxes,i+1)
        det=find_colors(frame,cfg['camera'],T)
        if any(det[k] is None for k in ('red','blue')):
            raise AssertionError('synthetic color/depth detection lost')
        for name in ('red','blue'):
            if np.linalg.norm(det[name]['point_base_m']-fixed_boxes[name]) > .003:
                raise AssertionError('static box estimated to move >3 mm')
        obs, raw=observation(cfg,command,det,tcp,'red')
        if first_obs is None: first_obs=raw.tolist()
        act=policy.predict(obs) if policy is not None else np.array([.7,0.0])
        next_command, target=proposed_command(cfg,act,command,origin)
        check_wrong_cube_clearance(cfg,det,origin,target,'red')
        commanded_progress=float((target-tcp) @ np.array([1.,0.,0.]))
        if commanded_progress<=0:raise AssertionError('policy did not advance toward target')
        # Mock servo: feedback reaches independently approved target exactly.
        tcp=np.asarray(target).copy()
        command=next_command.copy()
        actions.append({'step':i,'action2':act.tolist(),
                        'target_m':target.tolist(),
                        'forward_progress_mm':1000*commanded_progress,
                        'mock_feedback_error_m':float(np.linalg.norm(tcp-target)),
                        'obs12':obs.tolist()})
    report['synthetic_loop'] = {'steps':steps,'first_obs12':first_obs,'steps_detail':actions,
                                'note':'Fixed synthetic boxes, moving pinhole camera, idealized mock servo; no real dynamics'}


def check_case(label, fn, report):
    try:
        details=fn()
        report['checks'].append({'name':label,'status':'PASS', 'details':details})
    except Exception as exc:
        report['checks'].append({'name':label,'status':'FAIL',
                                 'error':f'{type(exc).__name__}: {exc}'})


def expects_safety_fault(fn, text=None):
    try:
        fn()
    except SafetyFault as exc:
        if text is not None and text not in str(exc):
            raise AssertionError(f'wrong fault {exc}, expected {text}')
        return str(exc)
    raise AssertionError('dangerous scenario was not rejected')


def local_checks(report, with_policy, loop_steps):
    cfg=demo_config()
    check_case('User checkpoint SHA256 and evaluation evidence', audit, report)
    check_case('Default real-world calibration/motion remain blocked', lambda:
        expects_safety_fault(lambda:checked_static_preflight(
            json.loads((ROOT/'config/deployment.json').read_text()),live=True)),report)
    cfg['calibration']['holdout_rmse_m']=.001  # synthetic, in-memory ONLY
    check_case('Mock calibration cannot authorize physical motion', lambda:
        expects_safety_fault(lambda:checked_static_preflight(cfg,live=True),'LIVE MOTION BLOCKED'),report)
    check_case('Observation schema length and bounds', lambda:
        (None if len(NAMES)==12 and len(OBS_HIGH)==12 and len(OBS_LOW)==12 and
         OBS_HIGH[0]==.19 and OBS_HIGH[1]==.125 else
         (_ for _ in ()).throw(AssertionError('schema mismatch'))),report)

    def rgbd_checks():
        d=find_colors(mock_frame(),cfg['camera'],fake_transform([0.,0.,.9]))
        assert all(d[c] is not None for c in ('red','blue'))
        assert abs(d['red']['point_base_m'][0]-.19)<1e-5
        assert abs(d['red']['point_base_m'][1])<0.003
        assert d['blue']['point_base_m'][1] > .11
        assert d['red']['pixels']>=12 and d['blue']['pixels']>=12
        obs, raw=observation(cfg,[0.,0.],d,[0.,0.,.9],'red')
        assert obs.shape==(12,) and obs[5]==1 and obs[9]==1
        assert np.allclose(obs,raw) and (obs[10],obs[11])==(1.,0.)
        return {'red_mask_px':d['red']['pixels'],'blue_mask_px':d['blue']['pixels'],
                'obs12':obs.tolist()}
    check_case('Full generated RGB-D -> calibrated optical point -> 12D', rgbd_checks, report)

    def invalid_depth():
        d=find_colors(mock_frame(bad_depth=True),cfg['camera'],fake_transform([0,0,.9]))
        assert d['red'] is None
        return expects_safety_fault(lambda:observation(cfg,[0,0],d,[0,0,.9],'red'))
    check_case('Missing color depth is rejected',invalid_depth,report)
    check_case('Stale RGB-D frame is rejected',lambda:expects_safety_fault(
        lambda:find_colors(mock_frame(old=True),cfg['camera']),'STALE_CAMERA_FRAME'),report)

    def bad_intrinsics():
        f=mock_frame();f.fx=0.
        return expects_safety_fault(lambda:find_colors(f,cfg['camera']),'intrinsics')
    check_case('Invalid camera intrinsics rejected',bad_intrinsics,report)

    def action_checks():
        c,t=proposed_command(cfg,[1.,1.],[0.,0.],[0.,0.,.9])
        assert abs(np.linalg.norm(t-[0,0,.9])-0.002)<1e-9
        return {'full_scale_diagonal_step_mm':float(np.linalg.norm(t-[0,0,.9])*1000)}
    check_case('2 mm Euclidean action bound and task mapping',action_checks,report)
    check_case('Out-of-range PPO action rejected', lambda:expects_safety_fault(
        lambda:proposed_command(cfg,[4,0],[0,0],[0,0,.9])) ,report)
    check_case('Overrun task bounds rejected',lambda:expects_safety_fault(
        lambda:proposed_command(cfg,[1,0],[.19,0],[0,0,.9])),report)
    def missing_workspace():
        bad=demo_config()
        bad['safety']['workspace_min_base_m']=None
        return expects_safety_fault(lambda:proposed_command(bad,[0.5,0.0],[0.,0.],[0.,0.,.9]),'not calibrated')
    check_case('Missing physical workspace bounds fail closed', missing_workspace, report)

    def wrong_clearance():
        det={'blue':{'point_base_m':np.array([.12,.045,.9])}}
        return expects_safety_fault(lambda:check_wrong_cube_clearance(
            cfg,det,np.array([0.,0.,.9]),np.array([.002,0.,.9]),'red'), 'WRONG_BOX_IN_PLATE_SWEEP')
    check_case('Wrong-box too close for plate rejected',wrong_clearance,report)

    def calibration_check():
        base=np.array([[.0,.0,.1],[.1,0,.1],[0,.1,.1],[0,0,.2],[.1,.1,.2]])
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        dst=(rot@base.T).T+np.array([.4,-.3,.7])
        T,err=rigid_fit(base,dst)
        assert err<1e-10 and np.allclose(T[:3,:3],rot)
        return {'synthetic_rmse_m':err}
    check_case('3D rigid calibration fit synthetic accuracy',calibration_check,report)
    check_case('Dobot numeric response parser',lambda:(
        None if np.allclose(parse_payload('0,{1,2,3,4,5,6},GetPose();',6),[1,2,3,4,5,6])
        else (_ for _ in ()).throw(AssertionError('bad response'))),report)
    check_case('Dobot controller error rejected',lambda:expects_safety_fault(
        lambda:parse_payload('-1,{0},GetPose();',6)),report)

    def mock_inverse_kinematics():
        r=object.__new__(ROSArm)  # Does NOT call rclpy.init or connect to a controller
        r.cfg=demo_config()
        r.cfg['robot']['tool_frame']=1
        r.cfg['robot']['joint_limits_deg']=[[-180,180]]*6
        sent=[]
        r.angles=lambda:np.zeros(6)
        def fake_call(label,fields):
            sent.append((label,fields))
            return SimpleNamespace(res=0,robot_return='0,{0,0,0,0,0,0},InverseKin();')
        r.call=fake_call
        out=r.ik_preflight([200,50,900,0,0,0])
        assert np.allclose(out,0)
        assert sent[0][0]=='InverseKin'
        assert set(sent[0][1])=={'x','y','z','rx','ry','rz','use_joint_near','joint_near','user','tool'}
        assert isinstance(sent[0][1]['x'],float) and isinstance(sent[0][1]['tool'],str)
        return {'request_fields':sorted(sent[0][1]),'real_ROS_calls':0}
    check_case('Mock InverseKin request and reply matches Dobot service fields',
               mock_inverse_kinematics,report)

    def resize_check():
        orig=np.full((720,1280,3),33,dtype=np.uint8)
        orig[300:410,640:750,:]=[210,46,30]
        d=np.full((720,1280),.19,np.float32)
        out,small=resize_aligned_frame(orig,d,128,72)
        assert out.shape==(72,128,3) and small.shape==(72,128)
        assert np.allclose(small,.19)
        return {'native_resolution':[1280,720],'policy_resolution':[128,72]}
    check_case('Full-resolution D405 -> 128x72 preprocessing without cv2',resize_check,report)

    def mock_d405_sdk():
        """Mock RealSense SDK, no USB hardware and no driver imported or installed."""
        class Intrinsics:
            fx=900.; fy=900.; ppx=640.; ppy=360.
        class ColorProfile:
            def as_video_stream_profile(self):return self
            def get_intrinsics(self):return Intrinsics()
        class ColorFrame:
            profile=ColorProfile()
            def __init__(self,n):self.number=n
            def get_frame_number(self):return self.number
            def get_data(self):
                img=np.zeros((720,1280,3),dtype=np.uint8)
                img[200:330,240:400]=(210,46,30)
                img[200:330,810:970]=(26,63,217)
                return img
        class DepthFrame:
            def get_data(self):return np.full((720,1280),1200,dtype=np.uint16)
        class FrameSet:
            def __init__(self,n):self.n=n
            def get_depth_frame(self):return DepthFrame()
            def get_color_frame(self):return ColorFrame(self.n)
        class Device:
            def get_info(self,item):return 'Intel RealSense D405'
            def first_depth_sensor(self):return SimpleNamespace(get_depth_scale=lambda: .0001)
        class DeviceProfile:
            def get_device(self):return Device()
        class Pipeline:
            def __init__(self):self.count=0;self.stopped=False
            def start(self,config):return DeviceProfile()
            def wait_for_frames(self,timeout_ms):
                self.count+=1
                return FrameSet(self.count)
            def stop(self):self.stopped=True
        class Config:
            def __init__(self):self.streams=[]
            def enable_device(self,serial):self.serial=serial
            def enable_stream(self,*a):self.streams.append(a)
        class Aligner:
            def process(self,frames):return frames
        fake=SimpleNamespace(
            pipeline=Pipeline,config=Config,
            stream=SimpleNamespace(color='color',depth='depth'),
            format=SimpleNamespace(rgb8='rgb8',z16='z16'),
            camera_info=SimpleNamespace(name='name'),
            align=lambda *_:Aligner(),
        )
        old=sys.modules.get('pyrealsense2')
        sys.modules['pyrealsense2']=fake
        try:
            with D405(cfg['camera']) as cam:
                frame=cam.read()
                assert frame.rgb.shape==(720,1280,3)
                assert frame.depth_m.shape==(720,1280)
                assert abs(frame.depth_m[0,0]-.12)<1e-5
                detected=find_colors(frame,cfg['camera'])
                assert all(detected[k] is not None for k in ('red','blue'))
                cam.last_number=999
                expects_safety_fault(lambda:cam.read(),'REPEATED_OR_OUT_OF_ORDER_CAMERA_FRAME')
            assert cam.pipeline.stopped
            return {'fake_sensor':'D405','frame_rgb':[1280,720],
                    'color_detections':list(detected),'repeat_detection':True,
                    'hardware_connected':False}
        finally:
            if old is None:sys.modules.pop('pyrealsense2',None)
            else:sys.modules['pyrealsense2']=old
    check_case('Mock RealSense D405 acquisition, alignment and repeated-frame rejection',
               mock_d405_sdk,report)

    if with_policy:
        def policy_check():
            from arm_bridge.policy import Policy
            policy=Policy(ROOT)
            try:
                o=[0,0,.08,-.10,.17,1,.08,.10,.18,1,1,0]
                a=policy.predict(o)
                b=policy.predict([0,0,.08,.10,.17,1,.08,-.10,.18,1,1,0])
                assert a.shape==(2,) and np.all(np.isfinite(a)) and np.max(np.abs(a))<=1
                assert np.linalg.norm(a-b)>1e-4, 'policy fails input sensitivity smoke check'
                return {'red_left':a.tolist(),'red_right':b.tolist(),
                        'source':'actual trained PPO and saved normalization'}
            finally:
                policy.close()
        check_case('Actual saved PPO inference and input sensitivity',policy_check,report)

    def loop_check():
        p=None
        try:
            if with_policy:
                from arm_bridge.policy import Policy
                p=Policy(ROOT)
            one_synthetic_loop(p,report,loop_steps)
            return {'steps':loop_steps,'uses_real_checkpoint':with_policy,'ROS_calls':0}
        finally:
            if p is not None:p.close()
    check_case('Multi-step synthetic camera -> policy -> gate -> mock feedback',loop_check,report)


def sim_parity(report,sim_project,dobot,policy_enabled):
    """Cross-check the *actual simulated pixels* vs the live bridge code.

    This is optional because the sim source and MuJoCo are on the user's Ubuntu PC.
    It does not modify the simulation or send ROS2 commands.
    """
    sim_project=Path(sim_project).expanduser().resolve()
    if not (sim_project/'cr10_2f85_lab'/'plate_scene.py').exists():
        raise FileNotFoundError('Simulation source folder not found')
    sys.path.insert(0,str(sim_project))
    from cr10_2f85_lab.plate_scene import build_plate_scene
    from cr10_2f85_lab.rl.color_target_push_env import ColorTargetPushEnv, COLOR_OBSERVATION_COMPONENTS
    assert tuple(COLOR_OBSERVATION_COMPONENTS)==tuple(NAMES), '12D feature order differs'
    scene=build_plate_scene(dobot)
    env=ColorTargetPushEnv(scene)
    errors=[]
    try:
        for col,sign in (('red',-1),('red',1),('blue',-1),('blue',1)):
            obs,info=env.reset(seed=207,options={'target_color':col,'red_lateral_sign':sign})
            f=env.forward_axis
            l=env.lateral_axis
            cfg=demo_config()
            cfg['task']['forward_axis_base']=f.tolist()
            cfg['task']['lateral_axis_base']=l.tolist()
            Rgl=env.data.cam_xmat[env.camera_id].reshape(3,3)
            T=np.eye(4)
            T[:3,:3]=Rgl @ np.diag([1.,-1.,-1.])
            T[:3,3]=env.data.cam_xpos[env.camera_id]
            assert np.linalg.det(T[:3,:3])>.99
            intr=env.intrinsics
            # Exactly the renderer's RGB-D arrays from the reset, no second render/noise draw.
            frame=RGBDFrame(env.last_rgb.copy(),env.last_depth_m.copy(),
                            float(intr.fx),float(intr.fy),float(intr.cx),float(intr.cy),
                            time.monotonic(),1)
            det=find_colors(frame,cfg['camera'],T)
            command=np.zeros(2) if env.interface is None else env.interface.planar_m
            tcp=env.data.site_xpos[env.tcp_sid].copy()
            obs2,raw=observation(cfg,command,det,tcp,col)
            deviation=np.abs(obs2-obs)
            errors.append({'target':col,'red_lateral_sign':sign,
                           'max_obs12_abs_diff':float(np.max(deviation)),
                           'mean_obs12_abs_diff':float(np.mean(deviation)),
                           'sim_obs12':obs.tolist(),'bridge_obs12':obs2.tolist()})
            if not np.allclose(obs2,obs,rtol=0,atol=0.003):
                raise AssertionError('SIM/BRIDGE OBSERVATION PARITY FAIL: '+str(errors[-1]))
        return {'scenarios':errors,'max_allowed_feature_difference':0.003,
                'note':'Real MuJoCo RGB-D rendered frames; no physical camera'}
    finally:
        env.close()


def check_ros_message_contract():
    """Inspect LOCAL installed ROS2 .srv types only, no ROS node or controller connection."""
    from dobot_msgs_v4 import srv
    required={
      'GetPose':{'user':'int32','tool':'int32'},
      'GetAngle':{},'RobotMode':{},
      'InverseKin':{'x':'float64','y':'float64','z':'float64',
                    'rx':'float64','ry':'float64','rz':'float64',
                    'use_joint_near':'string','joint_near':'string','user':'string','tool':'string'},
      'MovL':{'mode':'boolean','a':'float64','b':'float64','c':'float64',
              'd':'float64','e':'float64','f':'float64','param_value':'sequence<string>'},
    }
    fields={}
    for label,expected in required.items():
        typ=getattr(srv,label).Request
        actual=typ.get_fields_and_field_types()
        # Normalize equivalent ROS 2 type names.
        def canonical(t):
            return {
                'float64': 'double',
                'bool': 'boolean',
                'sequence<string, 0>': 'sequence<string>'
            }.get(t, t)

        for key,value in expected.items():
            if key not in actual or canonical(actual[key]) != canonical(value):
                raise AssertionError(f'{label}.{key}: expected {value}, found {actual.get(key)}')
        if set(actual)!=set(expected):
            raise AssertionError(f'{label} fields mismatch: {actual}')
        fields[label]=actual
    return {'local_generated_ros2_service_contracts':fields,
            'ros_node_created':False,'robot_network_contact':False}


def main():
    ap=argparse.ArgumentParser(description='Zero-hardware offline preflight; no real motion mode')
    ap.add_argument('--ros-contract',action='store_true',help='Check installed ROS2 service schemas; no robot connection')
    ap.add_argument('--skip-policy',action='store_true',help='For environments without Stable-Baselines3; explicitly skips network inference')
    ap.add_argument('--loop-steps',type=int,default=4)
    ap.add_argument('--sim-parity',action='store_true',help='Use existing MuJoCo project to compare rendered RGB-D -> 12D with training env')
    ap.add_argument('--sim-project',default='~/Downloads/CR10_D405_PLATE_ColorPush_SIM_v0_3')
    ap.add_argument('--dobot',default='~/cr10_2f85_ws/src/DOBOT_6Axis_ROS2_V4')
    ap.add_argument('--report',default='verification/offline_preflight.json')
    args=ap.parse_args()
    if not 1<=args.loop_steps<=20:ap.error('loop-steps must be 1..20')
    report={'package':'CR10_D405_PLATE_PPO_OFFLINE_READY_v0_3',
            'physical_camera_connected':False,'physical_robot_connected':False,
            'physical_motion_possible_via_this_script':False,
            'policy_inference_requested':not args.skip_policy,
            'checks':[]}
    local_checks(report,not args.skip_policy,args.loop_steps)
    if args.ros_contract:
        check_case('Installed ROS2 Dobot V4 message schema (no connection)',
                   check_ros_message_contract,report)
    if args.sim_parity:
        check_case('Actual MuJoCo RGB-D vs hardware bridge 12D parity',
                   lambda:sim_parity(report,args.sim_project,args.dobot,not args.skip_policy),report)
    report['passed']=sum(x['status']=='PASS' for x in report['checks'])
    report['failed']=sum(x['status']=='FAIL' for x in report['checks'])
    report['status']='PASS' if report['failed']==0 else 'FAIL'
    target=(ROOT/args.report).resolve() if not Path(args.report).is_absolute() else Path(args.report)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
    for x in report['checks']:
        print(f"[{x['status']}] {x['name']}")
        if x['status']=='FAIL': print(f"    {x['error']}")
    print(f"OFFLINE PREFLIGHT: {report['status']} ({report['passed']} passed, {report['failed']} failed)")
    print('Report:',target)
    print('NO CAMERA / NO ROS 2 / NO PHYSICAL MOTION')
    return 0 if report['status']=='PASS' else 2

if __name__=='__main__':
    raise SystemExit(main())
