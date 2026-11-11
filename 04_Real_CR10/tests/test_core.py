from pathlib import Path
import json,sys,time
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arm_bridge.core import (SafetyFault, observation, proposed_command, camera_transform,
                             checked_static_preflight, check_wrong_cube_clearance)
from arm_bridge.perception import RGBDFrame,find_colors
from arm_bridge.ros_arm import parse_payload
from calibrate_from_pairs import rigid_fit

@pytest.fixture
def cfg():
    c=json.loads((ROOT/'config/deployment.json').read_text())
    c['task']['forward_axis_base']=[1.,0.,0.]
    c['task']['lateral_axis_base']=[0.,1.,0.]
    c['safety']['workspace_min_base_m']=[-.5,-.5,.6]
    c['safety']['workspace_max_base_m']=[.5,.5,1.2]
    c['calibration']['T_base_camera_at_baseline']=np.eye(4).tolist()
    c['calibration']['baseline_tcp_pose_mm_deg']=[0,0,900,0,0,0]
    c['calibration']['approved']=True
    c['safety']['box_lateral_width_m']=.05
    return c

def test_action_bound_and_radial(cfg):
    cmd,target=proposed_command(cfg,[1,1],[0,0],[0,0,.9])
    assert abs(np.linalg.norm(target-[0,0,.9])-0.002)<1e-10
    assert np.isclose(target[2],.9)
    with pytest.raises(SafetyFault): proposed_command(cfg,[2,0],[0,0],[0,0,.9])
    with pytest.raises(SafetyFault): proposed_command(cfg,[1,0],[.19,0],[0,0,.9])

def test_observation_layout(cfg):
    tcp=np.array([0.,0.,.9])
    det={'red':{'point_base_m':tcp+np.array([.05,-.04,0]),'depth_m':.12},
         'blue':{'point_base_m':tcp+np.array([.06,.04,0]),'depth_m':.16}}
    o,raw=observation(cfg,[0,0],det,tcp,'red')
    assert len(o)==12 and np.allclose(o,[0,0,.05,-.04,.12,1,.06,.04,.16,1,1,0])

def test_camera_transform_moving_tool(cfg):
    p=[20,10,900,0,0,0]
    T=camera_transform(cfg,p)
    assert np.allclose(T[:3,3],[.02,.01,0])
    with pytest.raises(SafetyFault):camera_transform(cfg,[0,0,900,0,2,0])

def test_fail_closed(cfg):
    cfg['safety']['real_motion_authorized']=True
    cfg['calibration']['holdout_rmse_m']=0.001
    with pytest.raises(SafetyFault,match='LIVE MOTION BLOCKED'):
        checked_static_preflight(cfg,live=True)
    cfg['calibration']['approved']=False
    with pytest.raises(SafetyFault):checked_static_preflight(cfg)

def test_wrong_cube_plate(cfg):
    det={'blue':{'point_base_m':np.array([.04,.06,.9])}}
    with pytest.raises(SafetyFault,match='WRONG_BOX_IN_PLATE_SWEEP'):
        check_wrong_cube_clearance(cfg,det,np.array([0,0,.9]),np.array([.01,0,.9]),'red')
    det['blue']['point_base_m'][1]=.20
    check_wrong_cube_clearance(cfg,det,np.array([0,0,.9]),np.array([.01,0,.9]),'red')

def test_sim_red_blue_detection(cfg):
    rgb=np.zeros((72,128,3),np.uint8)
    depth=np.ones((72,128),np.float32)*.15
    rgb[20:35,20:35]=[210,46,30]
    rgb[20:35,75:90]=[26,63,217]
    frame=RGBDFrame(rgb,depth,80,80,64,36,time.monotonic(),1)
    result=find_colors(frame,cfg['camera'],np.eye(4))
    assert result['red'] is not None and result['blue'] is not None
    assert np.isclose(result['red']['depth_m'],.15,atol=1e-6)

def test_parse_ros_response():
    assert np.allclose(parse_payload('0,{1,2,3,4,5,6},GetPose();',6),[1,2,3,4,5,6])
    assert parse_payload('0,{5},RobotMode();',1)[0]==5
    with pytest.raises(SafetyFault):parse_payload('-1,{},GetPose();',6)

def test_fit_rigid():
    src=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]],float)
    R=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    dst=(R@src.T).T+np.array([.5,-.2,1.0])
    T,err=rigid_fit(src,dst)
    assert err<1e-10
    assert np.allclose(T[:3,:3],R)
