from pathlib import Path
import json, sys, zipfile
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arm_bridge.core import OBS_LOW,OBS_HIGH,SafetyFault,observation,proposed_command,checked_static_preflight
from checkpoint_audit import audit

def config():
    c=json.loads((ROOT/'config/deployment.json').read_text())
    c['task']['forward_axis_base']=[1,0,0]
    c['task']['lateral_axis_base']=[0,1,0]
    c['safety']['workspace_min_base_m']=[-.5,-.5,.7]
    c['safety']['workspace_max_base_m']=[.7,.5,1.2]
    return c

def test_saved_checkpoint_hashes_and_evaluation():
    a=audit()
    assert a['checkpoint_steps']==100352
    assert a['heldout_successes']==200

def test_new_workspace_and_observation_clip_match_plate():
    assert np.isclose(OBS_HIGH[0],0.19) and np.isclose(OBS_HIGH[1],0.125)
    assert np.isclose(OBS_LOW[1],-0.125)
    c=config()
    tcp=np.array([0,0,.9])
    det={'red':{'point_base_m':tcp+[.07,.10,0], 'depth_m':.15},'blue':{'point_base_m':tcp+[.09,-.10,0], 'depth_m':.16}}
    clipped,raw=observation(c,[.16,.11],det,tcp,'red')
    assert np.allclose(raw,[.16,.11,.07,.10,.15,1,.09,-.10,.16,1,1,0])
    assert np.allclose(clipped,raw)
    for key in ('forward_bounds_m','lateral_bounds_m'):
        assert c['task'][key]==([0,0.19] if key=='forward_bounds_m' else [-0.125,0.125])

def test_action_limit_and_block_unapproved_live():
    c=config()
    cmd,point=proposed_command(c,[1,1],[0,0],[0,0,.9])
    assert np.isclose(np.linalg.norm(point-[0,0,.9]),.002)
    assert np.allclose(cmd,np.array([.002,.002])/np.sqrt(2))
    with pytest.raises(SafetyFault):checked_static_preflight(c,live=True)

def test_no_execute_exposed_by_cli():
    import subprocess
    cp=subprocess.run([sys.executable,str(ROOT/'run.py'),'--stage','execute'],capture_output=True,text=True)
    assert cp.returncode!=0 and 'invalid choice' in cp.stderr

def test_reference_observation_schema_order():
    required=['tcp_command_forward_m','tcp_command_lateral_m','red_surface_minus_tcp_forward_m','red_surface_minus_tcp_lateral_m','red_surface_depth_m','red_visible','blue_surface_minus_tcp_forward_m','blue_surface_minus_tcp_lateral_m','blue_surface_depth_m','blue_visible','requested_red','requested_blue']
    from arm_bridge.core import NAMES
    assert list(NAMES)==required

def test_policy_paths_are_new_plate_artifacts():
    s=(ROOT/'arm_bridge/policy.py').read_text()
    assert 'plate_color_ppo.zip' in s and 'plate_color_ppo_vecnormalize.pkl' in s
    assert 'pack5_v02_color_ppo_100k_seed7' not in s
