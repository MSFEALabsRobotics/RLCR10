"""Real source-locked CR10+2F85 MuJoCo geometry, without RL or hardware calls."""
from pathlib import Path
import sys, math, time
import numpy as np
import mujoco
import mujoco.viewer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cr10_2f85_lab.config import load_config
scene=ROOT/'models/generated/cr10_2f85_known_cube.xml'
if not scene.is_file(): raise SystemExit('Model missing. Run: bash setup.sh')
model=mujoco.MjModel.from_xml_path(str(scene))
data=mujoco.MjData(model)
config=load_config()
home=np.deg2rad(np.asarray(config['model']['home_deg'],dtype=float))
for i,q in enumerate(home,1):
    j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'joint{i}')
    a=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,f'arm_servo_{i}')
    data.qpos[int(model.jnt_qposadr[j])]=float(q)
    data.ctrl[a]=float(q)
gripper=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,'gripper_actuator')
data.ctrl[gripper]=float(config['verification']['open_command'])
mujoco.mj_forward(model,data)
print('Robot: simulated CR10 + Robotiq 2F-85. No RL. No hardware connection.')
print('MuJoCo: nq=',model.nq,'nv=',model.nv,'nu=',model.nu)
print('First six controls: arm joint target angles; seventh: gripper command')
with mujoco.viewer.launch_passive(model,data) as viewer:
    with viewer.lock():
        viewer.cam.lookat[:] = (0.2, -0.1, 0.65)
        viewer.cam.distance=2.3
        viewer.cam.azimuth=135
        viewer.cam.elevation=-22
    started=time.monotonic()
    last_print=-1
    while viewer.is_running() and time.monotonic()-started<28:
        elapsed=time.monotonic()-started
        data.ctrl[0]=float(home[0]+0.08*math.sin(0.45*elapsed))
        data.ctrl[gripper]=float(config['verification']['close_command'] if elapsed>12 else config['verification']['open_command'])
        mujoco.mj_step(model,data)
        # reduce UI overhead; simulation physics still advances at 0.002 seconds
        if int(elapsed*40)!=int((elapsed-float(model.opt.timestep))*40):
            viewer.sync()
        sec=int(elapsed)
        if sec>last_print and sec%5==0:
            last_print=sec
            print(f'wall time={sec:2d}s | joint1 command={data.ctrl[0]:+.3f}rad | actual={data.qpos[0]:+.3f}rad')
        time.sleep(float(model.opt.timestep))
