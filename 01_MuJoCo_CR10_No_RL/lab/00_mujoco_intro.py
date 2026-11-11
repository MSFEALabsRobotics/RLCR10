"""First MuJoCo exercise: a deliberately simplified two-link arm, NOT the CR10."""
import math
import time
import mujoco
import mujoco.viewer

xml = """<mujoco model="two_link_intro">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual><global azimuth="140" elevation="-25"/></visual>
  <worldbody>
    <light pos="0 -2 4"/>
    <geom type="plane" size="2 2 0.1" rgba="0.92 0.92 0.88 1"/>
    <body name="base" pos="0 0 0.5">
      <geom type="cylinder" size="0.11 0.08" rgba=".2 .2 .2 1"/>
      <body name="link1" pos="0 0 0.08">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-100 100"/>
        <geom type="capsule" fromto="0 0 0 0 0 .55" size=".045" rgba="0.3 0.5 0.8 1" mass="1"/>
        <body name="link2" pos="0 0 .55">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-120 120"/>
          <geom type="capsule" fromto="0 0 0 0 0 .4" size=".038" rgba="0.9 0.45 0.2 1" mass="0.6"/>
          <site name="tcp" pos="0 0 .4" size=".03" rgba="0 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="shoulder" kp="120"/>
    <position joint="elbow" kp="120"/>
  </actuator>
</mujoco>"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
print('MODEL: MuJoCo MjModel;', model.nq, 'joint positions,', model.nu, 'actuators')
print('DATA: MuJoCo MjData; qpos, qvel, ctrl and site positions')
print('This is a simple conceptual robot, not a CR10 geometry or controller.')
with mujoco.viewer.launch_passive(model, data) as viewer:
    with viewer.lock():
        viewer.cam.distance = 2.7
        viewer.cam.lookat[:] = (0.0, 0.0, 0.6)
    start=time.monotonic()
    last_print=-1
    while viewer.is_running() and time.monotonic() - start < 24:
        elapsed=time.monotonic()-start
        data.ctrl[0] = 0.35 * math.sin(0.8*elapsed)
        data.ctrl[1] = 0.45 * math.sin(0.5*elapsed)
        mujoco.mj_step(model,data)
        viewer.sync()
        second=int(elapsed)
        if second>last_print and second%3==0:
            last_print=second
            print('t=',round(elapsed,1),'qpos=',data.qpos.round(3).tolist(), 'ctrl=',data.ctrl.round(3).tolist())
        time.sleep(float(model.opt.timestep))
