"""Dobot ROS2 V4 adapter. Read-only unless execute mode passes external gates."""
from __future__ import annotations
import re
import time
import numpy as np
from .core import SafetyFault

# The V4 driver exposes explicit /dobot_bringup_ros2/srv/{GetPose,...} names.
_NUM = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def parse_payload(value, count):
    if not isinstance(value, str):
        raise SafetyFault('Dobot response has no readable payload')
    text = value.strip()
    # Standard reply: 0,{v1,...,vn},GetPose(...);
    code = re.match(r'^\s*(-?\d+)\s*,',text)
    if code and int(code.group(1)) != 0:
        raise SafetyFault(f'Dobot command returned error {code.group(1)}')
    groups = re.findall(r'\{([^{}]*)\}',text)
    for group in groups:
        tokens = group.split(',')
        if len(tokens) == count and all(re.fullmatch(_NUM, x.strip()) for x in tokens):
            a=np.asarray([float(x) for x in tokens],dtype=float)
            if np.all(np.isfinite(a)):
                return a
    if count == 1 and re.fullmatch(_NUM,text):
        return np.array([float(text)])
    raise SafetyFault(f'cannot parse Dobot {count}-value reply: {text[:120]}')


class ROSArm:
    def __init__(self, cfg):
        try:
            import rclpy
            from dobot_msgs_v4 import srv
        except ImportError as e:
            raise RuntimeError('Source ROS2 Humble and ~/cr10_2f85_ws/install/setup.bash before running') from e
        self.rclpy=rclpy; self.srv=srv; self.cfg=cfg
        rclpy.init(args=None)
        self.node=rclpy.create_node('cr10_d405_ppo_bridge')
        self.clients={}
        self.prefix=cfg['robot']['service_prefix'].rstrip('/')

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()

    def call(self, label, request_fields=None, timeout_s=2.0):
        service_type=getattr(self.srv,label)
        if label not in self.clients:
            name=f'{self.prefix}/{label}'
            c=self.node.create_client(service_type,name)
            if not c.wait_for_service(timeout_sec=timeout_s):
                raise SafetyFault(f'Dobot service unavailable: {name}')
            self.clients[label]=c
        req=service_type.Request()
        for k,v in (request_fields or {}).items():
            setattr(req,k,v)
        future=self.clients[label].call_async(req)
        self.rclpy.spin_until_future_complete(self.node,future,timeout_sec=timeout_s)
        if not future.done():
            raise SafetyFault(f'Dobot service {label} timed out; status is unknown')
        if future.exception() is not None:
            raise SafetyFault(f'Dobot service {label} failed: {future.exception()}')
        reply=future.result()
        if int(reply.res)!=0:
            raise SafetyFault(f'Dobot {label} returned res={reply.res} {getattr(reply,"robot_return","")}')
        return reply

    def raw_flange_pose_read_only(self):
        # READ ONLY: tool 0 is the vendor reference, NOT the pusher TCP.
        ret = self.call('GetPose', {'user': 0, 'tool': 0})
        return parse_payload(ret.robot_return, 6)

    def pose(self):
        r=self.cfg['robot']
        if r['tool_frame'] is None:
            raise SafetyFault('tool_frame (calibrated pusher TCP) missing')
        ret=self.call('GetPose',{'user':int(r['user_frame']),'tool':int(r['tool_frame'])})
        return parse_payload(ret.robot_return,6)

    def angles(self):
        return parse_payload(self.call('GetAngle').robot_return,6)

    def robot_mode(self):
        return int(parse_payload(self.call('RobotMode').robot_return,1)[0])

    def ensure_idle(self):
        mode=self.robot_mode()
        if mode!=int(self.cfg['robot']['controller_mode_idle']):
            raise SafetyFault(f'RobotMode {mode}; required enabled and idle (5)')

    def ik_preflight(self, target_pose_mm_deg):
        current=self.angles()
        user=self.cfg['robot']['user_frame']; tool=self.cfg['robot']['tool_frame']
        vals=np.asarray(target_pose_mm_deg,float)
        args=dict(zip(('x','y','z','rx','ry','rz'),[float(x) for x in vals]))
        args.update({'use_joint_near':'1','joint_near':'{'+','.join(f'{x:.5f}' for x in current)+'}',
                     'user':str(user),'tool':str(tool)})
        ik=parse_payload(self.call('InverseKin',args).robot_return,6)
        r=self.cfg['robot']; limits=np.asarray(r['joint_limits_deg'],dtype=float)
        margin=float(r['joint_margin_deg'])
        if np.any(ik < limits[:,0]+margin) or np.any(ik>limits[:,1]-margin):
            raise SafetyFault('INVERSE_KIN_JOINT_LIMIT_MARGIN')
        if np.max(np.abs(ik-current)) > float(r['max_joint_delta_deg']):
            raise SafetyFault('INVERSE_KIN_JOINT_STEP_TOO_LARGE')
        return ik

    def move_and_wait(self, target_pose_mm_deg):
        self.ensure_idle()
        self.ik_preflight(target_pose_mm_deg)
        vals=[float(x) for x in target_pose_mm_deg]
        r=self.cfg['robot']
        args=dict(zip(('a','b','c','d','e','f'),vals))
        args['mode']=False  # Verified driver parseTool.cpp: false => pose={x,y,z,rx,ry,rz}
        args['param_value']=[f'user={int(r["user_frame"])}',f'tool={int(r["tool_frame"])}',
                            f'a={int(r["acceleration_percent"])}',
                            f'speed={int(r["linear_speed_mm_s"])}','cp=0']
        ret=self.call('MovL',args,timeout_s=3.0)
        if hasattr(ret,'robot_return') and ret.robot_return:
            response=ret.robot_return
            code=re.match(r'^\s*(-?\d+)\s*,',response)
            if code and int(code.group(1))!=0:
                raise SafetyFault(f'MovL controller error: {response}')
        deadline=time.monotonic()+float(r['move_timeout_s'])
        stable=0
        while time.monotonic() < deadline:
            current=self.pose()
            mode=self.robot_mode()
            if mode in (9,10,11,4,6,2,1):
                raise SafetyFault(f'ROBOT_MODE_DURING_MOVEMENT={mode}')
            cart_error=np.linalg.norm((current[:3]-target_pose_mm_deg[:3])*.001)
            delta_ang=np.mod(current[3:]-target_pose_mm_deg[3:]+180,360)-180
            if cart_error < float(r['max_tcp_tracking_error_m']) and np.max(np.abs(delta_ang)) < 0.5 and mode==5:
                stable+=1
                if stable>=2:
                    return current
            else:
                stable=0
            time.sleep(0.08)
        raise SafetyFault('ROBOT_TRACKING_TIMEOUT_UNCERTAIN_QUEUED_MOTION')

    def stop_best_effort(self):
        try:
            self.call('Stop',timeout_s=1.0)
        except Exception:
            pass  # An independent physical safety system is mandatory.
