import json
import math
import time
import numpy as np
from arm_bridge.ros_arm import ROSArm

START = np.array([191.737, 661.957, -27.159, 91.33, 3.48, -177.69], dtype=float)
RECORDED_STOP = np.array([82.362, 783.5669, -27.1588], dtype=float)
PPO_MOVES = np.array([
    [-3.789, 2.171],
    [-3.640, 2.368],
    [-3.659, 2.418],
    [-3.657, 2.492],
    [-3.542, 2.677],
    [-3.473, 2.814],
    [-3.449, 2.918],
    [-3.394, 3.042],
    [-3.245, 3.287],
], dtype=float)
ALIGN_MM = 77.5
MAX_STEP_MM = 20

def angle_delta(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0

def read_pose(arm):
    value = np.asarray(arm.raw_flange_pose_read_only(), dtype=float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise RuntimeError('Invalid CR10 pose feedback')
    return value

def alarms_clear(arm):
    data = arm.call('GetErrorID').robot_return.strip()
    if not (data.startswith('{') and data.endswith('}')):
        raise RuntimeError('Cannot read CR10 alarms')
    value = json.loads(data[1:-1])
    if not isinstance(value, list) or len(value) != 7 or any(value):
        raise RuntimeError('CR10 alarm: ' + data)

def verify_idle(arm):
    mode = arm.robot_mode()
    if mode != 5:
        raise RuntimeError(f'CR10 mode {mode}, expected idle mode 5')
    alarms_clear(arm)

def move_to(arm, goal, name):
    goal = np.asarray(goal, dtype=float)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise RuntimeError('Invalid target position')
    initial = read_pose(arm)[:3]
    count = max(1, math.ceil(np.linalg.norm(goal - initial) / MAX_STEP_MM))
    for i in range(1, count + 1):
        target = initial + (goal - initial) * (i / count)
        verify_idle(arm)
        before = read_pose(arm)
        if np.max(np.abs(angle_delta(before[3:], START[3:]))) > 0.6:
            raise RuntimeError('Wrist orientation changed')
        d = target - before[:3]
        if np.linalg.norm(d) > MAX_STEP_MM + 0.5:
            raise RuntimeError('Movement exceeded 8 mm step allowance')
        args = dict(zip(('a', 'b', 'c', 'd', 'e', 'f'), [float(x) for x in d] + [0.0] * 3))
        args['param_value'] = []
        arm.call('RelMovLUser', args, timeout_s=5.0)
        deadline = time.monotonic() + 20.0
        consecutive = 0
        while time.monotonic() < deadline:
            mode = arm.robot_mode()
            if mode not in (5, 7):
                raise RuntimeError(f'Robot mode changed to {mode}')
            actual = read_pose(arm)
            if np.max(np.abs(angle_delta(actual[3:], START[3:]))) > 0.6:
                raise RuntimeError('Unexpected wrist rotation')
            if np.linalg.norm(actual[:3] - target) < 0.5 and mode == 5:
                consecutive += 1
                if consecutive >= 2:
                    break
            else:
                consecutive = 0
            time.sleep(0.08)
        else:
            raise RuntimeError(f'Tracking timeout during {name}')
    print(f'FINISHED {name}: {np.round(read_pose(arm)[:3], 3).tolist()}', flush=True)

def main():
    cfg = json.load(open('config/deployment.json'))
    arm = ROSArm(cfg)
    try:
        verify_idle(arm)
        actual = read_pose(arm)
        ppo_end = START[:3].copy()
        ppo_end[:2] += PPO_MOVES.sum(axis=0)
        aligned = ppo_end.copy()
        aligned[0] -= ALIGN_MM
        if np.linalg.norm(actual[:3] - RECORDED_STOP) > 3.0:
            raise RuntimeError('Robot has moved since the reported STOP pose; this recorded reverse path is invalid')
        if abs(actual[0] - aligned[0]) > 3.0:
            raise RuntimeError('Current robot X does not match the recorded aligned path')
        if actual[1] < ppo_end[1] - 2.0 or actual[1] > ppo_end[1] + 256.5:
            raise RuntimeError('Current robot Y is outside recorded contact approach')
        if abs(actual[2] - START[2]) > 1.0:
            raise RuntimeError('Z changed since the recorded path')
        if np.max(np.abs(angle_delta(actual[3:], START[3:]))) > 0.6:
            raise RuntimeError('Wrist changed since the recorded path')
        print('CURRENT:', np.round(actual, 3).tolist(), flush=True)
        print('1. BACKWARD -Y to:', [round(actual[0], 3), round(ppo_end[1], 3), round(actual[2], 3)], flush=True)
        print('2. UNALIGN +X to:', np.round(ppo_end, 3).tolist(), flush=True)
        print('3. REVERSE 9 PPO moves to START:', START[:3].tolist(), flush=True)
        print('Only proceed if the plate is free and this previously traveled path is clear.', flush=True)
        if input('Type RETURN to move: ').strip() != 'RETURN':
            print('CANCELLED')
            return
        move_to(arm, np.array([actual[0], ppo_end[1], actual[2]]), 'BACKWARD')
        move_to(arm, ppo_end, 'UNALIGN')
        cursor = ppo_end.copy()
        for i, d in enumerate(PPO_MOVES[::-1], 1):
            cursor[:2] -= d
            move_to(arm, cursor.copy(), f'REVERSE PPO {i}/9')
        move_to(arm, START[:3], 'FINAL START')
        final = read_pose(arm)
        if np.linalg.norm(final[:3] - START[:3]) > 1.0:
            raise RuntimeError('Return completed but final position differs from START')
        print('RETURNED TO ORIGINAL SESSION START:', np.round(final, 3).tolist(), flush=True)
    except BaseException:
        arm.stop_best_effort()
        raise
    finally:
        arm.close()

if __name__ == '__main__':
    main()
