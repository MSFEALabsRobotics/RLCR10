from pathlib import Path
import py_compile

src = Path('cr10_hybrid_full_hardware_v11_depth_filter.py')
out = Path('cr10_hybrid_v11_visible_fix.py')
s = src.read_text()

old = 'EDGE_PX = 8.0'
if old not in s:
    raise SystemExit('Cannot locate camera edge margin. No file written.')
s = s.replace(old, 'EDGE_PX = 4.0', 1)

old = "    unstable_frames = 0\n    for step in range(max_steps):"
new = """    unstable_frames = 0
    initial_points = []
    last_detection = {}
    for _ in range(30):
        current = pose(arm, home)
        _, _, initial_boxes = see(camera, cfg, r, current, home, color, offset, side_offset)
        last_detection = initial_boxes[color]
        if last_detection.get('visible'):
            initial_points.append(np.asarray(last_detection['box_base_mm'], dtype=float))
            initial_points = initial_points[-10:]
            center = np.median(np.asarray(initial_points), axis=0)
            stable = [p for p in initial_points if np.linalg.norm(p - center) <= 15.0]
            if len(stable) >= 4:
                print('TARGET LOCKED:', color, 'stable_frames:', len(stable), flush=True)
                break
        time.sleep(0.10)
    else:
        print('STOP: No stable RGB-D detection:', json.dumps(last_detection), flush=True)
        return
    for step in range(max_steps):"""
if old not in s:
    raise SystemExit('Cannot locate PPO loop start. No file written.')
s = s.replace(old, new, 1)

old = "print('STOP: No reliable previous position for target')"
new = "print('STOP: No reliable previous position for target:', json.dumps(boxes.get(color, {})), flush=True)"
if old not in s:
    raise SystemExit('Cannot locate vision failure diagnostic. No file written.')
s = s.replace(old, new, 1)
out.write_text(s)
py_compile.compile(str(out), doraise=True)
print('Created', out)
print('Original preserved; PPO and fixed movement sizes unchanged.')
