from pathlib import Path
import mujoco
root = Path(__file__).resolve().parents[1]
for scene in ('cr10_2f85_known_cube.xml', 'cr10_2f85_color_target.xml'):
    file = root/'models/generated'/scene
    if not file.is_file():
        raise SystemExit(f'Missing {file}. Run setup.sh.')
    model = mujoco.MjModel.from_xml_path(str(file))
    print(f'[PASS] {scene}: {model.nq} qpos, {model.nu} actuators')
print('[PASS] MuJoCo arm ready. No RL checkpoint used.')
