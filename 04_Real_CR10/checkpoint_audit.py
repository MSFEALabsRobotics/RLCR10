#!/usr/bin/env python3
"""No camera, ROS, MuJoCo, or SB3 required."""
import hashlib, json, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def audit():
    root = ROOT / 'artifacts'
    manifest = json.loads((root/'manifest.json').read_text())
    for filename, expected in manifest['files_sha256'].items():
        p = root / filename
        if not p.is_file():
            raise RuntimeError(f'missing artifact: {filename}')
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'checkpoint SHA256 mismatch: {filename}')
    with zipfile.ZipFile(root/'plate_color_ppo.zip') as z:
        if not {'policy.pth','data','_stable_baselines3_version'}.issubset(z.namelist()):
            raise RuntimeError('saved PPO archive is incomplete')
        model_data = json.loads(z.read('data'))
        if int(model_data['num_timesteps']) < 100000:
            raise RuntimeError('unexpected PPO training length')
    metadata = json.loads((root/'plate_color_ppo_metadata.json').read_text())
    evaluation = json.loads((root/'plate_policy_heldout_eval.json').read_text())
    if metadata['end_effector'] != 'plastic_plate_89mm_square_4p5mm_thick':
        raise RuntimeError('wrong end-effector metadata')
    if not (evaluation['evaluation_episodes']==200 and evaluation['evaluation_successes']==200):
        raise RuntimeError('held-out evaluation report does not match uploaded results')
    return {'status':'PASS','checkpoint_steps':int(model_data['num_timesteps']),
            'model':'plate_color_ppo.zip','heldout_successes':200,'heldout_episodes':200,
            'checkpoint_and_normalizer_hashes_verified':True,
            'hardware_motion_commanded':False}

if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
