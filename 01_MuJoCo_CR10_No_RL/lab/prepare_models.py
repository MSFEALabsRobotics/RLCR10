from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cr10_2f85_lab.runtime import prepare_locked_scene
from cr10_2f85_lab.color_scene import build_color_task_scene
from cr10_2f85_lab.config import load_config

base = prepare_locked_scene(ROOT)
scene = build_color_task_scene(base, load_config())
print('[READY] Base CR10 scene:', base.relative_to(ROOT))
print('[READY] Color pushing scene:', scene.relative_to(ROOT))
