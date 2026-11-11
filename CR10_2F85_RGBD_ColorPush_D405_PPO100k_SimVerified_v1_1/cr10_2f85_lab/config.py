from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_config() -> dict:
    return json.loads((ROOT / "config/lab.json").read_text(encoding="utf-8"))
