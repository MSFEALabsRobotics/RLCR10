from __future__ import annotations

from pathlib import Path

from .build_model import build
from .config import load_config
from .source_lock import import_locked_sources, validate_sources


def prepare_locked_scene(
    root: Path,
    *,
    dobot: str | Path | None = None,
    robotiq: str | Path | None = None,
    menagerie: str | Path | None = None,
) -> Path:
    """Validate frozen sources, import only required files, and build MuJoCo."""

    root = Path(root).resolve()
    locked = load_config()["source_lock"]
    paths = {
        "dobot": Path(dobot or locked["dobot_repo"]).expanduser(),
        "robotiq": Path(robotiq or locked["robotiq_repo"]).expanduser(),
        "menagerie": Path(menagerie or locked["menagerie_repo"]).expanduser(),
    }
    manifest = validate_sources(**paths)
    import_locked_sources(root, **paths, manifest=manifest)
    return build()

