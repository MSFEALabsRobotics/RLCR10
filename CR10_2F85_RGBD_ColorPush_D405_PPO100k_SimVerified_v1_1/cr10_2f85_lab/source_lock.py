from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

DOBOT_COMMIT = "0f21e9839888f188ac06b0a9fea622d13a29a74f"
DOBOT_BRANCH = "feature/v4-optimization"
DOBOT_URDF_REL = Path("dobot_rviz/urdf/cr10_robot.urdf")
DOBOT_MESH_REL = Path("dobot_rviz/meshes/cr10")
DOBOT_URDF_SHA256 = "4012ab3033761485025222332744f43ffdb58e61e9917aa714fd4f0ee8c25cda"

ROBOTIQ_COMMIT = "3ab3befccaa10468f80803ba687105c9d224d567"
ROBOTIQ_MACRO_REL = Path("grippers/robotiq_description/urdf/robotiq_2f_85_macro.urdf.xacro")
ROBOTIQ_CONTROL_REL = Path("grippers/robotiq_description/urdf/2f_85.ros2_control.xacro")

MENAGERIE_COMMIT = "da76818e269b82289eba39808e2fb91d679d6994"
MENAGERIE_MODEL_REL = Path("robotiq_2f85/2f85.xml")
MENAGERIE_ASSETS_REL = Path("robotiq_2f85/assets")

class LockError(RuntimeError):
    pass

def _run(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        raise LockError(f"git query failed for {repo}: {exc}") from exc

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _require_clean(repo: Path, paths: list[Path]) -> None:
    rels = [str(p) for p in paths]
    dirty = _run(repo, "status", "--porcelain", "--", *rels)
    if dirty:
        raise LockError(f"source-locked files have local modifications in {repo}:\n{dirty}")

def validate_sources(dobot: Path, robotiq: Path, menagerie: Path) -> dict:
    dobot = dobot.expanduser().resolve(); robotiq = robotiq.expanduser().resolve(); menagerie = menagerie.expanduser().resolve()
    for label, repo in [("Dobot", dobot), ("Robotiq", robotiq), ("Menagerie", menagerie)]:
        if not (repo / ".git").exists():
            raise LockError(f"{label} repository missing: {repo}")

    d_commit = _run(dobot, "rev-parse", "HEAD")
    if d_commit != DOBOT_COMMIT:
        raise LockError(f"Dobot commit mismatch: {d_commit} != {DOBOT_COMMIT}")
    d_branch = _run(dobot, "rev-parse", "--abbrev-ref", "HEAD")
    if d_branch != DOBOT_BRANCH:
        raise LockError(f"Dobot branch mismatch: {d_branch!r} != {DOBOT_BRANCH!r}")
    d_urdf = dobot / DOBOT_URDF_REL
    if not d_urdf.is_file() or not (dobot / DOBOT_MESH_REL).is_dir():
        raise LockError("Dobot CR10 URDF/mesh directory missing")
    if _sha(d_urdf) != DOBOT_URDF_SHA256:
        raise LockError("Dobot CR10 URDF hash mismatch")
    _require_clean(dobot, [DOBOT_URDF_REL, DOBOT_MESH_REL])

    r_commit = _run(robotiq, "rev-parse", "HEAD")
    if r_commit != ROBOTIQ_COMMIT:
        raise LockError(f"Robotiq commit mismatch: {r_commit} != {ROBOTIQ_COMMIT}")
    for rel in [ROBOTIQ_MACRO_REL, ROBOTIQ_CONTROL_REL]:
        if not (robotiq / rel).is_file():
            raise LockError(f"Robotiq required file missing: {rel}")
    _require_clean(robotiq, [ROBOTIQ_MACRO_REL, ROBOTIQ_CONTROL_REL])

    m_commit = _run(menagerie, "rev-parse", "HEAD")
    if m_commit != MENAGERIE_COMMIT:
        raise LockError(f"Menagerie commit mismatch: {m_commit} != {MENAGERIE_COMMIT}")
    if not (menagerie / MENAGERIE_MODEL_REL).is_file() or not (menagerie / MENAGERIE_ASSETS_REL).is_dir():
        raise LockError("Menagerie Robotiq 2F-85 model/assets missing")
    _require_clean(menagerie, [Path("robotiq_2f85")])

    return {
        "dobot": {"path": str(dobot), "branch": d_branch, "commit": d_commit, "urdf_sha256": _sha(d_urdf)},
        "robotiq": {"path": str(robotiq), "commit": r_commit, "macro_sha256": _sha(robotiq / ROBOTIQ_MACRO_REL), "ros2_control_sha256": _sha(robotiq / ROBOTIQ_CONTROL_REL)},
        "menagerie": {"path": str(menagerie), "commit": m_commit, "model_sha256": _sha(menagerie / MENAGERIE_MODEL_REL)},
    }

def import_locked_sources(root: Path, dobot: Path, robotiq: Path, menagerie: Path, manifest: dict) -> None:
    dst = root / "models/source"
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "dobot_cr10/meshes/cr10").mkdir(parents=True, exist_ok=True)
    (dst / "robotiq_ros").mkdir(parents=True, exist_ok=True)
    (dst / "menagerie_2f85").mkdir(parents=True, exist_ok=True)

    shutil.copy2(dobot / DOBOT_URDF_REL, dst / "dobot_cr10/cr10_robot.urdf")
    shutil.copytree(dobot / DOBOT_MESH_REL, dst / "dobot_cr10/meshes/cr10", dirs_exist_ok=True)
    shutil.copy2(robotiq / ROBOTIQ_MACRO_REL, dst / "robotiq_ros/robotiq_2f_85_macro.urdf.xacro")
    shutil.copy2(robotiq / ROBOTIQ_CONTROL_REL, dst / "robotiq_ros/2f_85.ros2_control.xacro")
    shutil.copy2(menagerie / MENAGERIE_MODEL_REL, dst / "menagerie_2f85/2f85.xml")
    shutil.copytree(menagerie / MENAGERIE_ASSETS_REL, dst / "menagerie_2f85/assets", dirs_exist_ok=True)

    files = []
    for p in sorted(x for x in dst.rglob("*") if x.is_file()):
        files.append({"path": str(p.relative_to(root)), "sha256": _sha(p)})
    out = {"locked_sources": manifest, "imported_files": files}
    (dst / "SOURCE_MANIFEST.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
