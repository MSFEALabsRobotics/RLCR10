from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parents[1]


def build_color_task_scene(
    base_scene: str | Path,
    config: dict,
) -> Path:
    """Build the separate red/blue task scene."""

    source = Path(base_scene).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"base MuJoCo scene missing: {source}")

    tree = ET.parse(source)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("base MuJoCo worldbody is missing")

    red_body = worldbody.find("body[@name='known_push_cube']")
    if red_body is None:
        raise RuntimeError("verified base cube is missing")
    red_joint = red_body.find("freejoint[@name='known_push_cube_free']")
    red_geom = red_body.find("geom[@name='known_push_cube_geom']")
    if red_joint is None or red_geom is None:
        raise RuntimeError("verified base cube structure changed")

    color_cfg = config["pack5"]["color_task"]["cube_colors"]
    red_body.set("name", "color_red_cube")
    red_joint.set("name", "color_red_cube_free")
    red_geom.set("name", "color_red_cube_geom")
    red_geom.set(
        "rgba",
        " ".join(str(float(v)) for v in color_cfg["red"]["rgba"]),
    )

    blue_body = copy.deepcopy(red_body)
    blue_body.set("name", "color_blue_cube")
    blue_body.set("pos", "5.2 5.2 5.2")
    blue_joint = blue_body.find("freejoint")
    blue_geom = blue_body.find("geom")
    if blue_joint is None or blue_geom is None:
        raise RuntimeError("failed to clone the verified cube")
    blue_joint.set("name", "color_blue_cube_free")
    blue_geom.set("name", "color_blue_cube_geom")
    blue_geom.set(
        "rgba",
        " ".join(str(float(v)) for v in color_cfg["blue"]["rgba"]),
    )
    worldbody.insert(list(worldbody).index(red_body) + 1, blue_body)

    output = ROOT / "models/generated/cr10_2f85_color_target.xml"
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="unicode")

    model = mujoco.MjModel.from_xml_path(str(output))
    required = (
        (mujoco.mjtObj.mjOBJ_BODY, "color_red_cube"),
        (mujoco.mjtObj.mjOBJ_BODY, "color_blue_cube"),
        (mujoco.mjtObj.mjOBJ_JOINT, "color_red_cube_free"),
        (mujoco.mjtObj.mjOBJ_JOINT, "color_blue_cube_free"),
        (mujoco.mjtObj.mjOBJ_GEOM, "color_red_cube_geom"),
        (mujoco.mjtObj.mjOBJ_GEOM, "color_blue_cube_geom"),
    )
    for object_type, name in required:
        if mujoco.mj_name2id(model, object_type, name) < 0:
            raise RuntimeError(f"compiled color-task element missing: {name}")
    if model.nu != 7:
        raise RuntimeError(
            "color-task derivation changed the seven-actuator robot contract"
        )
    return output
