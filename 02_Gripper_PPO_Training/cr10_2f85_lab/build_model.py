from __future__ import annotations
import copy
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco

from .config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _ensure(parent: ET.Element, tag: str) -> ET.Element:
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    return node


def _rpy_to_quat(rpy: list[float]) -> str:
    r, p, y = map(float, rpy)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yy = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return f"{w:.16g} {x:.16g} {yy:.16g} {z:.16g}"


def _camera_xyaxes(
    position: list[float], target: list[float]
) -> tuple[str, list[float]]:
    """Return camera x/y axes and backward (+Z) axis."""

    forward = [float(t) - float(p) for p, t in zip(position, target)]
    norm = math.sqrt(sum(value * value for value in forward))
    if norm <= 0.0:
        raise ValueError("Pack 5 camera target must differ from its position")
    forward = [value / norm for value in forward]

    reference_x = [1.0, 0.0, 0.0]
    projection = sum(a * b for a, b in zip(reference_x, forward))
    x_axis = [
        reference_x[i] - projection * forward[i] for i in range(3)
    ]
    x_norm = math.sqrt(sum(value * value for value in x_axis))
    if x_norm <= 1e-12:
        raise ValueError("Pack 5 camera direction is parallel to mount +X")
    x_axis = [value / x_norm for value in x_axis]
    z_axis = [-value for value in forward]
    y_axis = [
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    ]
    xyaxes = " ".join(str(value) for value in x_axis + y_axis)
    return xyaxes, z_axis


def _remove_if(parent: ET.Element, child: ET.Element | None) -> None:
    if child is not None:
        parent.remove(child)


def _check_unique_named_elements(root: ET.Element) -> None:
    seen: dict[tuple[str, str], int] = {}
    for e in root.iter():
        name = e.get("name")
        if not name:
            continue
        key = (e.tag, name)
        seen[key] = seen.get(key, 0) + 1
    dup = [f"{tag}:{name}" for (tag, name), n in seen.items() if n > 1 and tag not in {"default"}]
    if dup:
        raise RuntimeError("duplicate named MuJoCo elements after merge: " + ", ".join(sorted(dup)))


def build() -> Path:
    cfg = load_config()
    src = ROOT / "models/source"
    gen = ROOT / "models/generated"
    if gen.exists():
        shutil.rmtree(gen)
    gen.mkdir(parents=True, exist_ok=True)

    shutil.copytree(src / "dobot_cr10/meshes", gen / "meshes", dirs_exist_ok=True)
    text = (src / "dobot_cr10/cr10_robot.urdf").read_text(encoding="utf-8")
    for prefix in [
        "package://dobot_rviz/meshes/",
        "package://cra_description/meshes/",
        "file://$(find cra_description)/meshes/",
        "$(find cra_description)/meshes/",
    ]:
        text = text.replace(prefix, "meshes/")
    local_urdf = gen / "cr10_local.urdf"
    local_urdf.write_text(text, encoding="utf-8")

    base_model = mujoco.MjModel.from_xml_path(str(local_urdf))
    base_xml = gen / "cr10_base.xml"
    mujoco.mj_saveLastXML(str(base_xml), base_model)
    tree = ET.parse(base_xml)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("worldbody missing")
    arm_visual_geoms = list(worldbody.findall("geom"))
    for body_name in [f"Link{i}" for i in range(1, 7)]:
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise RuntimeError(f"official CR10 {body_name} not found after MuJoCo URDF import")
        arm_visual_geoms.extend(body.findall("geom"))
    if len(arm_visual_geoms) != 7:
        raise RuntimeError(
            f"expected exactly 7 imported CR10 visual mesh geoms, got {len(arm_visual_geoms)}"
        )
    for geom in arm_visual_geoms:
        geom.set("contype", "2")
        geom.set("conaffinity", "0")

    option = _ensure(root, "option")
    option.set("timestep", str(cfg["model"]["physics_timestep_s"]))
    option.set("integrator", "implicitfast")
    option.set("gravity", "0 0 -9.81")
    option.set("cone", "elliptic")
    option.set("impratio", "10")

    link6 = root.find(".//body[@name='Link6']")
    if link6 is None:
        raise RuntimeError("official CR10 Link6 not found after MuJoCo URDF import")

    if cfg["model"].get("idealized_arm_gravity_compensation", False):
        for name in [f"Link{i}" for i in range(1, 7)]:
            b = root.find(f".//body[@name='{name}']")
            if b is not None:
                b.set("gravcomp", "1")

    old_act = root.find("actuator")
    _remove_if(root, old_act)
    actuator = ET.SubElement(root, "actuator")
    for i, kp in enumerate(cfg["model"]["arm_position_servo_kp"], 1):
        ET.SubElement(actuator, "position", {
            "name": f"arm_servo_{i}", "joint": f"joint{i}", "kp": str(kp),
            "dampratio": "1", "inheritrange": "1"
        })

    gtree = ET.parse(src / "menagerie_2f85/2f85.xml")
    groot = gtree.getroot()
    gdefault = groot.find("default")
    if gdefault is None:
        raise RuntimeError("Menagerie 2F-85 default block missing")
    bdefault = root.find("default")
    if bdefault is None:
        bdefault = ET.Element("default")
        world = root.find("worldbody")
        idx = list(root).index(world) if world is not None else 0
        root.insert(idx, bdefault)
    for child in list(gdefault):
        bdefault.append(copy.deepcopy(child))

    shutil.copytree(src / "menagerie_2f85/assets", gen / "robotiq_2f85_assets", dirs_exist_ok=True)
    basset = _ensure(root, "asset")
    gasset = groot.find("asset")
    if gasset is None:
        raise RuntimeError("Menagerie 2F-85 asset block missing")
    for child in list(gasset):
        c = copy.deepcopy(child)
        if c.tag == "mesh" and c.get("file"):
            c.set("file", "robotiq_2f85_assets/" + Path(c.get("file")).name)
        basset.append(c)

    gworld = groot.find("worldbody")
    if gworld is None or len(list(gworld)) != 1 or list(gworld)[0].tag != "body":
        raise RuntimeError("unexpected Menagerie 2F-85 worldbody structure")

    mount = ET.SubElement(link6, "body", {
        "name": "robotiq_mount_frame",
        "pos": " ".join(str(float(v)) for v in cfg["model"]["mount_xyz_m"]),
        "quat": _rpy_to_quat(cfg["model"]["mount_rpy_rad"]),
    })
    gripper_body = copy.deepcopy(list(gworld)[0])
    mount.append(gripper_body)

    if cfg["model"].get("idealized_arm_gravity_compensation", False):
        mount.set("gravcomp", "1")
        for body in gripper_body.iter("body"):
            body.set("gravcomp", "1")

    pinch = gripper_body.find(".//site[@name='pinch']")
    if pinch is None:
        raise RuntimeError("Menagerie 2F-85 pinch site missing")
    pinch.set("name", "menagerie_pinch_site")

    ET.SubElement(mount, "site", {
        "name": str(cfg["model"]["control_site_name"]),
        "pos": " ".join(str(float(v)) for v in cfg["model"]["nominal_tcp_xyz_m"]),
        "quat": _rpy_to_quat(cfg["model"]["nominal_tcp_rpy_rad"]),
        "size": "0.006", "group": "5"
    })

    p5 = cfg.get("pack5")
    if p5 is not None:
        native_width, native_height = (
            int(value) for value in p5["native_resolution_px"]
        )
        visual = _ensure(root, "visual")
        visual_global = visual.find("global")
        if visual_global is None:
            visual_global = ET.SubElement(visual, "global")
        visual_global.set("offwidth", str(native_width))
        visual_global.set("offheight", str(native_height))

        camera_pos = [
            float(value)
            for value in p5["optical_center_xyz_m_in_mount_frame"]
        ]
        camera_target = [
            float(value) for value in p5["look_at_xyz_m_in_mount_frame"]
        ]
        camera_xyaxes, camera_z_axis = _camera_xyaxes(
            camera_pos, camera_target
        )
        principal_x, principal_y = (
            float(value) for value in p5["principal_point_px"]
        )
        mujoco_principal_offset_px = [
            native_width / 2.0 - principal_x,
            native_height / 2.0 - principal_y,
        ]
        camera_attrs = {
            "name": str(p5["camera_name"]),
            "mode": "fixed",
            "pos": " ".join(str(value) for value in camera_pos),
            "xyaxes": camera_xyaxes,
            "resolution": f"{native_width} {native_height}",
            "sensorsize": " ".join(
                str(float(value))
                for value in p5["projection_sensor_size_m"]
            ),
            "focalpixel": " ".join(
                str(float(value)) for value in p5["focal_length_px"]
            ),
            "principalpixel": " ".join(
                str(value) for value in mujoco_principal_offset_px
            ),
        }
        ET.SubElement(mount, "camera", camera_attrs)

        housing_size = [float(value) for value in p5["housing_size_m"]]
        housing_half = [value / 2.0 for value in housing_size]
        housing_center = [
            camera_pos[i] + camera_z_axis[i] * (housing_half[2] + 0.002)
            for i in range(3)
        ]
        ET.SubElement(mount, "geom", {
            "name": "d405_visual_housing",
            "type": "box",
            "size": " ".join(str(value) for value in housing_half),
            "pos": " ".join(str(value) for value in housing_center),
            "xyaxes": camera_xyaxes,
            "mass": "0",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
            "rgba": "0.06 0.07 0.08 1",
        })

    for section in ["contact", "tendon", "equality"]:
        gs = groot.find(section)
        if gs is None:
            continue
        bs = _ensure(root, section)
        for child in list(gs):
            bs.append(copy.deepcopy(child))

    gact = groot.find("actuator")
    if gact is None:
        raise RuntimeError("Menagerie 2F-85 actuator missing")
    for child in list(gact):
        c = copy.deepcopy(child)
        if c.get("name") == "fingers_actuator":
            c.set("name", "gripper_actuator")
        actuator.append(c)

    cg = cfg["known_cube_push"]
    half = float(cg["cube_size_m"]) / 2.0
    cube_body = ET.SubElement(worldbody, "body", {"name": "known_push_cube", "pos": "5 5 5"})
    ET.SubElement(cube_body, "freejoint", {"name": "known_push_cube_free"})
    ET.SubElement(cube_body, "geom", {
        "name": "known_push_cube_geom", "type": "box",
        "size": f"{half} {half} {half}", "mass": str(float(cg["cube_mass_kg"])),
        "friction": " ".join(str(float(x)) for x in cg["cube_contact_friction"]),
        "rgba": "0.82 0.18 0.12 1"
    })

    tg = cfg["table_gravity_push"]
    table_half = [float(x) / 2.0 for x in tg["table_size_m"]]
    table_body = ET.SubElement(worldbody, "body", {"name": "pack2_table", "pos": "5 5 5"})
    ET.SubElement(table_body, "geom", {
        "name": "pack2_table_geom", "type": "box",
        "size": " ".join(str(x) for x in table_half),
        "friction": " ".join(str(float(x)) for x in tg["table_friction"]),
        "contype": "1", "conaffinity": "3",
        "rgba": "0.48 0.36 0.24 1"
    })

    _check_unique_named_elements(root)
    scene = gen / "cr10_2f85_known_cube.xml"
    tree.write(scene, encoding="unicode")

    model = mujoco.MjModel.from_xml_path(str(scene))
    required = [
        (mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator"),
        (mujoco.mjtObj.mjOBJ_SITE, str(cfg["model"]["control_site_name"])),
        (mujoco.mjtObj.mjOBJ_BODY, "robotiq_mount_frame"),
        (mujoco.mjtObj.mjOBJ_CAMERA, str(cfg["pack5"]["camera_name"])),
        (mujoco.mjtObj.mjOBJ_GEOM, "d405_visual_housing"),
        (mujoco.mjtObj.mjOBJ_JOINT, "known_push_cube_free"),
        (mujoco.mjtObj.mjOBJ_GEOM, "known_push_cube_geom"),
        (mujoco.mjtObj.mjOBJ_BODY, "pack2_table"),
        (mujoco.mjtObj.mjOBJ_GEOM, "pack2_table_geom"),
        (mujoco.mjtObj.mjOBJ_GEOM, "left_pad1"),
        (mujoco.mjtObj.mjOBJ_GEOM, "left_pad2"),
        (mujoco.mjtObj.mjOBJ_GEOM, "right_pad1"),
        (mujoco.mjtObj.mjOBJ_GEOM, "right_pad2"),
    ]
    for obj, name in required:
        if mujoco.mj_name2id(model, obj, name) < 0:
            raise RuntimeError(f"compiled required element missing: {name}")
    for j in [f"joint{i}" for i in range(1, 7)]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) < 0:
            raise RuntimeError(f"compiled CR10 joint missing: {j}")
    if model.nu != 7:
        raise RuntimeError(f"expected exactly 7 actuators (6 arm + 1 gripper), got {model.nu}")
    return scene
