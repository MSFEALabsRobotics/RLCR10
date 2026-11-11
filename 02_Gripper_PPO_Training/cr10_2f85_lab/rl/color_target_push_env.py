from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ..camera import CameraIntrinsics, raycast_aligned_depth
from ..config import load_config
from ..control.bounded_planar_action import (
    BoundedPlanarActionInterface,
    PlanarActionDecision,
)
from ..control.joint_layout import JointLayout
from ..rgbd_perception import ColorDetection, detect_configured_colors


COLOR_OBSERVATION_COMPONENTS = (
    "tcp_command_forward_m",
    "tcp_command_lateral_m",
    "red_surface_minus_tcp_forward_m",
    "red_surface_minus_tcp_lateral_m",
    "red_surface_depth_m",
    "red_visible",
    "blue_surface_minus_tcp_forward_m",
    "blue_surface_minus_tcp_lateral_m",
    "blue_surface_depth_m",
    "blue_visible",
    "requested_red",
    "requested_blue",
)


def _id(model: mujoco.MjModel, object_type: Any, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise ValueError(f"required model element missing: {name}")
    return int(value)


def _descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    current = int(body_id)
    while current > 0:
        if current == ancestor_id:
            return True
        current = int(model.body_parentid[current])
    return current == ancestor_id


@dataclass(frozen=True)
class CubeIds:
    body: int
    geom: int
    joint: int
    qpos_address: int
    dof_address: int


@dataclass(frozen=True)
class ColorContactState:
    table_contact: dict[str, bool]
    gripper_contact: dict[str, bool]
    robot_table_geoms: tuple[str, ...]
    cube_cube_contact: bool
    maximum_table_penetration_m: float


class ColorTargetPushEnv(gym.Env[np.ndarray, np.ndarray]):
    """Color-conditioned RGB-D planar pushing environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scene: str | Path,
        *,
        render_mode: None = None,
        config: dict | None = None,
        diagnostic_cpu_rgbd: bool = False,
    ):
        super().__init__()
        if render_mode is not None:
            raise ValueError("use color_task_preview.py for RGB-D inspection")
        self.render_mode = render_mode
        self.scene = Path(scene).expanduser().resolve()
        if not self.scene.is_file():
            raise FileNotFoundError(f"color-task MuJoCo scene missing: {self.scene}")

        self.cfg = load_config() if config is None else config
        self.mc = self.cfg["model"]
        self.vc = self.cfg["verification"]
        self.ac = self.cfg["action"]
        self.cg = self.cfg["known_cube_push"]
        self.tg = self.cfg["table_gravity_push"]
        self.p5 = self.cfg["pack5"]
        self.ct = self.p5["color_task"]

        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        self.layout = JointLayout.from_model(self.model)
        self.model.opt.gravity[:] = np.asarray(self.tg["gravity_m_s2"], dtype=float)

        self.arm_aids = [
            _id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm_servo_{i}")
            for i in range(1, 7)
        ]
        self.gripper_aid = _id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator"
        )
        self.tcp_sid = _id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            str(self.mc["control_site_name"]),
        )
        self.mount_bid = _id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_mount_frame"
        )
        self.table_bid = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pack2_table")
        self.table_gid = _id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pack2_table_geom")
        self.camera_id = _id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            str(self.p5["camera_name"]),
        )
        self.cubes = {color: self._cube_ids(color) for color in ("red", "blue")}

        width, height = (int(value) for value in self.ct["policy_resolution_px"])
        self.intrinsics = CameraIntrinsics.from_config(self.p5).scaled(width, height)
        self.diagnostic_cpu_rgbd = bool(diagnostic_cpu_rgbd)
        self.perception_backend = (
            "CPU_GEOM_ID_DIAGNOSTIC_ONLY"
            if self.diagnostic_cpu_rgbd
            else "MUJOCO_OPENGL_RGB_PLUS_METRIC_DEPTH"
        )
        self.renderer: mujoco.Renderer | None = None
        if not self.diagnostic_cpu_rgbd:
            try:
                self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            except Exception as exc:
                raise RuntimeError(
                    "Pack 5 color task requires an OpenGL context. Run from the "
                    "Ubuntu desktop, or use MUJOCO_GL=egl on a configured GPU."
                ) from exc

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        forward_bounds = np.asarray(self.ct["tcp_forward_bounds_m"], dtype=float)
        lateral_bounds = np.asarray(self.ct["tcp_lateral_bounds_m"], dtype=float)
        low = np.asarray(
            [
                forward_bounds[0],
                lateral_bounds[0],
                -0.30,
                -0.30,
                0.0,
                0.0,
                -0.30,
                -0.30,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        high = np.asarray(
            [
                forward_bounds[1],
                lateral_bounds[1],
                0.30,
                0.30,
                float(self.ct["maximum_valid_depth_m"]),
                1.0,
                0.30,
                0.30,
                float(self.ct["maximum_valid_depth_m"]),
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.goal_displacement_m = float(self.ct["cube_goal_displacement_m"])
        self.max_episode_steps = int(self.ct["max_episode_steps"])
        self.q_target = np.zeros(6, dtype=float)
        self.gripper_command = float(self.vc["close_command"])
        self.axis_origin = np.zeros(3, dtype=float)
        self.forward_axis = np.zeros(3, dtype=float)
        self.lateral_axis = np.zeros(3, dtype=float)
        self.vertical_axis = np.zeros(3, dtype=float)
        self.tcp_rotation = np.eye(3, dtype=float)
        self.task_basis = np.eye(3, dtype=float)
        self.cube_start = {color: np.zeros(3, dtype=float) for color in ("red", "blue")}
        self.target_color = "red"
        self.interface: BoundedPlanarActionInterface | None = None
        self.episode_steps = 0
        self._episode_done = True
        self.reset_diagnostics: dict[str, Any] = {}
        self.last_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self.last_depth_m = np.zeros((height, width), dtype=np.float32)
        self.last_detections: dict[str, ColorDetection | None] = {
            "red": None,
            "blue": None,
        }

    def _cube_ids(self, color: str) -> CubeIds:
        joint = _id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"color_{color}_cube_free",
        )
        return CubeIds(
            body=_id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"color_{color}_cube",
            ),
            geom=_id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"color_{color}_cube_geom",
            ),
            joint=joint,
            qpos_address=int(self.model.jnt_qposadr[joint]),
            dof_address=int(self.model.jnt_dofadr[joint]),
        )

    def _finite(self) -> bool:
        return bool(
            np.all(np.isfinite(self.data.qpos))
            and np.all(np.isfinite(self.data.qvel))
            and np.all(np.isfinite(self.data.ctrl))
        )

    def _hold(self) -> None:
        for aid, value in zip(self.arm_aids, self.q_target):
            self.data.ctrl[aid] = float(value)
        self.data.ctrl[self.gripper_aid] = float(self.gripper_command)

    def _set_cube_pose(self, color: str, position: np.ndarray) -> None:
        cube = self.cubes[color]
        self.data.qpos[cube.qpos_address : cube.qpos_address + 3] = position
        self.data.qpos[cube.qpos_address + 3 : cube.qpos_address + 7] = np.asarray(
            [1.0, 0.0, 0.0, 0.0]
        )
        self.data.qvel[cube.dof_address : cube.dof_address + 6] = 0.0

    def _park_task_objects(self) -> None:
        self._set_cube_pose("red", np.asarray([5.0, 5.0, 5.0]))
        self._set_cube_pose("blue", np.asarray([5.2, 5.2, 5.2]))
        self.model.body_pos[self.table_bid] = np.asarray([8.0, 8.0, 8.0])

    def _contacts(self) -> ColorContactState:
        table_contact = {"red": False, "blue": False}
        gripper_contact = {"red": False, "blue": False}
        robot_table: set[str] = set()
        cube_cube = False
        maximum_penetration = 0.0
        geom_to_color = {cube.geom: color for color, cube in self.cubes.items()}
        cube_geoms = set(geom_to_color)

        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            pair = {geom1, geom2}
            if pair == cube_geoms:
                cube_cube = True

            for color, cube in self.cubes.items():
                if pair == {cube.geom, self.table_gid}:
                    table_contact[color] = True
                    maximum_penetration = max(
                        maximum_penetration, max(0.0, -float(contact.dist))
                    )
                if cube.geom not in (geom1, geom2):
                    continue
                other = geom2 if geom1 == cube.geom else geom1
                if _descendant(
                    self.model,
                    int(self.model.geom_bodyid[other]),
                    self.mount_bid,
                ):
                    gripper_contact[color] = True

            if self.table_gid in pair:
                other = geom2 if geom1 == self.table_gid else geom1
                if other not in cube_geoms:
                    name = (
                        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                        or f"geom_{other}"
                    )
                    robot_table.add(name)

        return ColorContactState(
            table_contact=table_contact,
            gripper_contact=gripper_contact,
            robot_table_geoms=tuple(sorted(robot_table)),
            cube_cube_contact=bool(cube_cube),
            maximum_table_penetration_m=float(maximum_penetration),
        )

    def _cube_tilt_deg(self, color: str) -> float:
        cube_up = self.data.xmat[self.cubes[color].body].reshape(3, 3)[:, 2]
        cosine = float(np.clip(cube_up[2], -1.0, 1.0))
        return float(math.degrees(math.acos(cosine)))

    def _cube_progress_m(self, color: str) -> float:
        return float(
            np.dot(
                self.data.xpos[self.cubes[color].body] - self.cube_start[color],
                self.forward_axis,
            )
        )

    def _cube_planar_displacement_m(self, color: str) -> float:
        delta = self.data.xpos[self.cubes[color].body] - self.cube_start[color]
        planar = np.asarray(
            [np.dot(delta, self.forward_axis), np.dot(delta, self.lateral_axis)]
        )
        return float(np.linalg.norm(planar))

    @property
    def wrong_color(self) -> str:
        return "blue" if self.target_color == "red" else "red"

    def _runtime_fault(
        self,
        contacts: ColorContactState,
        *,
        perception_valid: bool = True,
    ) -> str | None:
        if not self._finite():
            return "NONFINITE_STATE"
        if contacts.robot_table_geoms:
            return "ROBOT_TABLE_CONTACT:" + ",".join(contacts.robot_table_geoms)
        if contacts.maximum_table_penetration_m > float(
            self.tg["max_contact_penetration_m"]
        ):
            return "CUBE_TABLE_PENETRATION"
        if contacts.cube_cube_contact:
            return "CUBE_CUBE_CONTACT"
        if contacts.gripper_contact[self.wrong_color]:
            return "WRONG_CUBE_CONTACT"
        if self._cube_planar_displacement_m(self.wrong_color) > float(
            self.ct["maximum_wrong_cube_displacement_m"]
        ):
            return "WRONG_CUBE_DISPLACEMENT"
        for color in ("red", "blue"):
            cube_z = float(self.data.xpos[self.cubes[color].body][2])
            if abs(cube_z - float(self.cube_start[color][2])) > float(
                self.tg["max_cube_vertical_drift_m"]
            ):
                return f"{color.upper()}_CUBE_VERTICAL_DRIFT"
            if self._cube_tilt_deg(color) > float(self.tg["max_cube_tilt_deg"]):
                return f"{color.upper()}_CUBE_TILT"
        if not perception_valid:
            return "RGBD_COLOR_TARGET_LOST"
        return None

    def _render_perception(self) -> dict[str, ColorDetection | None]:
        if self.diagnostic_cpu_rgbd:
            depth, geom_ids = raycast_aligned_depth(
                self.model, self.data, self.camera_id, self.intrinsics
            )
            rgb = np.zeros(
                (self.intrinsics.height, self.intrinsics.width, 3),
                dtype=np.uint8,
            )
            for color, cube in self.cubes.items():
                rgba = np.asarray(
                    self.ct["cube_colors"][color]["rgba"][:3], dtype=float
                )
                rgb[geom_ids == cube.geom] = np.asarray(
                    np.rint(255.0 * rgba), dtype=np.uint8
                )
        else:
            if self.renderer is None:
                raise RuntimeError("OpenGL renderer is not initialized")
            self.renderer.update_scene(self.data, camera=self.camera_id)
            rgb = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            try:
                self.renderer.update_scene(self.data, camera=self.camera_id)
                depth = self.renderer.render().copy()
            finally:
                self.renderer.disable_depth_rendering()
        detections = detect_configured_colors(
            rgb=rgb,
            depth_m=depth,
            intrinsics=self.intrinsics,
            camera_position_world_m=self.data.cam_xpos[self.camera_id],
            camera_rotation_world=self.data.cam_xmat[self.camera_id].reshape(3, 3),
            color_task_config=self.ct,
        )
        self.last_rgb = rgb
        self.last_depth_m = depth
        self.last_detections = detections
        return detections

    def rgbd_frame(
        self,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, ColorDetection | None]]:
        detections = self._render_perception()
        return self.last_rgb.copy(), self.last_depth_m.copy(), dict(detections)

    def _observation(
        self,
        detections: dict[str, ColorDetection | None],
        contacts: ColorContactState,
    ) -> np.ndarray:
        if self.interface is None:
            command = np.zeros(2, dtype=float)
        else:
            command = self.interface.planar_m
        tcp = self.data.site_xpos[self.tcp_sid]
        perceived: list[float] = []
        for color in ("red", "blue"):
            detection = detections[color]
            if detection is None:
                perceived.extend([0.0, 0.0, 0.0, 0.0])
                continue
            relative = self.task_basis.T @ (detection.point_world_xyz_m - tcp)
            perceived.extend(
                [
                    float(relative[0]),
                    float(relative[1]),
                    float(detection.median_depth_m),
                    1.0,
                ]
            )
        requested = [
            float(self.target_color == "red"),
            float(self.target_color == "blue"),
        ]
        observation = np.asarray(
            [
                float(command[0]),
                float(command[1]),
                *perceived,
                *requested,
            ],
            dtype=np.float32,
        )
        if observation.shape != (len(COLOR_OBSERVATION_COMPONENTS),):
            raise RuntimeError("internal color-task observation shape mismatch")
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("non-finite color-task observation")
        return np.clip(
            observation,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32)

    def _is_success(self, contacts: ColorContactState) -> bool:
        return bool(
            self._cube_progress_m(self.target_color) >= self.goal_displacement_m
            and contacts.table_contact[self.target_color]
            and contacts.table_contact[self.wrong_color]
            and self._cube_planar_displacement_m(self.wrong_color)
            <= float(self.ct["maximum_wrong_cube_displacement_m"])
        )

    def _base_info(
        self,
        contacts: ColorContactState,
        detections: dict[str, ColorDetection | None],
    ) -> dict[str, Any]:
        return {
            "observation_schema_version": str(self.ct["observation_schema"]),
            "observation_components": list(COLOR_OBSERVATION_COMPONENTS),
            "target_color": self.target_color,
            "wrong_color": self.wrong_color,
            "target_progress_m": self._cube_progress_m(self.target_color),
            "wrong_cube_displacement_m": self._cube_planar_displacement_m(
                self.wrong_color
            ),
            "goal_displacement_m": self.goal_displacement_m,
            "is_success": self._is_success(contacts),
            "table_contact": dict(contacts.table_contact),
            "gripper_contact": dict(contacts.gripper_contact),
            "cube_cube_contact": contacts.cube_cube_contact,
            "robot_table_contact_geoms": list(contacts.robot_table_geoms),
            "perception": {
                color: detection.as_dict() if detection is not None else None
                for color, detection in detections.items()
            },
            "rgbd_resolution_px": [
                self.intrinsics.width,
                self.intrinsics.height,
            ],
            "perception_backend": self.perception_backend,
            "episode_step": int(self.episode_steps),
            "hardware_motion_commanded": False,
        }

    def _place_task(self, layout_sign: int) -> None:
        mujoco.mj_forward(self.model, self.data)
        tcp = self.data.site_xpos[self.tcp_sid].copy()
        rotation = self.data.site_xmat[self.tcp_sid].reshape(3, 3).copy()
        forward = rotation[:, 2].copy()
        forward /= np.linalg.norm(forward)
        if abs(float(forward[2])) > 0.03:
            raise RuntimeError("color-task push axis is not horizontal")
        lateral = np.cross(np.asarray([0.0, 0.0, 1.0]), forward)
        lateral /= np.linalg.norm(lateral)
        vertical = np.cross(forward, lateral)
        vertical /= np.linalg.norm(vertical)
        self.axis_origin = tcp
        self.forward_axis = forward
        self.lateral_axis = lateral
        self.vertical_axis = vertical
        self.tcp_rotation = rotation
        self.task_basis = np.column_stack((forward, lateral, vertical))

        cube_half = float(self.cg["cube_size_m"]) / 2.0
        forward_distance = cube_half + float(self.cg["initial_gap_m"])
        offset = float(self.ct["cube_lateral_offset_m"])
        jitter = float(self.ct["cube_lateral_jitter_m"])
        red_lateral = float(layout_sign) * offset + float(
            self.np_random.uniform(-jitter, jitter)
        )
        blue_lateral = -float(layout_sign) * offset + float(
            self.np_random.uniform(-jitter, jitter)
        )
        if abs(red_lateral - blue_lateral) <= float(self.cg["cube_size_m"]):
            raise RuntimeError("randomized color cubes overlap")
        center = tcp + forward * forward_distance
        self._set_cube_pose("red", center + lateral * red_lateral)
        self._set_cube_pose("blue", center + lateral * blue_lateral)

        table_length = float(self.tg["table_size_m"][0])
        table_height = float(self.tg["table_size_m"][2])
        table_center = center + forward * (
            table_length / 2.0 - float(self.tg["table_back_support_m"])
        )
        table_center[2] = center[2] - cube_half - table_height / 2.0
        table_yaw = math.atan2(float(forward[1]), float(forward[0]))
        self.model.body_pos[self.table_bid] = table_center
        self.model.body_quat[self.table_bid] = np.asarray(
            [math.cos(table_yaw / 2.0), 0.0, 0.0, math.sin(table_yaw / 2.0)]
        )
        mujoco.mj_forward(self.model, self.data)

    def _reset_task(self, options: dict[str, Any]) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.q_target = np.deg2rad(np.asarray(self.mc["push_pose_deg"], dtype=float))
        self.layout.set_q(self.data, self.q_target)
        self.layout.set_qdot_zero(self.data)
        self.gripper_command = float(self.vc["close_command"])
        self._park_task_objects()
        self._hold()
        mujoco.mj_forward(self.model, self.data)

        gripper_steps = int(
            math.ceil(float(self.vc["gripper_settle_s"]) / self.model.opt.timestep)
        )
        for _ in range(gripper_steps):
            self._hold()
            mujoco.mj_step(self.model, self.data)
            if not self._finite():
                raise RuntimeError("non-finite color-task gripper settle")

        target = options.get("target_color")
        if target is None:
            target = ("red", "blue")[int(self.np_random.integers(0, 2))]
        if target not in ("red", "blue"):
            raise ValueError("target_color must be 'red' or 'blue'")
        self.target_color = str(target)
        layout_sign_option = options.get("red_lateral_sign")
        if layout_sign_option is None:
            layout_sign = -1 if int(self.np_random.integers(0, 2)) == 0 else 1
        else:
            layout_sign = int(layout_sign_option)
            if layout_sign not in (-1, 1):
                raise ValueError("red_lateral_sign must be -1 or +1")
        self._place_task(layout_sign)

        settle_steps = int(
            math.ceil(float(self.tg["cube_settle_s"]) / self.model.opt.timestep)
        )
        maximum_penetration = 0.0
        table_counts = {"red": 0, "blue": 0}
        for _ in range(settle_steps):
            self._hold()
            mujoco.mj_step(self.model, self.data)
            contacts = self._contacts()
            for color in ("red", "blue"):
                table_counts[color] += int(contacts.table_contact[color])
            maximum_penetration = max(
                maximum_penetration, contacts.maximum_table_penetration_m
            )
            if contacts.robot_table_geoms or contacts.cube_cube_contact:
                raise RuntimeError("unsafe contact during color-task reset")
            if any(contacts.gripper_contact.values()):
                raise RuntimeError("color cube overlaps gripper at reset")

        contacts = self._contacts()
        if not all(contacts.table_contact.values()):
            raise RuntimeError("both color cubes must retain table support")
        minimum_table_steps = int(self.tg["min_table_contact_substeps"])
        if any(count < minimum_table_steps for count in table_counts.values()):
            raise RuntimeError("color cubes did not establish sustained support")
        if maximum_penetration > float(self.tg["max_contact_penetration_m"]):
            raise RuntimeError("color cube/table penetration exceeded gate")

        self.cube_start = {
            color: self.data.xpos[cube.body].copy()
            for color, cube in self.cubes.items()
        }
        self.interface = BoundedPlanarActionInterface(
            self.model,
            self.data,
            site_name=str(self.mc["control_site_name"]),
            origin_world_m=self.axis_origin,
            forward_axis_world=self.forward_axis,
            lateral_axis_world=self.lateral_axis,
            fixed_tcp_rotation_world=self.tcp_rotation,
            forward_bounds_m=self.ct["tcp_forward_bounds_m"],
            lateral_bounds_m=self.ct["tcp_lateral_bounds_m"],
            fixed_gripper_command=self.gripper_command,
            action_config=self.ac,
            model_config=self.mc,
        )
        self.episode_steps = 0
        self._episode_done = False

        detections = self._render_perception()
        if any(detection is None for detection in detections.values()):
            raise RuntimeError("RGB-D reset does not detect both configured colors")
        tcp = self.data.site_xpos[self.tcp_sid]
        lateral_errors: dict[str, float] = {}
        for color in ("red", "blue"):
            detection = detections[color]
            assert detection is not None
            perceived_lateral = float(
                np.dot(
                    detection.point_world_xyz_m - tcp,
                    self.lateral_axis,
                )
            )
            actual_lateral = float(
                np.dot(
                    self.data.xpos[self.cubes[color].body] - tcp,
                    self.lateral_axis,
                )
            )
            lateral_errors[color] = abs(perceived_lateral - actual_lateral)
        maximum_error = max(lateral_errors.values())
        if maximum_error > float(self.ct["maximum_reset_lateral_perception_error_m"]):
            raise RuntimeError(
                "RGB-D lateral localization error exceeds configured gate: "
                f"{maximum_error:.6f} m"
            )
        self.reset_diagnostics = {
            "target_color": self.target_color,
            "red_lateral_sign": int(layout_sign),
            "table_contact_substeps": table_counts,
            "maximum_table_penetration_m": float(maximum_penetration),
            "lateral_perception_error_m": lateral_errors,
            "color_mask_pixels": {
                color: int(detections[color].mask_pixels)  # type: ignore[union-attr]
                for color in ("red", "blue")
            },
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        selected_options = {} if options is None else dict(options)
        unknown = set(selected_options) - {"target_color", "red_lateral_sign"}
        if unknown:
            raise ValueError(f"unsupported color-task reset options: {sorted(unknown)}")
        self._reset_task(selected_options)
        contacts = self._contacts()
        detections = self.last_detections
        info = self._base_info(contacts, detections)
        info.update(
            {
                "reset_is_deterministic_for_seed_and_options": True,
                "reset_randomization": (
                    "target color, red/blue side assignment, and bounded lateral jitter"
                ),
                "reset_diagnostics": dict(self.reset_diagnostics),
                "safety_fault": None,
            }
        )
        return self._observation(detections, contacts), info

    def _rejected_step(
        self,
        decision: PlanarActionDecision,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.episode_steps += 1
        contacts = self._contacts()
        detections = self._render_perception()
        reward = float(self.ct["invalid_action_penalty"])
        truncated = self.episode_steps >= self.max_episode_steps
        info = self._base_info(contacts, detections)
        info.update(
            {
                "action_accepted": False,
                "action_code": decision.code,
                "action_detail": decision.detail,
                "simulation_substeps": 0,
                "tcp_tracking_error_m": 0.0,
                "safety_fault": None,
                "reward_terms": {
                    "approach": 0.0,
                    "target_progress": 0.0,
                    "wrong_cube_motion": 0.0,
                    "success_bonus": 0.0,
                    "invalid_action_penalty": reward,
                    "fault_penalty": 0.0,
                },
                "time_limit_reached": bool(truncated),
            }
        )
        self._episode_done = bool(truncated)
        return self._observation(detections, contacts), reward, False, truncated, info

    def _target_tcp_distance_m(self) -> float:
        tcp = self.data.site_xpos[self.tcp_sid]
        cube = self.data.xpos[self.cubes[self.target_color].body]
        delta = cube - tcp
        return float(
            np.linalg.norm(
                [np.dot(delta, self.forward_axis), np.dot(delta, self.lateral_axis)]
            )
        )

    def step(
        self,
        action: Sequence[float],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("episode is done; call reset() before step()")
        if self.interface is None:
            raise RuntimeError("color-task interface is not initialized")
        mujoco.mj_forward(self.model, self.data)
        decision = self.interface.plan_normalized(action)
        if not decision.accepted:
            return self._rejected_step(decision)

        target_progress_before = self._cube_progress_m(self.target_color)
        wrong_motion_before = self._cube_planar_displacement_m(self.wrong_color)
        approach_before = self._target_tcp_distance_m()
        self.interface.commit(decision)
        assert decision.q_target_rad is not None
        assert decision.gripper_command is not None
        assert decision.target_tcp_world_m is not None
        self.q_target = decision.q_target_rad.copy()
        self.gripper_command = float(decision.gripper_command)

        minimum_substeps = int(
            math.ceil(float(self.mc["control_timestep_s"]) / self.model.opt.timestep)
        )
        maximum_substeps = int(
            math.ceil(float(self.cg["waypoint_timeout_s"]) / self.model.opt.timestep)
        )
        required_streak = int(self.cg["waypoint_settle_substeps"])
        tolerance = float(self.cg["waypoint_tracking_tolerance_m"])
        streak = 0
        tracking_error = math.inf
        fault: str | None = None
        reached = False
        substeps = 0
        for substep in range(1, maximum_substeps + 1):
            self._hold()
            mujoco.mj_step(self.model, self.data)
            substeps = substep
            contacts = self._contacts()
            fault = self._runtime_fault(contacts)
            if fault is not None:
                break
            tracking_error = float(
                np.linalg.norm(
                    self.data.site_xpos[self.tcp_sid] - decision.target_tcp_world_m
                )
            )
            if tracking_error <= tolerance:
                streak += 1
            else:
                streak = 0
            if substep >= minimum_substeps and streak >= required_streak:
                reached = True
                break
        if fault is None and not reached:
            fault = "TCP_ACTION_TIMEOUT"

        self.episode_steps += 1
        contacts = self._contacts()
        detections = self._render_perception()
        perception_valid = all(
            detection is not None for detection in detections.values()
        )
        if fault is None:
            fault = self._runtime_fault(contacts, perception_valid=perception_valid)

        target_progress_after = self._cube_progress_m(self.target_color)
        wrong_motion_after = self._cube_planar_displacement_m(self.wrong_color)
        approach_after = self._target_tcp_distance_m()
        approach_reward = (
            float(self.ct["approach_reward_weight"])
            * (approach_before - approach_after)
            / float(self.ac["max_xy_step_m"])
        )
        progress_reward = (
            target_progress_after - target_progress_before
        ) / self.goal_displacement_m
        wrong_penalty = (
            -float(self.ct["wrong_cube_motion_penalty_weight"])
            * max(0.0, wrong_motion_after - wrong_motion_before)
            / float(self.ct["maximum_wrong_cube_displacement_m"])
        )
        success = bool(fault is None and self._is_success(contacts))
        success_bonus = float(self.ct["success_bonus"]) if success else 0.0
        fault_penalty = float(self.ct["fault_penalty"]) if fault else 0.0
        reward = float(
            approach_reward
            + progress_reward
            + wrong_penalty
            + success_bonus
            + fault_penalty
        )
        terminated = success
        time_limit = self.episode_steps >= self.max_episode_steps
        truncated = bool(fault is not None or (time_limit and not success))
        self._episode_done = bool(terminated or truncated)

        info = self._base_info(contacts, detections)
        info.update(
            {
                "action_accepted": True,
                "action_code": decision.code,
                "action_detail": decision.detail,
                "normalized_action": list(decision.normalized_action or ()),
                "mapped_planar_increment_m": (
                    decision.delta_planar_m.tolist()
                    if decision.delta_planar_m is not None
                    else None
                ),
                "commanded_planar_m": self.interface.planar_m.tolist(),
                "simulation_substeps": int(substeps),
                "tcp_tracking_error_m": float(tracking_error),
                "safety_fault": fault,
                "reward_terms": {
                    "approach": float(approach_reward),
                    "target_progress": float(progress_reward),
                    "wrong_cube_motion": float(wrong_penalty),
                    "success_bonus": float(success_bonus),
                    "invalid_action_penalty": 0.0,
                    "fault_penalty": float(fault_penalty),
                },
                "time_limit_reached": bool(time_limit),
                "is_success": success,
            }
        )
        return (
            self._observation(detections, contacts),
            reward,
            terminated,
            truncated,
            info,
        )

    def state_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.ctrl.copy(),
        )

    def contract(self) -> dict[str, Any]:
        if self.interface is None:
            raise RuntimeError("call reset() before reading the contract")
        return {
            "environment_id": str(self.ct["environment_id"]),
            "observation_schema_version": str(self.ct["observation_schema"]),
            "observation_components": list(COLOR_OBSERVATION_COMPONENTS),
            "observation_shape": list(self.observation_space.shape),
            "perception": {
                "rgb": "configured HSV masks identify red and blue",
                "depth": "aligned metric depth back-projects visible mask pixels",
                "policy_pixels": [self.intrinsics.width, self.intrinsics.height],
                "policy_receives_raw_images": False,
                "policy_receives_compact_rgbd_features": True,
                "backend": self.perception_backend,
            },
            "target_command": "one-hot requested color: red or blue",
            "action": self.interface.contract(),
            "fixed_gripper_normalized": float(self.ct["fixed_gripper_normalized"]),
            "goal_displacement_m": self.goal_displacement_m,
            "maximum_wrong_cube_displacement_m": float(
                self.ct["maximum_wrong_cube_displacement_m"]
            ),
            "max_episode_steps": self.max_episode_steps,
            "reward": {
                "approach_weight": float(self.ct["approach_reward_weight"]),
                "target_progress": "delta target progress / goal displacement",
                "wrong_cube_motion_penalty_weight": float(
                    self.ct["wrong_cube_motion_penalty_weight"]
                ),
                "success_bonus": float(self.ct["success_bonus"]),
                "invalid_action_penalty": float(self.ct["invalid_action_penalty"]),
                "fault_penalty": float(self.ct["fault_penalty"]),
                "force_used_in_reward": False,
            },
            "reset_randomization": [
                "requested target color",
                "red/blue side assignment",
                "bounded lateral jitter",
            ],
            "hardware_interface": False,
            "claim_boundary": str(self.ct["claim"]),
        }

    def close(self) -> None:
        self._episode_done = True
        renderer = getattr(self, "renderer", None)
        if renderer is not None:
            renderer.close()
            self.renderer = None


def scripted_color_action(
    observation: np.ndarray,
    *,
    lateral_tolerance_m: float = 0.0025,
    max_step_m: float = 0.002,
) -> np.ndarray:
    """Reference controller using only the declared compact observation."""

    obs = np.asarray(observation, dtype=float).reshape(
        len(COLOR_OBSERVATION_COMPONENTS)
    )
    if obs[10] > 0.5:
        relative_forward, relative_lateral, visible = obs[2], obs[3], obs[5]
    else:
        relative_forward, relative_lateral, visible = obs[6], obs[7], obs[9]
    if visible < 0.5:
        return np.zeros(2, dtype=np.float32)
    lateral_command = float(np.clip(relative_lateral / max_step_m, -1.0, 1.0))
    if abs(relative_lateral) > lateral_tolerance_m:
        return np.asarray([0.0, lateral_command], dtype=np.float32)
    forward_command = 1.0 if relative_forward > -0.02 else 0.0
    return np.asarray([forward_command, lateral_command], dtype=np.float32)
