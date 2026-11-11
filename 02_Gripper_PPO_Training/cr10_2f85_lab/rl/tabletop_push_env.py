from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ..config import load_config
from ..control.bounded_push_action import (
    ActionDecision,
    BoundedPushActionInterface,
)
from ..control.joint_layout import JointLayout
from ..telemetry import SimulationTelemetry


OBSERVATION_SCHEMA_VERSION = "cr10_2f85_pack4_state_observation_v1"
OBSERVATION_COMPONENTS = (
    "tcp_actual_segment_m",
    "tcp_commanded_segment_m",
    "cube_progress_along_goal_m",
    "cube_goal_remaining_m",
    "cube_minus_tcp_along_m",
    "cube_minus_tcp_lateral_m",
    "cube_minus_tcp_vertical_m",
    "cube_linear_velocity_along_m_s",
    "cube_linear_velocity_lateral_m_s",
    "cube_linear_velocity_vertical_m_s",
    "gripper_position_command_normalized",
    "gripper_cube_contact",
    "cube_table_contact",
)


def _id(model: mujoco.MjModel, obj: Any, name: str) -> int:
    value = mujoco.mj_name2id(model, obj, name)
    if value < 0:
        raise ValueError(f"required model element missing: {name}")
    return int(value)


def _descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    body = int(body_id)
    while body > 0:
        if body == ancestor_id:
            return True
        body = int(model.body_parentid[body])
    return body == ancestor_id


@dataclass(frozen=True)
class ContactState:
    cube_table: bool
    gripper_cube: bool
    robot_table_geoms: tuple[str, ...]
    cube_table_max_penetration_m: float


class TabletopPushEnv(gym.Env[np.ndarray, np.ndarray]):
    """State-based tabletop push environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scene: str | Path,
        *,
        render_mode: None = None,
        config: dict | None = None,
    ):
        super().__init__()
        if render_mode is not None:
            raise ValueError(
                "Pack 4 has no image observation/render contract; use the "
                "current color-task replay utility for visualization"
            )
        self.render_mode = render_mode
        self.scene = Path(scene).expanduser().resolve()
        if not self.scene.is_file():
            raise FileNotFoundError(f"generated MuJoCo scene missing: {self.scene}")

        self.cfg = load_config() if config is None else config
        self.mc = self.cfg["model"]
        self.vc = self.cfg["verification"]
        self.ac = self.cfg["action"]
        self.cg = self.cfg["known_cube_push"]
        self.tg = self.cfg["table_gravity_push"]
        self.p4 = self.cfg["pack4"]

        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        self.layout = JointLayout.from_model(self.model)
        self.model.opt.gravity[:] = np.asarray(
            self.tg["gravity_m_s2"], dtype=float
        )

        self.arm_aids = [
            _id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"arm_servo_{index}",
            )
            for index in range(1, 7)
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
        self.cube_bid = _id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "known_push_cube"
        )
        self.cube_gid = _id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "known_push_cube_geom"
        )
        self.cube_jid = _id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "known_push_cube_free"
        )
        self.table_bid = _id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pack2_table"
        )
        self.table_gid = _id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "pack2_table_geom"
        )
        self.cube_qadr = int(self.model.jnt_qposadr[self.cube_jid])
        self.cube_dadr = int(self.model.jnt_dofadr[self.cube_jid])
        self.telemetry = SimulationTelemetry(self.model, self.data)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        obs_low = np.full(len(OBSERVATION_COMPONENTS), -np.inf, dtype=np.float32)
        obs_high = np.full(len(OBSERVATION_COMPONENTS), np.inf, dtype=np.float32)
        obs_low[10:] = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
        obs_high[10:] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        self.goal_displacement_m = float(
            self.p4["cube_goal_displacement_m"]
        )
        self.max_episode_steps = int(self.p4["max_episode_steps"])
        if not 0.0 < self.goal_displacement_m < float(
            self.cg["push_distance_m"]
        ):
            raise ValueError("Pack 4 cube goal must lie inside the push segment")
        if self.max_episode_steps <= 0:
            raise ValueError("Pack 4 episode horizon must be positive")

        self.q_target = np.zeros(6, dtype=float)
        self.gripper_command = float(self.vc["close_command"])
        self.axis_origin = np.zeros(3, dtype=float)
        self.axis_direction = np.zeros(3, dtype=float)
        self.axis_rotation = np.eye(3, dtype=float)
        self.task_basis = np.eye(3, dtype=float)
        self.cube_start = np.zeros(3, dtype=float)
        self.interface: BoundedPushActionInterface | None = None
        self.reset_diagnostics: dict[str, float | int] = {}
        self.episode_steps = 0
        self._episode_done = True

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

    def _park_task_objects(self) -> None:
        self.data.qpos[self.cube_qadr : self.cube_qadr + 3] = np.array(
            [5.0, 5.0, 5.0]
        )
        self.data.qpos[self.cube_qadr + 3 : self.cube_qadr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0]
        )
        self.data.qvel[self.cube_dadr : self.cube_dadr + 6] = 0.0
        self.model.body_pos[self.table_bid] = np.array([8.0, 8.0, 8.0])

    def _place_supported_cube(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        tcp = self.data.site_xpos[self.tcp_sid].copy()
        rotation = self.data.site_xmat[self.tcp_sid].reshape(3, 3).copy()
        axis = rotation[:, 2].copy()
        axis /= np.linalg.norm(axis)
        if abs(float(axis[2])) > 0.03:
            raise RuntimeError("Pack 4 push axis is not horizontal")

        cube_half = float(self.cg["cube_size_m"]) / 2.0
        cube_center = tcp + axis * (
            cube_half + float(self.cg["initial_gap_m"])
        )
        table_length = float(self.tg["table_size_m"][0])
        table_height = float(self.tg["table_size_m"][2])
        table_axis = axis.copy()
        table_axis[2] = 0.0
        table_axis /= np.linalg.norm(table_axis)
        table_center = cube_center + table_axis * (
            table_length / 2.0 - float(self.tg["table_back_support_m"])
        )
        table_center[2] = cube_center[2] - cube_half - table_height / 2.0
        table_yaw = math.atan2(float(table_axis[1]), float(table_axis[0]))
        self.model.body_pos[self.table_bid] = table_center
        self.model.body_quat[self.table_bid] = np.array(
            [math.cos(table_yaw / 2.0), 0.0, 0.0, math.sin(table_yaw / 2.0)]
        )

        self.data.qpos[self.cube_qadr : self.cube_qadr + 3] = cube_center
        self.data.qpos[self.cube_qadr + 3 : self.cube_qadr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0]
        )
        self.data.qvel[self.cube_dadr : self.cube_dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        world_up = np.array([0.0, 0.0, 1.0])
        lateral = np.cross(world_up, axis)
        lateral /= np.linalg.norm(lateral)
        vertical = np.cross(axis, lateral)
        vertical /= np.linalg.norm(vertical)
        self.axis_origin = tcp
        self.axis_direction = axis
        self.axis_rotation = rotation
        self.task_basis = np.column_stack((axis, lateral, vertical))

    def _contacts(self) -> ContactState:
        cube_table = False
        gripper_cube = False
        robot_table: set[str] = set()
        max_penetration = 0.0
        for contact_id in range(int(self.data.ncon)):
            contact = self.data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if {geom1, geom2} == {self.cube_gid, self.table_gid}:
                cube_table = True
                max_penetration = max(
                    max_penetration, max(0.0, -float(contact.dist))
                )
            elif self.table_gid in (geom1, geom2):
                other = geom2 if geom1 == self.table_gid else geom1
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, other
                ) or f"geom_{other}"
                robot_table.add(name)

            if self.cube_gid not in (geom1, geom2):
                continue
            other = geom2 if geom1 == self.cube_gid else geom1
            if _descendant(
                self.model,
                int(self.model.geom_bodyid[other]),
                self.mount_bid,
            ):
                gripper_cube = True

        return ContactState(
            cube_table=cube_table,
            gripper_cube=gripper_cube,
            robot_table_geoms=tuple(sorted(robot_table)),
            cube_table_max_penetration_m=float(max_penetration),
        )

    def _cube_tilt_deg(self) -> float:
        cube_up = self.data.xmat[self.cube_bid].reshape(3, 3)[:, 2]
        cosine = float(
            np.clip(np.dot(cube_up, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)
        )
        return float(math.degrees(math.acos(cosine)))

    def _runtime_fault(self, contacts: ContactState) -> str | None:
        if not self._finite():
            return "NONFINITE_STATE"
        if contacts.robot_table_geoms:
            return "ROBOT_TABLE_CONTACT:" + ",".join(
                contacts.robot_table_geoms
            )
        if contacts.cube_table_max_penetration_m > float(
            self.tg["max_contact_penetration_m"]
        ):
            return "CUBE_TABLE_PENETRATION"
        cube_now = self.data.xpos[self.cube_bid]
        if abs(float(cube_now[2] - self.cube_start[2])) > float(
            self.tg["max_cube_vertical_drift_m"]
        ):
            return "CUBE_VERTICAL_DRIFT"
        if self._cube_tilt_deg() > float(self.tg["max_cube_tilt_deg"]):
            return "CUBE_TILT"
        return None

    def _cube_progress_m(self) -> float:
        return float(
            np.dot(
                self.data.xpos[self.cube_bid] - self.cube_start,
                self.axis_direction,
            )
        )

    def _observation(self) -> np.ndarray:
        if self.interface is None:
            raise RuntimeError("Pack 4 task interface is not initialized")
        contacts = self._contacts()
        tcp = self.data.site_xpos[self.tcp_sid]
        cube = self.data.xpos[self.cube_bid]
        cube_minus_tcp = self.task_basis.T @ (cube - tcp)
        cube_spatial_velocity = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.cube_bid,
            cube_spatial_velocity,
            0,
        )
        cube_velocity_world = cube_spatial_velocity[3:]
        cube_velocity_task = self.task_basis.T @ cube_velocity_world
        cube_progress = self._cube_progress_m()
        actual_segment = float(np.dot(tcp - self.axis_origin, self.axis_direction))
        gripper_normalized = self.interface.gripper_command_to_normalized(
            float(self.data.ctrl[self.gripper_aid])
        )
        observation = np.asarray(
            [
                actual_segment,
                float(self.interface.segment_m),
                cube_progress,
                self.goal_displacement_m - cube_progress,
                *cube_minus_tcp.tolist(),
                *cube_velocity_task.tolist(),
                gripper_normalized,
                float(contacts.gripper_cube),
                float(contacts.cube_table),
            ],
            dtype=np.float32,
        )
        if observation.shape != self.observation_space.shape:
            raise RuntimeError("internal Pack 4 observation shape mismatch")
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("non-finite Pack 4 observation")
        return observation

    def _base_info(self, contacts: ContactState) -> dict[str, Any]:
        progress = self._cube_progress_m()
        return {
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "observation_components": list(OBSERVATION_COMPONENTS),
            "cube_progress_m": progress,
            "cube_goal_displacement_m": self.goal_displacement_m,
            "cube_goal_remaining_m": self.goal_displacement_m - progress,
            "is_success": bool(
                progress >= self.goal_displacement_m and contacts.cube_table
            ),
            "gripper_command_0_to_255": float(
                self.data.ctrl[self.gripper_aid]
            ),
            "gripper_cube_contact": bool(contacts.gripper_cube),
            "cube_table_contact": bool(contacts.cube_table),
            "cube_table_max_penetration_m": float(
                contacts.cube_table_max_penetration_m
            ),
            "robot_table_contact_geoms": list(contacts.robot_table_geoms),
            "episode_step": int(self.episode_steps),
            "hardware_motion_commanded": False,
        }

    def _reset_task(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.q_target = np.deg2rad(
            np.asarray(self.mc["push_pose_deg"], dtype=float)
        )
        self.layout.set_q(self.data, self.q_target)
        self.layout.set_qdot_zero(self.data)
        self.gripper_command = float(self.vc["close_command"])
        self._park_task_objects()
        self._hold()
        mujoco.mj_forward(self.model, self.data)

        gripper_settle_steps = int(
            math.ceil(
                float(self.vc["gripper_settle_s"]) / self.model.opt.timestep
            )
        )
        for _ in range(gripper_settle_steps):
            self._hold()
            mujoco.mj_step(self.model, self.data)
            if not self._finite():
                raise RuntimeError("non-finite state during Pack 4 gripper settle")

        self._place_supported_cube()
        initial_contacts = self._contacts()
        max_penetration = initial_contacts.cube_table_max_penetration_m
        table_contact_substeps = int(initial_contacts.cube_table)
        if initial_contacts.robot_table_geoms:
            raise RuntimeError(
                "robot/table contact at Pack 4 reset: "
                + ",".join(initial_contacts.robot_table_geoms)
            )
        if initial_contacts.gripper_cube:
            raise RuntimeError("cube initially overlaps the Pack 4 gripper")
        cube_settle_steps = int(
            math.ceil(float(self.tg["cube_settle_s"]) / self.model.opt.timestep)
        )
        for _ in range(cube_settle_steps):
            self._hold()
            mujoco.mj_step(self.model, self.data)
            if not self._finite():
                raise RuntimeError("non-finite state during Pack 4 cube settle")
            contacts = self._contacts()
            table_contact_substeps += int(contacts.cube_table)
            max_penetration = max(
                max_penetration, contacts.cube_table_max_penetration_m
            )
            if contacts.robot_table_geoms:
                raise RuntimeError(
                    "robot/table contact during Pack 4 reset: "
                    + ",".join(contacts.robot_table_geoms)
                )
            if contacts.gripper_cube:
                raise RuntimeError(
                    "cube contacted gripper before the Pack 4 episode"
                )

        contacts = self._contacts()
        if not contacts.cube_table:
            raise RuntimeError("cube did not retain table support at Pack 4 reset")
        if table_contact_substeps < int(
            self.tg["min_table_contact_substeps"]
        ):
            raise RuntimeError(
                "cube did not establish sustained Pack 4 table support: "
                f"substeps={table_contact_substeps}"
            )

        cube_half = float(self.cg["cube_size_m"]) / 2.0
        table_height = float(self.tg["table_size_m"][2])
        expected_cube_z = (
            float(self.model.body_pos[self.table_bid][2])
            + table_height / 2.0
            + cube_half
        )
        rest_height_error = abs(
            float(self.data.xpos[self.cube_bid][2]) - expected_cube_z
        )
        rest_speed = float(
            np.linalg.norm(
                self.data.qvel[self.cube_dadr : self.cube_dadr + 3]
            )
        )
        if rest_height_error > float(self.tg["max_rest_height_error_m"]):
            raise RuntimeError(
                "Pack 4 cube rest-height error exceeded Pack 2 gate: "
                f"{rest_height_error:.6f} m"
            )
        if rest_speed > float(self.tg["max_rest_speed_m_s"]):
            raise RuntimeError(
                "Pack 4 cube did not settle below the Pack 2 speed gate: "
                f"{rest_speed:.6f} m/s"
            )
        if max_penetration > float(self.tg["max_contact_penetration_m"]):
            raise RuntimeError("cube/table penetration exceeded Pack 2 reset gate")

        self.reset_diagnostics = {
            "table_contact_substeps": int(table_contact_substeps),
            "rest_height_error_m": float(rest_height_error),
            "rest_speed_m_s": float(rest_speed),
            "max_cube_table_penetration_m": float(max_penetration),
        }

        self.cube_start = self.data.xpos[self.cube_bid].copy()
        self.interface = BoundedPushActionInterface(
            self.model,
            self.data,
            site_name=str(self.mc["control_site_name"]),
            segment_origin_world_m=self.axis_origin,
            segment_direction_world=self.axis_direction,
            fixed_tcp_rotation_world=self.axis_rotation,
            segment_length_m=float(self.cg["push_distance_m"]),
            initial_segment_m=0.0,
            action_config=self.ac,
            model_config=self.mc,
        )
        self.episode_steps = 0
        self._episode_done = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options not in (None, {}):
            raise ValueError("Pack 4 v0 has no reset randomization options")
        self._reset_task()
        contacts = self._contacts()
        info = self._base_info(contacts)
        info.update(
            {
                "reset_is_deterministic": True,
                "reset_seed_used_for_randomization": False,
                "reset_diagnostics": dict(self.reset_diagnostics),
            }
        )
        return self._observation(), info

    def _rejected_step(
        self, decision: ActionDecision
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.episode_steps += 1
        contacts = self._contacts()
        reward = float(self.p4["invalid_action_penalty"])
        truncated = self.episode_steps >= self.max_episode_steps
        info = self._base_info(contacts)
        info.update(
            {
                "action_accepted": False,
                "action_code": decision.code,
                "action_detail": decision.detail,
                "simulation_substeps": 0,
                "tcp_tracking_error_m": 0.0,
                "safety_fault": None,
                "reward_terms": {
                    "tcp_progress": 0.0,
                    "progress": 0.0,
                    "success_bonus": 0.0,
                    "invalid_action_penalty": reward,
                    "fault_penalty": 0.0,
                },
                "time_limit_reached": bool(truncated),
            }
        )
        self._episode_done = bool(truncated)
        return self._observation(), reward, False, truncated, info

    def step(
        self, action: Sequence[float]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("episode is done; call reset() before step()")
        if self.interface is None:
            raise RuntimeError("Pack 4 task interface is not initialized")

        mujoco.mj_forward(self.model, self.data)
        decision = self.interface.plan_normalized(action)
        if not decision.accepted:
            return self._rejected_step(decision)

        progress_before = self._cube_progress_m()
        tcp_segment_before = float(
            np.dot(
                self.data.site_xpos[self.tcp_sid] - self.axis_origin,
                self.axis_direction,
            )
        )
        self.interface.commit(decision)
        assert decision.q_target_rad is not None
        assert decision.gripper_command is not None
        assert decision.target_tcp_world_m is not None
        self.q_target = decision.q_target_rad.copy()
        self.gripper_command = float(decision.gripper_command)

        min_substeps = int(
            math.ceil(
                float(self.mc["control_timestep_s"]) / self.model.opt.timestep
            )
        )
        max_substeps = int(
            math.ceil(
                float(self.cg["waypoint_timeout_s"]) / self.model.opt.timestep
            )
        )
        required_streak = int(self.cg["waypoint_settle_substeps"])
        tolerance = float(self.cg["waypoint_tracking_tolerance_m"])
        streak = 0
        tracking_error = math.inf
        fault: str | None = None
        reached = False
        contact_peak = {"gripper_cube": 0.0, "cube_table": 0.0}
        substeps = 0

        for substep in range(1, max_substeps + 1):
            self._hold()
            mujoco.mj_step(self.model, self.data)
            substeps = substep
            contacts = self._contacts()
            fault = self._runtime_fault(contacts)

            snapshot = self.telemetry.snapshot("pack4_step")
            for contact in snapshot["contacts"]:
                pair = str(contact["pair_type"])
                contact_peak[pair] = max(
                    contact_peak[pair], float(contact["force_norm_N"])
                )
            if fault is not None:
                break

            tracking_error = float(
                np.linalg.norm(
                    self.data.site_xpos[self.tcp_sid]
                    - decision.target_tcp_world_m
                )
            )
            if tracking_error <= tolerance:
                streak += 1
            else:
                streak = 0
            if substep >= min_substeps and streak >= required_streak:
                reached = True
                break

        if fault is None and not reached:
            fault = "TCP_ACTION_TIMEOUT"
        self.episode_steps += 1

        contacts = self._contacts()
        progress_after = self._cube_progress_m()
        tcp_segment_after = float(
            np.dot(
                self.data.site_xpos[self.tcp_sid] - self.axis_origin,
                self.axis_direction,
            )
        )
        tcp_progress_reward = float(self.p4["tcp_progress_weight"]) * (
            tcp_segment_after - tcp_segment_before
        ) / float(self.cg["push_distance_m"])
        progress_reward = (
            progress_after - progress_before
        ) / self.goal_displacement_m
        success = bool(
            fault is None
            and progress_after >= self.goal_displacement_m
            and contacts.cube_table
        )
        success_bonus = float(self.p4["success_bonus"]) if success else 0.0
        fault_penalty = float(self.p4["fault_penalty"]) if fault else 0.0
        reward = float(
            tcp_progress_reward
            + progress_reward
            + success_bonus
            + fault_penalty
        )
        terminated = success
        time_limit = self.episode_steps >= self.max_episode_steps
        truncated = bool(fault is not None or (time_limit and not success))
        self._episode_done = bool(terminated or truncated)

        info = self._base_info(contacts)
        info.update(
            {
                "action_accepted": True,
                "action_code": decision.code,
                "action_detail": decision.detail,
                "normalized_action": list(decision.normalized_action or ()),
                "simulation_substeps": int(substeps),
                "tcp_tracking_error_m": float(tracking_error),
                "safety_fault": fault,
                "gripper_cube_peak_force_norm_N": float(
                    contact_peak["gripper_cube"]
                ),
                "cube_table_peak_force_norm_N": float(
                    contact_peak["cube_table"]
                ),
                "reward_terms": {
                    "tcp_progress": float(tcp_progress_reward),
                    "progress": float(progress_reward),
                    "success_bonus": float(success_bonus),
                    "invalid_action_penalty": 0.0,
                    "fault_penalty": float(fault_penalty),
                },
                "time_limit_reached": bool(time_limit),
                "is_success": success,
            }
        )
        return self._observation(), reward, terminated, truncated, info

    def state_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return copies for verification that rejected actions are read-only."""

        return (
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.ctrl.copy(),
        )

    def contract(self) -> dict[str, Any]:
        if self.interface is None:
            raise RuntimeError("call reset() before reading the Pack 4 contract")
        return {
            "environment_id": str(self.p4["environment_id"]),
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "observation_components": list(OBSERVATION_COMPONENTS),
            "observation_shape": list(self.observation_space.shape),
            "observation_units": "SI values plus one normalized command and two binary contacts",
            "action": self.interface.contract(),
            "cube_goal_displacement_m": self.goal_displacement_m,
            "max_episode_steps": self.max_episode_steps,
            "reward": {
                "tcp_progress": (
                    f"{float(self.p4['tcp_progress_weight'])} * delta actual "
                    "TCP segment / full push segment"
                ),
                "progress": "delta cube progress / cube goal displacement",
                "success_bonus": float(self.p4["success_bonus"]),
                "invalid_action_penalty": float(
                    self.p4["invalid_action_penalty"]
                ),
                "fault_penalty": float(self.p4["fault_penalty"]),
                "force_used_in_reward": False,
            },
            "reset_rest_gate": dict(self.reset_diagnostics),
            "reset_randomization": False,
            "image_observation": False,
            "hardware_interface": False,
        }

    def close(self) -> None:
        self._episode_done = True
