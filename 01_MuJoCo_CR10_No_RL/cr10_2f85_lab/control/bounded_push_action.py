from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import mujoco
import numpy as np

from .joint_layout import JointLayout
from .mujoco_ik import DampedLeastSquaresIK


ACTION_SCHEMA_VERSION = "cr10_2f85_pack3_bounded_push_action_v1"


@dataclass(frozen=True)
class ActionDecision:
    accepted: bool
    code: str
    detail: str
    normalized_action: tuple[float, float] | None
    start_segment_m: float
    target_segment_m: float | None
    delta_push_m: float | None
    gripper_command: float | None
    target_tcp_world_m: np.ndarray | None
    q_target_rad: np.ndarray | None
    ik_position_error_m: float | None
    ik_rotation_error_rad: float | None
    scaled_jacobian_sigma_min: float | None
    max_joint_target_delta_rad: float | None

    def as_dict(self) -> dict:
        return {
            "accepted": bool(self.accepted),
            "code": self.code,
            "detail": self.detail,
            "normalized_action": (
                list(self.normalized_action)
                if self.normalized_action is not None
                else None
            ),
            "start_segment_m": float(self.start_segment_m),
            "target_segment_m": (
                float(self.target_segment_m)
                if self.target_segment_m is not None
                else None
            ),
            "delta_push_m": (
                float(self.delta_push_m) if self.delta_push_m is not None else None
            ),
            "gripper_command": (
                float(self.gripper_command)
                if self.gripper_command is not None
                else None
            ),
            "target_tcp_world_m": (
                self.target_tcp_world_m.tolist()
                if self.target_tcp_world_m is not None
                else None
            ),
            "q_target_rad": (
                self.q_target_rad.tolist()
                if self.q_target_rad is not None
                else None
            ),
            "ik_position_error_m": self.ik_position_error_m,
            "ik_rotation_error_rad": self.ik_rotation_error_rad,
            "scaled_jacobian_sigma_min": self.scaled_jacobian_sigma_min,
            "max_joint_target_delta_rad": self.max_joint_target_delta_rad,
        }


class BoundedPushActionInterface:
    """Admission gate for bounded push and gripper commands."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        site_name: str,
        segment_origin_world_m: Sequence[float],
        segment_direction_world: Sequence[float],
        fixed_tcp_rotation_world: np.ndarray,
        segment_length_m: float,
        initial_segment_m: float,
        action_config: dict,
        model_config: dict,
    ):
        self.model = model
        self.data = data
        self.layout = JointLayout.from_model(model)
        self.site_name = str(site_name)
        self.origin = np.asarray(segment_origin_world_m, dtype=float).reshape(3)
        direction = np.asarray(segment_direction_world, dtype=float).reshape(3)
        direction_norm = float(np.linalg.norm(direction))
        if not math.isfinite(direction_norm) or direction_norm <= 0.0:
            raise ValueError("push segment direction must be finite and nonzero")
        self.direction = direction / direction_norm
        self.rotation = np.asarray(
            fixed_tcp_rotation_world, dtype=float
        ).reshape(3, 3)
        if not np.all(np.isfinite(self.origin)) or not np.all(
            np.isfinite(self.rotation)
        ):
            raise ValueError("push segment frame must be finite")
        orthogonality_error = float(
            np.linalg.norm(self.rotation.T @ self.rotation - np.eye(3))
        )
        if orthogonality_error > 1e-6 or np.linalg.det(self.rotation) <= 0.0:
            raise ValueError("fixed TCP rotation is not a proper rotation matrix")

        self.segment_length_m = float(segment_length_m)
        self.segment_m = float(initial_segment_m)
        if (
            not math.isfinite(self.segment_length_m)
            or self.segment_length_m <= 0.0
            or not 0.0 <= self.segment_m <= self.segment_length_m
        ):
            raise ValueError("invalid push segment length or initial position")

        self.max_step_m = float(action_config["max_xy_step_m"])
        self.joint_margin_rad = math.radians(
            float(action_config["joint_margin_deg"])
        )
        self.speed_caps_rad_s = np.asarray(
            action_config["software_joint_speed_cap_rad_s"], dtype=float
        ).reshape(6)
        self.control_timestep_s = float(model_config["control_timestep_s"])
        self.max_target_delta_rad = (
            self.speed_caps_rad_s * self.control_timestep_s
        )
        self.min_sigma = float(action_config["min_scaled_jacobian_sigma"])
        self.max_ik_pos_error_m = float(action_config["max_ik_pos_error_m"])
        self.max_ik_rot_error_rad = float(action_config["max_ik_rot_error_rad"])
        if (
            self.max_step_m <= 0.0
            or self.control_timestep_s <= 0.0
            or np.any(self.speed_caps_rad_s <= 0.0)
        ):
            raise ValueError("Pack 3 action bounds must be positive")

        self.gripper_aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator"
        )
        if self.gripper_aid < 0:
            raise ValueError("required gripper actuator missing")
        if not bool(model.actuator_ctrllimited[self.gripper_aid]):
            raise ValueError("gripper actuator does not have a finite control range")
        self.gripper_low, self.gripper_high = map(
            float, model.actuator_ctrlrange[self.gripper_aid]
        )
        if not self.gripper_low < self.gripper_high:
            raise ValueError("invalid gripper actuator control range")

        self.ik = DampedLeastSquaresIK(
            model,
            site_name=self.site_name,
            rot_scale=float(model_config["rotation_characteristic_length_m"]),
            damping=float(model_config["ik_damping"]),
            max_iterations=int(model_config["ik_max_iterations"]),
            max_joint_step=float(model_config["ik_max_joint_step_rad"]),
        )

    def _reject(
        self,
        code: str,
        detail: str,
        normalized_action: tuple[float, float] | None = None,
        *,
        target_segment_m: float | None = None,
        delta_push_m: float | None = None,
        gripper_command: float | None = None,
        ik_position_error_m: float | None = None,
        ik_rotation_error_rad: float | None = None,
        scaled_sigma: float | None = None,
        max_joint_delta: float | None = None,
    ) -> ActionDecision:
        return ActionDecision(
            accepted=False,
            code=code,
            detail=detail,
            normalized_action=normalized_action,
            start_segment_m=float(self.segment_m),
            target_segment_m=target_segment_m,
            delta_push_m=delta_push_m,
            gripper_command=gripper_command,
            target_tcp_world_m=None,
            q_target_rad=None,
            ik_position_error_m=ik_position_error_m,
            ik_rotation_error_rad=ik_rotation_error_rad,
            scaled_jacobian_sigma_min=scaled_sigma,
            max_joint_target_delta_rad=max_joint_delta,
        )

    def plan_normalized(self, action: Sequence[float]) -> ActionDecision:
        try:
            values = np.asarray(action, dtype=float)
        except (TypeError, ValueError):
            return self._reject("INVALID_ACTION_TYPE", "action must be numeric")
        if values.shape != (2,):
            return self._reject(
                "INVALID_ACTION_SHAPE",
                f"expected shape (2,), received {values.shape}",
            )
        if not np.all(np.isfinite(values)):
            return self._reject("NONFINITE_ACTION", "action contains NaN or inf")
        action_tuple = (float(values[0]), float(values[1]))
        if np.any(values < -1.0) or np.any(values > 1.0):
            return self._reject(
                "ACTION_OUT_OF_RANGE",
                "both normalized action components must lie in [-1, 1]",
                action_tuple,
            )

        delta_push_m = float(values[0]) * self.max_step_m
        gripper_command = self.gripper_low + 0.5 * (float(values[1]) + 1.0) * (
            self.gripper_high - self.gripper_low
        )
        proposed = self.segment_m + delta_push_m
        eps = 1e-12
        if proposed < -eps or proposed > self.segment_length_m + eps:
            return self._reject(
                "SEGMENT_BOUND",
                (
                    f"target {proposed:.9f} m leaves [0, "
                    f"{self.segment_length_m:.9f}] m"
                ),
                action_tuple,
                target_segment_m=float(proposed),
                delta_push_m=delta_push_m,
                gripper_command=gripper_command,
            )
        proposed = min(max(proposed, 0.0), self.segment_length_m)
        target_tcp = self.origin + proposed * self.direction
        seed_q = self.layout.get_q(self.data).copy()
        ik_result = self.ik.solve(
            target_tcp,
            self.rotation,
            seed_q,
            pos_tol=self.max_ik_pos_error_m,
            rot_tol=self.max_ik_rot_error_rad,
        )
        if not ik_result.success:
            return self._reject(
                "IK_FAILURE",
                "DLS IK did not satisfy configured position/orientation errors",
                action_tuple,
                target_segment_m=proposed,
                delta_push_m=delta_push_m,
                gripper_command=gripper_command,
                ik_position_error_m=float(ik_result.pos_error_m),
                ik_rotation_error_rad=float(ik_result.rot_error_rad),
                scaled_sigma=float(ik_result.scaled_sigma_min),
            )
        if float(ik_result.scaled_sigma_min) < self.min_sigma:
            return self._reject(
                "JACOBIAN_CONDITION",
                "scaled minimum Jacobian singular value is below the configured gate",
                action_tuple,
                target_segment_m=proposed,
                delta_push_m=delta_push_m,
                gripper_command=gripper_command,
                ik_position_error_m=float(ik_result.pos_error_m),
                ik_rotation_error_rad=float(ik_result.rot_error_rad),
                scaled_sigma=float(ik_result.scaled_sigma_min),
            )

        for index, joint_id in enumerate(self.layout.joint_ids):
            if not bool(self.model.jnt_limited[joint_id]):
                continue
            lower, upper = map(float, self.model.jnt_range[joint_id])
            if not (
                lower + self.joint_margin_rad
                <= float(ik_result.q[index])
                <= upper - self.joint_margin_rad
            ):
                return self._reject(
                    "JOINT_MARGIN",
                    f"{self.layout.joint_names[index]} enters configured margin",
                    action_tuple,
                    target_segment_m=proposed,
                    delta_push_m=delta_push_m,
                    gripper_command=gripper_command,
                    ik_position_error_m=float(ik_result.pos_error_m),
                    ik_rotation_error_rad=float(ik_result.rot_error_rad),
                    scaled_sigma=float(ik_result.scaled_sigma_min),
                )

        joint_delta = np.abs(ik_result.q - seed_q)
        max_joint_delta = float(np.max(joint_delta))
        violating = np.flatnonzero(joint_delta > self.max_target_delta_rad + 1e-12)
        if violating.size:
            index = int(violating[0])
            return self._reject(
                "JOINT_TARGET_DELTA",
                (
                    f"{self.layout.joint_names[index]} target delta "
                    f"{joint_delta[index]:.6f} rad exceeds "
                    f"{self.max_target_delta_rad[index]:.6f} rad"
                ),
                action_tuple,
                target_segment_m=proposed,
                delta_push_m=delta_push_m,
                gripper_command=gripper_command,
                ik_position_error_m=float(ik_result.pos_error_m),
                ik_rotation_error_rad=float(ik_result.rot_error_rad),
                scaled_sigma=float(ik_result.scaled_sigma_min),
                max_joint_delta=max_joint_delta,
            )

        return ActionDecision(
            accepted=True,
            code="ACCEPTED",
            detail="bounded simulation task action admitted",
            normalized_action=action_tuple,
            start_segment_m=float(self.segment_m),
            target_segment_m=float(proposed),
            delta_push_m=delta_push_m,
            gripper_command=float(gripper_command),
            target_tcp_world_m=target_tcp.copy(),
            q_target_rad=ik_result.q.copy(),
            ik_position_error_m=float(ik_result.pos_error_m),
            ik_rotation_error_rad=float(ik_result.rot_error_rad),
            scaled_jacobian_sigma_min=float(ik_result.scaled_sigma_min),
            max_joint_target_delta_rad=max_joint_delta,
        )

    def commit(self, decision: ActionDecision) -> None:
        if not decision.accepted or decision.target_segment_m is None:
            raise ValueError("cannot commit a rejected action")
        if abs(float(decision.start_segment_m) - self.segment_m) > 1e-12:
            raise ValueError("cannot commit a stale action decision")
        self.segment_m = float(decision.target_segment_m)

    def gripper_command_to_normalized(self, command: float) -> float:
        command = float(command)
        if not math.isfinite(command):
            raise ValueError("gripper command must be finite")
        normalized = 2.0 * (command - self.gripper_low) / (
            self.gripper_high - self.gripper_low
        ) - 1.0
        return float(np.clip(normalized, -1.0, 1.0))

    def contract(self) -> dict:
        return {
            "schema_version": ACTION_SCHEMA_VERSION,
            "normalized_action": {
                "shape": [2],
                "bounds": [-1.0, 1.0],
                "components": [
                    "push_increment_along_validated_segment",
                    "gripper_position_command",
                ],
            },
            "push_mapping": {
                "max_abs_increment_m": self.max_step_m,
                "segment_min_m": 0.0,
                "segment_max_m": self.segment_length_m,
            },
            "gripper_mapping": {
                "normalized_open": -1.0,
                "normalized_closed": 1.0,
                "actuator_ctrlrange": [self.gripper_low, self.gripper_high],
                "meaning": "position-like closure target, not force",
            },
            "admission_gates": [
                "shape and finite values",
                "normalized bounds",
                "validated push segment",
                "DLS IK residuals",
                "scaled Jacobian minimum singular value",
                "joint-limit margin",
                "per-command joint-target delta derived from configured speed cap",
            ],
            "claim_boundary": (
                "Simulation action admission only. This does not establish real-robot "
                "workspace, collision safety, motor velocity tracking, or hardware safety."
            ),
        }
