from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import mujoco
import numpy as np

from .joint_layout import JointLayout
from .mujoco_ik import DampedLeastSquaresIK


PLANAR_ACTION_SCHEMA_VERSION = "cr10_2f85_pack5_bounded_planar_action_v1"


@dataclass(frozen=True)
class PlanarActionDecision:
    accepted: bool
    code: str
    detail: str
    normalized_action: tuple[float, float] | None
    start_planar_m: np.ndarray
    target_planar_m: np.ndarray | None
    delta_planar_m: np.ndarray | None
    target_tcp_world_m: np.ndarray | None
    q_target_rad: np.ndarray | None
    gripper_command: float | None
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
            "start_planar_m": self.start_planar_m.tolist(),
            "target_planar_m": (
                self.target_planar_m.tolist()
                if self.target_planar_m is not None
                else None
            ),
            "delta_planar_m": (
                self.delta_planar_m.tolist()
                if self.delta_planar_m is not None
                else None
            ),
            "target_tcp_world_m": (
                self.target_tcp_world_m.tolist()
                if self.target_tcp_world_m is not None
                else None
            ),
            "q_target_rad": (
                self.q_target_rad.tolist() if self.q_target_rad is not None else None
            ),
            "gripper_command": self.gripper_command,
            "ik_position_error_m": self.ik_position_error_m,
            "ik_rotation_error_rad": self.ik_rotation_error_rad,
            "scaled_jacobian_sigma_min": self.scaled_jacobian_sigma_min,
            "max_joint_target_delta_rad": self.max_joint_target_delta_rad,
        }


class BoundedPlanarActionInterface:
    """Read-only admission planning for fixed-height planar TCP increments."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        site_name: str,
        origin_world_m: Sequence[float],
        forward_axis_world: Sequence[float],
        lateral_axis_world: Sequence[float],
        fixed_tcp_rotation_world: np.ndarray,
        forward_bounds_m: Sequence[float],
        lateral_bounds_m: Sequence[float],
        fixed_gripper_command: float,
        action_config: dict,
        model_config: dict,
    ):
        self.model = model
        self.data = data
        self.layout = JointLayout.from_model(model)
        self.site_name = str(site_name)
        self.origin = np.asarray(origin_world_m, dtype=float).reshape(3)
        self.forward = self._unit(forward_axis_world, "forward")
        self.lateral = self._unit(lateral_axis_world, "lateral")
        if abs(float(np.dot(self.forward, self.lateral))) > 1e-6:
            raise ValueError("planar forward/lateral axes are not orthogonal")
        self.rotation = np.asarray(fixed_tcp_rotation_world, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(self.origin)) or not np.all(
            np.isfinite(self.rotation)
        ):
            raise ValueError("planar action frame must be finite")
        if (
            np.linalg.norm(self.rotation.T @ self.rotation - np.eye(3)) > 1e-6
            or np.linalg.det(self.rotation) <= 0.0
        ):
            raise ValueError("fixed TCP rotation is not a proper rotation")

        self.forward_bounds = self._bounds(forward_bounds_m, "forward")
        self.lateral_bounds = self._bounds(lateral_bounds_m, "lateral")
        if not self.forward_bounds[0] <= 0.0 <= self.forward_bounds[1]:
            raise ValueError("initial forward coordinate is outside bounds")
        if not self.lateral_bounds[0] <= 0.0 <= self.lateral_bounds[1]:
            raise ValueError("initial lateral coordinate is outside bounds")
        self.planar_m = np.zeros(2, dtype=float)

        self.max_step_m = float(action_config["max_xy_step_m"])
        self.joint_margin_rad = math.radians(float(action_config["joint_margin_deg"]))
        self.speed_caps_rad_s = np.asarray(
            action_config["software_joint_speed_cap_rad_s"], dtype=float
        ).reshape(6)
        self.control_timestep_s = float(model_config["control_timestep_s"])
        self.max_target_delta_rad = self.speed_caps_rad_s * self.control_timestep_s
        self.min_sigma = float(action_config["min_scaled_jacobian_sigma"])
        self.max_ik_pos_error_m = float(action_config["max_ik_pos_error_m"])
        self.max_ik_rot_error_rad = float(action_config["max_ik_rot_error_rad"])
        if self.max_step_m <= 0.0 or self.control_timestep_s <= 0.0:
            raise ValueError("planar action bounds must be positive")

        self.gripper_aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator"
        )
        if self.gripper_aid < 0:
            raise ValueError("required gripper actuator missing")
        gripper_low, gripper_high = map(
            float, model.actuator_ctrlrange[self.gripper_aid]
        )
        self.fixed_gripper_command = float(fixed_gripper_command)
        if not gripper_low <= self.fixed_gripper_command <= gripper_high:
            raise ValueError("fixed gripper command leaves actuator range")

        self.ik = DampedLeastSquaresIK(
            model,
            site_name=self.site_name,
            rot_scale=float(model_config["rotation_characteristic_length_m"]),
            damping=float(model_config["ik_damping"]),
            max_iterations=int(model_config["ik_max_iterations"]),
            max_joint_step=float(model_config["ik_max_joint_step_rad"]),
        )

    @staticmethod
    def _unit(values: Sequence[float], label: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vector))
        if not np.all(np.isfinite(vector)) or norm <= 0.0:
            raise ValueError(f"{label} axis must be finite and nonzero")
        return vector / norm

    @staticmethod
    def _bounds(values: Sequence[float], label: str) -> np.ndarray:
        bounds = np.asarray(values, dtype=float).reshape(2)
        if not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError(f"invalid {label} planar bounds")
        return bounds

    def _reject(
        self,
        code: str,
        detail: str,
        normalized_action: tuple[float, float] | None = None,
        **kwargs,
    ) -> PlanarActionDecision:
        return PlanarActionDecision(
            accepted=False,
            code=code,
            detail=detail,
            normalized_action=normalized_action,
            start_planar_m=self.planar_m.copy(),
            target_planar_m=kwargs.get("target_planar_m"),
            delta_planar_m=kwargs.get("delta_planar_m"),
            target_tcp_world_m=None,
            q_target_rad=None,
            gripper_command=self.fixed_gripper_command,
            ik_position_error_m=kwargs.get("ik_position_error_m"),
            ik_rotation_error_rad=kwargs.get("ik_rotation_error_rad"),
            scaled_jacobian_sigma_min=kwargs.get("scaled_sigma"),
            max_joint_target_delta_rad=kwargs.get("max_joint_delta"),
        )

    def plan_normalized(self, action: Sequence[float]) -> PlanarActionDecision:
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
                "both normalized planar components must lie in [-1, 1]",
                action_tuple,
            )

        action_norm = float(np.linalg.norm(values))
        mapped = values / max(1.0, action_norm)
        delta = mapped * self.max_step_m
        proposed = self.planar_m + delta
        if not (
            self.forward_bounds[0] - 1e-12
            <= proposed[0]
            <= self.forward_bounds[1] + 1e-12
            and self.lateral_bounds[0] - 1e-12
            <= proposed[1]
            <= self.lateral_bounds[1] + 1e-12
        ):
            return self._reject(
                "PLANAR_BOUND",
                "target leaves the configured forward/lateral rectangle",
                action_tuple,
                target_planar_m=proposed.copy(),
                delta_planar_m=delta.copy(),
            )

        target_tcp = (
            self.origin + proposed[0] * self.forward + proposed[1] * self.lateral
        )
        seed_q = self.layout.get_q(self.data).copy()
        ik_result = self.ik.solve(
            target_tcp,
            self.rotation,
            seed_q,
            pos_tol=self.max_ik_pos_error_m,
            rot_tol=self.max_ik_rot_error_rad,
        )
        common = {
            "target_planar_m": proposed.copy(),
            "delta_planar_m": delta.copy(),
            "ik_position_error_m": float(ik_result.pos_error_m),
            "ik_rotation_error_rad": float(ik_result.rot_error_rad),
            "scaled_sigma": float(ik_result.scaled_sigma_min),
        }
        if not ik_result.success:
            return self._reject(
                "IK_FAILURE",
                "DLS IK did not satisfy configured residual bounds",
                action_tuple,
                **common,
            )
        if float(ik_result.scaled_sigma_min) < self.min_sigma:
            return self._reject(
                "JACOBIAN_CONDITION",
                "scaled Jacobian singular value is below the gate",
                action_tuple,
                **common,
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
                    f"{self.layout.joint_names[index]} enters joint margin",
                    action_tuple,
                    **common,
                )

        joint_delta = np.abs(ik_result.q - seed_q)
        max_joint_delta = float(np.max(joint_delta))
        violating = np.flatnonzero(joint_delta > self.max_target_delta_rad + 1e-12)
        if violating.size:
            index = int(violating[0])
            return self._reject(
                "JOINT_TARGET_DELTA",
                f"{self.layout.joint_names[index]} exceeds per-command speed cap",
                action_tuple,
                max_joint_delta=max_joint_delta,
                **common,
            )

        return PlanarActionDecision(
            accepted=True,
            code="ACCEPTED",
            detail="bounded planar simulation action admitted",
            normalized_action=action_tuple,
            start_planar_m=self.planar_m.copy(),
            target_planar_m=proposed.copy(),
            delta_planar_m=delta.copy(),
            target_tcp_world_m=target_tcp.copy(),
            q_target_rad=ik_result.q.copy(),
            gripper_command=self.fixed_gripper_command,
            ik_position_error_m=float(ik_result.pos_error_m),
            ik_rotation_error_rad=float(ik_result.rot_error_rad),
            scaled_jacobian_sigma_min=float(ik_result.scaled_sigma_min),
            max_joint_target_delta_rad=max_joint_delta,
        )

    def commit(self, decision: PlanarActionDecision) -> None:
        if not decision.accepted or decision.target_planar_m is None:
            raise ValueError("cannot commit a rejected planar action")
        if not np.array_equal(decision.start_planar_m, self.planar_m):
            raise ValueError("cannot commit a stale planar decision")
        self.planar_m = decision.target_planar_m.copy()

    def contract(self) -> dict:
        return {
            "schema_version": PLANAR_ACTION_SCHEMA_VERSION,
            "normalized_action": {
                "shape": [2],
                "bounds": [-1.0, 1.0],
                "components": ["tcp_forward_increment", "tcp_lateral_increment"],
            },
            "mapping": (
                "a is radially saturated to the unit disk, then multiplied "
                "by max_step_m"
            ),
            "maximum_euclidean_increment_m": self.max_step_m,
            "forward_bounds_m": self.forward_bounds.tolist(),
            "lateral_bounds_m": self.lateral_bounds.tolist(),
            "fixed_height": True,
            "fixed_orientation": True,
            "fixed_gripper_command": self.fixed_gripper_command,
            "admission_gates": [
                "shape, finite values, and normalized bounds",
                "planar task rectangle",
                "DLS IK residuals",
                "scaled Jacobian minimum singular value",
                "joint-limit margin",
                "per-command joint-target speed cap",
            ],
            "claim_boundary": (
                "Simulation-only action admission; not a physical collision, "
                "velocity, force, or hardware safety certificate."
            ),
        }
