from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np


SCHEMA_VERSION = "cr10_2f85_pack3_sim_telemetry_v1"
PAIR_TYPES = ("gripper_cube", "cube_table")


def _id(model: mujoco.MjModel, obj, name: str) -> int:
    value = mujoco.mj_name2id(model, obj, name)
    if value < 0:
        raise ValueError(f"required model element missing: {name}")
    return value


def _descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    body = int(body_id)
    while body > 0:
        if body == ancestor_id:
            return True
        body = int(model.body_parentid[body])
    return body == ancestor_id


def _name(model: mujoco.MjModel, obj, value: int) -> str:
    return mujoco.mj_id2name(model, obj, int(value)) or f"id_{int(value)}"


@dataclass(frozen=True)
class TelemetryIds:
    gripper_actuator: int
    left_driver_dof: int
    right_driver_dof: int
    gripper_root_body: int
    cube_geom: int
    table_geom: int


class SimulationTelemetry:
    """Read MuJoCo solver telemetry without modifying simulation state."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        left_jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
        right_jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
        self.ids = TelemetryIds(
            gripper_actuator=_id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_actuator"
            ),
            left_driver_dof=int(model.jnt_dofadr[left_jid]),
            right_driver_dof=int(model.jnt_dofadr[right_jid]),
            gripper_root_body=_id(
                model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_mount_frame"
            ),
            cube_geom=_id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "known_push_cube_geom"
            ),
            table_geom=_id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "pack2_table_geom"
            ),
        )

    def _pair_type(self, geom1: int, geom2: int) -> str | None:
        pair = {int(geom1), int(geom2)}
        if pair == {self.ids.cube_geom, self.ids.table_geom}:
            return "cube_table"
        if self.ids.cube_geom not in pair:
            return None
        other = int(geom2) if int(geom1) == self.ids.cube_geom else int(geom1)
        if _descendant(
            self.model,
            int(self.model.geom_bodyid[other]),
            self.ids.gripper_root_body,
        ):
            return "gripper_cube"
        return None

    def snapshot(self, phase: str = "unspecified") -> dict:
        contacts = []
        for contact_id in range(int(self.data.ncon)):
            contact = self.data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            pair_type = self._pair_type(geom1, geom2)
            if pair_type is None:
                continue

            wrench_contact = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(
                self.model, self.data, contact_id, wrench_contact
            )
            frame_axes_world = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            force_world = frame_axes_world.T @ wrench_contact[:3]
            torque_world = frame_axes_world.T @ wrench_contact[3:]
            contacts.append(
                {
                    "contact_id": contact_id,
                    "pair_type": pair_type,
                    "geom1": _name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1
                    ),
                    "geom2": _name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2
                    ),
                    "position_world_m": [float(x) for x in contact.pos],
                    "distance_m": float(contact.dist),
                    "contact_frame_axes_world_rows": frame_axes_world.tolist(),
                    "force_contact_frame_N": wrench_contact[:3].tolist(),
                    "torque_contact_frame_Nm": wrench_contact[3:].tolist(),
                    "force_world_N": force_world.tolist(),
                    "torque_world_Nm": torque_world.tolist(),
                    "normal_force_abs_N": abs(float(wrench_contact[0])),
                    "tangential_force_norm_N": float(
                        np.linalg.norm(wrench_contact[1:3])
                    ),
                    "force_norm_N": float(np.linalg.norm(wrench_contact[:3])),
                    "torque_norm_Nm": float(np.linalg.norm(wrench_contact[3:])),
                }
            )

        aid = self.ids.gripper_actuator
        actuator_scalar = float(self.data.actuator_force[aid])
        driver_torques = {
            "left_driver_joint": float(
                self.data.qfrc_actuator[self.ids.left_driver_dof]
            ),
            "right_driver_joint": float(
                self.data.qfrc_actuator[self.ids.right_driver_dof]
            ),
        }
        finite_values = [actuator_scalar, *driver_torques.values()]
        for contact in contacts:
            finite_values.extend(contact["force_contact_frame_N"])
            finite_values.extend(contact["torque_contact_frame_Nm"])
            finite_values.extend(contact["force_world_N"])
            finite_values.extend(contact["torque_world_Nm"])

        return {
            "schema_version": SCHEMA_VERSION,
            "simulation_time_s": float(self.data.time),
            "phase": str(phase),
            "all_values_finite": bool(np.all(np.isfinite(finite_values))),
            "gripper": {
                "command_0_to_255": float(self.data.ctrl[aid]),
                "actuator_force_scalar_model_units": actuator_scalar,
                "driver_joint_qfrc_actuator_Nm": driver_torques,
            },
            "contacts": contacts,
        }


class TelemetryAccumulator:
    """Accumulate telemetry statistics."""

    def __init__(self, reader: SimulationTelemetry):
        self.reader = reader
        self.sample_count = 0
        self.all_finite = True
        self.gripper_actuator_peak_abs = 0.0
        self.driver_peak_abs = {
            "left_driver_joint": 0.0,
            "right_driver_joint": 0.0,
        }
        self.pair_stats = {
            pair: {
                "contact_points": 0,
                "contact_substeps": 0,
                "peak_normal_force_abs_N": 0.0,
                "peak_tangential_force_norm_N": 0.0,
                "peak_force_norm_N": 0.0,
                "peak_torque_norm_Nm": 0.0,
                "phase_contact_substeps": {},
            }
            for pair in PAIR_TYPES
        }

    def observe(self, phase: str) -> dict:
        snapshot = self.reader.snapshot(phase)
        self.sample_count += 1
        self.all_finite = self.all_finite and bool(snapshot["all_values_finite"])
        gripper = snapshot["gripper"]
        self.gripper_actuator_peak_abs = max(
            self.gripper_actuator_peak_abs,
            abs(float(gripper["actuator_force_scalar_model_units"])),
        )
        for joint, value in gripper["driver_joint_qfrc_actuator_Nm"].items():
            self.driver_peak_abs[joint] = max(
                self.driver_peak_abs[joint], abs(float(value))
            )

        seen_pairs: set[str] = set()
        for contact in snapshot["contacts"]:
            pair = str(contact["pair_type"])
            stats = self.pair_stats[pair]
            stats["contact_points"] += 1
            stats["peak_normal_force_abs_N"] = max(
                stats["peak_normal_force_abs_N"],
                float(contact["normal_force_abs_N"]),
            )
            stats["peak_tangential_force_norm_N"] = max(
                stats["peak_tangential_force_norm_N"],
                float(contact["tangential_force_norm_N"]),
            )
            stats["peak_force_norm_N"] = max(
                stats["peak_force_norm_N"], float(contact["force_norm_N"])
            )
            stats["peak_torque_norm_Nm"] = max(
                stats["peak_torque_norm_Nm"], float(contact["torque_norm_Nm"])
            )
            seen_pairs.add(pair)
        for pair in seen_pairs:
            stats = self.pair_stats[pair]
            stats["contact_substeps"] += 1
            phases = stats["phase_contact_substeps"]
            phases[phase] = int(phases.get(phase, 0)) + 1
        return snapshot

    def summary(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "read_only": True,
            "sample_count": int(self.sample_count),
            "all_values_finite": bool(self.all_finite),
            "gripper": {
                "actuator_force_source": "MjData.actuator_force[gripper_actuator]",
                "actuator_force_scalar_units": (
                    "actuator-space model generalized-force units; not calibrated "
                    "fingertip newtons"
                ),
                "actuator_force_scalar_peak_abs": float(
                    self.gripper_actuator_peak_abs
                ),
                "driver_joint_source": "MjData.qfrc_actuator at hinge DOFs",
                "driver_joint_units": "N*m generalized actuator torque in this SI model",
                "driver_joint_peak_abs_Nm": {
                    key: float(value) for key, value in self.driver_peak_abs.items()
                },
            },
            "contact": {
                "source": "mujoco.mj_contactForce",
                "contact_force_units": "N",
                "contact_torque_units": "N*m about the contact point",
                "frame_contract": (
                    "raw wrench is resolved in the MuJoCo contact frame for the "
                    "recorded ordered geom1/geom2 pair; world vector = frame.T @ "
                    "contact-frame vector"
                ),
                "pair_statistics": self.pair_stats,
            },
            "claim_boundary": (
                "MuJoCo solver telemetry only. Values are not calibrated physical "
                "2F-85 force/current readings, not a wrist F/T sensor, and not "
                "validated physical contact forces."
            ),
        }


def validate_telemetry_summary(summary: dict) -> dict:
    """Validate telemetry structure and signal presence."""

    if summary.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("unexpected Pack 3 telemetry schema")
    if not bool(summary.get("read_only")):
        raise AssertionError("telemetry contract is not marked read-only")
    if int(summary.get("sample_count", 0)) <= 0:
        raise AssertionError("telemetry recorder did not sample the simulation")
    if not bool(summary.get("all_values_finite")):
        raise AssertionError("non-finite value found in Pack 3 telemetry")

    pair_stats = summary["contact"]["pair_statistics"]
    for pair in PAIR_TYPES:
        stats = pair_stats[pair]
        if int(stats["contact_points"]) <= 0:
            raise AssertionError(f"no {pair} contact wrench was recorded")
        if int(stats["contact_substeps"]) <= 0:
            raise AssertionError(f"no {pair} contact substep was recorded")
        peak = float(stats["peak_force_norm_N"])
        if not math.isfinite(peak) or peak <= 0.0:
            raise AssertionError(f"{pair} contact wrench never became nonzero")

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_count": int(summary["sample_count"]),
        "gripper_actuator_force_scalar_peak_abs_model_units": float(
            summary["gripper"]["actuator_force_scalar_peak_abs"]
        ),
        "left_driver_joint_torque_peak_abs_Nm": float(
            summary["gripper"]["driver_joint_peak_abs_Nm"]["left_driver_joint"]
        ),
        "right_driver_joint_torque_peak_abs_Nm": float(
            summary["gripper"]["driver_joint_peak_abs_Nm"]["right_driver_joint"]
        ),
        "gripper_cube_peak_force_norm_N": float(
            pair_stats["gripper_cube"]["peak_force_norm_N"]
        ),
        "cube_table_peak_force_norm_N": float(
            pair_stats["cube_table"]["peak_force_norm_N"]
        ),
    }


def format_snapshot(snapshot: dict) -> list[str]:
    gripper = snapshot["gripper"]
    driver = gripper["driver_joint_qfrc_actuator_Nm"]
    lines = [
        (
            "gripper actuator scalar(model units)="
            f"{gripper['actuator_force_scalar_model_units']:.6f}"
        ),
        (
            "driver actuator torque(N*m): "
            f"left={driver['left_driver_joint']:.6f} "
            f"right={driver['right_driver_joint']:.6f}"
        ),
    ]
    if not snapshot["contacts"]:
        lines.append("cube contact wrenches: none at this simulation step")
        return lines
    for contact in snapshot["contacts"]:
        lines.append(
            f"{contact['pair_type']} {contact['geom1']} <-> {contact['geom2']}: "
            f"normal_abs={contact['normal_force_abs_N']:.6f} N "
            f"tangent_norm={contact['tangential_force_norm_N']:.6f} N "
            f"world_force={np.round(contact['force_world_N'], 6).tolist()} N"
        )
    return lines
