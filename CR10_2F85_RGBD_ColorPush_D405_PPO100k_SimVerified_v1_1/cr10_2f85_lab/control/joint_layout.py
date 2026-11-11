from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mujoco

@dataclass(frozen=True)
class JointLayout:
    joint_names: tuple[str, ...]
    joint_ids: tuple[int, ...]
    qpos_adrs: tuple[int, ...]
    dof_adrs: tuple[int, ...]

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "JointLayout":
        names = tuple(f"joint{i}" for i in range(1, 7))
        jids, qadrs, dadrs = [], [], []
        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"required CR10 joint missing: {name}")
            if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(f"{name} is not a hinge joint")
            jids.append(jid)
            qadrs.append(int(model.jnt_qposadr[jid]))
            dadrs.append(int(model.jnt_dofadr[jid]))
        return cls(names, tuple(jids), tuple(qadrs), tuple(dadrs))

    def get_q(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray([data.qpos[a] for a in self.qpos_adrs], dtype=float)

    def set_q(self, data: mujoco.MjData, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float).reshape(6)
        for a, value in zip(self.qpos_adrs, q):
            data.qpos[a] = float(value)

    def set_qdot_zero(self, data: mujoco.MjData) -> None:
        for a in self.dof_adrs:
            data.qvel[a] = 0.0

    def jacobian_columns(self, full: np.ndarray) -> np.ndarray:
        return np.asarray(full, dtype=float)[:, list(self.dof_adrs)]

    def clamp(self, model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(6).copy()
        for i, jid in enumerate(self.joint_ids):
            if bool(model.jnt_limited[jid]):
                lo, hi = map(float, model.jnt_range[jid])
                q[i] = np.clip(q[i], lo, hi)
        return q
