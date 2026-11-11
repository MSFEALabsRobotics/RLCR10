from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mujoco
from .joint_layout import JointLayout

@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    success: bool
    pos_error_m: float
    rot_error_rad: float
    scaled_sigma_min: float
    iterations: int


def orientation_error_world(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    q_des = np.zeros(4, dtype=float)
    q_cur = np.zeros(4, dtype=float)
    local = np.zeros(3, dtype=float)
    mujoco.mju_mat2Quat(q_des, np.asarray(R_des, dtype=float).reshape(-1))
    mujoco.mju_mat2Quat(q_cur, np.asarray(R_cur, dtype=float).reshape(-1))
    mujoco.mju_subQuat(local, q_des, q_cur)
    return np.asarray(R_cur, dtype=float).reshape(3, 3) @ local


class DampedLeastSquaresIK:
    """Simulation-only 6D IK. Real CR10 execution must use Dobot InverseKin."""

    def __init__(self, model: mujoco.MjModel, site_name: str, rot_scale: float, damping: float,
                 max_iterations: int, max_joint_step: float):
        self.model = model
        self.data = mujoco.MjData(model)
        self.layout = JointLayout.from_model(model)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"site missing: {site_name}")
        self.rot_scale = float(rot_scale)
        self.damping = float(damping)
        self.max_iterations = int(max_iterations)
        self.max_joint_step = float(max_joint_step)

    def _terms(self, p_des: np.ndarray, R_des: np.ndarray):
        p = self.data.site_xpos[self.site_id].copy()
        R = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        ep = np.asarray(p_des, dtype=float).reshape(3) - p
        er = orientation_error_world(R_des, R)
        jp = np.zeros((3, self.model.nv), dtype=float)
        jr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site_id)
        J = np.vstack([
            self.layout.jacobian_columns(jp),
            self.rot_scale * self.layout.jacobian_columns(jr),
        ])
        e = np.concatenate([ep, self.rot_scale * er])
        return J, e, float(np.linalg.norm(ep)), float(np.linalg.norm(er))

    def solve(self, p_des: np.ndarray, R_des: np.ndarray, seed_q: np.ndarray,
              pos_tol: float, rot_tol: float) -> IKResult:
        p_des = np.asarray(p_des, dtype=float).reshape(3)
        R_des = np.asarray(R_des, dtype=float).reshape(3, 3)
        q = self.layout.clamp(self.model, np.asarray(seed_q, dtype=float).reshape(6))
        mujoco.mj_resetData(self.model, self.data)
        self.layout.set_q(self.data, q)
        self.layout.set_qdot_zero(self.data)
        used = 0
        for k in range(self.max_iterations):
            used = k + 1
            mujoco.mj_forward(self.model, self.data)
            J, e, pe, re = self._terms(p_des, R_des)
            if pe <= pos_tol and re <= rot_tol:
                break
            A = J @ J.T + (self.damping ** 2) * np.eye(6)
            try:
                dq = J.T @ np.linalg.solve(A, e)
            except np.linalg.LinAlgError:
                dq = J.T @ np.linalg.lstsq(A, e, rcond=None)[0]
            dq = np.clip(dq, -self.max_joint_step, self.max_joint_step)
            q = self.layout.clamp(self.model, self.layout.get_q(self.data) + dq)
            self.layout.set_q(self.data, q)
        mujoco.mj_forward(self.model, self.data)
        J, _, pe, re = self._terms(p_des, R_des)
        sigma = float(np.linalg.svd(J, compute_uv=False)[-1])
        q = self.layout.get_q(self.data)
        return IKResult(q=q, success=bool(pe <= pos_tol and re <= rot_tol),
                        pos_error_m=pe, rot_error_rad=re,
                        scaled_sigma_min=sigma, iterations=used)
