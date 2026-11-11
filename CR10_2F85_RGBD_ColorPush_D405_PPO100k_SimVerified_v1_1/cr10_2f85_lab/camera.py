from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_config(cls, pack5: dict[str, Any]) -> "CameraIntrinsics":
        width, height = (int(v) for v in pack5["native_resolution_px"])
        fx, fy = (float(v) for v in pack5["focal_length_px"])
        cx, cy = (float(v) for v in pack5["principal_point_px"])
        return cls(width, height, fx, fy, cx, cy)

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        sx = float(width) / float(self.width)
        sy = float(height) / float(self.height)
        return CameraIntrinsics(
            int(width),
            int(height),
            self.fx * sx,
            self.fy * sy,
            self.cx * sx,
            self.cy * sy,
        )

    def matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )


def recover_mujoco_pixel_intrinsics(
    model: mujoco.MjModel, camera_id: int
) -> CameraIntrinsics:
    width, height = (int(v) for v in model.cam_resolution[camera_id])
    sensor_width, sensor_height = (
        float(v) for v in model.cam_sensorsize[camera_id]
    )
    intrinsic = model.cam_intrinsic[camera_id]
    principal_x_offset_px = float(intrinsic[2]) / sensor_width * width
    principal_y_offset_px = float(intrinsic[3]) / sensor_height * height
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=float(intrinsic[0]) / sensor_width * width,
        fy=float(intrinsic[1]) / sensor_height * height,
        cx=width / 2.0 - principal_x_offset_px,
        cy=height / 2.0 - principal_y_offset_px,
    )


def camera_pose_in_body(
    data: mujoco.MjData, camera_id: int, body_id: int
) -> tuple[np.ndarray, np.ndarray]:
    body_rotation = data.xmat[body_id].reshape(3, 3)
    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
    relative_position = body_rotation.T @ (
        data.cam_xpos[camera_id] - data.xpos[body_id]
    )
    relative_rotation = body_rotation.T @ camera_rotation
    return relative_position, relative_rotation


def project_world_point(
    data: mujoco.MjData,
    camera_id: int,
    intrinsics: CameraIntrinsics,
    point_world: np.ndarray,
) -> tuple[float, float, float]:
    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
    point_camera = camera_rotation.T @ (
        np.asarray(point_world, dtype=float) - data.cam_xpos[camera_id]
    )
    depth = -float(point_camera[2])
    if depth <= 0.0:
        return float("nan"), float("nan"), depth
    u = intrinsics.fx * float(point_camera[0]) / depth + intrinsics.cx
    v = -intrinsics.fy * float(point_camera[1]) / depth + intrinsics.cy
    return u, v, depth


def projected_box_depth_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    geom_id: int,
) -> tuple[float, float]:
    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
    camera_forward = -camera_rotation[:, 2]
    center_depth = float(
        np.dot(
            data.geom_xpos[geom_id] - data.cam_xpos[camera_id],
            camera_forward,
        )
    )
    geom_rotation = data.geom_xmat[geom_id].reshape(3, 3)
    half_size = model.geom_size[geom_id]
    projected_half_extent = sum(
        abs(float(np.dot(camera_forward, geom_rotation[:, axis])))
        * float(half_size[axis])
        for axis in range(3)
    )
    return (
        center_depth - projected_half_extent,
        center_depth + projected_half_extent,
    )


def raycast_aligned_depth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Return axial depth and first-hit geometry IDs."""

    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
    camera_origin = data.cam_xpos[camera_id].copy()
    camera_forward = -camera_rotation[:, 2]
    depth = np.zeros((intrinsics.height, intrinsics.width), dtype=np.float32)
    geom_ids = np.full(
        (intrinsics.height, intrinsics.width), -1, dtype=np.int32
    )
    geom_id = np.empty(1, dtype=np.int32)
    geom_groups = np.ones(6, dtype=np.uint8)

    for v in range(intrinsics.height):
        camera_y = -(float(v) - intrinsics.cy) / intrinsics.fy
        for u in range(intrinsics.width):
            ray_camera = np.asarray(
                [
                    (float(u) - intrinsics.cx) / intrinsics.fx,
                    camera_y,
                    -1.0,
                ],
                dtype=float,
            )
            ray_world = camera_rotation @ ray_camera
            ray_world /= np.linalg.norm(ray_world)
            distance = float(
                mujoco.mj_ray(
                    model,
                    data,
                    camera_origin,
                    ray_world,
                    geom_groups,
                    True,
                    -1,
                    geom_id,
                )
            )
            if distance < 0.0:
                continue
            axial_depth = distance * float(np.dot(ray_world, camera_forward))
            if axial_depth <= 0.0:
                continue
            depth[v, u] = np.float32(axial_depth)
            geom_ids[v, u] = int(geom_id[0])
    return depth, geom_ids
