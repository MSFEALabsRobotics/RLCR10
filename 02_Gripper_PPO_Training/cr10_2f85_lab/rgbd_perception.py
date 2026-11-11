from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .camera import CameraIntrinsics


@dataclass(frozen=True)
class ColorDetection:
    color: str
    mask_pixels: int
    center_pixel_uv: np.ndarray
    median_depth_m: float
    point_camera_xyz_m: np.ndarray
    point_world_xyz_m: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "color": self.color,
            "mask_pixels": int(self.mask_pixels),
            "center_pixel_uv": self.center_pixel_uv.tolist(),
            "median_depth_m": float(self.median_depth_m),
            "visible_surface_point_camera_xyz_m": (self.point_camera_xyz_m.tolist()),
            "visible_surface_point_world_xyz_m": self.point_world_xyz_m.tolist(),
        }


def rgb_to_hsv_0_to_1(rgb: np.ndarray) -> np.ndarray:
    """Vectorized uint8/float RGB to HSV with all channels in [0, 1]."""

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (H,W,3), got {array.shape}")
    values = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer):
        values /= 255.0
    elif float(np.max(values, initial=0.0)) > 1.0 + 1e-6:
        values /= 255.0
    values = np.clip(values, 0.0, 1.0)

    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    maximum = np.max(values, axis=2)
    minimum = np.min(values, axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    chromatic = delta > 1e-8

    red_max = chromatic & (maximum == red)
    green_max = chromatic & (maximum == green) & ~red_max
    blue_max = chromatic & ~(red_max | green_max)
    hue[red_max] = np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6.0)
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2.0
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0
    hue /= 6.0
    saturation = np.zeros_like(maximum)
    nonblack = maximum > 1e-8
    saturation[nonblack] = delta[nonblack] / maximum[nonblack]
    return np.stack((hue, saturation, maximum), axis=2)


def color_mask_from_hsv(
    hsv: np.ndarray,
    color_config: dict[str, Any],
) -> np.ndarray:
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    hue_match = np.zeros(hue.shape, dtype=bool)
    for lower, upper in color_config["hue_ranges_0_to_1"]:
        hue_match |= (hue >= float(lower)) & (hue <= float(upper))
    return (
        hue_match
        & (saturation >= float(color_config["minimum_saturation"]))
        & (value >= float(color_config["minimum_value"]))
    )


def detect_colored_surface(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_position_world_m: np.ndarray,
    camera_rotation_world: np.ndarray,
    color: str,
    color_config: dict[str, Any],
    minimum_pixels: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> ColorDetection | None:
    """Back-project the median visible RGB-D surface point for one color."""

    rgb_array = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float32)
    expected_rgb = (intrinsics.height, intrinsics.width, 3)
    expected_depth = (intrinsics.height, intrinsics.width)
    if rgb_array.shape != expected_rgb:
        raise ValueError(
            f"unexpected RGB shape {rgb_array.shape}, expected {expected_rgb}"
        )
    if depth.shape != expected_depth:
        raise ValueError(
            f"unexpected depth shape {depth.shape}, expected {expected_depth}"
        )
    hsv = rgb_to_hsv_0_to_1(rgb_array)
    color_mask = color_mask_from_hsv(hsv, color_config)
    valid_depth = (
        np.isfinite(depth)
        & (depth >= float(minimum_depth_m))
        & (depth <= float(maximum_depth_m))
    )
    mask = color_mask & valid_depth
    rows, columns = np.nonzero(mask)
    if rows.size < int(minimum_pixels):
        return None

    z = depth[rows, columns].astype(np.float64)
    x = (columns.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = -(rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    points_camera = np.column_stack((x, y, -z))
    point_camera = np.median(points_camera, axis=0)
    rotation = np.asarray(camera_rotation_world, dtype=float).reshape(3, 3)
    position = np.asarray(camera_position_world_m, dtype=float).reshape(3)
    point_world = position + rotation @ point_camera
    return ColorDetection(
        color=str(color),
        mask_pixels=int(rows.size),
        center_pixel_uv=np.asarray([np.median(columns), np.median(rows)], dtype=float),
        median_depth_m=float(np.median(z)),
        point_camera_xyz_m=point_camera,
        point_world_xyz_m=point_world,
    )


def detect_configured_colors(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_position_world_m: np.ndarray,
    camera_rotation_world: np.ndarray,
    color_task_config: dict[str, Any],
) -> dict[str, ColorDetection | None]:
    return {
        color: detect_colored_surface(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            camera_position_world_m=camera_position_world_m,
            camera_rotation_world=camera_rotation_world,
            color=color,
            color_config=color_config,
            minimum_pixels=int(color_task_config["minimum_color_mask_pixels"]),
            minimum_depth_m=float(color_task_config["minimum_valid_depth_m"]),
            maximum_depth_m=float(color_task_config["maximum_valid_depth_m"]),
        )
        for color, color_config in color_task_config["cube_colors"].items()
    }
