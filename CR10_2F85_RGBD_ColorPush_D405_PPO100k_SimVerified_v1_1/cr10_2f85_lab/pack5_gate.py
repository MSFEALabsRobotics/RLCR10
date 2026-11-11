from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .camera import (
    CameraIntrinsics,
    camera_pose_in_body,
    project_world_point,
    projected_box_depth_bounds,
    raycast_aligned_depth,
    recover_mujoco_pixel_intrinsics,
)
from .config import load_config
from .rl import FixedGripPushEnv


ROOT = Path(__file__).resolve().parents[1]


def _id(model: mujoco.MjModel, obj: Any, name: str) -> int:
    value = mujoco.mj_name2id(model, obj, name)
    if value < 0:
        raise AssertionError(f"required Pack 5 model element missing: {name}")
    return int(value)


def _render_rgbd_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    intrinsics: CameraIntrinsics,
    cube_geom_id: int,
    table_geom_id: int,
    cr10_arm_geom_ids: list[int],
) -> dict[str, Any]:
    renderer = mujoco.Renderer(
        model, height=intrinsics.height, width=intrinsics.width
    )
    try:
        renderer.update_scene(data, camera=camera_id)
        rgb_a = renderer.render().copy()
        renderer.update_scene(data, camera=camera_id)
        rgb_b = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera_id)
        depth_a = renderer.render().copy()
        renderer.update_scene(data, camera=camera_id)
        depth_b = renderer.render().copy()
        renderer.disable_depth_rendering()

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera_id)
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
    finally:
        renderer.close()

    if rgb_a.shape != (intrinsics.height, intrinsics.width, 3):
        raise AssertionError(f"unexpected RGB shape: {rgb_a.shape}")
    if rgb_a.dtype != np.uint8:
        raise AssertionError(f"unexpected RGB dtype: {rgb_a.dtype}")
    if depth_a.shape != (intrinsics.height, intrinsics.width):
        raise AssertionError(f"unexpected depth shape: {depth_a.shape}")
    if depth_a.dtype != np.float32:
        raise AssertionError(f"unexpected depth dtype: {depth_a.dtype}")
    if segmentation.shape != (intrinsics.height, intrinsics.width, 2):
        raise AssertionError(
            f"unexpected segmentation shape: {segmentation.shape}"
        )
    if not np.array_equal(rgb_a, rgb_b):
        raise AssertionError("two unchanged-state RGB renders differ")
    if not np.array_equal(depth_a, depth_b):
        raise AssertionError("two unchanged-state depth renders differ")
    if float(np.std(rgb_a)) <= 0.0:
        raise AssertionError("RGB render is constant")

    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    cube_mask = (segmentation[:, :, 0] == cube_geom_id) & (
        segmentation[:, :, 1] == geom_type
    )
    table_mask = (segmentation[:, :, 0] == table_geom_id) & (
        segmentation[:, :, 1] == geom_type
    )
    cr10_arm_mask = np.isin(
        segmentation[:, :, 0], np.asarray(cr10_arm_geom_ids, dtype=int)
    ) & (segmentation[:, :, 1] == geom_type)
    cube_pixels = int(np.sum(cube_mask))
    table_pixels = int(np.sum(table_mask))
    cr10_arm_pixels = int(np.sum(cr10_arm_mask))
    if cube_pixels <= 0 or table_pixels <= 0:
        raise AssertionError(
            "native RGB-D render does not contain both cube and table"
        )
    if cr10_arm_pixels != 0:
        raise AssertionError(
            "native wrist view contains CR10 arm pixels; camera is not clear "
            "of the flange"
        )
    if not np.all(np.isfinite(depth_a[cube_mask])):
        raise AssertionError("cube depth contains non-finite values")
    if np.any(depth_a[cube_mask] <= 0.0):
        raise AssertionError("cube depth contains non-positive values")

    projected_u, projected_v, _ = project_world_point(
        data,
        camera_id,
        intrinsics,
        data.geom_xpos[cube_geom_id],
    )
    projected_x = int(np.rint(projected_u))
    projected_y = int(np.rint(projected_v))
    if not (
        0 <= projected_x < intrinsics.width
        and 0 <= projected_y < intrinsics.height
    ):
        raise AssertionError("projected cube center is outside the RGB image")
    if not bool(cube_mask[projected_y, projected_x]):
        raise AssertionError(
            "rendered cube does not cover its projected center pixel; the "
            "view is occluded or raster projection is inconsistent"
        )
    cube_y, cube_x = np.nonzero(cube_mask)
    cube_bbox = [
        int(np.min(cube_x)),
        int(np.min(cube_y)),
        int(np.max(cube_x)),
        int(np.max(cube_y)),
    ]
    cube_touches_border = bool(
        cube_bbox[0] == 0
        or cube_bbox[1] == 0
        or cube_bbox[2] == intrinsics.width - 1
        or cube_bbox[3] == intrinsics.height - 1
    )
    if cube_touches_border:
        raise AssertionError("rendered cube is clipped by the image border")

    cube_rgb_mean = np.mean(rgb_a[cube_mask], axis=0)
    if not (
        float(cube_rgb_mean[0]) > float(cube_rgb_mean[1])
        and float(cube_rgb_mean[0]) > float(cube_rgb_mean[2])
    ):
        raise AssertionError("rendered red cube is not red-channel dominant")

    lower, upper = projected_box_depth_bounds(
        model, data, camera_id, cube_geom_id
    )
    cube_depth = depth_a[cube_mask]
    raster_tolerance_m = 1e-3
    if float(np.min(cube_depth)) < lower - raster_tolerance_m:
        raise AssertionError("rendered cube depth is before its geometric bound")
    if float(np.max(cube_depth)) > upper + raster_tolerance_m:
        raise AssertionError("rendered cube depth is behind its geometric bound")

    return {
        "rgb_shape": list(rgb_a.shape),
        "rgb_dtype": str(rgb_a.dtype),
        "depth_shape": list(depth_a.shape),
        "depth_dtype": str(depth_a.dtype),
        "deterministic_rgb": True,
        "deterministic_depth": True,
        "cube_pixels": cube_pixels,
        "table_pixels": table_pixels,
        "cr10_arm_pixels": cr10_arm_pixels,
        "projected_cube_center_pixel": [projected_u, projected_v],
        "projected_center_pixel_is_cube": True,
        "cube_bbox_xyxy_px": cube_bbox,
        "cube_touches_image_border": False,
        "cube_rgb_mean": [float(v) for v in cube_rgb_mean],
        "cube_depth_min_m": float(np.min(cube_depth)),
        "cube_depth_max_m": float(np.max(cube_depth)),
        "cube_geometric_depth_bounds_m": [float(lower), float(upper)],
    }


def run(
    scene: Path,
    pack4_result: dict,
    *,
    require_opengl_render: bool,
) -> dict:
    """Verify the provisional wrist camera without changing Pack 1--4."""

    cfg = load_config()
    p5 = cfg["pack5"]
    native = CameraIntrinsics.from_config(p5)
    preview_width, preview_height = (
        int(v) for v in p5["policy_preview_resolution_px"]
    )
    preview = native.scaled(preview_width, preview_height)

    env = FixedGripPushEnv(scene)
    env.reset(seed=23)
    model, data = env.model, env.data
    camera_id = _id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, str(p5["camera_name"])
    )
    mount_id = _id(
        model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_mount_frame"
    )
    cube_geom_id = _id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "known_push_cube_geom"
    )
    table_geom_id = _id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "pack2_table_geom"
    )
    housing_geom_id = _id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "d405_visual_housing"
    )
    cr10_arm_body_ids = {
        _id(model, mujoco.mjtObj.mjOBJ_BODY, f"Link{index}")
        for index in range(1, 7)
    }
    cr10_arm_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in cr10_arm_body_ids
    ]
    if len(cr10_arm_geom_ids) != 6:
        raise AssertionError("expected one imported visual geom on each CR10 link")

    if int(model.geom_contype[housing_geom_id]) != 0:
        raise AssertionError("camera housing unexpectedly participates in contact")
    if int(model.geom_conaffinity[housing_geom_id]) != 0:
        raise AssertionError("camera housing unexpectedly accepts contact")
    if model.nq != 21 or model.nv != 20 or model.nu != 7:
        raise AssertionError("Pack 5 camera changed the frozen dynamic state")
    if int(model.vis.global_.offwidth) < native.width:
        raise AssertionError("offscreen framebuffer is narrower than native RGB-D")
    if int(model.vis.global_.offheight) < native.height:
        raise AssertionError("offscreen framebuffer is shorter than native RGB-D")

    recovered = recover_mujoco_pixel_intrinsics(model, camera_id)
    expected_values = np.asarray(
        [native.fx, native.fy, native.cx, native.cy], dtype=float
    )
    recovered_values = np.asarray(
        [recovered.fx, recovered.fy, recovered.cx, recovered.cy], dtype=float
    )
    intrinsic_error = np.abs(recovered_values - expected_values)
    sensor_width, sensor_height = (
        float(value) for value in model.cam_sensorsize[camera_id]
    )
    compiled_principal_offset_px = np.asarray(
        [
            float(model.cam_intrinsic[camera_id][2])
            / sensor_width
            * native.width,
            float(model.cam_intrinsic[camera_id][3])
            / sensor_height
            * native.height,
        ],
        dtype=float,
    )
    expected_principal_offset_px = np.asarray(
        [native.width / 2.0 - native.cx, native.height / 2.0 - native.cy],
        dtype=float,
    )
    principal_offset_error = np.abs(
        compiled_principal_offset_px - expected_principal_offset_px
    )
    if recovered.width != native.width or recovered.height != native.height:
        raise AssertionError("compiled D405 native resolution changed")
    if float(np.max(intrinsic_error)) > float(
        p5["intrinsics_tolerance_px"]
    ):
        raise AssertionError(
            "compiled D405 intrinsics changed: "
            f"max_error={float(np.max(intrinsic_error)):.6g} px"
        )
    if float(np.max(principal_offset_error)) > float(
        p5["intrinsics_tolerance_px"]
    ):
        raise AssertionError(
            "compiled MuJoCo principal-point offset changed: "
            f"max_error={float(np.max(principal_offset_error)):.6g} px"
        )

    render_camera = mujoco.MjvCamera()
    render_camera.fixedcamid = camera_id
    render_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    render_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(render_option)
    render_scene = mujoco.MjvScene(model, maxgeom=10000)
    mujoco.mjv_updateScene(
        model,
        data,
        render_option,
        None,
        render_camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        render_scene,
    )
    gl_camera = render_scene.camera[0]
    frustum_left = float(
        gl_camera.frustum_center - gl_camera.frustum_width
    )
    frustum_right = float(
        gl_camera.frustum_center + gl_camera.frustum_width
    )
    frustum_bottom = float(gl_camera.frustum_bottom)
    frustum_top = float(gl_camera.frustum_top)
    raster_optical_axis_px = np.asarray(
        [
            (0.0 - frustum_left)
            / (frustum_right - frustum_left)
            * native.width,
            (frustum_top - 0.0)
            / (frustum_top - frustum_bottom)
            * native.height,
        ],
        dtype=float,
    )
    raster_axis_error = np.abs(
        raster_optical_axis_px - np.asarray([native.cx, native.cy])
    )
    if float(np.max(raster_axis_error)) > float(
        p5["intrinsics_tolerance_px"]
    ):
        raise AssertionError(
            "MuJoCo raster frustum does not place the optical axis at the "
            "supplied D405 principal point: "
            f"max_error={float(np.max(raster_axis_error)):.6g} px"
        )

    expected_position = np.asarray(
        p5["optical_center_xyz_m_in_mount_frame"], dtype=float
    )
    initial_position, initial_rotation = camera_pose_in_body(
        data, camera_id, mount_id
    )
    if not np.allclose(initial_position, expected_position, atol=1e-12, rtol=0.0):
        raise AssertionError("compiled wrist-camera translation changed")

    projected_u, projected_v, projected_depth = project_world_point(
        data,
        camera_id,
        native,
        data.geom_xpos[cube_geom_id],
    )
    if not (
        0.0 <= projected_u < native.width
        and 0.0 <= projected_v < native.height
        and projected_depth > 0.0
    ):
        raise AssertionError("initial cube center is outside the native image")

    first_depth, first_geom_ids = raycast_aligned_depth(
        model, data, camera_id, preview
    )
    repeated_depth, repeated_geom_ids = raycast_aligned_depth(
        model, data, camera_id, preview
    )
    if not np.array_equal(first_depth, repeated_depth):
        raise AssertionError("unchanged-state CPU depth frames differ")
    if not np.array_equal(first_geom_ids, repeated_geom_ids):
        raise AssertionError("unchanged-state CPU geom-ID frames differ")

    cube_mask = first_geom_ids == cube_geom_id
    table_mask = first_geom_ids == table_geom_id
    first_arm_pixels = int(
        np.sum(np.isin(first_geom_ids, np.asarray(cr10_arm_geom_ids)))
    )
    if not np.any(cube_mask) or not np.any(table_mask):
        raise AssertionError("preview frame does not see both cube and table")
    if first_arm_pixels != 0:
        raise AssertionError("CPU wrist view contains CR10 arm pixels")
    first_cube_depth = first_depth[cube_mask]
    lower, upper = projected_box_depth_bounds(
        model, data, camera_id, cube_geom_id
    )
    if float(np.min(first_cube_depth)) < lower - 1e-9:
        raise AssertionError("ray depth is before cube geometric bound")
    if float(np.max(first_cube_depth)) > upper + 1e-9:
        raise AssertionError("ray depth is behind cube geometric bound")

    render_snapshot = None
    if require_opengl_render:
        try:
            render_snapshot = _render_rgbd_snapshot(
                model,
                data,
                camera_id,
                native,
                cube_geom_id,
                table_geom_id,
                cr10_arm_geom_ids,
            )
        except Exception as exc:
            env.close()
            raise RuntimeError(
                "MuJoCo could not complete the native 1280x720 RGB-D render "
                "gate. Run from the Ubuntu desktop session with a working "
                "OpenGL context. --skip-camera-render is diagnostic only. "
                f"Underlying error: {exc}"
            ) from exc

    minimum_cube_pixels = int(np.sum(cube_mask))
    minimum_table_pixels = int(np.sum(table_mask))
    minimum_cube_depth = float(np.min(first_cube_depth))
    maximum_cube_depth = float(np.max(first_cube_depth))
    maximum_position_drift = 0.0
    maximum_rotation_drift = 0.0
    maximum_cr10_arm_pixels = first_arm_pixels
    frames_checked = 1
    final_info: dict[str, Any] | None = None

    for _ in range(env.max_episode_steps):
        _, _, terminated, truncated, info = env.step(
            np.asarray([1.0], dtype=np.float32)
        )
        depth, geom_ids = raycast_aligned_depth(
            model, data, camera_id, preview
        )
        current_cube_mask = geom_ids == cube_geom_id
        current_table_mask = geom_ids == table_geom_id
        cr10_arm_pixels = int(
            np.sum(np.isin(geom_ids, np.asarray(cr10_arm_geom_ids)))
        )
        cube_pixels = int(np.sum(current_cube_mask))
        table_pixels = int(np.sum(current_table_mask))
        if cube_pixels <= 0 or table_pixels <= 0:
            raise AssertionError(
                f"camera lost cube/table visibility at action {info['episode_step']}"
            )
        if cr10_arm_pixels != 0:
            raise AssertionError(
                f"camera sees CR10 arm at action {info['episode_step']}"
            )
        cube_depth = depth[current_cube_mask]
        if np.any(cube_depth <= 0.0) or not np.all(np.isfinite(cube_depth)):
            raise AssertionError("trajectory cube depth is invalid")

        relative_position, relative_rotation = camera_pose_in_body(
            data, camera_id, mount_id
        )
        maximum_position_drift = max(
            maximum_position_drift,
            float(np.max(np.abs(relative_position - initial_position))),
        )
        maximum_rotation_drift = max(
            maximum_rotation_drift,
            float(np.max(np.abs(relative_rotation - initial_rotation))),
        )
        maximum_cr10_arm_pixels = max(
            maximum_cr10_arm_pixels, cr10_arm_pixels
        )
        minimum_cube_pixels = min(minimum_cube_pixels, cube_pixels)
        minimum_table_pixels = min(minimum_table_pixels, table_pixels)
        minimum_cube_depth = min(
            minimum_cube_depth, float(np.min(cube_depth))
        )
        maximum_cube_depth = max(
            maximum_cube_depth, float(np.max(cube_depth))
        )
        frames_checked += 1
        final_info = info
        if terminated or truncated:
            break

    if final_info is None or not bool(final_info["is_success"]):
        raise AssertionError("camera trajectory did not preserve Pack 4 success")
    if final_info["safety_fault"] is not None:
        raise AssertionError("camera trajectory ended with a safety fault")
    if maximum_position_drift > 1e-12 or maximum_rotation_drift > 1e-12:
        raise AssertionError("camera is not rigidly attached to the wrist mount")

    result = {
        "verification_level": (
            "CR10_2F85_D405_WRIST_RGBD_RENDER_GATE"
            if require_opengl_render
            else "CR10_2F85_D405_WRIST_CAMERA_GEOMETRY_ONLY"
        ),
        "pack4_regression_level": pack4_result["verification_level"],
        "camera": {
            "model": p5["camera_model"],
            "name": p5["camera_name"],
            "stream_convention": p5["stream_convention"],
            "native_resolution_px": [native.width, native.height],
            "preview_resolution_px": [preview.width, preview.height],
            "camera_matrix_px": native.matrix().tolist(),
            "principal_point_convention": p5[
                "principal_point_convention"
            ],
            "compiled_mujoco_principal_offset_px": (
                compiled_principal_offset_px.tolist()
            ),
            "compiled_raster_optical_axis_px": (
                raster_optical_axis_px.tolist()
            ),
            "compiled_intrinsic_max_error_px": float(
                np.max(intrinsic_error)
            ),
            "offscreen_framebuffer_px": [
                int(model.vis.global_.offwidth),
                int(model.vis.global_.offheight),
            ],
            "optical_center_xyz_m_in_mount_frame": expected_position.tolist(),
            "initial_cube_center_pixel": [projected_u, projected_v],
            "initial_cube_center_depth_m": projected_depth,
            "mount_status": p5["mount_status"],
            "intrinsics_status": p5["intrinsics_status"],
            "intrinsics_source": p5["intrinsics_source"],
            "nominal_field_of_view_deg": p5["nominal_field_of_view_deg"],
            "ideal_operating_range_m": p5["ideal_operating_range_m"],
        },
        "cpu_depth_gate": {
            "depth_unit": "m",
            "invalid_depth_value": 0.0,
            "deterministic_unchanged_state": True,
            "trajectory_frames_checked": int(frames_checked),
            "cube_visible_every_frame": True,
            "table_visible_every_frame": True,
            "minimum_cube_pixels": int(minimum_cube_pixels),
            "minimum_table_pixels": int(minimum_table_pixels),
            "cube_depth_min_over_trajectory_m": float(minimum_cube_depth),
            "cube_depth_max_over_trajectory_m": float(maximum_cube_depth),
            "maximum_cr10_arm_pixels": int(maximum_cr10_arm_pixels),
        },
        "rigid_attachment": {
            "maximum_position_drift_m": float(maximum_position_drift),
            "maximum_rotation_matrix_entry_drift": float(
                maximum_rotation_drift
            ),
        },
        "opengl_rgbd_render": {
            "required": bool(require_opengl_render),
            "validated": render_snapshot is not None,
            "raw_background_depth_semantics": (
                "MuJoCo represents unrendered background at its far plane; "
                "camera_preview.py remaps unsegmented pixels to 0"
            ),
            "snapshot": render_snapshot,
        },
        "reference_task": {
            "episode_steps": int(final_info["episode_step"]),
            "cube_progress_m": float(final_info["cube_progress_m"]),
            "success": True,
            "safety_fault": None,
        },
        "hardware_motion_commanded": False,
        "claims": {
            "validated": [
                "compiled 1280x720 pinhole projection matches the configured nominal D405 matrix",
                "camera transform is rigidly attached to the Robotiq mount",
                "metric axial depth and geom visibility are deterministic",
                "cube and table remain visible through the complete verified push",
                "no CR10 arm mesh is visible in the provisional wrist view",
                "zero-mass non-colliding camera visual preserves the Pack 4 task",
            ]
            + (
                [
                    "native-resolution MuJoCo RGB, depth, and segmentation rendering",
                    "rendered cube color and depth agree with model geometry",
                ]
                if require_opengl_render
                else []
            ),
            "not_validated": [
                "physical wrist-camera extrinsic",
                "unit-specific D405 intrinsics and distortion",
                "lens distortion or stereo depth artifacts",
                "physical depth scale and invalid-pixel behavior",
                "camera mass or mount collisions",
                "vision-policy learning or robustness",
                "sim-to-real transfer",
            ],
        },
    }
    env.close()
    suffix = (
        "CR10_2F85_D405_WRIST_RGBD_RENDER_GATE.json"
        if require_opengl_render
        else "CR10_2F85_D405_WRIST_CAMERA_GEOMETRY_ONLY.json"
    )
    output = ROOT / "verification" / suffix
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
