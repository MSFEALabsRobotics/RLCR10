#!/usr/bin/env python3
"""Read calibration from a connected RealSense D405."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read D405 1280x720 aligned-depth-to-color intrinsics"
    )
    parser.add_argument(
        "--output",
        default="artifacts/d405_calibration_1280x720.json",
    )
    args = parser.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is required only for this hardware inspection script. "
            "Install the RealSense SDK Python bindings that match the lab SDK."
        ) from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    profile = pipeline.start(config)
    try:
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        if "D405" not in device_name.upper():
            raise RuntimeError(f"connected device is not a D405: {device_name}")

        align = rs.align(rs.stream.color)
        aligned = None
        for _ in range(15):
            aligned = align.process(pipeline.wait_for_frames())
        if aligned is None or not aligned.get_depth_frame() or not aligned.get_color_frame():
            raise RuntimeError("D405 did not produce aligned depth and color frames")

        color_profile = aligned.get_color_frame().profile.as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        depth_sensor = device.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        hfov = math.degrees(2.0 * math.atan(intr.width / (2.0 * intr.fx)))
        vfov = math.degrees(2.0 * math.atan(intr.height / (2.0 * intr.fy)))
        report = {
            "device_model": device_name,
            "stream_convention": "depth_aligned_to_color",
            "resolution_px": [int(intr.width), int(intr.height)],
            "camera_matrix_px": [
                [float(intr.fx), 0.0, float(intr.ppx)],
                [0.0, float(intr.fy), float(intr.ppy)],
                [0.0, 0.0, 1.0],
            ],
            "distortion_model": str(intr.model),
            "distortion_coefficients": [float(value) for value in intr.coeffs],
            "derived_field_of_view_deg": [hfov, vfov],
            "depth_scale_m_per_unit": depth_scale,
            "hardware_motion_commanded": False,
            "note": "Use fx, fy, ppx, and ppy for this exact aligned 1280x720 profile. A separate hand-eye calibration is still required.",
        }
    finally:
        pipeline.stop()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")
    print("No robot motion was commanded.")


if __name__ == "__main__":
    main()
