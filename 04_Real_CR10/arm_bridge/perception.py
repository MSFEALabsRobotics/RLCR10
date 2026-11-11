"""D405 color-aligned depth to physical 3D surface points (no MuJoCo state)."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np

from .core import SafetyFault

@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    receipt_monotonic_s: float
    frame_number: int
    intrinsics: object | None = None

class D405:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = None
        self.last_number = -1

    def __enter__(self):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError('pyrealsense2 missing: install lab-compatible RealSense SDK') from e
        self.rs = rs
        self.pipeline = rs.pipeline()
        c = rs.config()
        if self.cfg.get('serial'):
            c.enable_device(self.cfg['serial'])
        width, height = self.cfg['resolution_px']
        fps = int(self.cfg['fps'])
        c.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        c.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(c)
        name = profile.get_device().get_info(rs.camera_info.name)
        if 'D405' not in name.upper():
            self.pipeline.stop()
            raise SafetyFault(f'not a D405: {name}')
        self.scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self.aligner = rs.align(rs.stream.color)
        for _ in range(8):
            self.pipeline.wait_for_frames(timeout_ms=3000)
        return self

    def __exit__(self, *_):
        if self.pipeline is not None:
            self.pipeline.stop()

    def read(self):
        aligned = self.aligner.process(self.pipeline.wait_for_frames(timeout_ms=1500))
        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()
        receipt = time.monotonic()
        if not depth or not color:
            raise SafetyFault('MISSING_ALIGNED_RGBD_FRAME')
        number = int(color.get_frame_number())
        if number <= self.last_number:
            raise SafetyFault('REPEATED_OR_OUT_OF_ORDER_CAMERA_FRAME')
        self.last_number = number
        intr = color.profile.as_video_stream_profile().get_intrinsics()
        rgb = np.asanyarray(color.get_data()).copy()
        raw_depth = np.asanyarray(depth.get_data())
        depth_m = raw_depth.astype(np.float32) * self.scale
        if rgb.shape[:2] != depth_m.shape:
            raise SafetyFault('RGB depth alignment resolution mismatch')
        return RGBDFrame(rgb, depth_m, float(intr.fx), float(intr.fy),
                         float(intr.ppx), float(intr.ppy), receipt, number, intr)


def rgb_to_hsv_sim(rgb):
    """Keep Pack 5 HSV thresholds/semantics unchanged."""
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    r,g,b = a[:,:,0],a[:,:,1],a[:,:,2]
    mx = np.max(a,axis=2); mn = np.min(a,axis=2); d = mx-mn
    h = np.zeros_like(mx)
    chrom = d > 1e-8
    rm = chrom & (mx == r)
    gm = chrom & (mx == g) & ~rm
    bm = chrom & ~(rm | gm)
    h[rm] = np.mod((g[rm]-b[rm])/d[rm],6)
    h[gm] = (b[gm]-r[gm])/d[gm]+2
    h[bm] = (r[bm]-g[bm])/d[bm]+4
    h /= 6.0
    s = np.zeros_like(mx)
    ok = mx > 1e-8
    s[ok] = d[ok]/mx[ok]
    return h,s,mx


def resize_aligned_frame(rgb, depth, w, h):
    """Aligned resize. NumPy exact-factor path removes OpenCV dependency offline.

    The D405 nominal 1280x720 -> PPO 128x72 scales by an integer factor 10.
    RGB area averaging and nearest-neighbor depth preserve prior conventions.
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise SafetyFault('RGB/depth dimensions mismatch')
    H, W = depth.shape
    if H == h and W == w:
        return rgb, depth
    if H >= h and W >= w and H % h == 0 and W % w == 0:
        sy, sx = H // h, W // w
        avg = rgb.reshape(h, sy, w, sx, 3).mean(axis=(1, 3))
        if np.issubdtype(rgb.dtype, np.integer):
            avg = np.floor(avg + 0.5).astype(rgb.dtype)
        return avg, depth[::sy, ::sx]
    try:
        import cv2
    except ImportError as exc:
        raise SafetyFault('non-integer RGB-D resizing requires OpenCV') from exc
    return (cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA),
            cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST))

def find_colors(frame, cfg, T_base_cam=None, *, rs_module=None):
    """Returns median visible surface point in robot base coordinates if calibrated."""
    if time.monotonic() - frame.receipt_monotonic_s > float(cfg['max_capture_age_s']):
        raise SafetyFault('STALE_CAMERA_FRAME')
    w,h = cfg['policy_resolution_px']
    rgb, depth = resize_aligned_frame(frame.rgb, frame.depth_m, w, h)
    if not np.all(np.isfinite([frame.fx, frame.fy, frame.cx, frame.cy])) or min(frame.fx,frame.fy)<=0:
        raise SafetyFault('invalid D405 intrinsics')
    sx,sy = w / frame.rgb.shape[1], h / frame.rgb.shape[0]
    fx,fy,cx,cy = frame.fx*sx, frame.fy*sy, frame.cx*sx, frame.cy*sy
    hue,sat,val = rgb_to_hsv_sim(rgb)
    depth_ok = np.isfinite(depth) & (depth >= cfg['min_valid_depth_m']) & (depth <= cfg['max_valid_depth_m'])
    result = {}
    for name,color in cfg['colors'].items():
        hm = np.zeros((h,w),bool)
        for lo,hi in color['hue_ranges_0_to_1']:
            hm |= (hue>=lo)&(hue<=hi)
        mask = hm & (sat>=color['minimum_saturation']) & (val>=color['minimum_value']) & depth_ok
        yy,xx = np.where(mask)
        if len(xx) < cfg['min_mask_pixels']:
            result[name]=None
            continue
        d = depth[yy,xx].astype(np.float64)
        px = (xx-cx)*d/fx
        py = (yy-cy)*d/fy
        p_opt = np.median(np.stack([px,py,d],axis=1),axis=0)
        # The sim's coordinate convention is x-right, y-up, z-back;
        # RealSense is x-right, y-down, z-forward. T_base_cam uses RealSense optical axes.
        if rs_module is not None and frame.intrinsics is not None:
            # Device intrinsics include real lens distortion. Use SDK deprojection of the median
            # surface pixel/depth rather than assuming the physical camera is ideal pinhole.
            uv_full = [float(np.median(xx))/sx, float(np.median(yy))/sy]
            p_opt = np.array(rs_module.rs2_deproject_pixel_to_point(
                frame.intrinsics, uv_full, float(np.median(d))),dtype=float)
        item = {
            'point_optical_m': p_opt,
            'depth_m': float(np.median(d)),
            'uv_policy_px':[float(np.median(xx)),float(np.median(yy))],
            'pixels':len(xx),
        }
        if T_base_cam is not None:
            item['point_base_m'] = T_base_cam[:3,:3] @ p_opt + T_base_cam[:3,3]
        result[name]=item
    return result
