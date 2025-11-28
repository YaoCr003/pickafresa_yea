"""
Real-time Object Detection Testing Tool for YOLOv11 Models

Interactive testing tool for evaluating object detection models on live video feeds.
Supports RealSense cameras (with optional depth overlay) and standard USB/built-in cameras.

Features:
- Model selection from available .pt files
- Camera source selection (RealSense or OpenCV VideoCapture)
- Real-time bounding box visualization with class-specific colors
- Depth information overlay for RealSense cameras
- Performance statistics (FPS, detection counts)
- Interactive controls for threshold adjustment and frame capture
- Configuration persistence across runs

Usage:
    python objd_testing.py

Keyboard Controls:
    q - Quit
    s - Save current frame
    p - Pause/Resume
    + - Increase confidence threshold by 0.05
    - - Decrease confidence threshold by 0.05
    r - Reset to default settings

@aldrick-t, 2025
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure repository root is on sys.path for absolute package imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False
    print("Warning: pyrealsense2 not available. RealSense cameras will not be accessible.")

# Import profile verification tools for automatic configuration
if HAVE_REALSENSE:
    try:
        from pickafresa_vision.realsense_tools.realsense_verify_color import (
            get_best_color_profile,
            get_camera_serial,
        )
        from pickafresa_vision.realsense_tools.realsense_verify_depth import (
            get_best_depth_profile,
        )
        from pickafresa_vision.realsense_tools.realsense_verify_full import (
            get_best_full_profile,
            get_working_full_profiles,
            load_working_profiles,
        )
        HAVE_VERIFICATION = True
    except ImportError:
        HAVE_VERIFICATION = False
        print("Warning: Could not import verification tools. Using default resolution.")
else:
    HAVE_VERIFICATION = False

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False
    print("Warning: pyyaml not available. Will use fallback class names.")

from pickafresa_vision.vision_nodes.inference_bbox import load_model, infer
from pickafresa_vision.vision_tools.config_store import (
    load_config,
    save_config,
    get_namespace,
    update_namespace,
)


def _load_objd_config_defaults() -> dict:
    """Load default values from objd_config.yaml if available."""
    config_path = REPO_ROOT / "pickafresa_vision" / "configs" / "objd_config.yaml"
    
    if not HAVE_YAML or not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# Configuration namespace for this tool
CONFIG_NAMESPACE = "objd_testing"

# Default class colors (BGR format for OpenCV)
DEFAULT_CLASS_COLORS = {
    "ripe": (0, 255, 0),      # Green
    "unripe": (0, 255, 255),  # Yellow
    "flower": (255, 0, 0),    # Blue
}


@dataclass
class TestConfig:
    """Configuration for the testing session."""
    model_path: str
    dataset_path: str
    camera_type: str  # "realsense" or "opencv"
    camera_index: int
    enable_depth: bool
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 15
    # Depth overlay rendering on color window
    depth_overlay_enabled: bool = False
    depth_overlay_alpha: float = 0.0  # 0.0 (no overlay) .. 1.0 (full depth colormap)
    # Persisted user preferences for RealSense fps selection (global)
    preferred_color_fps: Optional[int] = None
    preferred_depth_fps: Optional[int] = None
    # Persisted preferred full profile selection when depth is enabled
    # ((color_w, color_h, color_fps), (depth_w, depth_h, depth_fps))
    preferred_full_profile: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None


class FPSCounter:
    """Simple FPS counter with exponential moving average."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.fps = 0.0
        self.last_time = time.time()
    
    def update(self) -> float:
        current_time = time.time()
        delta = current_time - self.last_time
        if delta > 0:
            instant_fps = 1.0 / delta
            self.fps = self.alpha * instant_fps + (1 - self.alpha) * self.fps
        self.last_time = current_time
        return self.fps


class RealSenseCamera:
    """Wrapper for RealSense camera with depth support.

    Supports color-only mode or combined color+depth with potentially different
    resolutions and FPS for each stream.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        auto_detect: bool = True,
        enable_depth: bool = True,
    ):
        if not HAVE_REALSENSE:
            raise RuntimeError("pyrealsense2 is required for RealSense cameras")

        # Color stream settings (primary)
        self.resolution = resolution
        self.fps = fps
        self.auto_detect = auto_detect
        self.enable_depth = enable_depth

        # Allow distinct depth settings
        self.depth_resolution: Tuple[int, int] = resolution
        self.depth_fps: int = fps

        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self.depth_scale = 0.0
        self._filters = []
        # Selected full profile pair if chosen from cache
        self.selected_full_profile = None

    def _get_device_by_serial(self, serial: Optional[str]) -> Optional[rs.device]:
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                return None
            if not serial:
                return devices[0]
            for d in devices:
                try:
                    if d.get_info(rs.camera_info.serial_number) == serial:
                        return d
                except Exception:
                    continue
            return devices[0]
        except Exception:
            return None

    def _enumerate_fps_options(
        self,
        device: Optional[rs.device],
        color_res: Tuple[int, int],
        depth_res: Optional[Tuple[int, int]],
        enable_depth: bool,
    ) -> Tuple[List[int], List[int]]:
        """Enumerate available fps options for the given resolutions.

        Returns (color_fps_options, depth_fps_options). If depth disabled, depth list may be [].
        """
        color_set: set[int] = set()
        depth_set: set[int] = set()
        if not device:
            return sorted(color_set), sorted(depth_set)
        try:
            for sensor in device.sensors:
                try:
                    for prof in sensor.get_stream_profiles():
                        try:
                            vprof = prof.as_video_stream_profile()
                        except Exception:
                            continue
                        stype = vprof.stream_type()
                        fmt = vprof.format()
                        w, h = vprof.width(), vprof.height()
                        fps = vprof.fps()
                        if stype == rs.stream.color and fmt == rs.format.bgr8 and (w, h) == color_res:
                            color_set.add(int(fps))
                        if enable_depth and depth_res and stype == rs.stream.depth and fmt == rs.format.z16 and (w, h) == depth_res:
                            depth_set.add(int(fps))
                except Exception:
                    continue
        except Exception:
            pass
        return sorted(color_set), sorted(depth_set)

    def _prompt_select_fps_color(self, options: List[int], saved: Optional[int]) -> Optional[int]:
        if not options:
            return None
        # If saved preference exists and is available, use it automatically
        if saved and saved in options:
            print(f"Using saved color fps preference: {saved}")
            return saved
        if len(options) == 1:
            print(f"Only one color fps available: {options[0]}")
            return options[0]
        print("\n=== Select Color FPS ===")
        for i, f in enumerate(options, 1):
            print(f"{i}. {f} fps")
        while True:
            choice = input(f"Select color fps (1-{len(options)}) or 'q' to cancel: ").strip().lower()
            if choice == 'q':
                return None
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except Exception:
                pass
            print("Please enter a valid option.")

    def _prompt_select_fps_pair(
        self,
        color_options: List[int],
        depth_options: List[int],
        saved_color: Optional[int],
        saved_depth: Optional[int],
        color_res: Tuple[int, int],
        depth_res: Tuple[int, int],
    ) -> Tuple[Optional[int], Optional[int]]:
        # If both saved exist and are available, use them
        if saved_color in color_options and saved_depth in depth_options:
            print(f"Using saved fps preferences: Color {saved_color}, Depth {saved_depth}")
            return saved_color, saved_depth
        # If only single options, pick them
        if len(color_options) == 1 and len(depth_options) == 1:
            print(f"Only one fps pair available: Color {color_options[0]}, Depth {depth_options[0]}")
            return color_options[0], depth_options[0]

        # Build cartesian pairs for user selection
        pairs: List[Tuple[int, int]] = [(c, d) for c in color_options for d in depth_options]
        print("\n=== Select FPS Pair (Color | Depth) ===")
        for i, (c, d) in enumerate(pairs, 1):
            print(f"{i}. Color {color_res[0]}x{color_res[1]} @ {c} | Depth {depth_res[0]}x{depth_res[1]} @ {d}")
        while True:
            choice = input(f"Select pair (1-{len(pairs)}) or 'q' to cancel: ").strip().lower()
            if choice == 'q':
                return None, None
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(pairs):
                    return pairs[idx]
            except Exception:
                pass
            print("Please enter a valid option.")

    def _prompt_select_cached_full_profile(
        self,
        cached: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
        saved_pref: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
    ) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """Prompt user to select a full (color+depth) profile from cache.

        If saved_pref is present and matches one of the cached entries, use it automatically.
        """
        if not cached:
            return None
        # Auto-use saved preference when available
        if saved_pref and saved_pref in cached:
            c, d = saved_pref
            print(
                f"Using saved full profile preference: Color {c[0]}x{c[1]}@{c[2]} | Depth {d[0]}x{d[1]}@{d[2]}"
            )
            return saved_pref

        # If differing fps pairs exist, prompt; otherwise just use the first
        fps_pairs = {(c[2], d[2]) for c, d in cached}
        if len(fps_pairs) <= 1 and len(cached) == 1:
            return cached[0]

        print("\n=== Select Cached Full Profile (Color | Depth) ===")
        for i, (c, d) in enumerate(cached, 1):
            print(f"{i}. Color {c[0]}x{c[1]} @ {c[2]} | Depth {d[0]}x{d[1]} @ {d[2]}")
        while True:
            choice = input(f"Select profile (1-{len(cached)}) or 'q' to cancel: ").strip().lower()
            if choice == 'q':
                return None
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(cached):
                    return cached[idx]
            except Exception:
                pass
            print("Please enter a valid option.")

    def start(self):
        """Initialize and start the RealSense pipeline."""
        color_w, color_h = self.resolution
        color_fps = self.fps

        # Auto-detect profile(s) if enabled
        if self.auto_detect and HAVE_VERIFICATION:
            print("\n=== Auto-detecting camera profiles ===")
            try:
                if self.enable_depth:
                    # Prefer cached full profiles; prompt user when fps differ
                    try:
                        serial = get_camera_serial()
                    except Exception:
                        serial = None

                    cached_profiles = load_working_profiles(serial) if serial else None
                    saved_cfg = load_previous_config()
                    saved_full_pref = saved_cfg.preferred_full_profile if saved_cfg else None

                    selected: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
                    if cached_profiles:
                        selected = self._prompt_select_cached_full_profile(cached_profiles, saved_full_pref)

                    if not selected:
                        # Fall back to picking a full profile (independent mode to widen options)
                        try:
                            pair = get_best_full_profile(mode="independent", verbose=True, use_cache=True, validate_cached=False)
                        except Exception as e:
                            print(f"Full verification unavailable ({e}), falling back to separate checks...")
                            pair = None
                        selected = pair

                    if selected:
                        (color_w, color_h, color_fps), (depth_w, depth_h, depth_fps) = selected
                        print(
                            f"[OK] Selected: Color {color_w}x{color_h}@{color_fps} | "
                            f"Depth {depth_w}x{depth_h}@{depth_fps}"
                        )
                        self.resolution = (color_w, color_h)
                        self.fps = color_fps
                        self.depth_resolution = (depth_w, depth_h)
                        self.depth_fps = depth_fps
                        self.selected_full_profile = selected
                        # Persist selected full profile globally for RealSense runs
                        try:
                            cfg = load_config()
                            update_namespace(cfg, CONFIG_NAMESPACE, {
                                "preferred_full_profile": {
                                    "color": [color_w, color_h, color_fps],
                                    "depth": [depth_w, depth_h, depth_fps],
                                }
                            })
                        except Exception:
                            pass
                    else:
                        # Fallback: pick best color and best depth independently
                        print("Detecting best color profile...")
                        best_color = get_best_color_profile(verbose=True, use_cache=True, validate_cached=False)
                        if best_color:
                            color_w, color_h, color_fps = best_color
                            print(f"[OK] Color: {color_w}x{color_h}@{color_fps}")
                            self.resolution = (color_w, color_h)
                            self.fps = color_fps

                        print("Detecting best depth profile...")
                        best_depth = get_best_depth_profile(verbose=True, use_cache=True, validate_cached=False)
                        if best_depth:
                            dw, dh, dfps = best_depth
                            print(f"[OK] Depth: {dw}x{dh}@{dfps}")
                            self.depth_resolution = (dw, dh)
                            self.depth_fps = dfps
                else:
                    # Only color stream needed
                    print("Detecting best color profile (depth disabled)...")
                    best_color = get_best_color_profile(verbose=True, use_cache=True, validate_cached=False)
                    if best_color:
                        color_w, color_h, color_fps = best_color
                        print(f"[OK] Selected color profile: {color_w}x{color_h}@{color_fps}fps")
                        self.resolution = (color_w, color_h)
                        self.fps = color_fps
            except Exception as e:
                print(
                    f"Warning: Auto-detection failed ({e}), using default "
                    f"{color_w}x{color_h}@{color_fps}fps"
                )

        # At this point we have chosen resolutions; if no full profile was explicitly selected,
        # and multiple fps options exist, prompt user for fps selection.
        if self.selected_full_profile is None:
            try:
                device_serial = get_camera_serial() if HAVE_VERIFICATION else None
            except Exception:
                device_serial = None
            device = self._get_device_by_serial(device_serial)

            # Enumerate available fps for these resolutions
            depth_res_tuple: Optional[Tuple[int, int]] = self.depth_resolution if self.enable_depth else None
            color_fps_opts, depth_fps_opts = self._enumerate_fps_options(device, (color_w, color_h), depth_res_tuple, self.enable_depth)

            # Load saved preferences if any
            saved_cfg = load_previous_config()
            saved_color_pref = saved_cfg.preferred_color_fps if saved_cfg else None
            saved_depth_pref = saved_cfg.preferred_depth_fps if saved_cfg else None

            if self.enable_depth:
                if color_fps_opts or depth_fps_opts:
                    sel_color, sel_depth = self._prompt_select_fps_pair(
                        color_fps_opts or [color_fps],
                        depth_fps_opts or [self.depth_fps],
                        saved_color_pref,
                        saved_depth_pref,
                        (color_w, color_h),
                        self.depth_resolution,
                    )
                    if sel_color:
                        color_fps = sel_color
                        self.fps = sel_color
                    if sel_depth:
                        self.depth_fps = sel_depth
            else:
                if color_fps_opts:
                    sel_color = self._prompt_select_fps_color(color_fps_opts, saved_color_pref)
                    if sel_color:
                        color_fps = sel_color
                        self.fps = sel_color

        # Small settle delay after any prior verification start/stop cycles
        time.sleep(0.4)

        self.pipeline = rs.pipeline()
        config = rs.config()
        # Bind to the detected device to prevent librealsense from probing other sensors
        try:
            device_serial = get_camera_serial() if HAVE_VERIFICATION else None
            if device_serial:
                config.enable_device(device_serial)
        except Exception:
            pass
        config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, color_fps)

        if self.enable_depth:
            depth_w, depth_h = self.depth_resolution
            depth_fps = self.depth_fps
            config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, depth_fps)

        # Try to start; if it fails (e.g., invalid fps pair), re-prompt if possible
        start_attempts = 0
        while True:
            try:
                profile = self.pipeline.start(config)
                break
            except Exception as e:
                start_attempts += 1
                print(f"Failed to start pipeline with selected fps (attempt {start_attempts}): {e}")
                if start_attempts >= 3:
                    raise
                # Re-prompt available fps options and update config
                color_fps_opts, depth_fps_opts = self._enumerate_fps_options(device, (color_w, color_h), self.depth_resolution if self.enable_depth else None, self.enable_depth)
                if self.enable_depth:
                    sel_color, sel_depth = self._prompt_select_fps_pair(
                        color_fps_opts or [color_fps],
                        depth_fps_opts or [self.depth_fps],
                        None,
                        None,
                        (color_w, color_h),
                        self.depth_resolution,
                    )
                    if sel_color:
                        color_fps = sel_color
                        self.fps = sel_color
                    if sel_depth:
                        self.depth_fps = sel_depth
                    # Reconfigure streams with new fps
                    config = rs.config()
                    try:
                        device_serial = get_camera_serial() if HAVE_VERIFICATION else None
                        if device_serial:
                            config.enable_device(device_serial)
                    except Exception:
                        pass
                    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, color_fps)
                    dw, dh = self.depth_resolution
                    config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, self.depth_fps)
                else:
                    sel_color = self._prompt_select_fps_color(color_fps_opts or [color_fps], None)
                    if sel_color:
                        color_fps = sel_color
                        self.fps = sel_color
                        config = rs.config()
                        try:
                            device_serial = get_camera_serial() if HAVE_VERIFICATION else None
                            if device_serial:
                                config.enable_device(device_serial)
                        except Exception:
                            pass
                        config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, color_fps)

        if self.enable_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            self.align = rs.align(rs.stream.color)

            # Setup depth filtering pipeline
            try:
                self._filters = [
                    rs.disparity_transform(True),
                    rs.spatial_filter(),
                    rs.temporal_filter(),
                    rs.disparity_transform(False),
                    rs.hole_filling_filter(),
                ]
                # Tune filters
                spatial: rs.spatial_filter = self._filters[1]
                spatial.set_option(rs.option.holes_fill, 3)
                temporal: rs.temporal_filter = self._filters[2]
                temporal.set_option(rs.option.alpha, 0.5)
            except Exception:
                self._filters = []

    def read(self) -> Tuple[bool, np.ndarray, Optional[rs.depth_frame]]:
        """Read aligned color and depth frames."""
        if not self.pipeline:
            return False, np.zeros((480, 640, 3), dtype=np.uint8), None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)

            if self.enable_depth and self.align:
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    return False, np.zeros((480, 640, 3), dtype=np.uint8), None

                # Apply depth filtering
                try:
                    f = depth_frame
                    for flt in self._filters:
                        f = flt.process(f)
                    depth_frame = f.as_depth_frame()
                except Exception:
                    pass

                color_image = np.asanyarray(color_frame.get_data())
                return True, color_image, depth_frame
            else:
                # Color only mode
                color_frame = frames.get_color_frame()
                if not color_frame:
                    return False, np.zeros((480, 640, 3), dtype=np.uint8), None

                color_image = np.asanyarray(color_frame.get_data())
                return True, color_image, None
        except Exception as e:
            print(f"Error reading RealSense frames: {e}")
            return False, np.zeros((480, 640, 3), dtype=np.uint8), None

    def stop(self):
        """Stop the RealSense pipeline."""
        if self.pipeline:
            try:
                self.pipeline.stop()
                # Give hardware a moment to power down cleanly (macOS/librealsense quirk)
                time.sleep(0.3)
            except Exception:
                pass
        self.pipeline = None
        self.align = None


class OpenCVCamera:
    """Wrapper for standard OpenCV camera (USB/built-in)."""
    
    def __init__(self, camera_index: int, resolution: Tuple[int, int] = (640, 480)):
        self.camera_index = camera_index
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
    
    def start(self):
        """Initialize and start the camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        width, height = self.resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def read(self) -> Tuple[bool, np.ndarray, Optional[rs.depth_frame]]:
        """Read a frame (no depth for OpenCV cameras)."""
        if not self.cap:
            return False, np.zeros((480, 640, 3), dtype=np.uint8), None
        
        ret, frame = self.cap.read()
        return ret, frame if ret else np.zeros((480, 640, 3), dtype=np.uint8), None
    
    def stop(self):
        """Release the camera."""
        if self.cap:
            self.cap.release()
        self.cap = None


def load_class_names_from_yaml(dataset_path: Path) -> Dict[str, str]:
    """Load class names from dataset's data.yaml file."""
    if not HAVE_YAML:
        return {}
    
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        print(f"Warning: data.yaml not found at {yaml_path}")
        return {}
    
    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        names = data.get("names", [])
        if isinstance(names, list):
            return {name: name for name in names}
        elif isinstance(names, dict):
            return {str(v): str(v) for v in names.values()}
        return {}
    except Exception as e:
        print(f"Error loading class names from YAML: {e}")
        return {}


def get_class_color(class_name: str, class_colors: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Get BGR color for a class name."""
    return class_colors.get(class_name, (128, 128, 128))  # Default gray


def calculate_bbox_median_depth(
    depth_frame: rs.depth_frame,
    bbox: Tuple[float, float, float, float],
    depth_scale: float
) -> Optional[float]:
    """Calculate median depth value within a bounding box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Ensure bbox is within frame bounds
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Extract depth values in bbox region
    depth_values = []
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            depth = depth_frame.get_distance(x, y)
            if depth > 0:  # Valid depth
                depth_values.append(depth)
    
    if not depth_values:
        return None
    
    # Return median depth in meters
    return float(np.median(depth_values))


def draw_overlay_stats(
    frame: np.ndarray,
    fps: float,
    total_detections: int,
    class_counts: Dict[str, int],
    model_name: str,
    camera_info: str,
    conf_threshold: float,
    resolution: Tuple[int, int],
    paused: bool = False,
    depth_overlay_enabled: bool = False,
    depth_overlay_alpha: float = 0.0,
) -> np.ndarray:
    """Draw statistics overlay on the frame."""
    overlay = frame.copy()
    
    # Semi-transparent background for text
    cv2.rectangle(overlay, (5, 5), (350, 220), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)
    y_offset = 25
    line_height = 25
    
    # Draw stats
    stats = [
        f"Model: {model_name}",
        f"Camera: {camera_info}",
        f"Resolution: {resolution[0]}x{resolution[1]}",
        f"FPS: {fps:.1f}",
        f"Conf Threshold: {conf_threshold:.2f}",
        (f"Depth Overlay: {'On' if depth_overlay_enabled else 'Off'}"
         + (f" (alpha={depth_overlay_alpha:.2f})" if depth_overlay_enabled else "")),
        f"Total Detections: {total_detections}",
        "",
        "Per-class counts:",
    ]
    
    for class_name, count in sorted(class_counts.items()):
        stats.append(f"  {class_name}: {count}")
    
    if paused:
        stats.insert(0, "*** PAUSED ***")
    
    for i, line in enumerate(stats):
        y = y_offset + i * line_height
        cv2.putText(frame, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Draw controls at bottom
    controls_y = frame.shape[0] - 75
    cv2.rectangle(frame, (5, controls_y - 5), (frame.shape[1] - 5, frame.shape[0] - 5), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.6, overlay, 0.4, 0, frame)
    
    control_text = [
        "Controls: [Q]uit | [S]ave | [P]ause | [+/-] Conf | [R]eset",
        "Overlay: [O] Toggle | [ ] Increase alpha | [ [ ] Decrease alpha",
    ]
    
    for i, line in enumerate(control_text):
        y = controls_y + 10 + i * 20
        cv2.putText(frame, line, (10, y), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame


def draw_detections(
    frame: np.ndarray,
    detections: List,
    bboxes: List[Tuple[float, float, float, float]],
    class_colors: Dict[str, Tuple[int, int, int]],
    depth_frame: Optional[rs.depth_frame] = None,
    depth_scale: float = 0.0
) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    for det, bbox in zip(detections, bboxes):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        class_name = det.clazz
        confidence = det.confidence
        
        # Get class-specific color
        color = get_class_color(class_name, class_colors)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{class_name} {confidence:.2f}"
        
        # Add depth if available
        if depth_frame is not None:
            depth = calculate_bbox_median_depth(depth_frame, bbox, depth_scale)
            if depth is not None:
                label += f" | {depth*1000:.0f}mm"
        
        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - baseline - 5), (x1 + text_w + 5, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return frame


def create_depth_colormap(depth_frame: rs.depth_frame) -> np.ndarray:
    """Create a colorized depth map for visualization."""
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET
    )
    return depth_colormap


def prompt_model_selection(models_dir: Path) -> Optional[Path]:
    """Prompt user to select a model from available .pt files."""
    pt_files = sorted(models_dir.glob("*.pt"))
    
    if not pt_files:
        print(f"No .pt model files found in {models_dir}")
        return None
    
    print("\n=== Available Models ===")
    for i, model_file in enumerate(pt_files, 1):
        print(f"{i}. {model_file.name}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(pt_files)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(pt_files):
                return pt_files[idx]
            else:
                print(f"Please enter a number between 1 and {len(pt_files)}")
        except (ValueError, KeyboardInterrupt):
            return None


def prompt_dataset_selection(datasets_dir: Path) -> Optional[Path]:
    """Prompt user to select a dataset folder for class names."""
    dataset_dirs = sorted([d for d in datasets_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    if not dataset_dirs:
        print(f"No dataset directories found in {datasets_dir}")
        return None
    
    print("\n=== Available Datasets ===")
    for i, dataset_dir in enumerate(dataset_dirs, 1):
        print(f"{i}. {dataset_dir.name}")
    
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(dataset_dirs)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(dataset_dirs):
                return dataset_dirs[idx]
            else:
                print(f"Please enter a number between 1 and {len(dataset_dirs)}")
        except (ValueError, KeyboardInterrupt):
            return None


def prompt_camera_selection() -> Tuple[Optional[str], Optional[int]]:
    """Prompt user to select camera type and index."""
    print("\n=== Camera Selection ===")
    realsense_status = " [Available]" if HAVE_REALSENSE else " [NOT AVAILABLE]"
    if HAVE_REALSENSE and HAVE_VERIFICATION:
        realsense_status += " + Auto-detect"
    print("1. RealSense Camera" + realsense_status)
    print("2. OpenCV Camera (USB/Built-in)")
    
    while True:
        try:
            choice = input("\nSelect camera type (1-2) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None, None
            
            if choice == '1':
                if not HAVE_REALSENSE:
                    print("RealSense not available. Please install pyrealsense2.")
                    continue
                return "realsense", 0
            elif choice == '2':
                cam_idx = input("Enter camera index (default 0): ").strip()
                cam_idx = int(cam_idx) if cam_idx else 0
                return "opencv", cam_idx
            else:
                print("Please enter 1 or 2")
        except (ValueError, KeyboardInterrupt):
            return None, None


def prompt_depth_mode(camera_type: str) -> bool:
    """Prompt user to enable depth overlay (only for RealSense)."""
    if camera_type != "realsense":
        return False
    
    print("\n=== Depth Mode ===")
    choice = input("Enable depth overlay? (y/n, default=y): ").strip().lower()
    return choice != 'n'


def load_previous_config() -> Optional[TestConfig]:
    """Load previous configuration if available."""
    cfg = load_config()
    ns = get_namespace(cfg, CONFIG_NAMESPACE)
    
    if not ns:
        return None
    
    try:
        # Parse preferred full profile if present
        pfp = ns.get("preferred_full_profile")
        preferred_full_profile = None
        try:
            if pfp and isinstance(pfp, dict) and "color" in pfp and "depth" in pfp:
                c = tuple(pfp.get("color", []))
                d = tuple(pfp.get("depth", []))
                if len(c) == 3 and len(d) == 3:
                    preferred_full_profile = (tuple(int(x) for x in c), tuple(int(x) for x in d))  # type: ignore
        except Exception:
            preferred_full_profile = None

        return TestConfig(
            model_path=ns.get("model_path", ""),
            dataset_path=ns.get("dataset_path", ""),
            camera_type=ns.get("camera_type", "opencv"),
            camera_index=ns.get("camera_index", 0),
            enable_depth=ns.get("enable_depth", False),
            conf_threshold=ns.get("conf_threshold", 0.25),
            iou_threshold=ns.get("iou_threshold", 0.45),
            resolution=tuple(ns.get("resolution", [640, 480])),
            fps=ns.get("fps", 30),
            depth_overlay_enabled=ns.get("depth_overlay_enabled", False),
            depth_overlay_alpha=float(ns.get("depth_overlay_alpha", 0.0)),
            preferred_color_fps=ns.get("preferred_color_fps"),
            preferred_depth_fps=ns.get("preferred_depth_fps"),
            preferred_full_profile=preferred_full_profile,
        )
    except Exception:
        return None


def save_current_config(test_config: TestConfig):
    """Save current configuration for future runs."""
    cfg = load_config()
    update_namespace(cfg, CONFIG_NAMESPACE, {
        "model_path": test_config.model_path,
        "dataset_path": test_config.dataset_path,
        "camera_type": test_config.camera_type,
        "camera_index": test_config.camera_index,
        "enable_depth": test_config.enable_depth,
        "conf_threshold": test_config.conf_threshold,
        "iou_threshold": test_config.iou_threshold,
        "resolution": list(test_config.resolution),
        "fps": test_config.fps,
        "depth_overlay_enabled": test_config.depth_overlay_enabled,
        "depth_overlay_alpha": float(test_config.depth_overlay_alpha),
        "preferred_color_fps": test_config.preferred_color_fps,
        "preferred_depth_fps": test_config.preferred_depth_fps,
        "preferred_full_profile": (
            {
                "color": list(test_config.preferred_full_profile[0]),
                "depth": list(test_config.preferred_full_profile[1]),
            }
            if test_config.preferred_full_profile else None
        ),
    })


def interactive_setup() -> Optional[TestConfig]:
    """Interactive configuration setup with option to use previous settings."""
    models_dir = REPO_ROOT / "pickafresa_vision" / "models"
    datasets_dir = REPO_ROOT / "pickafresa_vision" / "datasets"
    
    # Load defaults from objd_config.yaml
    objd_defaults = _load_objd_config_defaults()
    
    # Check for previous config
    prev_config = load_previous_config()
    if prev_config and Path(prev_config.model_path).exists():
        print("\n=== Previous Configuration Found ===")
        print(f"Model: {Path(prev_config.model_path).name}")
        print(f"Dataset: {Path(prev_config.dataset_path).name}")
        print(f"Camera: {prev_config.camera_type} (index {prev_config.camera_index})")
        print(f"Depth: {'Enabled' if prev_config.enable_depth else 'Disabled'}")
        print(f"Confidence: {prev_config.conf_threshold}")
        
        use_prev = input("\nUse previous configuration? (y/n, default=y): ").strip().lower()
        if use_prev != 'n':
            return prev_config
    
    # New configuration
    print("\n=== New Configuration ===")
    
    # Model selection (with default from objd_config.yaml)
    default_model = objd_defaults.get("model_path")
    if default_model:
        default_model_path = REPO_ROOT / default_model
        if default_model_path.exists():
            print(f"\nDefault model from config: {default_model_path.name}")
            use_default = input("Use this model? (y/n, default=y): ").strip().lower()
            if use_default != 'n':
                model_path = default_model_path
            else:
                model_path = prompt_model_selection(models_dir)
        else:
            model_path = prompt_model_selection(models_dir)
    else:
        model_path = prompt_model_selection(models_dir)
    
    if model_path is None:
        return None
    
    # Dataset selection
    dataset_path = prompt_dataset_selection(datasets_dir)
    if dataset_path is None:
        return None
    
    # Camera selection
    camera_type, camera_index = prompt_camera_selection()
    if camera_type is None:
        return None
    
    # Depth mode
    enable_depth = prompt_depth_mode(camera_type)
    
    # Get default inference parameters from config
    inference_defaults = objd_defaults.get("inference", {})
    default_conf = inference_defaults.get("confidence", 0.25)
    default_iou = inference_defaults.get("iou", 0.45)
    
    # Get camera defaults from config
    camera_defaults = objd_defaults.get("camera", {})
    default_resolution = tuple(camera_defaults.get("resolution", [640, 480]))
    default_fps = camera_defaults.get("fps", 30)
    
    return TestConfig(
        model_path=str(model_path),
        dataset_path=str(dataset_path),
        camera_type=camera_type,
        camera_index=camera_index,
        enable_depth=enable_depth,
        conf_threshold=default_conf,
        iou_threshold=default_iou,
        resolution=default_resolution,
        fps=default_fps,
    )


def main():
    """Main testing loop."""
    print("=" * 60)
    print("Object Detection Model Testing Tool")
    print("Team YEA, 2025")
    print("=" * 60)
    
    # Setup configuration
    config = interactive_setup()
    if config is None:
        print("\nSetup cancelled.")
        return
    
    # Save configuration
    save_current_config(config)
    
    # Load model
    print(f"\nLoading model: {Path(config.model_path).name}")
    try:
        model = load_model(config.model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load class names and colors
    class_names = load_class_names_from_yaml(Path(config.dataset_path))
    class_colors = DEFAULT_CLASS_COLORS.copy()
    
    # Initialize camera
    print(f"\nInitializing camera: {config.camera_type}")
    try:
        if config.camera_type == "realsense":
            # Enable auto-detection for RealSense cameras
                camera = RealSenseCamera(
                    resolution=config.resolution, 
                    fps=config.fps, 
                    auto_detect=True,
                    enable_depth=config.enable_depth
                )
        else:
            camera = OpenCVCamera(camera_index=config.camera_index, resolution=config.resolution)
        
        camera.start()
        print("Camera initialized successfully!")
        
        # Update config with actual resolution/fps (may have changed with auto-detection)
        if config.camera_type == "realsense":
            config.resolution = camera.resolution
            config.fps = camera.fps
            # Persist user fps preferences globally for RealSense runs
            config.preferred_color_fps = getattr(camera, 'fps', None)
            config.preferred_depth_fps = getattr(camera, 'depth_fps', None) if config.enable_depth else None
            # If a full profile pair was selected, persist it in the config object too
            if hasattr(camera, 'selected_full_profile') and camera.selected_full_profile:
                config.preferred_full_profile = camera.selected_full_profile
            save_current_config(config)
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return
    
    # Initialize FPS counter
    fps_counter = FPSCounter()
    
    # State variables
    paused = False
    frame_count = 0
    
    # Main loop
    print("\n" + "=" * 60)
    print("Starting real-time inference...")
    print("Press 'q' to quit, 's' to save frame, 'p' to pause")
    print("=" * 60 + "\n")
    
    window_name_color = "YOLOv11 - Color (Detections)"
    window_name_depth = "YOLOv11 - Depth"
    cv2.namedWindow(window_name_color, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_depth, cv2.WINDOW_NORMAL)
    
    # Track last displayed frames for paused mode
    last_color_frame: Optional[np.ndarray] = None
    last_depth_frame: Optional[np.ndarray] = None
    last_fps: float = 0.0
    last_detections: List = []
    last_bboxes: List[Tuple[float, float, float, float]] = []
    last_class_counts: Dict[str, int] = {}
    camera_info = f"{config.camera_type}:{config.camera_index}" + (" (depth)" if config.enable_depth else "")

    try:
        while True:
            # Read frame
            ret, frame, depth_frame = camera.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Prepare visualization frames
            vis_frame_color = None
            vis_frame_depth = None

            if not paused:
                # Run inference
                detections, bboxes = infer(
                    model,
                    frame,
                    conf=config.conf_threshold,
                    iou=config.iou_threshold,
                    bbox_format="xyxy",
                    normalized=False,
                )
                
                # Count detections per class
                class_counts = {}
                for det in detections:
                    class_counts[det.clazz] = class_counts.get(det.clazz, 0) + 1
                
                # Update FPS
                fps = fps_counter.update()
                
                # Draw detections
                vis_frame_color = frame.copy()
                if config.enable_depth and depth_frame is not None:
                    depth_scale = camera.depth_scale if hasattr(camera, 'depth_scale') else 0.001
                    vis_frame_color = draw_detections(vis_frame_color, detections, bboxes, class_colors, depth_frame, depth_scale)
                else:
                    vis_frame_color = draw_detections(vis_frame_color, detections, bboxes, class_colors)
                
                # Optional depth overlay blended into color window
                if config.enable_depth and depth_frame is not None and config.depth_overlay_enabled and config.depth_overlay_alpha > 0.0:
                    depth_colormap = create_depth_colormap(depth_frame)
                    if depth_colormap.shape[:2] != vis_frame_color.shape[:2]:
                        depth_colormap = cv2.resize(depth_colormap, (vis_frame_color.shape[1], vis_frame_color.shape[0]))
                    alpha = max(0.0, min(1.0, float(config.depth_overlay_alpha)))
                    vis_frame_color = cv2.addWeighted(vis_frame_color, 1.0 - alpha, depth_colormap, alpha, 0)

                # Draw stats overlay on color window
                vis_frame_color = draw_overlay_stats(
                    vis_frame_color,
                    fps,
                    len(detections),
                    class_counts,
                    Path(config.model_path).name,
                    camera_info,
                    config.conf_threshold,
                    config.resolution,
                    paused,
                    depth_overlay_enabled=config.depth_overlay_enabled and config.enable_depth,
                    depth_overlay_alpha=float(config.depth_overlay_alpha),
                )
                
                # Prepare depth-only window image (if depth enabled)
                if config.enable_depth and depth_frame is not None:
                    vis_frame_depth = create_depth_colormap(depth_frame)
                    # Keep depth window size similar to color window height
                    if vis_frame_depth.shape[0] != vis_frame_color.shape[0]:
                        scale_w = int(vis_frame_depth.shape[1] * (vis_frame_color.shape[0] / vis_frame_depth.shape[0]))
                        vis_frame_depth = cv2.resize(vis_frame_depth, (scale_w, vis_frame_color.shape[0]))
                
                frame_count += 1

                # Keep last states for pause display
                last_color_frame = vis_frame_color.copy() if vis_frame_color is not None else frame.copy()
                last_depth_frame = vis_frame_depth.copy() if vis_frame_depth is not None else None
                last_fps = fps
                last_detections = detections
                last_bboxes = bboxes
                last_class_counts = class_counts
            else:
                # Paused - show the last frames with pause indicator
                if last_color_frame is None:
                    last_color_frame = frame.copy()
                vis_frame_color = draw_overlay_stats(
                    last_color_frame.copy(),
                    last_fps,
                    len(last_detections),
                    last_class_counts,
                    Path(config.model_path).name,
                    camera_info,
                    config.conf_threshold,
                    config.resolution,
                    paused=True,
                    depth_overlay_enabled=config.depth_overlay_enabled and config.enable_depth,
                    depth_overlay_alpha=float(config.depth_overlay_alpha),
                )
                vis_frame_depth = last_depth_frame.copy() if last_depth_frame is not None else None
            
            # Display frames in separate windows
            if vis_frame_color is not None:
                cv2.imshow(window_name_color, vis_frame_color)
            if config.enable_depth and vis_frame_depth is not None:
                cv2.imshow(window_name_depth, vis_frame_depth)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"objd_test_{timestamp}.jpg"
                save_path = REPO_ROOT / "pickafresa_vision" / "images" / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                # Save color window image
                to_save = vis_frame_color if vis_frame_color is not None else frame
                cv2.imwrite(str(save_path), to_save)
                print(f"Frame saved: {filename}")
            elif key == ord('p'):
                # Toggle pause
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('+') or key == ord('='):
                # Increase confidence threshold
                config.conf_threshold = min(0.95, config.conf_threshold + 0.05)
                print(f"Confidence threshold: {config.conf_threshold:.2f}")
                save_current_config(config)
            elif key == ord('-') or key == ord('_'):
                # Decrease confidence threshold
                config.conf_threshold = max(0.05, config.conf_threshold - 0.05)
                print(f"Confidence threshold: {config.conf_threshold:.2f}")
                save_current_config(config)
            elif key == ord('r'):
                # Reset to defaults
                config.conf_threshold = 0.25
                config.iou_threshold = 0.45
                print("Reset to default thresholds")
                save_current_config(config)
            elif key == ord('o'):
                # Toggle depth overlay on color window
                if config.enable_depth:
                    config.depth_overlay_enabled = not config.depth_overlay_enabled
                    state = 'enabled' if config.depth_overlay_enabled else 'disabled'
                    print(f"Depth overlay {state}")
                    save_current_config(config)
            elif key == ord(']'):
                # Increase overlay alpha
                if config.enable_depth and config.depth_overlay_enabled:
                    config.depth_overlay_alpha = min(1.0, float(config.depth_overlay_alpha) + 0.05)
                    print(f"Overlay alpha: {config.depth_overlay_alpha:.2f}")
                    save_current_config(config)
            elif key == ord('['):
                # Decrease overlay alpha
                if config.enable_depth and config.depth_overlay_enabled:
                    config.depth_overlay_alpha = max(0.0, float(config.depth_overlay_alpha) - 0.05)
                    print(f"Overlay alpha: {config.depth_overlay_alpha:.2f}")
                    save_current_config(config)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")
        print("Done!")


if __name__ == "__main__":
    main()
