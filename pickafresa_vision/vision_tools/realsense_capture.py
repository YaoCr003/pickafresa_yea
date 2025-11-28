"""
RealSense D435 Camera Capture Module

Provides robust frame capture API for Intel RealSense D400 series cameras with:
- Automatic device detection and initialization
- Profile verification and caching for reliable startup
- Depth-to-color alignment
- Intrinsics extraction from both PyRealSense SDK and YAML calibration files
- macOS-specific workarounds for stability
- Error recovery and graceful degradation

Key Features:
- Single-frame capture API optimized for frame-by-frame processing
- Validates intrinsics from multiple sources
- Handles common RealSense issues (segfaults, device locks, bandwidth errors)
- Configuration-driven with sensible defaults

Usage:
    from pickafresa_vision.vision_tools.realsense_capture import RealSenseCapture
    
    # Initialize camera
    camera = RealSenseCapture()
    camera.start()
    
    # Capture a single aligned frame
    color_frame, depth_frame = camera.capture_frame()
    
    # Get camera intrinsics
    intrinsics = camera.get_intrinsics(source="auto")
    
    # Save raw image
    camera.save_raw_image(color_frame, "path/to/output.png")
    
    # Cleanup
    camera.stop()

# @aldrick-t, 2025
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False
    from typing import Any as rs  # type: ignore

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False
    yaml = None  # type: ignore

# Repository root for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import verification tools if available
REALSENSE_TOOLS_PATH = REPO_ROOT / "pickafresa_vision" / "realsense_tools"
if str(REALSENSE_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(REALSENSE_TOOLS_PATH))

try:
    from realsense_verify_color import get_camera_serial  # type: ignore
    from realsense_verify_full import (  # type: ignore
        get_best_full_profile,
        load_working_profiles,
        save_working_profiles,
    )
    HAVE_VERIFICATION_TOOLS = True
except ImportError:
    HAVE_VERIFICATION_TOOLS = False


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int
    distortion_model: str = "none"
    distortion_coeffs: Optional[np.ndarray] = None
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 camera matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "width": int(self.width),
            "height": int(self.height),
            "distortion_model": self.distortion_model,
            "distortion_coeffs": self.distortion_coeffs.tolist() if self.distortion_coeffs is not None else None,
        }


class RealSenseCaptureError(Exception):
    """Base exception for RealSense capture errors."""
    pass


class RealSenseCapture:
    """
    RealSense D435 camera capture manager.
    
    Handles device initialization, frame capture, and intrinsics management
    with robust error handling and macOS-specific workarounds.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize RealSense capture manager.
        
        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        if not HAVE_REALSENSE:
            raise RealSenseCaptureError(
                "pyrealsense2 is required. Install with: pip install pyrealsense2"
            )
        
        # Load configuration
        if config_path is None:
            config_path = REPO_ROOT / "pickafresa_vision" / "configs" / "realsense_capture_config.yaml"
        self.config = self._load_config(config_path)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Camera state
        self.pipeline: Any = None  # rs.pipeline when available
        self.pipeline_profile: Any = None  # rs.pipeline_profile when available
        self.align: Any = None  # rs.align when available
        self.device_serial: Optional[str] = None
        self.is_running = False
        
        # Frame statistics
        self.consecutive_failures = 0
        self.total_frames_captured = 0
        
        # Intrinsics cache
        self._cached_intrinsics_rs: Optional[CameraIntrinsics] = None
        self._cached_intrinsics_yaml: Optional[CameraIntrinsics] = None
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not HAVE_YAML or not config_path.exists():
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "camera": {"auto_detect": True, "device_timeout_ms": 5000},
            "streams": {
                "color": {"width": 640, "height": 480, "fps": 30, "format": "RGB8"},
                "depth": {"width": 640, "height": 480, "fps": 30, "format": "Z16"},
            },
            "alignment": {"align_to_color": True},
            "profile_verification": {"use_cached_profiles": True, "max_startup_retries": 3},
            "initialization": {"cleanup_existing_contexts": True, "stabilization_frames": 5},
            "capture": {"timeout_ms": 10000, "max_frame_retries": 10},
            "intrinsics": {"source": "auto", "validate_yaml_match": True},
            "error_handling": {"auto_recover": True, "max_consecutive_failures": 5},
            "macos_workarounds": {"enabled": True, "extra_init_delay_ms": 500},
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for capture module."""
        logger = logging.getLogger("realsense_capture")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            
            # File handler if logging enabled
            if self.config.get("logging", {}).get("enabled", True):
                log_dir = REPO_ROOT / "pickafresa_vision" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / self.config.get("logging", {}).get("log_file", "realsense_capture.log")
                
                fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
        
        return logger
    
    def _cleanup_existing_contexts(self) -> None:
        """Force cleanup of existing RealSense contexts (macOS workaround)."""
        if not self.config.get("initialization", {}).get("cleanup_existing_contexts", True):
            return
        
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                try:
                    dev.hardware_reset()
                    self.logger.debug(f"Reset device: {dev.get_info(rs.camera_info.serial_number)}")
                except Exception as e:
                    self.logger.debug(f"Could not reset device: {e}")
            
            # macOS-specific: extra delay after reset
            if self.config.get("macos_workarounds", {}).get("enabled", True):
                delay_ms = self.config.get("macos_workarounds", {}).get("extra_init_delay_ms", 500)
                time.sleep(delay_ms / 1000.0)
        except Exception as e:
            self.logger.debug(f"Context cleanup failed: {e}")
    
    def _get_device_serial(self) -> Optional[str]:
        """Get RealSense device serial number."""
        if HAVE_VERIFICATION_TOOLS:
            try:
                return get_camera_serial()
            except Exception as e:
                self.logger.warning(f"Could not get camera serial via verification tools: {e}")
        
        # Fallback: manual detection
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                return None
            return devices[0].get_info(rs.camera_info.serial_number)
        except Exception as e:
            self.logger.error(f"Failed to get device serial: {e}")
            return None
    
    def _select_profile(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Select best color and depth stream profiles.
        
        Returns:
            ((color_w, color_h, color_fps), (depth_w, depth_h, depth_fps))
        """
        # Check for cached profiles
        if self.config.get("profile_verification", {}).get("use_cached_profiles", True):
            if HAVE_VERIFICATION_TOOLS and self.device_serial:
                cached = load_working_profiles(self.device_serial)
                if cached and len(cached) > 0:
                    self.logger.info(f"Using cached profile for device {self.device_serial}")
                    return cached[0]  # Use first cached profile
        
        # Use config defaults
        color_cfg = self.config.get("streams", {}).get("color", {})
        depth_cfg = self.config.get("streams", {}).get("depth", {})
        
        color_profile = (
            color_cfg.get("width", 640),
            color_cfg.get("height", 480),
            color_cfg.get("fps", 30)
        )
        depth_profile = (
            depth_cfg.get("width", 640),
            depth_cfg.get("height", 480),
            depth_cfg.get("fps", 30)
        )
        
        return color_profile, depth_profile
    
    def start(self) -> None:
        """
        Start RealSense camera pipeline.
        
        Raises:
            RealSenseCaptureError: If initialization fails after retries.
        """
        if self.is_running:
            self.logger.warning("Camera is already running")
            return
        
        max_retries = self.config.get("profile_verification", {}).get("max_startup_retries", 3)
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Starting RealSense camera (attempt {attempt + 1}/{max_retries})...")
                
                # Cleanup existing contexts
                self._cleanup_existing_contexts()
                
                # Get device serial
                self.device_serial = self._get_device_serial()
                if self.device_serial:
                    self.logger.info(f"Detected device: {self.device_serial}")
                
                # Select profiles
                color_profile, depth_profile = self._select_profile()
                self.logger.info(f"Color: {color_profile[0]}x{color_profile[1]}@{color_profile[2]}fps")
                self.logger.info(f"Depth: {depth_profile[0]}x{depth_profile[1]}@{depth_profile[2]}fps")
                
                # Configure pipeline
                self.pipeline = rs.pipeline()
                config = rs.config()
                
                if self.device_serial:
                    config.enable_device(self.device_serial)
                
                # Enable streams
                config.enable_stream(
                    rs.stream.color,
                    color_profile[0], color_profile[1],
                    rs.format.rgb8,  # Always use RGB8 internally
                    color_profile[2]
                )
                config.enable_stream(
                    rs.stream.depth,
                    depth_profile[0], depth_profile[1],
                    rs.format.z16,
                    depth_profile[2]
                )
                
                # Start pipeline
                self.pipeline_profile = self.pipeline.start(config)
                
                # Setup alignment
                if self.config.get("alignment", {}).get("align_to_color", True):
                    self.align = rs.align(rs.stream.color)
                    self.logger.info("Depth alignment to color enabled")
                
                # Stabilization: discard first N frames
                stabilization_frames = self.config.get("initialization", {}).get("stabilization_frames", 5)
                for i in range(stabilization_frames):
                    self.pipeline.wait_for_frames(timeout_ms=10000)
                self.logger.info(f"Discarded {stabilization_frames} stabilization frames")
                
                self.is_running = True
                self.consecutive_failures = 0
                self.logger.info("[OK] RealSense camera started successfully")
                return
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                self.stop()  # Cleanup before retry
                if attempt < max_retries - 1:
                    retry_delay = self.config.get("profile_verification", {}).get("retry_delay_seconds", 1.0)
                    time.sleep(retry_delay)
        
        raise RealSenseCaptureError(f"Failed to start camera after {max_retries} attempts")
    
    def stop(self) -> None:
        """Stop RealSense camera pipeline."""
        if self.pipeline:
            try:
                self.pipeline.stop()
                self.logger.info("Camera pipeline stopped")
            except Exception as e:
                self.logger.debug(f"Error stopping pipeline: {e}")
            finally:
                self.pipeline = None
                self.pipeline_profile = None
        
        self.is_running = False
    
    def capture_frame(self) -> Tuple[np.ndarray, Any]:
        """
        Capture a single aligned color and depth frame.
        
        Returns:
            (color_image, depth_frame): 
                - color_image: RGB numpy array (H, W, 3)
                - depth_frame: RealSense depth_frame object
        
        Raises:
            RealSenseCaptureError: If frame capture fails.
        """
        if not self.is_running:
            raise RealSenseCaptureError("Camera is not running. Call start() first.")
        
        max_retries = self.config.get("capture", {}).get("max_frame_retries", 10)
        timeout_ms = self.config.get("capture", {}).get("timeout_ms", 2000)
        
        for attempt in range(max_retries):
            try:
                # Wait for frames
                frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
                
                # Align if configured
                if self.align:
                    frames = self.align.process(frames)
                
                # Get color and depth frames
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    raise RealSenseCaptureError("Failed to get valid frames")
                
                # Convert color frame to numpy array (RGB)
                color_image = np.asanyarray(color_frame.get_data())
                
                # Success
                self.consecutive_failures = 0
                self.total_frames_captured += 1
                return color_image, depth_frame
                
            except Exception as e:
                self.logger.warning(f"Frame capture attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    backoff_ms = self.config.get("capture", {}).get("retry_backoff_ms", 100)
                    time.sleep(backoff_ms / 1000.0)
        
        # All retries failed
        self.consecutive_failures += 1
        
        # Check if we should attempt recovery
        if self.config.get("error_handling", {}).get("auto_recover", True):
            max_failures = self.config.get("error_handling", {}).get("max_consecutive_failures", 5)
            if self.consecutive_failures >= max_failures:
                self.logger.error("Too many consecutive failures, attempting recovery...")
                self._attempt_recovery()
        
        raise RealSenseCaptureError("Failed to capture frame after all retries")
    
    def _attempt_recovery(self) -> None:
        """Attempt to recover from repeated failures."""
        self.logger.warning("Attempting camera recovery...")
        try:
            self.stop()
            time.sleep(1.0)
            self.start()
            self.consecutive_failures = 0
            self.logger.info("Recovery successful")
        except Exception as e:
            self.logger.error(f"Recovery failed: {e}")
            raise RealSenseCaptureError("Camera recovery failed")
    
    def get_intrinsics(self, source: str = "auto") -> CameraIntrinsics:
        """
        Get camera intrinsics from specified source.
        
        Args:
            source: "auto", "realsense", or "yaml"
                - auto: Try YAML first, fallback to RealSense
                - realsense: Use RealSense SDK intrinsics
                - yaml: Use YAML calibration file
        
        Returns:
            CameraIntrinsics object
        
        Raises:
            RealSenseCaptureError: If intrinsics cannot be obtained.
        """
        if source == "realsense":
            return self._get_intrinsics_realsense()
        elif source == "yaml":
            return self._get_intrinsics_yaml()
        elif source == "auto":
            # Try YAML first
            try:
                yaml_intrinsics = self._get_intrinsics_yaml()
                
                # Validate against RealSense if configured
                if self.config.get("intrinsics", {}).get("validate_yaml_match", True):
                    rs_intrinsics = self._get_intrinsics_realsense()
                    self._validate_intrinsics_match(yaml_intrinsics, rs_intrinsics)
                
                return yaml_intrinsics
            except Exception as e:
                self.logger.warning(f"YAML intrinsics failed: {e}, falling back to RealSense")
                return self._get_intrinsics_realsense()
        else:
            raise ValueError(f"Invalid intrinsics source: {source}")
    
    def _get_intrinsics_realsense(self) -> CameraIntrinsics:
        """Extract intrinsics from RealSense SDK."""
        if self._cached_intrinsics_rs is not None:
            return self._cached_intrinsics_rs
        
        if not self.is_running or not self.pipeline_profile:
            raise RealSenseCaptureError("Cannot get intrinsics: camera not running")
        
        try:
            # Get color stream profile
            color_stream = self.pipeline_profile.get_stream(rs.stream.color)
            color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            # Extract parameters
            intrinsics = CameraIntrinsics(
                fx=color_intrinsics.fx,
                fy=color_intrinsics.fy,
                cx=color_intrinsics.ppx,
                cy=color_intrinsics.ppy,
                width=color_intrinsics.width,
                height=color_intrinsics.height,
                distortion_model=str(color_intrinsics.model),
                distortion_coeffs=np.array(color_intrinsics.coeffs, dtype=np.float32)
            )
            
            self._cached_intrinsics_rs = intrinsics
            self.logger.info("Loaded intrinsics from RealSense SDK")
            return intrinsics
            
        except Exception as e:
            raise RealSenseCaptureError(f"Failed to get RealSense intrinsics: {e}")
    
    def _get_intrinsics_yaml(self) -> CameraIntrinsics:
        """Load intrinsics from YAML calibration file."""
        if self._cached_intrinsics_yaml is not None:
            return self._cached_intrinsics_yaml
        
        if not HAVE_YAML:
            raise RealSenseCaptureError("PyYAML required for YAML intrinsics")
        
        # Find calibration file
        calib_dir = REPO_ROOT / "pickafresa_vision" / self.config.get("intrinsics", {}).get("calibration_directory", "camera_calibration")
        pattern = self.config.get("intrinsics", {}).get("calibration_file_pattern", "calib*.yaml")
        
        calib_files = sorted(calib_dir.glob(pattern))
        if not calib_files:
            raise RealSenseCaptureError(f"No calibration files found in {calib_dir}")
        
        # Use most recent
        calib_file = calib_files[-1]
        self.logger.info(f"Loading intrinsics from: {calib_file.name}")
        
        try:
            with open(calib_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Extract camera matrix
            cm = data.get("camera_matrix", {})
            if isinstance(cm, dict) and "data" in cm:
                cm_data = cm["data"]
                camera_matrix = np.array(cm_data, dtype=np.float32).reshape(3, 3)
            else:
                raise ValueError("Invalid camera_matrix format")
            
            # Extract distortion coefficients
            dc = data.get("dist_coeffs", {})
            if isinstance(dc, dict) and "data" in dc:
                dist_coeffs = np.array(dc["data"], dtype=np.float32)
            else:
                dist_coeffs = np.zeros(5, dtype=np.float32)
            
            # Get image dimensions
            width = data.get("image_width", 640)
            height = data.get("image_height", 480)
            
            intrinsics = CameraIntrinsics(
                fx=camera_matrix[0, 0],
                fy=camera_matrix[1, 1],
                cx=camera_matrix[0, 2],
                cy=camera_matrix[1, 2],
                width=width,
                height=height,
                distortion_model=data.get("distortion_model", "plumb_bob"),
                distortion_coeffs=dist_coeffs
            )
            
            self._cached_intrinsics_yaml = intrinsics
            return intrinsics
            
        except Exception as e:
            raise RealSenseCaptureError(f"Failed to load YAML intrinsics: {e}")
    
    def _validate_intrinsics_match(self, yaml_intr: CameraIntrinsics, rs_intr: CameraIntrinsics) -> None:
        """Validate that YAML and RealSense intrinsics match within tolerance."""
        tolerance = self.config.get("intrinsics", {}).get("match_tolerance_pixels", 5.0)
        
        diffs = {
            "fx": abs(yaml_intr.fx - rs_intr.fx),
            "fy": abs(yaml_intr.fy - rs_intr.fy),
            "cx": abs(yaml_intr.cx - rs_intr.cx),
            "cy": abs(yaml_intr.cy - rs_intr.cy),
        }
        
        mismatches = {k: v for k, v in diffs.items() if v > tolerance}
        
        if mismatches:
            self.logger.warning(f"Intrinsics mismatch detected: {mismatches}")
            self.logger.warning("YAML intrinsics may be outdated. Consider recalibration.")
        else:
            self.logger.info("[OK] YAML and RealSense intrinsics match within tolerance")
    
    def save_raw_image(self, color_image: np.ndarray, output_path: Path) -> None:
        """
        Save raw color image to file.
        
        Args:
            color_image: RGB numpy array
            output_path: Path to save image
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr_image)
        self.logger.debug(f"Saved raw image: {output_path}")
    
    def __enter__(self) -> "RealSenseCapture":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


# Convenience function for quick single-frame capture
def capture_single_frame(
    config_path: Optional[Path] = None
) -> Tuple[np.ndarray, Any, CameraIntrinsics]:
    """
    Capture a single frame with automatic camera management.
    
    Args:
        config_path: Optional path to configuration file
    
    Returns:
        (color_image, depth_frame, intrinsics)
    """
    with RealSenseCapture(config_path=config_path) as camera:
        color_image, depth_frame = camera.capture_frame()
        intrinsics = camera.get_intrinsics(source="auto")
    
    return color_image, depth_frame, intrinsics


if __name__ == "__main__":
    # Quick test
    print("Testing RealSense capture...")
    try:
        color, depth, intr = capture_single_frame()
        print(f"[OK] Captured frame: {color.shape}")
        print(f"[OK] Intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.cx:.2f}, cy={intr.cy:.2f}")
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
