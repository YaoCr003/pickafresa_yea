"""
Perspective-n-Point (PnP) Pose Estimation for Strawberry Detection

This module provides a robust API for estimating 6DOF poses of detected strawberries
using bounding box corners, RealSense depth data, and OpenCV's PnP solver.

Key Pipeline:
1. Take RGB+Depth aligned frame (from realsense_capture)
2. Get detections with bounding boxes (from inference_bbox in cxcywh format)
3. Sample depth at bbox using configurable strategy (closest/center/offset variants)
4. Validate depth consistency across corners
5. Solve PnP using known strawberry dimensions and bbox corners
6. Return 4x4 homogeneous transformation matrix (Camera → Fruit frame)

Features:
- Multi-detection processing (sorted by confidence)
- Configurable class filtering (default: "ripe" only)
- Configurable depth output strategies:
  * "closest": Minimum depth from ROI (finds fruit surface)
  * "center": Depth at ROI center
  * "closest_offset": Minimum depth + 0.5× strawberry width (deeper)
  * "center_offset": Center depth + 0.4× strawberry width (deeper)
- Adaptive depth variance thresholds based on distance
- Comprehensive error reporting with failure reasons
- JSON-serializable results for data logging

Usage:
    from pickafresa_vision.vision_nodes.pnp_calc import FruitPoseEstimator
    
    # Initialize estimator
    estimator = FruitPoseEstimator()
    
    # Estimate poses for all detections
    results = estimator.estimate_poses(
        color_image=color_img,
        depth_frame=depth_frame,
        detections=detections,
        bboxes_cxcywh=bboxes,
        camera_matrix=intrinsics.to_matrix(),
        dist_coeffs=intrinsics.distortion_coeffs
    )
    
    # Process results
    for result in results:
        if result.success:
            print(f"Fruit at {result.position_cam} with confidence {result.confidence}")
            print(f"Transform:\n{result.T_cam_fruit}")

# @aldrick-t, 2025
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Setup logging
LOG_DIR = REPO_ROOT / "pickafresa_vision" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "pnp_calc.log"

# Log to file only, not console (to avoid spam during preview)
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a')
    ],
    force=True
)
logger = logging.getLogger(__name__)


@dataclass
class PoseEstimationResult:
    """Result of PnP pose estimation for a single detection.
    
    Attributes:
        bbox_cxcywh: Bounding box in (center_x, center_y, width, height) format
        confidence: Detection confidence [0, 1]
        class_name: Detected class label
        class_id: Numeric class ID
        success: Whether PnP estimation succeeded
        T_cam_fruit: 4x4 homogeneous transformation matrix (Camera → Fruit frame)
        position_cam: 3D position in camera frame [x, y, z] meters (from T_cam_fruit)
        rotation_matrix: 3x3 rotation matrix (from T_cam_fruit)
        rvec: Rotation vector from PnP solver (3,)
        tvec: Translation vector from PnP solver (3,)
        error_reason: Reason for failure if success=False
        depth_samples: Depth values at bbox corners [TL, TR, BR, BL] in meters
        median_depth: Median depth across corners
        depth_variance: Maximum depth variance ratio
        sampling_strategy: Strategy used for depth sampling (corner/inset/center/bbox_median)
    """
    bbox_cxcywh: Tuple[float, float, float, float]
    confidence: float
    class_name: str
    class_id: int
    success: bool
    T_cam_fruit: Optional[np.ndarray] = None
    position_cam: Optional[np.ndarray] = None
    rotation_matrix: Optional[np.ndarray] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    error_reason: Optional[str] = None
    depth_samples: Optional[List[float]] = None
    median_depth: Optional[float] = None
    depth_variance: Optional[float] = None
    sampling_strategy: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "bbox_cxcywh": [float(v) for v in self.bbox_cxcywh],
            "confidence": float(self.confidence),
            "class_name": self.class_name,
            "class_id": int(self.class_id),
            "success": bool(self.success),
            "T_cam_fruit": self.T_cam_fruit.tolist() if self.T_cam_fruit is not None else None,
            "position_cam": self.position_cam.tolist() if self.position_cam is not None else None,
            "rotation_matrix": self.rotation_matrix.tolist() if self.rotation_matrix is not None else None,
            "rvec": self.rvec.tolist() if self.rvec is not None else None,
            "tvec": self.tvec.tolist() if self.tvec is not None else None,
            "error_reason": self.error_reason,
            "depth_samples": [float(d) for d in self.depth_samples] if self.depth_samples is not None else None,
            "median_depth": float(self.median_depth) if self.median_depth is not None else None,
            "depth_variance": float(self.depth_variance) if self.depth_variance is not None else None,
            "sampling_strategy": self.sampling_strategy,
        }


class DepthSampler:
    """Samples depth values with median filtering for robustness."""
    
    def __init__(self, window_size: int = 5, min_valid_samples: int = 3):
        """
        Initialize depth sampler.
        
        Args:
            window_size: Size of sampling window (must be odd)
            min_valid_samples: Minimum number of valid readings required
        """
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd number")
        
        self.window_size = window_size
        self.min_valid_samples = max(1, min_valid_samples)
    
    def sample(self, depth_frame: Any, pixel: Tuple[int, int]) -> Optional[float]:
        """
        Sample depth at a pixel location using median filtering.
        
        Args:
            depth_frame: RealSense depth_frame object
            pixel: (x, y) pixel coordinates
        
        Returns:
            Median depth in meters, or None if insufficient valid samples
        """
        if not HAVE_REALSENSE:
            raise RuntimeError("pyrealsense2 required for depth sampling")
        
        cx, cy = pixel
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        half = self.window_size // 2
        
        values: List[float] = []
        
        for yy in range(max(0, cy - half), min(height, cy + half + 1)):
            for xx in range(max(0, cx - half), min(width, cx + half + 1)):
                depth = depth_frame.get_distance(xx, yy)
                if depth > 0:  # Valid depth
                    values.append(depth)
        
        if len(values) < self.min_valid_samples:
            return None
        
        return float(np.median(values))


class FruitPoseEstimator:
    """
    Estimate 6DOF poses of detected strawberries using PnP.
    
    This class provides the main API for pose estimation from detections.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize pose estimator.
        
        Args:
            config_path: Path to configuration YAML. If None, uses default.
        """
        if config_path is None:
            config_path = REPO_ROOT / "pickafresa_vision" / "configs" / "pnp_calc_config.yaml"
        
        logger.info("Initializing FruitPoseEstimator...")
        logger.info(f"Config path: {config_path}")
        
        self.config = self._load_config(config_path)
        self.depth_sampler = DepthSampler(
            window_size=self.config.get("depth_sampling", {}).get("window_size", 5),
            min_valid_samples=self.config.get("depth_sampling", {}).get("min_valid_samples", 3)
        )
        
        # Build object points for PnP (strawberry corners in its own frame)
        self.object_points_4pt = self._build_object_points_4pt()
        self.object_points_8pt = self._build_object_points_8pt()
        
        # Log configuration
        use_dual_plane = self.config.get("pnp_solver", {}).get("use_dual_plane", False)
        logger.info(f"PnP mode: {'8-point dual-plane' if use_dual_plane else '4-point simple'}")
        logger.info(f"Target classes: {self.config.get('target_classes', ['ripe'])}")
        logger.info(f"Strawberry dimensions: {self.config.get('strawberry_dimensions', {})}")
        logger.info("[OK] FruitPoseEstimator initialized")
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not HAVE_YAML or not config_path.exists():
            return self._get_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "target_classes": ["ripe"],
            "strawberry_dimensions": {"width_mm": 32.5, "height_mm": 34.3},
            "depth_sampling": {
                "window_size": 5,
                "min_valid_samples": 3,
                "variance_threshold_close": 0.50,
                "variance_threshold_mid": 0.40,
                "variance_threshold_far": 0.30,
                "distance_threshold_close": 0.5,
                "distance_threshold_mid": 2.0,
                "inset_ratio": 0.15,  # Sample 15% inside from corners
                "bbox_sample_grid": 10,  # 10x10 grid for bbox median sampling
                "roi_sample_grid": 20,  # 20x20 grid for ROI nearest depth sampling
            },
            "pnp_solver": {
                "method": "SOLVEPNP_ITERATIVE",
                "use_ransac": False,
                "refinement_iterations": 10,
                "use_dual_plane": False,  # Use 8-point (true) or 4-point (false) model
            },
            "validation": {
                "min_confidence": 0.25,
                "max_depth_meters": 3.0,
                "min_depth_meters": 0.105,  # D435 minimum range (~10cm)
                "min_bbox_area": 100,
            }
        }
    
    def _build_object_points_4pt(self) -> np.ndarray:
        """
        Build 3D object points for simple 4-point PnP.
        
        Creates a single plane of bbox corners at the fruit surface.
        This is the standard approach for small object pose estimation.
        
        IMPORTANT: This method accounts for camera mounting rotation to ensure
        accurate PnP results. The camera is mounted with a pitch angle relative
        to the robot TCP, which must be compensated during PnP solving.
        
        Returns:
            (4, 3) array of corner positions in object frame (meters)
            Order: [TL, TR, BR, BL] at Z=0
        """
        dims = self.config.get("strawberry_dimensions", {})
        width_mm = dims.get("width_mm", 32.5)
        height_mm = dims.get("height_mm", 34.3)
        
        # Convert to meters
        hw = (width_mm / 2.0) / 1000.0  # half-width
        hh = (height_mm / 2.0) / 1000.0  # half-height
        
        # Define corners in object frame (Z=0 plane, centered at origin)
        # Object frame: strawberry with Z-axis pointing away from camera
        corners = np.array([
            [-hw, -hh, 0.0],  # Top-left
            [+hw, -hh, 0.0],  # Top-right
            [+hw, +hh, 0.0],  # Bottom-right
            [-hw, +hh, 0.0],  # Bottom-left
        ], dtype=np.float32)
        
        # Apply camera mounting rotation compensation
        # The camera is mounted with a pitch angle relative to the robot TCP.
        # This rotation must be accounted for so that PnP solver gets the correct
        # geometric relationship between image points and 3D object points.
        camera_pitch_deg = self.config.get("camera_mounting", {}).get("pitch_deg", 0.0)
        
        if abs(camera_pitch_deg) > 0.01:  # Only apply if non-zero
            logger.debug(f"Applying camera mounting pitch compensation: {camera_pitch_deg}°")
            
            # Rotation around X-axis (pitch)
            # Positive pitch = camera tilted down (looking downward)
            # We need to rotate object points in the opposite direction
            pitch_rad = np.deg2rad(camera_pitch_deg)
            R_x = np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                [0.0, np.sin(pitch_rad), np.cos(pitch_rad)]
            ], dtype=np.float32)
            
            # Apply rotation to each corner
            corners = (R_x @ corners.T).T
        
        return corners
    
    def _build_object_points_8pt(self) -> np.ndarray:
        """
        Build 3D object points for dual-plane 8-point PnP.
        
        Creates two sets of corners:
        1. Front plane (Z=0): bbox corners at strawberry surface
        2. Back plane (Z=diameter): bbox corners projected to strawberry depth
        
        This provides additional geometric constraints but may not significantly
        improve accuracy for small objects at typical distances.
        
        Returns:
            (8, 3) array of corner positions in object frame (meters)
            Order: [Front: TL, TR, BR, BL, Back: TL, TR, BR, BL]
        """
        dims = self.config.get("strawberry_dimensions", {})
        width_mm = dims.get("width_mm", 32.5)
        height_mm = dims.get("height_mm", 34.3)
        
        # Convert to meters
        hw = (width_mm / 2.0) / 1000.0  # half-width
        hh = (height_mm / 2.0) / 1000.0  # half-height
        
        # Estimate strawberry diameter (use max dimension as approximation)
        diameter = max(width_mm, height_mm) / 1000.0  # meters
        
        # Front plane: bbox corners at strawberry surface (Z=0)
        front_plane = np.array([
            [-hw, -hh, 0.0],  # Top-left
            [+hw, -hh, 0.0],  # Top-right
            [+hw, +hh, 0.0],  # Bottom-right
            [-hw, +hh, 0.0],  # Bottom-left
        ], dtype=np.float32)
        
        # Back plane: bbox corners projected to strawberry depth
        back_plane = np.array([
            [-hw, -hh, diameter],  # Top-left back
            [+hw, -hh, diameter],  # Top-right back
            [+hw, +hh, diameter],  # Bottom-right back
            [-hw, +hh, diameter],  # Bottom-left back
        ], dtype=np.float32)
        
        # Apply camera mounting rotation compensation
        camera_pitch_deg = self.config.get("camera_mounting", {}).get("pitch_deg", 0.0)
        
        if abs(camera_pitch_deg) > 0.01:  # Only apply if non-zero
            logger.debug(f"Applying camera mounting pitch compensation to 8-point model: {camera_pitch_deg}°")
            
            pitch_rad = np.deg2rad(camera_pitch_deg)
            R_x = np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                [0.0, np.sin(pitch_rad), np.cos(pitch_rad)]
            ], dtype=np.float32)
            
            # Apply rotation to both planes
            front_plane = (R_x @ front_plane.T).T
            back_plane = (R_x @ back_plane.T).T
        
        # Combine both planes
        return np.vstack([front_plane, back_plane])
    
    def _cxcywh_to_corners(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Convert cxcywh bbox to corner coordinates.
        
        Args:
            bbox: (center_x, center_y, width, height)
        
        Returns:
            (4, 2) array of corners [TL, TR, BR, BL]
        """
        cx, cy, w, h = bbox
        hw = w / 2.0
        hh = h / 2.0
        
        return np.array([
            [cx - hw, cy - hh],  # Top-left
            [cx + hw, cy - hh],  # Top-right
            [cx + hw, cy + hh],  # Bottom-right
            [cx - hw, cy + hh],  # Bottom-left
        ], dtype=np.float32)
    
    def _get_inset_corners(
        self,
        bbox_cxcywh: Tuple[float, float, float, float],
        inset_ratio: float = 0.15
    ) -> np.ndarray:
        """
        Get corner positions inset from bbox edges.
        
        This samples slightly inside the bbox to avoid edge pixels
        and background discontinuities.
        
        Args:
            bbox_cxcywh: (center_x, center_y, width, height)
            inset_ratio: Fraction to inset from edges (0.15 = 15% inset)
        
        Returns:
            (4, 2) array of inset corners [TL, TR, BR, BL]
        """
        cx, cy, w, h = bbox_cxcywh
        # Reduce effective size by inset
        w_inset = w * (1.0 - 2 * inset_ratio)
        h_inset = h * (1.0 - 2 * inset_ratio)
        hw = w_inset / 2.0
        hh = h_inset / 2.0
        
        return np.array([
            [cx - hw, cy - hh],  # Top-left inset
            [cx + hw, cy - hh],  # Top-right inset
            [cx + hw, cy + hh],  # Bottom-right inset
            [cx - hw, cy + hh],  # Bottom-left inset
        ], dtype=np.float32)
    
    def _sample_bbox_median_depth(
        self,
        color_image: np.ndarray,
        depth_frame: Any,
        bbox_cxcywh: Tuple[float, float, float, float],
        grid_size: int = 10
    ) -> Optional[float]:
        """
        Sample depth across entire bbox using grid sampling with optional color filtering.
        
        This is a fallback strategy when corner sampling fails.
        
        Args:
            color_image: RGB image for color filtering
            depth_frame: RealSense depth frame
            bbox_cxcywh: (center_x, center_y, width, height)
            grid_size: Number of sample points per dimension
        
        Returns:
            Median depth in meters, or None if insufficient samples
        """
        cx, cy, w, h = bbox_cxcywh
        x1 = int(cx - w / 2.0)
        y1 = int(cy - h / 2.0)
        x2 = int(cx + w / 2.0)
        y2 = int(cy + h / 2.0)
        
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        
        # Clamp to frame bounds
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Create color mask if enabled
        color_mask = self._create_color_mask(color_image, (x1, y1, x2, y2))
        
        # Sample on a grid
        values: List[float] = []
        step_x = max(1, (x2 - x1) // grid_size)
        step_y = max(1, (y2 - y1) // grid_size)
        
        for yy in range(y1, y2 + 1, step_y):
            for xx in range(x1, x2 + 1, step_x):
                # Check color mask if enabled
                if color_mask is not None:
                    mask_y = min(yy - y1, color_mask.shape[0] - 1)
                    mask_x = min(xx - x1, color_mask.shape[1] - 1)
                    if color_mask[mask_y, mask_x] == 0:  # Skip non-red pixels (keep only red)
                        continue
                
                depth = depth_frame.get_distance(int(xx), int(yy))
                if depth > 0:
                    values.append(depth)
        
        if len(values) < 3:  # Need at least 3 valid samples
            return None
        
        return float(np.median(values))
    
    def _create_color_mask(self, color_image: np.ndarray, roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Create HSV-based color mask to keep only red pixels (strawberry surface).
        
        Two modes:
        - adaptive: Automatically finds red hue range from bbox pixels using percentiles
        - preset: Uses fixed red hue ranges (0-10 and 160-179)
        
        Args:
            color_image: RGB image
            roi: Region of interest as (x1, y1, x2, y2)
        
        Returns:
            Binary mask where 255 = red (keep), 0 = not red (filter)
            Returns None if color filtering is disabled
        """
        color_cfg = self.config.get("depth_sampling", {}).get("color_filter", {})
        if not color_cfg.get("enabled", False):
            return None
        
        x1, y1, x2, y2 = roi
        
        # Extract ROI from image
        h, w = color_image.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        roi_image = color_image[y1:y2, x1:x2]
        
        # Convert RGB to HSV
        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
        
        mode = color_cfg.get("mode", "adaptive")
        
        if mode == "adaptive":
            # Extract saturation and value thresholds
            adaptive_cfg = color_cfg.get("adaptive", {})
            sat_min = adaptive_cfg.get("saturation_min", 50)
            val_min = adaptive_cfg.get("value_min", 50)
            
            # Create initial mask for saturated, bright pixels (potential red candidates)
            sat_val_mask = (hsv_roi[:, :, 1] >= sat_min) & (hsv_roi[:, :, 2] >= val_min)
            
            if not sat_val_mask.any():
                # No candidates - fall back to preset mode
                mode = "preset"
            else:
                # Extract hue values from candidate pixels
                hue_values = hsv_roi[:, :, 0][sat_val_mask]
                
                # Red hues wrap around 0/180 - normalize to 0-180 range
                # Values near 0 (0-10) and near 180 (160-179) are both red
                # Map 160-179 -> -20 to -1 for unified range
                hue_normalized = hue_values.astype(np.float32)
                hue_normalized[hue_values > 90] -= 180  # Shift upper red range to negative
                
                # Calculate percentile-based range
                percentile_low = adaptive_cfg.get("percentile_low", 10)
                percentile_high = adaptive_cfg.get("percentile_high", 90)
                expansion_factor = adaptive_cfg.get("expansion_factor", 1.2)
                min_red_pixels = adaptive_cfg.get("min_red_pixels", 20)
                
                if len(hue_normalized) < min_red_pixels:
                    # Not enough red pixels - fall back to preset
                    mode = "preset"
                else:
                    hue_low = np.percentile(hue_normalized, percentile_low)
                    hue_high = np.percentile(hue_normalized, percentile_high)
                    
                    # Expand range
                    hue_range = hue_high - hue_low
                    hue_low -= hue_range * (expansion_factor - 1.0) / 2.0
                    hue_high += hue_range * (expansion_factor - 1.0) / 2.0
                    
                    # Clamp to valid red range (-20 to 10)
                    hue_low = max(-20, hue_low)
                    hue_high = min(10, hue_high)
                    
                    # Create mask using adaptive range
                    # Split into two ranges if wrapping
                    if hue_low < 0:
                        # Range wraps: use [hue_low+180, 179] and [0, hue_high]
                        lower1 = np.array([int(hue_low + 180), sat_min, val_min], dtype=np.uint8)
                        upper1 = np.array([179, 255, 255], dtype=np.uint8)
                        lower2 = np.array([0, sat_min, val_min], dtype=np.uint8)
                        upper2 = np.array([int(hue_high), 255, 255], dtype=np.uint8)
                        
                        mask1 = cv2.inRange(hsv_roi, lower1, upper1)
                        mask2 = cv2.inRange(hsv_roi, lower2, upper2)
                        red_mask = cv2.bitwise_or(mask1, mask2)
                    else:
                        # Simple range [hue_low, hue_high]
                        lower = np.array([int(hue_low), sat_min, val_min], dtype=np.uint8)
                        upper = np.array([int(hue_high), 255, 255], dtype=np.uint8)
                        red_mask = cv2.inRange(hsv_roi, lower, upper)
                    
                    return red_mask
        
        # Preset mode (fallback or explicit)
        if mode == "preset":
            preset_cfg = color_cfg.get("preset", {})
            sat_min = preset_cfg.get("saturation_min", 50)
            val_min = preset_cfg.get("value_min", 50)
            
            # Red wraps around: [0-10] and [160-179]
            hue_min_1 = preset_cfg.get("hue_min_1", 0)
            hue_max_1 = preset_cfg.get("hue_max_1", 10)
            hue_min_2 = preset_cfg.get("hue_min_2", 160)
            hue_max_2 = preset_cfg.get("hue_max_2", 179)
            
            lower1 = np.array([hue_min_1, sat_min, val_min], dtype=np.uint8)
            upper1 = np.array([hue_max_1, 255, 255], dtype=np.uint8)
            lower2 = np.array([hue_min_2, sat_min, val_min], dtype=np.uint8)
            upper2 = np.array([hue_max_2, 255, 255], dtype=np.uint8)
            
            mask1 = cv2.inRange(hsv_roi, lower1, upper1)
            mask2 = cv2.inRange(hsv_roi, lower2, upper2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            return red_mask
        
        return None
    
    def _sample_roi_nearest_depth(
        self,
        color_image: np.ndarray,
        depth_frame: Any,
        bbox_cxcywh: Tuple[float, float, float, float],
        grid_size: int = 20
    ) -> Optional[float]:
        """
        Sample nearest (minimum) depth within bbox ROI with optional color filtering.
        
        This is the most robust strategy for fruit detection:
        - Samples across full bbox area
        - Uses minimum depth to find fruit surface (ignoring background)
        - Dense grid sampling for reliability
        - Optional HSV filtering to keep only red pixels (strawberry surface)
        
        Args:
            color_image: RGB image for color filtering
            depth_frame: RealSense depth frame
            bbox_cxcywh: (center_x, center_y, width, height)
            grid_size: Number of sample points per dimension (higher = more robust)
        
        Returns:
            Minimum valid depth in meters, or None if insufficient samples
        """
        cx, cy, w, h = bbox_cxcywh
        x1 = int(cx - w / 2.0)
        y1 = int(cy - h / 2.0)
        x2 = int(cx + w / 2.0)
        y2 = int(cy + h / 2.0)
        
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        
        # Clamp to frame bounds
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Create color mask if enabled
        color_mask = self._create_color_mask(color_image, (x1, y1, x2, y2))
        
        # Sample on a dense grid
        values: List[float] = []
        step_x = max(1, (x2 - x1) // grid_size)
        step_y = max(1, (y2 - y1) // grid_size)
        
        for yy in range(y1, y2 + 1, step_y):
            for xx in range(x1, x2 + 1, step_x):
                # Check color mask if enabled
                if color_mask is not None:
                    mask_y = min(yy - y1, color_mask.shape[0] - 1)
                    mask_x = min(xx - x1, color_mask.shape[1] - 1)
                    if color_mask[mask_y, mask_x] == 0:  # Skip non-red pixels (keep only red)
                        continue
                
                depth = depth_frame.get_distance(int(xx), int(yy))
                if depth > 0:
                    values.append(depth)
        
        if len(values) < 3:  # Need at least 3 valid samples
            return None
        
        # Return minimum depth (nearest point = fruit surface)
        return float(np.min(values))
    
    def _validate_detection(
        self,
        bbox_cxcywh: Tuple[float, float, float, float],
        confidence: float,
        class_name: str
    ) -> Optional[str]:
        """
        Validate detection against configuration thresholds.
        
        Returns:
            Error message if validation fails, None otherwise
        """
        # Check confidence
        min_conf = self.config.get("validation", {}).get("min_confidence", 0.25)
        if confidence < min_conf:
            return f"confidence {confidence:.3f} < threshold {min_conf}"
        
        # Check class filter
        target_classes = self.config.get("target_classes", ["ripe"])
        if class_name not in target_classes:
            return f"class '{class_name}' not in target classes {target_classes}"
        
        # Check bbox area
        _, _, w, h = bbox_cxcywh
        area = w * h
        min_area = self.config.get("validation", {}).get("min_bbox_area", 100)
        max_area = self.config.get("validation", {}).get("max_bbox_area", 250000)
        
        if area < min_area:
            return f"bbox area {area:.0f} < min {min_area}"
        if area > max_area:
            return f"bbox area {area:.0f} > max {max_area}"
        
        return None
    
    def _sample_corner_depths(
        self,
        color_image: np.ndarray,
        depth_frame: Any,
        bbox_cxcywh: Tuple[float, float, float, float],
        corners_2d: np.ndarray
    ) -> Tuple[Optional[List[float]], Optional[str], str, float]:
        """
        Adaptive depth sampling with configurable output strategies.
        
        Primary strategies (configurable via output_strategy in config):
        - "closest": Minimum depth across full bbox (finds fruit surface, ignores background)
        - "center": Depth at bbox center point
        - "closest_offset": Minimum depth + 0.5× strawberry width (pushes deeper into fruit)
        - "center_offset": Center depth + 0.4× strawberry width (deeper from center)
        
        Fallback strategies (if primary fails):
        1. Inset sampling: Sample 15% inside from corners
        2. Center fallback: Use bbox center depth for all corners
        3. Bbox median: Last resort - use grid median across bbox
        
        NOTE: When enforce_depth_constraint=true (pinhole mode), the offset is applied
        AFTER PnP solving to only affect Z. Otherwise, offset affects initial guess.
        
        Args:
            color_image: RGB image for color filtering
            depth_frame: RealSense depth frame
            bbox_cxcywh: Bounding box in (cx, cy, w, h) format
            corners_2d: (4, 2) array of corner positions (kept for inset calculation)
        
        Returns:
            (depths_list, error_message, strategy_used, depth_offset):
                - depths_list: Raw depths without offset (for PnP solving)
                - error_message: Error if sampling failed
                - strategy_used: Name of strategy that succeeded
                - depth_offset: Offset to apply after PnP (0.0 if no offset)
        """
        min_depth = self.config.get("validation", {}).get("min_depth_meters", 0.105)
        max_depth = self.config.get("validation", {}).get("max_depth_meters", 3.0)
        
        logger.debug(f"Depth sampling for bbox {bbox_cxcywh}")
        
        # Get configured output strategy
        ds_cfg = self.config.get("depth_sampling", {})
        output_strategy = ds_cfg.get("output_strategy", "closest")
        
        # Get strawberry width for offset calculations
        width_mm = self.config.get("strawberry_dimensions", {}).get("width_mm", 32.5)
        width_m = width_mm / 1000.0
        
        # Check if enforce_depth_constraint is enabled (pinhole mode)
        enforce_depth_constraint = self.config.get("pnp_solver", {}).get("enforce_depth_constraint", False)
        
        # Sample depth based on strategy
        raw_depth = None
        strategy_name = None
        depth_offset = 0.0
        
        if output_strategy == "closest" or output_strategy == "closest_offset":
            # Sample minimum depth from ROI
            roi_grid_size = ds_cfg.get("roi_sample_grid", 20)
            raw_depth = self._sample_roi_nearest_depth(color_image, depth_frame, bbox_cxcywh, roi_grid_size)
            strategy_name = output_strategy
            
            if raw_depth is not None and min_depth <= raw_depth <= max_depth:
                # Calculate offset but don't apply yet in pinhole mode
                if output_strategy == "closest_offset":
                    offset_multiplier = ds_cfg.get("closest_offset_multiplier", 0.5)
                    depth_offset = width_m * offset_multiplier
                    
                    if enforce_depth_constraint:
                        # Pinhole mode: offset applied after PnP
                        logger.info(f"[OK] ROI closest depth: {raw_depth:.3f}m (offset {depth_offset*1000:.1f}mm will be applied after PnP)")
                        depths = [raw_depth] * 4
                    else:
                        # Non-pinhole: offset affects PnP initial guess
                        final_depth = raw_depth + depth_offset
                        logger.info(f"[OK] ROI closest depth: {raw_depth:.3f}m + offset {depth_offset*1000:.1f}mm = {final_depth:.3f}m (affects PnP)")
                        depths = [final_depth] * 4
                else:
                    logger.info(f"[OK] ROI closest depth: {raw_depth:.3f}m")
                    depths = [raw_depth] * 4
                
                return depths, None, strategy_name, depth_offset
            else:
                logger.debug(f"ROI closest failed: depth={raw_depth}")
        
        elif output_strategy == "center" or output_strategy == "center_offset":
            # Sample depth at bbox center
            cx, cy = bbox_cxcywh[0], bbox_cxcywh[1]
            raw_depth = self.depth_sampler.sample(depth_frame, (int(cx), int(cy)))
            strategy_name = output_strategy
            
            if raw_depth is not None and min_depth <= raw_depth <= max_depth:
                # Calculate offset but don't apply yet in pinhole mode
                if output_strategy == "center_offset":
                    offset_multiplier = ds_cfg.get("center_offset_multiplier", 0.4)
                    depth_offset = width_m * offset_multiplier
                    
                    if enforce_depth_constraint:
                        # Pinhole mode: offset applied after PnP
                        logger.info(f"[OK] ROI center depth: {raw_depth:.3f}m (offset {depth_offset*1000:.1f}mm will be applied after PnP)")
                        depths = [raw_depth] * 4
                    else:
                        # Non-pinhole: offset affects PnP initial guess
                        final_depth = raw_depth + depth_offset
                        logger.info(f"[OK] ROI center depth: {raw_depth:.3f}m + offset {depth_offset*1000:.1f}mm = {final_depth:.3f}m (affects PnP)")
                        depths = [final_depth] * 4
                else:
                    logger.info(f"[OK] ROI center depth: {raw_depth:.3f}m")
                    depths = [raw_depth] * 4
                
                return depths, None, strategy_name, depth_offset
            else:
                logger.debug(f"ROI center failed: depth={raw_depth}")
        
        else:
            logger.warning(f"Unknown output_strategy '{output_strategy}', falling back to closest")
            # Fallback to closest strategy
            roi_grid_size = ds_cfg.get("roi_sample_grid", 20)
            raw_depth = self._sample_roi_nearest_depth(color_image, depth_frame, bbox_cxcywh, roi_grid_size)
            
            if raw_depth is not None and min_depth <= raw_depth <= max_depth:
                depths = [raw_depth] * 4
                logger.info(f"[OK] ROI closest depth (fallback): {raw_depth:.3f}m")
                return depths, None, "closest_fallback", 0.0
            else:
                logger.debug(f"ROI closest fallback failed: depth={raw_depth}")
        
        # Fallback Strategy 1: Inset corner sampling (backup for when primary strategy fails)
        logger.debug("Primary strategy failed, trying inset corner sampling")
        inset_ratio = self.config.get("depth_sampling", {}).get("inset_ratio", 0.15)
        inset_corners = self._get_inset_corners(bbox_cxcywh, inset_ratio)
        depths, error = self._sample_corner_depths_strategy(
            depth_frame, inset_corners, "inset"
        )
        if depths is not None:
            return depths, None, "inset", 0.0
        
        # Fallback Strategy 2: Bbox center depth for all corners
        logger.debug("Inset sampling failed, trying bbox center")
        cx, cy = bbox_cxcywh[0], bbox_cxcywh[1]
        center_depth = self.depth_sampler.sample(depth_frame, (int(cx), int(cy)))
        
        if center_depth is not None and min_depth <= center_depth <= max_depth:
            # Use center depth for all 4 corners
            depths = [center_depth] * 4
            return depths, None, "center_fallback", 0.0
        
        # Fallback Strategy 3: Last resort - bbox median depth
        logger.debug("Center sampling failed, trying bbox median")
        grid_size = self.config.get("depth_sampling", {}).get("bbox_sample_grid", 10)
        median_depth = self._sample_bbox_median_depth(color_image, depth_frame, bbox_cxcywh, grid_size)
        
        if median_depth is not None and min_depth <= median_depth <= max_depth:
            # Use median depth for all 4 corners
            depths = [median_depth] * 4
            return depths, None, "bbox_median", 0.0
        
        # All strategies failed
        return None, f"all depth sampling strategies failed (primary={output_strategy}, fallbacks=inset/center/bbox_median)", "failed", 0.0
    
    def _sample_corner_depths_strategy(
        self,
        depth_frame: Any,
        corners_2d: np.ndarray,
        strategy_name: str
    ) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Sample depth at corners using a specific strategy.
        
        Helper method for inset sampling strategy.
        
        Args:
            depth_frame: RealSense depth frame
            corners_2d: (4, 2) array of corner positions
            strategy_name: Name of strategy for error messages
        
        Returns:
            (depths_list, error_message): depths_list is None if sampling fails
        """
        depths = []
        
        for i, (cx, cy) in enumerate(corners_2d):
            depth = self.depth_sampler.sample(depth_frame, (int(cx), int(cy)))
            
            if depth is None:
                return None, f"{strategy_name}: insufficient valid depth at corner {i}"
            
            # Validate depth range
            min_depth = self.config.get("validation", {}).get("min_depth_meters", 0.105)
            max_depth = self.config.get("validation", {}).get("max_depth_meters", 3.0)
            
            if depth < min_depth or depth > max_depth:
                return None, f"{strategy_name}: depth {depth:.3f}m at corner {i} out of range [{min_depth}, {max_depth}]"
            
            depths.append(depth)
        
        return depths, None
    
    def _validate_depth_consistency(
        self,
        depths: List[float]
    ) -> Tuple[bool, float, float]:
        """
        Validate depth consistency across corners.
        
        Args:
            depths: List of depth values at corners
        
        Returns:
            (is_valid, median_depth, max_variance)
        """
        median_depth = float(np.median(depths))
        
        # Calculate variance ratios
        variances = [abs(d - median_depth) / max(median_depth, 1e-6) for d in depths]
        max_variance = max(variances)
        
        # Adaptive threshold based on distance
        ds_cfg = self.config.get("depth_sampling", {})
        dist_close = ds_cfg.get("distance_threshold_close", 0.5)
        dist_mid = ds_cfg.get("distance_threshold_mid", 2.0)
        
        if median_depth < dist_close:
            threshold = ds_cfg.get("variance_threshold_close", 0.50)
        elif median_depth < dist_mid:
            threshold = ds_cfg.get("variance_threshold_mid", 0.40)
        else:
            threshold = ds_cfg.get("variance_threshold_far", 0.30)
        
        is_valid = max_variance <= threshold
        return is_valid, median_depth, max_variance
    
    def _build_dual_plane_image_points(
        self,
        corners_2d: np.ndarray,
        depths: List[float],
        camera_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Build 8 image points for dual-plane PnP using corrected geometry.
        
        CORRECTED APPROACH:
        - Front plane: Bbox corners at measured depth
        - Back plane: Project corners along camera rays, offset by diameter
        
        This ensures geometric consistency: the same rays from camera,
        just at different depths along those rays.
        
        Args:
            corners_2d: (4, 2) array of corner pixel positions
            depths: List of 4 depth values (should all be identical with roi_nearest)
            camera_matrix: 3x3 camera intrinsic matrix
        
        Returns:
            (8, 2) array of image points [Front corners (4), Back corners (4)]
        """
        # Use median as the uniform depth
        uniform_depth = float(np.median(depths))
        
        # Get strawberry diameter
        dims = self.config.get("strawberry_dimensions", {})
        width_mm = dims.get("width_mm", 32.5)
        height_mm = dims.get("height_mm", 34.3)
        diameter = max(width_mm, height_mm) / 1000.0  # meters
        
        # Extract intrinsics
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # Front plane: use original corner positions
        front_points = corners_2d.copy()
        
        # Back plane: project corners along rays from camera center
        back_points = []
        for u, v in corners_2d:
            # Create unit ray from camera center through pixel
            x_ray = (u - cx) / fx
            y_ray = (v - cy) / fy
            z_ray = 1.0
            
            # Normalize to unit vector
            ray_length = np.sqrt(x_ray**2 + y_ray**2 + z_ray**2)
            ray_unit = np.array([x_ray, y_ray, z_ray]) / ray_length
            
            # Front point in 3D camera frame
            P_front = ray_unit * uniform_depth
            
            # Back point: Move diameter distance along the ray
            # This puts it at depth = uniform_depth + diameter
            P_back = ray_unit * (uniform_depth + diameter)
            
            # Project back point to image plane
            if P_back[2] > 0:  # Ensure point is in front of camera
                u_back = fx * P_back[0] / P_back[2] + cx
                v_back = fy * P_back[1] / P_back[2] + cy
            else:
                # Fallback (shouldn't happen with positive depths)
                u_back = u
                v_back = v
            
            back_points.append([u_back, v_back])
        
        back_points = np.array(back_points, dtype=np.float32)
        
        # Combine front and back planes
        return np.vstack([front_points, back_points])
    
    def _solve_pnp(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        measured_depth: Optional[float] = None,
        bbox_center: Optional[Tuple[float, float]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """
        Solve PnP to get rotation and translation.
        
        Args:
            object_points: 3D object points in object frame
            image_points: 2D image points in pixel coordinates
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            measured_depth: Optional measured depth from sensor (meters)
            bbox_center: Optional bbox center (cx, cy) for depth constraint initialization
        
        Returns:
            (rvec, tvec, error_message): rvec and tvec are None if solving fails
        """
        pnp_cfg = self.config.get("pnp_solver", {})
        method_name = pnp_cfg.get("method", "SOLVEPNP_ITERATIVE")
        use_ransac = pnp_cfg.get("use_ransac", False)
        use_depth_constraint = pnp_cfg.get("use_depth_constraint", False)
        enforce_depth_constraint = pnp_cfg.get("enforce_depth_constraint", False)
        
        # Map method name to OpenCV constant
        method_map = {
            "SOLVEPNP_ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
            "SOLVEPNP_P3P": cv2.SOLVEPNP_P3P,
            "SOLVEPNP_EPNP": cv2.SOLVEPNP_EPNP,
            "SOLVEPNP_IPPE": cv2.SOLVEPNP_IPPE,
        }
        method = method_map.get(method_name, cv2.SOLVEPNP_ITERATIVE)
        
        # Prepare initial guess if using depth constraint
        rvec_init = None
        tvec_init = None
        use_extrinsic_guess = False
        
        if use_depth_constraint and measured_depth is not None and bbox_center is not None:
            # Compute 3D position of bbox center at measured depth
            cx, cy = bbox_center
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            px = camera_matrix[0, 2]
            py = camera_matrix[1, 2]
            
            # Back-project bbox center to 3D at measured depth
            X = (cx - px) * measured_depth / fx
            Y = (cy - py) * measured_depth / fy
            Z = measured_depth
            
            tvec_init = np.array([[X], [Y], [Z]], dtype=np.float32)
            rvec_init = np.zeros((3, 1), dtype=np.float32)  # Identity rotation
            use_extrinsic_guess = True
            
            logger.debug(f"Using depth constraint: initial tvec=[{X:.4f}, {Y:.4f}, {Z:.4f}]")
        
        try:
            if use_ransac:
                # Use solvePnPRansac
                reprojection_error = pnp_cfg.get("ransac_reprojection_error", 8.0)
                confidence = pnp_cfg.get("ransac_confidence", 0.99)
                iterations = pnp_cfg.get("ransac_iterations", 100)
                
                logger.debug(f"Solving PnP with RANSAC (error={reprojection_error}, conf={confidence}, iter={iterations})")
                
                # Note: RANSAC doesn't support useExtrinsicGuess parameter
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    reprojectionError=reprojection_error,
                    confidence=confidence,
                    iterationsCount=iterations,
                    flags=method
                )
                
                if success:
                    logger.debug(f"RANSAC found {len(inliers)} inliers out of {len(object_points)} points")
            else:
                # Use standard solvePnP
                logger.debug(f"Solving PnP with method {method_name}, use_guess={use_extrinsic_guess}")
                
                if use_extrinsic_guess:
                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        camera_matrix,
                        dist_coeffs,
                        rvec=rvec_init,
                        tvec=tvec_init,
                        useExtrinsicGuess=True,
                        flags=method
                    )
                else:
                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        camera_matrix,
                        dist_coeffs,
                        flags=method
                    )
            
            if not success:
                logger.warning("PnP solver failed to converge")
                return None, None, "PnP solver failed to converge"
            
            # Enforce depth constraint if enabled
            if enforce_depth_constraint and measured_depth is not None:
                original_z = tvec[2, 0]
                tvec[2, 0] = measured_depth
                logger.debug(f"Enforced depth constraint: Z {original_z:.4f}m → {measured_depth:.4f}m (Δ={abs(original_z - measured_depth)*1000:.1f}mm)")
            
            logger.debug(f"PnP converged - tvec: {tvec.flatten()}, rvec: {rvec.flatten()}")
            return rvec, tvec, None
            
        except Exception as e:
            logger.error(f"PnP solver exception: {str(e)}")
            return None, None, f"PnP solver exception: {str(e)}"
    
    def _build_transform_matrix(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build 4x4 homogeneous transformation matrix from rvec and tvec.
        
        Returns:
            (T_cam_fruit, rotation_matrix): 4x4 transform and 3x3 rotation
        """
        # Convert rotation vector to matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Build 4x4 homogeneous transform
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation_matrix.astype(np.float32)
        T[:3, 3] = tvec.ravel().astype(np.float32)
        
        return T, rotation_matrix.astype(np.float32)
    
    def estimate_single_pose(
        self,
        bbox_cxcywh: Tuple[float, float, float, float],
        confidence: float,
        class_name: str,
        class_id: int,
        color_image: np.ndarray,
        depth_frame: Any,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> PoseEstimationResult:
        """
        Estimate pose for a single detection.
        
        Args:
            bbox_cxcywh: Bounding box in (cx, cy, w, h) format
            confidence: Detection confidence
            class_name: Class label
            class_id: Numeric class ID
            color_image: RGB image for color filtering
            depth_frame: RealSense depth frame
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        
        Returns:
            PoseEstimationResult object
        """
        # Validate detection
        error = self._validate_detection(bbox_cxcywh, confidence, class_name)
        if error:
            return PoseEstimationResult(
                bbox_cxcywh=bbox_cxcywh,
                confidence=confidence,
                class_name=class_name,
                class_id=class_id,
                success=False,
                error_reason=error
            )
        
        # Convert bbox to corners
        corners_2d = self._cxcywh_to_corners(bbox_cxcywh)
        
        # Sample depths at corners with adaptive fallback
        depths, error, strategy, depth_offset = self._sample_corner_depths(color_image, depth_frame, bbox_cxcywh, corners_2d)
        if error:
            return PoseEstimationResult(
                bbox_cxcywh=bbox_cxcywh,
                confidence=confidence,
                class_name=class_name,
                class_id=class_id,
                success=False,
                error_reason=error,
                depth_samples=depths,
                sampling_strategy=strategy
            )
        
        # Validate depth consistency
        is_valid, median_depth, max_variance = self._validate_depth_consistency(depths)
        if not is_valid:
            return PoseEstimationResult(
                bbox_cxcywh=bbox_cxcywh,
                confidence=confidence,
                class_name=class_name,
                class_id=class_id,
                success=False,
                error_reason=f"depth variance {max_variance:.1%} too high for distance {median_depth:.2f}m",
                depth_samples=depths,
                median_depth=median_depth,
                depth_variance=max_variance,
                sampling_strategy=strategy
            )
        
        # Choose PnP model based on configuration
        use_dual_plane = self.config.get("pnp_solver", {}).get("use_dual_plane", False)
        
        logger.debug(f"Using {'8-point dual-plane' if use_dual_plane else '4-point simple'} PnP model")
        
        if use_dual_plane:
            # Dual-plane 8-point model: More constraints but may not help for small objects
            image_points = self._build_dual_plane_image_points(
                corners_2d,
                depths,
                camera_matrix
            )
            object_points = self.object_points_8pt
            logger.debug(f"Dual-plane image points shape: {image_points.shape}")
        else:
            # Simple 4-point model: Standard approach, recommended for small objects
            image_points = corners_2d
            object_points = self.object_points_4pt
            logger.debug(f"Simple 4-point corners: {corners_2d}")
        
        # Solve PnP with optional depth constraint
        logger.debug(f"Solving PnP with {len(object_points)} correspondences")
        bbox_center = (bbox_cxcywh[0], bbox_cxcywh[1])  # Extract (cx, cy)
        
        rvec, tvec, error = self._solve_pnp(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            measured_depth=median_depth,
            bbox_center=bbox_center
        )
        
        if error:
            logger.warning(f"PnP estimation failed: {error}")
            return PoseEstimationResult(
                bbox_cxcywh=bbox_cxcywh,
                confidence=confidence,
                class_name=class_name,
                class_id=class_id,
                success=False,
                error_reason=error,
                depth_samples=depths,
                median_depth=median_depth,
                depth_variance=max_variance,
                sampling_strategy=strategy
            )
        
        # Build transformation matrix
        T_cam_fruit, rotation_matrix = self._build_transform_matrix(rvec, tvec)
        position_cam = T_cam_fruit[:3, 3]
        
        # Apply depth offset after PnP (only affects Z in pinhole mode)
        enforce_depth_constraint = self.config.get("pnp_solver", {}).get("enforce_depth_constraint", False)
        if enforce_depth_constraint and depth_offset > 0.0:
            # In pinhole mode, offset is applied after PnP to only affect Z
            original_z = position_cam[2]
            T_cam_fruit[2, 3] += depth_offset
            position_cam = T_cam_fruit[:3, 3]
            logger.info(f"[Offset Applied] Z: {original_z:.4f}m → {position_cam[2]:.4f}m (+{depth_offset*1000:.1f}mm)")
            logger.info(f"[OK] Final pose for {class_name} - Position: X={position_cam[0]:.4f}, Y={position_cam[1]:.4f}, Z={position_cam[2]:.4f}")
        else:
            logger.info(f"[OK] Pose estimated for {class_name} - Position: X={position_cam[0]:.4f}, Y={position_cam[1]:.4f}, Z={position_cam[2]:.4f}")
        
        logger.debug(f"Rotation matrix:\n{rotation_matrix}")
        
        return PoseEstimationResult(
            bbox_cxcywh=bbox_cxcywh,
            confidence=confidence,
            class_name=class_name,
            class_id=class_id,
            success=True,
            T_cam_fruit=T_cam_fruit,
            position_cam=position_cam,
            rotation_matrix=rotation_matrix,
            rvec=rvec,
            tvec=tvec,
            depth_samples=depths,
            median_depth=median_depth,
            depth_variance=max_variance,
            sampling_strategy=strategy
        )
    
    def estimate_poses(
        self,
        color_image: np.ndarray,
        depth_frame: Any,
        detections: List[Any],
        bboxes_cxcywh: List[Tuple[float, float, float, float]],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> List[PoseEstimationResult]:
        """
        Estimate poses for multiple detections.
        
        Args:
            color_image: RGB image (used for color filtering in depth sampling)
            depth_frame: RealSense depth frame
            detections: List of Detection objects from inference_bbox
            bboxes_cxcywh: List of bounding boxes in (cx, cy, w, h) format
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        
        Returns:
            List of PoseEstimationResult objects (sorted by confidence descending)
        """
        if len(detections) != len(bboxes_cxcywh):
            raise ValueError("Number of detections and bboxes must match")
        
        results = []
        
        for detection, bbox in zip(detections, bboxes_cxcywh):
            result = self.estimate_single_pose(
                bbox_cxcywh=bbox,
                confidence=detection.confidence,
                class_name=detection.clazz,
                class_id=detection.class_id,
                color_image=color_image,
                depth_frame=depth_frame,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs
            )
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Quick test (requires actual RealSense hardware and detections)
    print("PnP Calculator module loaded successfully")
    print("Use in conjunction with realsense_capture and inference_bbox for pose estimation")
