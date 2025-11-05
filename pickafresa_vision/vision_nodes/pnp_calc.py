"""
Perspective-n-Point (PnP) Pose Estimation for Strawberry Detection

This module provides a robust API for estimating 6DOF poses of detected strawberries
using bounding box corners, RealSense depth data, and OpenCV's PnP solver.

Key Pipeline:
1. Take RGB+Depth aligned frame (from realsense_capture)
2. Get detections with bounding boxes (from inference_bbox in cxcywh format)
3. Sample depth at bbox corners using median filtering
4. Validate depth consistency across corners
5. Solve PnP using known strawberry dimensions and bbox corners
6. Return 4x4 homogeneous transformation matrix (Camera → Fruit frame)

Features:
- Multi-detection processing (sorted by confidence)
- Configurable class filtering (default: "ripe" only)
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

by: Aldrick T, 2025 
for Team YEA
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

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
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
        logger.info("✓ FruitPoseEstimator initialized")
    
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
        return np.array([
            [-hw, -hh, 0.0],  # Top-left
            [+hw, -hh, 0.0],  # Top-right
            [+hw, +hh, 0.0],  # Bottom-right
            [-hw, +hh, 0.0],  # Bottom-left
        ], dtype=np.float32)
    
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
        depth_frame: Any,
        bbox_cxcywh: Tuple[float, float, float, float],
        grid_size: int = 10
    ) -> Optional[float]:
        """
        Sample depth across entire bbox using grid sampling.
        
        This is a fallback strategy when corner sampling fails.
        
        Args:
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
        
        # Sample on a grid
        values: List[float] = []
        step_x = max(1, (x2 - x1) // grid_size)
        step_y = max(1, (y2 - y1) // grid_size)
        
        for yy in range(y1, y2 + 1, step_y):
            for xx in range(x1, x2 + 1, step_x):
                depth = depth_frame.get_distance(int(xx), int(yy))
                if depth > 0:
                    values.append(depth)
        
        if len(values) < 3:  # Need at least 3 valid samples
            return None
        
        return float(np.median(values))
    
    def _sample_roi_nearest_depth(
        self,
        depth_frame: Any,
        bbox_cxcywh: Tuple[float, float, float, float],
        grid_size: int = 20
    ) -> Optional[float]:
        """
        Sample nearest (minimum) depth within bbox ROI.
        
        This is the most robust strategy for fruit detection:
        - Samples across full bbox area
        - Uses minimum depth to find fruit surface (ignoring background)
        - Dense grid sampling for reliability
        
        Args:
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
        
        # Sample on a dense grid
        values: List[float] = []
        step_x = max(1, (x2 - x1) // grid_size)
        step_y = max(1, (y2 - y1) // grid_size)
        
        for yy in range(y1, y2 + 1, step_y):
            for xx in range(x1, x2 + 1, step_x):
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
        depth_frame: Any,
        bbox_cxcywh: Tuple[float, float, float, float],
        corners_2d: np.ndarray
    ) -> Tuple[Optional[List[float]], Optional[str], str]:
        """
        Adaptive depth sampling with multiple fallback strategies.
        
        Strategy priority (most robust to least):
        1. ROI Nearest: Minimum depth across full bbox (ignores background)
        2. Inset sampling: Sample 15% inside from corners
        3. Center fallback: Use bbox center depth for all corners
        4. Bbox median: Last resort - use grid median across bbox
        
        Note: Corner sampling removed - fundamentally flawed when fruit
        has background at different depth.
        
        Args:
            depth_frame: RealSense depth frame
            bbox_cxcywh: Bounding box in (cx, cy, w, h) format
            corners_2d: (4, 2) array of corner positions (kept for inset calculation)
        
        Returns:
            (depths_list, error_message, strategy_used):
                depths_list is None only if all strategies fail
        """
        min_depth = self.config.get("validation", {}).get("min_depth_meters", 0.105)
        max_depth = self.config.get("validation", {}).get("max_depth_meters", 3.0)
        
        logger.debug(f"Depth sampling for bbox {bbox_cxcywh}")
        
        # Strategy 1: ROI Nearest Depth (PRIMARY - most robust)
        roi_grid_size = self.config.get("depth_sampling", {}).get("roi_sample_grid", 20)
        nearest_depth = self._sample_roi_nearest_depth(depth_frame, bbox_cxcywh, roi_grid_size)
        
        if nearest_depth is not None and min_depth <= nearest_depth <= max_depth:
            # Use nearest depth for all 4 corners (uniform depth model)
            depths = [nearest_depth] * 4
            logger.info(f"✓ ROI nearest depth: {nearest_depth:.3f}m")
            return depths, None, "roi_nearest"
        else:
            logger.debug(f"ROI nearest failed: depth={nearest_depth}")
        
        # Strategy 2: Inset corner sampling (backup for when ROI fails)
        inset_ratio = self.config.get("depth_sampling", {}).get("inset_ratio", 0.15)
        inset_corners = self._get_inset_corners(bbox_cxcywh, inset_ratio)
        depths, error = self._sample_corner_depths_strategy(
            depth_frame, inset_corners, "inset"
        )
        if depths is not None:
            return depths, None, "inset"
        
        # Strategy 3: Bbox center depth for all corners
        cx, cy = bbox_cxcywh[0], bbox_cxcywh[1]
        center_depth = self.depth_sampler.sample(depth_frame, (int(cx), int(cy)))
        
        if center_depth is not None and min_depth <= center_depth <= max_depth:
            # Use center depth for all 4 corners
            depths = [center_depth] * 4
            return depths, None, "center"
        
        # Strategy 4: Last resort - bbox median depth
        grid_size = self.config.get("depth_sampling", {}).get("bbox_sample_grid", 10)
        median_depth = self._sample_bbox_median_depth(depth_frame, bbox_cxcywh, grid_size)
        
        if median_depth is not None and min_depth <= median_depth <= max_depth:
            # Use median depth for all 4 corners
            depths = [median_depth] * 4
            return depths, None, "bbox_median"
        
        # All strategies failed
        return None, "all depth sampling strategies failed (roi_nearest/inset/center/bbox_median)", "failed"
    
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
        dist_coeffs: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """
        Solve PnP to get rotation and translation.
        
        Returns:
            (rvec, tvec, error_message): rvec and tvec are None if solving fails
        """
        pnp_cfg = self.config.get("pnp_solver", {})
        method_name = pnp_cfg.get("method", "SOLVEPNP_ITERATIVE")
        use_ransac = pnp_cfg.get("use_ransac", False)
        
        # Map method name to OpenCV constant
        method_map = {
            "SOLVEPNP_ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
            "SOLVEPNP_P3P": cv2.SOLVEPNP_P3P,
            "SOLVEPNP_EPNP": cv2.SOLVEPNP_EPNP,
            "SOLVEPNP_IPPE": cv2.SOLVEPNP_IPPE,
        }
        method = method_map.get(method_name, cv2.SOLVEPNP_ITERATIVE)
        
        try:
            if use_ransac:
                # Use solvePnPRansac
                reprojection_error = pnp_cfg.get("ransac_reprojection_error", 8.0)
                confidence = pnp_cfg.get("ransac_confidence", 0.99)
                iterations = pnp_cfg.get("ransac_iterations", 100)
                
                logger.debug(f"Solving PnP with RANSAC (error={reprojection_error}, conf={confidence}, iter={iterations})")
                
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
                logger.debug(f"Solving PnP with method {method_name}")
                
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
        depths, error, strategy = self._sample_corner_depths(depth_frame, bbox_cxcywh, corners_2d)
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
        
        # Solve PnP
        logger.debug(f"Solving PnP with {len(object_points)} correspondences")
        rvec, tvec, error = self._solve_pnp(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs
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
        
        logger.info(f"✓ Pose estimated for {class_name} - Position: X={position_cam[0]:.4f}, Y={position_cam[1]:.4f}, Z={position_cam[2]:.4f}")
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
            color_image: RGB image (not currently used, kept for API consistency)
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
