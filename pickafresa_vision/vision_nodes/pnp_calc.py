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

Team YEA, 2025
"""

from __future__ import annotations

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "bbox_cxcywh": list(self.bbox_cxcywh),
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
            "depth_samples": self.depth_samples,
            "median_depth": self.median_depth,
            "depth_variance": self.depth_variance,
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
        
        self.config = self._load_config(config_path)
        self.depth_sampler = DepthSampler(
            window_size=self.config.get("depth_sampling", {}).get("window_size", 5),
            min_valid_samples=self.config.get("depth_sampling", {}).get("min_valid_samples", 3)
        )
        
        # Build object points for PnP (strawberry corners in its own frame)
        self.object_points = self._build_object_points()
    
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
            },
            "pnp_solver": {
                "method": "SOLVEPNP_ITERATIVE",
                "use_ransac": False,
                "refinement_iterations": 10,
            },
            "validation": {
                "min_confidence": 0.25,
                "max_depth_meters": 3.0,
                "min_depth_meters": 0.1,
                "min_bbox_area": 100,
            }
        }
    
    def _build_object_points(self) -> np.ndarray:
        """
        Build 3D object points for strawberry bbox corners.
        
        Returns:
            (4, 3) array of corner positions in object frame (meters)
            Order: [TL, TR, BR, BL] (top-left, top-right, bottom-right, bottom-left)
        """
        dims = self.config.get("strawberry_dimensions", {})
        width_mm = dims.get("width_mm", 32.5)
        height_mm = dims.get("height_mm", 34.3)
        
        # Convert to meters
        hw = (width_mm / 2.0) / 1000.0  # half-width
        hh = (height_mm / 2.0) / 1000.0  # half-height
        
        # Define corners in object frame (Z=0 plane, centered at origin)
        # Clockwise from top-left
        return np.array([
            [-hw, -hh, 0.0],  # Top-left
            [+hw, -hh, 0.0],  # Top-right
            [+hw, +hh, 0.0],  # Bottom-right
            [-hw, +hh, 0.0],  # Bottom-left
        ], dtype=np.float32)
    
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
        corners_2d: np.ndarray
    ) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Sample depth at each corner with validation.
        
        Returns:
            (depths_list, error_message): depths_list is None if sampling fails
        """
        depths = []
        
        for i, (cx, cy) in enumerate(corners_2d):
            depth = self.depth_sampler.sample(depth_frame, (int(cx), int(cy)))
            
            if depth is None:
                return None, f"insufficient valid depth at corner {i}"
            
            # Validate depth range
            min_depth = self.config.get("validation", {}).get("min_depth_meters", 0.1)
            max_depth = self.config.get("validation", {}).get("max_depth_meters", 3.0)
            
            if depth < min_depth or depth > max_depth:
                return None, f"depth {depth:.3f}m at corner {i} out of range [{min_depth}, {max_depth}]"
            
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
            else:
                # Use standard solvePnP
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=method
                )
            
            if not success:
                return None, None, "PnP solver failed to converge"
            
            return rvec, tvec, None
            
        except Exception as e:
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
        
        # Sample depths at corners
        depths, error = self._sample_corner_depths(depth_frame, corners_2d)
        if error:
            return PoseEstimationResult(
                bbox_cxcywh=bbox_cxcywh,
                confidence=confidence,
                class_name=class_name,
                class_id=class_id,
                success=False,
                error_reason=error,
                depth_samples=depths
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
                depth_variance=max_variance
            )
        
        # Solve PnP
        rvec, tvec, error = self._solve_pnp(
            self.object_points,
            corners_2d,
            camera_matrix,
            dist_coeffs
        )
        
        if error:
            return PoseEstimationResult(
                bbox_cxcywh=bbox_cxcywh,
                confidence=confidence,
                class_name=class_name,
                class_id=class_id,
                success=False,
                error_reason=error,
                depth_samples=depths,
                median_depth=median_depth,
                depth_variance=max_variance
            )
        
        # Build transformation matrix
        T_cam_fruit, rotation_matrix = self._build_transform_matrix(rvec, tvec)
        position_cam = T_cam_fruit[:3, 3]
        
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
            depth_variance=max_variance
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
