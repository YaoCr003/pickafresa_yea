"""
PnP Data Handler for Robot Testing Tool

Handles fruit pose estimation data from two sources:
1. Live API: FruitPoseEstimator from pickafresa_vision
2. Offline JSON: Pre-saved capture files

Features:
- Load and parse JSON files with PnP results
- Call FruitPoseEstimator API for live capture
- Transform coordinates from camera frame to robot base frame
- Handle multiple detections with user selection
- Filter detections by confidence and class

by: Aldrick T, 2025
for Team YEA
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

# Repository root for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FruitDetection:
    """Container for a single fruit detection with pose."""
    
    def __init__(self, detection_dict: Dict[str, Any]):
        """
        Initialize from detection dictionary.
        
        Args:
            detection_dict: Dictionary from PnP result or JSON
        """
        self.bbox_cxcywh = detection_dict.get("bbox_cxcywh", [0, 0, 0, 0])
        self.confidence = detection_dict.get("confidence", 0.0)
        self.class_name = detection_dict.get("class_name", "unknown")
        self.class_id = detection_dict.get("class_id", -1)
        self.success = detection_dict.get("success", False)
        
        # Pose information
        T_cam_fruit_list = detection_dict.get("T_cam_fruit")
        if T_cam_fruit_list is not None:
            self.T_cam_fruit = np.array(T_cam_fruit_list, dtype=np.float64)
            # Ensure it's a 4x4 matrix
            if self.T_cam_fruit.shape != (4, 4):
                self.T_cam_fruit = self.T_cam_fruit.reshape(4, 4)
        else:
            self.T_cam_fruit = None
        
        pos_list = detection_dict.get("position_cam")
        self.position_cam = np.array(pos_list) if pos_list else None
        
        # Additional metadata
        self.error_reason = detection_dict.get("error_reason")
        self.depth_samples = detection_dict.get("depth_samples")
        self.median_depth = detection_dict.get("median_depth")
        self.sampling_strategy = detection_dict.get("sampling_strategy")
        
        # Robot base frame pose (computed later)
        self.T_base_fruit: Optional[np.ndarray] = None
        self.position_base: Optional[np.ndarray] = None
    
    def __repr__(self) -> str:
        status = "[OK]" if self.success else "[FAIL]"
        if self.position_cam is not None:
            pos_str = f"[{self.position_cam[0]:.3f}, {self.position_cam[1]:.3f}, {self.position_cam[2]:.3f}]m"
        else:
            pos_str = "N/A"
        
        return (f"FruitDetection({status} {self.class_name}, "
                f"conf={self.confidence:.2f}, pos_cam={pos_str})")


class PnPDataHandler:
    """Handler for PnP data from API or JSON files."""
    
    def __init__(
        self,
        T_flange_cameraTCP: np.ndarray,
        T_flange_gripperTCP: np.ndarray,
        logger: Optional[any] = None
    ):
        """
        Initialize PnP data handler.
        
        Args:
            T_flange_cameraTCP: 4x4 transformation from flange to camera TCP
            T_flange_gripperTCP: 4x4 transformation from flange to gripper TCP
            logger: Logger instance (optional)
        """
        self.T_flange_cameraTCP = T_flange_cameraTCP
        self.T_flange_gripperTCP = T_flange_gripperTCP
        self.logger = logger
        
        self._log_info("PnP Data Handler initialized")
        self._log_info(f"Flange-to-Camera TCP transform:\n{T_flange_cameraTCP}")
        self._log_info(f"Flange-to-Gripper TCP transform:\n{T_flange_gripperTCP}")
    
    def _log_info(self, message: str) -> None:
        """Log info message."""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"[INFO] {message}")
    
    def _log_warn(self, message: str) -> None:
        """Log warning message."""
        if self.logger:
            self.logger.warn(message)
        else:
            print(f"[WARN] {message}")
    
    def _log_error(self, message: str) -> None:
        """Log error message."""
        if self.logger:
            self.logger.error(message)
        else:
            print(f"[ERROR] {message}")
    
    def load_json_file(
        self,
        json_path: Path,
        min_confidence: float = 0.0,
        preferred_class: Optional[str] = None
    ) -> List[FruitDetection]:
        """
        Load PnP results from JSON file.
        
        Args:
            json_path: Path to JSON file
            min_confidence: Minimum confidence threshold
            preferred_class: Preferred class name (for filtering)
        
        Returns:
            List of FruitDetection objects
        """
        self._log_info(f"Loading PnP data from: {json_path}")
        
        if not json_path.exists():
            self._log_error(f"JSON file not found: {json_path}")
            return []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract detections
            detections_raw = data.get("detections", [])
            self._log_info(f"Found {len(detections_raw)} detections in JSON")
            
            # Create FruitDetection objects
            detections = [FruitDetection(d) for d in detections_raw]
            
            # Filter by success and confidence
            detections = [
                d for d in detections
                if d.success and d.confidence >= min_confidence
            ]
            self._log_info(f"After filtering (success + conf>={min_confidence}): {len(detections)} detections")
            
            # Sort by preferred class, then by confidence
            if preferred_class:
                detections.sort(
                    key=lambda d: (d.class_name != preferred_class, -d.confidence)
                )
            else:
                detections.sort(key=lambda d: -d.confidence)
            
            return detections
        
        except Exception as e:
            self._log_error(f"Failed to load JSON file: {e}")
            return []
    
    def call_api_live(
        self,
        objd_config_path: Path,
        pnp_config_path: Path,
        realsense_config_path: Optional[Path] = None,
        min_confidence: float = 0.0
    ) -> List[FruitDetection]:
        """
        Call FruitPoseEstimator API for live capture.
        
        Args:
            objd_config_path: Path to object detection config
            pnp_config_path: Path to PnP calculation config
            realsense_config_path: Path to RealSense config (optional)
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of FruitDetection objects
        """
        self._log_info("Calling FruitPoseEstimator API (live capture)...")
        
        try:
            # Import vision modules
            from pickafresa_vision.vision_nodes.pnp_calc import FruitPoseEstimator
            from pickafresa_vision.vision_nodes.inference_bbox import load_model, infer
            from pickafresa_vision.vision_tools.realsense_capture import RealSenseCapture
            
            # Load object detection config
            import yaml
            with open(objd_config_path, 'r') as f:
                objd_config = yaml.safe_load(f)
            
            # Initialize components
            self._log_info("Initializing FruitPoseEstimator...")
            estimator = FruitPoseEstimator(config_path=pnp_config_path)
            
            self._log_info("Loading YOLO model...")
            model_path = objd_config.get('model_path', '')
            if not Path(model_path).is_absolute():
                model_path = str(REPO_ROOT / model_path)
            model = load_model(model_path)
            
            self._log_info("Initializing RealSense camera...")
            camera = RealSenseCapture(config_path=realsense_config_path)
            
            # Capture frame
            self._log_info("Capturing aligned RGB+Depth frame...")
            color_img, depth_frame, intrinsics = camera.capture_aligned_frame()
            
            if color_img is None or depth_frame is None:
                self._log_error("Failed to capture frame from RealSense")
                camera.release()
                return []
            
            # Run YOLO detection
            self._log_info("Running YOLO detection...")
            conf_threshold = objd_config.get('conf_threshold', 0.25)
            iou_threshold = objd_config.get('iou_threshold', 0.45)
            
            results = infer(
                model=model,
                image=color_img,
                conf=conf_threshold,
                iou=iou_threshold
            )
            
            detections_yolo, bboxes = results
            self._log_info(f"YOLO detected {len(detections_yolo)} objects")
            
            # Estimate poses
            self._log_info("Estimating 6DOF poses...")
            pose_results = estimator.estimate_poses(
                color_image=color_img,
                depth_frame=depth_frame,
                detections=detections_yolo,
                bboxes_cxcywh=bboxes,
                camera_matrix=intrinsics.to_matrix(),
                dist_coeffs=intrinsics.distortion_coeffs
            )
            
            # Release camera
            camera.release()
            
            # Convert to FruitDetection objects
            detections = [
                FruitDetection(r.to_dict())
                for r in pose_results
                if r.success and r.confidence >= min_confidence
            ]
            
            self._log_info(f"Successfully estimated {len(detections)} poses")
            
            return detections
        
        except Exception as e:
            self._log_error(f"Failed to call API: {e}")
            import traceback
            self._log_error(traceback.format_exc())
            return []
    
    def transform_to_base_frame(
        self,
        detection: FruitDetection,
        T_base_gripperTCP: np.ndarray
    ) -> FruitDetection:
        """
        Transform fruit pose from camera frame to robot base frame.
        
        Transform chain when robot is at Foto position:
        1. T_base_gripperTCP: Gripper TCP pose at Foto (from RoboDK, in MILLIMETERS)
        2. Convert to meters for consistent units
        3. T_base_flange = T_base_gripperTCP @ inv(T_flange_gripperTCP)
        4. T_base_camera = T_base_flange @ T_flange_cameraTCP
        5. T_base_fruit = T_base_camera @ T_cam_fruit
        
        Args:
            detection: FruitDetection object with T_cam_fruit (in METERS)
            T_base_gripperTCP: 4x4 transformation from robot base to gripper TCP (from RoboDK, in MILLIMETERS!)
        
        Returns:
            Updated FruitDetection with T_base_fruit and position_base (in METERS)
        """
        if detection.T_cam_fruit is None:
            self._log_warn("Cannot transform: T_cam_fruit is None")
            return detection
        
        # Ensure all transformation matrices are proper 4x4 arrays
        if detection.T_cam_fruit.shape != (4, 4):
            self._log_error(f"T_cam_fruit has invalid shape: {detection.T_cam_fruit.shape}, expected (4, 4)")
            return detection
        
        if T_base_gripperTCP.shape != (4, 4):
            self._log_error(f"T_base_gripperTCP has invalid shape: {T_base_gripperTCP.shape}, expected (4, 4)")
            return detection
        
        # CRITICAL: Convert RoboDK pose from millimeters to meters
        # RoboDK uses millimeters, but our vision system uses meters
        T_base_gripperTCP_m = T_base_gripperTCP.copy()
        T_base_gripperTCP_m[:3, 3] = T_base_gripperTCP[:3, 3] / 1000.0  # Convert translation from mm to m
        
        # Step 1: Compute flange position from gripper TCP position
        # T_base_flange = T_base_gripperTCP @ inv(T_flange_gripperTCP)
        T_base_flange = T_base_gripperTCP_m @ np.linalg.inv(self.T_flange_gripperTCP)
        
        # Step 2: Compute camera position from flange position
        # T_base_camera = T_base_flange @ T_flange_cameraTCP
        T_base_camera = T_base_flange @ self.T_flange_cameraTCP
        
        # Step 3: Compute fruit position from camera
        # T_base_fruit = T_base_camera @ T_cam_fruit
        T_base_fruit = T_base_camera @ detection.T_cam_fruit
        
        # Verify result is a proper 4x4 matrix
        if T_base_fruit.shape != (4, 4):
            self._log_error(f"T_base_fruit has invalid shape: {T_base_fruit.shape}, expected (4, 4)")
            return detection
        
        # Extract position (in meters)
        position_base = T_base_fruit[:3, 3]
        
        # Update detection
        detection.T_base_fruit = T_base_fruit
        detection.position_base = position_base
        
        self._log_info(
            f"Transformed to base frame: "
            f"[{position_base[0]:.3f}, {position_base[1]:.3f}, {position_base[2]:.3f}]m "
            f"= [{position_base[0]*1000:.1f}, {position_base[1]*1000:.1f}, {position_base[2]*1000:.1f}]mm"
        )
        
        return detection
    
    def select_detection_interactive(
        self,
        detections: List[FruitDetection]
    ) -> Optional[FruitDetection]:
        """
        Let user select a detection from multiple options.
        
        Args:
            detections: List of available detections
        
        Returns:
            Selected detection or None if cancelled
        """
        if not detections:
            self._log_warn("No detections available to select")
            return None
        
        if len(detections) == 1:
            self._log_info("Only one detection available, auto-selecting...")
            return detections[0]
        
        # Display options
        print("\n" + "=" * 70)
        print("Multiple fruit detections found. Please select one:")
        print("=" * 70)
        
        for i, det in enumerate(detections, 1):
            pos_str = "N/A"
            if det.position_cam is not None:
                pos_str = f"[{det.position_cam[0]:.3f}, {det.position_cam[1]:.3f}, {det.position_cam[2]:.3f}]m"
            
            print(f"  [{i}] {det.class_name:<10} conf={det.confidence:.2f}  pos_cam={pos_str}")
        
        print("=" * 70)
        
        # Get user input
        while True:
            try:
                choice = input(f"Select detection [1-{len(detections)}] or 'q' to cancel: ").strip()
                
                if choice.lower() == 'q':
                    self._log_info("Selection cancelled by user")
                    return None
                
                idx = int(choice) - 1
                
                if 0 <= idx < len(detections):
                    selected = detections[idx]
                    self._log_info(f"Selected detection #{choice}: {selected.class_name} (conf={selected.confidence:.2f})")
                    return selected
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(detections)}.")
            
            except ValueError:
                print("Invalid input. Please enter a number or 'q'.")
            except KeyboardInterrupt:
                print("\nSelection cancelled.")
                return None


def create_transform_matrix(
    translation_mm: List[float],
    rotation_deg: List[float]
) -> np.ndarray:
    """
    Create 4x4 homogeneous transformation matrix from translation and rotation.
    
    Args:
        translation_mm: [x, y, z] in millimeters
        rotation_deg: [u, v, w] axis-angle in degrees
    
    Returns:
        4x4 transformation matrix
    """
    import cv2
    
    # Convert to meters
    t = np.array(translation_mm) / 1000.0  # mm to m
    
    # Convert rotation to radians
    r_deg = np.array(rotation_deg)
    r_rad = np.deg2rad(r_deg)
    
    # Convert axis-angle to rotation matrix
    R, _ = cv2.Rodrigues(r_rad)
    
    # Build 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


# Example usage
if __name__ == "__main__":
    # Create camera TCP offset from flange
    T_flange_cameraTCP = create_transform_matrix(
        translation_mm=[-11.080, -53.400, 24.757],
        rotation_deg=[0.0, 0.0, 0.0]
    )
    
    # Create gripper TCP offset from flange
    T_flange_gripperTCP = create_transform_matrix(
        translation_mm=[0.0, 0.0, 77.902],
        rotation_deg=[0.0, 0.0, 0.0]
    )
    
    # Create handler
    handler = PnPDataHandler(
        T_flange_cameraTCP=T_flange_cameraTCP,
        T_flange_gripperTCP=T_flange_gripperTCP
    )
    
    # Load from JSON
    json_file = REPO_ROOT / "pickafresa_vision/captures/20251104_161710_data.json"
    detections = handler.load_json_file(json_file, min_confidence=0.5)
    
    print(f"\nLoaded {len(detections)} detections:")
    for det in detections:
        print(f"  {det}")
