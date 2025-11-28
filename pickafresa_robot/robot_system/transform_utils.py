"""
Transform Utilities for Robot PnP System

Provides coordinate frame transformations, TCP conversions, and offset calculations
for the robotic pick-and-place system.

Key Concepts:
- All transformations use 4x4 homogeneous matrices
- Units: METERS for internal calculations, MILLIMETERS for RoboDK interface
- Rotation: Axis-angle representation (Rodrigues formula via OpenCV)

Transform Chain:
    T_base_fruit = T_base_camera @ T_camera_fruit
    T_base_gripper = T_base_fruit @ T_fruit_gripper

Coordinate Frames:
- Robot base frame: Z+ up, X+ forward
- Fruit/berry frame: Z+ away from camera (into fruit)
- Camera frame: OpenCV convention (Z+ forward, X+ right, Y+ down)

by: Aldrick T, 2025
for Team YEA
"""

import numpy as np
from typing import List, Tuple, Optional
import cv2


class TransformUtils:
    """Utilities for 3D transformations and coordinate frame conversions."""
    
    # Unit conversion constants
    MM_TO_M = 0.001
    M_TO_MM = 1000.0
    
    @staticmethod
    def create_transform_matrix(
        translation: List[float],
        rotation_deg: List[float],
        input_units: str = "mm"
    ) -> np.ndarray:
        """
        Create 4x4 homogeneous transformation matrix.
        
        Args:
            translation: [x, y, z] translation
            rotation_deg: [u, v, w] axis-angle rotation in degrees
            input_units: "mm" or "m" (output is always meters internally)
        
        Returns:
            4x4 transformation matrix (translation in meters)
        """
        # Convert translation to meters
        t = np.array(translation, dtype=np.float64)
        if input_units == "mm":
            t = t * TransformUtils.MM_TO_M
        
        # Convert rotation to radians
        r_deg = np.array(rotation_deg, dtype=np.float64)
        r_rad = np.deg2rad(r_deg)
        
        # Convert axis-angle to rotation matrix using Rodrigues formula
        R, _ = cv2.Rodrigues(r_rad)
        
        # Build 4x4 homogeneous matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
    
    @staticmethod
    def convert_robodk_pose_to_meters(T_robodk: np.ndarray) -> np.ndarray:
        """
        Convert RoboDK pose (millimeters) to internal representation (meters).
        
        Args:
            T_robodk: 4x4 transformation matrix with translation in millimeters
        
        Returns:
            4x4 transformation matrix with translation in meters
        """
        T_meters = T_robodk.copy()
        T_meters[:3, 3] = T_robodk[:3, 3] * TransformUtils.MM_TO_M
        return T_meters
    
    @staticmethod
    def convert_meters_to_robodk_pose(T_meters: np.ndarray) -> np.ndarray:
        """
        Convert internal pose (meters) to RoboDK pose (millimeters).
        
        Args:
            T_meters: 4x4 transformation matrix with translation in meters
        
        Returns:
            4x4 transformation matrix with translation in millimeters
        """
        T_robodk = T_meters.copy()
        T_robodk[:3, 3] = T_meters[:3, 3] * TransformUtils.M_TO_MM
        return T_robodk
    
    @staticmethod
    def transform_fruit_to_base(
        T_cam_fruit: np.ndarray,
        T_base_gripperTCP: np.ndarray,
        T_flange_cameraTCP: np.ndarray,
        T_flange_gripperTCP: np.ndarray
    ) -> np.ndarray:
        """
        Transform fruit pose from camera frame to robot base frame.
        
        Transform chain when robot is at capture position:
        1. T_base_gripperTCP: Gripper TCP pose (from RoboDK, in MILLIMETERS)
        2. Convert to meters for consistent units
        3. T_base_flange = T_base_gripperTCP @ inv(T_flange_gripperTCP)
        4. T_base_camera = T_base_flange @ T_flange_cameraTCP
        5. T_base_fruit = T_base_camera @ T_cam_fruit
        
        Args:
            T_cam_fruit: 4x4 fruit pose in camera frame (meters)
            T_base_gripperTCP: 4x4 gripper TCP pose in base frame (MILLIMETERS!)
            T_flange_cameraTCP: 4x4 camera TCP offset from flange (meters)
            T_flange_gripperTCP: 4x4 gripper TCP offset from flange (meters)
        
        Returns:
            T_base_fruit: 4x4 fruit pose in base frame (meters)
        """
        # Convert RoboDK pose from millimeters to meters
        T_base_gripperTCP_m = TransformUtils.convert_robodk_pose_to_meters(T_base_gripperTCP)
        
        # Compute flange position from gripper TCP position
        T_base_flange = T_base_gripperTCP_m @ np.linalg.inv(T_flange_gripperTCP)
        
        # Compute camera position from flange position
        T_base_camera = T_base_flange @ T_flange_cameraTCP
        
        # Compute fruit position from camera
        T_base_fruit = T_base_camera @ T_cam_fruit
        
        return T_base_fruit
    
    @staticmethod
    def apply_offset_in_frame(
        T_base_frame: np.ndarray,
        offset_mm: List[float],
        rotation_deg: Optional[List[float]] = None,
        rotation_mode: str = "absolute"
    ) -> np.ndarray:
        """
        Apply translation and rotation offset in a reference frame.
        
        Args:
            T_base_frame: 4x4 transformation of reference frame in base
            offset_mm: [x, y, z] translation offset in millimeters (in reference frame)
            rotation_deg: [u, v, w] rotation in degrees (optional)
            rotation_mode: "absolute" (in base frame axes) or "cumulative" (in current frame)
        
        Returns:
            T_base_offset: New pose with offset applied
        """
        # Convert offset to meters
        offset_m = np.array(offset_mm, dtype=np.float64) * TransformUtils.MM_TO_M
        
        # Create offset transformation in reference frame
        T_frame_offset = np.eye(4, dtype=np.float64)
        T_frame_offset[:3, 3] = offset_m
        
        # Apply rotation if specified
        if rotation_deg is not None:
            r_rad = np.deg2rad(np.array(rotation_deg, dtype=np.float64))
            R_offset, _ = cv2.Rodrigues(r_rad)
            
            if rotation_mode == "absolute":
                # Rotation in BASE frame axes
                # Extract base frame orientation, apply rotation, combine with offset
                R_base = T_base_frame[:3, :3]
                R_new = R_offset @ R_base
                T_frame_offset[:3, :3] = R_base.T @ R_new  # Convert to frame-local rotation
            else:
                # Cumulative rotation in CURRENT frame axes (default)
                T_frame_offset[:3, :3] = R_offset
        
        # Apply offset: T_base_offset = T_base_frame @ T_frame_offset
        T_base_offset = T_base_frame @ T_frame_offset
        
        return T_base_offset
    
    @staticmethod
    def compute_gripper_target_from_fruit(
        T_base_fruit: np.ndarray,
        T_flange_cameraTCP: np.ndarray,
        T_flange_gripperTCP: np.ndarray,
        offset_mm: List[float],
        rotation_deg: Optional[List[float]] = None,
        rotation_mode: str = "absolute"
    ) -> np.ndarray:
        """
        Compute gripper TCP target pose for picking a fruit.
        
        This accounts for the difference between camera TCP (for vision) and
        gripper TCP (for picking).
        
        Args:
            T_base_fruit: 4x4 fruit pose in base frame (meters)
            T_flange_cameraTCP: Camera TCP offset from flange (meters)
            T_flange_gripperTCP: Gripper TCP offset from flange (meters)
            offset_mm: Picking offset in fruit frame [x, y, z] (millimeters)
            rotation_deg: Optional rotation offset [u, v, w] (degrees)
            rotation_mode: "absolute" or "cumulative"
        
        Returns:
            T_base_gripperTCP: 4x4 gripper TCP target pose (meters)
        """
        # Apply offset in fruit frame
        T_base_offset = TransformUtils.apply_offset_in_frame(
            T_base_frame=T_base_fruit,
            offset_mm=offset_mm,
            rotation_deg=rotation_deg,
            rotation_mode=rotation_mode
        )
        
        # Convert from camera TCP to gripper TCP
        # T_base_gripperTCP = T_base_flange @ T_flange_gripperTCP
        # T_base_flange = T_base_offset @ inv(T_flange_cameraTCP)
        # Therefore: T_base_gripperTCP = T_base_offset @ inv(T_flange_cameraTCP) @ T_flange_gripperTCP
        
        T_cameraTCP_gripperTCP = np.linalg.inv(T_flange_cameraTCP) @ T_flange_gripperTCP
        T_base_gripperTCP = T_base_offset @ T_cameraTCP_gripperTCP
        
        return T_base_gripperTCP
    
    @staticmethod
    def apply_joint_deltas(
        current_joints_deg: List[float],
        joint_deltas_deg: List[float]
    ) -> List[float]:
        """
        Apply joint deltas for cumulative joint space movement.
        
        Args:
            current_joints_deg: Current joint positions in degrees [j0, j1, j2, j3, j4, j5]
            joint_deltas_deg: Joint deltas in degrees [deltaj0, deltaj1, deltaj2, deltaj3, deltaj4, deltaj5]
        
        Returns:
            New joint positions in degrees
        """
        current = np.array(current_joints_deg, dtype=np.float64)
        deltas = np.array(joint_deltas_deg, dtype=np.float64)
        return (current + deltas).tolist()
    
    @staticmethod
    def extract_position_mm(T: np.ndarray) -> List[float]:
        """
        Extract translation from transformation matrix in millimeters.
        
        Args:
            T: 4x4 transformation matrix (translation in meters)
        
        Returns:
            [x, y, z] in millimeters
        """
        position_m = T[:3, 3]
        position_mm = position_m * TransformUtils.M_TO_MM
        return position_mm.tolist()
    
    @staticmethod
    def extract_position_m(T: np.ndarray) -> List[float]:
        """
        Extract translation from transformation matrix in meters.
        
        Args:
            T: 4x4 transformation matrix (translation in meters)
        
        Returns:
            [x, y, z] in meters
        """
        return T[:3, 3].tolist()
    
    @staticmethod
    def matrix_to_pose_mm(T: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Convert transformation matrix to pose representation (RoboDK format).
        
        Args:
            T: 4x4 transformation matrix (translation in meters)
        
        Returns:
            Tuple of (translation_mm, rotation_deg)
            - translation_mm: [x, y, z] in millimeters
            - rotation_deg: [u, v, w] axis-angle in degrees
        """
        # Extract translation (convert to mm)
        translation_mm = TransformUtils.extract_position_mm(T)
        
        # Extract rotation (convert to axis-angle degrees)
        R = T[:3, :3]
        rvec, _ = cv2.Rodrigues(R)
        rotation_deg = np.rad2deg(rvec.flatten()).tolist()
        
        return translation_mm, rotation_deg
    
    @staticmethod
    def validate_transform_matrix(T: np.ndarray, name: str = "T") -> bool:
        """
        Validate that a matrix is a proper 4x4 homogeneous transformation.
        
        Args:
            T: Matrix to validate
            name: Name for error messages
        
        Returns:
            True if valid, False otherwise (prints errors)
        """
        if not isinstance(T, np.ndarray):
            print(f"[ERROR] {name} is not a numpy array")
            return False
        
        if T.shape != (4, 4):
            print(f"[ERROR] {name} has invalid shape {T.shape}, expected (4, 4)")
            return False
        
        # Check bottom row is [0, 0, 0, 1]
        if not np.allclose(T[3, :], [0, 0, 0, 1]):
            print(f"[WARN] {name} bottom row is not [0, 0, 0, 1]: {T[3, :]}")
        
        # Check rotation matrix is orthogonal (R @ R.T = I)
        R = T[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            print(f"[WARN] {name} rotation matrix is not orthogonal")
            print(f"  R @ R.T =\n{R @ R.T}")
        
        # Check determinant is +1 (proper rotation, not reflection)
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=1e-6):
            print(f"[WARN] {name} rotation matrix determinant is {det:.6f}, expected 1.0")
        
        return True


# Example usage and unit tests
if __name__ == "__main__":
    print("Transform Utils - Unit Tests")
    print("=" * 70)
    
    # Test 1: Create transform matrix
    print("\n[Test 1] Create transform matrix")
    T_flange_camera = TransformUtils.create_transform_matrix(
        translation=[-11.08, -53.4, 24.757],
        rotation_deg=[0, 0, 0],
        input_units="mm"
    )
    print(f"T_flange_camera (from [-11.08, -53.4, 24.757] mm):")
    print(T_flange_camera)
    print(f"Position in meters: {T_flange_camera[:3, 3]}")
    
    # Test 2: Unit conversion
    print("\n[Test 2] Unit conversion")
    T_mm = np.eye(4)
    T_mm[:3, 3] = [100, 200, 300]  # mm
    print(f"Original (mm): {T_mm[:3, 3]}")
    
    T_m = TransformUtils.convert_robodk_pose_to_meters(T_mm)
    print(f"Converted to meters: {T_m[:3, 3]}")
    
    T_back = TransformUtils.convert_meters_to_robodk_pose(T_m)
    print(f"Converted back to mm: {T_back[:3, 3]}")
    
    # Test 3: Apply offset
    print("\n[Test 3] Apply offset in frame")
    T_base_fruit = np.eye(4)
    T_base_fruit[:3, 3] = [0.5, 0.3, 0.4]  # meters
    
    T_offset = TransformUtils.apply_offset_in_frame(
        T_base_frame=T_base_fruit,
        offset_mm=[10, 20, -50],  # mm
        rotation_deg=None
    )
    
    print(f"Original position (m): {T_base_fruit[:3, 3]}")
    print(f"Offset applied: [10, 20, -50] mm")
    print(f"New position (m): {T_offset[:3, 3]}")
    print(f"Delta (mm): {(T_offset[:3, 3] - T_base_fruit[:3, 3]) * 1000}")
    
    # Test 4: Joint deltas
    print("\n[Test 4] Apply joint deltas")
    current_joints = [0, -90, 90, 0, 90, 0]
    deltas = [0, 0, 0, 0, 0, 360]  # Full rotation on wrist
    new_joints = TransformUtils.apply_joint_deltas(current_joints, deltas)
    print(f"Current joints: {current_joints}")
    print(f"Deltas: {deltas}")
    print(f"New joints: {new_joints}")
    
    # Test 5: Validation
    print("\n[Test 5] Matrix validation")
    T_valid = np.eye(4)
    print(f"Valid matrix: {TransformUtils.validate_transform_matrix(T_valid, 'T_valid')}")
    
    T_invalid = np.zeros((3, 3))
    print(f"Invalid matrix: {TransformUtils.validate_transform_matrix(T_invalid, 'T_invalid')}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
