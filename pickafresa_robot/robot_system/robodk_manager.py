"""
RoboDK Integration Manager for Robot PnP Testing

Handles all RoboDK interactions including:
- Station loading
- Robot selection and configuration
- Target discovery and creation
- Movement execution with confirmations
- Reference frame visualization

Features:
- Auto-discover targets from RDK station
- Create dynamic targets for fruit positions
- Simulate or run on real robot
- Movement with user confirmations
- Collision checking

# @aldrick-t, 2025
"""

import time
import struct
import platform
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    from robolink import Robolink, ITEM_TYPE_ROBOT, ITEM_TYPE_TARGET, ITEM_TYPE_FRAME, RUNMODE_SIMULATE, RUNMODE_RUN_ROBOT
    from robodk import robomath
    HAVE_ROBODK = True
except ImportError:
    HAVE_ROBODK = False
    print("Warning: RoboDK Python API not available. Install with: pip install robodk")

# Keyboard library requires root/admin on macOS and causes threading errors
# Disable on macOS to avoid OSError: Error 13 - Must be run as administrator
IS_MACOS = platform.system() == "Darwin"

try:
    import keyboard
    # Disable keyboard on macOS due to permission requirements
    HAVE_KEYBOARD = not IS_MACOS
except ImportError:
    HAVE_KEYBOARD = False


class RobotSafetyError(Exception):
    """Critical safety error requiring immediate stop."""
    pass


class RobotEmergencyStopError(RobotSafetyError):
    """Emergency stop activated on robot."""
    pass


class RobotCollisionError(RobotSafetyError):
    """Collision detected during robot operation."""
    pass


class RoboDKManager:
    """Manager for RoboDK station, robot, and movements."""
    
    def __init__(
        self,
        station_file: Path,
        robot_model: str = "UR3e",
        run_mode: str = "simulate",
        logger: Optional[any] = None
    ):
        """
        Initialize RoboDK manager.
        
        Args:
            station_file: Path to .rdk station file
            robot_model: Name of robot model to use
            run_mode: "simulate" or "real_robot"
            logger: Logger instance (optional)
        """
        if not HAVE_ROBODK:
            raise RuntimeError("RoboDK API not available. Cannot initialize RoboDKManager.")
        
        self.station_file = station_file
        self.robot_model = robot_model
        self.run_mode = run_mode
        self.logger = logger
        
        # RoboDK connection
        self.RDK: Optional[Robolink] = None
        self.robot = None
        
        # Discovered targets
        self.targets: Dict[str, any] = {}
        
        # Created dynamic targets
        self.dynamic_targets: Dict[str, any] = {}
        
        # Safety state
        self.emergency_stop_active: bool = False
        self.collision_halt_active: bool = False
        self.last_safety_check: float = 0.0
        self.safety_check_interval: float = 0.1  # Check every 100ms
        
        self._log_info("RoboDK Manager initialized")
    
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
    
    def _log_debug(self, message: str) -> None:
        """Log debug message."""
        if self.logger:
            self.logger.debug(message)
        else:
            # Debug messages only shown if verbose
            pass
    
    def _is_elbow_down(self, joints: any) -> bool:
        """
        Determine if a joint configuration is elbow-down.
        
        For UR robots, elbow-down typically means joint 2 (elbow) is negative.
        Joint 2 is the shoulder lift joint, and when negative, the elbow points down.
        
        Args:
            joints: Joint angles (can be robodk.robomath.Mat or list)
        
        Returns:
            True if elbow-down configuration
        """
        try:
            if hasattr(joints, 'list'):
                joint_list = joints.list()
            else:
                joint_list = list(joints)
            
            # For UR robots: Joint 2 (index 2, elbow joint)
            # Elbow down: joint[2] < 0
            elbow_joint = joint_list[2] if len(joint_list) > 2 else 0
            return elbow_joint < 0
        except Exception as e:
            self._log_debug(f"Could not determine elbow config: {e}")
            return False
    
    def _get_joint_config_flags(self, joints: any) -> List[int]:
        """
        Get joint configuration flags for UR robot.
        Returns [shoulder, elbow, wrist] configuration.
        
        For UR robots:
        - shoulder: 1 if joint[0] >= 0 else -1
        - elbow: 1 if joint[2] >= 0 (elbow up) else -1 (elbow down)
        - wrist: 1 if joint[4] >= 0 else -1
        
        Args:
            joints: Joint angles
        
        Returns:
            [shoulder_flag, elbow_flag, wrist_flag]
        """
        try:
            if hasattr(joints, 'list'):
                joint_list = joints.list()
            else:
                joint_list = list(joints)
            
            shoulder = 1 if joint_list[0] >= 0 else -1
            elbow = 1 if joint_list[2] >= 0 else -1  # 1 = elbow up, -1 = elbow down
            wrist = 1 if joint_list[4] >= 0 else -1
            
            return [shoulder, elbow, wrist]
        except Exception as e:
            self._log_debug(f"Could not get joint config flags: {e}")
            return [1, 1, 1]  # Default to elbow up
    
    def connect(self) -> bool:
        """
        Connect to RoboDK and load station.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._log_info("Connecting to RoboDK...")
            self.RDK = Robolink()
            
            # Check if the robot already exists in the current station
            # If it does, we don't need to load the station again
            test_robot = self.RDK.Item(self.robot_model, ITEM_TYPE_ROBOT)
            
            if test_robot.Valid():
                self._log_info(f"Station already loaded (found '{self.robot_model}')")
            elif self.station_file.exists():
                # Only load if robot not found (station not loaded)
                self._log_info(f"Loading station: {self.station_file}")
                self.RDK.AddFile(str(self.station_file))
            else:
                self._log_error(f"Station file not found: {self.station_file}")
                return False
            
            # Set run mode
            if self.run_mode.lower() == "real_robot":
                self._log_info("Setting run mode: REAL ROBOT")
                self.RDK.setRunMode(RUNMODE_RUN_ROBOT)
            else:
                self._log_info("Setting run mode: SIMULATION")
                self.RDK.setRunMode(RUNMODE_SIMULATE)
            
            self._log_info("[OK] Connected to RoboDK")
            return True
        
        except Exception as e:
            self._log_error(f"Failed to connect to RoboDK: {e}")
            return False
    
    def select_robot(self, robot_name: Optional[str] = None) -> bool:
        """
        Select robot from station.
        
        Args:
            robot_name: Name of robot (None = use default from config)
        
        Returns:
            True if successful, False otherwise
        """
        if self.RDK is None:
            self._log_error("Not connected to RoboDK")
            return False
        
        try:
            name = robot_name or self.robot_model
            self._log_info(f"Selecting robot: {name}")
            
            self.robot = self.RDK.Item(name, ITEM_TYPE_ROBOT)
            
            if not self.robot.Valid():
                self._log_error(f"Robot '{name}' not found in station")
                return False
            
            self._log_info(f"[OK] Robot selected: {name}")
            return True
        
        except Exception as e:
            self._log_error(f"Failed to select robot: {e}")
            return False
    
    def set_speed(
        self,
        linear_speed: float = 100.0,
        joint_speed: float = 60.0
    ) -> bool:
        """
        Set robot speed.
        
        Args:
            linear_speed: Linear speed (mm/s or %)
            joint_speed: Joint speed (deg/s or %)
        
        Returns:
            True if successful
        """
        if self.robot is None:
            self._log_error("No robot selected")
            return False
        
        try:
            self.robot.setSpeed(linear_speed, joint_speed)
            self._log_info(f"Speed set: linear={linear_speed}, joint={joint_speed}")
            return True
        except Exception as e:
            self._log_error(f"Failed to set speed: {e}")
            return False
    
    def check_robot_safety_status(self) -> Tuple[bool, Optional[str]]:
        """
        Check robot safety status including emergency stop, protective stop, and errors.
        
        For UR robots connected via RoboDK driver:
        - Checks robot connection status
        - Detects emergency stop activation
        - Detects protective stops
        - Checks for robot errors
        
        Returns:
            Tuple of (is_safe: bool, error_message: Optional[str])
            is_safe=False means critical safety issue detected
        """
        if self.robot is None or self.RDK is None:
            return (False, "Robot not initialized")
        
        try:
            # For real robot mode, check robot status
            if self.run_mode.lower() == "real_robot":
                # Check if robot is connected
                if not self.robot.Connect():
                    return (False, "Robot disconnected")
                
                # Try to get robot joints - if this fails, robot has an issue
                try:
                    joints = self.robot.Joints()
                    if joints is None:
                        return (False, "Cannot read robot position - possible safety stop")
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Check for emergency stop indicators
                    if "emergency" in error_str or "e-stop" in error_str or "estop" in error_str:
                        self.emergency_stop_active = True
                        return (False, "Emergency stop activated on robot")
                    
                    # Check for protective stop
                    if "protective" in error_str or "safeguard" in error_str:
                        return (False, "Protective stop activated on robot")
                    
                    # Check for communication errors
                    if "timeout" in error_str or "connection" in error_str:
                        return (False, f"Robot communication error: {e}")
                    
                    # Generic error
                    return (False, f"Robot error: {e}")
                
                # Check robot status flags via RoboDK
                try:
                    # Use RoboDK's robot.Busy() to check if robot is executing
                    # If robot is in error state, this will fail
                    busy_status = self.robot.Busy()
                    
                    # Check for collision detection during movement
                    if self.RDK.Collisions() > 0:
                        self.collision_halt_active = True
                        return (False, "Collision detected by RoboDK")
                    
                except Exception as status_error:
                    return (False, f"Cannot read robot status: {status_error}")
            
            # Robot is safe
            return (True, None)
            
        except Exception as e:
            self._log_error(f"Safety check failed: {e}")
            return (False, f"Safety check error: {e}")
    
    def reset_safety_state(self) -> bool:
        """
        Reset safety state flags after manual intervention.
        
        Should only be called after:
        - Emergency stop has been physically reset on teach pendant
        - Collision has been cleared
        - Robot is verified to be in safe state
        
        Returns:
            True if reset successful
        """
        try:
            # Verify robot is actually safe before resetting
            is_safe, error_msg = self.check_robot_safety_status()
            
            if not is_safe:
                self._log_error(f"Cannot reset safety state - robot still unsafe: {error_msg}")
                return False
            
            # Reset flags
            self.emergency_stop_active = False
            self.collision_halt_active = False
            
            self._log_info("Safety state reset - system ready to resume")
            return True
            
        except Exception as e:
            self._log_error(f"Failed to reset safety state: {e}")
            return False
    
    def _recover_from_api_error(self) -> bool:
        """
        Recover from RoboDK API communication errors.
        
        When the RoboDK API encounters errors (like UnicodeDecodeError or struct errors),
        the connection can become corrupted. This method attempts to recover by:
        1. Clearing the socket buffer
        2. Reconnecting if necessary
        3. Restoring robot state
        
        Returns:
            True if recovery successful, False otherwise
        """
        try:
            self._log_debug("Attempting API recovery...")
            
            # Try to get robot position to test connection
            try:
                _ = self.robot.Joints()
                self._log_debug("API connection OK, no recovery needed")
                return True
            except Exception as e:
                self._log_debug(f"Connection test failed: {e}, attempting recovery...")
            
            # Save current robot reference
            robot_name = self.robot_model if self.robot is None else self.robot.Name()
            
            # Try to reconnect to RoboDK
            self._log_debug("Reconnecting to RoboDK...")
            try:
                # Close existing connection if any
                if self.RDK is not None:
                    try:
                        self.RDK.Close()
                    except:
                        pass
                
                # Create new connection
                self.RDK = Robolink()
                
                # Re-select robot
                self.robot = self.RDK.Item(robot_name, ITEM_TYPE_ROBOT)
                
                if not self.robot.Valid():
                    self._log_error(f"Failed to re-select robot: {robot_name}")
                    return False
                
                # Test connection
                _ = self.robot.Joints()
                
                self._log_debug("API recovery successful")
                return True
                
            except Exception as e:
                self._log_error(f"API recovery failed: {e}")
                return False
                
        except Exception as e:
            self._log_error(f"Recovery attempt failed: {e}")
            return False
    
    def enable_motion_planner(
        self,
        enable: bool = True,
        max_time_ms: int = 5000,
        max_iterations: int = 1000
    ) -> bool:
        """
        Configure collision-aware movement settings.
        
        Note: RoboDK's Python API doesn't expose direct PRM motion planner control.
        This method configures collision detection and movement verification settings
        that help find collision-free paths through our own planning strategies.
        
        Args:
            enable: True to enable collision checking, False to disable
            max_time_ms: Maximum planning time (for future extensions)
            max_iterations: Maximum iterations (for future extensions)
        
        Returns:
            True if successful
        """
        if self.RDK is None:
            self._log_error("Not connected to RoboDK")
            return False
        
        try:
            if enable:
                self._log_info(f"Enabling collision-aware movements...")
                
                # Enable collision checking (this is the main RoboDK API feature)
                self.RDK.setCollisionActive(1)
                
                self._log_info("[OK] Collision checking enabled")
            else:
                self._log_info("Disabling collision checking...")
                self.RDK.setCollisionActive(0)
                self._log_info("[OK] Collision checking disabled")
            
            return True
            
        except Exception as e:
            self._log_error(f"Failed to configure collision checking: {e}")
            return False
    
    def discover_targets(self) -> Dict[str, any]:
        """
        Auto-discover all targets in the station.
        
        Returns:
            Dictionary of target_name: target_object
        """
        if self.RDK is None:
            self._log_error("Not connected to RoboDK")
            return {}
        
        try:
            self._log_info("Discovering targets in station...")
            
            # Get all items in station
            item_list = self.RDK.ItemList(ITEM_TYPE_TARGET)
            
            targets = {}
            for item in item_list:
                if item.Valid():
                    name = item.Name()
                    targets[name] = item
            
            self._log_info(f"[OK] Discovered {len(targets)} targets: {list(targets.keys())}")
            self.targets = targets
            
            return targets
        
        except Exception as e:
            self._log_error(f"Failed to discover targets: {e}")
            return {}
    
    def create_target_from_pose(
        self,
        name: str,
        T_base_target: np.ndarray,
        create_frame: bool = True,
        frame_size: float = 50.0,
        color: Optional[List[float]] = None
    ) -> Optional[any]:
        """
        Create a target at specified pose in robot base frame.
        
        Args:
            name: Name for the target
            T_base_target: 4x4 transformation matrix (base → target)
            create_frame: Also create a reference frame for visualization
            frame_size: Size of frame axes (mm)
            color: RGB color [r, g, b] 0-255 (optional)
        
        Returns:
            Created target object or None if failed
        """
        if self.RDK is None or self.robot is None:
            self._log_error("Not connected or no robot selected")
            return None
        
        try:
            # Convert numpy matrix to RoboDK matrix
            pose_robodk = robomath.Mat(T_base_target.tolist())
            
            # Get robot base reference frame to ensure targets are created in absolute coordinates
            robot_base_frame = self.robot.Parent()  # Get robot's parent (usually the station/world)
            
            # Create target with robot as parent (this makes it use robot base coordinates)
            target = self.RDK.AddTarget(name, itemparent=robot_base_frame, itemrobot=self.robot)
            target.setPose(pose_robodk)
            
            self._log_info(f"[OK] Created target: {name} (parent: {robot_base_frame.Name() if robot_base_frame.Valid() else 'None'})")
            
            # Create reference frame for visualization
            if create_frame:
                # Create frame with same parent as target to ensure consistent coordinate system
                frame = self.RDK.AddFrame(f"{name}_frame", itemparent=robot_base_frame)
                frame.setPose(pose_robodk)
                
                # Set frame size
                frame.setVisible(True)
                
                # Set color if specified
                if color:
                    frame.setColor(color)
                
                self._log_info(f"[OK] Created reference frame: {name}_frame (parent: {robot_base_frame.Name() if robot_base_frame.Valid() else 'None'})")
            
            # Store in dynamic targets
            self.dynamic_targets[name] = target
            
            return target
        
        except Exception as e:
            self._log_error(f"Failed to create target '{name}': {e}")
            return None
    
    def create_target_from_joints(
        self,
        name: str,
        joints: List[float],
        color: Optional[List[float]] = None
    ) -> bool:
        """
        Create a target with specific joint configuration.
        
        This is useful for joint-space control where you want to specify exact
        joint angles rather than Cartesian positions. The target will be created
        at the Cartesian pose corresponding to the given joint configuration.
        
        Args:
            name: Name for the target
            joints: List of 6 joint angles in degrees [j0, j1, j2, j3, j4, j5]
            color: RGB color [r, g, b] 0-255 (optional)
        
        Returns:
            True if successful, False otherwise
        """
        if self.RDK is None or self.robot is None:
            self._log_error("Not connected or no robot selected")
            return False
        
        try:
            # Validate joint count
            if len(joints) != 6:
                self._log_error(f"Invalid joint count: expected 6, got {len(joints)}")
                return False
            
            # Get robot base reference frame
            robot_base_frame = self.robot.Parent()
            
            # Save current robot configuration
            current_joints = self.robot.Joints()
            
            # Temporarily set robot to target joints to get the Cartesian pose
            joints_mat = robomath.Mat([joints])
            self.robot.setJoints(joints_mat.tr())
            
            # Get the Cartesian pose at these joints
            target_pose = self.robot.Pose()
            
            # Restore original robot configuration
            self.robot.setJoints(current_joints)
            
            # Create target with the pose
            target = self.RDK.AddTarget(name, itemparent=robot_base_frame, itemrobot=self.robot)
            target.setPose(target_pose)
            
            # IMPORTANT: Set the joint configuration
            # This ensures the robot uses these specific joints when moving to this target
            target.setJoints(joints_mat.tr())
            
            # Set color if specified
            if color:
                target.setColor(color)
            
            self._log_info(f"[OK] Created joint-space target: {name}")
            self._log_debug(f"  Joints: {[f'{j:.2f}' for j in joints]}")
            
            # Store in dynamic targets
            self.dynamic_targets[name] = target
            
            return True
        
        except Exception as e:
            self._log_error(f"Failed to create joint-space target '{name}': {e}")
            return False
    
    def cleanup_dynamic_targets(self, fixed_targets: Optional[List[str]] = None) -> int:
        """
        Clean up all dynamically created targets and frames from RoboDK station.
        Preserves fixed targets (like Home, Foto) that should not be deleted.
        
        Args:
            fixed_targets: List of target names to preserve (e.g., ["Home", "Foto"])
        
        Returns:
            Number of targets/frames cleaned up
        """
        if self.RDK is None:
            self._log_error("Not connected to RoboDK")
            return 0
        
        if fixed_targets is None:
            fixed_targets = []
        
        cleaned_count = 0
        
        # Clean up ALL targets in the station (not just tracked ones)
        try:
            all_targets = self.RDK.ItemList(ITEM_TYPE_TARGET)
            for target in all_targets:
                if target and target.Valid():
                    target_name = target.Name()
                    # Delete if not in fixed list
                    if target_name not in fixed_targets:
                        try:
                            target.Delete()
                            self._log_info(f"Deleted target: {target_name}")
                            cleaned_count += 1
                        except Exception as e:
                            self._log_warn(f"Failed to delete target '{target_name}': {e}")
        except Exception as e:
            self._log_warn(f"Error during target cleanup: {e}")
        
        # Clean up ALL frames in the station
        try:
            all_frames = self.RDK.ItemList(ITEM_TYPE_FRAME)
            for frame in all_frames:
                if frame and frame.Valid():
                    frame_name = frame.Name()
                    # Skip robot base frame and other system frames
                    if frame_name not in fixed_targets and not frame_name.endswith(" Base"):
                        try:
                            frame.Delete()
                            self._log_info(f"Deleted frame: {frame_name}")
                            cleaned_count += 1
                        except Exception as e:
                            self._log_warn(f"Failed to delete frame '{frame_name}': {e}")
        except Exception as e:
            self._log_warn(f"Error during frame cleanup: {e}")
        
        # Clear tracking dict
        self.dynamic_targets.clear()
        
        if cleaned_count > 0:
            self._log_info(f"[OK] Cleaned up {cleaned_count} targets/frames")
        
        return cleaned_count
    
    def get_target(self, name: str) -> Optional[any]:
        """
        Get target by name (from discovered or dynamic targets).
        
        Args:
            name: Target name
        
        Returns:
            Target object or None if not found
        """
        # Check dynamic targets first
        if name in self.dynamic_targets:
            target = self.dynamic_targets[name]
            # Validate that the target is still valid in RoboDK
            if target and target.Valid():
                return target
            else:
                # Target became invalid, remove from cache
                self._log_warn(f"Dynamic target '{name}' became invalid, removing from cache")
                del self.dynamic_targets[name]
        
        # Check discovered targets
        if name in self.targets:
            target = self.targets[name]
            if target and target.Valid():
                return target
            else:
                self._log_warn(f"Discovered target '{name}' became invalid")
                del self.targets[name]
        
        # Try to find in RoboDK station
        if self.RDK:
            target = self.RDK.Item(name, ITEM_TYPE_TARGET)
            if target.Valid():
                return target
        
        self._log_warn(f"Target not found: {name}")
        return None
    
    def get_tcp_pose(self) -> Optional[np.ndarray]:
        """
        Get current TCP pose in robot base frame.
        
        Returns:
            4x4 transformation matrix or None if failed
        """
        if self.robot is None:
            self._log_error("No robot selected")
            return None
        
        try:
            pose_robodk = self.robot.Pose()
            
            # RoboDK's Mat object can be converted row by row
            # Build 4x4 matrix from the rows
            T = np.array([
                pose_robodk[0, :],  # First row
                pose_robodk[1, :],  # Second row
                pose_robodk[2, :],  # Third row
                pose_robodk[3, :]   # Fourth row
            ], dtype=np.float64)
            
            # Squeeze any extra dimensions (RoboDK may return (4,4,1) instead of (4,4))
            T = np.squeeze(T)
            
            # Verify it's a proper 4x4 matrix
            if T.shape != (4, 4):
                self._log_error(f"Unexpected pose shape: {T.shape}")
                return None
            
            return T
        except Exception as e:
            self._log_error(f"Failed to get TCP pose: {e}")
            import traceback
            self._log_error(traceback.format_exc())
            return None
    
    def check_collision(
        self,
        target_name: str,
        move_type: str = "joint"
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if movement to target would cause collision.
        
        Args:
            target_name: Name of target to check
            move_type: "joint" or "linear"
        
        Returns:
            (has_collision, collision_info): Tuple of collision status and info string
        """
        if self.robot is None:
            return (False, "No robot selected")
        
        target = self.get_target(target_name)
        if target is None:
            return (False, f"Target '{target_name}' not found")
        
        try:
            # Enable collision checking
            self.RDK.setCollisionActive(1)
            
            # Save current robot state - with error recovery
            try:
                current_joints = self.robot.Joints()
                target_pose = target.Pose()
            except (UnicodeDecodeError, struct.error) as e:
                self._log_warn(f"API error getting robot state: {e}, attempting recovery...")
                if self._recover_from_api_error():
                    current_joints = self.robot.Joints()
                    target_pose = target.Pose()
                else:
                    raise
            
            # First check if target is reachable (IK solution exists)
            target_joints = self.robot.SolveIK(target_pose)
            
            if target_joints is None or len(target_joints) == 0:
                error_msg = "Cannot solve IK for target (unreachable or out of workspace)"
                self._log_warn(f"{error_msg} - Target: {target_name}")
                target_pos = target_pose.Pos()
                self._log_debug(f"Target position: [{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}] mm")
                return (True, error_msg)
            
            # Use RoboDK's built-in test methods for accurate collision checking
            # Note: MoveJ_Test/MoveL_Test signatures:
            # - MoveJ_Test(j1, j2, minstep_deg=-1) requires Mat objects or lists of joints
            # - MoveL_Test(j1, pose, minstep_mm=-1) requires joints and pose
            use_test_methods = True  # Re-enabled with correct API usage
            
            if use_test_methods:
                try:
                    # Get current joints as a list for proper formatting
                    current_joints_list = current_joints.list() if hasattr(current_joints, 'list') else list(current_joints)
                    target_joints_list = target_joints.list() if hasattr(target_joints, 'list') else list(target_joints)
                    
                    # MoveJ_Test and MoveL_Test require proper data types
                    # Signature: MoveJ_Test(j1, j2, minstep_deg=-1) -> collision_count
                    collision_result = None
                    
                    if move_type.lower() == "linear":
                        # For linear moves: MoveL_Test(joints, pose, minstep_mm=-1)
                        # Note: target_pose should be used, not target_joints
                        try:
                            collision_result = self.robot.MoveL_Test(current_joints_list, target_pose, minstep_mm=5.0)
                        except (TypeError, AttributeError, Exception) as e:
                            self._log_debug(f"MoveL_Test failed: {e}, using fallback")
                            use_test_methods = False
                    else:
                        # For joint moves, use MoveJ_Test with proper signature
                        try:
                            # minstep_deg: smaller = more accurate but slower (default -1 = auto)
                            collision_result = self.robot.MoveJ_Test(current_joints_list, target_joints_list, minstep_deg=5.0)
                        except (TypeError, AttributeError, Exception) as e:
                            self._log_debug(f"MoveJ_Test failed: {e}, using fallback")
                            use_test_methods = False
                    
                    if use_test_methods and collision_result is not None:
                        # Restore robot position after test
                        self.robot.setJoints(current_joints)
                        
                        # collision_result: 0 = no collision, >0 = number of collision pairs
                        if collision_result > 0:
                            collision_info = f"Collision detected during {move_type} path (count: {collision_result})"
                            self._log_debug(f"{collision_info}")
                            return (True, collision_info)
                        
                        # No collision detected
                        return (False, None)
                    
                except (AttributeError, TypeError) as e:
                    # MoveJ_Test or MoveL_Test not available or wrong signature
                    self._log_debug(f"Move{move_type[0].upper()}_Test() not available or incompatible: {e}")
                    use_test_methods = False
            
            # Use fallback collision checking
            return self._check_collision_fallback(target_name, target_pose, target_joints, move_type, current_joints)
            
        except Exception as e:
            self._log_error(f"Collision check failed: {e}")
            import traceback
            self._log_debug(f"Traceback: {traceback.format_exc()}")
            
            # Check if this is an API communication error
            if isinstance(e, (UnicodeDecodeError, struct.error)) or "unpack" in str(e) or "decode" in str(e):
                self._log_warn("Detected API communication error, attempting recovery...")
                if self._recover_from_api_error():
                    self._log_info("API recovered, retrying collision check...")
                    # Retry collision check once after recovery
                    try:
                        return self._check_collision_fallback(target_name, target.Pose(), 
                                                              self.robot.SolveIK(target.Pose()), 
                                                              move_type, self.robot.Joints())
                    except Exception as retry_error:
                        self._log_error(f"Retry after recovery failed: {retry_error}")
            
            # Restore robot position on error
            try:
                if 'current_joints' in locals():
                    self.robot.setJoints(current_joints)
            except:
                pass
            return (False, f"Collision check error: {e}")
    
    def _check_collision_fallback(
        self,
        target_name: str,
        target_pose: any,
        target_joints: any,
        move_type: str,
        current_joints: any
    ) -> Tuple[bool, Optional[str]]:
        """
        Fallback collision checking when MoveJ_Test/MoveL_Test not available.
        
        Args:
            target_name: Target name
            target_pose: Target pose
            target_joints: Target joint configuration
            move_type: Movement type
            current_joints: Current joint configuration
        
        Returns:
            (has_collision, collision_info)
        """
        try:
            # Check collision at target position
            self.robot.setJoints(target_joints)
            collision_status = self.RDK.Collisions()
            
            # Also check intermediate points along the path
            path_collision = False
            current_pose = self.robot.Pose()
            
            if move_type.lower() == "linear":
                # For linear moves, check more intermediate points
                num_samples = 10
                
                for i in range(1, num_samples):
                    fraction = i / num_samples
                    
                    # Interpolate pose
                    interp_pose = current_pose.copy()
                    current_pos = current_pose.Pos()
                    target_pos = target_pose.Pos()
                    
                    interp_pos = [
                        current_pos[j] + (target_pos[j] - current_pos[j]) * fraction
                        for j in range(3)
                    ]
                    interp_pose.setPos(interp_pos)
                    
                    # Check IK and collision at this point
                    interp_joints = self.robot.SolveIK(interp_pose)
                    if interp_joints:
                        self.robot.setJoints(interp_joints)
                        if self.RDK.Collisions() > 0:
                            path_collision = True
                            collision_status = max(collision_status, self.RDK.Collisions())
                            break
            
            # Restore original position
            self.robot.setJoints(current_joints)
            
            if collision_status > 0 or path_collision:
                collision_info = f"Collision detected (count: {collision_status})"
                if path_collision:
                    collision_info += " [path collision]"
                self._log_debug(collision_info)
                return (True, collision_info)
            
            return (False, None)
            
        except Exception as e:
            self._log_debug(f"Fallback collision check error: {e}")
            self.robot.setJoints(current_joints)
            return (False, f"Collision check error: {e}")
    
    def move_to_target_with_collision_avoidance(
        self,
        target_name: str,
        move_type: str = "joint",
        confirm: bool = True,
        highlight: bool = True,
        enable_collision_avoidance: bool = True,
        collision_config: Optional[Dict[str, any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Move robot to target with collision avoidance fallback strategies.
        
        Fallback order:
        1. Try requested move_type (MoveJ or MoveL)
        2. If collision, try alternative move type
        3. If still collision, try alternative IK solution
        4. If still collision, try with intermediate waypoints
        5. If all fail, abort and notify user
        
        Args:
            target_name: Name of target to move to
            move_type: "joint" (MoveJ) or "linear" (MoveL)
            confirm: Ask user confirmation before moving
            highlight: Highlight target before moving
            enable_collision_avoidance: Enable collision checking and avoidance
            collision_config: Collision avoidance configuration dict (optional)
        
        Returns:
            Tuple of (success: bool, message: str or None)
        """
        if collision_config is None:
            collision_config = {}
        
        if not enable_collision_avoidance:
            # Collision avoidance disabled, use standard movement
            result = self.move_to_target(target_name, move_type, confirm, highlight)
            return (result, None if result else "Movement failed")
        
        if self.robot is None:
            self._log_error("No robot selected")
            return (False, "No robot selected")
        
        target = self.get_target(target_name)
        if target is None:
            self._log_error(f"Target '{target_name}' not found")
            return (False, f"Target '{target_name}' not found")
        
        # Highlight target
        if highlight:
            try:
                self.RDK.ShowTarget(target)
                time.sleep(0.5)
            except:
                pass
        
        # Check collision with requested move type
        self._log_info(f"Checking collision for {move_type} to '{target_name}'...")
        has_collision, collision_info = self.check_collision(target_name, move_type)
        
        if not has_collision:
            # No collision in pre-check, try the movement
            self._log_debug("Pre-check passed, attempting movement...")
            result = self.move_to_target(target_name, move_type, confirm, highlight=False)
            if result:
                return (True, None)
            else:
                # Movement failed despite passing pre-check (path collision or other issue)
                self._log_warn(f"Movement failed despite passing pre-check, trying fallback strategies...")
                has_collision = True  # Treat as collision to trigger fallback strategies
                collision_info = "Movement failed (likely path collision)"
        
        # Collision detected (or movement failed), try fallback strategies
        if has_collision:
            self._log_warn(f"Collision detected with {move_type}: {collision_info}")
            self._log_info("Attempting collision avoidance strategies...")
        
        # Strategy 1: Try alternative IK configuration (HIGHEST PRIORITY - fixes elbow collisions)
        # This is the main strategy: find an elbow-up configuration that doesn't hit the table
        self._log_info("Strategy 1: Trying alternative IK configuration (prioritizing elbow-up)...")
        max_ik_attempts = collision_config.get('max_ik_attempts', 16)
        success = self._try_alternative_ik(target_name, move_type, confirm, max_ik_attempts)
        if success:
            return (True, "Used alternative IK solution")
        
        self._log_warn("Strategy 1 failed: No collision-free IK solution found")
        
        # Strategy 2: Try alternative move type
        alt_move_type = "linear" if move_type.lower() == "joint" else "joint"
        self._log_info(f"Strategy 2: Trying {alt_move_type} move...")
        
        has_collision, collision_info = self.check_collision(target_name, alt_move_type)
        if not has_collision:
            self._log_info(f"[OK] Alternative move type ({alt_move_type}) passed pre-check, attempting movement...")
            result = self.move_to_target(target_name, alt_move_type, confirm, highlight=False)
            if result:
                message = f"Used alternative move type: {alt_move_type}"
                self._log_info(f"[OK] {message}")
                return (True, message)
            else:
                self._log_warn("Strategy 2 movement failed (collision during path or other error)")
        else:
            self._log_warn(f"Strategy 2 pre-check failed: {collision_info}")
        
        # Strategy 3: Try intelligent path sampling (collision-free path finding)
        motion_planner_config = collision_config.get('motion_planner', {})
        if motion_planner_config.get('enabled', True):
            self._log_info("Strategy 3: Trying intelligent path sampling...")
            max_time = motion_planner_config.get('max_planning_time_ms', 5000)
            max_iter = motion_planner_config.get('max_iterations', 1000)
            success = self._try_motion_planner(target_name, move_type, confirm, max_time, max_iter)
            if success:
                return (True, "Used collision-free path sampling")
            
            self._log_warn("Strategy 3 failed: Path sampling could not find collision-free path")
        else:
            self._log_debug("Strategy 3 (path sampling) disabled in configuration")
        
        # Strategy 4: Try with intermediate waypoints
        self._log_info("Strategy 4: Trying with intermediate waypoints...")
        success = self._try_with_waypoints(target_name, move_type, confirm)
        if success:
            return (True, "Used intermediate waypoints")
        
        self._log_warn("Strategy 4 failed: Could not find collision-free path with waypoints")
        
        # All strategies failed
        self._log_error("="*60)
        self._log_error("COLLISION AVOIDANCE FAILED")
        self._log_error(f"Cannot reach target '{target_name}' without collision")
        self._log_error("All fallback strategies exhausted")
        self._log_error("="*60)
        
        # Get run_mode and simulation_mode from collision_config
        run_mode = collision_config.get('run_mode', 'manual_confirm')
        simulation_mode = collision_config.get('simulation_mode', 'simulate')
        is_autonomous = run_mode == 'autonomous'
        is_real_robot = simulation_mode == 'real_robot'
        
        # CRITICAL SAFETY: Always prompt user about collision
        # In autonomous mode with simulation, auto-abort
        # In real_robot mode, NEVER allow force move
        print("\n" + "="*60)
        print("⚠️  COLLISION WARNING - MOVEMENT CANNOT BE COMPLETED SAFELY")
        print("="*60)
        print(f"Target: {target_name}")
        print(f"Mode: {run_mode} | Robot: {simulation_mode}")
        print("="*60)
        
        if is_real_robot:
            # REAL ROBOT: Absolute safety - no force option
            print("❌ REAL ROBOT MODE - Movement automatically ABORTED for safety")
            print("   Collision detected with physical robot")
            print("   Manual intervention or RoboDK station adjustment required")
            self._log_error("Real robot mode: Auto-abort due to collision (safety protocol)")
            return (False, "Collision detected - auto-aborted for real robot safety")
        
        elif is_autonomous:
            # AUTONOMOUS + SIMULATION: Auto-abort but inform user
            print("🤖 AUTONOMOUS MODE - Movement automatically ABORTED")
            print("   Collision detected in simulation")
            print("   System will skip this operation and continue if possible")
            self._log_warn("Autonomous mode: Auto-abort due to collision")
            input("\nPress Enter to acknowledge and continue...")
            return (False, "Collision detected - auto-aborted in autonomous mode")
        
        else:
            # MANUAL MODE + SIMULATION: Give user options
            print("\nAll collision avoidance strategies failed. Options:")
            print("  1. Abort movement (SAFE - recommended)")
            print("  2. Force movement anyway (DANGEROUS - simulation only)")
            print("  3. Manually intervene in RoboDK station")
            
            response = input("\nChoose option [1/2/3]: ").strip()
            
            if response == "2":
                # Only allow force in simulation mode
                self._log_warn("⚠️  USER OVERRIDE: Forcing movement despite collision risk")
                self._log_warn("   This should ONLY be used in simulation for testing!")
                confirm_force = input("Type 'FORCE' to confirm dangerous override: ").strip()
                
                if confirm_force == "FORCE":
                    # Disable collision checking and force move
                    self.RDK.setCollisionActive(0)
                    result = self.move_to_target(target_name, move_type, confirm=False, highlight=False)
                    self.RDK.setCollisionActive(1)  # Re-enable
                    message = "Movement forced by user (collision checking disabled)" if result else "Forced movement failed"
                    return (result, message)
                else:
                    self._log_info("Force override cancelled")
                    return (False, "User cancelled force override")
            else:
                self._log_info("Movement aborted by user")
                return (False, "Collision detected - movement aborted")
    
    def _try_alternative_ik(
        self,
        target_name: str,
        move_type: str,
        confirm: bool,
        max_attempts: int = 16
    ) -> bool:
        """
        Try alternative IK solutions to avoid collision.
        Uses RoboDK's SolveIK_All() to get ALL valid joint configurations.
        Prioritizes elbow-up configurations (except for Foto target).
        
        Args:
            target_name: Target name
            move_type: Movement type
            confirm: User confirmation
            max_attempts: Maximum number of IK configurations to try
        
        Returns:
            True if successful
        """
        try:
            target = self.get_target(target_name)
            if target is None:
                return False
            
            # Get target pose - with error recovery
            try:
                target_pose = target.Pose()
            except (UnicodeDecodeError, struct.error) as e:
                self._log_warn(f"API error getting target pose: {e}, attempting recovery...")
                if self._recover_from_api_error():
                    target = self.get_target(target_name)
                    target_pose = target.Pose()
                else:
                    raise
            
            # Determine if this is the Foto target (elbow-down allowed)
            is_foto_target = "foto" in target_name.lower()
            
            self._log_info(f"Testing ALL valid joint configurations from RoboDK...")
            if not is_foto_target:
                self._log_info("Prioritizing elbow-up configurations to avoid collisions")
            
            # Use RoboDK's SolveIK_All to get all valid joint configurations
            # This returns all possible configurations for the robot
            try:
                all_joint_solutions = self.robot.SolveIK_All(target_pose)
                
                if all_joint_solutions is None or len(all_joint_solutions) == 0:
                    self._log_warn("No IK solutions found for target pose")
                    return False
                
                self._log_info(f"Found {len(all_joint_solutions)} valid joint configurations from RoboDK")
                
                # Sort configurations: elbow-up first (unless Foto target)
                sorted_solutions = []
                for joints in all_joint_solutions:
                    # Ensure joints is in the right format (Mat object or list compatible)
                    # SolveIK_All may return Mat objects or lists depending on RoboDK version
                    is_elbow_down = self._is_elbow_down(joints)
                    elbow_status = "elbow-down" if is_elbow_down else "elbow-up"
                    
                    # Priority: elbow-up = 0 (first), elbow-down = 1 (last)
                    # Unless it's Foto target, then reverse priority
                    if is_foto_target:
                        priority = 0 if is_elbow_down else 1
                    else:
                        priority = 1 if is_elbow_down else 0
                    
                    sorted_solutions.append((priority, elbow_status, joints))
                
                # Sort by priority (0 first, 1 last)
                sorted_solutions.sort(key=lambda x: x[0])
                
                # Try each configuration in order
                for config_idx, (priority, elbow_status, joints) in enumerate(sorted_solutions):
                    if config_idx >= max_attempts:
                        break
                    
                    self._log_debug(f"Testing configuration {config_idx + 1}/{len(sorted_solutions)} ({elbow_status})...")
                    
                    try:
                        # Save current robot position
                        current_joints = self.robot.Joints()
                        
                        # STEP 1: Check if this configuration itself is collision-free (static check)
                        # This is the KEY - we need to check if the ROBOT AT THIS CONFIGURATION
                        # collides with the table/obstacles, not the path to it
                        self.robot.setJoints(joints)
                        static_collision_count = self.RDK.Collisions()
                        
                        # Restore robot position
                        self.robot.setJoints(current_joints)
                        
                        if static_collision_count > 0:
                            self._log_debug(f"Configuration #{config_idx + 1} ({elbow_status}) has STATIC collision (count: {static_collision_count}) - likely elbow hitting table")
                            continue  # Skip this configuration
                        
                        # STEP 2: Configuration is collision-free at target, now check path
                        # Create temporary target with this joint configuration
                        temp_target_name = f"_temp_ik_config_{config_idx}"
                        temp_target = self.RDK.AddTarget(temp_target_name, itemparent=self.robot.Parent())
                        
                        # setJoints accepts Mat objects or lists
                        # Ensure we have the right type
                        if hasattr(joints, 'list'):
                            # It's a Mat object, use it directly
                            temp_target.setJoints(joints)
                        else:
                            # It's already a list, convert to Mat if needed
                            from robodk import robomath
                            joints_mat = robomath.Mat([joints])
                            temp_target.setJoints(joints_mat.tr())
                        
                        temp_target.setPose(target_pose)
                        
                        # Check path collision to this configuration
                        has_collision, collision_info = self.check_collision(temp_target_name, move_type)
                        
                        if not has_collision:
                            self._log_info(f"[OK] Found collision-free configuration #{config_idx + 1} ({elbow_status}) - both static and path are clear")
                            
                            # Actually move to this configuration
                            result = self.move_to_target(temp_target_name, move_type, confirm, highlight=False)
                            
                            # Clean up temp target
                            try:
                                temp_target.Delete()
                            except:
                                pass
                            
                            if result:
                                return True
                            else:
                                self._log_warn(f"Configuration #{config_idx + 1} failed during execution")
                        else:
                            self._log_debug(f"Configuration #{config_idx + 1} ({elbow_status}) has PATH collision: {collision_info}")
                        
                        # Clean up temp target
                        try:
                            temp_target.Delete()
                        except:
                            pass
                            
                    except Exception as e:
                        self._log_debug(f"Configuration #{config_idx + 1} failed: {e}")
                        # Clean up on error
                        try:
                            temp = self.RDK.Item(temp_target_name)
                            if temp.Valid():
                                temp.Delete()
                        except:
                            pass
                
                self._log_warn(f"Tested {min(len(sorted_solutions), max_attempts)} configurations, all had collisions or failed")
                return False
                
            except AttributeError:
                # SolveIK_All not available, fall back to manual perturbation
                self._log_warn("SolveIK_All() not available, using fallback method...")
                return self._try_alternative_ik_fallback(target_name, move_type, confirm, max_attempts, is_foto_target)
        
        except Exception as e:
            self._log_error(f"Alternative IK failed: {e}")
            import traceback
            self._log_debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _try_alternative_ik_fallback(
        self,
        target_name: str,
        move_type: str,
        confirm: bool,
        max_attempts: int = 16,
        is_foto_target: bool = False
    ) -> bool:
        """
        Fallback method for alternative IK when SolveIK_All is not available.
        Tries manual perturbations of joint angles, prioritizing elbow-up.
        
        Args:
            target_name: Target name
            move_type: Movement type
            confirm: User confirmation
            max_attempts: Maximum number of attempts
            is_foto_target: Whether this is the Foto target (allows elbow-down)
        
        Returns:
            True if successful
        """
        try:
            target = self.get_target(target_name)
            if target is None:
                return False
            
            target_pose = target.Pose()
            current_joints = self.robot.Joints().list()
            
            self._log_info(f"Trying {max_attempts} joint perturbations...")
            if not is_foto_target:
                self._log_info("Prioritizing elbow-up configurations")
            
            # Create list of seed configurations to try
            # Order: elbow-up first, then others, then elbow-down last
            seed_configs = []
            
            # 1. Current configuration
            seed_configs.append(("current", current_joints.copy(), self._is_elbow_down(current_joints)))
            
            # 2. Elbow-up configuration (HIGH PRIORITY)
            elbow_up = current_joints.copy()
            if elbow_up[2] < 0:  # If currently elbow-down
                elbow_up[2] = abs(elbow_up[2])  # Make positive (elbow-up)
            seed_configs.append(("elbow-up", elbow_up, False))
            
            # 3. Flip shoulder (maintain elbow-up if possible)
            shoulder_flip = current_joints.copy()
            shoulder_flip[0] = current_joints[0] + 3.14159
            if shoulder_flip[2] < 0:
                shoulder_flip[2] = abs(shoulder_flip[2])
            seed_configs.append(("shoulder-flip+elbow-up", shoulder_flip, False))
            
            # 4-6. Wrist flips (maintain elbow-up)
            for i, joint_idx in enumerate([3, 4, 5]):
                wrist_flip = current_joints.copy()
                wrist_flip[joint_idx] = current_joints[joint_idx] + 3.14159
                if wrist_flip[2] < 0:
                    wrist_flip[2] = abs(wrist_flip[2])
                is_elbow_down = self._is_elbow_down(wrist_flip)
                seed_configs.append((f"wrist-flip-{i+1}+elbow-up", wrist_flip, is_elbow_down))
            
            # 7. Elbow-down configuration (LOW PRIORITY - only for Foto)
            elbow_down = current_joints.copy()
            if elbow_down[2] > 0:  # If currently elbow-up
                elbow_down[2] = -abs(elbow_down[2])  # Make negative (elbow-down)
            seed_configs.append(("elbow-down", elbow_down, True))
            
            # Sort configurations: elbow-up first, elbow-down last (unless Foto target)
            if is_foto_target:
                # For Foto, elbow-down has priority
                seed_configs.sort(key=lambda x: (0 if x[2] else 1))
            else:
                # For others, elbow-up has priority
                seed_configs.sort(key=lambda x: (1 if x[2] else 0))
            
            # Try each specific configuration
            for config_name, seed_joints, is_elbow_down_config in seed_configs:
                elbow_status = "elbow-down" if is_elbow_down_config else "elbow-up"
                self._log_debug(f"Trying {config_name} configuration ({elbow_status})...")
                
                try:
                    joints = self.robot.SolveIK(target_pose, joints_approx=seed_joints)
                    
                    if joints is not None and len(joints.list()) > 0:
                        temp_target_name = f"_temp_ik_{config_name}"
                        temp_target = self.RDK.AddTarget(temp_target_name, itemparent=self.robot.Parent())
                        temp_target.setJoints(joints)
                        temp_target.setPose(target_pose)
                        
                        has_collision, collision_info = self.check_collision(temp_target_name, move_type)
                        
                        if not has_collision:
                            self._log_info(f"[OK] Found collision-free IK solution: {config_name} ({elbow_status})")
                            
                            result = self.move_to_target(temp_target_name, move_type, confirm, highlight=False)
                            try:
                                temp_target.Delete()
                            except:
                                pass
                            if result:
                                return True
                        else:
                            self._log_debug(f"{config_name} ({elbow_status}) has collision: {collision_info}")
                        
                        try:
                            temp_target.Delete()
                        except:
                            pass
                except Exception as e:
                    self._log_debug(f"{config_name} IK solve failed: {e}")
            
            # Try random perturbations
            remaining = max_attempts - len(seed_configs)
            if remaining > 0:
                self._log_info(f"Trying {remaining} random perturbations...")
                
                for config_id in range(remaining):
                    try:
                        import random
                        perturbed_joints = [j + random.uniform(-0.5, 0.5) for j in current_joints]
                        joints = self.robot.SolveIK(target_pose, joints_approx=perturbed_joints)
                        
                        if joints is not None and len(joints.list()) > 0:
                            temp_target_name = f"_temp_ik_rand_{config_id}"
                            temp_target = self.RDK.AddTarget(temp_target_name, itemparent=self.robot.Parent())
                            temp_target.setJoints(joints)
                            temp_target.setPose(target_pose)
                            
                            has_collision, _ = self.check_collision(temp_target_name, move_type)
                            
                            if not has_collision:
                                self._log_info(f"[OK] Found collision-free IK solution (random #{config_id+1})")
                                
                                if self._verify_movement_in_simulation(temp_target_name, move_type):
                                    result = self.move_to_target(temp_target_name, move_type, confirm, highlight=False)
                                    try:
                                        temp_target.Delete()
                                    except:
                                        pass
                                    if result:
                                        return True
                            
                            try:
                                temp_target.Delete()
                            except:
                                pass
                    except Exception as e:
                        self._log_debug(f"Random perturbation {config_id} failed: {e}")
            
            return False
        
        except Exception as e:
            self._log_error(f"Fallback IK method failed: {e}")
            return False
    
    def _try_motion_planner(
        self,
        target_name: str,
        move_type: str,
        confirm: bool,
        max_time_ms: int = 5000,
        max_iterations: int = 1000
    ) -> bool:
        """
        Try using intelligent path sampling to find collision-free path.
        
        Note: RoboDK's Python API doesn't expose direct PRM planner control.
        Instead, this method implements a sampling-based approach:
        1. Sample intermediate configurations between current and target
        2. Test each for collisions
        3. Build path through collision-free samples
        4. Execute path if found
        
        Args:
            target_name: Target name
            move_type: Movement type ("linear" or "joint")
            confirm: User confirmation
            max_time_ms: Maximum planning time (number of samples derived from this)
            max_iterations: Maximum sampling iterations
        
        Returns:
            True if successful
        """
        try:
            target = self.get_target(target_name)
            if target is None:
                return False
            
            self._log_info(f"Sampling collision-free configurations...")
            
            # Get current and target joint configurations
            current_joints = self.robot.Joints()
            target_pose = target.Pose()
            target_joints = self.robot.SolveIK(target_pose)
            
            if target_joints is None:
                self._log_warn("Cannot solve IK for target")
                return False
            
            # Convert to lists for easier manipulation
            current_j = current_joints.list() if hasattr(current_joints, 'list') else list(current_joints)
            target_j = target_joints.list() if hasattr(target_joints, 'list') else list(target_joints)
            
            # Calculate number of samples based on max_time (rough heuristic)
            # Assume ~10ms per collision check, so samples = max_time_ms / 10
            num_samples = min(max(3, max_time_ms // 10), 50)  # Between 3 and 50 samples
            
            self._log_debug(f"Testing {num_samples} intermediate configurations...")
            
            # Sample intermediate configurations along joint-space path
            collision_free_path = []
            has_collision_on_path = False
            
            for i in range(num_samples + 1):
                fraction = i / num_samples
                
                # Linear interpolation in joint space
                interp_joints = [
                    current_j[j] + (target_j[j] - current_j[j]) * fraction
                    for j in range(len(current_j))
                ]
                
                # Test this configuration for collision
                self.robot.setJoints(interp_joints)
                collision_count = self.RDK.Collisions()
                
                if collision_count > 0:
                    self._log_debug(f"Sample {i}/{num_samples} has collision (count: {collision_count})")
                    has_collision_on_path = True
                    break
                else:
                    collision_free_path.append(interp_joints)
            
            # Restore original position
            self.robot.setJoints(current_joints)
            
            if has_collision_on_path:
                self._log_warn(f"Path sampling found collisions on direct path")
                # Could try alternative sampling strategies here (future enhancement)
                return False
            
            # Path is collision-free, execute it
            self._log_info(f"[OK] Found collision-free path with {len(collision_free_path)} waypoints")
            
            # Execute movement to target
            # Since we verified the path is clear, we can move directly
            result = self.move_to_target(target_name, move_type, confirm, highlight=False)
            
            if result:
                self._log_info("[OK] Path planning successful - movement completed!")
                return True
            else:
                self._log_warn("Path was collision-free but movement failed")
                return False
                
        except Exception as e:
            self._log_error(f"Path sampling strategy failed: {e}")
            import traceback
            self._log_debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _verify_movement_in_simulation(
        self,
        target_name: str,
        move_type: str
    ) -> bool:
        """
        Verify movement is safe in simulation before executing on real robot.
        Always runs in simulation mode first, even if real_robot mode is enabled.
        
        Args:
            target_name: Target name
            move_type: Movement type
        
        Returns:
            True if movement successful in simulation
        """
        if self.robot is None or self.RDK is None:
            return False
        
        try:
            # Save current run mode
            current_mode = self.RDK.RunMode()
            
            # Force simulation mode for verification
            self.RDK.setRunMode(RUNMODE_SIMULATE)
            
            # Save current robot position
            current_joints = self.robot.Joints()
            
            # Get target
            target = self.get_target(target_name)
            if target is None:
                self.RDK.setRunMode(current_mode)
                return False
            
            # Try the movement in simulation
            self._log_debug(f"Verifying movement to '{target_name}' in simulation...")
            
            try:
                if move_type.lower() == "linear":
                    move_result = self.robot.MoveL(target)
                else:
                    move_result = self.robot.MoveJ(target)
                
                # Restore position
                self.robot.setJoints(current_joints)
                
                # Restore run mode
                self.RDK.setRunMode(current_mode)
                
                # Check result
                if move_result is None or move_result == 0:
                    self._log_debug("Simulation verification passed")
                    return True
                else:
                    self._log_warn(f"Simulation verification failed with code: {move_result}")
                    return False
                    
            except Exception as e:
                # Restore position and mode on error
                try:
                    self.robot.setJoints(current_joints)
                    self.RDK.setRunMode(current_mode)
                except:
                    pass
                self._log_warn(f"Simulation verification failed: {e}")
                return False
                
        except Exception as e:
            self._log_error(f"Simulation verification error: {e}")
            return False
    
    def _try_with_waypoints(
        self,
        target_name: str,
        move_type: str,
        confirm: bool
    ) -> bool:
        """
        Try reaching target with intermediate waypoints.
        Attempts multiple waypoint strategies to find collision-free path.
        
        Args:
            target_name: Target name
            move_type: Movement type
            confirm: User confirmation
        
        Returns:
            True if successful
        """
        try:
            target = self.get_target(target_name)
            if target is None:
                return False
            
            # Get current and target poses - with error recovery
            try:
                current_pose = self.robot.Pose()
                target_pose = target.Pose()
            except (UnicodeDecodeError, struct.error) as e:
                self._log_warn(f"API error getting poses: {e}, attempting recovery...")
                if self._recover_from_api_error():
                    current_pose = self.robot.Pose()
                    target_pose = target.Pose()
                else:
                    raise
            
            # Get positions
            target_position = target_pose.Pos()
            current_position = current_pose.Pos()
            
            self._log_info(f"Trying waypoint strategies...")
            self._log_debug(f"Current: [{current_position[0]:.1f}, {current_position[1]:.1f}, {current_position[2]:.1f}]")
            self._log_debug(f"Target: [{target_position[0]:.1f}, {target_position[1]:.1f}, {target_position[2]:.1f}]")
            
            # Try multiple waypoint strategies
            waypoint_strategies = [
                ("above", [0, 0, 150]),      # 150mm above midpoint
                ("high-above", [0, 0, 250]),  # 250mm above midpoint
                ("retract-first", None),      # Move up first, then approach
            ]
            
            for strategy_name, offset in waypoint_strategies:
                self._log_debug(f"Trying waypoint strategy: {strategy_name}")
                
                try:
                    if strategy_name == "retract-first":
                        # Strategy: Move up from current position first, then to target
                        retract_pose = current_pose.copy()
                        retract_pos = current_position.copy()
                        retract_pos[2] += 150  # Move 150mm up
                        retract_pose.setPos(retract_pos)
                        
                        # Create retract waypoint
                        waypoint1 = self.RDK.AddTarget("_temp_retract", itemparent=self.robot.Parent())
                        waypoint1.setPose(retract_pose)
                        
                        # Create approach waypoint (above target)
                        approach_pose = target_pose.copy()
                        approach_pos = target_position.copy()
                        approach_pos[2] += 100  # 100mm above target
                        approach_pose.setPos(approach_pos)
                        
                        waypoint2 = self.RDK.AddTarget("_temp_approach", itemparent=self.robot.Parent())
                        waypoint2.setPose(approach_pose)
                        
                        # Check collisions for this path
                        col1, _ = self.check_collision("_temp_retract", "joint")
                        col2, _ = self.check_collision("_temp_approach", "joint") if not col1 else (True, None)
                        col3, _ = self.check_collision(target_name, move_type) if not col2 else (True, None)
                        
                        if not (col1 or col2 or col3):
                            self._log_info(f"[OK] Found collision-free path: {strategy_name}")
                            
                            # Execute waypoint sequence
                            if (self.move_to_target("_temp_retract", "joint", confirm=False, highlight=False) and
                                self.move_to_target("_temp_approach", "joint", confirm=False, highlight=False) and
                                self.move_to_target(target_name, move_type, confirm=False, highlight=False)):
                                
                                waypoint1.Delete()
                                waypoint2.Delete()
                                return True
                        
                        waypoint1.Delete()
                        waypoint2.Delete()
                        
                    else:
                        # Calculate midpoint with offset
                        mid_position = [
                            (current_position[0] + target_position[0]) / 2 + offset[0],
                            (current_position[1] + target_position[1]) / 2 + offset[1],
                            (current_position[2] + target_position[2]) / 2 + offset[2]
                        ]
                        
                        mid_pose = current_pose.copy()
                        mid_pose.setPos(mid_position)
                        
                        # Create temporary waypoint target
                        waypoint_target = self.RDK.AddTarget(f"_temp_waypoint_{strategy_name}", 
                                                            itemparent=self.robot.Parent())
                        waypoint_target.setPose(mid_pose)
                        
                        # Check collision for path to waypoint and then to target
                        has_collision_1, info1 = self.check_collision(f"_temp_waypoint_{strategy_name}", "joint")
                        has_collision_2, info2 = self.check_collision(target_name, move_type) if not has_collision_1 else (True, None)
                        
                        if not has_collision_1 and not has_collision_2:
                            self._log_info(f"[OK] Found collision-free path: {strategy_name}")
                            
                            # Move to waypoint first
                            success_1 = self.move_to_target(f"_temp_waypoint_{strategy_name}", "joint", 
                                                          confirm=False, highlight=False)
                            if success_1:
                                # Then move to final target
                                success_2 = self.move_to_target(target_name, move_type, 
                                                              confirm=False, highlight=False)
                                waypoint_target.Delete()
                                if success_2:
                                    return True
                        else:
                            self._log_debug(f"{strategy_name} has collision - wp: {has_collision_1}, target: {has_collision_2}")
                        
                        waypoint_target.Delete()
                        
                except Exception as e:
                    self._log_debug(f"Waypoint strategy '{strategy_name}' failed: {e}")
                    # Clean up any temp targets
                    try:
                        for temp_name in [f"_temp_waypoint_{strategy_name}", "_temp_retract", "_temp_approach"]:
                            temp = self.RDK.Item(temp_name)
                            if temp.Valid():
                                temp.Delete()
                    except:
                        pass
            
            self._log_warn("All waypoint strategies failed")
            return False
            
        except Exception as e:
            self._log_error(f"Waypoint strategy failed: {e}")
            import traceback
            self._log_debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def move_to_target(
        self,
        target_name: str,
        move_type: str = "joint",
        confirm: bool = True,
        highlight: bool = True
    ) -> bool:
        """
        Move robot to target.
        
        Args:
            target_name: Name of target to move to
            move_type: "joint" (MoveJ) or "linear" (MoveL)
            confirm: Ask user confirmation before moving
            highlight: Highlight target before moving
        
        Returns:
            True if movement successful, False otherwise
        """
        if self.robot is None:
            self._log_error("No robot selected")
            return False
        
        # Get target
        target = self.get_target(target_name)
        if target is None:
            self._log_error(f"Target '{target_name}' not found")
            return False
        
        # Highlight target
        if highlight:
            try:
                self.RDK.ShowTarget(target)
                time.sleep(0.5)
            except:
                pass
        
        # User confirmation
        if confirm:
            print(f"\n{'='*60}")
            print(f"Ready to move to target: {target_name}")
            print(f"Movement type: {move_type.upper()}")
            print(f"{'='*60}")
            
            response = input("Continue? [Y/n/q]: ").strip().lower()
            
            if response == 'q':
                self._log_warn("Movement cancelled by user (quit)")
                return False
            elif response == 'n':
                self._log_warn("Movement skipped by user")
                return False
        
        # If in real robot mode, verify in simulation first
        if self.run_mode.lower() == "real_robot":
            self._log_info("Real robot mode: Verifying movement in simulation first...")
            if not self._verify_movement_in_simulation(target_name, move_type):
                self._log_error("Simulation verification failed! Aborting real robot movement for safety.")
                return False
            self._log_info("[OK] Simulation verification passed, proceeding with real robot...")
        
        # Execute movement
        try:
            self._log_info(f"Moving to '{target_name}' ({move_type})...")
            
            # Check safety status before movement
            is_safe, safety_error = self.check_robot_safety_status()
            if not is_safe:
                self._log_error(f"Safety check failed before movement: {safety_error}")
                if self.emergency_stop_active:
                    raise RobotEmergencyStopError(f"E-stop active: {safety_error}")
                elif self.collision_halt_active:
                    raise RobotCollisionError(f"Collision halt active: {safety_error}")
                else:
                    raise RobotSafetyError(f"Safety error: {safety_error}")
            
            # Get current position for error reporting - with error recovery
            try:
                current_joints = self.robot.Joints().list()
                target_pose = target.Pose()
                target_pos = target_pose.Pos()
            except (UnicodeDecodeError, struct.error) as e:
                self._log_warn(f"API error getting position data: {e}, attempting recovery...")
                if self._recover_from_api_error():
                    current_joints = self.robot.Joints().list()
                    target_pose = target.Pose()
                    target_pos = target_pose.Pos()
                else:
                    raise
            
            self._log_debug(f"Current joints: {[f'{j:.2f}' for j in current_joints]}")
            self._log_debug(f"Target position: [{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}] mm")
            
            # Execute movement with proper error handling
            move_result = None
            try:
                if move_type.lower() == "linear":
                    move_result = self.robot.MoveL(target)
                else:  # joint
                    move_result = self.robot.MoveJ(target)
            except Exception as move_error:
                # Check if movement was interrupted by safety issue
                is_safe, safety_error = self.check_robot_safety_status()
                if not is_safe:
                    self._log_error(f"Movement interrupted by safety issue: {safety_error}")
                    if self.emergency_stop_active:
                        raise RobotEmergencyStopError(f"E-stop during movement: {safety_error}")
                    elif self.collision_halt_active:
                        raise RobotCollisionError(f"Collision during movement: {safety_error}")
                    else:
                        raise RobotSafetyError(f"Safety error during movement: {safety_error}")
                # Re-raise original error if not a safety issue
                raise
            
            # Post-movement safety check
            is_safe, safety_error = self.check_robot_safety_status()
            if not is_safe:
                self._log_error(f"Safety check failed after movement: {safety_error}")
                if self.emergency_stop_active:
                    raise RobotEmergencyStopError(f"E-stop after movement: {safety_error}")
                elif self.collision_halt_active:
                    raise RobotCollisionError(f"Collision after movement: {safety_error}")
                else:
                    raise RobotSafetyError(f"Safety error after movement: {safety_error}")
            
            # Check if movement completed successfully
            # RoboDK returns: None or 0 = success, positive integer = error
            if move_result is not None and move_result != 0:
                self._log_error(f"Movement returned error code: {move_result}")
                
                # Try to get more details about the error
                error_msg = "Unknown movement error"
                if move_result == 1:
                    error_msg = "Target not reachable (IK failed or out of workspace)"
                elif move_result == 2:
                    error_msg = "Collision detected during movement"
                    self.collision_halt_active = True
                    raise RobotCollisionError(f"RoboDK detected collision: {error_msg}")
                elif move_result == 3:
                    error_msg = "Joint limits exceeded"
                elif move_result == 4:
                    error_msg = "Singularity detected"
                
                self._log_error(f"Error details: {error_msg}")
                self._log_error(f"Target: {target_name}, Move type: {move_type}")
                
                # Check current collision status
                collision_count = self.RDK.Collisions()
                if collision_count > 0:
                    self._log_error(f"Collision count: {collision_count}")
                    try:
                        collision_items = self.RDK.getCollisionItems()
                        if collision_items:
                            items_str = ", ".join([item.Name() for item in collision_items if item.Valid()])
                            self._log_error(f"Colliding items: {items_str}")
                    except:
                        pass
                    self.collision_halt_active = True
                    raise RobotCollisionError(f"Collision detected (count: {collision_count}): {error_msg}")
                
                return False
            
            self._log_info(f"[OK] Movement complete: {target_name}")
            return True
        
        except RobotSafetyError as safety_err:
            # Safety errors should halt immediately and propagate up
            self._log_error(f"SAFETY ERROR - HALTING OPERATIONS: {safety_err}")
            self._log_error(f"Target: {target_name}, Move type: {move_type}")
            # Re-raise to ensure calling code knows this is a safety issue
            raise
        
        except Exception as e:
            self._log_error(f"Movement exception: {type(e).__name__}: {e}")
            self._log_error(f"Target: {target_name}, Move type: {move_type}")
            
            # Check if movement failure was due to safety issue
            is_safe, safety_error = self.check_robot_safety_status()
            if not is_safe:
                self._log_error(f"Post-error safety check failed: {safety_error}")
                if self.emergency_stop_active:
                    raise RobotEmergencyStopError(f"E-stop detected after error: {safety_error}")
                elif self.collision_halt_active:
                    raise RobotCollisionError(f"Collision detected after error: {safety_error}")
            
            # Check if this is an API communication error and attempt recovery
            if isinstance(e, (UnicodeDecodeError, struct.error)) or "unpack" in str(e) or "decode" in str(e):
                self._log_warn("Detected API communication error during movement")
                if self._recover_from_api_error():
                    self._log_info("API recovered, but movement did not complete")
                else:
                    self._log_error("API recovery failed")
            
            # Additional error context
            try:
                import traceback
                self._log_debug(f"Traceback: {traceback.format_exc()}")
            except:
                pass
            
            return False
    
    def check_emergency_stop(self) -> bool:
        """
        Check if emergency stop key is pressed.
        
        Returns:
            True if ESC pressed, False otherwise
        """
        if HAVE_KEYBOARD:
            if keyboard.is_pressed("esc"):
                self._log_error("EMERGENCY STOP: ESC key pressed!")
                return True
        return False
    
    def cleanup(self) -> None:
        """Clean up and close RoboDK connection."""
        self._log_info("Cleaning up RoboDK connection...")
        
        # Delete dynamic targets (optional, comment out to keep them)
        # for name, target in self.dynamic_targets.items():
        #     try:
        #         target.Delete()
        #     except:
        #         pass
        
        self._log_info("RoboDK cleanup complete")


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    
    # Initialize manager
    station_path = Path("pickafresa_robot/rdk/SETUP Fresas.rdk")
    
    manager = RoboDKManager(
        station_file=station_path,
        robot_model="UR3e",
        run_mode="simulate"
    )
    
    # Connect and setup
    if manager.connect():
        if manager.select_robot():
            manager.set_speed(linear_speed=50, joint_speed=30)
            
            # Discover targets
            targets = manager.discover_targets()
            
            # Move to home
            if "Home" in targets:
                manager.move_to_target("Home", move_type="joint", confirm=True)
            
            # Create a test target
            T_test = np.eye(4)
            T_test[:3, 3] = [0.3, 0.2, 0.4]  # Position in meters
            
            manager.create_target_from_pose(
                name="test_fruit",
                T_base_target=T_test,
                create_frame=True,
                color=[255, 0, 0]
            )
            
            manager.cleanup()
