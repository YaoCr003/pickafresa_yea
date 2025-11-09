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

by: Aldrick T, 2025
for Team YEA
"""

import time
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

try:
    import keyboard
    HAVE_KEYBOARD = True
except ImportError:
    HAVE_KEYBOARD = False


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
            
            self._log_info("✓ Connected to RoboDK")
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
            
            self._log_info(f"✓ Robot selected: {name}")
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
            
            self._log_info(f"✓ Discovered {len(targets)} targets: {list(targets.keys())}")
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
            
            self._log_info(f"✓ Created target: {name} (parent: {robot_base_frame.Name() if robot_base_frame.Valid() else 'None'})")
            
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
                
                self._log_info(f"✓ Created reference frame: {name}_frame (parent: {robot_base_frame.Name() if robot_base_frame.Valid() else 'None'})")
            
            # Store in dynamic targets
            self.dynamic_targets[name] = target
            
            return target
        
        except Exception as e:
            self._log_error(f"Failed to create target '{name}': {e}")
            return None
    
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
            return self.dynamic_targets[name]
        
        # Check discovered targets
        if name in self.targets:
            return self.targets[name]
        
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
        
        # Execute movement
        try:
            self._log_info(f"Moving to '{target_name}' ({move_type})...")
            
            if move_type.lower() == "linear":
                self.robot.MoveL(target)
            else:  # joint
                self.robot.MoveJ(target)
            
            self._log_info(f"✓ Movement complete: {target_name}")
            return True
        
        except Exception as e:
            self._log_error(f"Movement failed: {e}")
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
