"""
Robot PnP Controller - Core Pick-and-Place Logic

This module contains the core pick-and-place sequence logic extracted from robot_pnp_cli.
It manages the robot operations, state transitions, and error handling.

Used by:
- robot_pnp_service (always-on service)
- robot_pnp_cli (testing tool)
- robot_pnp_manager (via service)
- robot_pnp_remote (via service)

by: Aldrick T, 2025
for Team YEA
"""

import sys
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import cv2  # For Rodrigues rotation in post-pick target creation

# Repository root setup
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pickafresa_robot.robot_system.state_machine import RobotStateMachine, RobotState, StateTransitionError
from pickafresa_robot.robot_system.config_manager import ConfigManager
from pickafresa_robot.robot_system.vision_client import VisionServiceClient, VisionServiceError, FruitDetection
from pickafresa_robot.robot_system.transform_utils import TransformUtils
from pickafresa_robot.robot_system.robodk_manager import RoboDKManager
from pickafresa_robot.robot_system.mqtt_gripper import MQTTGripperController
from pickafresa_robot.robot_system.pnp_handler import PnPDataHandler


class ControllerError(Exception):
    """Raised when controller encounters an error."""
    pass


class RobotPnPController:
    """
    Core pick-and-place controller.
    
    Manages robot operations, state transitions, and error handling.
    Can be used standalone or embedded in a service.
    """
    
    def __init__(self, config: ConfigManager, logger=None):
        """
        Initialize controller.
        
        Args:
            config: ConfigManager instance
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger
        self.state_machine = RobotStateMachine(initial_state=RobotState.INITIALIZING, logger=logger)
        
        # Components (initialized lazily)
        self.robodk_manager: Optional[RoboDKManager] = None
        self.mqtt_gripper: Optional[MQTTGripperController] = None
        self.pnp_handler: Optional[PnPDataHandler] = None
        self.vision_client: Optional[VisionServiceClient] = None
        
        # State
        self.is_initialized = False
        self.current_berry_index = 0
        
        # Offline/debug mode
        self.offline_mode = False
        self.offline_json_data: Optional[List[Dict[str, Any]]] = None
        
        self._log("info", "Robot PnP Controller created")
    
    def _log(self, level: str, message: str):
        """Internal logging helper."""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def initialize(self) -> bool:
        """
        Initialize all subsystems.
        
        Returns:
            True if initialization succeeded
        """
        if self.is_initialized:
            self._log("warning", "Controller already initialized")
            return True
        
        self._log("info", "Initializing controller subsystems...")
        
        try:
            # Initialize RoboDK
            if not self._init_robodk():
                raise ControllerError("Failed to initialize RoboDK")
            
            # Initialize MQTT gripper (optional)
            self._init_mqtt_gripper()
            
            # Initialize PnP handler
            if not self._init_pnp_handler():
                raise ControllerError("Failed to initialize PnP handler")
            
            # Initialize vision client
            if not self._init_vision_client():
                raise ControllerError("Failed to initialize vision client")
            
            self.is_initialized = True
            
            # Transition to IDLE state
            self._log("info", "Transitioning to IDLE state...")
            try:
                self.state_machine.transition_to(RobotState.IDLE, "Initialization complete")
            except Exception as sm_error:
                self._log("warning", f"State machine transition failed: {sm_error}")
                # Don't fail initialization just because state transition failed
            
            self._log("info", "✓ Controller initialization complete")
            return True
        
        except Exception as e:
            self._log("error", f"Controller initialization failed: {e}")
            try:
                self.state_machine.to_error(f"Initialization failed: {e}")
            except:
                pass  # State machine might not be working
            return False
    
    def _init_robodk(self) -> bool:
        """Initialize RoboDK manager with full configuration."""
        try:
            self._log("info", "Initializing RoboDK...")
            
            robodk_config = self.config.get('robodk', {})
            station_file = self.config.resolve_path(robodk_config.get('station_file', ''))
            
            if not station_file.exists():
                self._log("error", f"RoboDK station file not found: {station_file}")
                return False
            
            # Get robot model and simulation mode
            robot_model = robodk_config.get('robot_model', 'UR3e')
            simulation_mode = robodk_config.get('simulation_mode', robodk_config.get('run_mode', 'simulate'))
            
            self._log("info", f"Station: {station_file.name}")
            self._log("info", f"Robot: {robot_model}")
            self._log("info", f"Mode: {simulation_mode}")
            
            self.robodk_manager = RoboDKManager(
                station_file=str(station_file),
                robot_model=robot_model,
                run_mode=simulation_mode,  # RoboDKManager expects 'run_mode' parameter
                logger=self.logger
            )
            
            # Connect to RoboDK
            if not self.robodk_manager.connect():
                self._log("error", "Failed to connect to RoboDK")
                return False
            
            # Select robot
            if not self.robodk_manager.select_robot():
                self._log("error", "Failed to select robot")
                return False
            
            # Set speed profile
            speed_config = self.config.get('movement', {})
            profile = speed_config.get('default_profile', 'slow')
            
            profiles = speed_config.get('speed_profiles', {
                'turtle': {'linear_speed': 20, 'joint_speed': 10},
                'slow': {'linear_speed': 50, 'joint_speed': 30},
                'normal': {'linear_speed': 100, 'joint_speed': 60}
            })
            
            speed = profiles.get(profile, profiles['slow'])
            self.robodk_manager.set_speed(
                linear_speed=speed.get('linear_speed', 50),
                joint_speed=speed.get('joint_speed', 30)
            )
            self._log("info", f"Speed profile: {profile} (linear={speed.get('linear_speed')} mm/s, joint={speed.get('joint_speed')} deg/s)")
            
            # Discover targets
            self._log("info", "Discovering targets in RoboDK station...")
            self.robodk_manager.discover_targets()
            
            # Move to Foto position (idle detection position)
            self._log("info", "Moving to Foto position for idle detection mode...")
            if not self.robodk_manager.move_to_target("Foto", "joint", confirm=False):
                self._log("warning", "Failed to move to Foto position - continuing anyway")
            
            self._log("info", "✓ RoboDK initialized successfully")
            return True
        
        except Exception as e:
            self._log("error", f"RoboDK initialization failed: {e}")
            import traceback
            self._log("debug", traceback.format_exc())
            return False
    
    def _init_mqtt_gripper(self) -> bool:
        """Initialize MQTT gripper (optional)."""
        mqtt_config = self.config.get('mqtt', {})
        
        if not mqtt_config.get('enabled', False):
            self._log("info", "MQTT gripper disabled in config")
            return True
        
        try:
            self._log("info", "Initializing MQTT gripper...")
            
            self.mqtt_gripper = MQTTGripperController(
                broker_address=mqtt_config.get('broker_address', '192.168.1.100'),
                broker_port=mqtt_config.get('broker_port', 1883),
                logger=self.logger
            )
            
            if self.mqtt_gripper.connect():
                self._log("info", "✓ MQTT gripper initialized")
                return True
            else:
                self._log("warning", "MQTT gripper connection failed (optional)")
                self.mqtt_gripper = None
                return True  # Non-fatal
        
        except Exception as e:
            self._log("warning", f"MQTT gripper initialization failed: {e} (optional)")
            self.mqtt_gripper = None
            return True  # Non-fatal
    
    def _init_pnp_handler(self) -> bool:
        """Initialize PnP data handler."""
        try:
            self._log("info", "Initializing PnP handler...")
            
            T_flange_cameraTCP = self.config.get_transform_matrix('camera_tcp')
            T_flange_gripperTCP = self.config.get_transform_matrix('gripper_tcp')
            
            self.pnp_handler = PnPDataHandler(
                T_flange_cameraTCP=T_flange_cameraTCP,
                T_flange_gripperTCP=T_flange_gripperTCP,
                logger=self.logger
            )
            
            self._log("info", "✓ PnP handler initialized")
            return True
        
        except Exception as e:
            self._log("error", f"PnP handler initialization failed: {e}")
            return False
    
    def _init_vision_client(self) -> bool:
        """Initialize vision service client."""
        try:
            self._log("info", "Initializing vision client...")
            
            vision_config = self.config.get('vision_service', {})
            host = vision_config.get('host', '127.0.0.1')
            port = vision_config.get('port', 5555)
            timeout = vision_config.get('timeout', 30.0)
            
            self.vision_client = VisionServiceClient(host=host, port=port, timeout=timeout, logger=self.logger)
            self.vision_client.connect()
            
            # Check status
            status = self.vision_client.check_status()
            if not status.get('alive', False):
                raise VisionServiceError("Service not alive")
            
            self._log("info", "✓ Vision client initialized")
            self.offline_mode = False
            return True
        
        except VisionServiceError as e:
            self._log("warning", f"Vision client initialization failed: {e}")
            self._log("warning", "Vision service unavailable - controller can run in offline/debug mode")
            self._log("warning", "  In offline mode, you will be prompted to provide JSON data files for detections")
            
            # Mark as offline mode, but don't fail initialization
            self.offline_mode = True
            self.vision_client = None
            
            self._log("info", "✓ Controller initialized in OFFLINE MODE (vision service unavailable)")
            return True
    
    def shutdown(self):
        """Graceful shutdown of all subsystems."""
        self._log("info", "Shutting down controller...")
        
        try:
            self.state_machine.transition_to(RobotState.SHUTDOWN, "Shutdown requested")
        except:
            pass
        
        # Return home and release gripper
        if self.robodk_manager and self.robodk_manager.robot:
            try:
                self._log("info", "Returning to home position...")
                self.robodk_manager.move_to_target("Home", "joint", confirm=False)
            except:
                pass
        
        if self.mqtt_gripper:
            try:
                self._log("info", "Releasing gripper...")
                self.mqtt_gripper.open_gripper()
            except:
                pass
        
        # Disconnect components
        if self.vision_client:
            try:
                self.vision_client.disconnect()
            except:
                pass
        
        if self.mqtt_gripper:
            try:
                self.mqtt_gripper.disconnect()
            except:
                pass
        
        if self.robodk_manager:
            try:
                self.robodk_manager.disconnect()
            except:
                pass
        
        self.is_initialized = False
        self._log("info", "✓ Controller shutdown complete")
    
    def execute_pick_sequence(self, berry_index: int = 0, json_path: Optional[str] = None) -> bool:
        """
        Execute full pick-and-place sequence for one berry.
        
        Args:
            berry_index: Index of berry to pick (for multi-berry mode)
            json_path: Optional path to JSON data file (for offline mode)
        
        Returns:
            True if sequence completed successfully
        """
        if not self.is_initialized:
            self._log("error", "Controller not initialized")
            return False
        
        if not self.state_machine.is_operational():
            self._log("error", f"Controller not operational (state: {self.state_machine.state_name})")
            return False
        
        self.current_berry_index = berry_index
        
        try:
            # === CLEANUP DYNAMIC TARGETS (following CLI pattern) ===
            # Clean up targets from previous berry cycles to prevent accumulation
            self._log("info", "Cleaning up dynamic targets from previous cycles...")
            fixed_targets = self.config.get('robodk', {}).get('fixed_targets', ['Home', 'Foto', 'Prepick_plane', 'place_final'])
            cleaned_count = self.robodk_manager.cleanup_dynamic_targets(fixed_targets)
            if cleaned_count > 0:
                self._log("info", f"✓ Cleaned up {cleaned_count} targets/frames from previous berry cycles")
            
            # Robot should already be at Foto position (idle detection mode)
            # Capture vision data
            self.state_machine.transition_to(RobotState.CAPTURING, "Requesting vision data")
            detections = self._capture_detections(json_path=json_path)
            
            if not detections:
                self._log("error", "No valid detections found")
                self.state_machine.to_idle("No detections")
                # Return to Foto position for idle detection
                if not self.robodk_manager.move_to_target("Foto", "joint", confirm=False):
                    self._log("warning", "Failed to return to Foto position")
                return False
            
            # Select berry
            fruit = self._select_berry(detections, berry_index)
            if fruit is None:
                self._log("error", f"Berry #{berry_index} not available")
                self.state_machine.to_idle("Berry selection failed")
                # Return to Foto position for idle detection
                if not self.robodk_manager.move_to_target("Foto", "joint", confirm=False):
                    self._log("warning", "Failed to return to Foto position")
                return False
            
            # Transform to base frame
            T_base_gripperTCP = self.robodk_manager.get_tcp_pose()
            fruit = self.pnp_handler.transform_to_base_frame(fruit, T_base_gripperTCP)
            
            # Execute pick sequence
            self.state_machine.transition_to(RobotState.PICKING, "Executing pick sequence")
            if not self._execute_pick(fruit):
                raise ControllerError("Pick sequence failed")
            
            # Return to Foto position for idle detection (NOT home)
            # Only transition if not already in MOVING state
            if self.state_machine.state != RobotState.MOVING:
                self.state_machine.transition_to(RobotState.MOVING, "Returning to Foto position")
            if not self.robodk_manager.move_to_target("Foto", "joint", confirm=False):
                self._log("warning", "Failed to return to Foto position")
            
            self.state_machine.to_idle("Sequence complete")
            self._log("info", f"✓ Pick sequence #{berry_index} completed successfully")
            return True
        
        except StateTransitionError as e:
            self._log("error", f"Invalid state transition: {e}")
            # Try to return to Foto position and recover to IDLE
            try:
                self._log("info", "Attempting recovery to Foto position...")
                if self.robodk_manager.move_to_target("Foto", "joint", confirm=False):
                    self.state_machine.to_idle("Recovery after state transition error")
                else:
                    self.state_machine.to_error("Failed to recover to Foto position")
            except Exception as recovery_error:
                self._log("error", f"Recovery failed: {recovery_error}")
                self.state_machine.to_error(f"Recovery failed: {recovery_error}")
            return False
        
        except Exception as e:
            self._log("error", f"Pick sequence failed: {e}")
            self.state_machine.to_error(str(e))
            
            # Attempt recovery to Foto position and IDLE state
            try:
                self._log("info", "Attempting recovery to Foto position...")
                if self.robodk_manager.move_to_target("Foto", "joint", confirm=False):
                    self._log("info", "✓ Returned to Foto position")
                    self.state_machine.to_idle("Recovery after pick failure")
                    self._log("info", "✓ Recovered to IDLE state")
                else:
                    self._log("warning", "Failed to return to Foto position during recovery")
            except Exception as recovery_error:
                self._log("error", f"Recovery to Foto failed: {recovery_error}")
            
            return False
    
    def execute_multi_berry_sequence(self, json_path: Optional[str] = None) -> bool:
        """
        Execute multi-berry picking sequence with advanced automation features.
        
        Features:
        - Sorting by confidence, distance, or position
        - Retry logic for failed picks
        - Safe home/foto return between picks
        - Abort on failure option
        - Success statistics tracking
        
        Args:
            json_path: Optional path to JSON data file (for offline mode)
        
        Returns:
            True if at least one berry was picked successfully
        """
        if not self.is_initialized:
            self._log("error", "Controller not initialized")
            return False
        
        if not self.state_machine.is_operational():
            self._log("error", f"Controller not operational (state: {self.state_machine.state_name})")
            return False
        
        # Get multi-berry configuration
        multi_berry_config = self.config.get('multi_berry', {})
        max_berries = multi_berry_config.get('max_berries_per_run', 10)
        sort_by = multi_berry_config.get('sort_by', 'confidence')
        retry_on_failure = multi_berry_config.get('retry_on_failure', True)
        max_retries = multi_berry_config.get('max_retries', 3)
        return_home_between = multi_berry_config.get('return_home_between_picks', False)
        abort_on_failure = multi_berry_config.get('abort_on_failure', False)
        
        self._log("info", "=" * 60)
        self._log("info", "MULTI-BERRY PICKING MODE")
        self._log("info", "=" * 60)
        self._log("info", f"Max berries per run: {max_berries}")
        self._log("info", f"Sort by: {sort_by}")
        self._log("info", f"Retry on failure: {retry_on_failure} (max {max_retries})")
        self._log("info", f"Abort on failure: {abort_on_failure}")
        self._log("info", "=" * 60)
        
        try:
            # Capture detections once at start
            self.state_machine.transition_to(RobotState.CAPTURING, "Requesting vision data")
            detections = self._capture_detections(json_path=json_path)
            
            if not detections:
                self._log("error", "No valid detections found")
                self.state_machine.to_idle("No detections")
                return False
            
            # Sort fruits based on configuration
            fruits_to_process = self._sort_fruits(detections, sort_by)
            
            # Limit to max_berries
            fruits_to_process = fruits_to_process[:max_berries]
            
            self._log("info", f"Found {len(detections)} detections, processing {len(fruits_to_process)} berries")
            
            # Track statistics
            total_attempts = 0
            successful_picks = 0
            failed_picks = 0
            
            # Process each fruit
            for i, fruit in enumerate(fruits_to_process, 1):
                berry_letter = chr(64 + i)  # 1->A, 2->B, etc.
                
                self._log("info", "=" * 60)
                self._log("info", f"BERRY {i}/{len(fruits_to_process)} (Berry {berry_letter})")
                self._log("info", f"Class: {fruit.class_name}, Confidence: {fruit.confidence:.2f}")
                self._log("info", "=" * 60)
                
                # For berry #2 onwards, ensure proper positioning
                if i > 1:
                    self._log("info", "=" * 60)
                    self._log("info", f"PRE-BERRY #{i} POSITIONING (Safety Protocol)")
                    self._log("info", "=" * 60)
                    
                    # Return to Home first
                    self._log("info", "Moving to Home before next berry...")
                    if not self._move_home():
                        self._log("error", "Failed to return home - ABORTING for safety")
                        self.state_machine.to_error("Failed to return home")
                        return False
                    
                    # Move to Foto for proper transformation
                    self._log("info", "Moving to Foto for proper berry transformation...")
                    if not self._move_foto():
                        self._log("error", "Failed to reach Foto position - ABORTING for safety")
                        self.state_machine.to_error("Failed to reach Foto")
                        return False
                    
                    self._log("info", "✓ Robot positioned at Foto for berry transformation")
                
                # Attempt to pick this berry (with retry logic)
                success = False
                retry_count = 0
                
                while not success and retry_count <= max_retries:
                    if retry_count > 0:
                        self._log("info", f"Retry attempt {retry_count}/{max_retries}")
                    
                    total_attempts += 1
                    success = self._process_single_fruit(fruit, index=i)
                    
                    if success:
                        successful_picks += 1
                        self._log("info", f"✓ Berry {berry_letter} picked successfully")
                        break
                    else:
                        retry_count += 1
                        failed_picks += 1
                        
                        if retry_on_failure and retry_count <= max_retries:
                            self._log("warning", f"Berry {berry_letter} failed, retrying...")
                            
                            # Return to home before retry
                            if not self._move_home():
                                self._log("error", "Failed to return home for retry")
                                break
                            
                            # Return to foto for retry
                            if not self._move_foto():
                                self._log("error", "Failed to return to foto for retry")
                                break
                        else:
                            self._log("error", f"Berry {berry_letter} failed after {retry_count} attempts")
                            
                            if abort_on_failure:
                                self._log("error", "Abort on failure enabled, stopping...")
                                self.state_machine.to_idle("Aborted due to failure")
                                return False
                            break
                
                # Return home between picks if configured
                if return_home_between and success and i < len(fruits_to_process):
                    self._log("info", "Returning home between picks...")
                    if not self._move_home():
                        self._log("error", "Failed to return home")
                        if abort_on_failure:
                            self.state_machine.to_error("Failed to return home")
                            return False
                    
                    if not self._move_foto():
                        self._log("error", "Failed to return to foto")
                        if abort_on_failure:
                            self.state_machine.to_error("Failed to return to foto")
                            return False
            
            # Print summary
            self._log("info", "=" * 60)
            self._log("info", "MULTI-BERRY PICKING SUMMARY")
            self._log("info", "=" * 60)
            self._log("info", f"Total attempts: {total_attempts}")
            self._log("info", f"Successful picks: {successful_picks}")
            self._log("info", f"Failed picks: {failed_picks}")
            success_rate = 100.0 * successful_picks / total_attempts if total_attempts > 0 else 0
            self._log("info", f"Success rate: {success_rate:.1f}%")
            self._log("info", "=" * 60)
            
            self.state_machine.to_idle("Multi-berry sequence complete")
            return successful_picks > 0
        
        except Exception as e:
            self._log("error", f"Multi-berry sequence failed: {e}")
            self.state_machine.to_error(f"Multi-berry failure: {e}")
            return False
    
    def _sort_fruits(self, fruits: List[FruitDetection], sort_by: str) -> List[FruitDetection]:
        """
        Sort fruits based on specified criteria.
        
        Args:
            fruits: List of fruit detections
            sort_by: Sorting criterion ('confidence', 'distance', 'position')
        
        Returns:
            Sorted list of fruits
        """
        if sort_by == 'confidence':
            # Sort by confidence (highest first)
            return sorted(fruits, key=lambda f: f.confidence, reverse=True)
        
        elif sort_by == 'distance':
            # Sort by distance from camera (closest first)
            # Use Z coordinate in camera frame
            return sorted(fruits, key=lambda f: f.position_cam[2] if f.position_cam is not None else float('inf'))
        
        elif sort_by == 'position':
            # Sort by position (left to right, then top to bottom in image)
            # Use bbox center
            return sorted(fruits, key=lambda f: (f.bbox[1] + f.bbox[3]) / 2.0 * 1000 + (f.bbox[0] + f.bbox[2]) / 2.0)
        
        else:
            self._log("warning", f"Unknown sort criterion '{sort_by}', using original order")
            return fruits
    
    def _process_single_fruit(self, fruit: FruitDetection, index: int = 1) -> bool:
        """
        Process a single fruit (full pick and place cycle).
        
        Args:
            fruit: FruitDetection object with pose information
            index: Berry index for multi-berry runs (1-based)
        
        Returns:
            True if successful, False otherwise
        """
        berry_letter = chr(64 + index)  # 1->A, 2->B, etc.
        
        self._log("info", f"Starting pick-and-place cycle for fruit #{index} (Berry {berry_letter})")
        
        # Cleanup dynamic targets from previous berry
        fixed_targets = self.config.get('robodk', {}).get('fixed_targets', ['Home', 'Foto', 'Prepick_plane', 'place_final'])
        cleaned_count = self.robodk_manager.cleanup_dynamic_targets(fixed_targets)
        if cleaned_count > 0:
            self._log("info", f"Cleaned up {cleaned_count} targets/frames from previous berry cycles")
        
        # Transform to base frame
        T_base_gripperTCP = self.robodk_manager.get_tcp_pose()
        fruit = self.pnp_handler.transform_to_base_frame(fruit, T_base_gripperTCP)
        
        # Execute pick sequence
        self.state_machine.transition_to(RobotState.PICKING, f"Executing pick for Berry {berry_letter}")
        if not self._execute_pick(fruit, berry_letter):
            self._log("error", f"Pick sequence failed for Berry {berry_letter}")
            return False
        
        self._log("info", f"✓ Berry {berry_letter} sequence completed")
        return True
    
    def _move_home(self) -> bool:
        """Move robot to Home position."""
        # Only transition if not already in MOVING state
        if self.state_machine.state != RobotState.MOVING:
            self.state_machine.transition_to(RobotState.MOVING, "Moving to Home")
        if not self.robodk_manager.move_to_target("Home", "joint", confirm=False):
            self._log("error", "Failed to move to Home position")
            return False
        self._log("info", "✓ Reached Home position")
        return True
    
    def _move_foto(self) -> bool:
        """Move robot to Foto position."""
        # Only transition if not already in MOVING state
        if self.state_machine.state != RobotState.MOVING:
            self.state_machine.transition_to(RobotState.MOVING, "Moving to Foto")
        if not self.robodk_manager.move_to_target("Foto", "joint", confirm=False):
            self._log("error", "Failed to move to Foto position")
            return False
        self._log("info", "✓ Reached Foto position")
        return True
    
    def _capture_detections(self, json_path: Optional[str] = None) -> List[FruitDetection]:
        """
        Capture detections from various sources based on config.
        
        Supports three modes (via pnp_data.source_mode):
        - "vision": Request from vision service via IPC
        - "api": Direct camera capture (not yet implemented in controller)
        - "json": Load from JSON file (offline mode)
        
        Args:
            json_path: Optional path to JSON file (overrides config)
        
        Returns:
            List of FruitDetection objects
        """
        # Check config for source mode
        pnp_config = self.config.get('pnp_data', {})
        source_mode = pnp_config.get('source_mode', 'json')
        
        # If json_path provided, force JSON mode
        if json_path:
            source_mode = 'json'
        
        # If in offline_mode, force JSON mode
        if self.offline_mode:
            source_mode = 'json'
        
        self._log("info", f"Loading PnP data (source_mode: {source_mode})...")
        
        if source_mode == 'vision':
            return self._capture_from_vision_service()
        elif source_mode == 'api':
            self._log("error", "API source mode not yet implemented in controller")
            self._log("info", "Falling back to JSON mode...")
            return self._load_detections_from_json(json_path=json_path)
        else:  # json
            return self._load_detections_from_json(json_path=json_path)
    
    def _capture_from_vision_service(self) -> List[FruitDetection]:
        """Capture detections from vision service via IPC."""
        if not self.vision_client:
            self._log("error", "Vision service not available!")
            return []
        
        vision_config = self.config.get('vision_service', {})
        pnp_config = self.config.get('pnp_data', {}).get('json', {})
        
        multi_frame = vision_config.get('multi_frame_enabled', True)
        num_frames = vision_config.get('num_frames', 10)
        min_confidence = pnp_config.get('min_confidence', 0.5)
        preferred_class = pnp_config.get('prefer_class', 'ripe')
        
        self._log("info", "Requesting capture from vision service...")
        
        try:
            detections = self.vision_client.request_capture(
                multi_frame=multi_frame,
                num_frames=num_frames,
                min_confidence=min_confidence,
                class_filter=[preferred_class] if preferred_class else None
            )
            
            self._log("info", f"✓ Received {len(detections)} valid detection(s) from vision service")
            return detections
        except Exception as e:
            self._log("error", f"Vision service request failed: {e}")
            return []
    
    def _load_detections_from_json(self, json_path: Optional[str] = None) -> List[FruitDetection]:
        """Load detections from JSON file in offline/debug mode."""
        captures_dir = REPO_ROOT / "pickafresa_vision/captures"
        
        # If json_path provided, use it directly
        if json_path:
            json_file = Path(json_path)
            if not json_file.is_absolute():
                json_file = captures_dir / json_path
            self._log("info", f"Using provided JSON file: {json_file.name}")
        else:
            # Auto-select latest JSON file
            if captures_dir.exists():
                json_files = sorted(captures_dir.glob("*_data.json"), reverse=True)
                if json_files:
                    json_file = json_files[0]  # Most recent
                    self._log("info", f"Auto-selected latest JSON: {json_file.name}")
                else:
                    self._log("error", "No JSON files found in captures directory")
                    return []
            else:
                self._log("error", f"Captures directory not found: {captures_dir}")
                return []
        
        # Load JSON file
        if not json_file.exists():
            self._log("error", f"JSON file not found: {json_file}")
            return []
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            self._log("info", f"✓ Loaded data from: {json_file.name}")
            
            # Extract detections (newer format has pose data embedded, older has separate pose_results)
            detections_data = data.get('detections', [])
            
            # Check if using old format (separate pose_results array) or new format (embedded)
            if 'pose_results' in data:
                # Old format: merge detection + pose_results
                pose_results = data.get('pose_results', [])
                merged = []
                for det, pose in zip(detections_data, pose_results):
                    merged_dict = {**det, **pose}
                    merged.append(merged_dict)
                detections_data = merged
            
            # Convert to FruitDetection objects (filter for successful pose estimation)
            fruit_detections = []
            for detection in detections_data:
                if detection.get('success', False):
                    fruit_detections.append(FruitDetection(detection))
            
            self._log("info", f"✓ Loaded {len(fruit_detections)} valid detection(s) from JSON")
            
            # Apply config filters if specified
            pnp_config = self.config.get('pnp_data', {}).get('json', {})
            min_confidence = pnp_config.get('min_confidence', 0.0)
            preferred_class = pnp_config.get('prefer_class')
            
            if min_confidence > 0.0:
                fruit_detections = [d for d in fruit_detections if d.confidence >= min_confidence]
                self._log("info", f"  After confidence filter (>={min_confidence}): {len(fruit_detections)} detection(s)")
            
            if preferred_class:
                preferred = [d for d in fruit_detections if d.class_name == preferred_class]
                if preferred:
                    fruit_detections = preferred
                    self._log("info", f"  Filtered to preferred class '{preferred_class}': {len(fruit_detections)} detection(s)")
            
            return fruit_detections
        
        except Exception as e:
            self._log("error", f"Failed to load JSON file: {e}")
            return []
    
    def _select_berry(self, detections: List[FruitDetection], index: int) -> Optional[FruitDetection]:
        """Select a berry from detections by index."""
        if index >= len(detections):
            self._log("error", f"Berry index {index} out of range (have {len(detections)})")
            return None
        
        berry = detections[index]
        self._log("info", f"Selected berry #{index}: {berry.class_name} (conf={berry.confidence:.2f})")
        return berry
    
    def _execute_pick(self, fruit: FruitDetection, berry_letter: Optional[str] = None) -> bool:
        """
        Execute pick-place sequence for a fruit.
        
        Routes to YAML-driven or legacy hardcoded sequence based on config.
        
        Args:
            fruit: FruitDetection object with pose information
            berry_letter: Optional override for berry letter (A, B, C, ...)
        """
        # Generate berry letter for target naming (A, B, C, ...)
        if berry_letter is None:
            berry_letter = chr(65 + self.current_berry_index)  # A=65 in ASCII
        
        # === DEBUG LOGGING (matches CLI pattern at lines 1610-1616) ===
        if fruit.position_base is not None:
            self._log("info", f"Fruit position in base frame (position_base):")
            self._log("info", f"  [{fruit.position_base[0]:.3f}, {fruit.position_base[1]:.3f}, {fruit.position_base[2]:.3f}] m")
            self._log("info", f"  [{fruit.position_base[0]*1000:.1f}, {fruit.position_base[1]*1000:.1f}, {fruit.position_base[2]*1000:.1f}] mm")
        
        if fruit.T_base_fruit is not None:
            self._log("info", f"T_base_fruit full matrix:")
            for i in range(4):
                self._log("info", f"  [{fruit.T_base_fruit[i,0]:9.6f}, {fruit.T_base_fruit[i,1]:9.6f}, {fruit.T_base_fruit[i,2]:9.6f}, {fruit.T_base_fruit[i,3]:9.6f}]")
        else:
            self._log("error", "T_base_fruit is None! Transform failed.")
            return False
        
        # Create prepick and pick targets using CLI's proven method
        if not self._create_berry_targets(fruit, berry_letter):
            self._log("error", "Failed to create berry targets")
            return False
        
        # === CHECK EXECUTION MODE (YAML vs Legacy) ===
        sequence_config = self.config.get('sequence', {})
        execution_mode = sequence_config.get('execution_mode', 'yaml')
        
        if execution_mode == 'yaml' and 'per_berry_steps' in sequence_config:
            # Use YAML-driven per-berry sequence
            self._log("info", f"Using YAML-driven sequence (execution_mode={execution_mode})")
            return self._execute_per_berry_yaml_sequence(fruit, berry_letter)
        else:
            # Use legacy hard-coded per-berry sequence
            self._log("info", f"Using legacy hardcoded sequence (execution_mode={execution_mode})")
            return self._execute_legacy_berry_sequence(fruit, berry_letter)
    
    def _execute_per_berry_yaml_sequence(self, fruit: FruitDetection, berry_letter: str) -> bool:
        """
        Execute per-berry sequence dynamically from YAML configuration.
        
        This method reads the sequence steps from the YAML config and executes them,
        allowing for complete configurability of the pick-place sequence.
        
        Args:
            fruit: Fruit detection with pose information
            berry_letter: Berry letter identifier (A, B, C, etc.)
        
        Returns:
            True if sequence completed successfully
        """
        self._log("info", "=" * 60)
        self._log("info", f"EXECUTING YAML PER-BERRY SEQUENCE (Berry {berry_letter})")
        self._log("info", "=" * 60)
        
        sequence_config = self.config.get('sequence', {})
        per_berry_steps = sequence_config.get('per_berry_steps', [])
        
        if not per_berry_steps:
            self._log("error", "No per_berry_steps defined in YAML!")
            return False
        
        self._log("info", f"Loaded {len(per_berry_steps)} per-berry steps from YAML")
        
        # Track if we've closed the gripper (to know when to insert post-pick)
        gripper_closed = False
        post_pick_executed = False
        
        # Execute each step
        for i, step in enumerate(per_berry_steps, 1):
            step = step.copy()  # Make a copy to avoid modifying original
            step_name = step.get('name', f'berry_step_{i}')
            step_type = step.get('type', '')
            
            # Replace {berry_letter} placeholder in target names
            if 'target' in step and '{berry_letter}' in step['target']:
                step['target'] = step['target'].replace('{berry_letter}', berry_letter)
            
            self._log("info", f"  [{i}/{len(per_berry_steps)}] {step_name}")
            
            # Execute the step
            success = self._execute_sequence_step(step, i, fruit, berry_letter)
            
            if not success:
                self._log("error", f"Per-berry step {i} ({step_name}) failed!")
                return False
            
            # Track gripper state
            if step_type == 'gripper' and step.get('action') == 'close':
                gripper_closed = True
            
            # Execute post-pick detachment sequence AFTER gripper closes and BEFORE next move/place
            if gripper_closed and not post_pick_executed:
                # Check if next step is a move (not another gripper action)
                next_step_is_move = False
                if i < len(per_berry_steps):
                    next_step = per_berry_steps[i]
                    if next_step.get('type') == 'move':
                        next_step_is_move = True
                
                # Execute post-pick if enabled and before next movement
                if next_step_is_move:
                    post_pick_config = self.config.get('post_pick', {})
                    if post_pick_config.get('enabled', False):
                        self._log("info", "")
                        self._log("info", "=" * 60)
                        self._log("info", "EXECUTING POST-PICK DETACHMENT SEQUENCE")
                        self._log("info", "=" * 60)
                        
                        # Get collision config
                        collision_config = self.config.get('collision_avoidance', {})
                        collision_config = self._enrich_collision_config(collision_config)
                        
                        # Execute post-pick detachment (creates targets dynamically during execution)
                        success = self._execute_post_pick_detachment(
                            fruit.T_base_fruit, 
                            berry_letter, 
                            collision_config
                        )
                        
                        if not success:
                            self._log("warning", "Post-pick sequence failed, continuing with placement...")
                        
                        post_pick_executed = True
        
        self._log("info", f"[OK] Berry {berry_letter} sequence completed")
        return True
    
    def _execute_legacy_berry_sequence(self, fruit: FruitDetection, berry_letter: str) -> bool:
        """
        Execute the legacy hard-coded per-berry sequence.
        
        This is the original implementation kept for backwards compatibility.
        """
        self._log("info", "Executing legacy per-berry sequence...")
        
        # Get collision configuration (enriched with run_mode and simulation_mode)
        collision_config = self.config.get('collision_avoidance', {})
        collision_config = self._enrich_collision_config(collision_config)
        use_collision_avoidance = collision_config.get('enabled', True)
        
        # Get movement confirmation settings (should be False for service mode)
        confirm_movement = False  # Service always runs without confirmation
        highlight = False  # No highlight in service mode
        
        # === STEP 1: Move to Prepick_plane (safe approach height) ===
        self._log("info", "Step 1: Moving to Prepick_plane (safe approach height)...")
        
        if use_collision_avoidance:
            success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                "Prepick_plane", "joint", confirm_movement, highlight,
                enable_collision_avoidance=True, collision_config=collision_config
            )
            if not success:
                self._log("error", f"Failed to reach Prepick_plane: {message}")
                return False
            if message:
                self._log("info", f"Prepick_plane movement: {message}")
        else:
            if not self.robodk_manager.move_to_target("Prepick_plane", "joint", confirm_movement, highlight):
                self._log("error", "Failed to reach Prepick_plane")
                return False
        
        # === STEP 2: Move to berry-specific prepick position ===
        self._log("info", f"Step 2: Moving to berry-specific prepick: prepick_{berry_letter}")
        
        if use_collision_avoidance:
            success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                f"prepick_{berry_letter}", "linear", confirm_movement, highlight,
                enable_collision_avoidance=True, collision_config=collision_config
            )
            if not success:
                self._log("error", f"Failed to reach prepick position: {message}")
                return False
            if message:
                self._log("info", f"Prepick movement: {message}")
        else:
            if not self.robodk_manager.move_to_target(f"prepick_{berry_letter}", "linear", confirm_movement, highlight):
                self._log("error", f"Failed to reach prepick position for berry {self.current_berry_index}")
                return False
        
        # === STEP 3: Move to pick position (linear approach) ===
        self._log("info", f"Step 3: Moving to pick position: pick_{berry_letter}")
        
        if use_collision_avoidance:
            success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                f"pick_{berry_letter}", "linear", confirm_movement, highlight,
                enable_collision_avoidance=True, collision_config=collision_config
            )
            if not success:
                self._log("error", f"Failed to reach pick position: {message}")
                return False
            if message:
                self._log("info", f"Pick movement: {message}")
        else:
            if not self.robodk_manager.move_to_target(f"pick_{berry_letter}", "linear", confirm_movement, highlight):
                self._log("error", f"Failed to reach pick position for berry {self.current_berry_index}")
                return False
        
        # === STEP 4: Close gripper ===
        if self.mqtt_gripper:
            self._log("info", "Step 4: Closing gripper...")
            try:
                self.mqtt_gripper.close_gripper()
                time.sleep(1.0)  # Wait for grip to stabilize
            except Exception as e:
                self._log("warning", f"Gripper close failed: {e}, continuing anyway")
        else:
            self._log("info", "Step 4: Gripper not configured (skipping)")
        
        # === STEP 5: Post-pick detachment sequence (if enabled) ===
        post_pick_config = self.config.get('post_pick', {})
        post_pick_enabled = post_pick_config.get('enabled', False)
        
        if post_pick_enabled:
            self._log("info", "=" * 60)
            self._log("info", "EXECUTING POST-PICK DETACHMENT SEQUENCE")
            self._log("info", "=" * 60)
            
            # Execute post-pick detachment (creates targets dynamically during execution)
            if not self._execute_post_pick_detachment(fruit.T_base_fruit, berry_letter, collision_config):
                self._log("warning", "Post-pick sequence failed, continuing...")
        
        # === STEP 6: Move to Home (NOT place_final!) ===
        # CLI moves to Home after post-pick to reset robot position
        self._log("info", "Step 6: Returning to Home position...")
        
        # Check if Home target exists
        if self.robodk_manager.get_target("Home") is None:
            self._log("error", "Target 'Home' not found in RoboDK station!")
            return False
        
        if use_collision_avoidance:
            success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                "Home", "joint", confirm_movement, highlight,
                enable_collision_avoidance=True, collision_config=collision_config
            )
            if not success:
                self._log("error", f"Failed to reach Home: {message}")
                return False
            if message:
                self._log("info", f"Home movement: {message}")
        else:
            if not self.robodk_manager.move_to_target("Home", "joint", confirm_movement, highlight):
                self._log("error", "Failed to reach Home")
                return False
        
        # === STEP 7: Open gripper (release berry at Home) ===
        if self.mqtt_gripper:
            self._log("info", "Step 7: Opening gripper (releasing berry)...")
            try:
                self.mqtt_gripper.open_gripper()
                time.sleep(0.5)  # Wait for release
            except Exception as e:
                self._log("warning", f"Gripper open failed: {e}, continuing anyway")
        else:
            self._log("info", "Step 7: Gripper not configured (skipping)")
        
        self._log("info", "[OK] Pick-place sequence completed successfully")
        return True
    
    def _execute_sequence_step(
        self,
        step: Dict[str, Any],
        step_index: int,
        fruit: FruitDetection,
        berry_letter: str
    ) -> bool:
        """
        Execute a single sequence step based on its type.
        
        Args:
            step: Step configuration dictionary
            step_index: Step number (for logging)
            fruit: Fruit detection (needed for some step types)
            berry_letter: Berry letter identifier
        
        Returns:
            True if step succeeded, False otherwise
        """
        step_type = step.get('type', 'unknown')
        step_name = step.get('name', f'step_{step_index}')
        
        # Route to appropriate handler based on step type
        if step_type == 'move':
            return self._execute_move_step(step)
        elif step_type == 'gripper':
            return self._execute_gripper_step(step)
        elif step_type == 'wait':
            return self._execute_wait_step(step)
        else:
            self._log("error", f"Unknown step type: '{step_type}' for step '{step_name}'")
            return False
    
    def _execute_move_step(self, step: Dict[str, Any]) -> bool:
        """
        Execute a movement step.
        
        Step format:
            type: move
            target: "Home" | "Foto" | "Prepick_plane" | etc.
            move_type: "joint" | "linear"
            confirm: true/false (ignored in service mode)
        """
        target = step.get('target', '')
        move_type = step.get('move_type', 'joint')
        confirm = False  # Service mode never confirms
        highlight = False
        
        if not target:
            self._log("error", "Move step missing 'target' parameter")
            return False
        
        self._log("info", f"Moving to target: {target} (mode: {move_type})")
        
        # Check if collision avoidance is enabled
        collision_config = self.config.get('collision_avoidance', {})
        collision_config = self._enrich_collision_config(collision_config)
        use_collision_avoidance = collision_config.get('enabled', True)
        
        if use_collision_avoidance:
            success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                target_name=target,
                move_type=move_type,
                confirm=confirm,
                highlight=highlight,
                enable_collision_avoidance=True,
                collision_config=collision_config
            )
            if not success:
                self._log("error", f"Failed to reach {target}: {message}")
                return False
            if message:
                self._log("info", f"Movement result: {message}")
            return True
        else:
            return self.robodk_manager.move_to_target(
                target_name=target,
                move_type=move_type,
                confirm=confirm,
                highlight=highlight
            )
    
    def _execute_gripper_step(self, step: Dict[str, Any]) -> bool:
        """
        Execute a gripper action step.
        
        Step format:
            type: gripper
            action: "open" | "close"
            confirm: true/false (ignored in service mode)
        """
        action = step.get('action', '').lower()
        
        if action not in ['open', 'close']:
            self._log("error", f"Invalid gripper action: '{action}' (must be 'open' or 'close')")
            return False
        
        if not self.mqtt_gripper:
            self._log("info", f"Gripper {action} requested but gripper not configured (skipping)")
            return True
        
        try:
            if action == 'close':
                self._log("info", "Closing gripper...")
                self.mqtt_gripper.close_gripper()
                time.sleep(1.0)  # Wait for grip to stabilize
            else:
                self._log("info", "Opening gripper...")
                self.mqtt_gripper.open_gripper()
                time.sleep(0.5)  # Wait for release
            return True
        except Exception as e:
            self._log("warning", f"Gripper {action} failed: {e}, continuing anyway")
            return True  # Don't fail sequence on gripper errors
    
    def _execute_wait_step(self, step: Dict[str, Any]) -> bool:
        """
        Execute a wait/pause step.
        
        Step format:
            type: wait
            duration: 2.0  # seconds
            message: "Waiting for stabilization..."  # optional
        """
        duration = step.get('duration', 0)
        message = step.get('message', '')
        
        if message:
            self._log("info", message)
        
        if duration > 0:
            self._log("info", f"Waiting for {duration} seconds...")
            time.sleep(duration)
        
        return True
    
    def _create_berry_targets(self, fruit: FruitDetection, berry_letter: str) -> bool:
        """
        Create RoboDK targets for prepick and pick positions.
        
        This follows the EXACT proven pattern from robot_pnp_cli._create_berry_targets().
        Offsets are applied directly in the fruit frame, then converted to millimeters.
        
        Args:
            fruit: Fruit detection with base frame transformation (T_base_fruit in METERS)
            berry_letter: Berry letter identifier (A, B, C, etc.)
        
        Returns:
            True if targets created successfully
        """
        # Get pick/prepick offset configuration
        pick_offset_config = self.config.get('transforms', {}).get('pick_offset', {})
        
        # Get prepick offsets (in mm, convert to meters for calculation)
        prepick_offset_mm = pick_offset_config.get('prepick_offset_mm', [0.0, 0.0, 100.0])
        prepick_offset_m = np.array(prepick_offset_mm) / 1000.0  # Convert mm to meters
        
        # Get prepick and pick rotations (absolute, in degrees)
        prepick_rotation_deg = pick_offset_config.get('prepick_rotation_deg', [0.0, 0.0, 0.0])
        pick_rotation_deg = pick_offset_config.get('pick_rotation_deg', [0.0, 0.0, 0.0])
        
        # Create prepick pose (offset from fruit in fruit frame)
        T_base_prepick = fruit.T_base_fruit.copy()
        
        # Apply translation offset in fruit frame
        # Fruit frame: Z+ away from camera (into fruit), X/Y aligned with image
        offset_in_base = T_base_prepick[:3, :3] @ prepick_offset_m
        T_base_prepick[:3, 3] += offset_in_base
        
        # Apply absolute rotation for prepick if specified
        if any(abs(r) > 0.01 for r in prepick_rotation_deg):
            prepick_rotation_rad = np.deg2rad(prepick_rotation_deg)
            R_prepick, _ = cv2.Rodrigues(prepick_rotation_rad)
            T_base_prepick[:3, :3] = R_prepick.astype(np.float64)
            self._log("debug", f"Applied prepick rotation: {prepick_rotation_deg}")
        
        # Create pick pose
        T_base_pick = fruit.T_base_fruit.copy()
        
        # Apply pick offset
        pick_offset_mm = pick_offset_config.get('pick_offset_mm', [0.0, 0.0, 0.0])
        pick_offset_m = np.array(pick_offset_mm) / 1000.0
        offset_in_base = T_base_pick[:3, :3] @ pick_offset_m
        T_base_pick[:3, 3] += offset_in_base
        
        # Apply absolute rotation for pick if specified
        if any(abs(r) > 0.01 for r in pick_rotation_deg):
            pick_rotation_rad = np.deg2rad(pick_rotation_deg)
            R_pick, _ = cv2.Rodrigues(pick_rotation_rad)
            T_base_pick[:3, :3] = R_pick.astype(np.float64)
            self._log("debug", f"Applied pick rotation: {pick_rotation_deg}")
        
        # CRITICAL: Convert from meters to millimeters for RoboDK
        # fruit.T_base_fruit is in meters, but RoboDK expects millimeters
        T_base_prepick_mm = T_base_prepick.copy()
        T_base_prepick_mm[:3, 3] = T_base_prepick[:3, 3] * 1000.0  # Convert to mm
        
        T_base_pick_mm = T_base_pick.copy()
        T_base_pick_mm[:3, 3] = T_base_pick[:3, 3] * 1000.0  # Convert to mm
        
        # Log created poses
        self._log("info", f"Prepick pose offset: {prepick_offset_mm} mm")
        self._log("info", f"Prepick position (base frame): [{T_base_prepick[0,3]*1000:.1f}, {T_base_prepick[1,3]*1000:.1f}, {T_base_prepick[2,3]*1000:.1f}] mm")
        self._log("info", f"Pick position (base frame): [{T_base_pick[0,3]*1000:.1f}, {T_base_pick[1,3]*1000:.1f}, {T_base_pick[2,3]*1000:.1f}] mm")
        
        # Create targets in RoboDK
        self._log("info", "Creating dynamic targets in RoboDK...")
        
        target_prepick = self.robodk_manager.create_target_from_pose(
            name=f"prepick_{berry_letter}",
            T_base_target=T_base_prepick_mm,  # RoboDK expects millimeters
            create_frame=True,
            color=[0, 255, 0]  # Green
        )
        
        target_pick = self.robodk_manager.create_target_from_pose(
            name=f"pick_{berry_letter}",
            T_base_target=T_base_pick_mm,  # RoboDK expects millimeters
            create_frame=True,
            color=[255, 0, 0]  # Red
        )
        
        if target_prepick is None or target_pick is None:
            self._log("error", "Failed to create targets")
            return False
        
        self._log("info", f"✓ Created targets: prepick_{berry_letter}, pick_{berry_letter}")
        return True
    
    def _execute_post_pick_detachment(self, T_base_fruit: np.ndarray, berry_label: str, collision_config: Dict[str, Any]) -> bool:
        """
        Execute post-pick detachment sequence.
        
        CRITICAL: Creates targets DYNAMICALLY during execution (like CLI).
        This is essential because joint state changes after each move, affecting subsequent targets.
        
        Args:
            T_base_fruit: Fruit pose in base frame (4x4 transform, in METERS)
            berry_label: Berry label for target naming (A, B, C, etc.)
            collision_config: Enriched collision configuration
        
        Returns:
            True if all detachment movements succeeded
        """
        post_pick_config = self.config.get('post_pick', {})
        num_targets = post_pick_config.get('num_targets', 3)
        target_configs = post_pick_config.get('targets', [])
        rotation_mode = post_pick_config.get('rotation_mode', 'cumulative')
        use_collision_avoidance = collision_config.get('enabled', True)
        
        if not target_configs:
            self._log("warning", "No post-pick target configs found")
            return True
        
        # Limit to configured number
        target_configs = target_configs[:num_targets]
        
        self._log("info", f"Executing post-pick detachment ({len(target_configs)} targets)...")
        self._log("debug", f"Rotation mode: {rotation_mode}")
        
        # Get current robot pose (should be at pick position)
        try:
            pose_robodk = self.robodk_manager.robot.Pose()
            T_current = np.array([
                pose_robodk[0, :],
                pose_robodk[1, :],
                pose_robodk[2, :],
                pose_robodk[3, :]
            ], dtype=np.float64)
            T_current = np.squeeze(T_current)
            T_current[:3, 3] /= 1000.0  # Convert mm to meters
        except Exception as e:
            self._log("error", f"Failed to get robot pose: {e}")
            return False
        
        # Get initial joint configuration
        try:
            current_joints = self.robodk_manager.robot.Joints().list()
            cumulative_joints = list(current_joints)
            self._log("debug", f"Initial joints at pick: {[f'{j:.2f}' for j in cumulative_joints]}")
        except Exception as e:
            self._log("error", f"Failed to get joint configuration: {e}")
            return False
        
        # Store initial fruit pose for absolute rotation mode
        # (passed as parameter - the fruit pose in base frame from PnP)
        # T_base_fruit is already in METERS (from transform_to_base_frame)
        
        # Execute each target dynamically
        for i, target_config in enumerate(target_configs, start=1):
            target_name = target_config.get('name', f'post_pick_{i}')
            move_type = target_config.get('move_type', 'linear')
            joint_deltas_deg = target_config.get('joint_deltas_deg', None)
            robodk_target_name = f"{target_name}_{berry_label}"
            
            if joint_deltas_deg is not None:
                # ==================== JOINT SPACE CONTROL ====================
                self._log("info", f"  Post-pick target {i}/{len(target_configs)}: {robodk_target_name} [JOINT CONTROL] ({move_type})")
                self._log("debug", f"    Joint deltas: {joint_deltas_deg} deg")
                
                if len(joint_deltas_deg) != 6:
                    self._log("error", f"Invalid joint_deltas_deg: expected 6, got {len(joint_deltas_deg)}")
                    return False
                
                # Apply cumulative joint deltas
                target_joints = [cumulative_joints[j] + joint_deltas_deg[j] for j in range(6)]
                self._log("debug", f"    Current joints: {[f'{j:.2f}' for j in cumulative_joints]}")
                self._log("debug", f"    Target joints:  {[f'{j:.2f}' for j in target_joints]}")
                
                # Create target NOW
                success = self.robodk_manager.create_target_from_joints(
                    name=robodk_target_name,
                    joints=target_joints,
                    color=[255, 165, 0]
                )
                
                if not success:
                    self._log("error", f"Failed to create joint target: {robodk_target_name}")
                    return False
                
                # Move to target
                if use_collision_avoidance:
                    success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                        robodk_target_name, move_type, confirm=False, highlight=False,
                        enable_collision_avoidance=True, collision_config=collision_config
                    )
                    if not success:
                        self._log("error", f"Post-pick target {i} failed: {message}")
                        return False
                    if message:
                        self._log("debug", f"  {message}")
                else:
                    if not self.robodk_manager.move_to_target(robodk_target_name, move_type, confirm=False, highlight=False):
                        self._log("error", f"Post-pick target {i} failed")
                        return False
                
                # Update cumulative joints for next iteration
                cumulative_joints = list(target_joints)
                
                # Update T_current for consistency
                try:
                    pose_robodk = self.robodk_manager.robot.Pose()
                    T_current = np.array([
                        pose_robodk[0, :],
                        pose_robodk[1, :],
                        pose_robodk[2, :],
                        pose_robodk[3, :]
                    ], dtype=np.float64)
                    T_current = np.squeeze(T_current)
                    T_current[:3, 3] /= 1000.0
                except Exception as e:
                    self._log("error", f"Failed to update robot state: {e}")
                    return False
                
            else:
                # ==================== CARTESIAN CONTROL ====================
                offset_mm = target_config.get('offset_mm', [0.0, 0.0, 0.0])
                rotation_deg = target_config.get('rotation_deg', [0.0, 0.0, 0.0])
                
                self._log("info", f"  Post-pick target {i}/{len(target_configs)}: {robodk_target_name} [CARTESIAN CONTROL] ({move_type})")
                self._log("debug", f"    Offset: {offset_mm} mm, Rotation: {rotation_deg} deg")
                
                # Apply translational offset
                offset_m = np.array(offset_mm) / 1000.0
                T_target = T_current.copy()
                
                if rotation_mode == 'absolute':
                    # Use original fruit frame for offset transform
                    offset_in_base = T_base_fruit[:3, :3] @ offset_m
                else:
                    # Use current accumulated frame
                    offset_in_base = T_current[:3, :3] @ offset_m
                
                T_target[:3, 3] += offset_in_base
                
                # Apply rotation
                if any(abs(r) > 0.01 for r in rotation_deg):
                    import cv2
                    rotation_rad = np.deg2rad(rotation_deg)
                    R_offset, _ = cv2.Rodrigues(np.array(rotation_rad))
                    R_offset = R_offset.astype(np.float64)
                    
                    if rotation_mode == 'absolute':
                        # Absolute: rotation in BASE frame, applied cumulatively
                        T_target[:3, :3] = R_offset @ T_base_fruit[:3, :3]
                    else:
                        # Cumulative: rotation in CURRENT gripper frame
                        T_target[:3, :3] = T_target[:3, :3] @ R_offset
                
                # Convert to millimeters for RoboDK
                T_target_mm = T_target.copy()
                T_target_mm[:3, 3] *= 1000.0
                
                # Create target NOW
                created_target = self.robodk_manager.create_target_from_pose(
                    name=robodk_target_name,
                    T_base_target=T_target_mm,
                    create_frame=False,
                    color=[255, 165, 0]
                )
                
                if created_target is None:
                    self._log("error", f"Failed to create Cartesian target: {robodk_target_name}")
                    return False
                
                # Move to target
                if use_collision_avoidance:
                    success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                        robodk_target_name, move_type, confirm=False, highlight=False,
                        enable_collision_avoidance=True, collision_config=collision_config
                    )
                    if not success:
                        self._log("error", f"Post-pick target {i} failed: {message}")
                        return False
                    if message:
                        self._log("debug", f"  {message}")
                else:
                    if not self.robodk_manager.move_to_target(robodk_target_name, move_type, confirm=False, highlight=False):
                        self._log("error", f"Post-pick target {i} failed")
                        return False
                
                # CRITICAL: Update joint state for next iteration (Cartesian moves don't track joints manually)
                try:
                    cumulative_joints = self.robodk_manager.robot.Joints().list()
                    self._log("debug", f"    Updated joints after Cartesian move: {[f'{j:.2f}' for j in cumulative_joints]}")
                except Exception as e:
                    self._log("error", f"Failed to update joint state: {e}")
                    return False
                
                # Update current transform
                T_current = T_target.copy()
        
        self._log("info", "[OK] Post-pick detachment sequence completed")
        return True
    
    def _create_post_pick_targets(self, T_base_fruit: np.ndarray, berry_label: str) -> bool:
        """
        [OBSOLETE] This method is no longer used.
        
        Post-pick targets are now created dynamically DURING execution in 
        _execute_post_pick_detachment() to avoid joint state divergence.
        
        Kept for reference only. DO NOT CALL THIS METHOD.
        """
        self._log("warning", "[OBSOLETE] _create_post_pick_targets() should not be called!")
        self._log("warning", "Post-pick targets are created dynamically during execution.")
        return False
    
    def _enrich_collision_config(self, collision_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich collision config with run_mode and simulation_mode.
        
        These parameters are needed by the collision avoidance system
        to make safety decisions about force overrides and auto-aborts.
        
        Args:
            collision_config: Base collision configuration
        
        Returns:
            Enriched collision configuration
        """
        enriched = collision_config.copy()
        
        # Get run mode from main config
        run_mode = self.config.get('run_mode', 'manual_confirm')
        enriched['run_mode'] = run_mode
        
        # Get simulation mode from RoboDK config
        robodk_config = self.config.get('robodk', {})
        simulation_mode = robodk_config.get('run_mode', 'simulate')
        enriched['simulation_mode'] = simulation_mode
        
        return enriched
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current controller status.
        
        Returns:
            Status dictionary
        """
        return {
            'initialized': self.is_initialized,
            'state': self.state_machine.state_name,
            'operational': self.state_machine.is_operational(),
            'busy': self.state_machine.is_busy(),
            'error': self.state_machine.is_error_state(),
            'robodk_connected': self.robodk_manager is not None and self.robodk_manager.robot is not None,
            'vision_connected': self.vision_client is not None,
            'gripper_connected': self.mqtt_gripper is not None,
            'current_berry': self.current_berry_index
        }


# Example usage
if __name__ == "__main__":
    from pickafresa_robot.robot_system.ros2_logger import create_logger
    
    print("Robot PnP Controller - Test")
    print("=" * 70)
    
    # Load config
    config_path = REPO_ROOT / "pickafresa_robot/configs/robot_pnp_config.yaml"
    config = ConfigManager(config_path)
    
    # Create logger
    logger = create_logger(node_name="controller_test", log_dir=str(REPO_ROOT / "pickafresa_robot/logs"))
    
    # Create controller
    controller = RobotPnPController(config=config, logger=logger)
    
    # Initialize
    print("\n[1] Initializing controller...")
    if controller.initialize():
        print("✓ Controller initialized")
        
        # Get status
        status = controller.get_status()
        print(f"\nStatus: {status}")
        
        # Execute pick (requires RoboDK and vision service running)
        # controller.execute_pick_sequence(berry_index=0)
        
        # Shutdown
        controller.shutdown()
        print("\n✓ Test complete")
    else:
        print("✗ Controller initialization failed")
