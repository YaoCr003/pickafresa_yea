"""
Interactive Robot Pick-and-Place Testing Tool

A comprehensive CLI tool for testing PnP-robot connection, robot movement,
and berry grabbing in RoboDK simulation environment.

Features:
- Interactive configuration (YAML or manual prompts)
- Multiple run modes (simulation/real robot)
- MQTT gripper control (optional)
- PnP data from API (live) or JSON (offline)
- User confirmations at each step for safety
- ROS2-style logging
- Multi-berry support

Usage:
    python pickafresa_robot/robot_testing/robot_pnp_cli.py [--config path/to/config.yaml]

# @aldrick-t, 2025
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import cv2

# Repository root setup
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local imports
from pickafresa_robot.robot_testing.ros2_logger import create_logger
from pickafresa_robot.robot_testing.mqtt_gripper import MQTTGripperController
from pickafresa_robot.robot_testing.pnp_handler import (
    PnPDataHandler, FruitDetection, create_transform_matrix
)
from pickafresa_robot.robot_testing.robodk_manager import RoboDKManager

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False
    yaml = None

try:
    import keyboard
    HAVE_KEYBOARD = True
except ImportError:
    HAVE_KEYBOARD = False

try:
    import socket
    import json
    HAVE_SOCKET = True
except ImportError:
    HAVE_SOCKET = False


class VisionServiceError(Exception):
    """Raised when vision service is not available or fails."""
    pass


class VisionServiceClient:
    """Client for communicating with the vision service via IPC."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: float = 30.0):
        """
        Initialize vision service client.
        
        Args:
            host: Vision service host address
            port: Vision service port
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.logger = None
    
    def set_logger(self, logger):
        """Set logger instance."""
        self.logger = logger
    
    def _log(self, level: str, message: str):
        """Internal logging helper."""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def connect(self) -> bool:
        """
        Connect to vision service.
        
        Returns:
            True if connected successfully
        
        Raises:
            VisionServiceError: If connection fails
        """
        if not HAVE_SOCKET:
            raise VisionServiceError("Socket module not available")
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self._log("info", f"[OK] Connected to vision service at {self.host}:{self.port}")
            return True
        
        except socket.timeout:
            raise VisionServiceError(f"Connection timeout to {self.host}:{self.port}")
        except ConnectionRefusedError:
            raise VisionServiceError(f"Connection refused by {self.host}:{self.port} - Is vision service running?")
        except Exception as e:
            raise VisionServiceError(f"Failed to connect to vision service: {e}")
    
    def disconnect(self):
        """Disconnect from vision service."""
        if self.socket:
            try:
                self.socket.close()
                self._log("info", "Disconnected from vision service")
            except:
                pass
            self.socket = None
    
    def check_status(self) -> Dict[str, Any]:
        """
        Check vision service status.
        
        Returns:
            Status dictionary with keys: alive, ready, error
        
        Raises:
            VisionServiceError: If request fails
        """
        request = {"command": "status"}
        return self._send_request(request)
    
    def request_capture(self, multi_frame: bool = False, num_frames: int = 10) -> Dict[str, Any]:
        """
        Request a capture from vision service.
        
        Args:
            multi_frame: Enable multi-frame averaging
            num_frames: Number of frames to average
        
        Returns:
            Response dictionary with keys: success, detections, error
            Each detection contains: class_name, confidence, bbox, T_cam_fruit, T_base_fruit
        
        Raises:
            VisionServiceError: If request fails
        """
        request = {
            "command": "capture",
            "multi_frame": multi_frame,
            "num_frames": num_frames
        }
        return self._send_request(request)
    
    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request to vision service and receive response.
        
        Args:
            request: Request dictionary
        
        Returns:
            Response dictionary
        
        Raises:
            VisionServiceError: If communication fails
        """
        if not self.socket:
            raise VisionServiceError("Not connected to vision service")
        
        try:
            # Send request
            request_json = json.dumps(request)
            self.socket.sendall(request_json.encode('utf-8') + b'\n')
            
            # Receive response (read until newline)
            response_data = b''
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    raise VisionServiceError("Connection closed by vision service")
                response_data += chunk
                if b'\n' in response_data:
                    break
            
            # Parse response
            response_json = response_data.decode('utf-8').strip()
            response = json.loads(response_json)
            
            return response
        
        except socket.timeout:
            raise VisionServiceError("Request timeout - vision service not responding")
        except json.JSONDecodeError as e:
            raise VisionServiceError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise VisionServiceError(f"Communication error: {e}")


class RobotPnPCLI:
    """Main CLI application for robot PnP testing."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the CLI application.
        
        Args:
            config_path: Path to configuration YAML file (optional)
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.logger = None
        self.robodk_manager: Optional[RoboDKManager] = None
        self.mqtt_controller: Optional[MQTTGripperController] = None
        self.pnp_handler: Optional[PnPDataHandler] = None
        self.vision_client: Optional[VisionServiceClient] = None
        
        # Runtime state
        self.use_config = True
        self.selected_fruits: List[FruitDetection] = []
        self.use_vision_service = False  # Whether to use vision service or direct capture
        
        print("\n" + "="*70)
        print(" "*15 + "ROBOT PNP TESTING TOOL")
        print(" "*20 + "Team YEA, 2025")
        print("="*70 + "\n")
    
    def run(self) -> int:
        """
        Main execution flow.
        
        Returns:
            Exit code (0 = success, 1 = error)
        """
        try:
            # Step 1: Configuration
            if not self._setup_configuration():
                return 1
            
            # Step 2: Initialize logger
            self._setup_logger()
            
            self.logger.info("="*60)
            self.logger.info("Robot PnP Testing Tool Started")
            self.logger.info("="*60)
            
            # Step 3: Initialize components
            if not self._initialize_components():
                return 1
            
            # Step 4: Main sequence
            if not self._execute_pnp_sequence():
                return 1
            
            self.logger.info("[OK] All operations completed successfully")
            
            return 0
        
        except KeyboardInterrupt:
            if self.logger:
                self.logger.error("\n!!! Program interrupted by user (Ctrl+C) !!!")
            else:
                print("\n!!! Program interrupted by user (Ctrl+C) !!!")
            return 1
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"!!! Fatal error: {e} !!!")
                import traceback
                self.logger.error(traceback.format_exc())
            else:
                print(f"!!! Fatal error: {e} !!!")
            return 1
        
        finally:
            self._cleanup()
    
    def _setup_configuration(self) -> bool:
        """Load or prompt for configuration."""
        print("CONFIGURATION SETUP")
        print("-" * 70)
        
        # Ask user: use config file or go through prompts?
        if self.config_path and self.config_path.exists():
            response = input(f"\nUse configuration from '{self.config_path.name}'? [Y/n]: ").strip().lower()
            self.use_config = response != 'n'
        else:
            response = input("\nUse configuration file? [y/N]: ").strip().lower()
            self.use_config = response == 'y'
            
            if self.use_config:
                # Prompt for config path
                default_config = REPO_ROOT / "pickafresa_robot/configs/robot_pnp_config.yaml"
                config_input = input(f"Config file path [{default_config}]: ").strip()
                
                if config_input:
                    self.config_path = Path(config_input)
                else:
                    self.config_path = default_config
        
        # Load or create config
        if self.use_config:
            return self._load_config_file()
        else:
            return self._create_config_interactive()
    
    def _load_config_file(self) -> bool:
        """Load configuration from YAML file."""
        if not HAVE_YAML:
            print("ERROR: PyYAML not installed. Cannot load config file.")
            print("Install with: pip install pyyaml")
            return False
        
        if not self.config_path or not self.config_path.exists():
            print(f"ERROR: Config file not found: {self.config_path}")
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            print(f"[OK] Configuration loaded from: {self.config_path}")
            return True
        
        except Exception as e:
            print(f"ERROR: Failed to load config: {e}")
            return False
    
    def _create_config_interactive(self) -> bool:
        """Create configuration interactively through prompts."""
        print("\nINTERACTIVE CONFIGURATION")
        print("-" * 70)
        
        config = {}
        
        # RoboDK settings
        print("\n[1/5] RoboDK Settings")
        station = input("  Station file path [pickafresa_robot/rdk/SETUP Fresas.rdk]: ").strip()
        robot = input("  Robot model [UR3e]: ").strip()
        mode = input("  Run mode - (s)imulate or (r)eal robot? [s]: ").strip().lower()
        
        config['robodk'] = {
            'station_file': station or "pickafresa_robot/rdk/SETUP Fresas.rdk",
            'robot_model': robot or "UR3e",
            'run_mode': "real_robot" if mode == 'r' else "simulate"
        }
        
        # MQTT settings
        print("\n[2/5] MQTT Gripper Control")
        use_mqtt = input("  Enable MQTT gripper control? [Y/n]: ").strip().lower()
        
        config['mqtt'] = {'enabled': use_mqtt != 'n'}
        
        if config['mqtt']['enabled']:
            broker = input("  MQTT broker IP [192.168.1.114]: ").strip()
            config['mqtt']['broker_ip'] = broker or "192.168.1.114"
            config['mqtt']['broker_port'] = 1883
            config['mqtt']['topics'] = {
                'command': "actuador/on_off",
                'state_feedback': "actuador/state"
            }
            config['mqtt']['states'] = {
                'inflated': "inflado",
                'deflated': "desinflado"
            }
            config['mqtt']['confirmation'] = {
                'enabled': True,
                'timeout_seconds': 10.0,
                'allow_override': True,
                'override_key': 'c'
            }
        
        # PnP data source
        print("\n[3/5] PnP Data Source")
        print("  (a) API - Live capture from camera")
        print("  (j) JSON - Read from saved file")
        source = input("  Select source [j]: ").strip().lower()
        
        config['pnp_data'] = {
            'source_mode': "api" if source == 'a' else "json"
        }
        
        if config['pnp_data']['source_mode'] == 'json':
            json_file = input("  JSON file path [pickafresa_vision/captures/20251104_161710_data.json]: ").strip()
            config['pnp_data']['json'] = {
                'default_file': json_file or "pickafresa_vision/captures/20251104_161710_data.json",
                'min_confidence': 0.5
            }
        
        # Movement speed
        print("\n[4/5] Movement Speed")
        print("  (t) Turtle - Very slow")
        print("  (s) Slow - Recommended")
        print("  (n) Normal - Default speed")
        print("  (c) Custom - Specify values")
        speed = input("  Select speed profile [s]: ").strip().lower()
        
        speed_map = {
            't': 'turtle',
            's': 'slow',
            'n': 'normal',
            'c': 'custom'
        }
        
        config['movement'] = {
            'default_profile': speed_map.get(speed, 'slow')
        }
        
        # Transforms
        print("\n[5/5] Camera Transform")
        print("  Using default: [20.0, -58.0, 0.0]mm | Rot[-10.0, 0.0, 0.0]deg")
        custom_transform = input("  Use custom transform? [y/N]: ").strip().lower()
        
        if custom_transform == 'y':
            print("  Enter translation (mm): x, y, z")
            tx = float(input("    x: ").strip() or "20.0")
            ty = float(input("    y: ").strip() or "-58.0")
            tz = float(input("    z: ").strip() or "0.0")
            print("  Enter rotation (deg): u, v, w")
            ru = float(input("    u: ").strip() or "-10.0")
            rv = float(input("    v: ").strip() or "0.0")
            rw = float(input("    w: ").strip() or "0.0")
            
            config['transforms'] = {
                'camera_tcp': {
                    'translation_mm': [tx, ty, tz],
                    'rotation_deg': [ru, rv, rw]
                }
            }
        else:
            config['transforms'] = {
                'camera_tcp': {
                    'translation_mm': [20.0, -58.0, 0.0],
                    'rotation_deg': [-10.0, 0.0, 0.0]
                }
            }
        
        config['transforms']['pick_offset'] = {
            'prepick_z_mm': -100.0,  # Negative for harvesting from below
            'pick_z_mm': 0.0
        }
        
        # Additional defaults
        config['logging'] = {
            'log_directory': "pickafresa_robot/logs",
            'log_filename_prefix': "robot_pnp",
            'log_level': "INFO",
            'console_level': "INFO"
        }
        
        config['safety'] = {
            'confirm_before_movement': True,
            'confirm_before_gripper': True
        }
        
        config['visualization'] = {
            'create_fruit_frames': True,
            'highlight_target': True
        }
        
        self.config = config
        
        print("\n[OK] Configuration created interactively")
        return True
    
    def _setup_logger(self) -> None:
        """Initialize logger from configuration."""
        log_config = self.config.get('logging', {})
        
        log_dir = REPO_ROOT / log_config.get('log_directory', 'pickafresa_robot/logs')
        log_prefix = log_config.get('log_filename_prefix', 'robot_pnp')
        console_level = log_config.get('console_level', 'INFO')
        file_level = log_config.get('log_level', 'DEBUG')
        use_timestamp = log_config.get('use_timestamp_in_filename', False)
        overwrite_log = log_config.get('overwrite_on_start', True)
        
        self.logger = create_logger(
            node_name="robot_pnp_cli",
            log_dir=log_dir,
            log_prefix=log_prefix,
            console_level=console_level,
            file_level=file_level,
            use_timestamp=use_timestamp,
            overwrite_log=overwrite_log
        )
    
    def _initialize_components(self) -> bool:
        """Initialize all components (RoboDK, MQTT, PnP handler, Vision Service)."""
        self.logger.info("Initializing components...")
        
        # Initialize RoboDK
        if not self._init_robodk():
            return False
        
        # Initialize MQTT (optional)
        if not self._init_mqtt():
            return False
        
        # Initialize PnP handler
        if not self._init_pnp_handler():
            return False
        
        # Initialize Vision Service (optional, but required in auto mode)
        if not self._init_vision_service():
            return False
        
        self.logger.info("[OK] All components initialized successfully")
        return True
    
    def _init_robodk(self) -> bool:
        """Initialize RoboDK manager."""
        robodk_config = self.config.get('robodk', {})
        
        station_file = REPO_ROOT / robodk_config.get('station_file', '')
        robot_model = robodk_config.get('robot_model', 'UR3e')
        run_mode = robodk_config.get('run_mode', 'simulate')
        
        self.logger.info(f"Initializing RoboDK (mode: {run_mode})...")
        
        try:
            self.robodk_manager = RoboDKManager(
                station_file=station_file,
                robot_model=robot_model,
                run_mode=run_mode,
                logger=self.logger
            )
            
            if not self.robodk_manager.connect():
                return False
            
            if not self.robodk_manager.select_robot():
                return False
            
            # Set speed
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
            
            # Discover targets
            self.robodk_manager.discover_targets()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to initialize RoboDK: {e}")
            return False
    
    def _init_mqtt(self) -> bool:
        """Initialize MQTT gripper controller (optional)."""
        mqtt_config = self.config.get('mqtt', {})
        
        if not mqtt_config.get('enabled', False):
            self.logger.info("MQTT disabled in configuration, skipping...")
            return True
        
        try:
            self.logger.info("Initializing MQTT gripper controller...")
            
            self.mqtt_controller = MQTTGripperController(
                broker_ip=mqtt_config.get('broker_ip', '192.168.1.114'),
                broker_port=mqtt_config.get('broker_port', 1883),
                command_topic=mqtt_config.get('topics', {}).get('command', 'actuador/on_off'),
                state_topic=mqtt_config.get('topics', {}).get('state_feedback', 'actuador/state'),
                inflated_state=mqtt_config.get('states', {}).get('inflated', 'inflado'),
                deflated_state=mqtt_config.get('states', {}).get('deflated', 'desinflado'),
                logger=self.logger
            )
            
            if not self.mqtt_controller.connect():
                self.logger.warn("Failed to connect to MQTT broker. Continuing without MQTT...")
                self.mqtt_controller = None
            
            return True
        
        except Exception as e:
            self.logger.error(f"MQTT initialization error: {e}")
            self.logger.warn("Continuing without MQTT...")
            self.mqtt_controller = None
            return True
    
    def _init_pnp_handler(self) -> bool:
        """Initialize PnP data handler."""
        try:
            self.logger.info("Initializing PnP data handler...")
            
            # Get camera TCP offset from flange
            camera_tcp_config = self.config.get('transforms', {}).get('camera_tcp', {})
            T_flange_cameraTCP = create_transform_matrix(
                translation_mm=camera_tcp_config.get('translation_mm', [-11.080, -53.400, 24.757]),
                rotation_deg=camera_tcp_config.get('rotation_deg', [0.0, 0.0, 0.0])
            )
            
            # Get gripper TCP offset from flange
            gripper_tcp_config = self.config.get('transforms', {}).get('gripper_tcp', {})
            T_flange_gripperTCP = create_transform_matrix(
                translation_mm=gripper_tcp_config.get('translation_mm', [0.0, 0.0, 77.902]),
                rotation_deg=gripper_tcp_config.get('rotation_deg', [0.0, 0.0, 0.0])
            )
            
            self.pnp_handler = PnPDataHandler(
                T_flange_cameraTCP=T_flange_cameraTCP,
                T_flange_gripperTCP=T_flange_gripperTCP,
                logger=self.logger
            )
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to initialize PnP handler: {e}")
            return False
    
    def _init_vision_service(self) -> bool:
        """
        Initialize vision service client.
        
        Vision service is optional but recommended. In auto mode, it's required.
        If not available, falls back to direct camera capture (if implemented).
        """
        vision_config = self.config.get('vision_service', {})
        pnp_config = self.config.get('pnp_data', {})
        
        # Check if we should try to use vision service
        # Required in auto mode with 'vision' source
        source = pnp_config.get('source', 'file')
        is_auto_mode = source == 'vision'
        
        # Check if vision service is enabled in config
        vision_enabled = vision_config.get('enabled', True)
        
        if not vision_enabled:
            self.logger.info("Vision service disabled in configuration")
            self.use_vision_service = False
            if is_auto_mode:
                self.logger.error("Cannot use auto mode without vision service!")
                return False
            return True
        
        try:
            self.logger.info("Initializing vision service client...")
            
            host = vision_config.get('host', '127.0.0.1')
            port = vision_config.get('port', 5555)
            timeout = vision_config.get('timeout', 30.0)
            
            self.vision_client = VisionServiceClient(host=host, port=port, timeout=timeout)
            self.vision_client.set_logger(self.logger)
            
            # Try to connect
            try:
                self.vision_client.connect()
                
                # Check service status
                status = self.vision_client.check_status()
                if not status.get('alive', False):
                    raise VisionServiceError("Service not alive")
                if not status.get('ready', False):
                    raise VisionServiceError("Service not ready")
                
                self.logger.info("[OK] Vision service connected and ready")
                self.use_vision_service = True
                return True
            
            except VisionServiceError as e:
                self.logger.warn(f"Vision service not available: {e}")
                
                # If auto mode, this is a failure
                if is_auto_mode:
                    self.logger.error("Auto mode requires vision service to be running!")
                    self.logger.error("Please start vision service with:")
                    self.logger.error("  python pickafresa_vision/vision_nodes/vision_service.py")
                    return False
                
                # Otherwise, fall back to direct capture
                self.logger.info("Will use direct camera capture instead")
                self.use_vision_service = False
                self.vision_client = None
                return True
        
        except Exception as e:
            self.logger.error(f"Vision service initialization error: {e}")
            
            if is_auto_mode:
                return False
            
            self.logger.warn("Continuing without vision service...")
            self.use_vision_service = False
            self.vision_client = None
            return True
    
    def _get_move_type_from_sequence(self, step_name: str, default: str = "joint") -> str:
        """
        Get move_type from sequence configuration for a specific step.
        
        Args:
            step_name: Name of the step to look for (e.g., "move_home", "move_foto")
            default: Default move type if not found in config
        
        Returns:
            Move type string: "linear" or "joint"
        """
        sequence_config = self.config.get('sequence', {})
        steps = sequence_config.get('steps', [])
        
        for step in steps:
            if step.get('name') == step_name:
                return step.get('move_type', default)
        
        return default
    
    def _execute_pnp_sequence(self) -> bool:
        """Execute the main PnP sequence based on configuration mode."""
        self.logger.info("Starting PnP sequence...")
        
        # Check execution mode
        sequence_config = self.config.get('sequence', {})
        execution_mode = sequence_config.get('execution_mode', 'yaml')
        
        if execution_mode == 'yaml':
            # Dynamic YAML-driven sequence
            return self._execute_yaml_sequence()
        elif execution_mode == 'interactive':
            # Interactive step-by-step prompts
            return self._execute_interactive_sequence()
        else:
            # Fallback to legacy hard-coded sequence
            self.logger.warn(f"Unknown execution_mode '{execution_mode}', using legacy sequence")
            return self._execute_legacy_sequence()
    
    def _execute_yaml_sequence(self) -> bool:
        """
        Execute sequence dynamically based on YAML configuration.
        
        This reads the sequence.steps from the YAML and executes each step
        in order, allowing full control of the robot workflow from config.
        """
        self.logger.info("Executing YAML-driven dynamic sequence...")
        
        sequence_config = self.config.get('sequence', {})
        steps = sequence_config.get('steps', [])
        
        if not steps:
            self.logger.error("No sequence steps defined in YAML configuration!")
            return False
        
        self.logger.info(f"Loaded {len(steps)} steps from YAML configuration")
        
        # Execute each step in sequence
        for i, step in enumerate(steps, 1):
            step_name = step.get('name', f'step_{i}')
            step_type = step.get('type', 'unknown')
            description = step.get('description', '')
            
            self.logger.info("=" * 60)
            self.logger.info(f"STEP {i}/{len(steps)}: {step_name}")
            if description:
                self.logger.info(f"Description: {description}")
            self.logger.info("=" * 60)
            
            # Execute step based on type
            success = self._execute_sequence_step(step, i)
            
            if not success:
                self.logger.error(f"Step {i} ({step_name}) failed!")
                return False
            
            self.logger.info(f"[OK] Step {i} ({step_name}) completed")
        
        self.logger.info("=" * 60)
        self.logger.info("YAML SEQUENCE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 60)
        
        return True
    
    def _execute_sequence_step(self, step: Dict[str, Any], step_index: int) -> bool:
        """
        Execute a single sequence step based on its type.
        
        Args:
            step: Step configuration dictionary
            step_index: Step number (for logging)
        
        Returns:
            True if step succeeded, False otherwise
        """
        step_type = step.get('type', 'unknown')
        step_name = step.get('name', f'step_{step_index}')
        
        # Route to appropriate handler based on step type
        if step_type == 'move':
            return self._execute_move_step(step)
        elif step_type == 'capture':
            return self._execute_capture_step(step)
        elif step_type == 'gripper':
            return self._execute_gripper_step(step)
        elif step_type == 'process_fruits':
            return self._execute_process_fruits_step(step)
        elif step_type == 'wait':
            return self._execute_wait_step(step)
        elif step_type == 'log':
            return self._execute_log_step(step)
        else:
            self.logger.error(f"Unknown step type: '{step_type}' for step '{step_name}'")
            return False
    
    def _execute_move_step(self, step: Dict[str, Any]) -> bool:
        """
        Execute a movement step.
        
        Step format:
            type: move
            target: "Home" | "Foto" | "Prepick_plane" | etc.
            move_type: "joint" | "linear"
            confirm: true/false
        """
        target = step.get('target', '')
        move_type = step.get('move_type', 'joint')
        confirm = step.get('confirm', self.config.get('safety', {}).get('confirm_before_movement', True))
        highlight = self.config.get('visualization', {}).get('highlight_target', True)
        
        if not target:
            self.logger.error("Move step missing 'target' parameter")
            return False
        
        self.logger.info(f"Moving to target: {target} (mode: {move_type})")
        
        # Check if collision avoidance is enabled
        collision_config = self.config.get('collision_avoidance', {})
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
                self.logger.error(f"Failed to reach {target}: {message}")
                return False
            if message:
                self.logger.info(f"Movement result: {message}")
            return True
        else:
            return self.robodk_manager.move_to_target(
                target_name=target,
                move_type=move_type,
                confirm=confirm,
                highlight=highlight
            )
    
    def _execute_capture_step(self, step: Dict[str, Any]) -> bool:
        """
        Execute a capture/load PnP data step.
        
        Step format:
            type: capture
            confirm: true/false
        """
        confirm = step.get('confirm', True)
        
        if confirm:
            response = input("\nProceed with capture/load? [Y/n]: ").strip().lower()
            if response == 'n':
                self.logger.warn("User cancelled capture step")
                return False
        
        return self._load_pnp_data()
    
    def _execute_gripper_step(self, step: Dict[str, Any]) -> bool:
        """
        Execute a gripper action step.
        
        Step format:
            type: gripper
            action: "open" | "close"
            confirm: true/false
        """
        action = step.get('action', '').lower()
        confirm = step.get('confirm', self.config.get('safety', {}).get('confirm_before_gripper', True))
        
        if action not in ['open', 'close']:
            self.logger.error(f"Invalid gripper action: '{action}' (must be 'open' or 'close')")
            return False
        
        return self._activate_gripper(action, confirm)
    
    def _execute_process_fruits_step(self, step: Dict[str, Any]) -> bool:
        """
        Execute the fruit processing step (handles multi-berry logic).
        
        Step format:
            type: process_fruits
            confirm: true/false
        """
        confirm = step.get('confirm', True)
        
        if confirm:
            response = input(f"\nProcess {len(self.selected_fruits)} detected fruit(s)? [Y/n]: ").strip().lower()
            if response == 'n':
                self.logger.warn("User cancelled fruit processing")
                return False
        
        return self._process_fruits()
    
    def _execute_wait_step(self, step: Dict[str, Any]) -> bool:
        """
        Execute a wait/pause step.
        
        Step format:
            type: wait
            duration: 2.0  # seconds (optional)
            message: "Waiting for stabilization..."  # optional
        """
        duration = step.get('duration', 0)
        message = step.get('message', '')
        
        if message:
            self.logger.info(message)
        
        if duration > 0:
            self.logger.info(f"Waiting for {duration} seconds...")
            time.sleep(duration)
        else:
            input("\nPress Enter to continue...")
        
        return True
    
    def _execute_log_step(self, step: Dict[str, Any]) -> bool:
        """
        Execute a logging step (display message).
        
        Step format:
            type: log
            message: "Custom log message"
            level: "info" | "warn" | "error"  # optional, default: info
        """
        message = step.get('message', '')
        level = step.get('level', 'info').lower()
        
        if not message:
            return True
        
        if level == 'warn':
            self.logger.warn(message)
        elif level == 'error':
            self.logger.error(message)
        else:
            self.logger.info(message)
        
        return True
    
    def _execute_interactive_sequence(self) -> bool:
        """
        Execute sequence with interactive step-by-step prompts.
        
        User is prompted for each action before execution.
        """
        self.logger.info("Starting INTERACTIVE sequence mode...")
        self.logger.warn("Interactive mode not fully implemented - falling back to legacy")
        return self._execute_legacy_sequence()
    
    def _execute_legacy_sequence(self) -> bool:
        """
        Execute the legacy hard-coded sequence.
        
        This is the original implementation kept for backwards compatibility.
        Sequence: Home → Foto → Capture/Load → Process Fruits → Home
        """
        self.logger.info("Executing legacy hard-coded sequence...")
        
        # Step 1: Move to Home
        if not self._move_home():
            return False
        
        # Step 2: Move to Foto position
        if not self._move_foto():
            return False
        
        # Step 3: Capture or load PnP data
        if not self._load_pnp_data():
            return False
        
        # Step 4: Process fruits (possibly multiple)
        if not self._process_fruits():
            return False
        
        # Step 5: Return home
        if not self._move_home():
            return False
        
        return True
    
    def _move_home(self) -> bool:
        """Move to home position."""
        self.logger.info("Moving to HOME position...")
        
        # Check if collision avoidance is enabled
        collision_config = self.config.get('collision_avoidance', {})
        use_collision_avoidance = collision_config.get('enabled', True)
        
        confirm = self.config.get('safety', {}).get('confirm_before_movement', True)
        highlight = self.config.get('visualization', {}).get('highlight_target', True)
        
        # Get move_type from sequence configuration
        move_type = self._get_move_type_from_sequence("move_home", default="joint")
        
        if use_collision_avoidance:
            success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                target_name="Home",
                move_type=move_type,
                confirm=confirm,
                highlight=highlight
            )
            if not success:
                self.logger.error(f"Failed to reach Home: {message}")
                return False
            if message:
                self.logger.info(f"Home movement: {message}")
            return True
        else:
            return self.robodk_manager.move_to_target(
                target_name="Home",
                move_type=move_type,
                confirm=confirm,
                highlight=highlight
            )
    
    def _move_foto(self) -> bool:
        """Move to foto/camera position."""
        self.logger.info("Moving to FOTO (camera) position...")
        
        # Check if collision avoidance is enabled
        collision_config = self.config.get('collision_avoidance', {})
        use_collision_avoidance = collision_config.get('enabled', True)
        
        confirm = self.config.get('safety', {}).get('confirm_before_movement', True)
        highlight = self.config.get('visualization', {}).get('highlight_target', True)
        
        # Get move_type from sequence configuration
        move_type = self._get_move_type_from_sequence("move_foto", default="joint")
        
        if use_collision_avoidance:
            success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                target_name="Foto",
                move_type=move_type,
                confirm=confirm,
                highlight=highlight
            )
            if not success:
                self.logger.error(f"Failed to reach Foto: {message}")
                return False
            if message:
                self.logger.info(f"Foto movement: {message}")
            return True
        else:
            return self.robodk_manager.move_to_target(
                target_name="Foto",
                move_type=move_type,
                confirm=confirm,
                highlight=highlight
            )
    
    def _load_pnp_data(self) -> bool:
        """Load PnP data from vision service, API, or JSON."""
        pnp_config = self.config.get('pnp_data', {})
        source_mode = pnp_config.get('source_mode', 'json')
        
        self.logger.info(f"Loading PnP data (source: {source_mode})...")
        
        if source_mode == 'vision':
            return self._load_pnp_from_vision_service()
        elif source_mode == 'api':
            return self._load_pnp_from_api()
        else:
            return self._load_pnp_from_json()
    
    def _load_pnp_from_vision_service(self) -> bool:
        """Load PnP data from vision service via IPC."""
        if not self.use_vision_service or self.vision_client is None:
            self.logger.error("Vision service not available!")
            return False
        
        try:
            self.logger.info("Requesting capture from vision service...")
            
            # Get vision config for multi-frame settings
            vision_config = self.config.get('vision_service', {})
            multi_frame = vision_config.get('multi_frame_enabled', True)
            num_frames = vision_config.get('num_frames', 10)
            
            # Request capture
            response = self.vision_client.request_capture(
                multi_frame=multi_frame,
                num_frames=num_frames
            )
            
            if not response.get('success', False):
                error = response.get('error', 'Unknown error')
                self.logger.error(f"Vision service capture failed: {error}")
                return False
            
            # Parse detections from response
            detections_data = response.get('detections', [])
            if not detections_data:
                self.logger.error("No detections returned from vision service")
                return False
            
            # Convert to FruitDetection objects
            detections = []
            for det_data in detections_data:
                # Extract data
                class_name = det_data.get('class_name', 'unknown')
                confidence = det_data.get('confidence', 0.0)
                bbox = det_data.get('bbox', [0, 0, 0, 0])
                T_cam_fruit = np.array(det_data.get('T_cam_fruit', np.eye(4).tolist()))
                T_base_fruit = np.array(det_data.get('T_base_fruit', np.eye(4).tolist()))
                
                # Create FruitDetection object
                fruit = FruitDetection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    T_cam_fruit=T_cam_fruit,
                    T_base_fruit=T_base_fruit
                )
                detections.append(fruit)
            
            self.logger.info(f"[OK] Received {len(detections)} detection(s) from vision service")
            
            # Filter by confidence if configured
            min_conf = self.config.get('pnp_data', {}).get('json', {}).get('min_confidence', 0.5)
            detections = [d for d in detections if d.confidence >= min_conf]
            
            if not detections:
                self.logger.error(f"No detections above confidence threshold {min_conf}")
                return False
            
            # Let user select detection if multiple
            if len(detections) > 1:
                selected = self.pnp_handler.select_detection_interactive(detections)
                if selected is None:
                    return False
                self.selected_fruits = [selected]
            else:
                self.selected_fruits = detections
            
            self.logger.info(f"[OK] Loaded {len(self.selected_fruits)} fruit detection(s)")
            return True
        
        except VisionServiceError as e:
            self.logger.error(f"Vision service error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to get detections from vision service: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _load_pnp_from_api(self) -> bool:
        """Load PnP data from live API."""
        self.logger.info("Capturing live PnP data from camera...")
        
        api_config = self.config.get('pnp_data', {}).get('api', {})
        
        objd_config = REPO_ROOT / api_config.get('objd_config_path', 'pickafresa_vision/configs/objd_config.yaml')
        pnp_config = REPO_ROOT / api_config.get('vision_config_path', 'pickafresa_vision/configs/pnp_calc_config.yaml')
        rs_config_path = api_config.get('realsense_config_path')
        rs_config = REPO_ROOT / rs_config_path if rs_config_path else None
        
        detections = self.pnp_handler.call_api_live(
            objd_config_path=objd_config,
            pnp_config_path=pnp_config,
            realsense_config_path=rs_config,
            min_confidence=0.5
        )
        
        if not detections:
            self.logger.error("No valid detections from API")
            return False
        
        self.selected_fruits = detections
        self.logger.info(f"[OK] Loaded {len(detections)} fruit detections from API")
        
        return True
    
    def _load_pnp_from_json(self) -> bool:
        """Load PnP data from JSON file."""
        json_config = self.config.get('pnp_data', {}).get('json', {})
        
        # Get JSON file path
        json_file_str = json_config.get('default_file', 'pickafresa_vision/captures/20251104_161710_data.json')
        json_file = REPO_ROOT / json_file_str
        
        # Prompt user to confirm or select different file
        print(f"\nDefault JSON file: {json_file}")
        response = input("Use this file? [Y/n/b(rowse)]: ").strip().lower()
        
        if response == 'n':
            file_input = input("Enter JSON file path: ").strip()
            if file_input:
                json_file = Path(file_input)
                if not json_file.is_absolute():
                    json_file = REPO_ROOT / file_input
        elif response == 'b':
            # List available JSON files
            search_dir = REPO_ROOT / json_config.get('search_directory', 'pickafresa_vision/captures')
            json_files = sorted(search_dir.glob("*_data.json"))
            
            if json_files:
                print("\nAvailable JSON files:")
                for i, f in enumerate(json_files, 1):
                    print(f"  [{i}] {f.name}")
                
                choice = input(f"Select file [1-{len(json_files)}]: ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(json_files):
                        json_file = json_files[idx]
                except:
                    self.logger.warn("Invalid selection, using default")
        
        # Load detections
        min_conf = json_config.get('min_confidence', 0.5)
        preferred_class = json_config.get('prefer_class', 'ripe')
        
        detections = self.pnp_handler.load_json_file(
            json_path=json_file,
            min_confidence=min_conf,
            preferred_class=preferred_class
        )
        
        if not detections:
            self.logger.error("No valid detections in JSON file")
            return False
        
        # Let user select detection if multiple
        if len(detections) > 1:
            selected = self.pnp_handler.select_detection_interactive(detections)
            if selected is None:
                return False
            self.selected_fruits = [selected]
        else:
            self.selected_fruits = detections
        
        self.logger.info(f"[OK] Loaded {len(self.selected_fruits)} fruit detection(s) from JSON")
        
        return True
    
    def _process_fruits(self) -> bool:
        """
        Process multiple fruits with advanced multi-berry picking logic.
        
        Features:
        - Sorting by confidence, distance, or position
        - Retry logic for failed picks
        - Semi-auto (with confirmations) or full-auto modes
        - Safe home return between picks (optional)
        - Abort on failure option
        """
        if not self.selected_fruits:
            self.logger.error("No fruits to process")
            return False
        
        # Get multi-berry configuration
        multi_berry_config = self.config.get('multi_berry', {})
        mode = multi_berry_config.get('mode', 'semi_auto')
        max_berries = multi_berry_config.get('max_berries_per_run', 10)
        sort_by = multi_berry_config.get('sort_by', 'confidence')
        confirm_between = multi_berry_config.get('confirm_between_picks', True)
        return_home_between = multi_berry_config.get('return_home_between_picks', False)
        abort_on_failure = multi_berry_config.get('abort_on_failure', False)
        retry_on_failure = multi_berry_config.get('retry_on_failure', True)
        max_retries = multi_berry_config.get('max_retries', 5)
        
        self.logger.info("=" * 60)
        self.logger.info("MULTI-BERRY PICKING MODE")
        self.logger.info("=" * 60)
        self.logger.info(f"Mode: {mode}")
        self.logger.info(f"Total detections: {len(self.selected_fruits)}")
        self.logger.info(f"Max berries per run: {max_berries}")
        self.logger.info(f"Sort by: {sort_by}")
        self.logger.info(f"Retry on failure: {retry_on_failure} (max {max_retries})")
        self.logger.info("=" * 60)
        
        # Sort fruits based on configuration
        fruits_to_process = self._sort_fruits(self.selected_fruits, sort_by)
        
        # Limit to max berries
        if len(fruits_to_process) > max_berries:
            self.logger.info(f"Limiting to first {max_berries} berries")
            fruits_to_process = fruits_to_process[:max_berries]
        
        # Track statistics
        total_attempts = 0
        successful_picks = 0
        failed_picks = 0
        
        # Process each fruit
        for i, fruit in enumerate(fruits_to_process, 1):
            self.logger.info("=" * 60)
            self.logger.info(f"BERRY {i}/{len(fruits_to_process)}")
            self.logger.info(f"Class: {fruit.class_name}, Confidence: {fruit.confidence:.2f}")
            self.logger.info("=" * 60)
            
            # Confirmation between picks (if enabled)
            if confirm_between and mode == 'semi_auto':
                response = input(f"\nProceed with berry {i}? [Y/n/q(uit)]: ").strip().lower()
                if response == 'n':
                    self.logger.info("Skipping berry...")
                    continue
                elif response == 'q':
                    self.logger.info("User requested quit")
                    break
            
            # Attempt to pick this berry (with retry logic)
            success = False
            retry_count = 0
            
            while not success and retry_count <= max_retries:
                if retry_count > 0:
                    self.logger.info(f"Retry attempt {retry_count}/{max_retries}")
                
                total_attempts += 1
                success = self._process_single_fruit(fruit, index=i)
                
                if success:
                    successful_picks += 1
                    self.logger.info(f"[OK] Berry {i} picked successfully")
                    break
                else:
                    retry_count += 1
                    failed_picks += 1
                    
                    if retry_on_failure and retry_count <= max_retries:
                        self.logger.warn(f"Berry {i} failed, retrying...")
                        
                        # Return to home position before retry
                        if not self._move_home():
                            self.logger.error("Failed to return home for retry")
                            break
                        
                        # Return to foto position for retry
                        if not self._move_foto():
                            self.logger.error("Failed to return to foto for retry")
                            break
                    else:
                        self.logger.error(f"Berry {i} failed after {retry_count} attempts")
                        
                        if abort_on_failure:
                            self.logger.error("Abort on failure enabled, stopping...")
                            return False
                        
                        # Ask user in semi-auto mode
                        if mode == 'semi_auto':
                            response = input("\nContinue with next berry? [Y/n]: ").strip().lower()
                            if response == 'n':
                                self.logger.info("User chose to stop")
                                return False
                        
                        break
            
            # Return home between picks if configured
            if return_home_between and success and i < len(fruits_to_process):
                self.logger.info("Returning home between picks...")
                if not self._move_home():
                    self.logger.error("Failed to return home")
                    if abort_on_failure:
                        return False
                
                # Return to foto for next berry
                if not self._move_foto():
                    self.logger.error("Failed to return to foto")
                    if abort_on_failure:
                        return False
        
        # Print summary
        self.logger.info("=" * 60)
        self.logger.info("MULTI-BERRY PICKING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total attempts: {total_attempts}")
        self.logger.info(f"Successful picks: {successful_picks}")
        self.logger.info(f"Failed picks: {failed_picks}")
        self.logger.info(f"Success rate: {100.0 * successful_picks / total_attempts if total_attempts > 0 else 0:.1f}%")
        self.logger.info("=" * 60)
        
        return successful_picks > 0
    
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
            self.logger.warn(f"Unknown sort criterion '{sort_by}', using original order")
            return fruits
    
    def _process_single_fruit(self, fruit: FruitDetection, index: int = 1) -> bool:
        """
        Process a single fruit (full pick and place cycle).
        
        This function handles the transformation, target creation, and execution
        of the per-berry sequence defined in YAML configuration.
        
        Args:
            fruit: FruitDetection object with pose information
            index: Berry index for multi-berry runs (1-based)
        
        Returns:
            True if successful, False otherwise
        """
        # Convert index to letter for target naming (1->A, 2->B, etc.)
        berry_letter = chr(64 + index)  # 65 = 'A', 66 = 'B', etc.
        
        self.logger.info(f"Starting pick-and-place cycle for fruit #{index} (Berry {berry_letter})")
        
        # CLEANUP DYNAMIC TARGETS at start of each berry cycle
        # This prevents accumulation of targets from previous berries
        fixed_targets = self.config.get('robodk', {}).get('fixed_targets', ['Home', 'Foto', 'Prepick_plane'])
        cleaned_count = self.robodk_manager.cleanup_dynamic_targets(fixed_targets)
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} targets/frames from previous berry cycles")
        
        # Get current robot pose (this is the FLANGE pose with active TCP)
        # We need to convert it to gripper TCP for the transformation chain
        T_base_activeTCP = self.robodk_manager.get_tcp_pose()
        if T_base_activeTCP is None:
            self.logger.error("Failed to get TCP pose")
            return False
        
        # CRITICAL: The transformation chain expects T_base_gripperTCP
        # but RoboDK might have Camera TCP active. We need to:
        # 1. Get the flange pose from current active TCP
        # 2. Apply gripper TCP offset to get T_base_gripperTCP
        
        # For now, assume we're at Foto position with gripper TCP active
        # TODO: Make this more robust by checking active TCP or computing from flange
        T_base_gripperTCP = T_base_activeTCP
        
        # Transform fruit to base frame
        fruit = self.pnp_handler.transform_to_base_frame(fruit, T_base_gripperTCP)
        
        if fruit.T_base_fruit is None:
            self.logger.error("Failed to transform fruit to base frame")
            return False
        
        # DEBUG: Log transformation results
        self.logger.info("=" * 60)
        self.logger.info("TRANSFORMATION DEBUG")
        self.logger.info("=" * 60)
        self.logger.info(f"T_base_gripperTCP (Foto gripper TCP) at index [0:3, 3]:")
        self.logger.info(f"  Position: [{T_base_gripperTCP[0,3]:.3f}, {T_base_gripperTCP[1,3]:.3f}, {T_base_gripperTCP[2,3]:.3f}]")
        self.logger.info(f"Fruit position in camera frame (position_cam):")
        self.logger.info(f"  [{fruit.position_cam[0]:.3f}, {fruit.position_cam[1]:.3f}, {fruit.position_cam[2]:.3f}]")
        self.logger.info(f"Fruit position in base frame (position_base):")
        self.logger.info(f"  [{fruit.position_base[0]:.3f}, {fruit.position_base[1]:.3f}, {fruit.position_base[2]:.3f}] m")
        self.logger.info(f"  [{fruit.position_base[0]*1000:.1f}, {fruit.position_base[1]*1000:.1f}, {fruit.position_base[2]*1000:.1f}] mm")
        self.logger.info(f"T_base_fruit full matrix:")
        for i in range(4):
            self.logger.info(f"  [{fruit.T_base_fruit[i,0]:9.6f}, {fruit.T_base_fruit[i,1]:9.6f}, {fruit.T_base_fruit[i,2]:9.6f}, {fruit.T_base_fruit[i,3]:9.6f}]")
        
        # MULTI-STAGE CAPTURE (Optional refinement)
        multi_stage_config = self.config.get('multi_stage_capture', {})
        
        # Stage 2: Alignment capture (center berry in image)
        stage2_enabled = multi_stage_config.get('stage2_alignment', {}).get('enabled', False)
        if stage2_enabled:
            self.logger.info("Executing Stage 2: Alignment capture...")
            fruit_aligned = self._execute_stage2_alignment(fruit, berry_letter)
            if fruit_aligned is not None:
                fruit = fruit_aligned
            else:
                self.logger.warn("Stage 2 failed, continuing with original detection")
        
        # CREATE PICK/PREPICK TARGETS
        if not self._create_berry_targets(fruit, berry_letter):
            return False
        
        # EXECUTE PER-BERRY SEQUENCE
        # Check if we should use YAML-driven or legacy sequence
        sequence_config = self.config.get('sequence', {})
        execution_mode = sequence_config.get('execution_mode', 'yaml')
        
        if execution_mode == 'yaml' and 'per_berry_steps' in sequence_config:
            # Use YAML-driven per-berry sequence
            return self._execute_per_berry_yaml_sequence(fruit, berry_letter)
        else:
            # Use legacy hard-coded per-berry sequence
            return self._execute_legacy_berry_sequence(fruit, berry_letter)
    
    def _create_berry_targets(self, fruit: FruitDetection, berry_letter: str) -> bool:
        """
        Create RoboDK targets for prepick and pick positions.
        
        Args:
            fruit: Fruit detection with base frame transformation
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
            self.logger.debug(f"Applied prepick rotation: {prepick_rotation_deg}")
        
        # Create pick pose
        T_base_pick = fruit.T_base_fruit.copy()
        
        # Apply pick offset (usually [0, 0, 0] for direct grasp)
        pick_offset_mm = pick_offset_config.get('pick_offset_mm', [0.0, 0.0, 0.0])
        pick_offset_m = np.array(pick_offset_mm) / 1000.0
        offset_in_base = T_base_pick[:3, :3] @ pick_offset_m
        T_base_pick[:3, 3] += offset_in_base
        
        # Apply absolute rotation for pick if specified
        if any(abs(r) > 0.01 for r in pick_rotation_deg):
            pick_rotation_rad = np.deg2rad(pick_rotation_deg)
            R_pick, _ = cv2.Rodrigues(pick_rotation_rad)
            T_base_pick[:3, :3] = R_pick.astype(np.float64)
            self.logger.debug(f"Applied pick rotation: {pick_rotation_deg}")
        
        # CRITICAL: Convert from meters to millimeters for RoboDK
        # fruit.T_base_fruit is in meters, but RoboDK expects millimeters
        T_base_prepick_mm = T_base_prepick.copy()
        T_base_prepick_mm[:3, 3] = T_base_prepick[:3, 3] * 1000.0  # Convert to mm
        
        T_base_pick_mm = T_base_pick.copy()
        T_base_pick_mm[:3, 3] = T_base_pick[:3, 3] * 1000.0  # Convert to mm
        
        # Log created poses
        self.logger.info(f"Prepick pose offset: {prepick_offset_mm} mm")
        self.logger.info(f"Prepick position (base frame): [{T_base_prepick[0,3]*1000:.1f}, {T_base_prepick[1,3]*1000:.1f}, {T_base_prepick[2,3]*1000:.1f}] mm")
        self.logger.info(f"Pick position (base frame): [{T_base_pick[0,3]*1000:.1f}, {T_base_pick[1,3]*1000:.1f}, {T_base_pick[2,3]*1000:.1f}] mm")
        
        # Create targets in RoboDK
        self.logger.info("Creating dynamic targets in RoboDK...")
        
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
            self.logger.error("Failed to create targets")
            return False
        
        return True
    
    def _execute_per_berry_yaml_sequence(self, fruit: FruitDetection, berry_letter: str) -> bool:
        """
        Execute per-berry sequence dynamically from YAML configuration.
        
        Args:
            fruit: Fruit detection
            berry_letter: Berry letter identifier (A, B, C, etc.)
        
        Returns:
            True if sequence completed successfully
        """
        self.logger.info("=" * 60)
        self.logger.info(f"EXECUTING YAML PER-BERRY SEQUENCE (Berry {berry_letter})")
        self.logger.info("=" * 60)
        
        sequence_config = self.config.get('sequence', {})
        per_berry_steps = sequence_config.get('per_berry_steps', [])
        
        if not per_berry_steps:
            self.logger.error("No per_berry_steps defined in YAML!")
            return False
        
        self.logger.info(f"Loaded {len(per_berry_steps)} per-berry steps from YAML")
        
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
            
            self.logger.info(f"  [{i}/{len(per_berry_steps)}] {step_name}")
            
            # Execute the step
            success = self._execute_sequence_step(step, i)
            
            if not success:
                self.logger.error(f"Per-berry step {i} ({step_name}) failed!")
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
                        self.logger.info("")
                        self.logger.info("=" * 60)
                        self.logger.info("EXECUTING POST-PICK DETACHMENT SEQUENCE")
                        self.logger.info("=" * 60)
                        
                        # Get pick pose from fruit
                        T_base_pick_used = fruit.T_base_fruit.copy()
                        
                        # Get collision config
                        collision_config = self.config.get('collision_avoidance', {})
                        use_collision_avoidance = collision_config.get('enabled', True)
                        confirm_movement = self.config.get('safety', {}).get('confirm_before_movement', True)
                        
                        # Execute post-pick targets
                        success = self._execute_post_pick_sequence(
                            T_base_pick_used,
                            fruit.T_base_fruit,
                            berry_letter,
                            use_collision_avoidance,
                            confirm_movement
                        )
                        
                        if not success:
                            self.logger.warn("Post-pick sequence failed, continuing with placement...")
                        
                        post_pick_executed = True
        
        self.logger.info(f"[OK] Berry {berry_letter} sequence completed")
        return True
    
    def _execute_legacy_berry_sequence(self, fruit: FruitDetection, berry_letter: str) -> bool:
        """
        Execute the legacy hard-coded per-berry sequence.
        
        This is the original implementation kept for backwards compatibility.
        
        Args:
            fruit: Fruit detection
            berry_letter: Berry letter identifier
        
        Returns:
            True if sequence completed successfully
        """
        self.logger.info("Executing legacy per-berry sequence...")
        
        # Get configuration
        confirm_movement = self.config.get('safety', {}).get('confirm_before_movement', True)
        confirm_gripper = self.config.get('safety', {}).get('confirm_before_gripper', True)
        highlight = self.config.get('visualization', {}).get('highlight_target', True)
        collision_config = self.config.get('collision_avoidance', {})
        use_collision_avoidance = collision_config.get('enabled', True)
        multi_stage_config = self.config.get('multi_stage_capture', {})
        pick_offset_config = self.config.get('transforms', {}).get('pick_offset', {})
        
        # 1. Move to Prepick_plane (fixed intermediate target)
        prepick_plane_target = self.config.get('robodk', {}).get('targets', {}).get('prepick_plane', 'Prepick_plane')
        self.logger.info(f"Moving to fixed prepick plane target: {prepick_plane_target}")
        
        if use_collision_avoidance:
            success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                prepick_plane_target, "joint", confirm_movement, highlight,
                enable_collision_avoidance=True, collision_config=collision_config
            )
            if not success:
                self.logger.error(f"Failed to reach prepick plane: {message}")
                return False
            if message:
                self.logger.info(f"Prepick plane movement: {message}")
        else:
            if not self.robodk_manager.move_to_target(prepick_plane_target, "joint", confirm_movement, highlight):
                return False
        
        # 2. Move to berry-specific prepick position
        self.logger.info(f"Moving to berry-specific prepick: prepick_{berry_letter}")
        
        if use_collision_avoidance:
            success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                f"prepick_{berry_letter}", "linear", confirm_movement, highlight,
                enable_collision_avoidance=True, collision_config=collision_config
            )
            if not success:
                self.logger.error(f"Failed to reach prepick: {message}")
                return False
            if message:
                self.logger.info(f"Prepick movement: {message}")
        else:
            if not self.robodk_manager.move_to_target(f"prepick_{berry_letter}", "linear", confirm_movement, highlight):
                return False
        
        # Stage 3: Prepick refinement capture (optional) - simplified for legacy mode
        pick_target_name = f"pick_{berry_letter}"
        stage3_enabled = multi_stage_config.get('stage3_prepick', {}).get('enabled', False)
        if stage3_enabled:
            self.logger.warn("Stage 3 refinement not fully supported in legacy mode - skipping")
        
        # 3. Move to pick (linear approach)
        self.logger.info(f"Moving to pick position: {pick_target_name}")
        
        if use_collision_avoidance:
            success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                pick_target_name, "linear", confirm_movement, highlight,
                enable_collision_avoidance=True, collision_config=collision_config
            )
            if not success:
                self.logger.error(f"Failed to reach pick: {message}")
                return False
            if message:
                self.logger.info(f"Pick movement: {message}")
        else:
            if not self.robodk_manager.move_to_target(pick_target_name, "linear", confirm_movement, highlight):
                return False
        
        # 4. Close gripper
        if not self._activate_gripper("close", confirm_gripper):
            return False
        
        # 5. Execute post-pick detachment sequence (if enabled)
        post_pick_config = self.config.get('post_pick', {})
        post_pick_enabled = post_pick_config.get('enabled', False)
        
        if post_pick_enabled:
            self.logger.info("Executing post-pick detachment sequence...")
            
            # Use fruit pose as pick pose
            T_base_pick_used = fruit.T_base_fruit.copy()
            
            # Generate and execute post-pick targets
            success = self._execute_post_pick_sequence(
                T_base_pick_used,
                fruit.T_base_fruit,
                berry_letter,
                use_collision_avoidance,
                confirm_movement
            )
            
            if not success:
                self.logger.warn("Post-pick sequence failed, continuing to home...")
        
        # 5. Move directly to home (skip prepick return)
        if not self._move_home():
            return False
        
        # 6. Open gripper
        if not self._activate_gripper("open", confirm_gripper):
            return False
        
        self.logger.info(f"[OK] Completed pick-and-place cycle for Berry {berry_letter}")
        
        return True
    
    def _execute_stage2_alignment(self, fruit: FruitDetection, berry_label: str) -> Optional[FruitDetection]:
        """
        Execute Stage 2: Alignment capture.
        
        Move camera TCP to align with berry position, making camera Z-axis
        collinear with berry Z-axis (pointing toward berry center).
        
        Args:
            fruit: Current fruit detection
            berry_label: Berry label (letter) for target naming
        
        Returns:
            Updated FruitDetection or None if failed
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: ALIGNMENT CAPTURE")
        self.logger.info("=" * 60)
        
        stage2_config = self.config.get('multi_stage_capture', {}).get('stage2_alignment', {})
        
        # Get current GRIPPER TCP pose (in mm from RoboDK) - this is the active TCP
        T_base_gripperTCP_mm = self.robodk_manager.get_tcp_pose()
        if T_base_gripperTCP_mm is None:
            self.logger.error("Failed to get current TCP pose")
            return None
        
        # Convert to meters for calculation
        T_base_gripperTCP = T_base_gripperTCP_mm.copy()
        T_base_gripperTCP[:3, 3] /= 1000.0
        
        # Get transforms from PnP handler
        T_flange_cameraTCP = self.pnp_handler.T_flange_cameraTCP
        T_flange_gripperTCP = self.pnp_handler.T_flange_gripperTCP
        
        # Calculate current camera TCP pose in base frame
        # T_base_cameraTCP = T_base_gripperTCP @ inv(T_flange_gripperTCP) @ T_flange_cameraTCP
        T_base_flange = T_base_gripperTCP @ np.linalg.inv(T_flange_gripperTCP)
        T_base_cameraTCP = T_base_flange @ T_flange_cameraTCP
        
        # Get berry position and frame in base frame (already in meters)
        berry_pos_base = fruit.T_base_fruit[:3, 3]
        berry_rotation_base = fruit.T_base_fruit[:3, :3]
        
        # Get berry Z-axis in base frame (points toward/into fruit, same as camera when aligned)
        berry_z_axis = berry_rotation_base[:, 2]
        
        self.logger.info(f"Current gripper TCP position (base frame): {T_base_gripperTCP[:3, 3]*1000} mm")
        self.logger.info(f"Current camera TCP position (base frame): {T_base_cameraTCP[:3, 3]*1000} mm")
        self.logger.info(f"Berry position (base frame): {berry_pos_base*1000} mm")
        self.logger.info(f"Berry Z-axis (base frame): {berry_z_axis}")
        
        # Get stage2 offset configuration
        stage2_offset_mm = stage2_config.get('stage2_offset_mm', [0.0, 0.0, 0.0])
        stage2_offset_m = np.array(stage2_offset_mm) / 1000.0
        stage2_rotation_deg = stage2_config.get('stage2_rotation_deg', [0.0, 0.0, 0.0])
        
        # =================================================================
        # STAGE 2 ALIGNMENT CALCULATION
        # =================================================================
        # Goal: Position camera TCP so that:
        #   1. Camera X,Y aligns with berry X,Y (no offset in XY plane)
        #   2. Camera is positioned along berry -Z axis at standoff distance
        #   3. Camera Z-axis is parallel to berry Z-axis (both point toward fruit)
        #
        # Only degree of freedom: distance along berry Z-axis (stage2_offset_mm[2])
        # =================================================================
        
        # Determine standoff distance (camera to berry distance along berry Z-axis)
        # stage2_offset_mm[2] sets this distance
        if abs(stage2_offset_mm[2]) > 1.0:  # If meaningful offset specified
            standoff_distance = abs(stage2_offset_mm[2]) / 1000.0  # Convert mm to m
        else:
            # Fallback: use current camera-to-berry distance
            current_distance = np.linalg.norm(berry_pos_base - T_base_cameraTCP[:3, 3])
            standoff_distance = current_distance
        
        # Calculate desired CAMERA TCP position:
        # Position = berry_pos - berry_Z * standoff_distance
        # This places camera back along berry's -Z axis
        desired_camera_pos = berry_pos_base - berry_z_axis * standoff_distance
        
        self.logger.info(f"Standoff distance: {standoff_distance*1000:.1f} mm")
        self.logger.info(f"Desired camera TCP position (base frame): {desired_camera_pos*1000} mm")
        
        # Create desired CAMERA TCP alignment transformation
        T_base_cameraTCP_aligned = T_base_cameraTCP.copy()
        T_base_cameraTCP_aligned[:3, 3] = desired_camera_pos
        
        # Adjust orientation to align camera Z-axis with berry Z-axis
        if stage2_config.get('adjust_orientation', True):
            # Camera Z-axis should be PARALLEL to berry Z-axis (both point toward fruit)
            new_z = berry_z_axis.copy()
            
            # Build orthonormal frame around new Z-axis
            # Try to keep berry X-axis as reference for new X-axis
            berry_x_axis = berry_rotation_base[:, 0]
            
            # Calculate Y-axis: perpendicular to both Z and X
            new_y = np.cross(new_z, berry_x_axis)
            if np.linalg.norm(new_y) < 0.01:  # X and Z are parallel, choose different reference
                # Use base frame Y-axis as reference
                new_y = np.cross(new_z, np.array([0, 1, 0]))
                if np.linalg.norm(new_y) < 0.01:  # Still parallel, use X-axis
                    new_y = np.cross(new_z, np.array([1, 0, 0]))
            new_y = new_y / np.linalg.norm(new_y)
            
            # Recalculate X-axis to ensure orthogonality
            new_x = np.cross(new_y, new_z)
            new_x = new_x / np.linalg.norm(new_x)
            
            # Update rotation matrix for camera TCP
            T_base_cameraTCP_aligned[:3, 0] = new_x
            T_base_cameraTCP_aligned[:3, 1] = new_y
            T_base_cameraTCP_aligned[:3, 2] = new_z
            
            self.logger.info(f"Camera Z-axis aligned with berry Z-axis: {new_z}")
            
            # Apply additional rotational offset if specified
            if any(abs(r) > 0.01 for r in stage2_rotation_deg):
                stage2_rotation_rad = np.deg2rad(stage2_rotation_deg)
                R_offset, _ = cv2.Rodrigues(stage2_rotation_rad)
                T_base_cameraTCP_aligned[:3, :3] = T_base_cameraTCP_aligned[:3, :3] @ R_offset.astype(np.float64)
                self.logger.info(f"Applied stage2 rotation offset: {stage2_rotation_deg}")
            
            self.logger.info("Camera orientation aligned perpendicular to berry frame")
        else:
            self.logger.info("Keeping original camera orientation")
        
        # =================================================================
        # CONVERT CAMERA TCP POSE TO GRIPPER TCP POSE
        # =================================================================
        # We calculated the desired camera TCP pose, but RoboDK controls the gripper TCP
        # Transform: T_base_gripperTCP = T_base_cameraTCP @ inv(T_flange_cameraTCP) @ T_flange_gripperTCP
        #
        # Actually: T_base_gripperTCP = T_base_flange @ T_flange_gripperTCP
        # where:    T_base_flange = T_base_cameraTCP @ inv(T_flange_cameraTCP)
        # =================================================================
        
        T_base_flange_aligned = T_base_cameraTCP_aligned @ np.linalg.inv(T_flange_cameraTCP)
        T_base_gripperTCP_aligned = T_base_flange_aligned @ T_flange_gripperTCP
        
        self.logger.info(f"Calculated aligned camera TCP pose")
        self.logger.info(f"Converting to gripper TCP pose for RoboDK control...")
        
        # Convert to mm for RoboDK
        T_base_gripperTCP_aligned_mm = T_base_gripperTCP_aligned.copy()
        T_base_gripperTCP_aligned_mm[:3, 3] *= 1000.0
        
        self.logger.info(f"Alignment gripper TCP position (base frame): {T_base_gripperTCP_aligned_mm[:3, 3]} mm")
        
        # Create alignment target in RoboDK (using gripper TCP pose)
        target_alignment = self.robodk_manager.create_target_from_pose(
            name=f"alignment_{berry_label}",
            T_base_target=T_base_gripperTCP_aligned_mm,
            create_frame=True,
            color=[0, 255, 255]  # Cyan
        )
        
        if target_alignment is None:
            self.logger.error("Failed to create alignment target")
            return None
        
        # Move to alignment target
        confirm = self.config.get('safety', {}).get('confirm_before_movement', True)
        self.logger.info("Moving to alignment position...")
        
        collision_config = self.config.get('collision_avoidance', {})
        collision_enabled = collision_config.get('enabled', True)
        if collision_enabled:
            success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                f"alignment_{berry_label}", "joint", confirm, True,
                enable_collision_avoidance=True, collision_config=collision_config
            )
            if not success:
                self.logger.error(f"Failed to reach alignment position: {message}")
                return None
        else:
            if not self.robodk_manager.move_to_target(f"alignment_{berry_label}", "joint", confirm, True):
                return None
        
        # Capture new detection at alignment position
        self.logger.info("Capturing at alignment position...")
        
        if self.use_vision_service and self.vision_client:
            # Use vision service
            try:
                vision_config = self.config.get('vision_service', {})
                response = self.vision_client.request_capture(
                    multi_frame=vision_config.get('multi_frame_enabled', True),
                    num_frames=vision_config.get('num_frames', 10)
                )
                
                if not response.get('success', False):
                    self.logger.error("Vision service capture failed")
                    return None
                
                detections_data = response.get('detections', [])
                if not detections_data:
                    self.logger.error("No detections at alignment position")
                    return None
                
                # Find the same berry (closest to previous position)
                best_detection = None
                min_distance = float('inf')
                
                for det_data in detections_data:
                    T_base_fruit_new = np.array(det_data.get('T_base_fruit', np.eye(4).tolist()))
                    distance = np.linalg.norm(T_base_fruit_new[:3, 3] - fruit.T_base_fruit[:3, 3])
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_detection = det_data
                
                if best_detection is None:
                    self.logger.error("Could not find matching berry")
                    return None
                
                # Create updated FruitDetection
                fruit_updated = FruitDetection(
                    class_name=best_detection.get('class_name', 'unknown'),
                    confidence=best_detection.get('confidence', 0.0),
                    bbox=best_detection.get('bbox', [0, 0, 0, 0]),
                    T_cam_fruit=np.array(best_detection.get('T_cam_fruit', np.eye(4).tolist())),
                    T_base_fruit=np.array(best_detection.get('T_base_fruit', np.eye(4).tolist()))
                )
                
                self.logger.info(f"[OK] Stage 2 complete - Updated position: [{fruit_updated.position_base[0]*1000:.1f}, {fruit_updated.position_base[1]*1000:.1f}, {fruit_updated.position_base[2]*1000:.1f}] mm")
                return fruit_updated
                
            except Exception as e:
                self.logger.error(f"Stage 2 capture failed: {e}")
                return None
        else:
            # TODO: Implement direct camera capture fallback
            self.logger.warn("Direct camera capture not implemented, using original detection")
            return fruit
    
    def _execute_stage3_prepick_refinement(
        self,
        fruit: FruitDetection,
        T_base_prepick: np.ndarray,
        berry_label: str
    ) -> Optional[np.ndarray]:
        """
        Execute Stage 3: Prepick refinement capture.
        
        Captures from an offset position relative to berry to refine coordinates
        using geometric PnP without depth constraint.
        
        Args:
            fruit: Current fruit detection
            T_base_prepick: Prepick pose (in meters)
            berry_label: Berry label (letter) for target naming
        
        Returns:
            Updated T_base_pick pose or None if failed
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: PREPICK REFINEMENT CAPTURE")
        self.logger.info("=" * 60)
        
        stage3_config = self.config.get('multi_stage_capture', {}).get('stage3_prepick', {})
        
        # Get stage3 offset configuration (relative to berry position in berry frame)
        stage3_offset_mm = stage3_config.get('stage3_offset_mm', [0.0, 0.0, 0.0])
        stage3_offset_m = np.array(stage3_offset_mm) / 1000.0
        stage3_rotation_deg = stage3_config.get('stage3_rotation_deg', [0.0, 0.0, 0.0])
        
        # Calculate stage3 capture position (relative to berry, not prepick)
        T_base_stage3 = fruit.T_base_fruit.copy()
        
        # Apply translational offset in berry frame
        offset_in_base = fruit.T_base_fruit[:3, :3] @ stage3_offset_m
        T_base_stage3[:3, 3] += offset_in_base
        
        # Apply rotational offset if specified
        if any(abs(r) > 0.01 for r in stage3_rotation_deg):
            stage3_rotation_rad = np.deg2rad(stage3_rotation_deg)
            R_offset, _ = cv2.Rodrigues(stage3_rotation_rad)
            T_base_stage3[:3, :3] = T_base_stage3[:3, :3] @ R_offset.astype(np.float64)
            self.logger.info(f"Applied stage3 rotation offset: {stage3_rotation_deg}")
        
        self.logger.info(f"Stage3 offset applied: {stage3_offset_mm} mm")
        self.logger.info(f"Stage3 position (base frame): {T_base_stage3[:3, 3]*1000} mm")
        
        # Create stage3 target if offset is non-zero
        if np.linalg.norm(stage3_offset_m) > 0.001:
            # Convert to mm for RoboDK
            T_base_stage3_mm = T_base_stage3.copy()
            T_base_stage3_mm[:3, 3] *= 1000.0
            
            # Create target
            target_stage3 = self.robodk_manager.create_target_from_pose(
                name=f"stage3_{berry_label}",
                T_base_target=T_base_stage3_mm,
                create_frame=True,
                color=[255, 255, 0]  # Yellow
            )
            
            if target_stage3 is None:
                self.logger.error("Failed to create stage3 target")
                return None
            
            # Move to stage3 position
            confirm = self.config.get('safety', {}).get('confirm_before_movement', True)
            self.logger.info("Moving to stage3 capture position...")
            
            collision_config = self.config.get('collision_avoidance', {})
            collision_enabled = collision_config.get('enabled', True)
            if collision_enabled:
                success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                    f"stage3_{berry_label}", "linear", confirm, True,
                    enable_collision_avoidance=True, collision_config=collision_config
                )
                if not success:
                    self.logger.error(f"Failed to reach stage3 position: {message}")
                    return None
            else:
                if not self.robodk_manager.move_to_target(f"stage3_{berry_label}", "linear", confirm, True):
                    return None
        else:
            self.logger.info("No stage3 offset configured, capturing from current position (prepick)")
        
        # Capture at stage3 position
        self.logger.info("Capturing at stage3 position...")
        
        if self.use_vision_service and self.vision_client:
            try:
                vision_config = self.config.get('vision_service', {})
                response = self.vision_client.request_capture(
                    multi_frame=vision_config.get('multi_frame_enabled', True),
                    num_frames=5  # Use fewer frames for refinement
                )
                
                if not response.get('success', False):
                    self.logger.warn("Stage 3 capture failed, using original position")
                    return None
                
                detections_data = response.get('detections', [])
                if not detections_data:
                    self.logger.warn("No detections at stage3, using original position")
                    return None
                
                # Find matching berry
                best_detection = None
                min_distance = float('inf')
                
                for det_data in detections_data:
                    T_base_fruit_new = np.array(det_data.get('T_base_fruit', np.eye(4).tolist()))
                    distance = np.linalg.norm(T_base_fruit_new[:3, 3] - fruit.T_base_fruit[:3, 3])
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_detection = det_data
                
                if best_detection is None:
                    self.logger.warn("Could not find matching berry at stage3")
                    return None
                
                # Check if depth is sufficient for reliable refinement
                T_cam_fruit_stage3 = np.array(best_detection.get('T_cam_fruit', np.eye(4).tolist()))
                depth_mm = T_cam_fruit_stage3[2, 3] * 1000.0
                min_depth_mm = stage3_config.get('min_depth_mm', 200.0)
                
                if depth_mm < min_depth_mm:
                    self.logger.warn(f"Depth too small ({depth_mm:.1f}mm < {min_depth_mm:.1f}mm), skipping refinement")
                    return None
                
                # Extract refined position in base frame
                T_base_fruit_refined = np.array(best_detection.get('T_base_fruit', np.eye(4).tolist()))
                
                # Apply refinement only to configured axes
                refine_axes = stage3_config.get('refine_axes', ['x', 'z'])
                
                T_base_pick_refined = fruit.T_base_fruit.copy()
                
                if 'x' in refine_axes:
                    T_base_pick_refined[0, 3] = T_base_fruit_refined[0, 3]
                    self.logger.info(f"Refined X: {fruit.T_base_fruit[0,3]*1000:.1f} → {T_base_fruit_refined[0,3]*1000:.1f} mm")
                
                if 'y' in refine_axes:
                    T_base_pick_refined[1, 3] = T_base_fruit_refined[1, 3]
                    self.logger.info(f"Refined Y: {fruit.T_base_fruit[1,3]*1000:.1f} → {T_base_fruit_refined[1,3]*1000:.1f} mm")
                
                if 'z' in refine_axes:
                    T_base_pick_refined[2, 3] = T_base_fruit_refined[2, 3]
                    self.logger.info(f"Refined Z: {fruit.T_base_fruit[2,3]*1000:.1f} → {T_base_fruit_refined[2,3]*1000:.1f} mm")
                
                self.logger.info("[OK] Stage 3 complete - Position refined")
                return T_base_pick_refined
                
            except Exception as e:
                self.logger.error(f"Stage 3 refinement failed: {e}")
                return None
        else:
            self.logger.warn("Vision service not available, skipping Stage 3")
            return None
    
    def _execute_post_pick_sequence(
        self,
        T_base_pick: np.ndarray,
        T_base_fruit: np.ndarray,
        berry_label: str,
        use_collision_avoidance: bool,
        confirm_movement: bool
    ) -> bool:
        """
        Execute post-pick detachment sequence.
        
        Generates and executes a series of cumulative movements relative to the pick
        position to help detach the berry from the plant.
        
        Args:
            T_base_pick: Pick pose in base frame (meters)
            T_base_fruit: Fruit pose in base frame (meters) for reference frame
            berry_label: Berry label (letter) for target naming
            use_collision_avoidance: Whether to use collision avoidance
            confirm_movement: Whether to confirm movements
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("=" * 60)
        self.logger.info("POST-PICK DETACHMENT SEQUENCE")
        self.logger.info("=" * 60)
        
        post_pick_config = self.config.get('post_pick', {})
        num_targets = post_pick_config.get('num_targets', 2)
        target_configs = post_pick_config.get('targets', [])
        rotation_mode = post_pick_config.get('rotation_mode', 'absolute')  # 'absolute' or 'cumulative'
        
        if not target_configs:
            self.logger.warn("No post-pick targets configured")
            return True
        
        # Limit to configured number of targets
        target_configs = target_configs[:num_targets]
        
        self.logger.info(f"Rotation mode: {rotation_mode}")
        
        # Get collision config
        collision_config = self.config.get('collision_avoidance', {})
        
        # Start from pick position for cumulative offsets
        T_current = T_base_pick.copy()
        
        for i, target_config in enumerate(target_configs, start=1):
            target_name = target_config.get('name', f'post_pick_{i}')
            offset_mm = target_config.get('offset_mm', [0.0, 0.0, 0.0])
            rotation_deg = target_config.get('rotation_deg', [0.0, 0.0, 0.0])
            move_type = target_config.get('move_type', 'linear')
            
            self.logger.info(f"Post-pick target {i}/{len(target_configs)}: {target_name}")
            self.logger.info(f"  Offset: {offset_mm} mm, Rotation: {rotation_deg} deg")
            self.logger.info(f"  Move type: {move_type}")
            
            # Apply translation offset
            # For absolute mode: always use original fruit frame for consistency
            # For cumulative mode: use current frame (rotation affects subsequent offsets)
            offset_m = np.array(offset_mm) / 1000.0
            if rotation_mode == 'absolute':
                # Use original fruit frame for all offsets (consistent with prepick/pick behavior)
                offset_in_base = T_base_fruit[:3, :3] @ offset_m
            else:
                # Use current accumulated frame (rotations affect subsequent offsets)
                offset_in_base = T_current[:3, :3] @ offset_m
            
            T_current[:3, 3] += offset_in_base
            
            # Log position after translation but before rotation
            self.logger.debug(f"Position after translation: [{T_current[0,3]*1000:.1f}, {T_current[1,3]*1000:.1f}, {T_current[2,3]*1000:.1f}] mm")
            
            # Apply rotation
            if any(abs(r) > 0.01 for r in rotation_deg):
                rotation_rad = np.deg2rad(rotation_deg)
                R_offset, _ = cv2.Rodrigues(rotation_rad)
                
                if rotation_mode == 'absolute':
                    # Absolute: Rotation specified in BASE frame, applied cumulatively
                    # The rotation_deg are euler angles in the BASE coordinate system
                    # We compose this with the current orientation to build up the rotation
                    # Result: Each rotation adds to the previous in base frame coordinates
                    T_current[:3, :3] = R_offset.astype(np.float64) @ T_current[:3, :3]
                    self.logger.info(f"  Applied ABSOLUTE rotation (base frame): {rotation_deg}")
                else:
                    # Cumulative: Rotation in current local frame
                    # The rotation happens around the current gripper's local axes
                    T_current[:3, :3] = T_current[:3, :3] @ R_offset.astype(np.float64)
                    self.logger.info(f"  Applied CUMULATIVE rotation (local frame): {rotation_deg}")
            
            # Convert to mm for RoboDK
            T_current_mm = T_current.copy()
            T_current_mm[:3, 3] *= 1000.0
            
            # Create target in RoboDK
            robodk_target_name = f"{target_name}_{berry_label}"
            target = self.robodk_manager.create_target_from_pose(
                name=robodk_target_name,
                T_base_target=T_current_mm,
                create_frame=True,
                color=[255, 128, 0]  # Orange
            )
            
            if target is None:
                self.logger.error(f"Failed to create post-pick target: {robodk_target_name}")
                return False
            
            # Move to post-pick target
            if use_collision_avoidance:
                success, message = self.robodk_manager.move_to_target_with_collision_avoidance(
                    robodk_target_name, move_type, confirm_movement, False,
                    enable_collision_avoidance=True, collision_config=collision_config
                )
                if not success:
                    self.logger.error(f"Failed to reach post-pick target {robodk_target_name}: {message}")
                    return False
                if message:
                    self.logger.info(f"Post-pick movement: {message}")
            else:
                if not self.robodk_manager.move_to_target(robodk_target_name, move_type, confirm_movement, False):
                    self.logger.error(f"Failed to reach post-pick target {robodk_target_name}")
                    return False
            
            self.logger.info(f"[OK] Completed post-pick target {i}/{len(target_configs)}")
        
        self.logger.info("[OK] Post-pick detachment sequence complete")
        return True
    
    def _activate_gripper(self, action: str, confirm: bool = True) -> bool:
        """Activate gripper (open or close)."""
        if self.mqtt_controller is None:
            self.logger.warn("MQTT not available, skipping gripper action...")
            
            if confirm:
                input(f"  [Manual Action Required] {action.upper()} gripper, then press Enter...")
            else:
                time.sleep(1.0)
            
            return True
        
        # User confirmation
        if confirm:
            print(f"\n{'='*60}")
            print(f"Ready to {action.upper()} gripper")
            print(f"{'='*60}")
            
            response = input("Continue? [Y/n]: ").strip().lower()
            if response == 'n':
                self.logger.warn(f"Gripper {action} cancelled by user")
                return False
        
        # Execute action
        mqtt_config = self.config.get('mqtt', {}).get('confirmation', {})
        
        if action.lower() == "close":
            self.mqtt_controller.close_gripper()
            desired_state = self.config.get('mqtt', {}).get('states', {}).get('inflated', 'inflado')
        else:
            self.mqtt_controller.open_gripper()
            desired_state = self.config.get('mqtt', {}).get('states', {}).get('deflated', 'desinflado')
        
        # Wait for confirmation
        if mqtt_config.get('enabled', True):
            timeout = mqtt_config.get('timeout_seconds', 10.0)
            allow_override = mqtt_config.get('allow_override', True)
            override_key = mqtt_config.get('override_key', 'c')
            
            return self.mqtt_controller.wait_for_state(
                desired_state=desired_state,
                timeout=timeout,
                allow_override=allow_override,
                override_key=override_key
            )
        
        return True
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.logger:
            self.logger.info("Cleaning up resources...")
        
        if self.vision_client:
            try:
                self.vision_client.disconnect()
            except:
                pass
        
        if self.mqtt_controller:
            try:
                self.mqtt_controller.disconnect()
            except:
                pass
        
        if self.robodk_manager:
            try:
                self.robodk_manager.cleanup()
            except:
                pass
        
        if self.logger:
            self.logger.info("Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Robot Pick-and-Place Testing Tool"
    )
    parser.add_argument(
        '--config',
        type=str,
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    
    app = RobotPnPCLI(config_path=config_path)
    exit_code = app.run()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
