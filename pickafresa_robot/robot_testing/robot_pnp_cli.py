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

by: Aldrick T, 2025
for Team YEA
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

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
        
        # Runtime state
        self.use_config = True
        self.selected_fruits: List[FruitDetection] = []
        
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
            
            self.logger.info("="*60)
            self.logger.info("✓ All operations completed successfully")
            self.logger.info("="*60)
            
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
            
            print(f"✓ Configuration loaded from: {self.config_path}")
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
        
        print("\n✓ Configuration created interactively")
        return True
    
    def _setup_logger(self) -> None:
        """Initialize logger from configuration."""
        log_config = self.config.get('logging', {})
        
        log_dir = REPO_ROOT / log_config.get('log_directory', 'pickafresa_robot/logs')
        log_prefix = log_config.get('log_filename_prefix', 'robot_pnp')
        console_level = log_config.get('console_level', 'INFO')
        file_level = log_config.get('log_level', 'DEBUG')
        
        self.logger = create_logger(
            node_name="robot_pnp_cli",
            log_dir=log_dir,
            log_prefix=log_prefix,
            console_level=console_level,
            file_level=file_level
        )
    
    def _initialize_components(self) -> bool:
        """Initialize all components (RoboDK, MQTT, PnP handler)."""
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
        
        self.logger.info("✓ All components initialized successfully")
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
    
    def _execute_pnp_sequence(self) -> bool:
        """Execute the main PnP sequence."""
        self.logger.info("Starting PnP sequence...")
        
        # Sequence: Home → Foto → Capture/Load → Prepick → Pick → Grip → Prepick → Place → Home
        
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
        
        return self.robodk_manager.move_to_target(
            target_name="Home",
            move_type="joint",
            confirm=self.config.get('safety', {}).get('confirm_before_movement', True),
            highlight=self.config.get('visualization', {}).get('highlight_target', True)
        )
    
    def _move_foto(self) -> bool:
        """Move to foto/camera position."""
        self.logger.info("Moving to FOTO (camera) position...")
        
        return self.robodk_manager.move_to_target(
            target_name="Foto",
            move_type="joint",
            confirm=self.config.get('safety', {}).get('confirm_before_movement', True),
            highlight=self.config.get('visualization', {}).get('highlight_target', True)
        )
    
    def _load_pnp_data(self) -> bool:
        """Load PnP data from API or JSON."""
        pnp_config = self.config.get('pnp_data', {})
        source_mode = pnp_config.get('source_mode', 'json')
        
        self.logger.info(f"Loading PnP data (source: {source_mode})...")
        
        if source_mode == 'api':
            return self._load_pnp_from_api()
        else:
            return self._load_pnp_from_json()
    
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
        self.logger.info(f"✓ Loaded {len(detections)} fruit detections from API")
        
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
        
        self.logger.info(f"✓ Loaded {len(self.selected_fruits)} fruit detection(s) from JSON")
        
        return True
    
    def _process_fruits(self) -> bool:
        """Process each selected fruit (pick and place)."""
        if not self.selected_fruits:
            self.logger.error("No fruits to process")
            return False
        
        self.logger.info(f"Processing {len(self.selected_fruits)} fruit(s)...")
        
        for i, fruit in enumerate(self.selected_fruits, 1):
            self.logger.info(f"Processing fruit {i}/{len(self.selected_fruits)}: {fruit.class_name}")
            
            if not self._process_single_fruit(fruit, index=i):
                self.logger.warn(f"Failed to process fruit {i}, skipping...")
                
                # Ask user if they want to continue
                if len(self.selected_fruits) > 1:
                    response = input("\nContinue with next fruit? [Y/n]: ").strip().lower()
                    if response == 'n':
                        return False
        
        return True
    
    def _process_single_fruit(self, fruit: FruitDetection, index: int = 1) -> bool:
        """Process a single fruit (full pick and place cycle)."""
        self.logger.info(f"Starting pick-and-place cycle for fruit #{index}")
        
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
        self.logger.info("=" * 60)
        
        # Create targets
        pick_offset_config = self.config.get('transforms', {}).get('pick_offset', {})
        prepick_offset_z = pick_offset_config.get('prepick_z_mm', 100.0) / 1000.0  # Convert to meters
        
        # Prepick position (offset from fruit - positive=above, negative=below for harvesting)
        T_base_prepick = fruit.T_base_fruit.copy()
        T_base_prepick[2, 3] += prepick_offset_z  # Add Z offset (negative value moves below)
        
        # CRITICAL: Convert from meters to millimeters for RoboDK
        # fruit.T_base_fruit is in meters, but RoboDK expects millimeters
        T_base_prepick_mm = T_base_prepick.copy()
        T_base_prepick_mm[:3, 3] = T_base_prepick[:3, 3] * 1000.0  # Convert to mm
        
        T_base_pick_mm = fruit.T_base_fruit.copy()
        T_base_pick_mm[:3, 3] = fruit.T_base_fruit[:3, 3] * 1000.0  # Convert to mm
        
        # Create targets in RoboDK
        self.logger.info("Creating dynamic targets in RoboDK...")
        
        target_prepick = self.robodk_manager.create_target_from_pose(
            name=f"prepick_{index}",
            T_base_target=T_base_prepick_mm,  # RoboDK expects millimeters
            create_frame=True,
            color=[0, 255, 0]  # Green
        )
        
        target_pick = self.robodk_manager.create_target_from_pose(
            name=f"pick_{index}",
            T_base_target=T_base_pick_mm,  # RoboDK expects millimeters
            create_frame=True,
            color=[255, 0, 0]  # Red
        )
        
        if target_prepick is None or target_pick is None:
            self.logger.error("Failed to create targets")
            return False
        
        # Execute sequence
        confirm_movement = self.config.get('safety', {}).get('confirm_before_movement', True)
        confirm_gripper = self.config.get('safety', {}).get('confirm_before_gripper', True)
        highlight = self.config.get('visualization', {}).get('highlight_target', True)
        
        # 1. Move to prepick
        if not self.robodk_manager.move_to_target(f"prepick_{index}", "joint", confirm_movement, highlight):
            return False
        
        # 2. Move to pick (linear approach)
        if not self.robodk_manager.move_to_target(f"pick_{index}", "linear", confirm_movement, highlight):
            return False
        
        # 3. Close gripper
        if not self._activate_gripper("close", confirm_gripper):
            return False
        
        # 4. Retract to prepick
        if not self.robodk_manager.move_to_target(f"prepick_{index}", "linear", False, False):
            return False
        
        # 5. Move to place position (for now, just home)
        # TODO: Add configurable place position
        if not self._move_home():
            return False
        
        # 6. Open gripper
        if not self._activate_gripper("open", confirm_gripper):
            return False
        
        self.logger.info(f"✓ Completed pick-and-place cycle for fruit #{index}")
        
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
