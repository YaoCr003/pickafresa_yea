# Robot Pick-and-Place Testing Tool

Interactive CLI tool for testing PNP-robot connection, robot movement, and berry grabbing in RoboDK simulation environment.

**Team YEA, 2025**

---

## Overview

This tool provides a comprehensive testing environment for the strawberry picking robot system, integrating:
- **RoboDK** simulation/real robot control
- **MQTT** gripper control
- **Vision system** PnP pose estimation (live API or offline JSON)
- **Safety confirmations** at each step
- **ROS2-style logging**

---

## Features

### Modes & Configuration
- ✅ **Configuration**: YAML file or interactive prompts
- ✅ **Run modes**: Simulation or real robot
- ✅ **MQTT**: Optional gripper control with state confirmation
- ✅ **Data sources**: Live camera API or pre-saved JSON files
- ✅ **Speed profiles**: Turtle, Slow, Normal, or Custom
- ✅ **Multi-berry**: Process multiple fruits in sequence

### Safety Features
- ✅ User confirmation before each movement
- ✅ User confirmation before gripper actions
- ✅ Override capability for MQTT state waiting (press 'c')
- ✅ Emergency stop (ESC key)
- ✅ Collision checking (RoboDK feature)

### Visualization
- ✅ Auto-discover targets from RoboDK station
- ✅ Create reference frames for fruit positions
- ✅ Highlight targets before movement
- ✅ Color-coded frames (green=prepick, red=pick)

### Logging
- ✅ ROS2-style format: `[timestamp] [level] [node]: message`
- ✅ Console and file logging
- ✅ Configurable log levels
- ✅ Timestamped log files

---

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install pyyaml paho-mqtt keyboard numpy opencv-python

# RoboDK Python API
pip install robodk

# Vision system dependencies (if using live API)
pip install pyrealsense2 ultralytics
```

### Repository Setup

```bash
cd pickafresa_yea
```

Ensure the following structure exists:
```
pickafresa_yea/
├── pickafresa_robot/
│   ├── configs/
│   │   └── robot_pnp_config.yaml
│   ├── logs/
│   ├── rdk/
│   │   └── SETUP Fresas.rdk
│   └── robot_testing/
│       ├── robot_pnp_cli.py       # Main CLI
│       ├── ros2_logger.py          # Logger module
│       ├── mqtt_gripper.py         # MQTT controller
│       ├── pnp_handler.py          # PnP data handler
│       └── robodk_manager.py       # RoboDK integration
└── pickafresa_vision/
    ├── captures/                   # JSON files
    └── configs/                    # Vision configs
```

---

## Usage

### Quick Start (YAML Config)

```bash
python pickafresa_robot/robot_testing/robot_pnp_cli.py --config pickafresa_robot/configs/robot_pnp_config.yaml
```

### Interactive Mode (No Config)

```bash
python pickafresa_robot/robot_testing/robot_pnp_cli.py
```

The tool will prompt you for:
1. Configuration source (YAML or interactive)
2. RoboDK settings (station, robot, mode)
3. MQTT settings (enable/disable, broker IP)
4. PnP data source (API or JSON)
5. Movement speed (turtle/slow/normal/custom)
6. Camera transform (default or custom)

### Example Session

```
=======================================================================
               ROBOT PNP TESTING TOOL
                    Team YEA, 2025
=======================================================================

CONFIGURATION SETUP
----------------------------------------------------------------------

Use configuration from 'robot_pnp_config.yaml'? [Y/n]: y
✓ Configuration loaded from: pickafresa_robot/configs/robot_pnp_config.yaml

[2025-11-05 14:32:10.123] [INFO] [robot_pnp_cli]: ==================================================
[2025-11-05 14:32:10.124] [INFO] [robot_pnp_cli]: Robot PnP Testing Tool Started
[2025-11-05 14:32:10.125] [INFO] [robot_pnp_cli]: ==================================================

[2025-11-05 14:32:10.200] [INFO] [robot_pnp_cli]: Initializing components...
[2025-11-05 14:32:10.201] [INFO] [robot_pnp_cli]: Initializing RoboDK (mode: simulate)...
[2025-11-05 14:32:11.050] [INFO] [robot_pnp_cli]: ✓ Connected to RoboDK
[2025-11-05 14:32:11.100] [INFO] [robot_pnp_cli]: ✓ Robot selected: UR3e
[2025-11-05 14:32:11.150] [INFO] [robot_pnp_cli]: ✓ Discovered 5 targets: ['Home', 'Foto', ...]

...

[2025-11-05 14:32:45.678] [INFO] [robot_pnp_cli]: ✓ Completed pick-and-place cycle for fruit #1
[2025-11-05 14:32:45.680] [INFO] [robot_pnp_cli]: ==================================================
[2025-11-05 14:32:45.681] [INFO] [robot_pnp_cli]: ✓ All operations completed successfully
[2025-11-05 14:32:45.682] [INFO] [robot_pnp_cli]: ==================================================
```

---

## Configuration

### YAML Configuration File

The `robot_pnp_config.yaml` file contains all configurable parameters:

#### Key Sections:

**1. RoboDK Settings**
```yaml
robodk:
  station_file: "pickafresa_robot/rdk/SETUP Fresas.rdk"
  robot_model: "UR3e"
  run_mode: "simulate"  # or "real_robot"
  auto_discover_targets: true
```

**2. Coordinate Transforms**
```yaml
transforms:
  camera_tcp:
    translation_mm: [20.0, -58.0, 0.0]  # [x, y, z]
    rotation_deg: [-10.0, 0.0, 0.0]     # [u, v, w] axis-angle
  pick_offset:
    prepick_z_mm: -100.0  # Offset for approach (negative = below fruit for harvesting)
```

**3. MQTT Gripper Control**
```yaml
mqtt:
  enabled: true
  broker_ip: "192.168.1.114"
  broker_port: 1883
  topics:
    command: "actuador/on_off"
    state_feedback: "actuador/state"
  confirmation:
    allow_override: true
    override_key: "c"
```

**4. PnP Data Source**
```yaml
pnp_data:
  source_mode: "json"  # or "api"
  json:
    default_file: "pickafresa_vision/captures/20251104_161710_data.json"
    min_confidence: 0.5
```

**5. Movement Speed**
```yaml
movement:
  speed_profiles:
    turtle: {linear_speed: 20, joint_speed: 10}
    slow: {linear_speed: 50, joint_speed: 30}
    normal: {linear_speed: 100, joint_speed: 60}
  default_profile: "slow"
```

**6. Safety Settings**
```yaml
safety:
  confirm_before_movement: true
  confirm_before_gripper: true
  emergency_stop_key: "esc"
```

---

## Movement Sequence

The tool executes the following sequence:

```
1. HOME         → Move to home position (joint movement)
2. FOTO         → Move to camera position (joint movement)
3. CAPTURE      → Capture PnP data or load from JSON
4. PREPICK      → Move to approach position (joint movement)
5. PICK         → Approach fruit (linear movement)
6. GRIPPER ON   → Close gripper, wait for confirmation
7. PREPICK      → Retract to approach position (linear movement)
8. PLACE/HOME   → Move to place position (joint movement)
9. GRIPPER OFF  → Open gripper, wait for confirmation
10. HOME        → Return to home position
```

### Harvesting Approach (Prepick Position)

For strawberry harvesting, the robot approaches **from below** the fruit:

- **Prepick Position**: Offset **below** the fruit (negative Z offset)
- **Configuration**: `prepick_z_mm: -100.0` (negative = 100mm below fruit)
- **Purpose**: Safe approach position before linear move to pick point
- **Direction**: Moves upward (linearly) from prepick to pick

**Example Configuration:**
```yaml
pick_offset:
  prepick_z_mm: -100.0  # 100mm below fruit (harvesting from below)
  pick_z_mm: 0.0        # No additional offset at pick point
```

To approach from **above** instead (e.g., for top-down picking), use a **positive** value:
```yaml
pick_offset:
  prepick_z_mm: 100.0   # 100mm above fruit (top-down approach)
```

Each step includes:
- ✅ User confirmation prompt (if enabled)
- ✅ Target highlighting in RoboDK
- ✅ Logging of action and status
- ✅ Emergency stop check

---

## Coordinate Transforms

The tool handles the following transform chain:

```
Camera Frame → TCP Frame → Robot Base Frame
```

### Camera-to-TCP Transform
Configured in YAML as:
```yaml
camera_tcp:
  translation_mm: [20.0, -58.0, 0.0]  # Camera position relative to TCP
  rotation_deg: [-10.0, 0.0, 0.0]     # Camera orientation (axis-angle)
```

### Fruit Position Calculation
```python
T_base_fruit = T_base_tcp @ inv(T_cam_tcp) @ T_cam_fruit
```

Where:
- `T_base_tcp`: Current robot TCP pose from RoboDK
- `T_cam_tcp`: Camera-to-TCP transform from config
- `T_cam_fruit`: Fruit pose from vision system

---

## PnP Data Sources

### 1. Live API (Camera Capture)
Uses the `FruitPoseEstimator` API from `pickafresa_vision`:
- Captures RGB+Depth frame from RealSense
- Runs YOLO detection
- Estimates 6DOF poses via PnP
- Returns list of detections with transforms

**Configuration:**
```yaml
pnp_data:
  source_mode: "api"
  api:
    vision_config_path: "pickafresa_vision/configs/pnp_calc_config.yaml"
    objd_config_path: "pickafresa_vision/configs/objd_config.yaml"
```

### 2. Offline JSON Files
Reads pre-saved capture files from `pickafresa_vision/captures/`:
- Loads JSON with detection results
- Filters by confidence and class
- Supports multiple detections with user selection

**Configuration:**
```yaml
pnp_data:
  source_mode: "json"
  json:
    default_file: "pickafresa_vision/captures/20251104_161710_data.json"
    min_confidence: 0.5
    prefer_class: "ripe"
```

**JSON Format:**
```json
{
  "detections": [
    {
      "bbox_cxcywh": [368.52, 294.42, 82.97, 81.40],
      "confidence": 0.82,
      "class_name": "ripe",
      "success": true,
      "T_cam_fruit": [[...], [...], [...], [0, 0, 0, 1]],
      "position_cam": [0.017, 0.016, 0.239]
    }
  ]
}
```

---

## MQTT Gripper Control

### State Confirmation
When MQTT is enabled, the tool:
1. Sends gripper command (`Gripper encendido` / `Gripper apagado`)
2. Waits for state confirmation (`inflado` / `desinflado`)
3. Allows override with keypress (default: 'c')

### Override Functionality
If state confirmation is taking too long:
```
[2025-11-05 14:35:10.123] [INFO] [robot_pnp_cli]: Waiting for gripper state: 'inflado' (timeout: 10s)
[2025-11-05 14:35:10.124] [INFO] [robot_pnp_cli]:   Press 'c' to continue without confirmation

[User presses 'c']

[2025-11-05 14:35:12.456] [WARN] [robot_pnp_cli]: State confirmation overridden by user (key: 'c')
```

### Manual Mode (No MQTT)
If MQTT is disabled or unavailable:
```
[2025-11-05 14:35:10.123] [WARN] [robot_pnp_cli]: MQTT not available, skipping gripper action...
[Manual Action Required] CLOSE gripper, then press Enter...
```

---

## Logging

### Log Files
Located in `pickafresa_robot/logs/`:
```
robot_pnp_20251105_143210.log
```

### Log Format (ROS2-Style)
```
[2025-11-05 14:32:10.123456] [INFO] [robot_pnp_cli]: Robot PnP Testing Tool Started
[2025-11-05 14:32:11.234567] [WARN] [robot_pnp_cli]: MQTT connection failed, continuing without MQTT
[2025-11-05 14:32:12.345678] [ERROR] [robot_pnp_cli]: Target 'invalid_target' not found
```

### Log Levels
- **DEBUG**: Detailed information for diagnostics
- **INFO**: General informational messages
- **WARN**: Warning messages (non-critical issues)
- **ERROR**: Error messages (critical issues)
- **CRITICAL**: Critical errors (program cannot continue)

---

## Troubleshooting

### Issue: RoboDK Connection Failed
```
[ERROR] Failed to connect to RoboDK
```
**Solution:**
- Ensure RoboDK application is running
- Check station file path in config
- Verify RoboDK Python API is installed: `pip install robodk`

### Issue: Station File Not Found
```
[ERROR] Station file not found: pickafresa_robot/rdk/SETUP Fresas.rdk
```
**Solution:**
- Verify station file exists at specified path
- Use absolute path or ensure running from repo root

### Issue: Robot Not Found
```
[ERROR] Robot 'UR3e' not found in station
```
**Solution:**
- Check robot name in RoboDK station
- Update `robot_model` in config to match station

### Issue: MQTT Connection Failed
```
[WARN] Failed to connect to MQTT broker. Continuing without MQTT...
```
**Solution:**
- Check broker IP address and port
- Verify MQTT broker is running: `mosquitto -v`
- Test connection: `mosquitto_pub -h 192.168.1.114 -t test -m "hello"`

### Issue: No Detections in JSON
```
[ERROR] No valid detections in JSON file
```
**Solution:**
- Check JSON file format matches expected structure
- Lower `min_confidence` threshold in config
- Verify JSON file contains successful detections

### Issue: Transform Error
```
[ERROR] Failed to transform fruit to base frame
```
**Solution:**
- Verify camera-to-TCP transform in config
- Check T_cam_fruit is valid in detection data
- Ensure robot TCP pose is accessible

---

## Advanced Usage

### Custom Speed Profile
```yaml
movement:
  speed_profiles:
    custom:
      linear_speed: 75
      joint_speed: 45
      acceleration: 50
  default_profile: "custom"
```

### Multiple Fruits Processing
```yaml
sequence:
  multi_berry:
    enabled: true
    max_berries: 5
    prompt_each: true
```

### Workspace Limits (Safety)
```yaml
safety:
  workspace_limits:
    enabled: true
    x_min: -500.0
    x_max: 500.0
    y_min: -500.0
    y_max: 500.0
    z_min: 0.0
    z_max: 800.0
```

---

## Module Documentation

### `robot_pnp_cli.py`
Main CLI application orchestrating the entire workflow.

### `ros2_logger.py`
ROS2-style logger with colored console output and file logging.

**Usage:**
```python
from pickafresa_robot.robot_testing.ros2_logger import create_logger

logger = create_logger(
    node_name="my_node",
    log_dir=Path("logs"),
    console_level="INFO"
)

logger.info("This is an info message")
logger.warn("This is a warning")
logger.error("This is an error")
```

### `mqtt_gripper.py`
MQTT gripper controller with state confirmation and override.

**Usage:**
```python
from pickafresa_robot.robot_testing.mqtt_gripper import MQTTGripperController

gripper = MQTTGripperController(
    broker_ip="192.168.1.114",
    logger=logger
)

gripper.connect()
gripper.close_gripper()
gripper.wait_for_state("inflado", timeout=10.0, allow_override=True)
gripper.disconnect()
```

### `pnp_handler.py`
PnP data handler for API calls and JSON loading with coordinate transforms.

**Usage:**
```python
from pickafresa_robot.robot_testing.pnp_handler import PnPDataHandler, create_transform_matrix

T_cam_tcp = create_transform_matrix([20, -58, 0], [-10, 0, 0])
handler = PnPDataHandler(T_cam_tcp=T_cam_tcp, logger=logger)

# From JSON
detections = handler.load_json_file(json_path, min_confidence=0.5)

# From API
detections = handler.call_api_live(objd_config, pnp_config)

# Transform to base frame
fruit = handler.transform_to_base_frame(detections[0], T_base_tcp)
```

### `robodk_manager.py`
RoboDK integration for station loading, robot control, and movement.

**Usage:**
```python
from pickafresa_robot.robot_testing.robodk_manager import RoboDKManager

manager = RoboDKManager(
    station_file=Path("rdk/SETUP Fresas.rdk"),
    robot_model="UR3e",
    run_mode="simulate",
    logger=logger
)

manager.connect()
manager.select_robot()
manager.discover_targets()
manager.move_to_target("Home", move_type="joint", confirm=True)
manager.create_target_from_pose("fruit_1", T_base_fruit, create_frame=True)
```

---

## Future Enhancements

- [ ] Support for custom place positions (not just home)
- [ ] Gripper animation in RoboDK
- [ ] Path planning with obstacle avoidance
- [ ] Multi-robot coordination
- [ ] Real-time visualization of fruit positions
- [ ] Automatic hand-eye calibration integration
- [ ] Support for different gripper types
- [ ] Batch processing mode (no confirmations)

---

## License

Team YEA, 2025

---

## Contact

For issues, questions, or contributions, please contact the Team YEA development team.
