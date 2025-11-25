"""
Robot PnP System Architecture Documentation
===========================================

This document describes the refactored robot pick-and-place system architecture.

by: Aldrick T, 2025
for Team YEA

## System Overview

The robot PnP system has been refactored into a modular, service-based architecture:

```
+-----------------------------------------------------------------+
|                    ROBOT PnP SYSTEM                             |
|-----------------------------------------------------------------+
|                                                                 |
|  +------------------+         +------------------+            |
|  | robot_pnp_cli    |         | robot_pnp_manager|            |
|  | (testing tool)   |         | (local admin CLI)|            |
|  \--------+---------+         \--------+---------+            |
|           |                             |                       |
|           \-------------+---------------+                       |
|                         |                                       |
|              +----------[DOWN]----------+                           |
|              | robot_pnp_service   |[LEFT]----+                     |
|              | (IPC server:5556)   |     |                     |
|              \----------+----------+     |                     |
|                         |                |                     |
|              +----------[DOWN]----------+     |                     |
|              | robot_pnp_controller|     |                     |
|              | (core logic)        |     |                     |
|              \----------+----------+     |                     |
|                         |                |                     |
|     +-------------------+----------------+---------+          |
|     |  Shared Modules   |                |         |          |
|     |-------------------+----------------+---------+          |
|     | transform_utils   |  config_manager|         |          |
|     | vision_client     |  state_machine |         |          |
|     \-------------------+----------------+---------+          |
|                         |                                      |
|     +-------------------+------------------------+            |
|     |                   |                        |            |
|     [DOWN]                   [DOWN]                        [DOWN]            |
|  RoboDK           Vision Service            MQTT Gripper      |
|  (robot)          (port 5555)               (broker)          |
|                                                                |
|  +------------------+                                         |
|  | robot_pnp_remote |[LEFT]---- MQTT Topics ------------+         |
|  | (MQTT bridge)    |                               |         |
|  \--------+---------+                               |         |
|           |                                         |         |
|           [DOWN]                                         |         |
|     Supabase (optional)               Remote Clients         |
|     (cloud logs)                                             |
\-------------------------------------------------------------+
```

## Components

### 1. Shared Modules (Foundation Layer)

#### transform_utils.py
- 3D coordinate transformation mathematics
- Homogeneous transform matrix operations
- Fruit-to-robot coordinate conversion
- Offset application in various reference frames

**Key Functions:**
- `create_transform_matrix(translation_mm, rotation_deg)` -> 4x4 matrix
- `transform_fruit_to_base(T_base_cameraTCP, T_cam_fruit)` -> T_base_fruit
- `compute_gripper_target_from_fruit(...)` -> T_base_gripper

#### config_manager.py
- YAML configuration with hot-reload capability
- MD5-based change detection
- Nested key access with dot notation
- Path resolution (absolute/relative)
- Callback notification on reload

**Hot-Reloadable Keys:**
- `run_mode`
- `transforms.pick_offset.*`
- `post_pick.*`
- `mqtt.*`
- `vision_service.*`
- `multi_berry.*`

**Cold Keys (require restart):**
- `robodk.*`
- `transforms.camera_tcp`
- `transforms.gripper_tcp`

#### vision_client.py
- Socket IPC client for vision_service (port 5555)
- Detection parsing and pose extraction
- `FruitDetection` class for type-safe handling
- Connection management with retry logic

**Protocol:**
- Request: `{"command": "capture", "multi_frame": true, ...}`
- Response: `{"status": "success", "data": {"detections": [...]}}`

#### state_machine.py
- Thread-safe robot state management
- State transition validation
- History tracking
- Callback registration per state

**States:**
- `IDLE` - Ready for commands
- `INITIALIZING` - Starting up
- `MOVING` - Robot in motion
- `CAPTURING` - Vision data acquisition
- `PICKING` - Pick sequence active
- `ERROR` - Recoverable error
- `EMERGENCY_STOP` - Emergency halt
- `SHUTDOWN` - Shutting down

### 2. Controller Layer

#### robot_pnp_controller.py
**Core pick-and-place logic extracted from robot_pnp_cli**

**Responsibilities:**
- Initialize all subsystems (RoboDK, MQTT, vision)
- Execute pick sequences
- State management
- Error handling
- Graceful shutdown

**Public API:**
```python
controller = RobotPnPController(config, logger)
controller.initialize() -> bool
controller.execute_pick_sequence(berry_index=0) -> bool
controller.get_status() -> dict
controller.shutdown()
```

**Pick Sequence Flow:**
1. Move to Foto position
2. Capture vision data
3. Select berry from detections
4. Transform to base frame
5. Compute prepick/pick/place targets
6. Execute movement sequence
7. Gripper close/open
8. Post-pick detachment (optional)
9. Return home

### 3. Service Layer

#### robot_pnp_service.py
**Always-on IPC server wrapping the controller**

**Protocol (port 5556):**
Request format:
```json
{
  "command": "execute_pick",
  "berry_index": 0
}
```

Response format:
```json
{
  "status": "success",
  "data": {...},
  "error": null
}
```

**Commands:**
- `status` - Get current state and statistics
- `initialize` - Initialize controller
- `execute_pick` - Run pick sequence (param: berry_index)
- `shutdown` - Graceful shutdown
- `reload_config` - Reload hot-reloadable config
- `stats` - Get operation statistics

**Features:**
- Multi-client support (threaded)
- Hot-reload config integration
- Statistics tracking (requests/picks success/failed)
- Signal handling (SIGINT/SIGTERM)
- Auto-cleanup on shutdown

**Usage:**
```bash
python pickafresa_robot/robot_system/robot_pnp_service.py \
  --config pickafresa_robot/configs/robot_pnp_config.yaml \
  --host 127.0.0.1 \
  --port 5556
```

### 4. Interface Layer

#### robot_pnp_manager.py
**Local CLI admin interface for operators**

**Menu Options:**
1. View Status - Check service and controller state
2. View Statistics - Request/pick success rates
3. Execute Pick - Run pick sequence for berry
4. View Logs - Tail service logs (live)
5. Edit Config - Modify hot-reloadable parameters
6. Reload Config - Force config reload
7. Emergency Stop - Halt operations immediately
8. Shutdown Service - Stop robot_pnp_service
9. Quit Manager - Exit (service continues)

**Features:**
- Interactive menu navigation
- Real-time log viewing (tail -f)
- Config editing with default editor
- Priority override (local > remote)

**Usage:**
```bash
python pickafresa_robot/robot_system/robot_pnp_manager.py \
  --config pickafresa_robot/configs/robot_pnp_config.yaml
```

#### robot_pnp_remote.py
**MQTT bridge for remote control with Supabase logging**

**MQTT Topics:**
- `robot/command/execute_pick` - Execute pick (payload: berry_index)
- `robot/command/status` - Request status
- `robot/command/emergency` - Emergency stop
- `robot/status/reply` - Status responses

**Priority Handling:**
- Local CLI commands have priority over remote MQTT
- Remote commands queue if local is active
- Local override flag prevents remote execution

**Supabase Integration (optional):**
- Logs operation events to cloud database
- Tracks: pick_success, pick_failed, emergency_stop
- Table: `robot_operations` (configurable)

**Usage:**
```bash
python pickafresa_robot/robot_system/robot_pnp_remote.py \
  --config pickafresa_robot/configs/robot_pnp_config.yaml
```

**Dependencies:**
```bash
pip install paho-mqtt  # Required
pip install supabase   # Optional (for cloud logging)
```

#### robot_pnp_cli.py
**Interactive testing tool (refactored to use shared modules)**

**Changes:**
- Removed inline VisionServiceClient (~158 lines)
- Uses TransformUtils for all transforms
- Uses ConfigManager for config
- Uses VisionServiceClient from vision_client module
- Uses FruitDetection objects instead of dicts

**Still provides:**
- Interactive testing workflow
- Manual confirmation mode
- Multi-stage capture (Stage 1/2/3)
- Offline testing with JSON files
- Direct keyboard control

### 5. Testing Tools

#### test_refactoring.py
**Validates refactored modules work correctly**

Tests:
- Module imports
- Transform creation
- Config loading
- FruitDetection parsing
- State machine transitions
- CLI instantiation

#### test_service_client.py
**Test client for robot_pnp_service validation**

Tests:
- Status command
- Stats command
- Reload config command
- Unknown command handling
- Initialize command (optional, requires RoboDK)

## Configuration Structure

### robot_pnp_config.yaml

```yaml
# Service configuration
service:
  host: "127.0.0.1"
  port: 5556

# Supabase configuration (optional)
supabase:
  enabled: false
  url: ""
  key: ""
  table_name: "robot_operations"

# Run mode
run_mode: "autonomous"  # or "manual_confirm"

# RoboDK configuration
robodk:
  station_file: "pickafresa_robot/rdk/SETUP Fresas.rdk"
  robot_model: "UR3e"
  simulation_mode: "simulate"

# Transforms
transforms:
  camera_tcp:
    translation_mm: [-11.08, -53.4, 24.757]
    rotation_deg: [0, 0, 0]
  gripper_tcp:
    translation_mm: [0, 0, 57]
    rotation_deg: [0, 0, 0]
  pick_offset:
    prepick:
      offset_mm: [0, 0, -200]  # 200mm back from fruit
    pick:
      offset_mm: [0, 0, 10]    # 10mm into fruit
    place:
      offset_mm: [0, 0, 50]

# Post-pick detachment
post_pick:
  enabled: true
  num_targets: 3

# MQTT
mqtt:
  enabled: true
  broker_address: "192.168.1.100"
  broker_port: 1883

# Vision service
vision_service:
  host: "127.0.0.1"
  port: 5555
  timeout: 30.0
  multi_frame_enabled: true
  num_frames: 10
```

## Usage Workflows

### Workflow 1: Local Testing with CLI
```bash
# Terminal 1: Start vision service
./realsense_venv_sudo pickafresa_vision/vision_nodes/vision_service.py --preview

# Terminal 2: Run testing CLI
python pickafresa_robot/robot_system/robot_pnp_cli.py \
  --config pickafresa_robot/configs/robot_pnp_config.yaml
```

### Workflow 2: Production Service Mode
```bash
# Terminal 1: Start vision service
./realsense_venv_sudo pickafresa_vision/vision_nodes/vision_service.py

# Terminal 2: Start robot service
python pickafresa_robot/robot_system/robot_pnp_service.py \
  --config pickafresa_robot/configs/robot_pnp_config.yaml

# Terminal 3: Use manager for local control
python pickafresa_robot/robot_system/robot_pnp_manager.py \
  --config pickafresa_robot/configs/robot_pnp_config.yaml
```

### Workflow 3: Remote MQTT Control
```bash
# Terminal 1: Start vision service
./realsense_venv_sudo pickafresa_vision/vision_nodes/vision_service.py

# Terminal 2: Start robot service
python pickafresa_robot/robot_system/robot_pnp_service.py

# Terminal 3: Start MQTT bridge
python pickafresa_robot/robot_system/robot_pnp_remote.py

# Remote device: Publish MQTT commands
mosquitto_pub -h 192.168.1.100 -t robot/command/execute_pick -m "0"
```

## Data Flow

### Pick Sequence Data Flow
```
1. Client (CLI/Manager/Remote)
   | IPC request: {"command": "execute_pick", "berry_index": 0}
2. robot_pnp_service
   | controller.execute_pick_sequence(0)
3. robot_pnp_controller
   | vision_client.request_capture()
4. vision_service (port 5555)
   | capture + PnP solve
   [UP] List[FruitDetection]
5. robot_pnp_controller
   | Transform to base frame (TransformUtils)
   | Compute targets (prepick/pick/place)
   | Execute movement (robodk_manager)
   | Gripper control (mqtt_gripper)
   [UP] Success/failure
6. robot_pnp_service
   [UP] IPC response: {"status": "success", "data": {...}}
7. Client
   (Display result, log to Supabase if remote)
```

## Error Handling

### State Machine Errors
- Invalid transitions raise `StateTransitionError`
- Controller catches and transitions to ERROR state
- Service returns error response to client

### Communication Errors
- Vision service timeout: Return empty detection list
- Service connection refused: Client handles with error message
- MQTT broker disconnect: Auto-reconnect with paho-mqtt

### Robot Errors
- RoboDK movement failure: Abort sequence, return to home
- Gripper timeout: Continue anyway (configurable)
- Collision detected: Emergency stop

## Performance Characteristics

### Timing
- Config hot-reload check: ~0.1ms (MD5 hash)
- IPC round-trip: ~2-5ms (local socket)
- Vision capture: ~1-2s (multi-frame) or ~0.3s (single frame)
- Pick sequence: ~15-30s (depends on distances)

### Resource Usage
- Memory: ~150MB per service (Python + RoboDK API)
- CPU: <5% idle, ~20-30% during motion
- Network: ~10KB/s IPC, ~1KB/s MQTT

## Security Considerations

### Network Exposure
- IPC service binds to 127.0.0.1 by default (local only)
- MQTT uses unencrypted connection (LAN only)
- Supabase uses HTTPS with API key

### Recommendations
- Use VPN for remote access (don't expose ports)
- Implement MQTT authentication (username/password)
- Rotate Supabase keys regularly
- Use firewall to restrict access

## Troubleshooting

### Service won't start
- Check port 5556 not already in use: `lsof -i :5556`
- Verify config file exists and is valid YAML
- Check RoboDK is running

### Client can't connect
- Verify service is running
- Check host/port in config matches
- Try `telnet 127.0.0.1 5556` to test connection

### Vision capture fails
- Verify vision_service is running on port 5555
- Check RealSense camera connected
- Try vision CLI first: `./realsense_venv_sudo pickafresa_vision/vision_tools/fruit_pose_cli.py`

### MQTT not receiving commands
- Check broker IP/port correct
- Verify topics match configuration
- Use `mosquitto_sub` to monitor: `mosquitto_sub -h 192.168.1.100 -t robot/#`

### Hot-reload not working
- Only hot-reloadable keys update at runtime
- Cold keys (TCP transforms) require service restart
- Check logs for "Configuration reloaded" message

## Development

### Adding New Commands

1. Add handler in `robot_pnp_controller.py`:
```python
def new_operation(self, param1, param2) -> bool:
    # Implementation
    return True
```

2. Add service command in `robot_pnp_service.py`:
```python
elif command == 'new_command':
    return self._cmd_new_command(request)

def _cmd_new_command(self, request):
    result = self.controller.new_operation(
        param1=request.get('param1'),
        param2=request.get('param2')
    )
    return {'status': 'success' if result else 'error'}
```

3. Add client method (optional):
```python
# In ServiceClient class
def new_command(self, param1, param2):
    return self.send_command('new_command', param1=param1, param2=param2)
```

### Testing Strategy

1. **Unit tests**: Individual modules (transform_utils, config_manager)
2. **Integration tests**: Controller + service (test_service_client.py)
3. **System tests**: Full workflow with RoboDK simulation
4. **Acceptance tests**: Real robot operations

## Migration from Old System

### robot_pnp_cli.py Changes
- [OK] Uses shared modules (transform_utils, config_manager, vision_client)
- [OK] Removed ~170 lines of duplicate code
- [OK] Backward compatible (same functionality)
- [WARNING] FruitDetection objects instead of dicts (minor API change)

### Deprecated Components
- None (all components still functional)

### New Dependencies
```bash
pip install paho-mqtt     # For robot_pnp_remote
pip install supabase      # Optional (cloud logging)
```

## Future Enhancements

### Planned Features
- [ ] Web dashboard for monitoring
- [ ] Multi-robot coordination
- [ ] Vision-guided alignment (Stage 2/3 capture)
- [ ] Predictive maintenance (log analysis)
- [ ] Auto-recovery from errors
- [ ] Real-time telemetry streaming

### API Versioning
- Current: v1.0 (initial refactor)
- Compatible with: pickafresa_vision v1.x

---

**Last Updated:** 2025-11-17  
**Author:** Aldrick T  
**Team:** YEA  
**Repository:** pickafresa_yea
"""

if __name__ == "__main__":
    print(__doc__)
