# MQTT & Supabase Integration - Implementation Summary

**Date**: November 23, 2025  
**Author**: Aldrick T
**Task**: Remote control and robot_pnp monitoring system with MQTT and Supabase

---

## Overview

Successfully implemented MQTT communication and Supabase cloud storage integration for the pickafresa_yea robot system. The implementation provides:

1. **MQTT Bridge** - Real-time robot monitoring and remote control
2. **Supabase Uploader** - Cloud storage for captured images and JSON metadata  
3. **Environment Configuration** - Centralized `.env` file for credentials
4. **Vision Service Integration** - Automatic uploads on capture

---

## Files Created/Modified

### Created Files:
1. **`/.env`** - Global environment variables (Roboflow, Supabase, MQTT credentials)
2. **`/.env.template`** - Template file with setup instructions
3. **`/pickafresa_vision/vision_tools/supabase_uploader.py`** - Supabase upload module
4. **`/pickafresa_robot/robot_system/mqtt_bridge.py`** - MQTT communication bridge

### Modified Files:
1. **`/pickafresa_vision/configs/vision_service_config.yaml`** - Added Supabase config section
2. **`/pickafresa_vision/vision_nodes/vision_service.py`** - Integrated Supabase uploader

---

## MQTT Bridge (`mqtt_bridge.py`)

### Topics Published (robot -> dashboard):

All published topics use simplified array format for maximum compatibility:

- **`robot/log`** (QoS 0) - Format: `[timestamp, level, message]`
  ```json
  ["2025-11-23T12:30:45.123456", "INFO", "Robot moving to FOTO position"]
  ```

- **`robot/status`** (QoS 1) - Format: `[timestamp, status, extra_info]`
  ```json
  ["2025-11-23T12:30:45.123456", "RUNNING", "paused=False"]
  ```

- **`robot/sequence`** (QoS 1) - Format: `[timestamp, step, details]`
  ```json
  ["2025-11-23T12:30:45.123456", "CAPTURING", "Capturing berry data"]
  ```

- **`robot/settings`** (QoS 1) - Format: `[timestamp, type, settings_string]`
  ```json
  ["2025-11-23T12:30:45.123456", "CONFIG", "run_mode=manual, speed=normal"]
  ```

### Topics Subscribed (dashboard -> robot):
- `robot/commands` - Remote commands with JSON payload:
  ```json
  {
    "command": "stop|start|pause|resume|emergency_stop",
    "params": {}
  }
  ```

### Usage Example:
```python
from pickafresa_robot.robot_system.mqtt_bridge import MQTTBridge

# Initialize with command callback
def handle_command(command: str, params: dict):
    if command == "stop":
        # Handle stop command
        pass

bridge = MQTTBridge(config, logger, command_callback=handle_command)
bridge.start()

# Publish updates (payloads are automatically formatted as arrays)
bridge.publish_status("RUNNING")
bridge.publish_log("INFO", "Robot initialized")
bridge.publish_sequence("MOVING_TO_FOTO", "Moving to idle position")
bridge.publish_settings({"run_mode": "autonomous", "speed": "normal"})

# Stop when done
bridge.stop()
```

### Configuration:
Uses existing MQTT settings from `robot_pnp_config.yaml`:
```yaml
mqtt:
  enabled: true
  broker_ip: "192.168.1.114"
  broker_port: 1883
  keepalive: 60
```

---

## Supabase Integration

### Uploader Module (`supabase_uploader.py`)

Handles uploading of images and JSON to Supabase storage + database.

#### Features:
- UUID-based unique filenames
- Synchronous and asynchronous uploads
- Automatic table insertion (images, json_files)
- Error handling and logging

#### Database Schema Required:

**Table: `images`**
```sql
CREATE TABLE images (
  id BIGSERIAL PRIMARY KEY,
  route TEXT NOT NULL,
  timestamp TIMESTAMPTZ DEFAULT NOW()
);
```

**Table: `json_files`**
```sql
CREATE TABLE json_files (
  id BIGSERIAL PRIMARY KEY,
  route TEXT NOT NULL,
  timestamp TIMESTAMPTZ DEFAULT NOW()
);
```

**Storage Bucket**: `pickafresa-captures` (or custom from .env)

#### Usage:
```python
from pickafresa_vision.vision_tools.supabase_uploader import SupabaseUploader

uploader = SupabaseUploader()

# Synchronous upload
results = uploader.upload_capture(
    image_path=Path("captures/20251123_120000_raw.png"),
    json_path=Path("captures/20251123_120000_data.json")
)

# Asynchronous upload (non-blocking)
def callback(success, message, results):
    print(f"Upload {'succeeded' if success else 'failed'}: {message}")

uploader.upload_capture_async(image_path, json_path, callback=callback)
```

### Vision Service Integration

#### Configuration (`vision_service_config.yaml`):
```yaml
capture:
  save_captures:
    enabled: true
    directory: "pickafresa_vision/captures"
    save_rgb: true
    save_json: true
  
  supabase:
    enabled: false            # Enable Supabase uploads
    upload_mode: "async"      # "async" (non-blocking) or "sync" (blocking)
    on_capture: false         # Upload on every capture (if false, only on explicit request)
    upload_rgb: true          # Upload RGB images
    upload_json: true         # Upload JSON metadata
```

#### Behavior:
- When `save_captures.enabled = true`, vision_service saves captures to disk
- If `supabase.enabled = true`, automatically uploads to Supabase cloud
- Upload mode configurable: async (non-blocking) or sync (blocking)
- Only uploads for IPC requests from robot (not for preview frames)

---

## Environment Variables (`.env`)

Located at repository root (`/pickafresa_yea/.env`):

```bash
# Roboflow
ROBOFLOW_API_KEY=your_key_here

# Supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=your_supabase_key
SUPABASE_BUCKET=pickafresa-captures

# MQTT (Optional - overrides config file)
# MQTT_BROKER_IP=192.168.1.114
# MQTT_BROKER_PORT=1883
# MQTT_USERNAME=
# MQTT_PASSWORD=
```

**Important**: The `.env` file is gitignored. Use `.env.template` as reference.

---

## Integration with robot_pnp_cli

### Step 1: Import MQTT Bridge

Add import to `robot_pnp_cli.py`:
```python
from pickafresa_robot.robot_system.mqtt_bridge import MQTTBridge
```

### Step 2: Initialize in `__init__`

Add to `RobotPnPCLI.__init__()`:
```python
# Initialize MQTT bridge
self.mqtt_bridge = None
mqtt_enabled = self.config.get('mqtt', {}).get('enabled', False)
if mqtt_enabled:
    self.mqtt_bridge = MQTTBridge(
        self.config,
        self.logger,
        command_callback=self._handle_mqtt_command
    )
    if self.mqtt_bridge.is_enabled():
        self.mqtt_bridge.start()
        self.logger.info("✓ MQTT bridge started")
    else:
        self.mqtt_bridge = None
```

### Step 3: Add Command Handler

```python
def _handle_mqtt_command(self, command: str, params: Dict[str, Any]):
    """
    Handle MQTT commands from remote dashboard.
    
    Args:
        command: Command name (stop, start, etc.)
        params: Command parameters
    """
    self.logger.info(f"MQTT command received: {command}")
    
    if command == "stop":
        # Set flag to stop continuous operation
        # Note: Implement proper state handling
        self.logger.info("Stop command received via MQTT")
        # TODO: Implement stop logic
    
    elif command == "start":
        self.logger.info("Start command received via MQTT")
        # TODO: Implement start logic
    
    else:
        self.logger.warn(f"Unknown MQTT command: {command}")
```

### Step 4: Publish Status Updates

Add calls throughout robot_pnp_cli.py state transitions:

```python
# In state machine transitions
if self.mqtt_bridge:
    if new_state == OperationState.STANDBY:
        self.mqtt_bridge.publish_status("STANDBY")
    elif new_state == OperationState.MOVING_TO_FOTO:
        self.mqtt_bridge.publish_status("RUNNING")
        self.mqtt_bridge.publish_sequence("MOVING_TO_FOTO")
    elif new_state == OperationState.CAPTURING:
        self.mqtt_bridge.publish_sequence("CAPTURING")
    elif new_state == OperationState.PROCESSING:
        self.mqtt_bridge.publish_sequence("PICKING")
```

### Step 5: Hook into Logger

Intercept log messages and publish to MQTT. Modify `ros2_logger.py` or add wrapper:

```python
class MQTTLoggingHandler(logging.Handler):
    """Custom handler to forward logs to MQTT."""
    
    def __init__(self, mqtt_bridge):
        super().__init__()
        self.mqtt_bridge = mqtt_bridge
    
    def emit(self, record):
        if self.mqtt_bridge and self.mqtt_bridge.is_connected():
            try:
                self.mqtt_bridge.publish_log(
                    level=record.levelname,
                    message=self.format(record)
                )
            except:
                pass  # Don't break logging if MQTT fails

# In robot_pnp_cli initialization:
if self.mqtt_bridge:
    mqtt_handler = MQTTLoggingHandler(self.mqtt_bridge)
    self.logger.logger.addHandler(mqtt_handler)
```

### Step 6: Publish Settings on Start

```python
# After loading configuration
if self.mqtt_bridge:
    settings = {
        "run_mode": self.config.get('run_mode', 'manual_confirm'),
        "continuous_operation": self.config.get('continuous_operation', {}).get('enabled', False),
        "speed_profile": self.config.get('movement', {}).get('default_profile', 'normal'),
        "mqtt_enabled": True,
        "vision_service": self.config.get('vision_service', {}).get('enabled', False)
    }
    self.mqtt_bridge.publish_settings(settings)
```

### Step 7: Cleanup on Exit

```python
def _cleanup(self):
    """Clean up resources."""
    # ... existing cleanup ...
    
    if self.mqtt_bridge:
        self.mqtt_bridge.stop()
```

---

## Testing

### Test MQTT Bridge Standalone:
```bash
python -m pickafresa_robot.robot_system.mqtt_bridge
```

### Test Supabase Uploader:
```bash
python -m pickafresa_vision.vision_tools.supabase_uploader \
  --image captures/test_raw.png \
  --json captures/test_data.json
```

### Test Vision Service with Supabase:
1. Update `vision_service_config.yaml`:
   ```yaml
   capture:
     supabase:
       enabled: true
       upload_mode: "async"
   ```

2. Ensure `.env` has Supabase credentials

3. Start vision service:
   ```bash
   python -m pickafresa_vision.vision_nodes.vision_service --preview
   ```

4. Request capture from robot_pnp_cli - uploads should happen automatically

---

## Setup Checklist

- [ ] Copy `.env.template` to `.env` and fill in credentials
- [ ] Create Supabase tables (`images`, `json_files`)
- [ ] Create Supabase storage bucket (`pickafresa-captures`)
- [ ] Configure MQTT broker address in `robot_pnp_config.yaml`
- [ ] Enable MQTT in `robot_pnp_config.yaml`: `mqtt.enabled: true`
- [ ] Enable Supabase in `vision_service_config.yaml`: `capture.supabase.enabled: true`
- [ ] Integrate MQTT bridge into `robot_pnp_cli.py` (see integration steps above)
- [ ] Test MQTT publishing to broker
- [ ] Test Supabase uploads
- [ ] Verify remote dashboard receives messages

---

## Known Limitations & Future Enhancements

### Current Limitations:
1. **Command Handling**: robot_pnp_cli doesn't yet handle pause/resume/emergency_stop commands (mentioned as future implementation)
2. **robot_pnp_service**: Deprecated, no integration added (as specified in requirements)
3. **Authentication**: MQTT authentication configured but not required yet

### Future Enhancements:
1. Add pause/resume/emergency_stop command handlers in robot_pnp_cli
2. Implement dashboard application for remote monitoring
3. Add MQTT authentication when security requirements increase
4. Add more detailed sequence steps (sub-steps within PICKING, etc.)
5. Implement bi-directional command acknowledgments

---

## Dependencies

All dependencies already in `requirements_macos.txt`:
- `paho-mqtt==2.1.0` - MQTT client
- `supabase==2.24.0` - Supabase client
- `python-dotenv==1.1.1` - Environment variable management

---

## Support & Documentation

### Module Documentation:
- `mqtt_bridge.py` - MQTT bridge for robot monitoring
- `supabase_uploader.py` - Cloud storage uploader
- `vision_service.py` - Vision system with Supabase integration

### Configuration Files:
- `.env.template` - Environment variables reference
- `robot_pnp_config.yaml` - Robot system configuration
- `vision_service_config.yaml` - Vision service configuration

### Test Scripts:
- `python -m pickafresa_robot.robot_system.mqtt_bridge` - Test MQTT
- `python -m pickafresa_vision.vision_tools.supabase_uploader` - Test uploads

---

**Implementation Complete**: All core functionality implemented and ready for integration. Follow integration steps above to connect MQTT bridge to robot_pnp_cli.
