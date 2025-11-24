# MQTT Topics Quick Reference

## Published Topics (Robot → Dashboard)

### `robot/log`
**Purpose**: Real-time log messages from robot system  
**QoS**: 0 (fire and forget)  
**Payload Format**:
```json
{
  "timestamp": "2025-11-23T12:30:45.123456",
  "level": "INFO|WARN|ERROR|DEBUG",
  "message": "Log message text"
}
```

**Example**:
```json
{
  "timestamp": "2025-11-23T12:30:45.123456",
  "level": "INFO",
  "message": "Robot moving to FOTO position"
}
```

---

### `robot/status`
**Purpose**: Current robot operational status  
**QoS**: 1 (at least once)  
**Payload Format**:
```json
{
  "timestamp": "2025-11-23T12:30:45.123456",
  "status": "RUNNING|STANDBY|ERROR|OFF",
  "...": "additional status fields"
}
```

**Status Values**:
- `RUNNING` - Robot actively executing tasks
- `STANDBY` - Robot idle, waiting for trigger or command
- `ERROR` - Robot in error state
- `OFF` - Robot system offline/shutdown

**Example**:
```json
{
  "timestamp": "2025-11-23T12:30:45.123456",
  "status": "RUNNING",
  "current_berry": 2,
  "total_berries": 5
}
```

---

### `robot/sequence`
**Purpose**: Current step in robot operation sequence  
**QoS**: 1 (at least once)  
**Payload Format**:
```json
{
  "timestamp": "2025-11-23T12:30:45.123456",
  "step": "sequence_step_name",
  "details": "Optional details about the step"
}
```

**Sequence Steps**:
- `STARTUP` - Initial system startup
- `MOVING_TO_FOTO` - Moving to idle/photo position
- `STANDBY` - Waiting at FOTO position
- `CAPTURING` - Requesting vision capture
- `PROCESSING` - Processing detected berries (pick-place loop)
- `PICKING` - Executing pick operation
- `PLACING` - Executing place operation
- `VERIFYING` - Verifying pick success
- `SHUTDOWN` - Graceful shutdown in progress

**Example**:
```json
{
  "timestamp": "2025-11-23T12:30:45.123456",
  "step": "PICKING",
  "details": "Berry 2/5 at position (150, 220, 350mm)"
}
```

---

### `robot/settings`
**Purpose**: Current robot configuration/settings  
**QoS**: 1 (at least once)  
**Payload Format**:
```json
{
  "timestamp": "2025-11-23T12:30:45.123456",
  "settings": "key1=value1, key2=value2, key3=value3",
  "settings_dict": {
    "key1": "value1",
    "key2": "value2",
    "key3": "value3"
  }
}
```

**Common Settings**:
- `run_mode` - "manual_confirm" or "autonomous"
- `continuous_operation` - true/false
- `speed_profile` - "turtle", "slow", "normal", "custom"
- `mqtt_enabled` - true/false
- `vision_service` - true/false
- `gripper_type` - MQTT gripper type

**Example**:
```json
{
  "timestamp": "2025-11-23T12:30:45.123456",
  "settings": "run_mode=autonomous, speed_profile=normal, continuous_operation=true",
  "settings_dict": {
    "run_mode": "autonomous",
    "speed_profile": "normal",
    "continuous_operation": true
  }
}
```

---

## Subscribed Topics (Dashboard → Robot)

### `robot/commands`
**Purpose**: Remote commands for robot control  
**Payload Format**:
```json
{
  "command": "command_name",
  "params": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

**Available Commands**:

#### `stop`
Stop continuous operation (graceful)
```json
{
  "command": "stop",
  "params": {}
}
```

#### `start`
Start continuous operation
```json
{
  "command": "start",
  "params": {}
}
```

#### `pause` (Future)
Pause current operation
```json
{
  "command": "pause",
  "params": {}
}
```

#### `resume` (Future)
Resume paused operation
```json
{
  "command": "resume",
  "params": {}
}
```

#### `emergency_stop` (Future)
Emergency stop (immediate halt)
```json
{
  "command": "emergency_stop",
  "params": {
    "reason": "User initiated emergency stop"
  }
}
```

---

## Dashboard Implementation Example

### Python (with paho-mqtt):
```python
import json
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # Subscribe to all robot topics
    client.subscribe("robot/log")
    client.subscribe("robot/status")
    client.subscribe("robot/sequence")
    client.subscribe("robot/settings")

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = json.loads(msg.payload.decode('utf-8'))
    
    if topic == "robot/log":
        print(f"[{payload['level']}] {payload['message']}")
    elif topic == "robot/status":
        print(f"Status: {payload['status']}")
    elif topic == "robot/sequence":
        print(f"Step: {payload['step']} - {payload.get('details', '')}")
    elif topic == "robot/settings":
        print(f"Settings: {payload['settings']}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("192.168.1.114", 1883, 60)

# Send command
def send_command(command, params=None):
    payload = {
        "command": command,
        "params": params or {}
    }
    client.publish("robot/commands", json.dumps(payload))

# Start listening
client.loop_start()

# Send stop command after some time
import time
time.sleep(10)
send_command("stop")
```

### Node.js (with mqtt.js):
```javascript
const mqtt = require('mqtt');
const client = mqtt.connect('mqtt://192.168.1.114:1883');

client.on('connect', () => {
    console.log('Connected to broker');
    client.subscribe('robot/log');
    client.subscribe('robot/status');
    client.subscribe('robot/sequence');
    client.subscribe('robot/settings');
});

client.on('message', (topic, message) => {
    const payload = JSON.parse(message.toString());
    
    if (topic === 'robot/log') {
        console.log(`[${payload.level}] ${payload.message}`);
    } else if (topic === 'robot/status') {
        console.log(`Status: ${payload.status}`);
    } else if (topic === 'robot/sequence') {
        console.log(`Step: ${payload.step}`);
    } else if (topic === 'robot/settings') {
        console.log(`Settings: ${payload.settings}`);
    }
});

// Send command
function sendCommand(command, params = {}) {
    const payload = JSON.stringify({ command, params });
    client.publish('robot/commands', payload);
}

// Example: send stop command
setTimeout(() => {
    sendCommand('stop');
}, 10000);
```

---

## Testing with mosquitto_pub/sub

### Subscribe to all robot topics:
```bash
# Terminal 1 - Logs
mosquitto_sub -h 192.168.1.114 -t "robot/log" -v

# Terminal 2 - Status
mosquitto_sub -h 192.168.1.114 -t "robot/status" -v

# Terminal 3 - Sequence
mosquitto_sub -h 192.168.1.114 -t "robot/sequence" -v

# Terminal 4 - Settings
mosquitto_sub -h 192.168.1.114 -t "robot/settings" -v

# Terminal 5 - All topics
mosquitto_sub -h 192.168.1.114 -t "robot/#" -v
```

### Send commands:
```bash
# Stop command
mosquitto_pub -h 192.168.1.114 -t "robot/commands" \
  -m '{"command": "stop", "params": {}}'

# Start command
mosquitto_pub -h 192.168.1.114 -t "robot/commands" \
  -m '{"command": "start", "params": {}}'
```

---

## Broker Configuration

Default broker from `robot_pnp_config.yaml`:
```yaml
mqtt:
  enabled: true
  broker_ip: "192.168.1.114"
  broker_port: 1883
  keepalive: 60
```

Can be overridden via environment variables in `.env`:
```bash
MQTT_BROKER_IP=192.168.1.114
MQTT_BROKER_PORT=1883
MQTT_USERNAME=  # Optional
MQTT_PASSWORD=  # Optional
```

---

## Topic Organization

```
robot/
├── log          (logs, QoS 0)
├── status       (state updates, QoS 1)
├── sequence     (operation steps, QoS 1)
├── settings     (configuration, QoS 1)
└── commands     (remote control, subscribed)
```

**Note**: All timestamps are in ISO 8601 format with microsecond precision.
