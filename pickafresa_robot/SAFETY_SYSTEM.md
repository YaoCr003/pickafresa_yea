# Robot Safety System Documentation

**Critical Safety Features for pickafresa_robot**  
**Last Updated: November 24, 2025**  
**Priority: HIGH - Read before operating robot**

---

## Overview

This document describes the comprehensive safety system implemented to protect operators, equipment, and the robot during operation. The safety system provides three critical layers of protection:

1. **Collision Detection & Halt** - Immediate stop on collision detection
2. **Emergency Stop Detection** - UR teach pendant e-stop monitoring
3. **Pause/Resume Control** - Operator override capability

---

## 1. Collision Detection & Halt System

### How It Works

The system continuously monitors for collisions during robot operation:

- **Pre-Movement Checking**: Before every movement, the system simulates the path to detect potential collisions
- **During Movement Monitoring**: Active collision checking during robot motion using RoboDK's collision detection API
- **Post-Movement Verification**: Safety status check after each movement completes
- **Immediate Halt**: On collision detection, the robot halts immediately and raises `RobotCollisionError`

### Collision Detection Methods

1. **RoboDK Collision API**:
   - `RDK.setCollisionActive(1)` - Enables collision checking
   - `robot.MoveJ_Test()` / `robot.MoveL_Test()` - Pre-check path for collisions
   - `RDK.Collisions()` - Returns collision count during/after movement

2. **Robot Status Monitoring**:
   - Checks robot connection status
   - Monitors for movement errors (error code 2 = collision)
   - Verifies robot can read joint positions

### When Collision Is Detected

```
⚠️  COLLISION DETECTED: Collision detected (count: 2): Collision detected during movement
System HALTED - Check robot and clear collision

Collision cleared? Retry movement? [y/N]: 
```

**Operator Actions**:
1. **STOP** - Do not approach robot until fully halted
2. **INSPECT** - Check robot workspace for obstructions
3. **CLEAR** - Remove any obstacles or fix the issue
4. **VERIFY** - Ensure robot is in safe position
5. **RESET** - Type 'y' to reset safety state and retry
6. **OR ABORT** - Type 'n' to abort operation

### Real Robot Mode

In `real_robot` mode, collision detection is **MANDATORY** and cannot be bypassed:
- All movements verified in simulation before execution
- Collision causes automatic abort with no force option
- Manual intervention required to resume

---

## 2. Emergency Stop Detection

### UR Teach Pendant E-Stop

The system monitors for emergency stop activation on the UR robot teach pendant:

**Detection Method**:
- Continuous polling of robot status via RoboDK UR driver
- Attempts to read robot joints - failure indicates safety stop
- Error message parsing for e-stop keywords: "emergency", "e-stop", "estop"

### When E-Stop Is Activated

```
🚨 EMERGENCY STOP DETECTED: Emergency stop activated on robot
System HALTED - Reset e-stop on teach pendant to continue

Press Enter after resetting emergency stop...
```

**Safety Protocol**:
1. **HALT** - System immediately stops all operations
2. **NO RESUME** - System will not continue until e-stop is physically reset
3. **MANUAL RESET** - Operator must:
   - Press e-stop reset button on teach pendant
   - Verify robot is in safe state
   - Press Enter in terminal to acknowledge
4. **STATE VERIFICATION** - System checks robot is safe before allowing resume
5. **OPERATION RETRY** - User may retry failed operation after reset

### Safety Status Checks

The `check_robot_safety_status()` method runs:
- Before each movement
- During movement execution
- After movement completion
- Returns: `(is_safe: bool, error_message: Optional[str])`

**Checked Conditions**:
- Robot connection status
- Emergency stop state
- Protective stop state  
- Communication errors
- Collision state

---

## 3. Pause/Resume Control

### Keyboard Control (Priority)

**NEW**: Keyboard monitoring provides immediate pause/resume:

**Keys**:
- `SPACE` - Toggle pause/resume
- `P` - Toggle pause/resume

**Behavior**:
```
⏸  PAUSED via keyboard (will pause at safe point)
[System completes current operation safely, then pauses]

⏵  RESUMED via keyboard
[System continues from paused state]
```

### MQTT Control (Secondary)

Remote pause/resume via MQTT commands:

**Topics**:
- `robot/commands` - Publish: `{"command": "pause"}`
- `robot/commands` - Publish: `{"command": "resume"}`

**Priority**: Keyboard commands override MQTT commands

### Safe Pause Points

System pauses only at safe points:
- After completing current movement
- After finishing pick-and-place cycle
- Before starting next berry

**NOT during**:
- Active robot movement
- Gripper actuation
- Vision capture

---

## 4. Safety Exception Hierarchy

### Exception Types

```python
RobotSafetyError           # Base safety exception
├── RobotEmergencyStopError  # E-stop activated
└── RobotCollisionError      # Collision detected
```

### Exception Propagation

1. **Detection**: Safety issue detected in `robodk_manager.py`
2. **Raise**: Specific exception raised immediately
3. **Catch**: Caught in `robot_pnp_cli.py` movement methods
4. **Handle**: User prompted for action
5. **Reset**: Safety state reset after user intervention
6. **Retry or Abort**: Based on user decision

### Error Handling Flow

```
Movement Request
    ↓
Pre-Movement Safety Check ────→ [HALT if unsafe]
    ↓
Execute Movement
    ↓
Movement Exception? ────→ Check Safety Status ────→ [HALT if e-stop/collision]
    ↓
Post-Movement Safety Check ────→ [HALT if unsafe]
    ↓
Movement Complete
```

---

## 5. Configuration

### Enable/Disable Safety Features

**`robot_pnp_config.yaml`**:

```yaml
# Collision avoidance
collision_avoidance:
  enabled: true  # Set false to disable (NOT RECOMMENDED for real_robot)
  
# RoboDK settings
robodk:
  simulation_mode: "real_robot"  # or "simulate"
  robot_model: "UR3e"

# MQTT control (for remote pause/resume)
mqtt:
  enabled: true
  broker_ip: "192.168.1.114"
```

### Safety Intervals

```python
# In robodk_manager.py __init__:
self.safety_check_interval = 0.1  # Check every 100ms
```

---

## 6. Testing Procedures

### Before Each Session

1. **E-Stop Test**:
   - Start robot system
   - Press e-stop during movement
   - Verify system halts immediately
   - Verify no resume without reset
   - Reset e-stop and verify recovery

2. **Collision Test** (Simulation):
   - Create intentional collision scenario
   - Verify collision detection
   - Verify system halts
   - Verify error messages
   - Test retry functionality

3. **Pause/Resume Test**:
   - Start operation
   - Press SPACE during execution
   - Verify pause at safe point
   - Press SPACE again
   - Verify resume

### Real Robot Testing

**NEVER test collision on real robot**. Use simulation only.

For real robot:
1. Test e-stop in safe, slow movements
2. Test pause/resume during non-critical operations
3. Always have e-stop accessible
4. Always have clear escape route

---

## 7. Emergency Procedures

### If Robot Behaves Unexpectedly

1. **IMMEDIATELY** press e-stop on teach pendant
2. **DO NOT** approach robot until fully stopped
3. Check terminal for error messages
4. Take photos/notes of robot position
5. Check logs: `pickafresa_robot/logs/robot_pnp.log`

### If System Doesn't Halt on Collision

1. Press Ctrl+C in terminal
2. Activate e-stop
3. **DO NOT RESTART** until issue is identified
4. Report incident with:
   - Log files
   - Configuration used
   - Description of collision
   - Robot position/state

### If E-Stop Not Detected

1. **STOP USING REAL ROBOT**
2. Test in simulation mode only
3. Verify RoboDK UR driver connection
4. Check logs for connection errors
5. Verify teach pendant functionality

---

## 8. Limitations & Known Issues

### Collision Detection

- **Simulation accuracy**: Collision models may not match real robot exactly
- **Soft collisions**: Very light contact may not be detected
- **Dynamic obstacles**: Moving obstacles not detected
- **Cable interference**: Robot cables not modeled

### E-Stop Detection

- **Polling interval**: 100ms interval means slight delay
- **Connection dependent**: Requires stable RoboDK UR driver connection
- **False positives**: Communication errors may trigger false e-stop

### Pause/Resume

- **macOS limitation**: Keyboard monitoring not available on macOS (use MQTT)
- **Pause delay**: Completes current operation before pausing
- **State sync**: MQTT status may lag slightly

---

## 9. Maintenance

### Regular Checks

**Daily** (when operating):
- Test e-stop functionality
- Verify collision detection in simulation
- Check log files for warnings

**Weekly**:
- Review safety system logs
- Test pause/resume controls
- Verify RoboDK station collision models up to date

**Monthly**:
- Full safety system test in simulation
- Review and update safety documentation
- Train operators on safety procedures

### Log Monitoring

Watch for these patterns in logs:

```bash
# Warning signs:
grep "collision" pickafresa_robot/logs/robot_pnp.log
grep "emergency" pickafresa_robot/logs/robot_pnp.log  
grep "safety" pickafresa_robot/logs/robot_pnp.log
grep "ERROR" pickafresa_robot/logs/robot_pnp.log
```

---

## 10. Support & Reporting

### If You Find a Safety Issue

**CRITICAL**: Do not continue operating until resolved.

**Report immediately** with:
1. Description of safety issue
2. Log files from incident
3. Configuration file used
4. Steps to reproduce
5. Robot state/position

### Contact

- **Development Team**: Team YEA
- **Safety Lead**: Aldrick T.
- **Emergency**: Stop all operations and secure area

---

## Version History

- **v1.0** (2025-11-24): Initial implementation
  - Collision detection with immediate halt
  - E-stop monitoring and detection
  - Keyboard pause/resume control
  - Comprehensive safety exception system

---

**Remember: Safety is not optional. When in doubt, stop and ask.**
