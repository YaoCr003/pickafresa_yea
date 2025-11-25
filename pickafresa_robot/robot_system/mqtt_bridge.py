"""
MQTT Bridge for Robot PnP CLI System

Provides MQTT communication for remote monitoring and control of robot_pnp_cli.
Publishes robot logs, status, sequence steps, and settings to MQTT topics.
Receives commands from remote dashboard for robot control.

MQTT Topics (Publish):
    - robot/log: Real-time robot logs from ros2_logger
    - robot/status: Robot state (RUNNING/STANDBY/ERROR/OFF)
    - robot/sequence: Current step in robot sequence
    - robot/settings: General robot settings/configuration

MQTT Topics (Subscribe):
    - robot/commands: Remote commands (stop, start, etc.)

Architecture:
    robot_pnp_cli <--> MQTTBridge <--> MQTT Broker <--> Remote Dashboard

Usage:
    from pickafresa_robot.robot_system.mqtt_bridge import MQTTBridge
    
    # Initialize bridge
    bridge = MQTTBridge(config, logger)
    bridge.start()
    
    # Publish updates
    bridge.publish_status("RUNNING")
    bridge.publish_log("INFO", "Robot moving to position")
    bridge.publish_sequence("CAPTURING")
    
    # Stop bridge
    bridge.stop()

@aldrick-t, 2025
for Team YEA
"""

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Repository root setup
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# MQTT client (optional dependency)
MQTT_AVAILABLE = False
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    mqtt = None


class MQTTBridgeError(Exception):
    """Base exception for MQTT bridge errors."""
    pass


class MQTTBridge:
    """
    MQTT bridge for robot_pnp_cli remote monitoring and control.
    
    Publishes logs, status, sequence steps, and settings to MQTT broker.
    Subscribes to command topic for remote control.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger=None,
        command_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        Initialize MQTT bridge.
        
        Args:
            config: Configuration dictionary (from robot_pnp_config.yaml)
            logger: Logger instance (ros2_logger)
            command_callback: Callback function for received commands: callback(command, params)
        """
        self.config = config
        self.logger = logger
        self.command_callback = command_callback
        
        # MQTT configuration
        mqtt_config = config.get('mqtt', {})
        self.enabled = mqtt_config.get('enabled', False)
        self.broker_ip = mqtt_config.get('broker_ip', '192.168.1.114')
        self.broker_port = mqtt_config.get('broker_port', 1883)
        self.keepalive = mqtt_config.get('keepalive', 60)
        
        # MQTT authentication (if configured)
        self.mqtt_username = config.get('MQTT_USERNAME')  # From env if needed
        self.mqtt_password = config.get('MQTT_PASSWORD')
        
        # Topics
        self.topic_log = "robot/log"
        self.topic_status = "robot/status"
        self.topic_sequence = "robot/sequence"
        self.topic_settings = "robot/settings"
        self.topic_commands = "robot/commands"
        
        # State
        self.running = False
        self.connected = False
        self.client = None
        
        # Check availability
        if not MQTT_AVAILABLE:
            self._log_warning("paho-mqtt not available. Install: pip install paho-mqtt")
            self.enabled = False
        
        if not self.enabled:
            self._log_info("MQTT bridge disabled in config")
            return
        
        # Initialize MQTT client
        try:
            self.client = mqtt.Client(client_id="robot_pnp_cli_bridge")
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            
            # Set authentication if configured
            if self.mqtt_username and self.mqtt_password:
                self.client.username_pw_set(self.mqtt_username, self.mqtt_password)
            
            self._log_info("MQTT bridge initialized")
            self._log_info(f"  Broker: {self.broker_ip}:{self.broker_port}")
        except Exception as e:
            self._log_error(f"Failed to initialize MQTT client: {e}")
            self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if MQTT bridge is enabled."""
        return self.enabled and self.client is not None
    
    def is_connected(self) -> bool:
        """Check if MQTT broker is connected."""
        return self.connected
    
    def start(self) -> bool:
        """
        Start MQTT bridge (connect to broker).
        
        Returns:
            True if started successfully
        """
        if not self.is_enabled():
            return False
        
        try:
            self._log_info("Connecting to MQTT broker...")
            self.client.connect(self.broker_ip, self.broker_port, self.keepalive)
            
            # Start MQTT loop in background thread
            self.running = True
            self.client.loop_start()
            
            # Wait for connection (with timeout)
            timeout = 5.0
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if self.connected:
                self._log_info("✓ MQTT bridge started")
                return True
            else:
                self._log_warning("MQTT connection timeout")
                return False
        
        except Exception as e:
            self._log_error(f"Failed to start MQTT bridge: {e}")
            return False
    
    def stop(self):
        """Stop MQTT bridge (disconnect from broker)."""
        if not self.is_enabled():
            return
        
        try:
            self._log_info("Stopping MQTT bridge...")
            self.running = False
            
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
            
            self.connected = False
            self._log_info("✓ MQTT bridge stopped")
        
        except Exception as e:
            self._log_error(f"Error stopping MQTT bridge: {e}")
    
    def publish_log(self, level: str, message: str):
        """
        Publish log message to MQTT.
        
        Args:
            level: Log level (INFO, WARN, ERROR, DEBUG)
            message: Log message
        
        Format: [timestamp, level, message]
        Example: ["2025-11-23T12:30:45.123456", "INFO", "Robot moving to FOTO position"]
        """
        if not self.is_connected():
            self._log_debug(f"Cannot publish log - not connected to MQTT broker")
            return
        
        try:
            payload = [
                datetime.now().isoformat(),
                level,
                message
            ]
            
            result = self.client.publish(
                self.topic_log,
                json.dumps(payload),
                qos=0  # Fire and forget for logs
            )
            self._log_debug(f"Published to {self.topic_log}: [{level}] {message[:50]}... (rc={result.rc})")
        
        except Exception as e:
            self._log_error(f"Failed to publish log: {e}")
    
    def publish_status(self, status: str, extra_data: Optional[Dict[str, Any]] = None):
        """
        Publish robot status to MQTT.
        
        Args:
            status: Robot status (RUNNING, STANDBY, ERROR, OFF, etc.)
            extra_data: Optional additional status data
        
        Format: [timestamp, status, extra_info]
        Example: ["2025-11-23T12:30:45.123456", "RUNNING", "paused=False"]
        """
        if not self.is_connected():
            self._log_debug(f"Cannot publish status - not connected to MQTT broker")
            return
        
        try:
            # Convert extra_data to string if provided
            extra_str = ""
            if extra_data:
                extra_str = ", ".join([f"{k}={v}" for k, v in extra_data.items()])
            
            payload = [
                datetime.now().isoformat(),
                status,
                extra_str
            ]
            
            result = self.client.publish(
                self.topic_status,
                json.dumps(payload),
                qos=1  # At least once delivery for status
            )
            self._log_debug(f"Published to {self.topic_status}: {status} {extra_str} (rc={result.rc})")
        
        except Exception as e:
            self._log_error(f"Failed to publish status: {e}")
    
    def publish_sequence(self, sequence_step: str, details: Optional[str] = None):
        """
        Publish current sequence step to MQTT.
        
        Args:
            sequence_step: Current step (MOVING_TO_FOTO, CAPTURING, PICKING, etc.)
            details: Optional details about the step
        
        Format: [timestamp, step, details]
        Example: ["2025-11-23T12:30:45.123456", "CAPTURING", "Capturing berry data"]
        """
        if not self.is_connected():
            self._log_debug(f"Cannot publish sequence - not connected to MQTT broker")
            return
        
        try:
            payload = [
                datetime.now().isoformat(),
                sequence_step,
                details if details else ""
            ]
            
            result = self.client.publish(
                self.topic_sequence,
                json.dumps(payload),
                qos=1  # At least once delivery
            )
            self._log_debug(f"Published to {self.topic_sequence}: {sequence_step} - {details} (rc={result.rc})")
        
        except Exception as e:
            self._log_error(f"Failed to publish sequence: {e}")
    
    def publish_settings(self, settings: Dict[str, Any]):
        """
        Publish robot settings/configuration to MQTT.
        
        Args:
            settings: Dictionary of settings to publish
        
        Format: [timestamp, type, settings_string]
        Example: ["2025-11-23T12:30:45.123456", "CONFIG", "run_mode=manual, speed=normal"]
        """
        if not self.is_connected():
            self._log_debug(f"Cannot publish settings - not connected to MQTT broker")
            return
        
        try:
            # Convert settings to comma-separated string
            settings_str = ", ".join([f"{k}={v}" for k, v in settings.items()])
            
            payload = [
                datetime.now().isoformat(),
                "CONFIG",
                settings_str
            ]
            
            result = self.client.publish(
                self.topic_settings,
                json.dumps(payload),
                qos=1  # At least once delivery
            )
            self._log_debug(f"Published to {self.topic_settings}: {settings_str} (rc={result.rc})")
        
        except Exception as e:
            self._log_error(f"Failed to publish settings: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self.connected = True
            self._log_info("✓ Connected to MQTT broker")
            self._log_debug(f"MQTT connection established with broker {self.broker_host}:{self.broker_port}")
            
            # Subscribe to command topic
            result = client.subscribe(self.topic_commands)
            print(f"[MQTT] Subscribed to {self.topic_commands} (result={result})")
            self._log_info(f"Subscribed to {self.topic_commands}")
            self._log_debug(f"Ready to publish to topics: {self.topic_log}, {self.topic_status}, {self.topic_sequence}, {self.topic_settings}")
        else:
            self.connected = False
            self._log_error(f"MQTT connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        self.connected = False
        
        if rc != 0:
            self._log_warning(f"Unexpected MQTT disconnection (code: {rc})")
            self._log_debug(f"Disconnect reason code: {rc}")
        else:
            self._log_info("Disconnected from MQTT broker")
            self._log_debug("Clean disconnection from MQTT broker")
    
    def _on_message(self, client, userdata, msg):
        """
        MQTT message callback (for received commands).
        
        Args:
            client: MQTT client
            userdata: User data
            msg: MQTT message
        """
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            # Use print for command reception to ensure visibility
            print(f"[MQTT] Received message on {topic}: {payload[:100]}...")
            self._log_info(f"Received MQTT message on {topic}")
            self._log_debug(f"Message payload: {payload[:100]}...")
            
            if topic == self.topic_commands:
                # Parse command
                try:
                    command_data = json.loads(payload)
                    command = command_data.get('command')
                    params = command_data.get('params', {})
                    
                    print(f"[MQTT] Command received: {command} with params: {params}")
                    self._log_info(f"Command received: {command}")
                    self._log_debug(f"Command parameters: {params}")
                    
                    # Call command callback if configured
                    if self.command_callback:
                        self.command_callback(command, params)
                    else:
                        self._log_warning("No command callback configured")
                
                except json.JSONDecodeError:
                    self._log_error(f"Invalid JSON in command: {payload}")
        
        except Exception as e:
            self._log_error(f"Error processing MQTT message: {e}")
    
    def _log_info(self, message: str):
        """Log info message (without MQTT publishing to avoid recursion)."""
        if self.logger:
            # Temporarily disable mqtt_callback to prevent infinite recursion
            old_callback = getattr(self.logger, 'mqtt_callback', None)
            self.logger.mqtt_callback = None
            self.logger.info(message)
            self.logger.mqtt_callback = old_callback
        else:
            print(f"[INFO] [mqtt_bridge]: {message}")
    
    def _log_warning(self, message: str):
        """Log warning message (without MQTT publishing to avoid recursion)."""
        if self.logger:
            # Temporarily disable mqtt_callback to prevent infinite recursion
            old_callback = getattr(self.logger, 'mqtt_callback', None)
            self.logger.mqtt_callback = None
            self.logger.warn(message)
            self.logger.mqtt_callback = old_callback
        else:
            print(f"[WARN] [mqtt_bridge]: {message}")
    
    def _log_error(self, message: str):
        """Log error message (without MQTT publishing to avoid recursion)."""
        if self.logger:
            # Temporarily disable mqtt_callback to prevent infinite recursion
            old_callback = getattr(self.logger, 'mqtt_callback', None)
            self.logger.mqtt_callback = None
            self.logger.error(message)
            self.logger.mqtt_callback = old_callback
        else:
            print(f"[ERROR] [mqtt_bridge]: {message}")
    
    def _log_debug(self, message: str):
        """Log debug message (without MQTT publishing to avoid recursion)."""
        if self.logger:
            # Temporarily disable mqtt_callback to prevent infinite recursion
            old_callback = getattr(self.logger, 'mqtt_callback', None)
            self.logger.mqtt_callback = None
            self.logger.debug(message)
            self.logger.mqtt_callback = old_callback
        else:
            print(f"[DEBUG] [mqtt_bridge]: {message}")


def main():
    """Test MQTT bridge."""
    from pickafresa_robot.robot_system.config_manager import ConfigManager
    from pickafresa_robot.robot_system.ros2_logger import create_logger
    
    # Load config
    config_path = REPO_ROOT / "pickafresa_robot" / "configs" / "robot_pnp_config.yaml"
    config_manager = ConfigManager(config_path)
    
    # Create logger
    logger = create_logger(
        node_name="mqtt_bridge_test",
        log_dir=str(REPO_ROOT / "pickafresa_robot/logs")
    )
    
    # Command callback
    def command_handler(command: str, params: Dict[str, Any]):
        logger.info(f"Handling command: {command} with params: {params}")
    
    # Create and start bridge
    bridge = MQTTBridge(config_manager.config, logger, command_callback=command_handler)
    
    if not bridge.is_enabled():
        logger.error("MQTT bridge not enabled")
        return 1
    
    if not bridge.start():
        logger.error("Failed to start MQTT bridge")
        return 1
    
    try:
        # Test publishing
        logger.info("Testing MQTT publishing...")
        
        bridge.publish_status("TESTING")
        time.sleep(0.5)
        
        bridge.publish_log("INFO", "Test log message")
        time.sleep(0.5)
        
        bridge.publish_sequence("TESTING", "Testing sequence step")
        time.sleep(0.5)
        
        bridge.publish_settings({
            "run_mode": "autonomous",
            "speed_profile": "normal",
            "continuous_operation": "enabled"
        })
        
        logger.info("Published test messages. Waiting for commands...")
        logger.info(f"Send commands to topic: {bridge.topic_commands}")
        logger.info("Example: {\"command\": \"stop\", \"params\": {}}")
        
        # Keep running
        while True:
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        bridge.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
