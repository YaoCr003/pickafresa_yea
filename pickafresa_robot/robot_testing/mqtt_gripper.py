"""
MQTT Gripper Controller for Robot PnP Testing

Handles MQTT communication for gripper control with optional state confirmation,
override capability, and comprehensive logging.

Features:
- Connect/disconnect to MQTT broker
- Send gripper commands (open/close)
- Wait for state confirmation with timeout
- Override waiting with keypress
- Thread-safe operation

by: Aldrick T, 2025
for Team YEA
"""

import time
import threading
from typing import Optional, Callable
import paho.mqtt.client as mqtt

try:
    import keyboard
    HAVE_KEYBOARD = True
except ImportError:
    HAVE_KEYBOARD = False
    print("Warning: 'keyboard' module not available. Override functionality disabled.")


class MQTTGripperController:
    """MQTT-based gripper controller with state confirmation."""
    
    def __init__(
        self,
        broker_ip: str,
        broker_port: int = 1883,
        keepalive: int = 60,
        command_topic: str = "actuador/on_off",
        state_topic: str = "actuador/state",
        inflated_state: str = "inflado",
        deflated_state: str = "desinflado",
        logger: Optional[any] = None
    ):
        """
        Initialize MQTT gripper controller.
        
        Args:
            broker_ip: MQTT broker IP address
            broker_port: MQTT broker port
            keepalive: Connection keepalive interval (seconds)
            command_topic: Topic for sending gripper commands
            state_topic: Topic for receiving gripper state
            inflated_state: Expected state message for "gripper closed"
            deflated_state: Expected state message for "gripper open"
            logger: Logger instance (optional)
        """
        self.broker_ip = broker_ip
        self.broker_port = broker_port
        self.keepalive = keepalive
        self.command_topic = command_topic
        self.state_topic = state_topic
        self.inflated_state = inflated_state.lower()
        self.deflated_state = deflated_state.lower()
        self.logger = logger
        
        # State tracking
        self.current_state: Optional[str] = None
        self.is_connected = False
        self.state_lock = threading.Lock()
        
        # MQTT client
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        self._log_info("MQTT Gripper Controller initialized")
    
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
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker."""
        if rc == 0:
            self.is_connected = True
            self._log_info(f"[OK] Connected to MQTT broker at {self.broker_ip}:{self.broker_port}")
            # Subscribe to state topic
            client.subscribe(self.state_topic)
            self._log_info(f"[OK] Subscribed to topic: {self.state_topic}")
        else:
            self.is_connected = False
            self._log_error(f"[FAIL] Failed to connect to MQTT broker (code: {rc})")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker."""
        self.is_connected = False
        if rc == 0:
            self._log_info("Disconnected from MQTT broker")
        else:
            self._log_warn(f"Unexpected disconnection from MQTT broker (code: {rc})")
    
    def _on_message(self, client, userdata, msg):
        """Callback when message received."""
        if msg.topic == self.state_topic:
            message = msg.payload.decode().strip().lower()
            with self.state_lock:
                if message in [self.inflated_state, self.deflated_state]:
                    self.current_state = message
                    self._log_info(f"Gripper state update: '{message}'")
                else:
                    self._log_warn(f"Unknown state received: '{message}'")
    
    def connect(self) -> bool:
        """
        Connect to MQTT broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._log_info(f"Connecting to MQTT broker at {self.broker_ip}:{self.broker_port}...")
            self.client.connect(self.broker_ip, self.broker_port, self.keepalive)
            self.client.loop_start()
            
            # Wait for connection confirmation (max 5 seconds)
            timeout = 5.0
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if not self.is_connected:
                self._log_error("Connection timeout")
                return False
            
            return True
        
        except Exception as e:
            self._log_error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        self._log_info("Disconnecting from MQTT broker...")
        self.client.loop_stop()
        self.client.disconnect()
        self.is_connected = False
    
    def open_gripper(self) -> bool:
        """
        Send command to open gripper (deflate).
        
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.is_connected:
            self._log_error("Cannot send command: not connected to MQTT broker")
            return False
        
        try:
            self._log_info("Sending command: OPEN gripper")
            self.client.publish(self.command_topic, "Gripper apagado")
            return True
        except Exception as e:
            self._log_error(f"Failed to send open command: {e}")
            return False
    
    def close_gripper(self) -> bool:
        """
        Send command to close gripper (inflate).
        
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.is_connected:
            self._log_error("Cannot send command: not connected to MQTT broker")
            return False
        
        try:
            self._log_info("Sending command: CLOSE gripper")
            self.client.publish(self.command_topic, "Gripper encendido")
            return True
        except Exception as e:
            self._log_error(f"Failed to send close command: {e}")
            return False
    
    def wait_for_state(
        self,
        desired_state: str,
        timeout: float = 10.0,
        allow_override: bool = True,
        override_key: str = "c"
    ) -> bool:
        """
        Wait for gripper to reach desired state.
        
        Args:
            desired_state: Expected state ("inflado" or "desinflado")
            timeout: Maximum time to wait (seconds)
            allow_override: Allow user to skip waiting with keypress
            override_key: Key to press to override waiting
        
        Returns:
            True if state confirmed or overridden, False if timeout
        """
        desired_state = desired_state.lower()
        
        if desired_state not in [self.inflated_state, self.deflated_state]:
            self._log_error(f"Invalid desired state: {desired_state}")
            return False
        
        self._log_info(f"Waiting for gripper state: '{desired_state}' (timeout: {timeout}s)")
        if allow_override and HAVE_KEYBOARD:
            self._log_info(f"  Press '{override_key}' to continue without confirmation")
        
        # Clear current state
        with self.state_lock:
            self.current_state = None
        
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # Check for state match
            with self.state_lock:
                if self.current_state == desired_state:
                    self._log_info(f"[OK] Gripper state confirmed: '{desired_state}'")
                    return True
            
            # Check for override keypress
            if allow_override and HAVE_KEYBOARD:
                if keyboard.is_pressed(override_key):
                    self._log_warn(f"State confirmation overridden by user (key: '{override_key}')")
                    time.sleep(0.2)  # Debounce
                    return True
            
            # Check for emergency stop
            if HAVE_KEYBOARD and keyboard.is_pressed("esc"):
                self._log_error("Emergency stop detected (ESC pressed)")
                return False
            
            time.sleep(0.1)
        
        # Timeout
        self._log_error(f"[FAIL] Timeout waiting for state '{desired_state}' (waited {timeout}s)")
        return False
    
    def get_current_state(self) -> Optional[str]:
        """
        Get current gripper state.
        
        Returns:
            Current state string or None if unknown
        """
        with self.state_lock:
            return self.current_state
    
    def is_gripper_closed(self) -> bool:
        """Check if gripper is in closed state."""
        with self.state_lock:
            return self.current_state == self.inflated_state
    
    def is_gripper_open(self) -> bool:
        """Check if gripper is in open state."""
        with self.state_lock:
            return self.current_state == self.deflated_state


# Example usage
if __name__ == "__main__":
    # Create gripper controller
    gripper = MQTTGripperController(
        broker_ip="192.168.1.114",
        command_topic="actuador/on_off",
        state_topic="actuador/state"
    )
    
    # Connect
    if gripper.connect():
        print("Connected successfully!")
        
        # Test close gripper
        gripper.close_gripper()
        gripper.wait_for_state("inflado", timeout=5.0, allow_override=True)
        
        time.sleep(2)
        
        # Test open gripper
        gripper.open_gripper()
        gripper.wait_for_state("desinflado", timeout=5.0, allow_override=True)
        
        # Disconnect
        gripper.disconnect()
    else:
        print("Failed to connect!")
