"""
Robot PnP Remote - MQTT Bridge with Supabase Upload

Provides remote control via MQTT with optional cloud database integration.
Bridges MQTT commands to local robot_pnp_service and uploads operation logs to Supabase.

MQTT Topics:
- robot/command/execute_pick  (publish: berry_index)
- robot/command/status        (publish: empty)
- robot/command/emergency     (publish: empty)
- robot/status/*              (subscribe: various status updates)

Priority: Local CLI (robot_pnp_manager) > Remote MQTT
- Local commands interrupt remote operations
- Remote commands queue if local is active

by: Aldrick T, 2025
for Team YEA
"""

import sys
import json
import socket
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

# Repository root setup
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("[FAIL] paho-mqtt not installed. Install: pip install paho-mqtt")
    sys.exit(1)

# Optional: Supabase for cloud logging
SUPABASE_AVAILABLE = False
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    print("[INFO] Supabase not available (optional). Install: pip install supabase")

from pickafresa_robot.robot_system.config_manager import ConfigManager
from pickafresa_robot.robot_system.ros2_logger import create_logger


class ServiceClient:
    """Lightweight client for robot_pnp_service."""
    
    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
    
    def send_command(self, command: str, **params) -> Dict[str, Any]:
        """Send command to service."""
        request = {'command': command, **params}
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            
            request_str = json.dumps(request) + '\n'
            sock.sendall(request_str.encode('utf-8'))
            
            buffer = b''
            while b'\n' not in buffer:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
            
            response_str = buffer.decode('utf-8').strip()
            response = json.loads(response_str)
            
            sock.close()
            return response
        
        except Exception as e:
            return {'status': 'error', 'error': f'Client error: {e}'}


class RobotPnPRemote:
    """
    MQTT bridge for remote robot control with optional Supabase logging.
    """
    
    def __init__(self, config: ConfigManager, logger=None):
        self.config = config
        self.logger = logger or create_logger(
            node_name="robot_pnp_remote",
            log_dir=str(REPO_ROOT / "pickafresa_robot/logs")
        )
        
        # Service client
        service_config = config.get('service', {})
        self.service_client = ServiceClient(
            host=service_config.get('host', '127.0.0.1'),
            port=service_config.get('port', 5556)
        )
        
        # MQTT client
        mqtt_config = config.get('mqtt', {})
        self.mqtt_broker = mqtt_config.get('broker_address', '192.168.1.100')
        self.mqtt_port = mqtt_config.get('broker_port', 1883)
        
        self.mqtt_client = mqtt.Client(client_id="robot_pnp_remote")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        
        # Supabase (optional)
        self.supabase_client: Optional[Client] = None
        self._init_supabase()
        
        # State
        self.running = False
        self.local_override = False  # True if local CLI is active
        
        # Topics
        self.topic_execute_pick = "robot/command/execute_pick"
        self.topic_status = "robot/command/status"
        self.topic_emergency = "robot/command/emergency"
        self.topic_status_reply = "robot/status/reply"
        
        self.logger.info("=" * 80)
        self.logger.info("Robot PnP Remote initialized")
        self.logger.info(f"MQTT Broker: {self.mqtt_broker}:{self.mqtt_port}")
        self.logger.info(f"Service: {self.service_client.host}:{self.service_client.port}")
        self.logger.info(f"Supabase: {'Enabled' if self.supabase_client else 'Disabled'}")
        self.logger.info("=" * 80)
    
    def _init_supabase(self):
        """Initialize Supabase client (optional)."""
        if not SUPABASE_AVAILABLE:
            return
        
        supabase_config = self.config.get('supabase', {})
        
        if not supabase_config.get('enabled', False):
            self.logger.info("Supabase disabled in config")
            return
        
        url = supabase_config.get('url')
        key = supabase_config.get('key')
        
        if not url or not key:
            self.logger.warning("Supabase credentials missing")
            return
        
        try:
            self.supabase_client = create_client(url, key)
            self.logger.info("[OK] Supabase client initialized")
        except Exception as e:
            self.logger.error(f"Supabase initialization failed: {e}")
    
    def start(self):
        """Start MQTT bridge."""
        try:
            # Connect to MQTT broker
            self.logger.info("Connecting to MQTT broker...")
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            
            # Start MQTT loop
            self.running = True
            self.mqtt_client.loop_start()
            
            self.logger.info("[OK] MQTT bridge started")
            
            # Status monitoring thread
            status_thread = threading.Thread(target=self._status_monitor, daemon=True)
            status_thread.start()
            
            # Keep running
            while self.running:
                time.sleep(1.0)
        
        except Exception as e:
            self.logger.error(f"MQTT bridge start failed: {e}")
            self.logger.debug(traceback.format_exc())
    
    def stop(self):
        """Stop MQTT bridge."""
        self.logger.info("Stopping MQTT bridge...")
        self.running = False
        
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        
        self.logger.info("[OK] MQTT bridge stopped")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self.logger.info("[OK] Connected to MQTT broker")
            
            # Subscribe to command topics
            client.subscribe(self.topic_execute_pick)
            client.subscribe(self.topic_status)
            client.subscribe(self.topic_emergency)
            
            self.logger.info(f"Subscribed to: {self.topic_execute_pick}")
            self.logger.info(f"Subscribed to: {self.topic_status}")
            self.logger.info(f"Subscribed to: {self.topic_emergency}")
        else:
            self.logger.error(f"MQTT connection failed: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        self.logger.info(f"MQTT message: {topic} = {payload}")
        
        try:
            if topic == self.topic_execute_pick:
                self._handle_execute_pick(payload)
            
            elif topic == self.topic_status:
                self._handle_status_request()
            
            elif topic == self.topic_emergency:
                self._handle_emergency()
            
            else:
                self.logger.warning(f"Unknown topic: {topic}")
        
        except Exception as e:
            self.logger.error(f"Message handling error: {e}")
            self.logger.debug(traceback.format_exc())
    
    def _handle_execute_pick(self, payload: str):
        """Handle execute_pick command."""
        if self.local_override:
            self.logger.warning("Local CLI override active - remote command ignored")
            self._publish_status({'status': 'error', 'error': 'Local override active'})
            return
        
        try:
            # Parse berry index
            berry_index = int(payload) if payload.strip() else 0
            
            self.logger.info(f"Executing pick for berry #{berry_index}...")
            
            # Send to service
            response = self.service_client.send_command('execute_pick', berry_index=berry_index)
            
            # Publish result
            self._publish_status(response)
            
            # Log to Supabase
            if response['status'] == 'success':
                self._log_to_supabase('pick_success', berry_index, response['data'])
            else:
                self._log_to_supabase('pick_failed', berry_index, {'error': response.get('error')})
        
        except ValueError:
            self.logger.error(f"Invalid berry_index: {payload}")
            self._publish_status({'status': 'error', 'error': 'Invalid berry_index'})
    
    def _handle_status_request(self):
        """Handle status request."""
        self.logger.info("Status requested")
        
        response = self.service_client.send_command('status')
        self._publish_status(response)
    
    def _handle_emergency(self):
        """Handle emergency stop."""
        self.logger.warning("[EMERGENCY] EMERGENCY STOP via MQTT")
        
        response = self.service_client.send_command('shutdown')
        self._publish_status(response)
        
        self._log_to_supabase('emergency_stop', None, {})
    
    def _publish_status(self, data: Dict[str, Any]):
        """Publish status to MQTT."""
        try:
            payload = json.dumps(data)
            self.mqtt_client.publish(self.topic_status_reply, payload)
        except Exception as e:
            self.logger.error(f"Status publish error: {e}")
    
    def _status_monitor(self):
        """Periodic status monitoring thread."""
        while self.running:
            try:
                # Check service status every 30 seconds
                time.sleep(30.0)
                
                if not self.running:
                    break
                
                response = self.service_client.send_command('status')
                
                if response['status'] == 'error':
                    self.logger.warning(f"Service health check failed: {response.get('error')}")
            
            except Exception as e:
                self.logger.error(f"Status monitor error: {e}")
    
    def _log_to_supabase(self, event_type: str, berry_index: Optional[int], data: Dict[str, Any]):
        """Log event to Supabase."""
        if not self.supabase_client:
            return
        
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': event_type,
                'berry_index': berry_index,
                'data': json.dumps(data)
            }
            
            table_name = self.config.get('supabase.table_name', 'robot_operations')
            
            self.supabase_client.table(table_name).insert(log_entry).execute()
            
            self.logger.debug(f"Logged to Supabase: {event_type}")
        
        except Exception as e:
            self.logger.error(f"Supabase logging error: {e}")


# Command-line entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Robot PnP Remote - MQTT Bridge")
    parser.add_argument('--config', type=str,
                       default='pickafresa_robot/configs/robot_pnp_config.yaml',
                       help='Path to config YAML')
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    
    if not config_path.exists():
        print(f"[FAIL] Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    config = ConfigManager(config_path)
    
    # Create and start remote
    remote = RobotPnPRemote(config=config)
    
    try:
        print("\n[OK] Robot PnP Remote running. Press Ctrl+C to stop.\n")
        remote.start()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        remote.stop()
    except Exception as e:
        print(f"\n[FAIL] Remote error: {e}")
        traceback.print_exc()
