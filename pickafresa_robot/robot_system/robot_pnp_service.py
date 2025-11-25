"""
Robot PnP Service - Always-On IPC Server

This service wraps the RobotPnPController and provides a socket-based IPC interface.
Runs continuously, accepts commands from local or remote clients.

IPC Protocol (port 5556):
- Request: JSON object with 'command' field + parameters
- Response: JSON object with 'status', 'data', and optional 'error'

Commands:
- status: Get current controller status
- initialize: Initialize controller subsystems
- execute_pick: Execute pick sequence (params: berry_index, json_path)
- execute_multi_berry: Execute multi-berry sequence (params: json_path)
- shutdown: Graceful shutdown
- reload_config: Reload configuration (hot-reloadable keys only)
- stats: Get persistent statistics

by: Aldrick T, 2025
for Team YEA
"""

import sys
import json
import socket
import threading
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

# Repository root setup
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pickafresa_robot.robot_system.robot_pnp_controller import RobotPnPController
from pickafresa_robot.robot_system.config_manager import ConfigManager
from pickafresa_robot.robot_system.state_machine import RobotState
from pickafresa_robot.robot_system.ros2_logger import create_logger
from pickafresa_robot.robot_system.persistent_stats import PersistentStats


class RobotPnPService:
    """
    Always-on IPC service for robot pick-and-place operations.
    
    Listens on socket, processes commands, manages controller lifecycle.
    """
    
    def __init__(self, config_path: Path, host: str = '127.0.0.1', port: int = 5556):
        """
        Initialize service.
        
        Args:
            config_path: Path to YAML config file
            host: Server bind address
            port: Server port
        """
        self.config_path = config_path
        self.host = host
        self.port = port
        
        # Logger
        self.logger = create_logger(
            node_name="robot_pnp_service",
            log_dir=str(REPO_ROOT / "pickafresa_robot/logs")
        )
        
        # Config manager (with hot-reload)
        self.config = ConfigManager(config_path, logger=self.logger)
        self.config.register_change_callback(self._on_config_reload)
        
        # Controller
        self.controller: Optional[RobotPnPController] = None
        
        # Server state
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.client_threads: list = []
        
        # Persistent statistics
        stats_file = REPO_ROOT / "pickafresa_robot/logs/service_stats.json"
        self.persistent_stats = PersistentStats(stats_file)
        
        self.logger.info("=" * 80)
        self.logger.info("Robot PnP Service initialized")
        self.logger.info(f"Config: {config_path}")
        self.logger.info(f"Listening on {host}:{port}")
        
        # Show lifetime stats
        lifetime = self.persistent_stats.get_lifetime()
        if lifetime['service_starts'] > 0:
            self.logger.info(f"Lifetime: {lifetime['picks_total']} picks, {lifetime['service_starts']} starts")
        
        self.logger.info("=" * 80)
    
    def _on_config_reload(self, new_config: Dict[str, Any]):
        """Called when config is reloaded."""
        self.logger.info("[REFRESH] Configuration reloaded")
        
        # Update controller config if it exists
        if self.controller:
            # Controller already has reference to self.config which was updated
            self.logger.info("[OK] Controller config updated")
    
    def start(self) -> bool:
        """
        Start service.
        
        Returns:
            True if service started successfully
        """
        try:
            # Mark session start
            self.persistent_stats.start_session()
            
            # Create controller
            self.controller = RobotPnPController(config=self.config, logger=self.logger)
            
            # Initialize controller
            self.logger.info("Initializing controller...")
            if not self.controller.initialize():
                self.logger.error("Controller initialization failed")
                return False
            
            # Log offline mode status if vision unavailable
            if self.controller.offline_mode:
                self.logger.warning("=" * 80)
                self.logger.warning("SERVICE RUNNING IN OFFLINE MODE")
                self.logger.warning("=" * 80)
                self.logger.warning("Vision service is not available.")
                self.logger.warning("")
                self.logger.warning("The service will:")
                self.logger.warning("  - Accept connections normally")
                self.logger.warning("  - Prompt for JSON files when executing picks")
                self.logger.warning("")
                self.logger.warning("To enable live vision:")
                self.logger.warning("  1. Start vision service: ./realsense_venv_sudo pickafresa_vision/vision_nodes/vision_service.py")
                self.logger.warning("  2. Restart this service")
                self.logger.warning("=" * 80)
            
            # Start server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # For graceful shutdown
            
            self.running = True
            self.logger.info(f"[OK] Service started on {self.host}:{self.port}")
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start accepting connections
            self._accept_connections()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Service start failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def _accept_connections(self):
        """Accept client connections (blocking)."""
        self.logger.info("Accepting connections...")
        
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()
                self.logger.info(f"Client connected: {client_address}")
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_thread.start()
                self.client_threads.append(client_thread)
            
            except socket.timeout:
                continue
            
            except Exception as e:
                if self.running:
                    self.logger.error(f"Error accepting connection: {e}")
    
    def _handle_client(self, client_socket: socket.socket, client_address):
        """Handle a client connection."""
        response = None
        try:
            # Read request (newline-terminated JSON)
            buffer = b''
            while b'\n' not in buffer:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                buffer += chunk
            
            if not buffer:
                self.logger.warning(f"Client {client_address} disconnected without sending data")
                return
            
            # Parse request
            request_str = buffer.decode('utf-8').strip()
            request = json.loads(request_str)
            
            command = request.get('command')
            self.logger.info(f"Request from {client_address}: {command}")
            
            # Process command and track stats
            self.persistent_stats.increment_both('requests_total')
            response = self._process_command(request)
            
            if response['status'] == 'success':
                self.persistent_stats.increment_both('requests_success')
            else:
                self.persistent_stats.increment_both('requests_failed')
            
            # Send response
            response_str = json.dumps(response) + '\n'
            client_socket.sendall(response_str.encode('utf-8'))
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON from {client_address}: {e}")
            error_response = {
                'status': 'error',
                'error': f'Invalid JSON: {str(e)}',
                'error_type': 'JSONDecodeError'
            }
            try:
                client_socket.sendall((json.dumps(error_response) + '\n').encode('utf-8'))
            except Exception as send_error:
                self.logger.error(f"Failed to send error response: {send_error}")
        
        except Exception as e:
            self.logger.error(f"Error handling client {client_address}: {e}")
            self.logger.debug(traceback.format_exc())
            
            # CRITICAL: Always send an error response to prevent client hanging
            error_response = {
                'status': 'error',
                'error': f'Internal server error: {str(e)}',
                'error_type': type(e).__name__
            }
            try:
                # Only send if we haven't sent a response yet
                if response is None:
                    client_socket.sendall((json.dumps(error_response) + '\n').encode('utf-8'))
            except Exception as send_error:
                self.logger.error(f"Failed to send error response: {send_error}")
        
        finally:
            try:
                client_socket.close()
            except Exception as close_error:
                self.logger.debug(f"Error closing socket: {close_error}")
    
    def _process_command(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a command request.
        
        Args:
            request: Request dictionary with 'command' field
        
        Returns:
            Response dictionary (always includes 'status' field)
        """
        command = request.get('command', '').lower()
        
        try:
            if command == 'status':
                return self._cmd_status(request)
            
            elif command == 'initialize':
                return self._cmd_initialize(request)
            
            elif command == 'execute_pick':
                return self._cmd_execute_pick(request)
            
            elif command == 'execute_multi_berry':
                return self._cmd_execute_multi_berry(request)
            
            elif command == 'shutdown':
                return self._cmd_shutdown(request)
            
            elif command == 'reload_config':
                return self._cmd_reload_config(request)
            
            elif command == 'stats':
                return self._cmd_stats(request)
            
            else:
                return {
                    'status': 'error',
                    'error': f'Unknown command: {command}',
                    'error_type': 'UnknownCommand'
                }
        
        except Exception as e:
            self.logger.error(f"Command processing error: {e}")
            self.logger.debug(traceback.format_exc())
            
            # Return structured error response
            return {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__,
                'command': command
            }
    
    def _cmd_status(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get controller status."""
        status = self.controller.get_status() if self.controller else {}
        stats = self.persistent_stats.get_stats()
        
        return {
            'status': 'success',
            'data': {
                'service': {
                    'running': self.running,
                    'host': self.host,
                    'port': self.port
                },
                'controller': status,
                'stats': {
                    'session': stats['current_session'],
                    'lifetime': stats['lifetime']
                }
            }
        }
    
    def _cmd_initialize(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize controller."""
        if not self.controller:
            return {'status': 'error', 'error': 'Controller not created'}
        
        success = self.controller.initialize()
        
        return {
            'status': 'success' if success else 'error',
            'data': {'initialized': success},
            'error': 'Initialization failed' if not success else None
        }
    
    def _cmd_execute_pick(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pick sequence."""
        if not self.controller or not self.controller.is_initialized:
            return {
                'status': 'error',
                'error': 'Controller not initialized',
                'error_type': 'ControllerNotInitialized'
            }
        
        berry_index = request.get('berry_index', 0)
        json_path = request.get('json_path', None)  # Optional JSON path for offline mode
        
        try:
            # Track pick attempt
            self.persistent_stats.increment_both('picks_total')
            
            # Execute the pick sequence
            success = self.controller.execute_pick_sequence(berry_index=berry_index, json_path=json_path)
            
            if success:
                self.persistent_stats.increment_both('picks_success')
            else:
                self.persistent_stats.increment_both('picks_failed')
            
            return {
                'status': 'success' if success else 'error',
                'data': {
                    'berry_index': berry_index,
                    'completed': success
                },
                'error': 'Pick sequence failed' if not success else None
            }
        
        except Exception as e:
            self.logger.error(f"Pick execution error: {e}")
            self.logger.debug(traceback.format_exc())
            self.persistent_stats.increment_both('picks_failed')
            return {
                'status': 'error',
                'error': f'Pick execution failed: {str(e)}',
                'error_type': type(e).__name__,
                'data': {
                    'berry_index': berry_index,
                    'completed': False
                }
            }
    
    def _cmd_execute_multi_berry(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-berry picking sequence."""
        if not self.controller or not self.controller.is_initialized:
            return {
                'status': 'error',
                'error': 'Controller not initialized',
                'error_type': 'ControllerNotInitialized'
            }
        
        json_path = request.get('json_path', None)  # Optional JSON path for offline mode
        
        try:
            # Track multi-berry run attempt
            self.persistent_stats.increment_both('multi_berry_runs')
            
            # Execute the multi-berry sequence
            success = self.controller.execute_multi_berry_sequence(json_path=json_path)
            
            if success:
                self.logger.info("[OK] Multi-berry sequence completed")
            else:
                self.logger.error("Multi-berry sequence failed")
            
            return {
                'status': 'success' if success else 'error',
                'data': {
                    'completed': success
                },
                'error': 'Multi-berry sequence failed' if not success else None
            }
        
        except Exception as e:
            self.logger.error(f"Multi-berry execution error: {e}")
            self.logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'error': f'Multi-berry execution failed: {str(e)}',
                'error_type': type(e).__name__,
                'data': {
                    'completed': False
                }
            }
    
    def _cmd_shutdown(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Shutdown service."""
        self.logger.info("Shutdown command received")
        
        # Stop in separate thread to allow response to be sent
        threading.Thread(target=self.stop, daemon=True).start()
        
        return {
            'status': 'success',
            'data': {'message': 'Shutdown initiated'}
        }
    
    def _cmd_reload_config(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Reload configuration."""
        try:
            self.config.reload()
            return {
                'status': 'success',
                'data': {'message': 'Configuration reloaded'}
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Config reload failed: {e}'
            }
    
    def _cmd_stats(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            'status': 'success',
            'data': self.persistent_stats.get_stats()
        }
    
    def stop(self):
        """Stop service gracefully."""
        if not self.running:
            return
        
        self.logger.info("Stopping service...")
        self.running = False
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Wait for client threads
        for thread in self.client_threads:
            thread.join(timeout=2.0)
        
        # Shutdown controller
        if self.controller:
            self.controller.shutdown()
        
        self.logger.info("[OK] Service stopped")


# Command-line entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Robot PnP Service - Always-On IPC Server")
    parser.add_argument('--config', type=str, 
                       default='pickafresa_robot/configs/robot_pnp_config.yaml',
                       help='Path to config YAML')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Bind address')
    parser.add_argument('--port', type=int, default=5556,
                       help='Service port')
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    
    if not config_path.exists():
        print(f"[FAIL] Config file not found: {config_path}")
        sys.exit(1)
    
    # Create and start service
    service = RobotPnPService(
        config_path=config_path,
        host=args.host,
        port=args.port
    )
    
    if service.start():
        print("\n[OK] Service running. Press Ctrl+C to stop.\n")
    else:
        print("\n[FAIL] Service failed to start\n")
        sys.exit(1)
