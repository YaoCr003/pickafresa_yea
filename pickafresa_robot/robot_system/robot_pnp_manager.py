"""
Robot PnP Manager - Local CLI Admin Interface

Interactive menu-driven CLI for managing the robot_pnp_service.
Provides local administration, monitoring, and control.

Features:
- Service status monitoring
- Execute pick commands
- View logs (tail -f style)
- Edit configuration (hot-reloadable params)
- View statistics
- Emergency stop

by: Aldrick T, 2025
for Team YEA
"""

import sys
import os
import json
import socket
import time
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

# Repository root setup
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pickafresa_robot.robot_system.config_manager import ConfigManager


class ServiceClient:
    """Lightweight client for communicating with robot_pnp_service."""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 5556, timeout: float = 180.0):
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
        
        except socket.timeout:
            return {'status': 'error', 'error': 'Connection timeout'}
        except ConnectionRefusedError:
            return {'status': 'error', 'error': 'Service not running'}
        except Exception as e:
            return {'status': 'error', 'error': f'Client error: {e}'}


class RobotPnPManager:
    """
    Local CLI admin interface for robot_pnp_service.
    """
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = ConfigManager(config_path)
        
        # Service connection
        service_config = self.config.get('service', {})
        host = service_config.get('host', '127.0.0.1')
        port = service_config.get('port', 5556)
        
        self.client = ServiceClient(host=host, port=port)
        
        self.running = True
    
    def run(self):
        """Run interactive menu."""
        self._print_header()
        
        while self.running:
            self._print_menu()
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                self._view_status()
            elif choice == '2':
                self._view_stats()
            elif choice == '3':
                self._execute_pick()
            elif choice == '4':
                self._execute_multi_berry()
            elif choice == '5':
                self._view_logs()
            elif choice == '6':
                self._edit_config()
            elif choice == '7':
                self._reload_config()
            elif choice == '8':
                self._emergency_stop()
            elif choice == '9':
                self._shutdown_service()
            elif choice == '0' or choice.lower() == 'q':
                self._quit()
            else:
                print("\n[FAIL] Invalid option")
            
            if self.running and choice not in ['5']:  # Don't pause after logs
                input("\nPress Enter to continue...")
    
    def _print_header(self):
        """Print header."""
        print("\n" + "=" * 80)
        print(" " * 25 + "ROBOT PnP MANAGER")
        print(" " * 20 + "Local CLI Admin Interface")
        print("=" * 80)
        print(f"Config: {self.config_path}")
        print(f"Service: {self.client.host}:{self.client.port}")
        print("=" * 80)
    
    def _print_menu(self):
        """Print menu options."""
        print("\n" + "-" * 80)
        print("MENU:")
        print("  [1] View Status        - Check service and controller state")
        print("  [2] View Statistics    - Request/pick statistics")
        print("  [3] Execute Pick       - Run single berry pick sequence")
        print("  [4] Execute Multi-Berry - Run multi-berry pick sequence")
        print("  [5] View Logs          - Tail service logs")
        print("  [6] Edit Config        - Modify hot-reloadable parameters")
        print("  [7] Reload Config      - Force config reload")
        print("  [8] Emergency Stop     - Halt operations immediately")
        print("  [9] Shutdown Service   - Stop robot_pnp_service")
        print("  [0] Quit Manager       - Exit (service keeps running)")
        print("-" * 80)
    
    def _view_status(self):
        """View service and controller status."""
        print("\n" + "-" * 80)
        print("SERVICE STATUS")
        print("-" * 80)
        
        response = self.client.send_command('status')
        
        if response['status'] == 'success':
            data = response['data']
            
            # Service info
            service = data['service']
            print(f"\nService:")
            print(f"  Running: {service['running']}")
            print(f"  Address: {service['host']}:{service['port']}")
            
            # Controller info
            controller = data['controller']
            print(f"\nController:")
            print(f"  Initialized: {controller.get('initialized', False)}")
            print(f"  State: {controller.get('state', 'N/A')}")
            print(f"  Operational: {controller.get('operational', False)}")
            print(f"  Busy: {controller.get('busy', False)}")
            print(f"  Error: {controller.get('error', False)}")
            
            # Connections
            print(f"\nConnections:")
            print(f"  RoboDK: {'[OK] Connected' if controller.get('robodk_connected') else '[FAIL] Disconnected'}")
            print(f"  Vision: {'[OK] Connected' if controller.get('vision_connected') else '[FAIL] Disconnected'}")
            print(f"  Gripper: {'[OK] Connected' if controller.get('gripper_connected') else '[FAIL] Disconnected'}")
            
            # Stats (show session stats)
            stats = data['stats']
            session = stats.get('session', stats)  # Fallback to old format if needed
            print(f"\nSession Statistics:")
            print(f"  Requests: {session['requests_total']} (success: {session['requests_success']}, failed: {session['requests_failed']})")
            print(f"  Picks: {session['picks_total']} (success: {session['picks_success']}, failed: {session['picks_failed']})")
        else:
            # Check if this is a connection error (service not running)
            error_msg = response.get('error', '')
            if 'not running' in error_msg.lower() or 'connection refused' in error_msg.lower() or 'timed out' in error_msg.lower():
                print("\n[INFO] Service is not running")
                print("\n   To start the service:")
                print("   $ python pickafresa_robot/robot_system/robot_pnp_service.py")
                print("\n   Or with config:")
                print("   $ python pickafresa_robot/robot_system/robot_pnp_service.py --config <path>")
            else:
                print(f"[FAIL] Error: {error_msg}")
    
    def _view_stats(self):
        """View detailed statistics."""
        print("\n" + "-" * 80)
        print("STATISTICS")
        print("-" * 80)
        
        response = self.client.send_command('stats')
        
        if response['status'] == 'success':
            stats = response['data']
            
            # Check if we have new format (session + lifetime) or old format
            if 'session' in stats and 'lifetime' in stats:
                # New format with persistent stats
                session = stats['session']
                lifetime = stats['lifetime']
                
                # Session statistics
                print("\n[STATS] Current Session Statistics:")
                print(f"  Started: {session.get('session_started', 'N/A')}")
                print(f"\n  Requests:")
                print(f"    Total: {session['requests_total']}")
                print(f"    Success: {session['requests_success']}")
                print(f"    Failed: {session['requests_failed']}")
                
                if session['requests_total'] > 0:
                    success_rate = (session['requests_success'] / session['requests_total']) * 100
                    print(f"    Success Rate: {success_rate:.1f}%")
                
                print(f"\n  Pick Operations:")
                print(f"    Total: {session['picks_total']}")
                print(f"    Success: {session['picks_success']}")
                print(f"    Failed: {session['picks_failed']}")
                
                if session['picks_total'] > 0:
                    success_rate = (session['picks_success'] / session['picks_total']) * 100
                    print(f"    Success Rate: {success_rate:.1f}%")
                
                # Lifetime statistics
                print("\n" + "-" * 80)
                print("[TROPHY] Lifetime Statistics (All Time):")
                print(f"  Service Starts: {lifetime.get('service_starts', 0)}")
                print(f"  First Started: {lifetime.get('first_started', 'N/A')}")
                print(f"  Last Started: {lifetime.get('last_started', 'N/A')}")
                print(f"\n  Requests:")
                print(f"    Total: {lifetime['requests_total']}")
                print(f"    Success: {lifetime['requests_success']}")
                print(f"    Failed: {lifetime['requests_failed']}")
                
                if lifetime['requests_total'] > 0:
                    success_rate = (lifetime['requests_success'] / lifetime['requests_total']) * 100
                    print(f"    Success Rate: {success_rate:.1f}%")
                
                print(f"\n  Pick Operations:")
                print(f"    Total: {lifetime['picks_total']}")
                print(f"    Success: {lifetime['picks_success']}")
                print(f"    Failed: {lifetime['picks_failed']}")
                
                if lifetime['picks_total'] > 0:
                    success_rate = (lifetime['picks_success'] / lifetime['picks_total']) * 100
                    print(f"    Success Rate: {success_rate:.1f}%")
            else:
                # Old format (fallback for backward compatibility)
                print(f"\nRequests:")
                print(f"  Total: {stats['requests_total']}")
                print(f"  Success: {stats['requests_success']}")
                print(f"  Failed: {stats['requests_failed']}")
                
                if stats['requests_total'] > 0:
                    success_rate = (stats['requests_success'] / stats['requests_total']) * 100
                    print(f"  Success Rate: {success_rate:.1f}%")
                
                print(f"\nPick Operations:")
                print(f"  Total: {stats['picks_total']}")
                print(f"  Success: {stats['picks_success']}")
                print(f"  Failed: {stats['picks_failed']}")
                
                if stats['picks_total'] > 0:
                    success_rate = (stats['picks_success'] / stats['picks_total']) * 100
                    print(f"  Success Rate: {success_rate:.1f}%")
        else:
            print(f"[FAIL] Error: {response.get('error')}")
    
    def _execute_pick(self):
        """Execute pick sequence."""
        print("\n" + "-" * 80)
        print("EXECUTE PICK")
        print("-" * 80)
        
        # Check if service is in offline mode
        status_response = self.client.send_command('status')
        offline_mode = False
        
        if status_response['status'] == 'success':
            controller_data = status_response['data'].get('controller', {})
            offline_mode = not controller_data.get('vision_connected', False)
        
        json_path = None
        
        # If offline mode, prompt for JSON file selection
        if offline_mode:
            print("\n[WARNING] Service running in OFFLINE MODE")
            print("Select JSON data file for pick sequence:\n")
            
            captures_dir = REPO_ROOT / "pickafresa_vision/captures"
            
            if captures_dir.exists():
                json_files = sorted(captures_dir.glob("*_data.json"), reverse=True)
                
                if json_files:
                    # Show latest file as default
                    latest_file = json_files[0]
                    print(f"  [ENTER] Use latest: {latest_file.name}")
                    print(f"\n  Recent captures:")
                    for idx, json_file in enumerate(json_files[:10], 1):
                        size_kb = json_file.stat().st_size / 1024
                        modified = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(json_file.stat().st_mtime))
                        print(f"    [{idx}] {json_file.name:<35} ({size_kb:>6.1f} KB, {modified})")
                    
                    print(f"\n  Select [1-{min(len(json_files), 10)}], ENTER for latest, or full path: ", end='')
                    choice = input().strip()
                    
                    if not choice:
                        # Use latest (default)
                        json_path = str(latest_file)
                        print(f"  -> Using: {latest_file.name}")
                    elif choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(json_files):
                            json_path = str(json_files[idx])
                            print(f"  -> Using: {json_files[idx].name}")
                        else:
                            print(f"  [FAIL] Invalid selection: {choice}")
                            return
                    else:
                        # Custom path
                        json_path = choice
                        print(f"  -> Using: {choice}")
                else:
                    print("  [FAIL] No JSON files found in captures directory")
                    print("  Enter full path to JSON file: ", end='')
                    json_path = input().strip()
                    if not json_path:
                        print("  Cancelled")
                        return
            else:
                print(f"  [FAIL] Captures directory not found: {captures_dir}")
                print("  Enter full path to JSON file: ", end='')
                json_path = input().strip()
                if not json_path:
                    print("  Cancelled")
                    return
            
            print()  # Blank line for readability
        
        # In service architecture, berry_index selects which detection to pick
        # For now, always pick first/best berry (index 0)
        # TODO: Implement multi-berry sequence or let user select after seeing detection count
        berry_index = 0
        
        confirm = input(f"\nExecute pick sequence? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Cancelled")
            return
        
        print(f"\n{'-' * 80}")
        print("EXECUTING PICK SEQUENCE...")
        print(f"{'-' * 80}")
        
        # Send command with optional json_path
        cmd_params = {'berry_index': berry_index}
        if json_path:
            cmd_params['json_path'] = json_path
        
        # Show execution progress
        print("\n[HOURGLASS] Sending command to service...")
        print("   (This may take 2-3 minutes for full pick sequence)")
        print("   Status updates:\n")
        
        response = self.client.send_command('execute_pick', **cmd_params)
        
        if response['status'] == 'success':
            print(f"\n{'-' * 80}")
            print("[OK] PICK SEQUENCE COMPLETED SUCCESSFULLY")
            print(f"{'-' * 80}")
            print(f"  Berry index: {response['data']['berry_index']}")
            print(f"  Completed: {response['data']['completed']}")
            if 'message' in response['data']:
                print(f"  Message: {response['data']['message']}")
        else:
            print(f"\n{'-' * 80}")
            print("[FAIL] PICK SEQUENCE FAILED")
            print(f"{'-' * 80}")
            error_msg = response.get('error', 'Unknown error')
            print(f"  Error: {error_msg}")
            if 'details' in response:
                print(f"  Details: {response['details']}")
    
    def _execute_multi_berry(self):
        """Execute multi-berry picking sequence."""
        print("\n" + "-" * 80)
        print("EXECUTE MULTI-BERRY SEQUENCE")
        print("-" * 80)
        
        # Check if service is in offline mode
        status_response = self.client.send_command('status')
        offline_mode = False
        
        if status_response['status'] == 'success':
            controller_data = status_response['data'].get('controller', {})
            offline_mode = not controller_data.get('vision_connected', False)
        
        json_path = None
        
        # If offline mode, prompt for JSON file selection
        if offline_mode:
            print("\n[WARNING] Service running in OFFLINE MODE")
            print("Select JSON data file for multi-berry sequence:\n")
            
            captures_dir = REPO_ROOT / "pickafresa_vision/captures"
            
            if captures_dir.exists():
                json_files = sorted(captures_dir.glob("*_data.json"), reverse=True)
                
                if json_files:
                    # Show latest file as default
                    latest_file = json_files[0]
                    print(f"  [ENTER] Use latest: {latest_file.name}")
                    print(f"\n  Recent captures:")
                    for idx, json_file in enumerate(json_files[:10], 1):
                        size_kb = json_file.stat().st_size / 1024
                        modified = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(json_file.stat().st_mtime))
                        print(f"    [{idx}] {json_file.name:<35} ({size_kb:>6.1f} KB, {modified})")
                    
                    print(f"\n  Select [1-{min(len(json_files), 10)}], ENTER for latest, or full path: ", end='')
                    choice = input().strip()
                    
                    if not choice:
                        # Use latest (default)
                        json_path = str(latest_file)
                        print(f"  -> Using: {latest_file.name}")
                    elif choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(json_files):
                            json_path = str(json_files[idx])
                            print(f"  -> Using: {json_files[idx].name}")
                        else:
                            print(f"  [FAIL] Invalid selection: {choice}")
                            return
                    else:
                        # Custom path
                        json_path = choice
                        print(f"  -> Using: {choice}")
                else:
                    print("  [FAIL] No JSON files found in captures directory")
                    print("  Enter full path to JSON file: ", end='')
                    json_path = input().strip()
                    if not json_path:
                        print("  Cancelled")
                        return
            else:
                print(f"  [FAIL] Captures directory not found: {captures_dir}")
                print("  Enter full path to JSON file: ", end='')
                json_path = input().strip()
                if not json_path:
                    print("  Cancelled")
                    return
            
            print()  # Blank line for readability
        
        # Get multi-berry config info
        multi_berry_config = self.config.get('multi_berry', {})
        max_berries = multi_berry_config.get('max_berries_per_run', 10)
        sort_by = multi_berry_config.get('sort_by', 'confidence')
        
        print(f"\nMulti-berry configuration:")
        print(f"  Max berries per run: {max_berries}")
        print(f"  Sort by: {sort_by}")
        
        confirm = input(f"\nExecute multi-berry sequence? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Cancelled")
            return
        
        print(f"\n{'-' * 80}")
        print("EXECUTING MULTI-BERRY SEQUENCE...")
        print(f"{'-' * 80}")
        
        # Send command with optional json_path
        cmd_params = {}
        if json_path:
            cmd_params['json_path'] = json_path
        
        # Show execution progress
        print("\n[HOURGLASS] Sending command to service...")
        print("   (This may take several minutes for multi-berry sequence)")
        print("   Status updates:\n")
        
        response = self.client.send_command('execute_multi_berry', **cmd_params)
        
        if response['status'] == 'success':
            print(f"\n{'-' * 80}")
            print("[OK] MULTI-BERRY SEQUENCE COMPLETED SUCCESSFULLY")
            print(f"{'-' * 80}")
            print(f"  Completed: {response['data']['completed']}")
            if 'message' in response['data']:
                print(f"  Message: {response['data']['message']}")
        else:
            print(f"\n{'-' * 80}")
            print("[FAIL] MULTI-BERRY SEQUENCE FAILED")
            print(f"{'-' * 80}")
            error_msg = response.get('error', 'Unknown error')
            print(f"  Error: {error_msg}")
            if 'details' in response:
                print(f"  Details: {response['details']}")
    
    def _view_logs(self):
        """View service logs (tail -f style)."""
        print("\n" + "-" * 80)
        print("SERVICE LOGS")
        print("-" * 80)
        
        log_dir = REPO_ROOT / "pickafresa_robot/logs"
        
        if not log_dir.exists():
            print(f"\n[FAIL] Log directory not found: {log_dir}")
            return
        
        # Find all .log files
        log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not log_files:
            print(f"\n[FAIL] No log files found in: {log_dir}")
            return
        
        # Show menu if multiple files
        if len(log_files) > 1:
            print("\nAvailable log files:")
            for idx, log_file in enumerate(log_files, 1):
                size_kb = log_file.stat().st_size / 1024
                modified = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log_file.stat().st_mtime))
                print(f"  [{idx}] {log_file.name:<30} ({size_kb:>7.1f} KB, modified: {modified})")
            
            print(f"\nSelect log file [1-{len(log_files)}] or press Enter for most recent: ", end='')
            choice = input().strip()
            
            if choice:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(log_files):
                        log_file = log_files[idx]
                    else:
                        print(f"[FAIL] Invalid selection. Using most recent.")
                        log_file = log_files[0]
                except ValueError:
                    print(f"[FAIL] Invalid input. Using most recent.")
                    log_file = log_files[0]
            else:
                log_file = log_files[0]
        else:
            log_file = log_files[0]
        
        print(f"\n[FILE] Viewing: {log_file.name}")
        print("-" * 80)
        print("(Press Ctrl+C to stop)\n")
        
        if not log_file.exists():
            print(f"\n[FAIL] Log file not found: {log_file}")
            return
        
        try:
            # Use tail -f to follow logs
            subprocess.run(['tail', '-f', str(log_file)])
        except KeyboardInterrupt:
            print("\n\nStopped viewing logs")
        except FileNotFoundError:
            # Fallback: Read last 50 lines
            print("\n(tail command not available, showing last 50 lines)\n")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    print(line, end='')
    
    def _edit_config(self):
        """Edit hot-reloadable config parameters."""
        print("\n" + "-" * 80)
        print("EDIT CONFIGURATION")
        print("-" * 80)
        
        print("\nHot-reloadable parameters:")
        hot_reload_keys = [
            'run_mode',
            'transforms.pick_offset.prepick',
            'transforms.pick_offset.pick',
            'transforms.pick_offset.place',
            'post_pick.enabled',
            'mqtt.enabled',
            'vision_service.multi_frame_enabled',
            'multi_berry.enabled'
        ]
        
        for i, key in enumerate(hot_reload_keys, 1):
            value = self.config.get(key)
            print(f"  [{i}] {key}: {value}")
        
        print("\nNOTE: Editing opens config file in default editor")
        print("      Changes take effect after saving and reloading")
        
        choice = input("\nOpen config file for editing? [y/N]: ").strip().lower()
        if choice == 'y':
            # Open in default editor
            editor = os.environ.get('EDITOR', 'nano')
            subprocess.run([editor, str(self.config_path)])
            
            # Prompt to reload
            reload_choice = input("\nReload configuration now? [Y/n]: ").strip().lower()
            if reload_choice != 'n':
                self._reload_config()
    
    def _reload_config(self):
        """Reload configuration."""
        print("\n" + "-" * 80)
        print("RELOAD CONFIGURATION")
        print("-" * 80)
        
        response = self.client.send_command('reload_config')
        
        if response['status'] == 'success':
            print("[OK] Configuration reloaded")
            print(f"  {response['data']['message']}")
            
            # Reload local config too
            self.config.reload()
        else:
            print(f"[FAIL] Error: {response.get('error')}")
    
    def _emergency_stop(self):
        """Emergency stop."""
        print("\n" + "-" * 80)
        print("EMERGENCY STOP")
        print("-" * 80)
        
        print("\n[WARNING] WARNING: This will halt all operations immediately!")
        confirm = input("Confirm emergency stop? [y/N]: ").strip().lower()
        
        if confirm != 'y':
            print("Cancelled")
            return
        
        # Send emergency stop (implementation depends on state machine)
        print("\n[EMERGENCY] EMERGENCY STOP TRIGGERED")
        print("(Implementation: Send to state machine or direct RoboDK stop)")
        
        # For now, just shutdown gracefully
        response = self.client.send_command('shutdown')
        
        if response['status'] == 'success':
            print("[OK] Service stopped")
        else:
            print(f"[FAIL] Error: {response.get('error')}")
    
    def _shutdown_service(self):
        """Shutdown service."""
        print("\n" + "-" * 80)
        print("SHUTDOWN SERVICE")
        print("-" * 80)
        
        confirm = input("\nShutdown robot_pnp_service? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Cancelled")
            return
        
        print("\nSending shutdown command...")
        
        response = self.client.send_command('shutdown')
        
        if response['status'] == 'success':
            print("[OK] Service shutdown initiated")
            print(f"  {response['data']['message']}")
        else:
            print(f"[FAIL] Error: {response.get('error')}")
    
    def _quit(self):
        """Quit manager."""
        print("\n" + "-" * 80)
        print("QUIT MANAGER")
        print("-" * 80)
        
        print("\nNOTE: Service will continue running in background")
        confirm = input("Exit manager? [Y/n]: ").strip().lower()
        
        if confirm != 'n':
            self.running = False
            print("\nGoodbye!")


# Command-line entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Robot PnP Manager - Local CLI Admin")
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
    
    # Create and run manager
    manager = RobotPnPManager(config_path=config_path)
    
    try:
        manager.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting...")
    except Exception as e:
        print(f"\n[FAIL] Manager error: {e}")
        import traceback
        traceback.print_exc()
