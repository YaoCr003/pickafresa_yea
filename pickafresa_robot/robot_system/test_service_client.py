"""
Robot PnP Service Test Client

Simple test client for validating the robot_pnp_service IPC interface.
Sends various commands and displays responses.

by: Aldrick T, 2025
for Team YEA
"""

import sys
import json
import socket
from pathlib import Path
from typing import Dict, Any

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]


class ServiceTestClient:
    """Test client for robot_pnp_service."""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 5556, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
    
    def send_command(self, command: str, **params) -> Dict[str, Any]:
        """
        Send a command to the service.
        
        Args:
            command: Command name
            **params: Additional parameters
        
        Returns:
            Response dictionary
        """
        request = {'command': command, **params}
        
        try:
            # Connect
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            
            # Send request
            request_str = json.dumps(request) + '\n'
            sock.sendall(request_str.encode('utf-8'))
            
            # Read response
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
            return {
                'status': 'error',
                'error': f'Client error: {e}'
            }
    
    def test_status(self):
        """Test status command."""
        print("\n[1] Testing status command...")
        response = self.send_command('status')
        
        if response['status'] == 'success':
            print("[OK] Status command successful")
            service_info = response['data']['service']
            controller_info = response['data']['controller']
            stats = response['data']['stats']
            
            print(f"  Service running: {service_info['running']}")
            print(f"  Controller state: {controller_info.get('state', 'N/A')}")
            print(f"  Controller initialized: {controller_info.get('initialized', False)}")
            print(f"  Operational: {controller_info.get('operational', False)}")
            print(f"  Requests: {stats['requests_total']} (success: {stats['requests_success']}, failed: {stats['requests_failed']})")
            return True
        else:
            print(f"[FAIL] Status command failed: {response.get('error')}")
            return False
    
    def test_initialize(self):
        """Test initialize command."""
        print("\n[2] Testing initialize command...")
        response = self.send_command('initialize')
        
        if response['status'] == 'success':
            print("[OK] Initialize command successful")
            print(f"  Initialized: {response['data']['initialized']}")
            return True
        else:
            print(f"[FAIL] Initialize command failed: {response.get('error')}")
            return False
    
    def test_stats(self):
        """Test stats command."""
        print("\n[3] Testing stats command...")
        response = self.send_command('stats')
        
        if response['status'] == 'success':
            print("[OK] Stats command successful")
            stats = response['data']
            print(f"  Total requests: {stats['requests_total']}")
            print(f"  Success: {stats['requests_success']}")
            print(f"  Failed: {stats['requests_failed']}")
            print(f"  Total picks: {stats['picks_total']}")
            print(f"  Pick success: {stats['picks_success']}")
            print(f"  Pick failed: {stats['picks_failed']}")
            return True
        else:
            print(f"[FAIL] Stats command failed: {response.get('error')}")
            return False
    
    def test_reload_config(self):
        """Test reload_config command."""
        print("\n[4] Testing reload_config command...")
        response = self.send_command('reload_config')
        
        if response['status'] == 'success':
            print("[OK] Reload config command successful")
            print(f"  Message: {response['data']['message']}")
            return True
        else:
            print(f"[FAIL] Reload config command failed: {response.get('error')}")
            return False
    
    def test_execute_pick(self, berry_index: int = 0):
        """Test execute_pick command (requires full system)."""
        print(f"\n[5] Testing execute_pick command (berry_index={berry_index})...")
        print("    NOTE: This requires RoboDK and vision service running")
        
        response = self.send_command('execute_pick', berry_index=berry_index)
        
        if response['status'] == 'success':
            print("[OK] Execute pick command successful")
            print(f"  Berry index: {response['data']['berry_index']}")
            print(f"  Completed: {response['data']['completed']}")
            return True
        else:
            print(f"[FAIL] Execute pick command failed: {response.get('error')}")
            return False
    
    def test_unknown_command(self):
        """Test unknown command handling."""
        print("\n[6] Testing unknown command...")
        response = self.send_command('invalid_command_xyz')
        
        if response['status'] == 'error':
            print("[OK] Unknown command correctly rejected")
            print(f"  Error: {response.get('error')}")
            return True
        else:
            print("[FAIL] Unknown command should have returned error")
            return False


def main():
    """Run test suite."""
    print("=" * 80)
    print("Robot PnP Service - Test Client")
    print("=" * 80)
    print("\nNOTE: Make sure robot_pnp_service.py is running before testing")
    print("      Start service: python pickafresa_robot/robot_system/robot_pnp_service.py")
    
    input("\nPress Enter to continue...")
    
    # Create client
    client = ServiceTestClient(host='127.0.0.1', port=5556)
    
    results = []
    
    # Run tests
    results.append(("Status", client.test_status()))
    results.append(("Stats", client.test_stats()))
    results.append(("Reload Config", client.test_reload_config()))
    results.append(("Unknown Command", client.test_unknown_command()))
    
    # Optional: Initialize test (requires RoboDK)
    print("\n" + "=" * 80)
    print("OPTIONAL TESTS (require full system)")
    print("=" * 80)
    
    user_input = input("\nTest initialize command? (requires RoboDK) [y/N]: ")
    if user_input.lower() == 'y':
        results.append(("Initialize", client.test_initialize()))
    
    # user_input = input("\nTest execute_pick command? (requires RoboDK + vision service) [y/N]: ")
    # if user_input.lower() == 'y':
    #     results.append(("Execute Pick", client.test_execute_pick(berry_index=0)))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED")
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
