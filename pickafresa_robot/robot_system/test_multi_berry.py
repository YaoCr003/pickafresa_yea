#!/usr/bin/env python3
"""
Test Multi-Berry Picking

Quick test script to demonstrate multi-berry automation with the service.

Usage:
    python test_multi_berry.py --json /path/to/capture.json
    python test_multi_berry.py  # Uses vision service

by: Aldrick T, 2025
for Team YEA
"""

import sys
import json
import socket
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def send_command(command: dict, host: str = "localhost", port: int = 5556) -> dict:
    """Send command to robot service."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            
            # Send command
            message = json.dumps(command)
            sock.sendall(message.encode('utf-8'))
            
            # Receive response
            response_data = b''
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
            
            return json.loads(response_data.decode('utf-8'))
    
    except ConnectionRefusedError:
        print(f"[ERROR] Could not connect to service at {host}:{port}")
        print("   Is robot_pnp_service.py running?")
        return {'status': 'error', 'error': 'Connection refused'}
    
    except Exception as e:
        print(f"[ERROR] Communication error: {e}")
        return {'status': 'error', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Test multi-berry picking")
    parser.add_argument('--json', type=str, help='Path to JSON capture file (offline mode)')
    parser.add_argument('--host', type=str, default='localhost', help='Service host')
    parser.add_argument('--port', type=int, default=5556, help='Service port')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MULTI-BERRY PICKING TEST")
    print("=" * 60)
    
    # Check service status
    print("\n1. Checking service status...")
    status = send_command({'command': 'status'}, args.host, args.port)
    
    if status.get('status') != 'success':
        print("[ERROR] Service not ready")
        print(f"   Response: {status}")
        return 1
    
    print(f"[OK] Service ready (state: {status['data']['state']})")
    
    # Initialize if needed
    if not status['data']['initialized']:
        print("\n2. Initializing controller...")
        init_resp = send_command({'command': 'initialize'}, args.host, args.port)
        
        if init_resp.get('status') != 'success':
            print("[ERROR] Initialization failed")
            print(f"   Response: {init_resp}")
            return 1
        
        print("[OK] Controller initialized")
    else:
        print("\n2. Controller already initialized")
    
    # Execute multi-berry sequence
    print("\n3. Executing multi-berry sequence...")
    print(f"   Mode: {'Offline (JSON)' if args.json else 'Online (Vision)'}")
    
    if args.json:
        json_path = Path(args.json).resolve()
        if not json_path.exists():
            print(f"[ERROR] JSON file not found: {json_path}")
            return 1
        print(f"   JSON: {json_path}")
    
    command = {
        'command': 'execute_multi_berry'
    }
    
    if args.json:
        command['json_path'] = str(json_path)
    
    print("\n   [Starting multi-berry sequence...]")
    print("   (Check service logs for detailed progress)")
    
    response = send_command(command, args.host, args.port)
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    
    if response.get('status') == 'success':
        print("[OK] Multi-berry sequence COMPLETED")
        print(f"   Completed: {response['data']['completed']}")
    else:
        print("[ERROR] Multi-berry sequence FAILED")
        print(f"   Error: {response.get('error', 'Unknown error')}")
        return 1
    
    # Get statistics
    print("\n4. Retrieving statistics...")
    stats = send_command({'command': 'stats'}, args.host, args.port)
    
    if stats.get('status') == 'success':
        data = stats['data']
        print(f"   Total picks: {data.get('picks_total', 0)}")
        print(f"   Successful: {data.get('picks_success', 0)}")
        print(f"   Failed: {data.get('picks_failed', 0)}")
        print(f"   Multi-berry runs: {data.get('multi_berry_runs', 0)}")
        
        total = data.get('picks_total', 0)
        success = data.get('picks_success', 0)
        if total > 0:
            success_rate = 100.0 * success / total
            print(f"   Success rate: {success_rate:.1f}%")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
