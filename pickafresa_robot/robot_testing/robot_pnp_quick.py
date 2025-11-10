#!/usr/bin/env python3
"""
Quick Start Script for Robot PnP Testing Tool

This script provides a simplified entry point with common configurations.

Usage:
    ./robot_pnp_quick.py [--mode MODE]

Modes:
    - sim-json: Simulation with JSON data (default, safest)
    - sim-api: Simulation with live camera
    - real-json: Real robot with JSON data (requires confirmation)
    - real-api: Real robot with live camera (requires confirmation)

# @aldrick-t, 2025
"""

import sys
import argparse
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pickafresa_robot.robot_testing.robot_pnp_cli import RobotPnPCLI


def main():
    parser = argparse.ArgumentParser(description="Quick Start - Robot PnP Testing Tool")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sim-json', 'sim-api', 'real-json', 'real-api'],
        default='sim-json',
        help="Operational mode (default: sim-json)"
    )
    
    args = parser.parse_args()
    
    # Configuration based on mode
    mode_configs = {
        'sim-json': {
            'name': 'Simulation with JSON data',
            'config': REPO_ROOT / 'pickafresa_robot/configs/robot_pnp_config.yaml',
            'safe': True
        },
        'sim-api': {
            'name': 'Simulation with live camera',
            'config': REPO_ROOT / 'pickafresa_robot/configs/robot_pnp_config.yaml',
            'safe': True,
            'modify': {'pnp_data': {'source_mode': 'api'}}
        },
        'real-json': {
            'name': 'REAL ROBOT with JSON data',
            'config': REPO_ROOT / 'pickafresa_robot/configs/robot_pnp_config.yaml',
            'safe': False,
            'modify': {'robodk': {'run_mode': 'real_robot'}}
        },
        'real-api': {
            'name': 'REAL ROBOT with live camera',
            'config': REPO_ROOT / 'pickafresa_robot/configs/robot_pnp_config.yaml',
            'safe': False,
            'modify': {
                'robodk': {'run_mode': 'real_robot'},
                'pnp_data': {'source_mode': 'api'}
            }
        }
    }
    
    mode_info = mode_configs[args.mode]
    
    # Display mode information
    print("\nQUICK START MODE: " + mode_info['name'])
    
    # Safety confirmation for real robot modes
    if not mode_info['safe']:
        print("\n[WARNING] This mode will control the REAL ROBOT!")
        print("   Ensure:")
        print("   - Robot is properly connected and powered")
        print("   - Workspace is clear of obstacles")
        print("   - Emergency stop is accessible")
        print("   - All safety protocols are followed")
        print()
        
        response = input("Are you SURE you want to continue? Type 'YES' to proceed: ").strip()
        
        if response != 'YES':
            print("\n[ERROR] Operation cancelled for safety.")
            return 1
    
    # Run the tool
    app = RobotPnPCLI(config_path=mode_info['config'])
    exit_code = app.run()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
