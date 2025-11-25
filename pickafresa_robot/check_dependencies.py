#!/usr/bin/env python3
"""
Robot PnP System - Dependency Checker and Installer

Checks for required and optional dependencies, offers to install missing ones.

by: Aldrick T, 2025
for Team YEA
"""

import sys
import subprocess
from pathlib import Path

# ANSI colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        return True, package_name
    except ImportError:
        return False, package_name

def print_header(text):
    """Print section header."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{text:^70}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

def print_status(name, available, optional=False):
    """Print dependency status."""
    if available:
        print(f"  {GREEN}[OK]{RESET} {name:30} {'(optional)' if optional else ''}")
    else:
        status = f"{YELLOW}[EMPTY]{RESET}" if optional else f"{RED}[FAIL]{RESET}"
        print(f"  {status} {name:30} {'(optional - not installed)' if optional else '(REQUIRED - not installed)'}")

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main dependency checker."""
    print_header("ROBOT PnP SYSTEM - DEPENDENCY CHECKER")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print(f"{RED}[FAIL] Python 3.8+ required{RESET}")
        sys.exit(1)
    else:
        print(f"{GREEN}[OK] Python version OK{RESET}")
    
    # Check required dependencies
    print_header("REQUIRED DEPENDENCIES")
    
    required = [
        ('numpy', 'numpy'),
        ('cv2', 'opencv-python'),
        ('yaml', 'pyyaml'),
        ('robodk.robolink', 'robodk'),
    ]
    
    missing_required = []
    for module, package in required:
        available, pkg = check_import(module, package)
        print_status(package, available, optional=False)
        if not available:
            missing_required.append(pkg)
    
    # Check optional dependencies
    print_header("OPTIONAL DEPENDENCIES")
    
    optional = [
        ('paho.mqtt.client', 'paho-mqtt', 'For robot_pnp_remote (MQTT bridge)'),
        ('supabase', 'supabase', 'For cloud logging'),
    ]
    
    missing_optional = []
    for module, package, desc in optional:
        available, pkg = check_import(module, package)
        print_status(f"{package:20} - {desc}", available, optional=True)
        if not available:
            missing_optional.append(pkg)
    
    # Summary
    print_header("SUMMARY")
    
    if not missing_required and not missing_optional:
        print(f"{GREEN}[OK] All dependencies installed!{RESET}\n")
        return
    
    if missing_required:
        print(f"{RED}Missing required packages: {', '.join(missing_required)}{RESET}")
    
    if missing_optional:
        print(f"{YELLOW}Missing optional packages: {', '.join(missing_optional)}{RESET}")
    
    # Offer to install
    print()
    install_all = missing_required + missing_optional
    
    if install_all:
        print(f"Would you like to install missing packages? ({len(install_all)} total)")
        print(f"Command: pip install {' '.join(install_all)}")
        
        choice = input("\nInstall now? [Y/n]: ").strip().lower()
        
        if choice != 'n':
            print(f"\n{BLUE}Installing packages...{RESET}\n")
            
            for package in install_all:
                print(f"Installing {package}...")
                if install_package(package):
                    print(f"{GREEN}[OK] {package} installed{RESET}")
                else:
                    print(f"{RED}[FAIL] Failed to install {package}{RESET}")
            
            print(f"\n{GREEN}[OK] Installation complete!{RESET}")
            print(f"Run this script again to verify.\n")
        else:
            print(f"\n{YELLOW}Skipped installation{RESET}")
            print(f"To install manually: pip install {' '.join(install_all)}\n")
    
    # Platform-specific notes
    print_header("PLATFORM-SPECIFIC NOTES")
    
    if sys.platform == 'darwin':
        print(f"{YELLOW}macOS detected:{RESET}")
        print("  - RealSense requires sudo access")
        print("  - Use ./realsense_venv_sudo script for vision tools")
        print("  - Keyboard library disabled (requires root)")
    elif sys.platform == 'win32':
        print(f"{YELLOW}Windows detected:{RESET}")
        print("  - Use requirements_win.txt for dependencies")
        print("  - Standard python commands work (no sudo needed)")
    
    print()
    
    # Additional requirements
    print_header("ADDITIONAL REQUIREMENTS")
    
    print("System Requirements:")
    print("  - RoboDK (simulation or real robot)")
    print("  - Intel RealSense D435 (for vision)")
    print("  - MQTT broker (for remote control)")
    print()
    
    # Documentation
    print_header("NEXT STEPS")
    
    print("Documentation:")
    print("  - README_REFACTORED.md  - Quick start guide")
    print("  - ARCHITECTURE.md        - Detailed system design")
    print("  - robot_system/README.md - Existing system docs")
    print()
    
    print("Quick Start:")
    print("  1. Start vision service:")
    print("     ./realsense_venv_sudo pickafresa_vision/vision_nodes/vision_service.py")
    print()
    print("  2. Start robot service:")
    print("     python pickafresa_robot/robot_system/robot_pnp_service.py")
    print()
    print("  3. Use manager for control:")
    print("     python pickafresa_robot/robot_system/robot_pnp_manager.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Interrupted{RESET}")
        sys.exit(0)
