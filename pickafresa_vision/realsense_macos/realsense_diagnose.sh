#!/usr/bin/env bash
# RealSense diagnostic script for macOS
# Checks for common issues and provides troubleshooting information
# @aldrick-t, 2025

set -euo pipefail

echo "=========================================="
echo "RealSense Camera Diagnostic (macOS)"
echo "=========================================="
echo ""

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Warning: Not running as root. Some checks may fail."
    echo "   Run with: sudo $0"
    echo ""
fi

# 1. Check USB connection
echo "1. Checking USB connection..."
if /usr/sbin/ioreg -p IOUSB -l | grep -qi "RealSense"; then
    echo "[OK] RealSense camera detected via USB"
    # Get detailed info
    /usr/sbin/ioreg -p IOUSB -l | grep -A 10 -i "RealSense" | grep -E "(idVendor|idProduct|USB Product Name)" || true
elif /usr/sbin/ioreg -p IOUSB -l | grep -qiE "idVendor = (0x8086|32902)"; then
    if /usr/sbin/ioreg -p IOUSB -l | grep -qiE "idProduct = (0x0b07|2823)"; then
        echo "[OK] Intel RealSense D435 detected (VID:PID 8086:0b07)"
    fi
else
    echo "[FAIL] RealSense camera NOT detected!"
    echo "  → Check USB cable connection"
    echo "  → Try a different USB port (use USB 3.0)"
    echo "  → Unplug and replug the camera"
fi
echo ""

# 2. Check for conflicting processes
echo "2. Checking for conflicting processes..."
CONFLICTS=0
for proc in VDCAssistant AppleCameraAssistant "com.apple.cmio.registerassistantservice"; do
    if pgrep -f "$proc" >/dev/null 2>&1; then
        echo "[WARNING]  Found running process: $proc"
        CONFLICTS=$((CONFLICTS + 1))
    fi
done

if [ $CONFLICTS -eq 0 ]; then
    echo "[OK] No conflicting camera processes found"
else
    echo ""
    echo "  → These macOS processes can interfere with RealSense"
    echo "  → To kill them: sudo killall -9 VDCAssistant AppleCameraAssistant"
fi
echo ""

# 3. Check realsense_guard status
echo "3. Checking realsense_guard service..."
if pgrep -f "realsense_guard" >/dev/null 2>&1; then
    echo "[OK] realsense_guard is running"
    pgrep -fl "realsense_guard" | head -3
else
    echo "[WARNING]  realsense_guard is NOT running"
    echo "  → This service prevents macOS from grabbing the camera"
    echo "  → Check: /usr/local/bin/realsense_guard.sh"
    echo "  → Check LaunchDaemon: ~/Library/LaunchAgents/com.*.realsense-guard.plist"
fi
echo ""

# 4. Check for Python processes using RealSense
echo "4. Checking for Python processes using RealSense..."
if ps aux | grep -i python | grep -iE "(realsense|bbox_depth|vision_nodes)" | grep -v grep >/dev/null 2>&1; then
    echo "[WARNING]  Found Python processes using RealSense:"
    ps aux | grep -i python | grep -iE "(realsense|bbox_depth|vision_nodes)" | grep -v grep
    echo ""
    echo "  → These may be holding the camera device"
    echo "  → Kill them before starting a new session"
else
    echo "[OK] No Python processes using RealSense"
fi
echo ""

# 5. Check file handles
echo "5. Checking open file handles to RealSense..."
if command -v lsof >/dev/null 2>&1; then
    RS_HANDLES=$(lsof 2>/dev/null | grep -i realsense | wc -l || echo "0")
    if [ "$RS_HANDLES" -gt 0 ]; then
        echo "[WARNING]  Found $RS_HANDLES open file handle(s) to RealSense"
        lsof 2>/dev/null | grep -i realsense | head -5 || true
    else
        echo "[OK] No open file handles to RealSense"
    fi
else
    echo "[WARNING]  lsof not available, skipping"
fi
echo ""

# 6. Recommendations
echo "=========================================="
echo "Recommendations:"
echo "=========================================="

if [ $CONFLICTS -gt 0 ]; then
    echo "1. Kill conflicting processes:"
    echo "   sudo killall -9 VDCAssistant AppleCameraAssistant"
    echo ""
fi

if ! pgrep -f "realsense_guard" >/dev/null 2>&1; then
    echo "2. Start realsense_guard service:"
    echo "   Check installation in realsense_macos/realsense_macos_guardinstall"
    echo ""
fi

echo "3. If camera is still not working:"
echo "   a. Unplug the RealSense camera"
echo "   b. Wait 5 seconds"
echo "   c. Plug it back in (use USB 3.0 port)"
echo "   d. Run this diagnostic again"
echo ""

echo "4. For persistent issues, try hardware reset:"
echo "   python3 -c 'import pyrealsense2 as rs; ctx = rs.context(); [d.hardware_reset() for d in ctx.query_devices()]'"
echo ""

echo "=========================================="
echo "Diagnostic complete"
echo "=========================================="
