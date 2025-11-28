# !/bin/bash
# RealSense firmware update script for macOS
# Uses the rs-fw-update tool to list and update firmware
# @aldrick-t, 2025

set -euo pipefail

# 1) Free the interfaces macOS likes to grab
sudo /usr/bin/killall -9 VDCAssistant 2>/dev/null || true
sudo /usr/bin/killall -9 AppleCameraAssistant 2>/dev/null || true
sleep 1

# 2) Quick USB presence check
if ! /usr/sbin/ioreg -p IOUSB -l | grep -qi "idVendor = 0x8086" ; then
  echo "RealSense not seen on USB (vendor 0x8086). Plug it in or use a powered USB3 hub."
  exit 1
fi

# 3) List devices (the tool may print a 'mutex lock failed' warning; that’s ok)
sudo rs-fw-update -l || true

# 4) Optional: pass a firmware .bin path to update
if [[ $# -ge 1 ]]; then
  FW="$1"
  if [[ ! -f "$FW" ]]; then echo "Firmware file not found: $FW"; exit 2; fi
  echo "Updating firmware from: $FW"
  sudo rs-fw-update -f "$FW"
fi