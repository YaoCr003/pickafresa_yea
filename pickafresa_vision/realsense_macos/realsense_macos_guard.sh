#!/usr/bin/env bash
# RealSense guard script for macOS
# Periodically kills macOS camera processes if a RealSense D435 is connected
# @aldrick-t, 2025

# MacOS only guard
set -euo pipefail
INTERVAL="${1:-3}"

while true; do
  if /usr/sbin/ioreg -p IOUSB -l | grep -qi "idVendor = 0x8086" && \
     /usr/sbin/ioreg -p IOUSB -l | grep -qi "idProduct = 0x0b07"; then
     /usr/bin/sudo -n /usr/bin/killall -9 VDCAssistant AppleCameraAssistant 2>/dev/null || true
  fi
  sleep "$INTERVAL"
done