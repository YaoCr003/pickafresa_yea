#!/usr/bin/env bash
# realsense_macos_guardinstall — Auto-installer for a background guard that
# auto-kills macOS camera agents when an Intel RealSense device (e.g., D435)
# is attached. Target: Apple Silicon (arm64) running macOS 26 "Tahoe".
# @aldrick-t, 2025

set -euo pipefail

# --- Env / detection ---------------------------------------------------------
arch="$(uname -m)"
if [[ "$arch" != "arm64" ]]; then
  echo "This installer supports Apple Silicon (arm64) Macs only. Detected: $arch" >&2
  exit 1
fi

product_ver="$(/usr/bin/sw_vers -productVersion)"
major="${product_ver%%.*}"
if [[ "$major" != "26" ]]; then
  echo "Warning: this installer is intended for macOS 26 (Tahoe). Detected: $product_ver" >&2
fi

USER_NAME="${SUDO_USER:-$USER}"
USER_HOME="$(/usr/bin/dscl . -read "/Users/$USER_NAME" NFSHomeDirectory 2>/dev/null | /usr/bin/awk '{print $2}')"
if [[ -z "${USER_HOME:-}" || ! -d "$USER_HOME" ]]; then
  USER_HOME="$HOME"
fi

PRIMARY_GROUP="$(/usr/bin/id -gn "$USER_NAME" 2>/dev/null || echo staff)"

LAUNCH_AGENTS_DIR="$USER_HOME/Library/LaunchAgents"
LABEL="com.${USER_NAME}.realsense-guard"
PLIST_PATH="$LAUNCH_AGENTS_DIR/${LABEL}.plist"
GUARD_PATH="/usr/local/bin/realsense_guard.sh"
SUDOERS_FILE="/etc/sudoers.d/realsense_guard"
INTERVAL="${REALSENSE_GUARD_INTERVAL:-3}"

# --- Optional uninstall mode -------------------------------------------------
if [[ "${1:-}" == "--uninstall" ]]; then
  UIDNUM="$(( $(/usr/bin/id -u "$USER_NAME") ))"
  /bin/launchctl bootout "gui/$UIDNUM/$LABEL" 2>/dev/null || true
  /bin/rm -f "$PLIST_PATH" 2>/dev/null || true
  /usr/bin/sudo /bin/rm -f "$GUARD_PATH" 2>/dev/null || true
  /usr/bin/sudo /bin/rm -f "$SUDOERS_FILE" 2>/dev/null || true
  echo "Uninstalled guard, LaunchAgent, and sudoers entry (if they existed)."
  exit 0
fi

# --- Create/install guard script --------------------------------------------
/usr/bin/sudo /usr/bin/install -d -m 755 /usr/local/bin
/bin/mkdir -p "$LAUNCH_AGENTS_DIR"

/usr/bin/sudo /usr/bin/tee "$GUARD_PATH" >/dev/null <<'GUARD'
#!/usr/bin/env bash
set -euo pipefail
INTERVAL="${1:-1}"

# Absolute paths (launchd does not inherit a full PATH)
IOREG="/usr/sbin/ioreg"
GREP="/usr/bin/grep"
SUDO="/usr/bin/sudo"
KILLALL="/usr/bin/killall"
PGREP="/usr/bin/pgrep"
DATE="/bin/date"
SLEEP="/bin/sleep"
ECHO="/bin/echo"
KILL="/bin/kill"

# Camera-related processes that grab UVC on macOS
PROC_REGEX='AppleCameraAssistant|AppleCamera|VDCAssistant|com\.apple\.cmio\.registerassistantservice'
KILL_CANDIDATES=(AppleCameraAssistant AppleCamera VDCAssistant com.apple.cmio.registerassistantservice)

# Safe logger (no xargs); truncate long lines
log() {
  local ts msg
  ts="$($DATE '+%Y-%m-%d %H:%M:%S')"
  msg="$*"
  if [ ${#msg} -gt 900 ]; then msg="${msg:0:900} ... [truncated]"; fi
  "$ECHO" "[realsense_guard] $ts $msg"
}

# Detect Intel RealSense D435 by name or VID:PID 8086:0b07 (accept hex or decimal)
detect_rs() {
  local dump
  dump="$($IOREG -p IOUSB -l 2>/dev/null || true)"
  echo "$dump" | "$GREP" -qi 'RealSense' && return 0
  if echo "$dump" | "$GREP" -qiE 'idVendor = (0x8086|32902)'; then
    if echo "$dump" | "$GREP" -qiE 'idProduct = (0x0b07|2823)'; then
      return 0
    fi
  fi
  return 1
}

kill_by_name() {
  local name rc pids pid
  name="$1"
  # 1) try killall first
  if "$SUDO" -n "$KILLALL" -9 "$name" 2>&1; then
    log "killed $name via killall"
    return 0
  fi
  rc=$?
  log "killall failed for $name (rc=$rc); trying kill -9 by PID"
  # 2) fall back to kill -9 each PID we find
  pids="$($PGREP -f "$name" 2>/dev/null || true)"
  for pid in $pids; do
    if "$SUDO" -n "$KILL" -9 "$pid" 2>&1; then
      log "killed pid $pid ($name) via kill -9"
    else
      log "ERROR: kill -9 $pid ($name) failed"
    fi
  done
}

# NOTE: Only act when a RealSense device is actually attached (no proactive kills)
loops=0
while true; do
  if detect_rs; then
    if "$PGREP" -alf "$PROC_REGEX" >/dev/null 2>&1; then
      procs="$($PGREP -alf "$PROC_REGEX" || true)"
      procs_oneline=$(echo "$procs" | tr '\n' '; ' | head -c 850)
      log "RS attached → killing camera helpers: $procs_oneline"
      for n in "${KILL_CANDIDATES[@]}"; do
        if "$PGREP" -x "$n" >/dev/null 2>&1 || "$PGREP" -f "$n" >/dev/null 2>&1; then
          kill_by_name "$n" || true
        fi
      done
    fi
  fi

  loops=$((loops + 1))
  if (( loops % 30 == 0 )); then
    if detect_rs; then
      log "HB: attached"
    else
      log "HB: absent"
    fi
  fi

  "$SLEEP" "$INTERVAL"

done
GUARD

/usr/bin/sudo /bin/chmod 755 "$GUARD_PATH"

# --- Sudoers rule for passwordless kill of camera agents --------------------
/usr/bin/sudo /usr/bin/tee "$SUDOERS_FILE" >/dev/null <<EOF
# Allow $USER_NAME to kill macOS camera agents without a password (needed by the RealSense guard)
$USER_NAME ALL=(root) NOPASSWD: /usr/bin/killall *, /usr/bin/pkill *, /bin/kill *
EOF

/usr/bin/sudo /usr/sbin/chown root:wheel "$SUDOERS_FILE"
/usr/bin/sudo /bin/chmod 440 "$SUDOERS_FILE"
/usr/bin/sudo /usr/sbin/visudo -c -f "$SUDOERS_FILE" >/dev/null

# --- LaunchAgent plist -------------------------------------------------------
/bin/cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
      <string>$GUARD_PATH</string>
      <string>$INTERVAL</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/${LABEL}.out</string>
    <key>StandardErrorPath</key><string>/tmp/${LABEL}.err</string>
  </dict>
</plist>
EOF

# Ensure user ownership (required by launchd for user agents)
/usr/sbin/chown "$USER_NAME":"$PRIMARY_GROUP" "$PLIST_PATH"
/bin/chmod 644 "$PLIST_PATH"

# --- Load (or reload) the LaunchAgent ---------------------------------------
UIDNUM="$(( $(/usr/bin/id -u "$USER_NAME") ))"

# Helper: try launchctl without sudo, then with sudo on failure
run_lctl() {
  local subcmd="$1"; shift
  if /bin/launchctl "$subcmd" "$@" 2>/dev/null; then
    return 0
  fi
  /usr/bin/sudo /bin/launchctl "$subcmd" "$@"
}

# Unload any existing instance (ignore errors)
run_lctl bootout "gui/$UIDNUM/$LABEL" || true

# Bootstrap and start the agent for this user's GUI session
run_lctl bootstrap "gui/$UIDNUM" "$PLIST_PATH"
run_lctl enable "gui/$UIDNUM/$LABEL"
run_lctl kickstart -k "gui/$UIDNUM/$LABEL"

cat <<EON
Installed and started $LABEL
- Guard script : $GUARD_PATH
- LaunchAgent  : $PLIST_PATH
- Sudoers rule : $SUDOERS_FILE
- Interval     : $INTERVAL seconds

Logs:   tail -f /tmp/${LABEL}.out /tmp/${LABEL}.err
Stop:   launchctl bootout gui/$UIDNUM/$LABEL
Start:  launchctl bootstrap gui/$UIDNUM $PLIST_PATH && launchctl kickstart -k gui/$UIDNUM/$LABEL
Uninstall:  $(/usr/bin/basename "$0") --uninstall
EON