#!/usr/bin/env bash
# Install RSI scanner as a background service (starts on login, restarts if it crashes).
# Usage: ./install_background_scanner.sh
# Remove: ./install_background_scanner.sh --uninstall

set -euo pipefail

LABEL="com.malikadil.rsi.scanner"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_DST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
RUN_SCRIPT="${PROJECT_DIR}/run_scanner.sh"
LOG_DIR="${PROJECT_DIR}/logs"

usage() {
  echo "Usage: $0          # install and start"
  echo "       $0 --uninstall"
}

if [[ "${1:-}" == "--uninstall" ]]; then
  launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || launchctl unload "$PLIST_DST" 2>/dev/null || true
  rm -f "$PLIST_DST"
  echo "Removed ${LABEL}. Scanner will not auto-start on login."
  exit 0
fi

if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
  echo "ERROR: ${PROJECT_DIR}/.env not found."
  echo "Create it first: cp .env.example .env  (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)"
  exit 1
fi

if [[ ! -x "$RUN_SCRIPT" ]]; then
  chmod +x "$RUN_SCRIPT"
fi

mkdir -p "$LOG_DIR"

cat > "$PLIST_DST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${RUN_SCRIPT}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${PROJECT_DIR}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/launchd_stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/launchd_stderr.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
  </dict>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)/${LABEL}"
launchctl kickstart -k "gui/$(id -u)/${LABEL}"

echo "Installed and started: ${LABEL}"
echo "  Project: ${PROJECT_DIR}"
echo "  Logs:    ${LOG_DIR}/scanner_*.log and launchd_*.log"
echo "  Telegram alerts when matches (from .env)"
echo ""
echo "Mac tips so scans keep running:"
echo "  • System Settings → Battery → keep power connected when possible"
echo "  • Optional: disable sleep on power adapter (System Settings → Lock Screen / Battery)"
echo ""
echo "Check status:  launchctl print gui/$(id -u)/${LABEL}"
echo "Stop/remove:   ./install_background_scanner.sh --uninstall"
