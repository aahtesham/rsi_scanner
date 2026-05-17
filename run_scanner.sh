#!/usr/bin/env bash
# Run RSI scanner in the project venv with local notifications.
#
# Usage:
#   ./run_scanner.sh              # default: scan every 5 min, notify on matches
#   SCAN_SLEEP_S=600 ./run_scanner.sh   # every 10 min
#   NOTIFY_MATCHES=0 ./run_scanner.sh # no macOS popups
#
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export SCAN_SLEEP_S="${SCAN_SLEEP_S:-300}"
export NOTIFY_MATCHES="${NOTIFY_MATCHES:-0}"
export NOTIFY_TELEGRAM="${NOTIFY_TELEGRAM:-1}"

mkdir -p logs
LOG="logs/scanner_$(date +%Y%m%d).log"

echo "Starting scanner | interval=${SCAN_SLEEP_S}s | log=$LOG"
PY="./.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="./venv/bin/python"
fi
exec "$PY" -u rsi_scanner_multi.py 2>&1 | tee -a "$LOG"
