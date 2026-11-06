#!/bin/bash
# Start the local Kronos web UI. Does not install packages or touch the virtualenv.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is not on PATH."
  exit 1
fi
exec python3 run.py
