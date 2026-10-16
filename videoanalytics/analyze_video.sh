#!/usr/bin/env bash
# ./analyze_video.sh <video-or-folder> ...   analyses and opens the report;  without arguments: opens the window.
cd "$(dirname "$0")" || exit 1
PY=python3
for v in venv .venv; do [ -x "$v/bin/python" ] && PY="$v/bin/python"; done
if [ $# -eq 0 ]; then exec "$PY" run_gui.pyw; fi
exec "$PY" -m videoanalytics analyze "$@"
