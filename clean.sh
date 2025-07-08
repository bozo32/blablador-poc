#!/usr/bin/env bash
###############################################################################
# clean.sh – Stop all running Citation-Checker processes (Linux / macOS bash)
#   • kills any python application.py parent
#   • kills all Streamlit processes
#   • kills all Uvicorn processes (including the ColBERT micro‑service)
###############################################################################

echo "🔍 Searching for runaway processes ..."

# Helper: kill by pattern if any exist
kill_if_running () {
  local pattern="$1"
  local pids
  pids=$(pgrep -f "$pattern")
  if [[ -n "$pids" ]]; then
    echo "⚠️  Killing $(echo "$pids" | wc -w) process(es) matching [$pattern]"
    kill $pids         2>/dev/null      # polite SIGTERM
    sleep 1
    # if still alive, force-kill
    pids=$(pgrep -f "$pattern")
    if [[ -n "$pids" ]]; then
      echo "🚨 Forcing kill on $(echo "$pids" | wc -w) stubborn process(es)"
      kill -9 $pids    2>/dev/null
    fi
  fi
}

# Kill Uvicorn (worker + reload-watcher)
pkill -f "[u]vicorn"

# Kill Streamlit (app + watcher)
pkill -f "[s]treamlit"

pkill -f "multiprocessing.spawn"
pkill -f "multiprocessing.resource_tracker"

# Kill any lingering application.py runs
kill_if_running "python .*application.py"
kill_if_running "python3 .*application.py"

# Also kill anything still listening on ports 8000, 8501, or 7001
if command -v lsof &>/dev/null; then
  for port in 8000 8501 7001; do
    pids=$(lsof -t -i tcp:$port)
    if [[ -n "$pids" ]]; then
      echo "⚠️ Killing processes listening on port $port: $pids"
      kill $pids 2>/dev/null
    fi
  done
fi

echo "✅ All citation-checker processes terminated."