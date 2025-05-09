#!/usr/bin/env bash
###############################################################################
# clean.sh – Stop all running Citation-Checker processes (Linux / macOS bash)
#   • kills any python application.py parent
#   • kills all Streamlit processes
#   • kills all Uvicorn processes
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

kill_if_running "python .*application.py"
kill_if_running "streamlit run"
kill_if_running "uvicorn .*backend.main:app"

echo "✅ All citation-checker processes terminated."