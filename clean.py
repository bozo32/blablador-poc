#!/usr/bin/env python3
"""Stop local dev processes (backend/frontend) safely.

Defaults target ports:
- backend: 8000 (uvicorn)
- frontend: 8501 (streamlit)

This script only manages processes; it does not delete project data.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Iterable, List


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _pids_on_port(port: int) -> List[int]:
    # macOS: lsof is available by default.
    proc = _run(["lsof", "-ti", f"tcp:{int(port)}"])
    if proc.returncode != 0:
        return []
    out = (proc.stdout or "").strip()
    if not out:
        return []
    pids: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return sorted(set(pids))


def _kill_pids(pids: Iterable[int], *, dry_run: bool) -> None:
    pids = sorted(set(int(p) for p in pids if int(p) > 0))
    if not pids:
        return

    if dry_run:
        print("Would stop PIDs:", ", ".join(str(p) for p in pids))
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    # Give processes a moment to exit.
    time.sleep(0.7)

    still_alive: List[int] = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        still_alive.append(pid)

    for pid in still_alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Stop local dev processes.")
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8000,
        help="Port to stop backend on (default: 8000)",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=8501,
        help="Port to stop Streamlit on (default: 8501)",
    )
    parser.add_argument(
        "--backend",
        action="store_true",
        help="Stop backend port only",
    )
    parser.add_argument(
        "--frontend",
        action="store_true",
        help="Stop frontend port only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Stop both backend and frontend (default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be stopped, but do nothing",
    )
    args = parser.parse_args(argv)

    stop_backend = bool(
        args.all or (not args.backend and not args.frontend) or args.backend
    )
    stop_frontend = bool(
        args.all or (not args.backend and not args.frontend) or args.frontend
    )

    any_found = False

    if stop_backend:
        pids = _pids_on_port(int(args.backend_port))
        if pids:
            any_found = True
            print(
                f"Stopping backend on :{int(args.backend_port)} "
                f"(PIDs: {', '.join(map(str, pids))})"
            )
            _kill_pids(pids, dry_run=bool(args.dry_run))
        else:
            print(f"No process found on backend port :{int(args.backend_port)}")

    if stop_frontend:
        pids = _pids_on_port(int(args.frontend_port))
        if pids:
            any_found = True
            print(
                f"Stopping frontend on :{int(args.frontend_port)} "
                f"(PIDs: {', '.join(map(str, pids))})"
            )
            _kill_pids(pids, dry_run=bool(args.dry_run))
        else:
            print(f"No process found on frontend port :{int(args.frontend_port)}")

    return 0 if any_found or args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
