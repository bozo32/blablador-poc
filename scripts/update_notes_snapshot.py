from __future__ import annotations

import datetime as _dt
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / ".planning" / "NOTES.md"

START = "<!-- SNAPSHOT:START -->"
END = "<!-- SNAPSHOT:END -->"


def _run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, cwd=str(ROOT), stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace").rstrip()


def _snapshot_block() -> str:
    now = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        branch = _run(["git", "branch", "--show-current"]).strip()
    except Exception:
        branch = ""
    try:
        head = _run(["git", "rev-parse", "--short", "HEAD"]).strip()
    except Exception:
        head = ""
    try:
        status = _run(["git", "status", "-sb"]).strip()
    except Exception:
        status = ""
    try:
        diffstat = _run(["git", "diff", "--stat"]).strip()
    except Exception:
        diffstat = ""
    try:
        log = _run(["git", "log", "-5", "--oneline", "--decorate"]).strip()
    except Exception:
        log = ""

    lines: list[str] = []
    lines.append(START)
    lines.append(f"Updated: `{now}`")
    if branch:
        lines.append(f"Branch: `{branch}`")
    if head:
        lines.append(f"HEAD: `{head}`")
    if status:
        lines.append("")
        lines.append("```text")
        lines.append(status)
        lines.append("```")
    if diffstat:
        lines.append("")
        lines.append("```text")
        lines.append(diffstat)
        lines.append("```")
    if log:
        lines.append("")
        lines.append("```text")
        lines.append(log)
        lines.append("```")
    lines.append(END)
    return "\n".join(lines)


def main() -> int:
    if not NOTES.exists():
        raise SystemExit(f"Missing notes file: {NOTES}")

    text = NOTES.read_text(encoding="utf-8")

    pattern = re.compile(
        re.escape(START) + r".*?" + re.escape(END),
        flags=re.DOTALL,
    )
    block = _snapshot_block()

    if pattern.search(text):
        updated = pattern.sub(block, text, count=1)
    else:
        updated = block + "\n\n" + text

    NOTES.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
