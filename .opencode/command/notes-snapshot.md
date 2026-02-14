---
description: Update the auto-generated snapshot section in .planning/NOTES.md
tools:
  bash: true
  read: true
---

<objective>
Refresh the SNAPSHOT block in `.planning/NOTES.md` so that reading it alone is enough to resume after context loss.
</objective>

<process>

1. Run: `python scripts/update_notes_snapshot.py`
2. Read `.planning/NOTES.md`.

</process>
