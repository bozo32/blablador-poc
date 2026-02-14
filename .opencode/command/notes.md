---
description: Read .planning/NOTES.md and resume casual work
tools:
  read: true
  bash: true
  question: true
---

<objective>
Restore working context quickly after a /clear by treating `.planning/NOTES.md` as the single source of truth.
</objective>

<process>

1. Read `.planning/NOTES.md`.
2. If the SNAPSHOT block is missing/stale (or git status looks different), run the snapshot updater:
   - `python scripts/update_notes_snapshot.py`
   Then re-read `.planning/NOTES.md`.
3. Present a short "where we are" and "next 3 actions" based strictly on `.planning/NOTES.md`.
4. Ask exactly one targeted question only if blocked.

</process>
