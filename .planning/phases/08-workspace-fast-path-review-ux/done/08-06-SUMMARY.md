---
phase: 08-workspace-fast-path-review-ux
plan: 06
subsystem: ui
tags: [streamlit, attachments, evidence, workspace, ux]

# Dependency graph
requires:
  - phase: 08-workspace-fast-path-review-ux
    provides: inline citation chips + stable right-rail collector (08-05)
provides:
  - Global Source bin upload surface backed by `POST/GET/PATCH /attachments`
  - Background processing UX (processing → matched → needs placement → placed)
  - Source placement onto a specific claim (claim-scoped assignment)
  - Auto evidence reruns on claim save and on source placement
  - Evidence view gating while reruns are queued/running; polling for updates
  - View PDF affordance (Cmd-F snippet + open local PDF when available)
affects: [workspace, chasing, evidence, attachments]

# Tech tracking
tech-stack:
  added:
    - streamlit-autorefresh (for polling while background work runs)
  patterns:
    - Query-param navigation for in-text citation clicks (`?doc=&cite=&target=`)
    - UI state hardened against Streamlit reruns (CSS reinjection + stable selection)

key-files:
  created: []
  modified:
    - frontend/attachment_queue.py
    - frontend/ui.py
    - frontend/components/chasing_panel.py
    - frontend/evidence_store.py
    - frontend/clipboard.py
    - frontend/assets/attachment_panel.css
    - frontend/assets/workspace.css
    - backend/attachment_pipeline.py
    - backend/evidence_matching/service.py
    - environment.yml

key-decisions:
  - "Sources are placed onto a specific claim id (not just a citing document) so evidence runs are claim-scoped."
  - "Keep segmentation usable without external LLM config by falling back to local deterministic segmentation."

completed: 2026-02-04
---

# Phase 8 Plan 06: Source Bin + Fast-Path Review UX Summary

**Replaced per-claim attachment dropzones with a single global Source bin, wired to backend `/attachments`, and hardened the fast-path review loop (placement → rerun → gated selection → PDF view) for real Streamlit sessions.**

## Accomplishments

- Source bin is the only cited-source upload surface; uploads immediately enqueue background conversion/parsing.
- Source bin items can be assigned/re-placed onto a specific claim, archived/unarchived, retried, and polled while processing.
- Claim saves and source placement automatically queue evidence reruns; evidence selection is disabled while the backend reports queued/running.
- View PDF copies a deterministic Cmd-F snippet and opens local PDFs when possible.
- Settings drawer reliably exposes Blablador configuration (key/base URL) with readable contrast.

## Task Commits

Each plan task was committed atomically:

1. **Task 1: Implement Source bin backend integration in attachment_queue** — `8455a3e` (feat)
2. **Task 2: Wire Source bin into 3-pane UI + auto rerun + View PDF snippet** — `f5b7825` (feat)

## Checkpoint Fixes (Human Verify Follow-ups)

User verification uncovered Streamlit-specific edge cases. Fixes were applied as small, atomic commits:

- Stable, readable Source bin + session controls: `e770a03`, `b43972f`
- Remove stray inline prefixes and keep chips stable: `71c3943`, `2a429a4`
- Prevent Streamlit widget/session_state conflicts: `cb05578`
- Make restarts and query-param navigation robust: `c0be563`, `08c0a89`, `c50b3d4`
- Prevent auto reruns for unassigned sources: `d82c4e8`
- Reduce noisy dependency logs: `c2c4f80`
- Unify PDF upload surface and keep citation clicks in-tab: `15676fc`
- Evidence rerun UX aligned to backend lock state: `1a64a0e`
- Deterministic segmentation fallback when LLM config missing: `1f94fed`
- Source bin polling + status pill fixes: `5f0f53f`
- Reclassify mis-uploaded PDFs (source → citing doc) + preserve filenames: `d984fa8`, `925cdab`
- Contrast/density/Cosmo-ish styling hardening: `853fc19`, `faf3e69`, `19c4bea`, `d9fc7db`, `3f6bbe1`, `1421083`, `eea6f05`

## Verification

- Automated: `pytest -q`
- Manual (checkpoint): upload → processing status → place onto a specific claim → auto rerun → gated selection while running → View PDF snippet + open.

## Notes

- A processed source showing "Needs placement" is expected until it is assigned to the exact claim id being rerun (e.g., `...:10a` vs `...:10b`).
- The longer-term Cosmo UI contract + explicit center panel mode plan is recorded in `.planning/phases/08-workspace-fast-path-review-ux/08-06-NOTES.md`.

---
*Phase: 08-workspace-fast-path-review-ux*
*Completed: 2026-02-04*
