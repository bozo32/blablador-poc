# Patch Probe + Next Work Runbook

This file is a self-check + execution prep checklist.

## 0) Non-Negotiables

- Do not modify real code until the patch probe passes.
- Keep the repo clean (or isolate unrelated dirt) before making changes.
- Stage files individually; never `git add .`.

## 1) Patch Tool Healthcheck

Goal: determine whether `apply_patch` is reliable, and which *category* of failure we have.

### 1.1 Probe Principles

- Use only sacrificial files under repo root.
- Probe should be deterministic and reversible.
- Prefer ASCII.

### 1.2 Probe A: Single-hunk, single-file (control)

Target: `.opencode_patch_probe.txt`

Patch:
- Add file with `probe: v1`
- Update line to `probe: v2`
- Delete file

Expected:
- Tool succeeds, filesystem changes reflect each step.

If this fails:
- Treat as *tool-layer abort* (infrastructure / OpenCode bug). Avoid apply_patch entirely.

### 1.3 Probe B: Multi-hunk, same file (position tracking)

Target: `.opencode_patch_probe_big.txt`

Create file with 200 numbered lines, then apply a patch with 3 hunks:
- Replace line 1
- Replace line 100
- Replace line 200

Expected:
- All hunks apply.

If A passes but B fails:
- Suspect multi-hunk offset/context tracking bug.
- Workaround: avoid multi-hunk patches; apply small single-hunk patches; or edit via guarded Python rewrites.

### 1.4 Probe C: Multi-file patch

Targets:
- `.opencode_patch_probe_a.txt`
- `.opencode_patch_probe_b.txt`

Patch:
- Add both files
- Update both files
- Delete both files

Expected:
- Entire patch applies.

If A/B pass but C fails:
- Suspect multi-file patch batching bug.

### 1.5 If apply_patch is unreliable

Fallback editing strategy (robust, slower):

- Use `python` in `bash` for guarded edits:
  - Read file
  - Assert exact anchor exists and is unique
  - Replace once
  - Write file
  - Abort if assertions fail
- After every change:
  - `python -m py_compile <touched files>`
  - `pre-commit run black --files ...` (as needed)
  - `pre-commit run flake8 --files ...`

## 2) What We’re Doing Next

You’ve been doing a targeted cleanup pass (10-04.5 style): compare backend truth vs UI behavior, then close the UX gaps before the full GSD discussion/planning.

The next work is **NOT** Phase 10-05 yet if we want to finish the umbrella `10-04.5` cleanup deliverables.

### 2.1 Umbrella 10-04.5 Deliverables (from `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-PLAN.md`)

- `docs/WORKFLOW_PROTOCOL.md`
  - Stage A–F checklist
  - How to run verifiers
  - Expected failure modes + remediation

- `scripts/dev/verify_10_04_5_spine_workflow.sh`
  - Orchestrates Stage A–F against the minimal corpus
  - Prints PASS/FAIL per stage
  - Leaves artifacts in spine only

- Cleanup / pruning
  - Remove or gate local-development residue (anything that assumes `data/**` is authoritative)
  - Ensure project scoping is consistent everywhere

### 2.2 Prep / Context To Load Before Working

- `.planning/STATE.md` (current position, constraints)
- `.planning/ROADMAP.md` (phase ordering + plan inventory)
- `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-PLAN.md` (umbrella requirements)
- `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-01-VERIFICATION.md` (what was verified)

### 2.3 Runtime Baseline Assumptions

- Everything should run in Compose (UI + API), not local conda.
- Avoid port conflicts (don’t leave ad-hoc `docker compose run` containers binding 8000).

Baseline commands:
- `docker compose ps`
- `docker compose up -d --build app-api app-ui`

### 2.4 Minimal Corpus

- Use the existing `corpus/workflow/*.pdf` paths already used by `scripts/dev/verify_10_04_5_01_intake.sh`.
- The umbrella workflow verifier should accept env overrides for corpus paths.

## 3) Checkpoints

If we proceed with umbrella 10-04.5 work, add a human checkpoint after:

- The new Stage A–F verifier script exists and passes Stage A–D automatically.
- UI smoke: drop citing + source; confirm links update.

## 4) Workspace Hygiene

Known untracked file (do not commit unless explicitly requested):
- `docs/WORKFLOW_STAGE_BACKEND_NOTES.md`

