# UX Style Conventions (Workspace UI)

Purpose: give new contributors/agents one place to follow when changing UX.

## Where To Start

When making any UX change, read in this order:

1. `.planning/codebase/UX_STYLE_CONVENTIONS.md` (this file)
2. `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-PRUNE-CANDIDATES.md`
3. `frontend/ui.py` and `frontend/components/*`

## Visual + Interaction Rules

- Use **Material Symbols Outlined** tokens only for icon-ish affordances.
- Status icon mapping (canonical):
  - working/in progress: `clock_loader_10`
  - running with issues / attention needed: `running_with_errors`
  - complete: `check_circle`
  - not yet complete / pending user completion: `incomplete_circle`
- Right rail citespans must be shown as **dropdown-like rows** with `>` closed / `v` open semantics.
- Left and right rails should use the same disclosure mental model.
- Avoid nested collapsibles inside collapsibles unless there is no alternative.
- Prefer left-aligned labels and avoid centered button text for document/citespan lists.

## Contract-First UX Changes (Migration Safety)

- Treat URL query params as navigation/bookmarking, not as state mutation control plane.
- Do not encode business workflow in widget rerun side-effects.
- Keep state transitions explicit and idempotent:
  - `follow_span`
  - `open_span`
  - `drop_span`
  - `place_relation`
- UI components should render projections from contracts, not mutate hidden state during render.

## Anti-Patterns To Avoid

- Mixing multiple queue sources without explicit precedence (session + URL + server).
- Auto-opening rows during render pass.
- Adding UI-only behavior that cannot be represented in backend contracts.
- Hard-coding pixel heights where viewport-relative layout is expected.

## Implementation Checklist For UX Edits

- Update/confirm icons against canonical mapping above.
- Confirm right rail accumulation behavior with multiple chip clicks.
- Confirm single-open/close behavior for citespan rows.
- Confirm retrieval panel handles multi-target spans.
- Add migration-impact notes to:
  - `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-PRUNE-CANDIDATES.md`
