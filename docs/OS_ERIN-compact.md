# OS-ERIN Compact

Use this file for fast startup context before any coding session.

## 1) What OS-ERIN is

OS-ERIN is a local-first citation validation system: upload citing PDFs, extract/resolve references, create reviewable claims, attach cited-source PDFs, run evidence retrieval/ranking/NLI, and record/export judgments.

Primary spec: `docs/REPO_SPEC.md`.

## 2) Core architectural rule

UI is projection and control surface. Backend + spine stores (Postgres + object store) are source of truth for durable workflow state.

Primary rationale: `docs/SYSTEM_DESIGN_RATIONALE.md`.

## 3) Critical contracts

- Scope and identity are explicit. Missing required scope/identity fails closed.
- Project membership and active scope session are backend-authoritative.
- Visibility contract for opinion events is fail-closed for incomplete ACL context.

Primary contract: `docs/IDENTITY_SCOPE_VISIBILITY_CONTRACT.md`.

## 4) Operational workflow contract

Stage A-F protocol defines ingestion through graph navigation checks and verification scripts.

Primary protocol: `docs/WORKFLOW_PROTOCOL.md`.

## 5) Current execution posture

- Active planning/execution state lives in `.planning/STATE.md`.
- Latest handoff and read order live in `.planning/README.md`.
- Bug-hunt notes live in `.planning/debug/`.

## 6) Session defaults

- If behavior mismatch appears, suspect stale containers/images first and rebuild (`app-api`, `app-ui`).
- Preserve strict scope contracts while debugging; do not reintroduce silent fallback.
- Avoid broad pruning during active bug hunts; log prune candidates and defer deletions.

## 7) Next read

Pick one use-case pack:

- `docs/OS_ERIN-usecase-bughunt-chasing.md`
- `docs/OS_ERIN-usecase-bughunt-surfing-graph.md`
- `docs/OS_ERIN-usecase-bughunt-reading.md`
- `docs/OS_ERIN-usecase-contract-validation.md`
- `docs/OS_ERIN-usecase-phase-execution.md`
