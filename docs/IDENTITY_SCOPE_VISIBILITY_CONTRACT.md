# Identity, Scope, and Visibility Contract

Status: active POC contract (updated 2026-03-02)

This document is intentionally written for two readers:

- Reader 1: technically aware, non-specialist (summary)
- Reader 2: implementation engineer (detailed contract and failure modes)

## Reader 1: Summary

### What this contract means

- Every meaningful action is scoped by two identities:
  - `user` (who is acting)
  - `project` (where the data belongs)
- A user can belong to multiple projects.
- A project can have multiple members.
- The UI now requires choosing and applying scope before doing work.

### What is now true

- Project membership exists in the backend (`/projects`, `/projects/select`, `/projects/active`).
- Active project can persist per user.
- Most sensitive write paths are strict-scoped and reject missing identity/scope.
- Visibility states exist for opinion events:
  - `private`
  - `public`
  - `selectable` (currently fail-closed unless explicit group context is present)

### What is intentionally not finished yet

- Full authentication/identity provider integration (JWT/OIDC/session-backed trust) is not complete.
- Full positive group-sharing semantics for `selectable` are intentionally deferred.

### Pitch-safe framing

This POC now demonstrates project/member-aware workflow control, explicit scope application in the UI, and fail-closed visibility handling for sensitive opinion data. It is not yet a production-grade IAM system.

---

## Reader 2: Engineering Contract

## 1) Scope Invariants

### 1.1 Request identity/scope invariants

- Scoped operations must carry `X-Project-Id`.
- Mutation operations must carry actor identity (`X-User-Id`; reviewer identity where reviewer-scoped).
- Reviewer-scoped operations must keep reviewer identities consistent when both header and query/body fields are present.
- Missing required scope/identity must fail closed (4xx), not silently default.

### 1.2 UI invariants

- UI maintains draft vs applied scope states.
- Activity is gated until scope is applied.
- Applied scope is authoritative for API calls; draft scope is not.

### 1.3 Project membership invariants

- User-project membership is persisted.
- Active project is persisted per user.
- Project selection must respect membership.

## 2) Data Authorities

- Membership authority:
  - `user_project_memberships`
- Active project authority:
  - `user_active_projects`
- Project metadata:
  - `project_meta` (metadata only; not membership authority)
- Opinion visibility data:
  - opinion events stream (`visibility`, `group_id`, `mode`)

## 3) API Contract (Current)

### 3.1 Project membership endpoints

- `GET /projects`
  - Requires `X-User-Id`
  - Returns user project list + active marker.
- `POST /projects`
  - Requires `X-User-Id`
  - Creates project membership record; can set active.
- `POST /projects/select`
  - Requires `X-User-Id`
  - Selects active project for that user.
- `GET /projects/active`
  - Requires `X-User-Id`
  - Returns current active project.

### 3.2 Scoped workspace routes

- Pilot write paths are strictified (ledger, attachment mutation family, opinion append path).
- Additional high-risk scoped paths were strictified in follow-up batches.
- Remaining deferred routes are tracked in phase prune/verification docs.

### 3.3 Visibility semantics (opinion stream)

- Canonical values: `private`, `selectable`, `public`.
- Alias handling: legacy `shared` maps to `selectable`.
- Current safety posture:
  - `private`: owner-only visibility
  - `public`: broadly readable (within route contract context)
  - `selectable`: fail-closed unless explicit group ACL context is available

## 4) Failure and Safety Model

### 4.1 Fail-closed principles

- Missing required identity/scope should block request.
- When ACL context is incomplete (`selectable` without usable group context), deny rather than over-disclose.
- No implicit escalation from private to shared/public.

### 4.2 Backward-compat principles

- Additive schema changes preferred.
- Legacy visibility values normalized at read/write boundaries.
- Old routes can be temporarily compatibility-preserving if explicitly tracked as debt.

## 5) Known Gaps (Engineering TODO)

- Auth trust boundary is still header-based for this POC (needs auth provider integration).
- Full positive group ACL implementation for `selectable` remains deferred.
- Some scoped-required routes may still require strictification cleanup; see phase verification/prune docs.

## 6) Verification and Traceability

Primary rollout artifacts and verification logs:

- `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-02-VERIFICATION.md`
- `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-PRUNE-CANDIDATES.md`
- `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-02-UID-PROJECT-SCOPE-STITCH-CHECKLIST.md`

For system context:

- `docs/REPO_SPEC.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/CONCERNS.md`
