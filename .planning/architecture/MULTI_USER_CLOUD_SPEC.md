# Multi-User Cloud Planning Spec (Web vs Thin Client)

This spec captures the minimum design decisions needed to support multiple
users in a future cloud deployment (e.g. Azure) without forcing a full auth/
tenancy implementation today.

## Goal

- Ensure current spine/work architecture can evolve into a multi-user,
  multi-project cloud system with minimal rewrites.

## Recommended Direction

- Default: web-based UI (browser) + API.
- Thin client is optional and should be justified by offline needs or extreme
  local-hardware coupling.

Rationale:
- Browser + API is easiest to deploy/upgrade, supports shared org identity, and
  avoids distributing binaries.

## Decisions To Lock In Now

### 1) Identity is first-class in the spine

- Every write must carry:
  - `project_id`
  - `created_by_user_id`
  - `work_id`
  - `attempt_id` / `job_id` when applicable

Even in POC mode, treat `created_by_user_id` as real and persist it.

### 2) Project is the primary tenancy boundary

- A user belongs to 1+ projects.
- Works are either:
  - owned by exactly one project, or
  - globally deduped but explicitly linked to multiple projects.

Pick one and reflect it in DB constraints and list endpoints.

### 3) Authorization can start as "soft" but must be enforceable

Short-term (POC):
- Accept `X-Project-Id` header.
- Use a single default user.

Long-term:
- Replace with OIDC (Azure Entra ID) and map:
  - `sub` -> user_id
  - groups/roles -> project membership

Important: do not bake auth into business logic; use middleware/dependencies.

### 4) Isolation + data access patterns

- All Postgres queries must be scoped by `project_id` (or must be provably safe
  if global).
- Object store keys must include `work_id` and should be project-aware if works
  are project-owned.
- Add auditability:
  - who triggered extract/fallback/resolve
  - who cancelled
  - who forced resolution

### 5) Concurrency, quotas, and fairness

- In cloud, the hard problem is not "can we run 100 jobs", it's "do we let one
  user starve everyone".
- Add primitives:
  - per-project concurrency limits (max running extract jobs)
  - per-user rate limits (enqueue calls)
  - bounded queues with explicit backpressure (503 + retry-after)

### 6) Sessions and long-running work

- Avoid UI sessions owning jobs.
- Jobs must outlive the UI tab.
- UI polls job state (`/ingest/{id}/spine` or job endpoints) and can reconnect.

### 7) Compatibility with both deployment styles

If later supporting both web and thin client:
- Keep API contracts stable.
- Treat the UI as a client of the API.
- Avoid server-side UI session state as the system of record.

## Web UI vs Thin Client

### Web UI (recommended)

Pros:
- Centralized updates
- Natural OIDC integration (Entra ID)
- Easy shared access
- Best fit for multi-user collaboration

Cons:
- Requires network connectivity

### Thin Client

Pros:
- Potential offline flows
- Can bundle local tooling (PDF viewers, local OCR)

Cons:
- Distribution/update complexity
- Auth + secrets handling complexity
- Harder to guarantee consistent environment

## Implementation Roadmap Hook

- Treat multi-user as a V2+ concern; do not block Phase 9.2/9.3.
- Ensure new endpoints always accept/emit `project_id` and attribute actions to
  a user id.
- Add an auth layer later without changing ingestion/extraction core.
