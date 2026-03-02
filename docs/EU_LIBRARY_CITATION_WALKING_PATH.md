# EU Library Citation Walking Path

Status: strategic direction document for the POC (updated 2026-03-02)

This document is intentionally written for two readers:

- Reader 1: technically aware, non-specialist (summary)
- Reader 2: implementation engineer (details and tradeoffs)

## Reader 1: Summary

### The target use case

- This POC is the seed of a front end for EU libraries that helps researchers walk citation trees.
- The system helps a user move from a citing sentence to cited evidence and record a review outcome.
- Over time it should support:
  - user annotations/markup,
  - Zotero-assisted writing workflow,
  - copyright-aware sharing between institutions by exchanging candidate spans, not full documents.

### What is already present and useful

- Citation walking core: upload citing doc, navigate callouts, resolve references, follow targets.
- Evidence review loop: candidate spans, reviewer selection, judgment capture, export.
- Scope foundations: user/project scoping, membership-aware project selection, visibility controls.
- Graph foundations: work/claim/span graph structures already exist and are used in workflow.

### Why this architecture still makes sense for the target

- It already separates durable metadata from large artifacts (Postgres + object storage), which is required for institutional operation.
- It already supports explicit reviewer actions over opaque automation, which matches research review practice.
- It already has project-level isolation and visibility controls, which are prerequisites for multi-library collaboration.

### Main gaps to close

- Production auth and federated identity trust (current identity is header-scoped POC contract).
- Full group-sharing semantics for selective disclosure (current `selectable` is fail-closed unless context exists).
- Annotation product surface and storage model.
- Zotero integration and sync contracts.
- Copyright-preserving cross-library span exchange protocol.
- Graph navigation UX hardening for large collaborative use.

---

## Reader 2: Engineering Details

## 1) Directional Product Model

### 1.1 Core user job

Given a claim context in a citing paper, a researcher should be able to:

1. identify the citation target,
2. inspect local and cited context,
3. evaluate evidence candidate spans,
4. record judgment and rationale,
5. share only allowed outputs based on disclosure policy.

### 1.2 Collaboration model

- Multiple users per project, multiple projects per user.
- Visibility-aware exposure of other users' work.
- Discovery-by-encounter principle: users see others' exposed/public work when encountered via their own navigation and allowed policy paths.

### 1.3 Graph model expectation

- The graph is both:
  - a workflow integrity mechanism (resolution/placement coherence), and
  - a user navigation surface.
- This dual role is intentional but requires careful API and UI contract hardening for predictable navigation behavior at scale.

## 2) What the Repo Already Delivers on This Path

### 2.1 Durable workflow spine

- Durable state in Postgres + object store.
- Explicit extraction/resolution/evidence runs and event trails.
- Exportable outcomes.

### 2.2 Citation chasing and evidence loop

- Callout navigation and context extraction.
- Reference resolution with editable selection pathways.
- Source attachment and evidence candidate generation.
- Reviewer judgment recording.

### 2.3 Scope and identity scaffolding

- Project membership persistence and active project selection APIs.
- Frontend apply/switch scope gating.
- Strict scope/identity wiring on major mutation surfaces.

### 2.4 Visibility baseline

- Opinion visibility normalization (`private`, `public`, `selectable`).
- Safety-first behavior: `selectable` currently fail-closed without explicit group ACL context.

## 3) Why Current Design Choices Are Rational

### 3.1 Spine-first over UI-first state

- Needed for reproducibility, auditability, and eventual institutional deployment.
- Prevents local session artifacts from becoming authoritative.

### 3.2 Explicit reruns and reviewer actions

- Needed for defensibility in scholarly use.
- Supports review accountability and disagreement analysis.

### 3.3 Graph as first-class domain object

- Enables consistent target resolution and future exploration UX.
- Supports both claim/evidence linkage and navigation.

### 3.4 Fail-closed ACL posture during transition

- Safer than permissive sharing while group semantics are incomplete.
- Avoids accidental over-disclosure across institutions.

## 4) Known Headaches (and Why They Matter)

### 4.1 Identity trust boundary

- Current model is scoped-header based and not full auth/IAM.
- Needs federation-compatible auth for real multi-institution deployment.

### 4.2 Graph navigation consistency

- Graph writes and reference-resolution pathways have historically diverged.
- Must stay deterministic so navigation is trustworthy.

### 4.3 Selectable visibility semantics

- Without mature group ACL context, positive selective sharing is not safely enforceable.
- Current fail-closed behavior is safe but not feature-complete.

### 4.4 Frontend coupling

- Streamlit-based orchestration is productive for POC but creates migration constraints for long-term multi-user UX scale.

## 5) Next-Step Work Packages

### WP-A: Institutional identity and trust

- Add production auth source (token/session-backed identity).
- Bind `X-User-Id` semantics to trusted principal.

### WP-B: Selective disclosure model

- Implement explicit group/membership ACL context for `selectable` visibility.
- Define and test cross-project/cross-library disclosure matrix.

### WP-C: Annotation subsystem

- Document-level and span-level markup model.
- Visibility and attribution controls aligned with scope contract.

### WP-D: Zotero integration

- Citation/reference synchronization boundaries.
- Linking review outputs back into author writing workflow.

### WP-E: Copyright-preserving exchange

- Define candidate-span exchange contract between institutions.
- Ensure full-document transfer is policy-governed and optional.

### WP-F: Graph navigation hardening

- Scale-tested navigation semantics.
- Encounter-based exposure policy enforcement in graph traversal APIs.

## 6) Success Criteria for "On Path"

The repo should be considered credibly on-path if:

- scope and membership remain authoritative in all critical routes,
- visibility controls remain fail-safe under uncertainty,
- graph-driven navigation remains consistent with evidence workflow,
- annotation/Zotero/copyright exchange can be added without architectural rewrites.

That is the current trajectory of this POC.
