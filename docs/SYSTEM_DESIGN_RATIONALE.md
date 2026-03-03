# System Design Rationale

## Governance Metadata

- Doc role: architecture rationale
- Authority tier: 3 (spec rationale)
- Status: active
- Owner: repo maintainers
- Last reviewed: 2026-03-03
- Canonical for: design tradeoffs and long-horizon posture
- Defers to: contract/protocol docs for runtime invariants

This document answers a practical question: "Why is the system built this way, and what does that buy us?"

## Reader 1: Summary

### Design intent in one paragraph

The system is designed as a durable review workflow, not a transient UI demo: the backend spine is the source of truth, reviewer actions are explicit, and outputs are attributable and exportable. This makes it suitable as a foundation for multi-institution citation review, even though identity federation, full selective disclosure semantics, and annotation/writing integrations are still in progress.

### Core choices and why they were chosen

- Spine-first persistence: reliable state, replayability, and auditing.
- Explicit user/project scope: safer collaboration and isolation.
- Graph-first domain model: enables both evidence integrity and navigation.
- Fail-closed visibility: safer collaboration while ACL semantics mature.

### Current maturity

- Strong POC foundation for citation walking and evidence review.
- Not yet full production IAM or full disclosure-policy completion.

---

## Reader 2: Engineering Details

## 1) Architectural choices

### 1.1 Spine-first persistence

- Durable state belongs to backend persistence layers (Postgres + object store).
- UI session state is orchestrational/cache-only, not authoritative.
- Reason: this minimizes state drift and supports eventual multi-user concurrency.

### 1.2 Explicit workflow transitions

- Expensive and consequential steps are explicit (extract, resolve, rerun, select, judge).
- Reason: review and traceability requirements beat hidden automation in scholarly workflows.

### 1.3 Scope-first API model

- Project scope and actor identity are progressively required for critical routes.
- Membership state is persisted and active scope is server-truth-backed via scope session APIs.
- Reason: prevents silent cross-project contamination and supports collaborative tenancy.

### 1.5 Client replaceability by contract

- Workflow authority is in backend contracts, not Streamlit widget/session behavior.
- New clients (hackathon UIs, SPA, mobile) should integrate by calling the same scope/workflow APIs.
- Reason: allows rapid UI evolution while keeping the spine stable and auditable.

### 1.4 Graph as architecture

- Graph artifacts support both correctness (resolution/placement coherence) and navigation.
- Reason: citation-walking and evidence review share graph constraints; one model avoids duplicate truth systems.

## 2) Deliberate compromises in POC stage

### 2.1 Header-carried identity before full auth

- Current scope/identity contracts are strong for POC, but not final IAM.
- This was chosen to validate workflow semantics before provider integration.

### 2.2 Conservative ACL for selective sharing

- `selectable` visibility is fail-closed when group context is incomplete.
- This intentionally trades feature completeness for disclosure safety.

### 2.3 Streamlit-led orchestration

- Fast for product iteration, but more coupled than a dedicated SPA architecture.
- Accepted as short-term velocity tradeoff; migration concerns are tracked.

## 3) Why this is still the right path

- The architecture already supports durable, reviewable scholarly workflows.
- The biggest remaining work is policy/compliance-grade identity + sharing semantics, not a full rewrite.
- Key features requested for target use (annotations, Zotero, cross-library candidate-span sharing) can be layered on top of current contracts.

## 4) Engineering risks to keep in view

- Any fallback to default scope/identity on critical paths is a regression risk.
- Graph resolution and navigation paths must stay deterministic.
- Selective sharing must remain fail-safe until group ACL is explicit and tested.

## 5) Near-term guidance for contributors

- Treat identity/scope/visibility contracts as architectural invariants.
- Prefer additive migrations and explicit compatibility notes.
- If behavior is temporarily compatibility-preserving, log it in prune/verification docs.
- Avoid introducing UI-only state as a source of truth for cross-user or cross-project behavior.
