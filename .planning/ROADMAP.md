# Roadmap: OS-ERIN

## Overview

OS-ERIN delivers an end-to-end citation validation workflow, starting with reliable PDF ingestion and citation extraction, then moving through claim review, evidence matching, and human judgment capture. The phases build progressively from structured extraction to evidence-backed validation and exportable judgments, with lightweight workspace organization utilities at the end.

For current "what to do next" ordering, see `.planning/PRIORITIES.md`.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Ingestion + Extraction** - PDFs ingested, parsed, and references normalized.
- [x] **Phase 2: Citation Context Navigation** - Citation callouts and trees are navigable.
- [x] **Phase 3: Claim Selection + Editing** - Claims are extracted and editable for review.
- [x] **Phase 4: Evidence Attachment** - Cited PDFs are attached and parsed for evidence search.
- [x] **Phase 5: Evidence Matching + Ranking** - Evidence candidates are matched and ranked.
- [x] **Phase 6: Evidence Review + Selection** - Evidence is highlighted and selectable.
- [x] **Phase 7: Validation + Export** - Judgments are recorded and exportable.
- [ ] **Phase 8: Workspace + Fast-Path Review UX** - Inline citation navigation, background processing, and a more space-efficient workspace.
- [ ] **Phase 8.1: Recursive Retrieval + Node Graph** (INSERTED) - Multi-document citation chasing with an integrated graph view.
- [ ] **Phase 8.2: Demo Dataset + Trace Replay** (INSERTED) - A packaged synthetic corpus + replayable review traces for a distributed-system tech demo.
- [ ] **Phase 9: Claim Graph + Multi-User Consensus** - Local multi-user judgments and an interactive claim graph showing consensus/disagreement.
- [ ] **Phase 9.1: Ingestion replumbing + solid spine** (INSERTED) - Make the system runnable/distributable and ingestion automatable via a robust job/attempt spine (containers, Postgres, object store, worker services).
- [ ] **Phase 9.2: Ingestion automation robustness + fallback extraction** (INSERTED) - Make ingestion runs resilient and automatable, with deterministic fallback extraction when TEI/GROBID fails.
- [ ] **Phase 9.3: Spine Everywhere + Legacy Removal** (INSERTED) - Move remaining workflow state off local disk and delete legacy filesystem stores.
- [ ] **Phase 10: Contracts + Core Workflow Simplification** (INSERTED) - Define stage contracts and simplify the core workflow so add-ons plug in cleanly.

## Phase Details

### Phase 1: Ingestion + Extraction
**Goal**: Users can ingest PDFs and see structured metadata, citations, and normalized references.
**Depends on**: Nothing (first phase)
**Requirements**: ING-01, ING-02, ING-03, ING-04, ING-05
**Success Criteria** (what must be TRUE):
  1. User can upload a PDF and see it available for analysis.
  2. User can view extracted metadata (title, authors, year) for the uploaded PDF.
  3. User can view in-text citations and bibliography entries extracted from the PDF.
  4. User can see resolved reference identifiers with confidence when available.
**Plans**: 4 plans
Plans:
- [x] 01-01-PLAN.md — Local ingestion storage and upload API
- [x] 01-02-PLAN.md — GROBID extraction and metadata/citation parsing
- [x] 01-03-PLAN.md — Reference resolution via Crossref
- [x] 01-04-PLAN.md — Streamlit ingestion UI

### Phase 2: Citation Context Navigation
**Goal**: Users can navigate citation context and follow cited works for validation.
**Depends on**: Phase 1
**Requirements**: NAV-01, NAV-02, NAV-03
**Success Criteria** (what must be TRUE):
  1. User can open citation context for a selected callout.
  2. User can follow a citation to load cited context into the validation pane.
  3. User can browse a citation tree of referenced and cited-by works when available.
**Plans**: 6 plans
Plans:
- [x] 02-01-PLAN.md — Backend citation context and graph endpoints
- [x] 02-02-PLAN.md — Streamlit callout navigation and citation tree UI
- [x] 02-03-PLAN.md — OpenAlex citation graph expansion wiring
- [x] 02-04-PLAN.md — Callout context selection fixes
- [x] 02-05-PLAN.md — GROBID consolidation + Crossref/OpenAlex mismatch resolution
- [x] 02-06-PLAN.md — Mismatch UI bibliography context + consolidation flag

### Phase 3: Claim Selection + Editing
**Goal**: Users can review and refine extracted claims from citing text.
**Depends on**: Phase 2
**Requirements**: CLM-01, CLM-02
**Success Criteria** (what must be TRUE):
  1. User can see candidate claims segmented from citing text.
  2. User can select and edit an extracted claim before validation.
**Plans**: 3 plans
Plans:
- [x] 03-01-PLAN.md — Claim queue state and segmentation helpers
- [x] 03-02-PLAN.md — Queue rendering and parsing confirmation UI
- [x] 03-03-PLAN.md — Modal claim editor with autosave and reset

### Phase 4: Evidence Attachment
**Goal**: Users can attach cited PDFs and prepare them for evidence retrieval.
**Depends on**: Phase 3
**Requirements**: EVD-01, EVD-02, EVD-03
**Success Criteria** (what must be TRUE):
  1. User can request retrieval instructions for a cited work from the claim context.
  2. User can drop a cited PDF onto a claim and see it attached.
  3. User can see the attached cited PDF parsed and ready for evidence search.
**Plans:** 6 plans
Plans:
- [x] 04-01-PLAN.md — Retrieval dossier API + claim UI action
- [x] 04-02-PLAN.md — Attachment queue UX with drop surfaces + accessibility fallback
- [x] 04-03-PLAN.md — Attachment persistence + parsing pipeline integration
- [x] 04-04-PLAN.md — Clipboard-backed retrieval instructions copy action
- [x] 04-05-PLAN.md — Attachment lifecycle statuses (pending → converting → parsing → matched)
- [x] 04-06-PLAN.md — Auto-match heuristics + manual reassignment controls

### Phase 5: Evidence Matching + Ranking
**Goal**: Users receive ranked evidence candidates that link claims to cited text.
**Depends on**: Phase 4
**Requirements**: EVD-04, EVD-05, EVD-06
**Success Criteria** (what must be TRUE):
  1. User can view evidence candidates matched to the selected claim.
  2. User sees a reranked list of top evidence options for the claim.
  3. User sees top-N evidence candidates with entail/contradict labels per ranking rules.
**Plans:** 6 plans
Plans:
- [x] 05-01-PLAN.md — Backend evidence pipeline foundation
- [x] 05-02-PLAN.md — Evidence service + FastAPI endpoints
- [x] 05-03-PLAN.md — Evidence store + claim sync wiring
- [x] 05-04-PLAN.md — Evidence board UI and rationale sidebar
- [x] 05-05-PLAN.md — Backend claim_text plumbing for evidence runs
- [x] 05-06-PLAN.md — Frontend claim_text wiring + reviewer guidance

### Phase 6: Evidence Review + Selection
**Goal**: Users can inspect evidence in-context and choose the best match.
**Depends on**: Phase 5
**Requirements**: EVD-07, EVD-08
**Success Criteria** (what must be TRUE):
  1. User can view paragraph-bounded excerpt previews for candidate evidence spans (with the nominated span highlighted).
  2. User can label candidates (supports/neutral/contradicts) and save an overall per-source assessment (supports/contradicts/inconsistent/silent).
  3. Saved assessment persists for the claim and is visible after reruns/reloads.
**Plans**: 2 plans
Plans:
- [x] 06-01-PLAN.md — Backend excerpt/jump APIs + selection persistence
- [x] 06-02-PLAN.md — Streamlit evidence review UI (excerpt previews + assessment)

### Phase 7: Validation + Export
**Goal**: Users can record verdicts, annotate them, and export results.
**Depends on**: Phase 6
**Requirements**: VAL-01, VAL-02, VAL-03, VAL-04
**Success Criteria** (what must be TRUE):
  1. User can record a verdict (support/contradict/uncertain) for a claim.
  2. User can add notes to a judgment.
  3. Citation callouts show validated vs unvalidated status.
  4. User can export judgments and metadata as CSV or JSON.
**Plans**: 3 plans
Plans:
- [x] 07-01-PLAN.md — Backend judgment persistence + export endpoints
- [x] 07-02-PLAN.md — Frontend judgment store + API helpers
- [x] 07-03-PLAN.md — Streamlit judgment UI, callout indicators, and export downloads

### Phase 8: Workspace + Fast-Path Review UX
**Goal**: Review is fast and interruption-free: citations are interacted with inline, sources are ingested in a global bin and processed in the background, and the UI is vertically dense without losing usability.
**Depends on**: Phase 7
**Requirements**: WS-01, WS-02, UX-01, UX-02, SEG-01, ATT-01, ATT-02, PDF-01, ML-01, ML-02
**Success Criteria** (what must be TRUE):
  1. In-text citation interactions are inline in the document text (underline/href-style is fine); end-of-block citation button rows are removed.
  2. If TEI/GROBID sentence boundaries are suspicious, the UI falls back to a local segmenter and still supports citation selection reliably.
  3. Users upload cited-source PDFs via a single global “Source bin” (no source-specific/per-claim dropzones).
  4. Newly uploaded sources are immediately queued for conversion + GROBID parsing in the background; the user does not wait at the point of review.
  5. After claims are saved, the system begins evidence runs automatically for the relevant sources so results are ready by the time the user opens the evidence view.
  6. Chasing claims view offers a “View PDF” affordance that copies a short excerpt snippet for Cmd-F (and opens the PDF).
  7. Layout is space-efficient: the big header is removed; settings are moved behind a gear/drawer; dense mode reduces vertical padding while keeping controls discoverable.
**Plans**: 8 plans
Plans:
- [x] 08-01-PLAN.md — Add robust TEI sentence fallback segmentation (SEG-01)
- [x] 08-02-PLAN.md — Add global background pause/resume state and wiring
- [x] 08-03-PLAN.md — Add backend Source Bin attachment endpoints (unassigned + archive + placement)
- [x] 08-04-PLAN.md — Rework Streamlit into dense 3-pane workspace shell (UX-02)
- [x] 08-05-PLAN.md — Make citations inline-clickable and remove end-of-block citation rows (UX-01)
- [x] 08-06-PLAN.md — Implement Source Bin UI + auto background processing + View PDF snippet UX (ATT-01/02, PDF-01)
- [x] 08-07-PLAN.md — Add execution profiles (Fast/Best) including ColBERT path (ML-01)
- [ ] 08-08-PLAN.md — Add optional Hugging Face Inference API mode for HF steps (ML-02)

### Phase 8.1: Recursive Retrieval + Node Graph (INSERTED)
**Goal**: Users can recursively chase a citation tree across multiple documents, with the node graph serving as a real navigation surface (not a dead-end tab).
**Depends on**: Phase 8
**Requirements**: NAV-04, GPH-01, DEC-01, DEC-02, DEC-03
**Success Criteria** (what must be TRUE):
  1. From a cited source, user can follow its citations to new targets and keep building the tree (recursive retrieval).
  2. Workspace holds multiple documents; switching documents preserves chase state and judgments.
  3. Node graph view is validated and integrated: selecting a node routes to the relevant doc/citation context and updates chase queue.
**Plans**: TBD

### Phase 8.2: Demo Dataset + Trace Replay (INSERTED)
**Goal**: A self-contained tech demo can show end-to-end citation-walking and annotation, including exports, without live external services.
**Depends on**: Phase 8.1
**Requirements**: DEMO-01
**Success Criteria** (what must be TRUE):
  1. A small PDF corpus (5–15 docs) with a known citation graph is packaged with the repo (or downloadable via a script).
  2. Ground-truth review artifacts exist (attachments parsed, evidence selections, judgments) to demonstrate the full workflow.
  3. A “trace” (event log) can replay a realistic walk + annotation session for the demo, and a Record mode can capture a real session into that trace format.
  4. Demo includes synthetic multi-actor graph behaviors: disputed-edge (orange) consensus with forward propagation; stacked supporting sources; and dotted contradiction links across alternate paths.
  5. Demo exports produce non-empty claim/callout JSON/CSV showing the captured judgments.
**Plans**: TBD

### Phase 9: Claim Graph + Multi-User Consensus
**Goal**: Review becomes collaborative: multiple local reviewer identities can record distinct judgments and vote/annotate claim-to-claim edges in an interactive claim graph with visible consensus vs disagreement.
**Depends on**: Phase 8
**Requirements**: MU-01, MU-02, CG-01, CG-02, CG-03
**Success Criteria** (what must be TRUE):
  1. A local "Current user" dropdown scopes all saved judgments, edge votes, and manual graph edits.
  2. Multiple users can disagree on the same claim-to-claim edge; the UI shows consensus vs disagreement with attributable votes/comments.
  3. Claim graph can be inspected interactively (zoom/pan; click edge opens a panel showing pro vs con annotations).
  4. Graph supports provenance layers: auto vs manual vs external_search (dotted).
  5. Compare mode allows selecting two reviewers and seeing side-by-side differences for both judgments and edge votes.
  6. MVP works with ~20 papers and multiple reviewer names without major usability/performance regressions.
**Plans**: 7 plans
Plans:
- [x] 09-01-PLAN.md — Persist reviewer identities in project metadata
- [x] 09-02-PLAN.md — Reviewer-scoped judgments + exports (backward compatible)
- [x] 09-03-PLAN.md — Claim graph store + edge votes + subgraph endpoints
- [x] 09-04-PLAN.md — Streamlit Current user dropdown + reviewer-scoped judgment UI
- [x] 09-05-PLAN.md — Interactive claim graph tab (renderer + edge inspection/voting)
- [x] 09-06-PLAN.md — Compare mode + end-to-end multi-user consensus verification checkpoint
- [x] 09-07-PLAN.md — Cytoscape Surfing POC: graph as architecture + plural segmentation + heat

### Phase 9.1: Ingestion replumbing + solid spine (INSERTED)
**Goal**: Ingestion has a stable execution spine (jobs/attempts/provenance/artifacts) and a portable runtime (Docker/Compose) so the system runs cleanly across host platforms and can move heavy services to remote Linux boxes.
**Depends on**: Phase 9
**Plans**: 15 plans

Plans:
- [x] 09.1-01-PLAN.md — Containerize API + UI with Compose (no behavior changes)
- [x] 09.1-02-PLAN.md — Add GROBID service to Compose + wire API config
- [x] 09.1-03-PLAN.md — Add Postgres + MinIO + DB/S3 wrappers + migrations
- [x] 09.1-04-PLAN.md — Dual-write ingest upload: filesystem + S3 + Postgres works
- [x] 09.1-05-PLAN.md — Primary extraction attempts/jobs + S3 artifacts + idempotency
- [x] 09.1-06-PLAN.md — Dev scripts + README + end-to-end verification checkpoint
- [x] 09.1-07-PLAN.md — Scope plumbing (project_id + user_id propagation)
- [x] 09.1-08-PLAN.md — Identity split (Work vs Document vs DocumentVersion)
- [x] 09.1-09-PLAN.md — Spine read default + legacy fallback
- [x] 09.1-10-PLAN.md — Settings/workflow versioning + safe edit surface
- [x] 09.1-11-PLAN.md — Stable locators primitives (entailment-ready)
- [x] 09.1-12-PLAN.md — Extraction reads PDFs from S3 (remove source.pdf runtime dependency)
- [x] 09.1-13-PLAN.md — Resolution artifacts in spine + spine-first resolution reads
- [x] 09.1-14-PLAN.md — Project membership list (project_documents) replaces filesystem scan
- [x] 09.1-15-PLAN.md — Spine-only mode + remove remaining filesystem read tails + wipe-all UX

**Details:**
- See `.planning/V2-REPLUMBING-PLAN.md` and `.planning/V2-PLANNING.md`.

### Phase 9.2: Ingestion automation robustness + fallback extraction (INSERTED)
**Goal**: Ingestion is resilient and automatable: background runs are idempotent and retry-safe, failures surface clearly, and extraction has a deterministic fallback path when primary parsing fails.
**Depends on**: Phase 9.1
**Plans**: 7 plans

**Cloud readiness note (post-9.2 hardening):** after fallback behavior is stable, add capability discovery (CPU/RAM/GPU) + derived concurrency defaults + strict readiness checks for cloud deployments (Azure-oriented). Spec: `.planning/architecture/CLOUD_CAPABILITIES_SPEC.md`.

Plans:
- [x] 09.2-01-PLAN.md — Corpus + regression fixtures for extraction failures
- [x] 09.2-02-PLAN.md — Fallback attempt contract (quality flags + S3 artifacts + DB pointers)
- [x] 09.2-03-PLAN.md — "Header 500 but refs OK" resilience + quality flags
- [x] 09.2-04-PLAN.md — PyMuPDF fallback extractor (text-layer) persisted as fallback attempts
- [x] 09.2-05-PLAN.md — Hybrid OCR worker service (per-page policy + adaptive DPI)
- [x] 09.2-06-PLAN.md — Orchestration: auto fallback on primary failure + manual force fallback + polling
- [x] 09.2-07-PLAN.md — Resolution gating based on refs quality + UI prompt

### Phase 9.3: Spine Everywhere + Legacy Removal (INSERTED)
**Goal**: Remove remaining local-disk persistence requirements by moving attachments, evidence, judgments, and project metadata onto the spine, and delete legacy filesystem stores.
**Depends on**: Phase 9.2
**Plans**: 6 plans

Plans:
- [x] 09.3-01-PLAN.md — Attachments persisted in spine (records in Postgres, blobs/artifacts in object store)
- [x] 09.3-02-PLAN.md — Judgments + exports persisted in spine (no data/judgments dependence)
- [x] 09.3-03-PLAN.md — Evidence runs + selections persisted in spine (history + artifacts)
- [x] 09.3-04-PLAN.md — Project metadata persisted in spine (reviewers/settings; no data/project.json)
- [x] 09.3-05-PLAN.md — Graph stores strategy (Postgres or rebuildable caches; no required local .db)
- [x] 09.3-06-PLAN.md — Delete legacy stores + remove compat flags; spine is the only runtime

**Details:**

- Verification script: `scripts/dev/verify_09_3_spine_cutover.sh`
- Verification doc: `.planning/phases/done/09.3-spine-everywhere-legacy-removal/09.3-VERIFICATION.md`

### Phase 10: Contracts + Core Workflow Simplification (INSERTED)
**Goal**: Make stage boundaries explicit (contracts) and simplify the core workflow so components can swap (e.g. rerankers) without rewiring callers.
**Depends on**: Phase 9.3

Plans:
- [x] `.planning/phases/10-contracts-core-workflow-simplification/done/10-01-PLAN.md` — Contracts arc (stage boundaries + artifact contracts)
- [x] `.planning/phases/10-contracts-core-workflow-simplification/done/10-02-PLAN.md` — Core workflow simplification (happy-path + minimal orchestrator)
- [x] `.planning/phases/10-contracts-core-workflow-simplification/done/10-03-PLAN.md` — Graph as navigation (Cytoscape routing + recursive retrieval)
- [x] `.planning/phases/10-contracts-core-workflow-simplification/done/10-04-PLAN.md` — Durable decisions/events (evidence decisions as events)
- [x] `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-01-01-PLAN.md` — Intake hardening (bytes-only uploads + project scoping + Source Inbox)
- [ ] 10-05-PLAN.md — Demo + trace replay (reproducible E2E)
- [ ] 10-06-PLAN.md — Optional add-ons (hosted inference, tri-level assessment)
- [ ] 10-07-PLAN.md — Deprioritized ops/cloud direction (doc-only backlog)

**Multi-user cloud note:** keep `project_id` and `created_by_user_id` first-class
in all spine tables and endpoints so OIDC (Azure Entra) can be layered on later.
Spec: `.planning/architecture/MULTI_USER_CLOUD_SPEC.md`.

### Post-9.2 Focus Notes

After Phase 9.2 stabilizes ingestion/extraction robustness, return to product UX stabilization in this order:

1) **Cytoscape reliability**: make the claim graph view fully functional and dependable (it was only partially working).
2) **Single settings pane**: consolidate settings into one comprehensive panel backed by the versioned settings/workflow spine (support swapping models/parameters; keep it non-graphical unless trivial).
3) **Real UX**: pursue a deliberate UX pass (and/or a UI stack upgrade) once the backend spine is boring and stable.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 8.1 → 8.2 → 9 → 9.1 → 9.2 → 9.3 → 10

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Ingestion + Extraction | 4/4 | Complete | 2026-01-23 |
| 2. Citation Context Navigation | 6/6 | Complete | 2026-01-24 |
| 3. Claim Selection + Editing | 3/3 | Complete | 2026-01-25 |
| 4. Evidence Attachment | 6/6 | Complete | 2026-01-27 |
| 5. Evidence Matching + Ranking | 6/6 | Complete | 2026-01-28 |
| 6. Evidence Review + Selection | 2/2 | Complete | 2026-02-01 |
| 7. Validation + Export | 3/3 | Complete | 2026-02-02 |
| 8. Workspace + Fast-Path Review UX | 7/8 | In progress | - |
| 8.1. Recursive Retrieval + Node Graph | 0/TBD | Not started | - |
| 8.2. Demo Dataset + Trace Replay | 0/TBD | Not started | - |
| 9. Claim Graph + Multi-User Consensus | 7/7 | Complete | 2026-02-09 |
| 9.1. Ingestion replumbing + solid spine | 15/15 | Complete | 2026-02-16 |
| 9.2. Ingestion automation robustness + fallback extraction | 7/7 | Complete | 2026-02-17 |
| 9.3. Spine Everywhere + Legacy Removal | 6/6 | Complete | 2026-02-20 |
| 10. Contracts + Core Workflow Simplification | 5/7 | In progress | - |
