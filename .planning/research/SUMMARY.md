# Project Research Summary

**Project:** OS-ERIN
**Domain:** Citation integrity and claim validation workflows for academic PDFs
**Researched:** 2026-01-23
**Confidence:** MEDIUM

## Executive Summary

OS-ERIN is a citation integrity web app that lets researchers ingest academic PDFs, follow citation trees, and validate whether cited sources support specific claims. Experts build these systems with a structured PDF extraction pipeline (GROBID), normalized citation graphs, and a human-in-the-loop review workflow that anchors claims to evidence passages with provenance. The recommended approach prioritizes reliable extraction and reference resolution first, then claim segmentation, retrieval/reranking, and review UX with evidence snapshotting for auditability.

The research points to a stack anchored on GROBID 0.8.2 for TEI/XML extraction, PostgreSQL 17.2 + pgvector 0.8.1 for metadata + embeddings, and PyTorch/Transformers for claim extraction and NLI scoring. This enables an async pipeline with checkpointed outputs, graph-backed citation navigation, and an evidence-first review surface using PDF.js for in-document context. Advanced differentiators (citation classification, multi-hop trails) should follow after the core workflow proves trustworthy.

Key risks cluster around extraction and provenance: assuming citation parsing is perfect, leaving GROBID on default CRF models, mislinking callouts to bibliography entries, and over-trusting automated entailment. Mitigate by enabling DL models, tracking extraction confidence, exposing callout-to-reference mappings for manual correction, snapshotting evidence, and keeping model outputs advisory with human confirmation.

## Key Findings

### Recommended Stack

The stack is optimized for structured academic PDFs and auditable claim validation. GROBID provides citation-aware TEI/XML, PostgreSQL + pgvector keep metadata and embeddings co-located, and PyTorch/Transformers support NLI and claim extraction. Use PDF.js for precise in-browser citation context and reserve Qdrant only if vector volume outgrows Postgres.

**Core technologies:**
- GROBID 0.8.2: scholarly PDF parsing and citation extraction — de facto standard for structured references.
- PostgreSQL 17.2 + pgvector 0.8.1: system of record and vector search — strong provenance joins and MVP ops simplicity.
- PyTorch 2.10.0 + Transformers 4.57.6: claim extraction and NLI scoring — broad model support and research-grade tooling.
- PDF.js 5.4.530: in-browser PDF rendering — robust text-layer alignment for citation context.

### Expected Features

MVP centers on ingestion, citation parsing, citation tree navigation, claim segmentation, evidence retrieval, and human judgments with export. Differentiators add automated citation classification, multi-hop trails, disagreement analysis, and collaboration, while v2+ defers heavy classification and retraction alerts.

**Must have (table stakes):**
- PDF upload + metadata extraction — entry point for all workflows.
- Citation extraction + reference resolution — enables citation tree and normalization.
- Citation tree navigation + context viewing — primary reviewer workflow.
- Claim segmentation + evidence retrieval — needed for claim-level validation.
- Human judgment capture + export — validates trust and audit needs.

**Should have (competitive):**
- Claim-level context scoring — focus reviewers on strongest evidence.
- Disagreement heatmap — surface conflicting evidence.
- Collaborative adjudication — required for institutional workflows.

**Defer (v2+):**
- Smart citation classification at scale — high labeling/training overhead.
- Multi-hop citation trails — complex graph + UI burden.
- Retraction/errata alerts — external data dependencies.

### Architecture Approach

The architecture favors an async pipeline with persisted checkpoints, a graph-backed citation store, and evidence snapshotting for auditability. Processing services (extraction, retrieval, validation) are isolated behind an orchestration API, and the UI integrates PDF viewing with claim review and judgment capture.

**Major components:**
1. PDF ingestion + GROBID extraction — store PDFs, parse TEI/XML, extract citations and contexts.
2. Reference resolution + citation graph service — normalize identifiers and serve citation trees.
3. Claim parsing + retrieval/reranking + NLI validation — produce evidence candidates and model scores.
4. Review UI + judgment store — human validation with evidence snapshots and provenance.

### Critical Pitfalls

1. **Treating extraction as ground truth** — track confidence, keep raw references, add manual correction paths.
2. **Leaving GROBID on default CRF models** — enable DL models and benchmark on target PDFs.
3. **Mislinking citation callouts** — persist mappings with confidence and allow user remapping.
4. **Naive Crossref consolidation at scale** — throttle, cache, include mailto, and plan backoff.
5. **Over-trusting NLI verdicts** — keep model outputs advisory and require human confirmation.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Ingestion + Extraction Calibration
**Rationale:** Downstream features depend on accurate structured text and normalized references.
**Delivers:** PDF upload, GROBID TEI/XML extraction, extraction confidence, DOI/reference normalization, citation graph scaffolding.
**Addresses:** PDF upload + metadata extraction; citation extraction + reference parsing; reference resolution.
**Avoids:** Treating extraction as ground truth; default CRF model risk.

### Phase 2: Claim Validation Workflow
**Rationale:** Claim segmentation, retrieval, and review UX build on the citation graph and structured text.
**Delivers:** Claim segmentation, evidence retrieval/reranking, citation context viewing, review UI with judgments and exports.
**Uses:** PostgreSQL + pgvector; PyTorch/Transformers; PDF.js.
**Avoids:** Mislinking callouts; naive Crossref consolidation under load.

### Phase 3: Trust + Collaboration Enhancements
**Rationale:** Differentiators require sufficient data volume and established core workflow.
**Delivers:** Evidence highlighting anchors, disagreement heatmaps, collaborative adjudication, initial model calibration feedback loops.
**Implements:** Evidence snapshotting and provenance-driven analytics.
**Avoids:** Over-trusting automated entailment verdicts.

### Phase 4: Advanced Intelligence + External Signals
**Rationale:** Smart classification and multi-hop trails depend on stable data and higher quality labels.
**Delivers:** Smart citation classification, multi-hop citation trails, retraction/errata alerts.
**Avoids:** Over-automation without evidence, reliance on fragile external data sources.

### Phase Ordering Rationale

- Citation extraction and reference resolution are prerequisites for citation tree navigation and claim-level linking.
- Claim validation must precede collaboration and analytics because it generates the judgments those features rely on.
- Advanced classification and multi-hop trails depend on labeled evidence and a stable provenance model.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** Evidence retrieval coverage and corpus access; validation of Crossref/OpenAlex rate-limit behavior.
- **Phase 4:** Smart citation classification models and multi-hop graph UX patterns.

Phases with standard patterns (skip research-phase):
- **Phase 1:** GROBID-based extraction + DOI normalization are well-documented.
- **Phase 3:** Collaboration workflows and disagreement aggregation are established patterns.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official release notes for core components and established tooling. |
| Features | MEDIUM | Competitor references plus inferred workflows; some assumptions need validation. |
| Architecture | MEDIUM | Standard patterns, but integration details are project-specific. |
| Pitfalls | MEDIUM | GROBID and Crossref docs are strong; practitioner notes need validation. |

**Overall confidence:** MEDIUM

### Gaps to Address

- Full-text evidence access: confirm coverage strategy (OA sources vs user-provided PDFs) and UI messaging.
- Citation context accuracy: benchmark callout-to-reference linking on target PDFs.
- Model calibration: define thresholds and human override metrics before scaling NLI usage.
- Data retention and privacy controls: align with EU expectations and permissions design.

## Sources

### Primary (HIGH confidence)
- https://github.com/kermitt2/grobid/releases — GROBID 0.8.2 release details.
- https://www.postgresql.org/docs/release/17.2/ — PostgreSQL 17.2 release notes.
- https://github.com/pgvector/pgvector/tags — pgvector 0.8.1 tag.
- https://github.com/pytorch/pytorch/releases — PyTorch 2.10.0 release.
- https://github.com/huggingface/transformers/releases — Transformers 4.57.6 release.
- https://github.com/mozilla/pdf.js/releases — PDF.js 5.4.530 release.
- https://docs.openalex.org/api-entities/works — OpenAlex citation graph API.
- https://www.crossref.org/documentation/retrieve-metadata/rest-api/ — Crossref REST API docs.
- https://grobid.readthedocs.io/en/latest/Grobid-service/ — GROBID service API.

### Secondary (MEDIUM confidence)
- https://scite.ai/?referenceCheck=true — Smart citation classification feature reference.
- https://api.semanticscholar.org/api-docs/graph — Semantic Scholar Graph API docs.
- https://github.com/stanford-futuredata/ColBERT — late-interaction retrieval reference.

### Tertiary (LOW confidence)
- Practitioner experience with citation validation workflows — requires validation.

---
*Research completed: 2026-01-23*
*Ready for roadmap: yes*
