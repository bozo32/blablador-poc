---
phase: 05-evidence-matching-ranking
plan: 01
subsystem: api
tags: [retrieval, bm25, faiss, nli]

# Dependency graph
requires:
  - phase: 04-evidence-attachment
    provides: attachment artifacts and sentence caches
provides:
  - Deterministic cite-span seeding with provenance metadata
  - EvidencePipeline orchestration plus JSON serializers
affects:
  - 05-02 evidence service endpoints
  - 05-03 evidence store + claim sync

# Tech tracking
tech-stack:
  added: []
  patterns:
    - EvidenceCandidate DTOs used across loader → matcher → pipeline stages
    - Attachment sentence caching reused by downstream retrieval

key-files:
  created:
    - backend/evidence_matching/__init__.py
    - backend/evidence_matching/types.py
    - backend/evidence_matching/loaders.py
    - backend/evidence_matching/deterministic_matcher.py
    - backend/evidence_matching/pipeline.py
    - backend/evidence_matching/serializers.py
    - tests/test_evidence_matching_pipeline.py
  modified:
    - backend/attachment_store.py
    - backend/settings.py

key-decisions:
  - Cached attachment sentences centrally so deterministic windows never reread disk artifacts.
  - Added spaCy-backed tokenization with regex fallback to keep BM25 seeding deterministic even when spaCy is unavailable.
  - Routed the evidence pipeline through a lightweight scoring combiner that still surfaces FAISS/SBERT/ColBERT slots and hands final labelling to the existing NLI module.

patterns-established:
  - EvidenceCandidate serialization contract (dict payload + highlights + scores)
  - Deterministic seed provenance chips (cited vs heuristic) available for UI badges

# Metrics
duration: 20 min
completed: 2026-01-28
---

# Phase 05 Plan 01: Evidence pipeline foundation Summary

**Deterministic BM25 seeding, EvidencePipeline orchestration, and JSON serializers now deliver ranked EvidenceCandidate payloads for downstream APIs.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-01-28T07:07:43Z
- **Completed:** 2026-01-28T07:28:11Z
- **Tasks:** 3
- **Files modified:** 9
- **Verification:**
  - `pytest tests/test_evidence_matching_pipeline.py -k loader`
  - `pytest tests/test_evidence_matching_pipeline.py -k matcher`
  - `pytest tests/test_evidence_matching_pipeline.py -k pipeline`
  - `pytest tests/test_evidence_matching_pipeline.py`

## Accomplishments

- Introduced EvidenceCandidate/CandidateSpan/RankScores dataclasses with attachment-aware loaders backed by a sentence cache.
- Delivered deterministic cite-span seeding that uses spaCy-tokenized BM25 scores, provenance chips, badges, and tunable thresholds surfaced via settings.
- Implemented EvidencePipeline + serializer utilities so seeds can be reranked, labelled via NLI, limited per settings, and emitted as JSON-friendly payloads with location metadata.

## Task Commits

1. **Task 1: Define evidence candidate data structures and loaders** - `a83623d`
2. **Task 2: Implement deterministic cite-span seeding** - `cb3503c`
3. **Task 3: Compose retrieve → rerank → NLI pipeline** - `90ce38f`

## Files Created/Modified

- `backend/evidence_matching/types.py` – EvidenceCandidate/CandidateSpan/RankScores DTOs and helpers.
- `backend/evidence_matching/loaders.py` – Attachment window builder with deterministic window IDs and tokenization.
- `backend/attachment_store.py` – Sentence cache loader + helpers for matched attachments.
- `backend/evidence_matching/deterministic_matcher.py` – SpaCy/BM25 seeding with provenance badges and tunables.
- `backend/settings.py` – Evidence-specific caps/thresholds exposed alongside attachment and retrieval settings.
- `backend/evidence_matching/pipeline.py` – EvidencePipeline scoring + NLI integration plus label summaries.
- `backend/evidence_matching/serializers.py` – JSON serializers with snippet trimming, bbox counts, and highlights.
- `tests/test_evidence_matching_pipeline.py` – Regression coverage for loaders, matcher, pipeline ordering, and serialization.

## Decisions Made

- Cached attachment sentences centrally and exposed `load_sentences_for_attachment` so all deterministic windows reuse the same artifact rather than rereading ndjson files.
- Defaulted BM25 tokenization to spaCy blank `en` with a regex fallback to keep seeds deterministic even when spaCy is absent in constrained environments.
- Added a best-effort HybridPipeline bootstrap hook and simple score combiner so the pipeline can run in lightweight unit tests while still aligning with the planned FAISS/SBERT/ColBERT cascade.

## Deviations from Plan

None – plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- Evidence candidates can now be requested directly from the pipeline, so Plan 05-02 can focus on exposing FastAPI endpoints without rebuilding retrieval logic.
- Serializer + settings caps ensure UI/streamlit consumers can ingest ranked candidates immediately once the evidence service is wired up.

---
*Phase: 05-evidence-matching-ranking*
*Completed: 2026-01-28*
