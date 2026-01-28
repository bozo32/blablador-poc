# OS-ERIN

## What This Is

OS-ERIN is a web app for researchers to upload an academic PDF, follow its citation tree, and validate whether cited sources actually support specific claims. It combines an in-document reading experience with a right-hand validation workspace that extracts claims, retrieves candidate evidence, and records user judgments.

## Core Value

Readers can validate a citation by linking a claim to supporting (or contradicting) evidence from the cited source, with the judgment captured and reusable.

## Requirements

### Validated

- ✓ Segment input text into candidate claims for review — existing pipeline
- ✓ Retrieve candidate evidence spans using hybrid retrieval and reranking — existing pipeline
- ✓ Score evidence candidates with NLI to support/contradict/neutral labels — existing pipeline

### Active

- [ ] PDF reader with clickable in-text citations as the primary workspace
- [ ] Right-click citation menu to follow and load cited context into the validation pane
- [ ] Citing-text panel that extracts claims and lets users select and edit parses
- [ ] Cited-source panel that accepts dropped PDFs, validates metadata/DOI, and processes silently
- [ ] Validation workflow that presents candidate spans and records user judgment + annotations
- [ ] Citation status indicators showing validated vs unvalidated state
- [ ] Copy-to-clipboard citation retrieval info (author/year/DOI)
- [ ] Left-pane project browser for folder-based PDF collections (single paper v1)
- [ ] Left-pane codebook for user-applied tags with future REFI-QDA export compatibility

### Out of Scope

- EOSC node governance artifacts and coalition planning — tracked in CIT-EOSC docs, not this app
- Large-corpus indexing or batch validation across hundreds of papers — defer beyond v1
- Fully automated validation without human judgment — requires user adjudication

## Context

- Brownfield codebase includes a FastAPI backend, Streamlit UI, and ColBERT reranker service for claim segmentation and evidence retrieval.
- Target workflow: Grobid-powered PDF reading experience with citation follow/validation actions.
- Project aligns with the CIT-EOSC vision for claim-level integrity, but this repo focuses on software delivery.

## Constraints

- **Deployment**: Local development is fine; user-facing delivery should be a web app due to processing needs.
- **Data protection**: EU data protection expectations; avoid unnecessary external data transfer.
- **Dependencies**: Grobid assumed available locally; blablador LLM API used but must remain replaceable.
- **ML stack**: Prefer Hugging Face models for most pipeline steps.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Project name is OS-ERIN | New identity for the product | — Pending |
| v1 success = end-to-end citation validation | Focus on complete user workflow | — Pending |
| Rework pipeline as needed | Existing flow needs improvement for new UX | — Pending |
| Primary users are researchers | Design for scholarly review workflows | — Pending |
| Persist attachment claim_text with metadata | Enables reruns to reuse the UI-provided text without manual seeding | Implemented (05-05) |
| Resolve claim_text via attachment metadata before failing | Avoids 409 errors on the first evidence fetch when text already exists | Implemented (05-05) |

---
*Last updated: 2026-01-28 after 05-05 plan*
