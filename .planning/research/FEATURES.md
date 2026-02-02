# Feature Research

**Domain:** Citation integrity workflows for academic PDF validation
**Researched:** 2026-01-23
**Confidence:** MEDIUM

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| PDF upload and metadata extraction | Users need to ingest their own papers; metadata is required to resolve citations. | MEDIUM | Leverage GROBID output and capture extraction confidence.
| Citation extraction and reference parsing | Core requirement to analyze citation integrity. | MEDIUM | Parse in-text citations and bibliography links.
| Reference resolution to canonical works | Users expect DOI/title matching to follow the citation tree. | HIGH | Use OpenAlex/Crossref-style metadata matching and dedupe.
| Citation tree navigation | Expected workflow: move from paper → references → citing papers. | MEDIUM | OpenAlex exposes referenced and cited-by relationships.
| Citation context viewing | Users must see the sentence/section where a citation is used. | MEDIUM | Surface surrounding text with section labeling.
| Claim segmentation | Validation requires claim-level granularity. | HIGH | Existing pipeline can drive this; store claim spans.
| Evidence retrieval from cited sources | Users expect to jump to source passages supporting the claim. | HIGH | Needs full-text access or OA fallback.
| Evidence highlighting and page anchors | Helps reviewers confirm support quickly. | MEDIUM | Track offsets/coordinates when possible.
| Human judgment capture | Validation workflows need a decision and rationale. | MEDIUM | Support verdict + notes + confidence.
| Export/reporting | Teams expect to export decisions for reporting. | LOW | CSV/JSON and per-claim reports.

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Smart citation classification (support/contradict/mention) | Speeds triage by showing whether citations support a claim. | HIGH | Scite highlights citation classification at scale; similar output is a differentiator when combined with claim-level validation.
| Claim-level citation context scoring | Aligns each claim to the strongest evidence passage with a confidence score. | HIGH | Uses NLI + reranking; aids reviewer focus.
| Multi-hop citation trails | Shows how claims propagate across the tree. | HIGH | Enables validation of indirect evidence chains.
| Disagreement heatmap across citations | Highlights conflicting evidence quickly. | MEDIUM | Aggregate support/contradict judgments by claim.
| Collaborative adjudication | Multiple reviewers, disagreement resolution, audit trail. | MEDIUM | Critical for institutional workflows.
| Alerts for retractions/errata | Maintains integrity over time. | MEDIUM | Requires linking to retraction data sources.
| Cross-source evidence search | Finds supporting/contradicting evidence beyond cited sources. | HIGH | Differentiates from citation-only tools.

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Fully automated accept/reject of claims | Saves reviewer time. | Creates false certainty; high error cost. | Human-in-the-loop workflow with confidence thresholds.
| Single “reliability score” without evidence | Simplifies reporting. | Hides nuance and misleads decision-makers. | Provide evidence-backed verdicts and context.
| Paywall scraping for full text | Maximizes recall. | Legal/compliance risk; brittle access. | Use OA sources, publisher APIs, and user-provided PDFs.
| Citation-count-as-truth ranking | Easy proxy for credibility. | Rewards popularity, not validity. | Rank by evidence quality and support/contradict signals.

## Feature Dependencies

```
[PDF upload + metadata]
    └──requires──> [Citation extraction + reference parsing]
                       └──requires──> [Reference resolution]
                                            └──enables──> [Citation tree navigation]

[Claim segmentation]
    └──requires──> [Evidence retrieval]
                       └──enables──> [Evidence highlighting]
                                            └──enables──> [Human judgment capture]

[Citation context viewing] ──enhances──> [Human judgment capture]

[Smart citation classification] ──enhances──> [Disagreement heatmap]
```

### Dependency Notes

- **Citation tree navigation requires reference resolution:** tree edges are only meaningful with normalized work IDs (eg, OpenAlex work IDs).
- **Evidence highlighting requires retrieval:** cannot anchor passages without retrieval results and offsets.
- **Judgment capture depends on context:** reviewers need citation context plus evidence passages to make reliable decisions.

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [ ] PDF upload + metadata extraction — enables ingestion.
- [ ] Citation extraction + reference resolution — unlocks navigation and linking.
- [ ] Citation tree navigation + context viewing — core workflow for reviewers.
- [ ] Claim segmentation + evidence retrieval — enables claim-level validation.
- [ ] Human judgment capture + export — validates whether users trust the workflow.

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] Evidence highlighting with precise anchors — after extraction quality is proven.
- [ ] Disagreement heatmap — once enough judgments exist.
- [ ] Collaboration/adjudication — when multiple reviewers are active.

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] Smart citation classification at scale — model training/labeling overhead.
- [ ] Multi-hop citation trails — heavy graph traversal and UI complexity.
- [ ] Retraction/errata alerts — requires external data integration.
- [ ] Open PDF at evidence location — from a candidate, open the local PDF at page/anchor (Preview/Skim integration) to speed verification.
- [ ] Source-scoped attachments — store one uploaded source PDF once, then link it to multiple claim segments/citations without duplicating processing.
- [ ] Phase 6 selection strictness — enforce primary-required for support/contradict and note-required for uncertain in the UI (backend already validates).
- [ ] True document-order chase queue — sort chased items by TEI paragraph/sentence position (not just citation index).
- [ ] Streamlit UX limitations — consider moving to a richer UI framework once core workflows stabilize (multi-pane interactions, inline click targets, complex layout control).

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Citation extraction + reference resolution | HIGH | HIGH | P1 |
| Citation tree navigation + context view | HIGH | MEDIUM | P1 |
| Claim segmentation + evidence retrieval | HIGH | HIGH | P1 |
| Human judgment capture + export | HIGH | MEDIUM | P1 |
| Evidence highlighting with anchors | MEDIUM | MEDIUM | P2 |
| Disagreement heatmap | MEDIUM | MEDIUM | P2 |
| Smart citation classification | HIGH | HIGH | P3 |
| Multi-hop citation trails | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | Competitor A (Scite) | Competitor B (OpenAlex-powered tools) | Our Approach |
|---------|----------------------|---------------------------------------|--------------|
| Citation context + classification | Provides Smart Citations with support/contradict labels and context. | Typically expose citation graph data but not per-claim validation. | Combine claim segmentation with evidence retrieval and NLI scoring.
| Citation graph navigation | Limited in Scite UI (citation report per paper). | Strong at references/cited-by links via OpenAlex work objects. | Provide citation tree with claim-level validations and reviewer judgments.
| Evidence anchoring | Scite shows citation context inside papers. | Usually metadata-only. | Provide claim-to-passage anchors in cited PDFs.

## Sources

- https://scite.ai/?referenceCheck=true (Smart Citations, citation context, classification) — MEDIUM confidence (official product page, marketing content)
- https://docs.openalex.org/api-entities/works (reference/cited-by relationships for citation graph) — HIGH confidence (official docs)

---
*Feature research for: citation integrity workflows*
*Researched: 2026-01-23*
