# Pitfalls Research

**Domain:** Citation integrity workflows for academic PDF validation
**Researched:** 2026-01-23
**Confidence:** MEDIUM

## Critical Pitfalls

### Pitfall 1: Treating citation extraction as ground truth

**What goes wrong:**
Extraction errors (missing or mis-parsed references, incorrect fields) propagate into the citation graph and lead to false validation outcomes.

**Why it happens:**
Teams assume the PDF parser is deterministic and accurate for all publishers and layouts, despite reported extraction F1 scores below 1.0.

**How to avoid:**
Track extraction confidence, keep raw reference strings, and add a human correction path for low-confidence extractions. Sample QA against the PDF to calibrate acceptable error rates.

**Warning signs:**
Reference counts differ from the PDF, DOI hit rates are low, or the same reference resolves to multiple records.

**Phase to address:**
Phase 1 (Ingestion + extraction calibration).

---

### Pitfall 2: Leaving GROBID on default CRF models

**What goes wrong:**
Citation parsing quality is lower than expected, especially for references and headers, leading to poor matching and recall.

**Why it happens:**
GROBID ships with CRF models by default; deep learning models must be explicitly enabled and configured.

**How to avoid:**
Enable recommended deep learning models for citation parsing and benchmark on representative PDFs. Document the accuracy/latency tradeoff and bake it into deployment requirements.

**Warning signs:**
Consistently missing DOIs, low recall on known references, or big improvements when running the same PDF through a DL-enabled test instance.

**Phase to address:**
Phase 1 (Extraction stack selection and configuration).

---

### Pitfall 3: Mislinking citation callouts to bibliography entries

**What goes wrong:**
Claims are validated against the wrong cited work because citation context resolution is imperfect, especially in dense references or multi-citation callouts.

**Why it happens:**
Teams assume citation context linking is exact, but reported F1 ranges show non-trivial error rates.

**How to avoid:**
Persist the callout-to-reference mapping with confidence, expose context for user verification, and allow manual remapping when ambiguous.

**Warning signs:**
Users flag “wrong paper” errors, or a high rate of conflicts between automated resolution and user judgment.

**Phase to address:**
Phase 2 (Claim-to-citation alignment and review workflow).

---

### Pitfall 4: Naive consolidation against Crossref at scale

**What goes wrong:**
Metadata enrichment is flaky or blocked; rate limiting/bans lead to missing identifiers, and the system silently degrades.

**Why it happens:**
Crossref REST API has rate limits and requires a mailto for reliable service; timeouts and aggressive concurrency can cause failures.

**How to avoid:**
Throttle and cache Crossref calls, include mailto in requests, and plan for a local consolidation service (e.g., biblio-glutton) if throughput increases.

**Warning signs:**
Spike in timeout errors, sudden drops in DOI resolution, or inconsistent enrichment success across batches.

**Phase to address:**
Phase 2 (Metadata enrichment + consolidation strategy).

---

### Pitfall 5: Over-trusting automated entailment verdicts

**What goes wrong:**
NLI or LLM scores are treated as final validation, producing false positives/negatives and eroding researcher trust.

**Why it happens:**
Automated models are attractive for scale, but they struggle with domain nuance, missing context, and non-textual evidence (figures, tables).

**How to avoid:**
Make model outputs advisory, require human confirmation for “unsupported” or “contradicted” judgments, and log disagreement rates for continuous calibration.

**Warning signs:**
High user override rates, low inter-annotator agreement, or repeated disputes on model-labeled decisions.

**Phase to address:**
Phase 3 (Human-in-the-loop validation and QA).

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Store only normalized identifiers (DOI/PMID) and discard raw references | Smaller storage footprint | Hard to debug parsing errors, impossible to re-run resolution without re-parsing PDFs | Only for short-lived prototypes |
| Skip versioning of PDFs, models, and extraction outputs | Faster iteration | Validation results become non-reproducible and unauditable | Never for production validation |
| Hardcode a single retrieval corpus | Simple pipeline | Biased “unsupported” outcomes when the cited paper is outside the corpus | MVP only, with explicit coverage warnings |

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| GROBID service | Leaving CRF defaults and assuming best accuracy | Enable recommended DL models; validate on target PDFs |
| Crossref REST API | No mailto, high concurrency, or low timeouts | Use polite pool, mailto, caching, and exponential backoff |
| biblio-glutton | Underestimating setup cost and data indexing | Plan infrastructure and index build time before scaling |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-citation external API calls | Long tail latency, rate-limit errors | Batch, cache, and prefetch metadata | Hundreds of PDFs per day |
| CPU-only DL parsing | Extraction queues grow, timeouts | Use GPU or mixed CRF/DL models | Tens of PDFs per hour |
| Re-parsing PDFs for every re-run | Duplicate compute, slow iteration | Persist TEI/structured outputs and re-use | As soon as multi-step workflows begin |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Sending full PDFs to third-party services without consent | Copyright or data leakage | Limit external calls to metadata-only APIs; document data handling |
| Exposing reviewer annotations publicly by default | Sensitive research notes leak | Use private-by-default permissions and audit logs |
| Storing uploaded PDFs indefinitely | Unclear data retention obligations | Add retention policy and deletion workflows |

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Presenting a binary “supported/unsupported” label | Users can’t judge nuance | Show evidence passages, uncertainty, and alternative sources |
| Hiding extraction confidence | Users over-trust system outputs | Expose confidence bands and allow manual fixes |
| No traceability from claim to PDF region | Users can’t verify context | Provide PDF coordinates and citation callouts |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Citation graph:** Often missing DOI consolidation — verify identifier resolution rate on a gold set
- [ ] **Claim validation:** Often missing reviewer override flow — verify manual correction is captured and audited
- [ ] **Extraction pipeline:** Often missing model configuration — verify GROBID DL models enabled and documented
- [ ] **Evidence retrieval:** Often missing coverage limits — verify corpus coverage is explicit in UI

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Extraction errors propagate | MEDIUM | Re-run parsing with DL models, update references, and backfill affected validations |
| Mislinked citations | HIGH | Rebuild callout-to-reference mapping and re-evaluate affected claims |
| Crossref throttling | MEDIUM | Queue and replay enrichment jobs with backoff and mailto |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Treating extraction as ground truth | Phase 1 | QA sample matches PDF reference counts and DOI recall benchmarks |
| Default CRF models | Phase 1 | Benchmark shows DL models enabled with measurable F1 gain |
| Mislinking citation callouts | Phase 2 | User review detects <5% mislink rate on sampled claims |
| Naive Crossref consolidation | Phase 2 | Enrichment logs show stable success rate under load |
| Over-trusting NLI verdicts | Phase 3 | Human override rate tracked and used to tune model thresholds |

## Sources

- https://grobid.readthedocs.io/en/latest/Deep-Learning-models/ (default CRF vs DL models, accuracy tradeoffs)
- https://grobid.readthedocs.io/en/latest/Consolidation/ (consolidation services, rate limits, mailto requirement)
- https://github.com/grobidOrg/grobid (citation extraction accuracy and citation context resolution F1 metrics)
- Practitioner experience with citation validation workflows (LOW confidence)

---
*Pitfalls research for: citation integrity workflows in academic PDF validation*
*Researched: 2026-01-23*
