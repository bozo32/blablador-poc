# Phase 5: Evidence Matching + Ranking - Research

**Researched:** 2026-01-27
**Domain:** Hybrid evidence retrieval, reranking, and Streamlit UI instrumentation
**Confidence:** HIGH

## Summary

Phase 4 already delivers parsed windows, FAISS indices, and Hugging Face NLI scores (`backend/hybrid.py`). Phase 5 must turn those ingredients into a deterministic, auditable evidence list per claim: seed candidates from cited spans, cascade dense and late-interaction ranking, then surface the top entailing/contradicting snippets with full metadata and UI affordances. That requires enforcing deterministic cite matching (EVD-04), codifying rerank logic (EVD-05), and shaping the payload that feeds the Streamlit cards and rationale sidebar (EVD-06).

Industry-standard retrieval stacks follow a “retrieve → rerank” pattern: lexical/BM25 or dense bi-encoders fetch the top 50–100 spans, CrossEncoder/NLI models rescore them, and late-interaction rerankers like ColBERT retain token-level signals for downstream UI explanations.[^sbert][^colbert] FAISS provides the performant vector index and τ-filtering needed to keep latency acceptable even with 3-sentence sliding windows per attachment.[^faiss]

UI delivery hinges on metadata fidelity. PyMuPDF already exposes per-word bounding boxes, letting us map NLI-positive spans back to PDF page + section + coordinates for “Open in PDF” jumps and inline highlights.[^pymupdf] Streamlit custom components (or `components.html`) are the sanctioned way to render the rich cards, keyboard-friendly accept/reject controls, sparkline visualizations, and rationale sidebar without forking the frontend stack.[^streamlit]

**Primary recommendation:** Extend the existing hybrid pipeline so it emits structured `EvidenceCandidate` objects (deterministic cited span seed → FAISS/SBERT/ColBERT rerank → HF NLI label) with PyMuPDF-backed location metadata and rank deltas, and expose them through Streamlit components that mirror the decided card layout and sidebar interactions.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `faiss-cpu` | 1.8.x | ANN index for window embeddings + τ-filtering | Provides Flat/IP, IVF, HNSW, and PQ indexes so we can balance accuracy vs speed for 3-sentence windows before reranking.[^faiss]
| `sentence-transformers` | 3.x | Bi-encoder retrieval + CrossEncoder reranking | Official retrieve & rerank pipeline documentation, pretrained MSMARCO/QA models, and CrossEncoder APIs align with our Stage 2/3 scoring.[^sbert]
| `colbert-ai` | v0.2.0 | Late-interaction reranker + token salience | README documents token-level MaxSim, FAISS-backed end-to-end retrieval, and the requirement to preprocess/index once then reuse for queries.[^colbert]
| `transformers` | 5.0.0 | HF NLI sequence classification for entail/contrad labels | Provides `AutoModelForSequenceClassification` pipelines for `cross-encoder/nli-deberta-v3-base`, matching `backend/nli.py`.[^hf]
| `streamlit` | 1.38+ | Frontend for evidence cards, keyboard actions, rationale sidebar | Custom components + HTML embedding allow us to build the full-width cards, sidebar, and shortcuts without leaving Streamlit.[^streamlit]

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `rank-bm25` | 0.2.x | Deterministic lexical scoring seeded from citations | Use for EVD-04 deterministic cite-span matching before dense retrieval.[^bm25]
| `PyMuPDF` (`fitz`) | 1.26.7 | Page/section extraction, word-level bbox for PDF jumps | Needed to attach page, section, coordinate metadata and build “Open in PDF” links + highlights.[^pymupdf]
| `spaCy` (blank en) | 3.7.x | Tokenization for BM25 and heuristic provenance chips | Cheap tokenizer that matches `rank-bm25` expectations; reuse existing config from `backend/hybrid.py`.
| `fastcoref` | 2.x | Optional pronoun patching before ranking | Already wired in hybrid pipeline; keep to improve snippet readability before cards render.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| FAISS Flat/IP (`IndexFlatIP`) | Elasticsearch / OpenSearch lexical search | Lexical-only misses synonyms and acronyms; SentenceTransformers docs show dense retrieval overcomes those gaps for QA tasks.[^sbert]
| Streamlit components | Standalone React SPA | Would violate existing Streamlit deployment flow and duplicate routing/keyboard handling already available via `components.html`.[^streamlit]
| HF CrossEncoder rerank | Pure ColBERT end-to-end retrieval | ColBERT README recommends FAISS indexing + token MaxSim but still expects reranking or diversity logic when combining entail/contrad slots; HF NLI scores remain needed for labels.[^colbert][^hf]

**Installation:**
```bash
uv pip install faiss-cpu sentence-transformers==3.* colbert-ai==0.2.* transformers==5.* rank-bm25 PyMuPDF==1.26.7 spacy fastcoref streamlit
python -m spacy download en_core_web_sm  # optional heavier tokenizer
```

## Architecture Patterns

### Recommended Project Structure
```
backend/
├── evidence_matching/
│   ├── deterministic_matcher.py   # BM25 / cite-graph spans (EVD-04)
│   ├── rerank_pipeline.py         # FAISS → SBERT → ColBERT → HF NLI orchestration
│   └── serializers.py             # EvidenceCandidate DTOs with rank deltas + metadata
├── pdf_links/
│   └── locator.py                 # PyMuPDF helpers for page/section/coords
frontend/
├── components/
│   ├── evidence_card.py           # Streamlit component for cards + inline actions
│   └── rationale_sidebar.py       # Hover-synced sidebar with rank history + sparkline
└── state/
    └── evidence_store.py          # Manages pinned/filtered candidates + load-more state
```

### Pattern 1: Deterministic cite-span seeding (EVD-04)
**What:** Use citation graph metadata + `rank_bm25.BM25Okapi` to align each claim with its cited spans before dense retrieval. Tokenize both claim and snippet identically (spaCy blank `en`) and record the top matches plus BM25 scores/provenance badges.

**When to use:** Every time attachments shift or claim text changes; rerun before FAISS queries so deterministic matches can short-circuit to ranked cards when rerank scores tie.

**Example:**
```python
bm25 = BM25Okapi(tokenized_cited_spans)  # deterministic corpus
scores = bm25.get_scores(tokenize(claim_text))
seed_ids = np.argsort(scores)[::-1][: settings.SEED_SPANS]
```
[^bm25]

### Pattern 2: Retrieve → rerank cascade (FAISS + SBERT + CrossEncoder)
**What:** Embed windows with `SentenceTransformer`, normalize, query FAISS `IndexFlatIP`, τ-filter, then rerank with SBERT cosine and CrossEncoder/NLI scores before labelling.[^faiss][^sbert]

**When to use:** Default hybrid run, manual reruns, and anytime rank history must be exportable (EVD-05/EVD-06).

**Example:** see Code Examples → “Retrieve & rerank cascade”.

### Pattern 3: Late-interaction ranking & salience (ColBERT)
**What:** Feed τ-filtered windows into ColBERT (internal or external) to capture token-level MaxSim scores + per-token salience for sparkline visualization and rationale sidebar explanations.[^colbert]

**When to use:** When diversity or contradiction slots need strong justification, or when reviewers request token-level rationale.

### Anti-Patterns to Avoid
- **Unlimited CrossEncoder scoring:** SentenceTransformers docs warn CrossEncoders are slow; never pass more than ~100 windows or UI latency will spike.[^sbert]
- **Rebuilding ColBERT indexes per query:** ColBERT README prescribes preprocess → index → search; skipping reuse makes reruns impossibly slow.[^colbert]
- **Losing bbox metadata:** Without PyMuPDF `extractWORDS`, “Open in PDF” buttons cannot jump accurately.[^pymupdf]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Lexical cite matching | Custom TF-IDF scorer | `rank_bm25.BM25Okapi` | Already implements Okapi BM25, BM25L/+, and expects tokenized corpora; deterministic and battle-tested.[^bm25]
| Vector indexing | Custom ANN graph | FAISS `IndexFlatIP`/`IndexIVFPQ` | Provides exhaustive + IVF/HNSW options and quantization knobs without reimplementing ANN math.[^faiss]
| NLI labelling | Homemade classifier | `transformers` pipeline with `AutoModelForSequenceClassification` | HF pipeline automatically handles tokenizer/model alignment, batching, and label mapping, matching `backend/nli.py`.[^hf]
| PDF coordinate mapping | Regex on plain text | PyMuPDF `TextPage.extractWORDS()` | Returns per-word bbox + block references, enabling page/section metadata and highlight quads.[^pymupdf]
| Rich Streamlit UI | Manual JS injections | `streamlit.components.v1` custom component | Supported way to embed HTML/JS with bidirectional data flow and theme awareness.[^streamlit]

**Key insight:** Each requirement (deterministic match, rerank, metadata-rich cards) is already covered by mature libraries; duplicating them would add maintenance without improving accuracy.

## Common Pitfalls

### Pitfall 1: Letting CrossEncoder rerank too many candidates
**What goes wrong:** Passing hundreds of windows to the CrossEncoder causes second-stage latency spikes; SentenceTransformers docs explicitly note CrossEncoders are accurate but slow.[^sbert]
**Why it happens:** τ-filter or BM25 seed caps aren’t enforced prior to rerank.
**How to avoid:** Cap FAISS results (e.g., `RETRIEVAL_MAX=200`), τ-filter to `best * tau`, and restrict CrossEncoder inputs to <=100 windows. Surface a warning toast if rerun queues would exceed that.
**Warning signs:** Rerun duration >30s, UI skeletons linger, accept/reject controls remain disabled.

### Pitfall 2: Using the wrong FAISS index type per corpus size
**What goes wrong:** Keeping everything in `IndexFlatIP` scales poorly for large attachment batches; FAISS docs recommend IVF/PQ or HNSW once corpus >~1e5 vectors.[^faiss]
**Why it happens:** Developers stick with Flat indexes from prototypes.
**How to avoid:** Choose `IndexIVFFlat` or `IndexIVFPQ` when attachments exceed a few thousand windows, persist quantizers on disk, and tune `nlist` ≈ C * sqrt(n) as documented.[^faiss]
**Warning signs:** Memory pressure, slow rerun logs, CPU pegged during FAISS search.

### Pitfall 3: Dropping snippet location metadata
**What goes wrong:** Without PyMuPDF word bboxes, you cannot supply page/section placeholders, inline highlights, or “Open in PDF” coordinates (breaking the decided UI experience).[^pymupdf]
**Why it happens:** Only raw text is stored after parsing, so metadata chips end up blank.
**How to avoid:** At parse time, persist `page`, `section`, and `bbox` from `TextPage.extractWORDS()` along with TEI metadata; include them in `EvidenceCandidate` serialization.
**Warning signs:** UI shows placeholder “Unknown location”, PDF button disabled, reviewers file bug reports.

### Pitfall 4: Rebuilding ColBERT index per rerun
**What goes wrong:** ColBERT README mandates preprocess → index → search; rerunning indexing per claim wastes minutes and blocks rerun queue.[^colbert]
**Why it happens:** The rerank step doesn’t persist indexes or share them across claims.
**How to avoid:** Cache ColBERT index paths keyed by attachment batch + checkpoint, reuse them across claims, and only rebuild when attachments actually change.
**Warning signs:** ColBERT logs show repeated `colbert.index` runs per rerun, queue stalls.

## Code Examples

### Retrieve & rerank cascade
```python
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss, numpy as np

seed_bm25 = BM25Okapi(tokenize_all(cited_spans))
seed_scores = seed_bm25.get_scores(tokenize(claim))
seed_ids = np.argsort(seed_scores)[::-1][: settings.SEED_SPANS]

encoder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")  # bi-encoder
embeddings = encoder.encode([windows[i]["text"] for i in seed_ids], convert_to_numpy=True)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

q = encoder.encode([claim], convert_to_numpy=True)
faiss.normalize_L2(q)
D, I = index.search(q, k=100)

cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
candidates = [windows[seed_ids[j]] for j in I[0]]
scores = cross.predict([(claim, c["text"]) for c in candidates])
ranked = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
```
*Source: SentenceTransformers retrieve & rerank documentation highlights the two-stage pipeline (lexical/dense retrieval → CrossEncoder rerank).*

### Persisting FAISS IVF indexes
```python
import faiss, numpy as np

d = 768
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, 4096, 64, 8)  # IVF + PQ
index.train(passages_matrix)  # call once per attachment batch
index.add(passages_matrix)
index.nprobe = 16  # trade off accuracy vs latency
faiss.write_index(index, index_path)

# later
index = faiss.read_index(index_path)
scores, ids = index.search(query_vector, k=50)
```
*Source: FAISS wiki describes Flat, IVF, PQ indexes and recommends tuning `nlist` ≈ C·√n plus `nprobe` at query time for speed/accuracy control.*[^faiss]

### Mapping snippets to PDF coordinates
```python
import fitz  # PyMuPDF

doc = fitz.open(pdf_path)
page = doc.load_page(page_no)
words = page.get_text("words")  # equivalent to TextPage.extractWORDS

def find_span(text):
    matches = [w for w in words if text in w[4]]
    for x0, y0, x1, y1, word, block_no, line_no, word_no in matches:
        yield {
            "bbox": (x0, y0, x1, y1),
            "block": block_no,
            "line": line_no,
            "word": word,
        }

coords = list(find_span("glucose tolerance"))
```
*Source: PyMuPDF `TextPage.extractWORDS()` returns word-level bounding boxes and metadata, enabling “Open in PDF” links and inline highlights.*[^pymupdf]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-stage lexical (BM25 only) | Retrieve (lexical or dense) + CrossEncoder rerank | Documented in SentenceTransformers retrieve & rerank guide (rev. 2026) | Improves QA relevance while controlling latency by limiting CrossEncoder inputs.[^sbert]
| Single-vector reranker | ColBERTv2 late interaction with FAISS | ColBERT README (v0.2.0, 2022) | Adds token-level similarity + MaxSim, enabling confidence sparklines and richer rationales.[^colbert]

**Deprecated/outdated:**
- Pure Flat indexes for large attachment corpora — FAISS recommends IVF/PQ or HNSW once vectors exceed ~1e5 for acceptable speed.[^faiss]
- Plain text extraction without bounding boxes — PyMuPDF now exposes `extractWORDS`/`extractBLOCKS`, so older regex-based methods should be retired.[^pymupdf]

## Open Questions

1. **Contradiction slot fallback logic:** When fewer than three contradicting candidates exist, should we relax the “default 3 entailing + 3 contradicting” rule or duplicate entailing ones? Recommendation: planner to confirm desired UX; backend can expose both counts so UI decides.
2. **Rank-change history retention:** Requirements mention downloadable rank-change history. Do we persist every rerun delta per claim indefinitely, or only the most recent N reruns? Recommendation: default to last 5 reruns per claim, purge older history unless auditor mode enabled.

## Sources

### Primary (HIGH confidence)
- SentenceTransformers – Retrieve & Re-Rank documentation (https://www.sbert.net/examples/applications/retrieve_rerank/README.html) – pipeline guidance on bi-encoders vs CrossEncoder rerankers.[^sbert]
- FAISS Wiki – Index types and tuning guidance (https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).[^faiss]
- ColBERT README (https://raw.githubusercontent.com/stanford-futuredata/ColBERT/master/README.md) – late interaction, FAISS integration, version info.[^colbert]
- PyMuPDF TextPage docs (https://pymupdf.readthedocs.io/en/latest/textpage.html) – word-level bbox extraction and search APIs.[^pymupdf]
- Streamlit custom components guide (https://docs.streamlit.io/develop/concepts/custom-components/intro) – sanctioned approach for rich UI/keyboard behaviors.[^streamlit]
- Hugging Face Transformers text classification guide (https://huggingface.co/docs/transformers/tasks/sequence_classification) – using `AutoModelForSequenceClassification` for NLI pipelines.[^hf]
- `backend/hybrid.py` & `backend/nli.py` (repo) – current hybrid pipeline and NLI usage patterns.

### Secondary (MEDIUM confidence)
- `rank-bm25` README (https://raw.githubusercontent.com/dorianbrown/rank_bm25/master/README.md) – BM25 scoring usage examples.[^bm25]

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | Based on official FAISS, SentenceTransformers, ColBERT, PyMuPDF, Streamlit, and HF docs.
| Architecture | HIGH | Grounded in existing `backend/hybrid.py` plus vendor docs for each stage.
| Pitfalls | MEDIUM | Derived from vendor guidance and observed constraints; some remediation steps still need validation in situ.

**Research date:** 2026-01-27  
**Valid until:** 2026-02-26 (lib versions stable; revisit if FAISS/ColBERT release major updates)

[^sbert]: SentenceTransformers Retrieve & Rerank docs, https://www.sbert.net/examples/applications/retrieve_rerank/README.html
[^faiss]: FAISS index reference, https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
[^colbert]: ColBERT README, https://raw.githubusercontent.com/stanford-futuredata/ColBERT/master/README.md
[^hf]: Hugging Face Transformers text classification guide, https://huggingface.co/docs/transformers/tasks/sequence_classification
[^bm25]: rank-bm25 README, https://raw.githubusercontent.com/dorianbrown/rank_bm25/master/README.md
[^pymupdf]: PyMuPDF TextPage docs, https://pymupdf.readthedocs.io/en/latest/textpage.html
[^streamlit]: Streamlit custom components intro, https://docs.streamlit.io/develop/concepts/custom-components/intro
