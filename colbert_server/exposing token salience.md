Exposing Token-Level Salience in ColBERT

Why Isn’t There Any Word-Coloring Yet?

ColBERT computes per-token similarities (“MaxSim”) internally, but the default code collapses those vectors down to a single passage score before results ever reach your API/UI.  Concretely:
	1.	segmented_maxsim_cpp (C++ extension) returns the sum of the per-query-token maxima.
	2.	colbert_score_reduce() (Python) does the same in the non-C++ path.
	3.	Searcher.dense_search() only receives aggregated scores, so your FastAPI wrapper & Streamlit UI never see token_scores.

Result: your UI can’t highlight which words drove retrieval.

Two Ways to Get Token Salience

Path	Effort	Runtime Cost	Risk
Python fallback (recommended)	~2 h	One extra BERT doc-encode per k passages	Low
C++ + Faiss patch	3–5 days	Minimal	High (native code)

1 · Python-Only Fallback (Implemented Here)
	1.	Add a return_vector flag to colbert_score_reduce() → returns the per-query-token MaxSim vector instead of the summed score when requested.
	2.	In Searcher.dense_search()
	•	After ranking, fetch the k passage texts from the collection.
	•	Re-encode them with checkpoint.docFromText().
	•	Call model.score(..., return_vector=True) to get the token_scores tensor [k × query_len].
	•	Attach token_scores[i] as .tolist() for each result.
	3.	FastAPI & UI already pass through the new field, colouring words when present.

Pros: Zero C++ work; easy to maintain.  Cons: Adds ≈30 ms GPU (or ≈300 ms CPU) per query at k = 10.

2 · Native C++ Path (Not Implemented)
	•	Modify segmented_maxsim.cpp to return the full [ndocs × query_len] tensor.
	•	Patch Faiss IndexScorer.rank() to retain per-token vectors.
	•	Expose through pybind & Searcher.

Fast but substantially harder; recommended only if you need large-scale, low-latency serving.

Usage

searcher = Searcher(index="default", checkpoint=CP)

pids, ranks, scores, tok = searcher.search(
    "climate change impact on peanuts", k=10, include_token_scores=True
)
assert tok.shape == (10, searcher.config.query_maxlen)

	•	Turn off colouring by omitting include_token_scores (default False).

⸻

File-Level Patch Summary

File	Key Changes
colbert/modeling/colbert.py	colbert_score_reduce() + flag threading
colbert/searcher.py	Re-encode top-k docs, compute & return token_scores
colbert_server/colbert.py	Pass include_token_scores=True in /search
backend/hybrid.py	Preserve token_scores in metadata
frontend/ui.py	Use color_tokens() + st.markdown(..., unsafe_allow_html=True)


⸻

Checklist
	•	Python fallback code compiled and tests pass.
	•	/search JSON now includes "token_scores": [...].
	•	Streamlit UI highlights words when salience present.
	•	(Optional) Mitigate re-encode cost with batch caching or GPU.

⸻

© 2025 Blablador team
# Exposing Token‑Level Salience in **ColBERT**

ColBERT’s late‑interaction model *does* compute per‑token similarities (“MaxSim”) between query and document tokens, but **the stock code collapses those vectors to a single passage score** before any downstream component sees them.  
As a result, your Streamlit UI cannot colour words to show which tokens drove retrieval.

| Internal step | What happens | Where |
|---------------|--------------|-------|
| 1 | `segmented_maxsim_cpp` returns **sum of query‑token maxima** | `colbert/modeling/segmented_maxsim.cpp` |
| 2 | `colbert_score_reduce()` does the same for the Python path | `colbert/modeling/colbert.py` |
| 3 | `Searcher.dense_search()` receives only passage‑level scores | `colbert/searcher.py` |

---

## Two Ways to Surface Salience

| Path | Effort | Runtime Cost | Risk |
|------|--------|--------------|------|
| **Python fallback (recommended)** | ≈ 2 h | One extra BERT doc‑encode per *k* passages | **Low** |
| Native  C++ + Faiss patch | 3 – 5 days | Minimal | **High** (native code, rebuild) |

---

## 1 · Python‑Only Fallback (Implemented)

1. **Return the vector**  
   *Add `return_vector` flag* to `colbert_score_reduce()` so it can return the per‑query‑token MaxSim vector instead of the summed score.  
2. **Post‑rank re‑encode** in `Searcher.dense_search()`  
   * After FAISS ranking, fetch the *k* passage texts.  
   * Re‑encode them with `checkpoint.docFromText()`.  
   * Call `model.score(..., return_vector=True)` → tensor `[k × query_len]`.  
   * Attach `token_scores[i].tolist()` to each result object.  
3. **FastAPI & UI** need no change—`token_scores` now flows through and `ui.py` highlights with `color_tokens()`.

**Pros:** No C++ changes, easy to maintain.  
**Cons:** Adds ≈ 30 ms on GPU (≈ 300 ms on CPU) per query at *k = 10*.

---

## 2 · Native C++ Path (Not Implemented)

* Modify `segmented_maxsim.cpp` to return the full `[ndocs × query_len]` tensor.  
* Patch `IndexScorer.rank()` to propagate per‑token vectors.  
* Expose via pybind & extend `Searcher`.

Fastest at runtime but substantially harder to develop.

---

## Usage Example

```python
searcher = Searcher(index="default", checkpoint=CP)

pids, ranks, scores, tok = searcher.search(
    "climate change impact on peanuts",
    k=10,
    include_token_scores=True
)
assert tok.shape == (10, searcher.config.query_maxlen)
# UI will now colour words; omit include_token_scores to disable.
```

---

## File‑Level Patch Summary

| File | Key Changes |
|------|-------------|
| `colbert/modeling/colbert.py` | `colbert_score_reduce()` + flag plumbing |
| `colbert/searcher.py` | Re‑encode top‑*k* docs, compute & attach `token_scores` |
| `colbert_server/colbert.py` | Pass `include_token_scores=True` in `/search` |
| `backend/hybrid.py` | Preserve `token_scores` in metadata |
| `frontend/ui.py` | Use `color_tokens()` + `st.markdown(..., unsafe_allow_html=True)` |

---

## Checklist

- [x] Python fallback code compiles and tests pass.  
- [x] `/search` JSON now includes `"token_scores": [...]`.  
- [x] Streamlit UI highlights words when salience present.  
- [ ] *(Optional)* Batch‑cache re‑encode step or move to GPU to cut overhead.

---

© 2025 Blablador team