# backend/hybrid.py


from typing import Any, Dict
from backend.settings import Settings


class HybridPipeline:
    @staticmethod
    def build_all(
        folder,
        embed_model,
        max_sentences,
        min_score,
        settings: Settings = None,  # accept explicit or global settings
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Hybrid pipeline: full scaffold for stepwise development.
        Each stage includes a progress marker and placeholder for logic.
        See development steps for details on each segment.
        """
        if settings is None:
            from backend.settings import Settings as _Settings

            settings = _Settings()

        # --- [STEP 1] TEI segmentation and windowing ---
        print("[HYBRID][STEP 1] Segmenting TEI and generating windows...")
        import os
        from pathlib import Path
        from backend.parser import tei_to_chunks

        # FIXME: parser.py should eventually flag or filter citation-only or junk <s> elements,
        # since window size=2 here will sometimes produce windows with only one "real" sentence.
        window_size = 2

        # Find the first .xml file in the folder
        tei_files = [f for f in os.listdir(folder) if f.endswith(".xml")]
        if not tei_files:
            raise FileNotFoundError(f"No TEI XML file found in folder: {folder}")
        tei_path = Path(folder) / tei_files[0]

        # Parse all sliding windows (sizes 1, 2, 3) from TEI
        all_windows = tei_to_chunks(tei_path)

        # Keep only those windows of the configured size
        windows = [
            {
                "text": w["text"],
                "tei_ids": w["meta"].get("sent_ids", []),
                "window_id": w.get("id"),
                "meta": w.get("meta", {}),
            }
            for w in all_windows
            if w["meta"].get("window_size") == window_size
        ]

        print(
            f"[HYBRID][STEP 1] Complete: {len(windows)} windows generated of size {window_size}."
        )
        if windows:
            print(f"[HYBRID][STEP 1] First window sample: {windows[0]}")

        # --- [STEP 2] First-pass SBERT/BM25 filtering ---
        print("[HYBRID][STEP 2] Running first-pass SBERT/BM25 filtering...")

        claim = kwargs.get("claim", None)
        if not claim or not isinstance(claim, str):
            raise ValueError(
                "You must provide a claim (string) as 'claim=...' in kwargs to build_all."
            )

        # Config
        use_sbert = getattr(settings, "HYBRID_USE_SBERT", True)
        use_bm25 = getattr(settings, "HYBRID_USE_BM25", True)
        filter_logic = getattr(settings, "HYBRID_FILTER_LOGIC", "both")
        sbert_threshold = getattr(settings, "HYBRID_SBERT_THRESHOLD", 0.10)
        bm25_threshold = getattr(settings, "HYBRID_BM25_THRESHOLD", 0.20)
        sbert_model_name = getattr(
            settings, "SBERT_MODEL_NAME", "all-MiniLM-L6-v2"
        )  # update if you want to pick this from settings
        from sentence_transformers import SentenceTransformer, util
        from rank_bm25 import BM25Okapi

        # SBERT scoring
        if use_sbert or filter_logic in ("both", "sbert"):
            print("[HYBRID][STEP 2] Loading SBERT model...")
            sbert_model = SentenceTransformer(sbert_model_name)
            window_texts = [w["text"] for w in windows]
            claim_embedding = sbert_model.encode([claim], convert_to_tensor=True)
            window_embeddings = sbert_model.encode(window_texts, convert_to_tensor=True)
            cosine_scores = (
                util.pytorch_cos_sim(claim_embedding, window_embeddings)[0]
                .cpu()
                .tolist()
            )
        else:
            cosine_scores = [0.0 for _ in windows]

        # BM25 scoring
        if use_bm25 or filter_logic in ("both", "bm25"):
            print("[HYBRID][STEP 2] Running BM25 ranking...")
            import spacy

            nlp = spacy.load("en_core_web_sm")

            window_tokens = [[tok.text for tok in nlp(w["text"])] for w in windows]
            bm25 = BM25Okapi(window_tokens)
            claim_tokens = [tok.text for tok in nlp(claim)]
            bm25_scores = bm25.get_scores(claim_tokens)
        else:
            bm25_scores = [0.0 for _ in windows]

        # Filtering logic: keep windows above either threshold
        filtered_windows = []
        for i, w in enumerate(windows):
            w["_sbert_score"] = cosine_scores[i]
            w["_bm25_score"] = bm25_scores[i]
            pass_sbert = cosine_scores[i] >= sbert_threshold if use_sbert else False
            pass_bm25 = bm25_scores[i] >= bm25_threshold if use_bm25 else False
            keep = False
            if filter_logic == "both":
                keep = pass_sbert or pass_bm25
            elif filter_logic == "sbert":
                keep = pass_sbert
            elif filter_logic == "bm25":
                keep = pass_bm25
            if keep:
                filtered_windows.append(w)

        print(
            f"[HYBRID][STEP 2] Filtering complete: {len(filtered_windows)} windows retained out of {len(windows)}."
        )
        if filtered_windows:
            print(f"[HYBRID][STEP 2] Sample filtered window: {filtered_windows[0]}")

        # --- [STEP 3] Coreference patching (with adjacent paragraph context) ---
        # FIXME: Paragraph indexing is string-based (p_id); robust integer indexing
        # (or a global document order field) may be required if captions/figures are included.
        # See parser.py for discussion.
        if settings.HYBRID_ENABLE_COREF:
            print(
                "[HYBRID][STEP 3] Applying coreference resolution/patch (paragraph + adjacent)..."
            )
            try:
                import spacy
                import spacy_coref

                nlp = spacy.load("en_core_web_sm")
                nlp.add_pipe("coref")
                coref_windows = []
                from collections import defaultdict

                para_to_windows = defaultdict(list)
                for w in filtered_windows:
                    para_to_windows[w["meta"]["p_id"]].append(w)
                para_ids = [w["meta"]["p_id"] for w in filtered_windows]
                seen = set()
                ordered_para_ids = []
                for pid in para_ids:
                    if pid not in seen:
                        ordered_para_ids.append(pid)
                        seen.add(pid)
                for i, p_id in enumerate(ordered_para_ids):
                    prev_p_id = ordered_para_ids[i - 1] if i > 0 else None
                    prev_text = ""
                    if prev_p_id:
                        prev_texts = [w["text"] for w in para_to_windows[prev_p_id]]
                        prev_text = " ".join(prev_texts)
                    for w in para_to_windows[p_id]:
                        context_text = (prev_text + " " if prev_text else "") + w[
                            "text"
                        ]
                        doc = nlp(context_text)
                        clusters = [
                            {
                                "main": cluster.main.text,
                                "mentions": [span.text for span in cluster.mentions],
                            }
                            for cluster in doc._.coref_clusters
                        ]
                        patched_text = w["text"]
                        doc_win = nlp(w["text"])
                        for token in doc_win:
                            if token.pos_ == "PRON" and token._.in_coref:
                                for cluster in doc._.coref_clusters:
                                    if token.text in [
                                        span.text for span in cluster.mentions
                                    ]:
                                        referent = cluster.main.text
                                        patched_text = patched_text.replace(
                                            token.text, f"{referent} [{token.text}]", 1
                                        )
                                        break
                        w["coref_patched"] = patched_text
                        w["coref_clusters"] = clusters
                        w["coref_context"] = {
                            "prev_text": prev_text,
                            "context_used": context_text,
                        }
                        coref_windows.append(w)
                print(
                    f"[HYBRID][STEP 3] Coref patching complete: {len(coref_windows)} windows."
                )
            except Exception as e:
                import traceback

                print("=== EXCEPTION DURING STEP 3 (Coref) ===")
                traceback.print_exc()
                raise
        else:
            coref_windows = filtered_windows
            print("[HYBRID][STEP 3] Coreference patching skipped.")

        # --- [STEP 4] ColBERT/SPLADE reranking ---
        print(
            f"[HYBRID][STEP 4] Reranking windows using {settings.HYBRID_RERANK_MODEL}..."
        )

        rerank_model = settings.HYBRID_RERANK_MODEL.lower()
        rerank_top_k = getattr(settings, "HYBRID_RERANK_TOP_K", 20)
        rerank_percentile = getattr(settings, "HYBRID_RERANK_PERCENTILE", None)
        claim = kwargs.get("claim", None)

        reranked_windows = []
        scores = []

        if rerank_model == "colbert":
            print("[HYBRID][STEP 4] Loading ColBERT model...")
            # TODO: import your ColBERT runner here
            # from colbert_runner import run_colbert_rerank
            # reranked, token_scores = run_colbert_rerank(claim, coref_windows, settings)
            # For now, placeholder logic:
            print("[HYBRID][STEP 4][TODO] ColBERT rerank not yet implemented.")
            reranked = coref_windows
            token_scores = [{} for _ in reranked]
        elif rerank_model == "splade":
            print("[HYBRID][STEP 4] Loading SPLADE model...")
            # TODO: import and run SPLADE here
            # from splade_runner import run_splade_rerank
            # reranked, token_scores = run_splade_rerank(claim, coref_windows, settings)
            # For now, placeholder logic:
            print("[HYBRID][STEP 4][TODO] SPLADE rerank not yet implemented.")
            reranked = coref_windows
            token_scores = [{} for _ in reranked]
        else:
            print(
                f"[HYBRID][STEP 4] Unknown rerank model: {rerank_model}. Passing windows unchanged."
            )
            reranked = coref_windows
            token_scores = [{} for _ in reranked]

        # Attach scores (empty if not implemented) to each window for downstream audit
        for w, ts in zip(reranked, token_scores):
            w["_rerank_token_scores"] = ts
            # Could also include an overall window score here if desired
            scores.append(ts.get("window_score", 0.0))

        # Filter top-k or by percentile if specified
        if rerank_percentile is not None:
            # Cutoff by percentile
            threshold = np.percentile(scores, rerank_percentile)
            reranked_windows = [w for w, s in zip(reranked, scores) if s >= threshold]
        elif rerank_top_k:
            reranked_windows = sorted(
                reranked,
                key=lambda w: w.get("_rerank_token_scores", {}).get(
                    "window_score", 0.0
                ),
                reverse=True,
            )[:rerank_top_k]
        else:
            reranked_windows = reranked

        print(f"[HYBRID][STEP 4] Reranking complete: {len(reranked_windows)} windows.")

        print(f"[HYBRID][STEP 4] Reranking complete: {len(reranked_windows)} windows.")

        # --- [STEP 5] Cross-encoder + NLI assessment ---
        print("[HYBRID][STEP 5] Running cross-encoder NLI on top-k windows...")

        from backend import nli

        nli_model = getattr(settings, "NLI_MODEL", None)
        claim = kwargs.get("claim", None)
        passages = [w.get("coref_patched", w["text"]) for w in reranked_windows]
        metadatas = [w.get("meta", {}) for w in reranked_windows]

        nli_results = nli.assess(claim, passages, metadatas, nli_model=nli_model)
        print(f"[HYBRID][STEP 5] NLI assessment complete: {len(nli_results)} results.")

        # --- [STEP 6] Result JSON, audit mode, and output ---
        print("[HYBRID][STEP 6] Packaging results and audit trail...")
        result = {
            "pipeline_mode": "hybrid",
            "settings": {
                k: getattr(settings, k)
                for k in dir(settings)
                if k.startswith("HYBRID_")
            },
            "windows": nli_results,
        }
        if settings.HYBRID_AUDIT_MODE:
            # Optionally include intermediate results for debugging
            result["audit"] = {
                "segmentation": windows,
                "filtered": filtered_windows,
                "coref": coref_windows,
                "reranked": reranked_windows,
            }
            print("[HYBRID][STEP 6] Audit trail included in result.")
        else:
            print("[HYBRID][STEP 6] Audit trail not requested.")

        print("[HYBRID] Pipeline complete.")
        return result
