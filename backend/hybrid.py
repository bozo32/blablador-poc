# backend/hybrid.py

import requests
from typing import Any, Dict
from backend.settings import AppSettings
from pathlib import Path
from backend.retriever import Retriever
from backend.utils import get_model
from backend import nli



class HybridPipeline:
    @staticmethod
    def build_all(
        folder,
        embed_model,
        max_sentences,
        min_score,
        settings: AppSettings = None,  # accept explicit or global settings
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Hybrid pipeline: full scaffold for stepwise development.
        Each stage includes a progress marker and placeholder for logic.
        See development steps for details on each segment.
        """
        if settings is None:
            settings = AppSettings()

        # --- [STEP 1] TEI segmentation and windowing ---
        print("[HYBRID][STEP 1] Segmenting TEI and generating windows...")
        import os
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

        # Build a mapping of paragraph ID to the max number of sentences in that paragraph
        para_max_len = {}
        for w in all_windows:
            p_id = w["meta"].get("p_id")
            sent_ids = w["meta"].get("sent_ids", [])
            para_max_len.setdefault(p_id, 0)
            para_max_len[p_id] = max(para_max_len[p_id], len(sent_ids))

        # Collect the standard windows, and also collect for short paras
        windows = []
        seen_full_para = set()
        for w in all_windows:
            p_id = w["meta"].get("p_id")
            sent_ids = w["meta"].get("sent_ids", [])
            win_size = w["meta"].get("window_size")
            # Standard window
            if win_size == window_size:
                windows.append(
                    {
                        "text": w["text"],
                        "tei_ids": sent_ids,
                        "window_id": w.get("id"),
                        "meta": w.get("meta", {}),
                    }
                )
            # For paragraphs too short for window_size, keep largest window covering all sentences
            elif (
                para_max_len[p_id] < window_size and len(sent_ids) == para_max_len[p_id]
            ):
                # Only include one "full paragraph" window per short para
                key = (p_id, tuple(sorted(sent_ids)))
                if key not in seen_full_para:
                    windows.append(
                        {
                            "text": w["text"],
                            "tei_ids": sent_ids,
                            "window_id": w.get("id"),
                            "meta": w.get("meta", {}),
                        }
                    )
                    seen_full_para.add(key)

        print(
            f"[HYBRID][STEP 1] Complete: {len(windows)} windows generated of size {window_size}."
        )
        if windows:
            print(f"[HYBRID][STEP 1] First window sample: {windows[0]}")

        # --- [STEP 2] Embed ALL windows ➜ FAISS ➜ τ-filter ➜ snap to 3-sentence ---
        print("[HYBRID][STEP 2] Building in-memory FAISS index...")

        import numpy as np, faiss

        claim = kwargs.get("claim")
        if not isinstance(claim, str) or not claim:
            raise ValueError("build_all requires claim=... (str)")

        embed_model_name = embed_model or getattr(settings, "EMBED_MODEL", "all-MiniLM-L6-v2")
        embedder = get_model(embed_model_name)

        # 2-1  embed passages
        p_vecs = embedder.encode([w["text"] for w in windows], show_progress_bar=False)
        p_vecs = np.asarray(p_vecs, dtype="float32"); faiss.normalize_L2(p_vecs)

        index = faiss.IndexFlatIP(p_vecs.shape[1]); index.add(p_vecs)

        # 2-2  embed query
        q_vec  = embedder.encode([claim], show_progress_bar=False)
        q_vec  = np.asarray(q_vec, dtype="float32"); faiss.normalize_L2(q_vec)

        K   = getattr(settings, "RETRIEVAL_K", 1500)
        tau = getattr(settings, "RETRIEVAL_TAU", 0.80)
        cap = getattr(settings, "RETRIEVAL_MAX", 200)

        D, I = index.search(q_vec, K)
        hits = [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i != -1]
        hits.sort(key=lambda t: t[1], reverse=True)

        best = hits[0][1]
        kept = [(i,s) for i,s in hits if s >= best * tau]
        if len(kept) > cap:
            kept = kept[:cap]
        print(f"[HYBRID][STEP 2] τ-filter kept {len(kept)} / {len(hits)} windows")

        # Build a quick lookup so we can safely resolve parent3 IDs
        id2win: dict[str, dict] = {w["window_id"]: w for w in windows}

        # 2‑3  snap every kept hit to its parent 3‑sentence window (if it exists)
        snap: dict[str, float] = {}
        for idx_i, score in kept:
            orig_win = windows[idx_i]
            # prefer the explicit parent3; fall back to the original window ID
            p3_id = orig_win["meta"].get("parent3") or orig_win["window_id"]
            if p3_id not in id2win:
                # parent3 window might not exist for 1‑sentence paras – fall back
                p3_id = orig_win["window_id"]
            # keep only the highest‑score hit per parent window
            if score > snap.get(p3_id, -1.0):
                snap[p3_id] = score

        filtered_windows = [
            {**id2win[p3_id], "faiss_score": sc}
            for p3_id, sc in sorted(snap.items(), key=lambda x: x[1], reverse=True)
        ]
        print(f"[HYBRID][STEP 2] Unique 3‑sentence windows: {len(filtered_windows)}")
        

        # ------------------------------------------------------------------ #
        #   [STEP 3]  OPTIONAL RERANK -- SBERT or BM25 or BOTH (existing cfg)
        # ------------------------------------------------------------------ #
        filter_logic = getattr(settings, "HYBRID_FILTER_LOGIC", "none").lower()
        use_sbert = filter_logic in ("sbert", "both")
        use_bm25  = filter_logic in ("bm25", "both")

        if use_sbert or use_bm25:
            print(f"[HYBRID][STEP 3] Reranking with {filter_logic.upper()} ...")
            # ---------- SBERT cosine ----------
            if use_sbert:
                # embedder from STEP 2 is still in scope
                import torch
                claim_emb   = embedder.encode([claim], convert_to_tensor=True)
                win_embs    = embedder.encode(
                    [w["text"] for w in filtered_windows], convert_to_tensor=True
                )

                sims = torch.nn.functional.cosine_similarity(
                    claim_emb.repeat(win_embs.size(0), 1),
                    win_embs,
                ).cpu().tolist()
                for w, s in zip(filtered_windows, sims):
                    w["_sbert_score"] = s
            # ---------- BM25  ----------
            if use_bm25:
                from rank_bm25 import BM25Okapi
                import spacy, textwrap
                # cheaper tokenizer than spaCy if you prefer
                nlp = spacy.blank("en")
                corpus_tok = [[tok.text for tok in nlp(w["text"])] for w in filtered_windows]
                bm25 = BM25Okapi(corpus_tok)
                claim_tok = [tok.text for tok in nlp(claim)]
                bm_scores = bm25.get_scores(claim_tok)
                for w, s in zip(filtered_windows, bm_scores):
                    w["_bm25_score"] = s
            # ---------- sort  ----------
            def _combined_key(w):
                # three cases: sbert only, bm25 only, both
                if use_sbert and use_bm25:
                    return (w.get("_sbert_score", 0.0) + w.get("_bm25_score", 0.0)) / 2
                elif use_sbert:
                    return w.get("_sbert_score", 0.0)
                else:  # bm25 only
                    return w.get("_bm25_score", 0.0)

            filtered_windows.sort(key=_combined_key, reverse=True)
            print(f"[HYBRID][STEP 3] Rerank complete. Top sample: {_combined_key(filtered_windows[0]):.3f}")
        else:
            print("[HYBRID][STEP 3] Reranking skipped (HYBRID_FILTER_LOGIC='none').")
        # ------------------------------------------------------------------ #
            
        # --- [STEP 4] Coreference patching (FastCoref/f-coref model) ---
        if settings.HYBRID_ENABLE_COREF:
            print("[HYBRID][STEP 3] Applying f-coref coreference (FastCoref direct)...")
            # -- PATCH: fastcoref integration --
            try:
                from fastcoref import FCoref

                coref_model_path = getattr(
                    settings, "HYBRID_COREF_MODEL", "biu-nlp/f-coref"
                )
                coref_model = FCoref(
                    model_name_or_path=coref_model_path, device="cpu"
                )  # adjust device as needed

                coref_windows = []
                for w in filtered_windows:
                    text = w["text"]
                    try:
                        preds = coref_model.predict(texts=[text])
                        # Extract clusters from the CorefResult object
                        if isinstance(preds, list) and preds:
                            coref_res = preds[0]
                        elif hasattr(preds, "get_clusters"):
                            coref_res = preds
                        else:
                            coref_res = None
                        # Retrieve cluster lists as strings
                        if coref_res:
                            cluster_list = coref_res.get_clusters(as_strings=True)
                        else:
                            cluster_list = []
                        clusters = cluster_list or []
                        patched = text
                        # Simple patching: replace each pronoun with the main mention
                        for cluster in cluster_list:
                            main = cluster[0]
                            for mention in cluster[1:]:
                                if mention != main and mention in patched:
                                    patched = patched.replace(mention, f"{main} [{mention}]", 1)
                        w["coref_clusters"] = clusters
                        w["coref_patched"] = patched
                        coref_windows.append(w)
                    except Exception as e:
                        print(
                            f"[HYBRID][STEP 3] Coref failed for window: {w.get('window_id', '')}: {e}"
                        )
                        w["coref_clusters"] = []
                        w["coref_patched"] = text
                        coref_windows.append(w)
                print(
                    f"[HYBRID][STEP 3] f-coref patch complete: {len(coref_windows)} windows."
                )
            except Exception as e:
                print("[HYBRID][STEP 3] Failed to load or run FastCoref:", e)
                coref_windows = filtered_windows
        else:
            coref_windows = filtered_windows
            print("[HYBRID][STEP 3] Coreference patching skipped.")

        # --- [STEP 5] ColBERT reranking ---------------------------------------------
        mode = settings.COLBERT_MODE.lower()
        if mode == "off":
            reranked_windows = coref_windows
        elif mode == "internal":
            # (keep your existing internal ColBERT code block here)
            ...
            reranked_windows = reranked_internal
        elif mode == "external":
            from backend.utils import colbert_api_rerank
            try:
                reranked_windows = colbert_api_rerank(
                    claim, coref_windows,
                    settings.COLBERT_API_URL,
                    settings.COLBERT_TOP_K,
                )
                print(f"[HYBRID][STEP 4] External ColBERT returned {len(reranked_windows)} wins.")
            except requests.RequestException as e:
                print(f"[HYBRID][STEP 4] ColBERT API failed → {e}.  Using coref_windows.")
                reranked_windows = coref_windows

        print(f"[HYBRID][DEBUG] Reranked windows: {reranked_windows[:3]}")
        print(f"[HYBRID][DEBUG] Claim: {claim}")
        passages = [w.get("coref_patched", w["text"]) for w in reranked_windows]
        # Build metadatas list, carrying ColBERT salience forward
        metadatas = []
        for w in reranked_windows:
            m = dict(w.get("meta", {}))          # copy original meta
            if "token_scores" in w:              # preserve salience
                m["token_scores"] = w["token_scores"]
            metadatas.append(m)
        print(f"[HYBRID][DEBUG] Passages: {passages[:3]}")
        print(f"[HYBRID][DEBUG] Metadatas: {metadatas[:3]}")

        # --- [STEP 5] Cross-encoder + NLI assessment ---


        # Prepare all required variables
        nli_model = getattr(settings, "NLI_MODEL", None)
        # claim, passages, metadatas already defined above

        print("[HYBRID][STEP 5] Running cross-encoder NLI on top-k windows...")

        try:
            nli_results = nli.assess(claim, passages, metadatas, nli_model=nli_model)
            print(f"[HYBRID][STEP 5] NLI assessment complete: {len(nli_results)} results.")
        except Exception as e:
            import traceback
            print("[HYBRID][ERROR] Exception in NLI step:", e)
            print(traceback.format_exc())
            raise
        # --- [STEP 6] Build and return a Retriever (classic interface) ---
        print("[HYBRID][STEP 6] Building Retriever for hybrid windows...")


        chunks = []
        for w in reranked_windows:
            meta = dict(w.get("meta", {}))
            meta["id"] = w.get("window_id") or meta.get("id")
            # Optionally include other meta fields here as needed
            if "coref_clusters" in w:
                meta["coref_clusters"] = w["coref_clusters"]
            if "_sbert_score" in w:
                meta["_sbert_score"] = w["_sbert_score"]
            if "_bm25_score" in w:
                meta["_bm25_score"] = w["_bm25_score"]
            if "token_scores" in w:
                meta["token_scores"] = w["token_scores"]
            chunks.append({
                "text": w.get("coref_patched", w["text"]),
                "meta": meta,
            })

        retr = Retriever(
            index_path=Path(folder) / "hybrid_default",
            max_sentences=max_sentences,
            min_score=min_score,
            embed_model=embed_model,
        )
        retr.build(chunks)
        print(f"[HYBRID][STEP 6] Retriever built with {len(chunks)} windows.")

        result = {"default": retr}

        if getattr(settings, "HYBRID_AUDIT_MODE", False):
            result["audit"] = {
                "segmentation": windows,
                "filtered": filtered_windows,
                "coref": coref_windows,
                "reranked": reranked_windows,
            }
            print("[HYBRID][STEP 6] Audit trail included in result.")
        else:
            print("[HYBRID][STEP 6] Audit trail not requested.")

        return result
    