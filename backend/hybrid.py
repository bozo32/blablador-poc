# backend/hybrid.py


from typing import Any, Dict
import numpy as np
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
                windows.append({
                    "text": w["text"],
                    "tei_ids": sent_ids,
                    "window_id": w.get("id"),
                    "meta": w.get("meta", {}),
                })
            # For paragraphs too short for window_size, keep largest window covering all sentences
            elif para_max_len[p_id] < window_size and len(sent_ids) == para_max_len[p_id]:
                # Only include one "full paragraph" window per short para
                key = (p_id, tuple(sorted(sent_ids)))
                if key not in seen_full_para:
                    windows.append({
                        "text": w["text"],
                        "tei_ids": sent_ids,
                        "window_id": w.get("id"),
                        "meta": w.get("meta", {}),
                    })
                    seen_full_para.add(key)

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

        # --- [STEP 3] Coreference patching (FastCoref/f-coref model) ---
        if settings.HYBRID_ENABLE_COREF:
            print("[HYBRID][STEP 3] Applying f-coref coreference (FastCoref direct)...")
            # -- PATCH: fastcoref integration --
            try:
                from fastcoref import FCoref
                coref_model_path = getattr(settings, "HYBRID_COREF_MODEL", "biu-nlp/f-coref")
                coref_model = FCoref(model_name=coref_model_path, device="cpu")  # adjust device as needed

                coref_windows = []
                for w in filtered_windows:
                    text = w['text']
                    try:
                        preds = coref_model.predict(texts=[text], max_length=512)
                        clusters = []
                        patched = text
                        if preds and preds.get_clusters(0):
                            cluster_list = preds.get_clusters(as_strings=True)[0]  # List[List[str]]
                            clusters = cluster_list
                            # Simple patching: replace each pronoun with main mention in cluster
                            # (for demonstration, could be smarter)
                            for cluster in cluster_list:
                                main = cluster[0]
                                for mention in cluster[1:]:
                                    if mention != main and mention in patched:
                                        patched = patched.replace(mention, f"{main} [{mention}]", 1)
                        w["coref_clusters"] = clusters
                        w["coref_patched"] = patched
                        coref_windows.append(w)
                    except Exception as e:
                        print(f"[HYBRID][STEP 3] Coref failed for window: {w.get('window_id', '')}: {e}")
                        w["coref_clusters"] = []
                        w["coref_patched"] = text
                        coref_windows.append(w)
                print(f"[HYBRID][STEP 3] f-coref patch complete: {len(coref_windows)} windows.")
            except Exception as e:
                print("[HYBRID][STEP 3] Failed to load or run FastCoref:", e)
                coref_windows = filtered_windows
        else:
            coref_windows = filtered_windows
            print("[HYBRID][STEP 3] Coreference patching skipped.")
                        
        # --- [STEP 4] ColBERT/SPLADE reranking ---
        print(f"[HYBRID][STEP 4] Reranking windows using {settings.HYBRID_RERANK_MODEL}...")

        rerank_model = getattr(settings, "HYBRID_RERANK_MODEL", "none").lower()
        if rerank_model in ("none", "", "pass", "skip"):
            print("[HYBRID][STEP 4] Reranking skipped for POC (pass-through, no ColBERT/SPLADE).")
            reranked_windows = coref_windows
        else:
            print(f"[HYBRID][STEP 4] Rerank model '{rerank_model}' not implemented in this POC. Passing windows unchanged.")
            reranked_windows = coref_windows

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
