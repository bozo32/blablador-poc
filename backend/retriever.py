from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from . import utils
from .parser import tei_and_csv_to_documents


class Retriever:
    def __init__(
        self,
        index_path: Path,
        max_sentences: Optional[int] = None,
        min_score: float = 0.20,
        embed_model: str = "all-MiniLM-L6-v2",  # HF default
        docstore: dict[str, dict] | None = None,
    ):
        self.index_path = index_path.with_suffix(".faiss")
        self.index = None
        self.chunks: List[Dict] = []
        self.max_sentences = max_sentences
        self.min_score = min_score
        self.embed_model = embed_model
        self.docstore = docstore or {}
        self._docs: List[Dict] = []
        self.id_list: List[str] = []

    def search(
        self, vectors: List[List[float]], k: int
    ) -> tuple[List[List[str]], List[float]]:
        arr = np.array(vectors, dtype="float32")
        D, I = self.index.search(arr, k)
        id_batches = [[self.id_list[pos] for pos in batch] for batch in I]
        return id_batches, D.tolist()

    def build(self, docs: List[Dict]) -> None:
        self._docs = docs
        filtered = docs[: self.max_sentences] if self.max_sentences else docs
        # *** Local embedding ***
        embeddings_list = utils.embed(
            [c["text"] for c in filtered],
            model_name=self.embed_model,
        )
        embeddings = np.array(embeddings_list, dtype="float32")
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        idx = faiss.IndexFlatIP(dimension)
        idx.add(embeddings)
        self.index = idx
        self.chunks = filtered
        self.id_list = [chunk["meta"]["id"] for chunk in filtered]
        for chunk in filtered:
            cid = chunk["meta"]["id"]
            self.docstore.setdefault(cid, chunk)
        faiss.write_index(self.index, str(self.index_path))

    def load(self):
        self.index = faiss.read_index(str(self.index_path))

    def query(self, text: str, k: int = 5) -> List[Dict]:
        if self.index is None or not self.chunks:
            raise ValueError(
                "Index and chunks must be loaded or built before querying."
            )
        raw_emb = utils.embed([text], model_name=self.embed_model)
        query_embedding = np.array(raw_emb, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist < self.min_score:
                continue
            chunk = self.chunks[idx]
            results.append(
                {"score": float(dist), "text": chunk["text"], "meta": chunk["meta"]}
            )
        return results

    def query_many(self, texts: List[str], k: int = 5) -> List[List[Dict]]:
        if self.index is None or not self.chunks:
            raise ValueError(
                "Index and chunks must be loaded or built before querying."
            )
        raw_embs = utils.embed(texts, model_name=self.embed_model)
        arr = np.array(raw_embs, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        faiss.normalize_L2(arr)
        distances, indices = self.index.search(arr, k)
        all_results: List[List[Dict]] = []
        for dist_row, idx_row in zip(distances, indices):
            results: List[Dict] = []
            for dist, idx in zip(dist_row, idx_row):
                if dist < self.min_score:
                    continue
                chunk = self.chunks[idx]
                results.append(
                    {"score": float(dist), "text": chunk["text"], "meta": chunk["meta"]}
                )
            all_results.append(results)
        return all_results


def build_all(
    folder: Path, embed_model: str, max_sentences: int = None, min_score: float = 0.2
) -> Dict[str, Retriever]:
    # find CSV in folder
    csv_file = next(folder.glob("*.csv"), None)
    if csv_file is None:
        raise ValueError(f"No CSV file found in {folder}")
    # combine TEI XML chunks and CSV entries into docs
    docs = tei_and_csv_to_documents(folder, str(csv_file))

    indexes = {}
    # Build per-paper indexes for CSV-derived docs (which have author/year)
    for doc in docs:
        if "author" not in doc or "year" not in doc:
            continue
        key = f"{doc['author']}-{doc['year']}"
        retr = Retriever(
            index_path=folder / key,
            max_sentences=max_sentences,
            min_score=min_score,
            embed_model=embed_model,
        )
        retr.build([doc])
        indexes[key] = retr

    # include both single sentences and our sliding‐window spans
    tei_docs = [
        doc
        for doc in docs
        if doc["meta"].get("type") in ("sentence", "sentence_window")
    ]

    default_retr = Retriever(
        index_path=folder / "default",
        max_sentences=max_sentences,
        min_score=min_score,
        embed_model=embed_model,
    )
    default_retr.build(tei_docs)
    indexes["default"] = default_retr

    return indexes
