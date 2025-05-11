from pathlib import Path
from . import utils
from .bl_client import embed_documents
from .parser import tei_and_csv_to_documents
from typing import List, Dict, Optional
from lxml import etree
import faiss
import numpy as np

class Retriever:
    def __init__(
        self,
        index_path: Path,
        max_sentences: Optional[int] = None,
        min_score: float = 0.20,
        embed_model: str = "alias-embeddings",
        api_key: str = None,
        base_url: str = None,
        docstore: dict[str, dict] | None = None,  # New parameter for chunk metadata
    ):
        self.index_path = index_path.with_suffix(".faiss")
        self.index = None
        self.chunks: List[Dict] = []
        self.max_sentences = max_sentences
        self.min_score = min_score
        self.embed_model = embed_model
        self.api_key = api_key
        self.base_url = base_url
        # Attach the chunk metadata store
        self.docstore = docstore or {}
        self._docs: List[Dict] = []
        # Parallel list of stable IDs for FAISS index positions
        self.id_list: List[str] = []

    def search(self, vectors: List[List[float]], k: int) -> tuple[List[List[str]], List[float]]:
        """
        Run a FAISS top‐k search over the prebuilt index.
        Returns: (list_of_id_batches, list_of_scores)
        """
        arr = np.array(vectors, dtype="float32")
        D, I = self.index.search(arr, k)
        # Map numeric indices → stable chunk IDs
        id_batches = [
            [self.id_list[pos] for pos in batch]
            for batch in I
        ]
        return id_batches, D.tolist()

    def build(self, docs: List[Dict]) -> None:
        """
        Build the FAISS index from the list of docs.
        """
        # retain full docs list so we can fetch by position later
        self._docs = docs
        filtered = docs[: self.max_sentences] if self.max_sentences else docs
        embeddings_list = embed_documents(
            [{"text": c["text"]} for c in filtered],
            model=self.embed_model,
            api_key=self.api_key,
            base_url=self.base_url
        )
        embeddings = np.array(embeddings_list, dtype="float32")
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        idx = faiss.IndexFlatIP(dimension)
        idx.add(embeddings)
        self.index = idx
        self.chunks = filtered
        # Build parallel ID list (position → chunk_id)
        self.id_list = [chunk["meta"]["id"] for chunk in filtered]
        # Ensure every chunk_id maps into the docstore
        for chunk in filtered:
            cid = chunk["meta"]["id"]
            self.docstore.setdefault(cid, chunk)
        faiss.write_index(self.index, str(self.index_path))

    def load(self):
        self.index = faiss.read_index(str(self.index_path))

    def query(self, text: str, k: int = 5) -> List[Dict]:
        if self.index is None or not self.chunks:
            raise ValueError("Index and chunks must be loaded or built before querying.")
        raw_emb = utils.embed([text])
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
            results.append({
                "score": float(dist),
                "text": chunk["text"],
                "meta": chunk["meta"]
            })
        return results

    def query_many(self, texts: List[str], k: int = 5) -> List[List[Dict]]:
        if self.index is None or not self.chunks:
            raise ValueError("Index and chunks must be loaded or built before querying.")
        raw_embs = utils.embed(texts)
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
                results.append({
                    "score": float(dist),
                    "text": chunk["text"],
                    "meta": chunk["meta"]
                })
            all_results.append(results)
        return all_results

def build_all(
    folder: Path,
    embed_model: str,
    api_key: str,
    base_url: str,
    max_sentences: int = None,
    min_score: float = 0.2
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
        if 'author' not in doc or 'year' not in doc:
            continue
        key = f"{doc['author']}-{doc['year']}"
        retr = Retriever(
            index_path=folder / key,
            max_sentences=max_sentences,
            min_score=min_score,
            embed_model=embed_model,
            api_key=api_key,
            base_url=base_url
        )
        retr.build([doc])
        indexes[key] = retr

    # Build default (global) FAISS index, but only over TEI sentence chunks
    tei_docs = [doc for doc in docs if doc["meta"].get("type") == "sentence"]
    default_retr = Retriever(
        index_path=folder / "default",
        max_sentences=max_sentences,
        min_score=min_score,
        embed_model=embed_model,
        api_key=api_key,
        base_url=base_url
    )
    default_retr.build(tei_docs)
    indexes["default"] = default_retr

    return indexes
