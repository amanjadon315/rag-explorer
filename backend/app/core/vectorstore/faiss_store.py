import os
import json
import time
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float               # cosine similarity (higher = more similar)
    rank: int
    metadata: dict


@dataclass
class SearchResults:
    query: str
    results: List[SearchResult]
    top_k: int
    latency_ms: float
    index_size: int            # total vectors in index
    index_type: str


class FAISSVectorStore:
    """
    FAISS HNSW index for approximate nearest-neighbor search.

    Why HNSW (Hierarchical Navigable Small World):
    - Builds a multi-layer graph — upper layers for coarse navigation,
      lower layers for fine-grained search.
    - O(log n) query time vs O(n) for brute-force flat index.
    - Trades a small amount of recall for large speed gains at scale.

    We store vectors normalized (L2 norm = 1), so inner product == cosine similarity.
    """

    def __init__(self, index_path: str = "./data/faiss", dimensions: int = 384):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.index_file = self.index_path / "index.faiss"
        self.meta_file = self.index_path / "metadata.json"

        # Metadata: maps FAISS integer ID → {chunk_id, text, ...}
        self._metadata: List[dict] = []
        self._index: Optional[faiss.Index] = None
        self._load_or_create()

    def _load_or_create(self):
        if self.index_file.exists() and self.meta_file.exists():
            self._index = faiss.read_index(str(self.index_file))
            with open(self.meta_file) as f:
                self._metadata = json.load(f)
        else:
            # IndexHNSWFlat: M=32 neighbors per layer, good balance of speed/recall
            self._index = faiss.IndexHNSWFlat(self.dimensions, 32)
            self._index.hnsw.efConstruction = 200   # higher = better index quality
            self._index.hnsw.efSearch = 64          # higher = better recall at query time
            self._metadata = []

    def add(self, chunk_id: str, text: str, vector: List[float], metadata: dict = None):
        """Add a single chunk to the index."""
        vec = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vec)
        self._index.add(vec)
        self._metadata.append({
            "chunk_id": chunk_id,
            "text": text,
            **(metadata or {}),
        })

    def add_batch(self, items: List[Tuple[str, str, List[float], dict]]):
        """
        Batch add: items = [(chunk_id, text, vector, metadata), ...]
        Much faster than adding one at a time.
        """
        vectors = np.array([item[2] for item in items], dtype=np.float32)
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        for chunk_id, text, _, meta in items:
            self._metadata.append({"chunk_id": chunk_id, "text": text, **(meta or {})})

    def search(self, query_vector: List[float], top_k: int = 5, query_text: str = "") -> SearchResults:
        t0 = time.perf_counter()
        if self._index.ntotal == 0:
            return SearchResults(query=query_text, results=[], top_k=top_k,
                                 latency_ms=0, index_size=0, index_type="HNSW")

        vec = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(vec)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:   # FAISS returns -1 for unfilled slots
                continue
            meta = self._metadata[idx]
            results.append(SearchResult(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                score=round(float(score), 4),
                rank=rank + 1,
                metadata={k: v for k, v in meta.items() if k not in ("chunk_id", "text")},
            ))

        return SearchResults(
            query=query_text,
            results=results,
            top_k=top_k,
            latency_ms=latency_ms,
            index_size=self._index.ntotal,
            index_type="HNSW",
        )

    def persist(self):
        faiss.write_index(self._index, str(self.index_file))
        with open(self.meta_file, "w") as f:
            json.dump(self._metadata, f)

    def reset(self):
        """Clear the index — useful for the explorer's 'start fresh' button."""
        self._load_or_create()
        if self.index_file.exists():
            self.index_file.unlink()
        if self.meta_file.exists():
            self.meta_file.unlink()

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0
