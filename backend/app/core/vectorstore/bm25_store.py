import time
from dataclasses import dataclass
from typing import List, Optional
from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    chunk_id: str
    text: str
    score: float
    rank: int


@dataclass
class BM25SearchResults:
    query: str
    results: List[BM25Result]
    top_k: int
    latency_ms: float
    corpus_size: int


class BM25Store:
    """
    BM25 (Best Match 25) — classic sparse TF-IDF-style retrieval.

    How it works:
    - TF (term frequency): how often does a query term appear in a chunk?
      BM25 caps this with a saturation parameter (k1) so one term can't dominate.
    - IDF (inverse document frequency): rare terms across all chunks score higher.
    - Length normalization (b): penalizes longer chunks from just having more term matches.

    BM25 is excellent for keyword-heavy queries ("What is the capital of France?").
    Vector search is better for semantic queries ("Tell me about European geography").
    Hybrid combines both — the best of both worlds.
    """

    def __init__(self):
        self._corpus: List[str] = []          # raw chunk texts
        self._chunk_ids: List[str] = []
        self._tokenized: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def add_batch(self, chunk_ids: List[str], texts: List[str]):
        self._chunk_ids.extend(chunk_ids)
        self._corpus.extend(texts)
        self._tokenized.extend([text.lower().split() for text in texts])
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 5) -> BM25SearchResults:
        t0 = time.perf_counter()
        if not self._bm25 or not self._corpus:
            return BM25SearchResults(query=query, results=[], top_k=top_k,
                                     latency_ms=0, corpus_size=0)

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Sort by score descending, take top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (idx, score) in enumerate(ranked):
            if score == 0:
                continue
            results.append(BM25Result(
                chunk_id=self._chunk_ids[idx],
                text=self._corpus[idx],
                score=round(float(score), 4),
                rank=rank + 1,
            ))

        return BM25SearchResults(
            query=query,
            results=results,
            top_k=top_k,
            latency_ms=latency_ms,
            corpus_size=len(self._corpus),
        )

    def reset(self):
        self._corpus = []
        self._chunk_ids = []
        self._tokenized = []
        self._bm25 = None

    @property
    def size(self) -> int:
        return len(self._corpus)
