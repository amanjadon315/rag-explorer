from dataclasses import dataclass
from typing import List
from app.core.vectorstore.faiss_store import FAISSVectorStore, SearchResult
from app.core.vectorstore.bm25_store import BM25Store


@dataclass
class HybridResult:
    chunk_id: str
    text: str
    rrf_score: float
    vector_rank: int | None      # rank in vector results (None if not retrieved)
    bm25_rank: int | None        # rank in BM25 results (None if not retrieved)
    vector_score: float | None
    bm25_score: float | None
    final_rank: int


@dataclass
class HybridSearchResults:
    query: str
    results: List[HybridResult]
    vector_results_count: int
    bm25_results_count: int
    rrf_k: int


class HybridRetriever:
    """
    Reciprocal Rank Fusion (RRF) — combines rankings from two retrievers.

    Formula: RRF_score(d) = Σ 1 / (k + rank(d))
    where k=60 is a constant that dampens the impact of very high ranks.

    Why RRF works well:
    - It doesn't require score normalization (vector cosine vs BM25 raw scores are incomparable).
    - Chunks ranked highly by BOTH retrievers get a big boost.
    - Chunks found by only one retriever still get credit.
    - k=60 is empirically shown to outperform other fusion strategies.
    """

    def __init__(self, vector_store: FAISSVectorStore, bm25_store: BM25Store, rrf_k: int = 60):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
    ) -> HybridSearchResults:
        # Get results from both retrievers (fetch more than top_k for better fusion)
        fetch_k = min(top_k * 3, 20)
        vector_results = self.vector_store.search(query_vector, top_k=fetch_k, query_text=query)
        bm25_results = self.bm25_store.search(query, top_k=fetch_k)

        # Build lookup dicts: chunk_id → result
        vec_by_id = {r.chunk_id: r for r in vector_results.results}
        bm25_by_id = {r.chunk_id: r for r in bm25_results.results}

        # Collect all unique chunk IDs
        all_ids = set(vec_by_id.keys()) | set(bm25_by_id.keys())

        # Compute RRF score for each chunk
        scored = []
        for chunk_id in all_ids:
            vec_r = vec_by_id.get(chunk_id)
            bm25_r = bm25_by_id.get(chunk_id)

            vec_rank = vec_r.rank if vec_r else None
            bm25_rank = bm25_r.rank if bm25_r else None

            rrf = 0.0
            if vec_rank is not None:
                rrf += 1.0 / (self.rrf_k + vec_rank)
            if bm25_rank is not None:
                rrf += 1.0 / (self.rrf_k + bm25_rank)

            text = (vec_r or bm25_r).text

            scored.append(HybridResult(
                chunk_id=chunk_id,
                text=text,
                rrf_score=round(rrf, 6),
                vector_rank=vec_rank,
                bm25_rank=bm25_rank,
                vector_score=round(vec_r.score, 4) if vec_r else None,
                bm25_score=round(bm25_r.score, 4) if bm25_r else None,
                final_rank=0,  # set below
            ))

        scored.sort(key=lambda x: x.rrf_score, reverse=True)
        for i, item in enumerate(scored[:top_k]):
            item.final_rank = i + 1

        return HybridSearchResults(
            query=query,
            results=scored[:top_k],
            vector_results_count=len(vector_results.results),
            bm25_results_count=len(bm25_results.results),
            rrf_k=self.rrf_k,
        )
