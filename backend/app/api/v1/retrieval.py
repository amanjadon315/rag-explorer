from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional, Literal
from app.dependencies import get_embedder, get_vector_store, get_bm25_store
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.prompt.builder import PromptBuilder
from app.core.ingestion.tokenizer import Tokenizer

router = APIRouter()

# ------------------------------------------------------------------ #
# Shared request models                                                #
# ------------------------------------------------------------------ #
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class IndexRequest(BaseModel):
    """Add a batch of chunks to the vector store and BM25 index."""
    chunks: List[dict]   # [{chunk_id, text}]


# ------------------------------------------------------------------ #
# Index endpoint — called after chunking to populate stores           #
# ------------------------------------------------------------------ #
@router.post("/index")
def index_chunks(
    body: IndexRequest,
    embedder=Depends(get_embedder),
    vector_store=Depends(get_vector_store),
    bm25_store=Depends(get_bm25_store),
):
    """Embed all chunks and store in FAISS + BM25."""
    texts = [c["text"] for c in body.chunks]
    ids = [c["chunk_id"] for c in body.chunks]

    embed_result = embedder.embed(texts)
    items = [(cid, txt, vec, {}) for cid, txt, vec in zip(ids, texts, embed_result.vectors)]
    vector_store.add_batch(items)
    bm25_store.add_batch(ids, texts)
    vector_store.persist()

    return {
        "indexed": len(body.chunks),
        "vector_index_size": vector_store.size,
        "bm25_index_size": bm25_store.size,
        "embedding_model": embed_result.model_name,
        "latency_ms": embed_result.latency_ms,
    }


@router.post("/vector")
def vector_search(
    body: QueryRequest,
    embedder=Depends(get_embedder),
    vector_store=Depends(get_vector_store),
):
    query_vec = embedder.embed_one(body.query)
    results = vector_store.search(query_vec, top_k=body.top_k, query_text=body.query)
    return {
        "query": results.query,
        "top_k": results.top_k,
        "latency_ms": results.latency_ms,
        "index_size": results.index_size,
        "index_type": results.index_type,
        "results": [
            {"rank": r.rank, "chunk_id": r.chunk_id, "score": r.score, "text": r.text}
            for r in results.results
        ],
    }


@router.post("/bm25")
def bm25_search(body: QueryRequest, bm25_store=Depends(get_bm25_store)):
    results = bm25_store.search(body.query, top_k=body.top_k)
    return {
        "query": results.query,
        "top_k": results.top_k,
        "latency_ms": results.latency_ms,
        "corpus_size": results.corpus_size,
        "results": [
            {"rank": r.rank, "chunk_id": r.chunk_id, "score": r.score, "text": r.text}
            for r in results.results
        ],
    }


@router.post("/hybrid")
def hybrid_search(
    body: QueryRequest,
    embedder=Depends(get_embedder),
    vector_store=Depends(get_vector_store),
    bm25_store=Depends(get_bm25_store),
):
    hybrid = HybridRetriever(vector_store, bm25_store)
    query_vec = embedder.embed_one(body.query)
    results = hybrid.search(body.query, query_vec, top_k=body.top_k)
    return {
        "query": results.query,
        "rrf_k": results.rrf_k,
        "vector_results_count": results.vector_results_count,
        "bm25_results_count": results.bm25_results_count,
        "results": [
            {
                "final_rank": r.final_rank,
                "chunk_id": r.chunk_id,
                "rrf_score": r.rrf_score,
                "vector_rank": r.vector_rank,
                "bm25_rank": r.bm25_rank,
                "vector_score": r.vector_score,
                "bm25_score": r.bm25_score,
                "text": r.text,
            }
            for r in results.results
        ],
    }
