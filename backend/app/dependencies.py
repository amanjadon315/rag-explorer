"""
Shared singletons injected via FastAPI's dependency injection system.
These are created once at startup and reused across requests.
"""
from functools import lru_cache
from app.config import get_settings

settings = get_settings()


@lru_cache()
def get_embedder():
    """
    Returns a singleton Embedder instance.
    The model weights are loaded once and kept in memory.
    Using lru_cache means this runs only on first call.
    """
    from app.core.embeddings.embedder import Embedder
    return Embedder(model_name=settings.embedding_model)


@lru_cache()
def get_vector_store():
    """
    Returns a singleton FAISSVectorStore.
    The index is loaded from disk if it exists, otherwise created fresh.
    """
    from app.core.vectorstore.faiss_store import FAISSVectorStore
    return FAISSVectorStore(index_path=settings.faiss_index_path)


@lru_cache()
def get_bm25_store():
    """
    Returns a singleton BM25Store (in-memory, rebuilt on restart).
    In production you'd persist the corpus to Postgres.
    """
    from app.core.vectorstore.bm25_store import BM25Store
    return BM25Store()
