from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional, Literal
from app.dependencies import get_embedder
from app.core.embeddings.reducer import DimensionalityReducer
from app.config import get_settings

router = APIRouter()
reducer = DimensionalityReducer()
settings = get_settings()


class EmbedRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = None


class SimilarityRequest(BaseModel):
    text_a: str
    text_b: str


class ReduceRequest(BaseModel):
    texts: List[str]
    labels: List[str]
    groups: Optional[List[str]] = None
    method: Literal["pca", "tsne"] = "pca"
    tsne_perplexity: int = 5


class NearestRequest(BaseModel):
    query: str
    top_k: int = 5


@router.get("/models")
def list_models():
    return {"models": settings.embedding_models_available, "default": settings.embedding_model}


@router.post("/generate")
def generate_embeddings(body: EmbedRequest, embedder=Depends(get_embedder)):
    """Embed a list of texts. Returns vectors + shape metadata."""
    result = embedder.embed(body.texts)
    return {
        "model": result.model_name,
        "dimensions": result.dimensions,
        "latency_ms": result.latency_ms,
        "n_texts": len(result.texts),
        "norms": result.norms,
        "vectors": result.vectors,   # shape: [n_texts, dimensions]
    }


@router.post("/similarity")
def cosine_similarity(body: SimilarityRequest, embedder=Depends(get_embedder)):
    """Compute similarity between two texts and explain the score."""
    result = embedder.cosine_similarity(body.text_a, body.text_b)
    return {
        "text_a": result.text_a,
        "text_b": result.text_b,
        "cosine_similarity": result.cosine_similarity,
        "dot_product": result.dot_product,
        "euclidean_distance": result.euclidean_distance,
        "interpretation": result.interpretation,
    }


@router.post("/reduce")
def reduce_dimensions(body: ReduceRequest, embedder=Depends(get_embedder)):
    """Generate embeddings then reduce to 2D for visualization."""
    embed_result = embedder.embed(body.texts)
    reduction = reducer.reduce(
        vectors=embed_result.vectors,
        labels=body.labels,
        method=body.method,
        groups=body.groups,
        tsne_perplexity=body.tsne_perplexity,
    )
    return {
        "method": reduction.method,
        "n_points": reduction.n_points,
        "n_input_dimensions": reduction.n_input_dimensions,
        "explained_variance_pct": reduction.explained_variance,
        "points": [
            {"x": p.x, "y": p.y, "label": p.label, "group": p.group, "index": p.original_index}
            for p in reduction.points
        ],
    }
