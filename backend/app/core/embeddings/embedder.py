import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from sentence_transformers import SentenceTransformer


@dataclass
class EmbedResult:
    texts: List[str]
    vectors: List[List[float]]    # shape: [n_texts, dimensions]
    dimensions: int
    model_name: str
    latency_ms: float
    norms: List[float]            # L2 norm of each vector — should be ~1.0 for normalized models


@dataclass
class SimilarityResult:
    text_a: str
    text_b: str
    cosine_similarity: float
    dot_product: float
    euclidean_distance: float
    interpretation: str           # "very similar" / "somewhat similar" / "dissimilar"


class Embedder:
    """
    Wraps sentence-transformers. Runs fully locally — no API key.

    Key concepts surfaced:
    - Vectors are L2-normalized → cosine similarity == dot product
    - Dimensions vary by model: MiniLM=384, mpnet=768
    - Batch embedding is much faster than one-at-a-time
    """

    _cache: dict = {}   # simple in-process cache keyed by (model, text)

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimensions = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], normalize: bool = True) -> EmbedResult:
        t0 = time.perf_counter()
        vectors = self._model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            batch_size=32,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        vecs_list = vectors.tolist()
        norms = [round(float(np.linalg.norm(v)), 4) for v in vectors]

        return EmbedResult(
            texts=texts,
            vectors=vecs_list,
            dimensions=self.dimensions,
            model_name=self.model_name,
            latency_ms=latency_ms,
            norms=norms,
        )

    def embed_one(self, text: str) -> List[float]:
        result = self.embed([text])
        return result.vectors[0]

    def cosine_similarity(self, text_a: str, text_b: str) -> SimilarityResult:
        result = self.embed([text_a, text_b])
        v_a = np.array(result.vectors[0])
        v_b = np.array(result.vectors[1])

        cosine = float(np.dot(v_a, v_b))   # valid because vectors are normalized
        dot = float(np.dot(v_a, v_b))
        euclidean = float(np.linalg.norm(v_a - v_b))

        if cosine >= 0.85:
            interp = "very similar"
        elif cosine >= 0.6:
            interp = "somewhat similar"
        elif cosine >= 0.3:
            interp = "weakly related"
        else:
            interp = "dissimilar"

        return SimilarityResult(
            text_a=text_a,
            text_b=text_b,
            cosine_similarity=round(cosine, 4),
            dot_product=round(dot, 4),
            euclidean_distance=round(euclidean, 4),
            interpretation=interp,
        )
