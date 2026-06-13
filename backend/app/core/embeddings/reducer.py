import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class Point2D:
    x: float
    y: float
    label: str
    original_index: int
    group: Optional[str] = None    # for coloring clusters in the UI


@dataclass
class ReductionResult:
    method: str
    points: List[Point2D]
    explained_variance: Optional[float]   # PCA only — % variance captured by 2 components
    n_input_dimensions: int
    n_points: int


class DimensionalityReducer:
    """
    Reduces high-dimensional embeddings to 2D for visualization.

    PCA:   Linear, fast, deterministic. Good for seeing global structure.
           explained_variance tells you how much information is preserved.

    t-SNE: Non-linear, slower, stochastic. Excellent at revealing clusters.
           Perplexity controls neighborhood size (5–50, lower = tighter clusters).
           Different runs may produce different layouts.
    """

    def reduce(
        self,
        vectors: List[List[float]],
        labels: List[str],
        method: Literal["pca", "tsne"] = "pca",
        groups: Optional[List[str]] = None,
        tsne_perplexity: int = 5,
    ) -> ReductionResult:
        X = np.array(vectors)
        n, d = X.shape

        if method == "pca":
            coords_2d, explained = self._pca(X)
        elif method == "tsne":
            coords_2d = self._tsne(X, perplexity=min(tsne_perplexity, n - 1))
            explained = None
        else:
            raise ValueError(f"Unknown method: {method}")

        points = []
        for i, (coord, label) in enumerate(zip(coords_2d, labels)):
            points.append(Point2D(
                x=round(float(coord[0]), 4),
                y=round(float(coord[1]), 4),
                label=label,
                original_index=i,
                group=groups[i] if groups else None,
            ))

        return ReductionResult(
            method=method,
            points=points,
            explained_variance=explained,
            n_input_dimensions=d,
            n_points=n,
        )

    def _pca(self, X: np.ndarray):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        explained = round(float(sum(pca.explained_variance_ratio_)) * 100, 1)
        return coords, explained

    def _tsne(self, X: np.ndarray, perplexity: int):
        from openTSNE import TSNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_jobs=-1,
            random_state=42,
        )
        return tsne.fit(X)
