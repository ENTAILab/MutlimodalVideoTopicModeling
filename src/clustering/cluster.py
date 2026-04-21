from __future__ import annotations

from typing import Any

import hdbscan
import numpy as np
import umap


def run_umap_hdbscan(
    vectors: np.ndarray,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 8,
    umap_metric: str = "cosine",
    hdbscan_min_cluster_size: int = 5,
    hdbscan_metric: str = "euclidean",
    random_state: int = 42,
) -> dict[str, Any]:
    if len(vectors) == 0:
        raise ValueError("No vectors for clustering")

    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        metric=umap_metric,
        random_state=random_state,
    )
    reduced = reducer.fit_transform(vectors)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        metric=hdbscan_metric,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)

    return {
        "reduced": reduced,
        "labels": labels,
        "probabilities": getattr(clusterer, "probabilities_", None),
    }
