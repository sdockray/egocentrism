"""
Dimensionality Reduction Module (`src/sonic/reduction.py`)

Reduces high-dimensional MFCC vectors (20D) to 2D spatial coordinates (X, Y)
using t-SNE, PCA, or SVD, normalizing coordinates into a [-1, 1] bounding box.
"""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_mfccs_to_2d(
    segments: List[Dict],
    method: str = "tsne",
    random_state: int = 42,
) -> List[Dict]:
    """
    Takes a list of segment dicts containing 'mfcc_vector' (20D float list).
    Computes 2D projection (X, Y) for each segment and normalizes to [-1, 1].
    Sets 'x' and 'y' fields on segment dicts.
    """
    if not segments:
        return segments

    X_matrix = np.array([seg["mfcc_vector"] for seg in segments], dtype=np.float32)

    # Standardize features (zero mean, unit variance)
    mean = np.mean(X_matrix, axis=0)
    std = np.std(X_matrix, axis=0) + 1e-6
    X_scaled = (X_matrix - mean) / std

    if method.lower() == "tsne":
        # Perplexity safety check based on sample size
        n_samples = len(X_scaled)
        perplexity = min(30, max(5, n_samples // 5))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca")
        coords_2d = reducer.fit_transform(X_scaled)
    else:
        reducer = PCA(n_components=2, random_state=random_state)
        coords_2d = reducer.fit_transform(X_scaled)

    # Min-Max Normalization to [-1.0, 1.0] bounding box
    min_x, max_x = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    min_y, max_y = coords_2d[:, 1].min(), coords_2d[:, 1].max()

    norm_x = (coords_2d[:, 0] - min_x) / (max_x - min_x + 1e-6) * 2.0 - 1.0
    norm_y = (coords_2d[:, 1] - min_y) / (max_y - min_y + 1e-6) * 2.0 - 1.0

    for idx, seg in enumerate(segments):
        seg["x"] = float(round(norm_x[idx], 4))
        seg["y"] = float(round(norm_y[idx], 4))

    return segments


if __name__ == "__main__":
    # Test reduction on dummy vectors
    dummy_segs = [
        {"segment_id": f"seg_{i}", "mfcc_vector": list(np.random.randn(20))}
        for i in range(50)
    ]
    reduced = reduce_mfccs_to_2d(dummy_segs, method="tsne")
    print("Sample reduced coordinate:", reduced[0]["x"], reduced[0]["y"])
