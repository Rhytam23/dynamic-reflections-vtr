"""Mutual k-NN alignment (Eq. 1 of the DynamicReflections paper / Huh et al. 2024).

Scoring mirrors the authors' `vprh.alignment.compute_score`: features are clamped at
the q-th abs-value quantile, then every (layer_x, layer_y) pair is scored, plus the
"all layers concatenated" option (reported as layer index -1).
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def remove_outliers(feats: np.ndarray, q: float = 0.95) -> np.ndarray:
    """Clamp to +-(mean over samples of the per-sample q-quantile of |feats|).

    Same as `vprh.metrics.remove_outliers(exact=False)`. q == 1 disables it.
    """
    if q >= 1:
        return feats
    flat = np.abs(feats.reshape(feats.shape[0], -1))
    q_val = float(np.quantile(flat, q, axis=1).mean())
    return np.clip(feats, -q_val, q_val)


def _knn_indices(feats: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest neighbours (cosine) of every row, excluding itself."""
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sim = feats @ feats.T
    np.fill_diagonal(sim, -np.inf)
    return np.argpartition(-sim, k - 1, axis=1)[:, :k]


def _overlap(nx: np.ndarray, ny: np.ndarray, k: int) -> float:
    n = nx.shape[0]
    return sum(len(set(a) & set(b)) for a, b in zip(nx, ny)) / (k * n)


def _check_k(n: int, k: int):
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < N (k={k}, N={n})")


def mutual_knn(x, y, k: int = 10) -> float:
    """Mean overlap of k-NN sets of x [N, p] and y [N, q]; 1.0 is perfect alignment."""
    x, y = _to_numpy(x), _to_numpy(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Expected [N, p] and [N, q] with equal N, got {x.shape} and {y.shape}")
    _check_k(x.shape[0], k)
    return _overlap(_knn_indices(x, k), _knn_indices(y, k), k)


def layer_sweep(
    feats_x, feats_y, k: int = 10, outlier_q: Optional[float] = 0.95, include_concat: bool = True
) -> Tuple[np.ndarray, Dict]:
    """Score every (layer_x, layer_y) pair of [N, L, D] features.

    The returned matrix has shape [Lx + 1, Ly + 1]; row/column 0 is the concatenation of
    all layers (as in the authors' code), row/column i + 1 is layer i. The best-pair dict
    reports layers as 0-based indices, with -1 meaning "all layers concatenated".
    """
    fx, fy = _to_numpy(feats_x), _to_numpy(feats_y)
    if fx.ndim != 3 or fy.ndim != 3 or fx.shape[0] != fy.shape[0]:
        raise ValueError(f"Expected [N, layers, dim] features with equal N, got {fx.shape}, {fy.shape}")
    n = fx.shape[0]
    _check_k(n, k)
    if outlier_q is not None:
        fx, fy = remove_outliers(fx, outlier_q), remove_outliers(fy, outlier_q)

    def neighbours(f):  # index 0 = concat, index i + 1 = layer i; computed once per layer
        sets = [_knn_indices(f.reshape(n, -1), k) if include_concat else None]
        sets += [_knn_indices(f[:, i], k) for i in range(f.shape[1])]
        return sets

    nx, ny = neighbours(fx), neighbours(fy)
    scores = np.zeros((len(nx), len(ny)), dtype=np.float32)
    for i, a in enumerate(nx):
        for j, b in enumerate(ny):
            if a is None or b is None:
                continue
            scores[i, j] = _overlap(a, b, k)
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    best = {"layer_x": int(i) - 1, "layer_y": int(j) - 1, "score": float(scores[i, j]), "k": k, "n": int(n)}
    logger.info("Best alignment %.4f at layers (%d, %d), N=%d, k=%d", best["score"], best["layer_x"], best["layer_y"], n, k)
    return scores, best
