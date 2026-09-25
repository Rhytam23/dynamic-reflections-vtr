"""Video -> text retrieval through the aligned spaces (the 'video to text' step).

The video and text spaces have different dimensionality, so mutual k-NN alignment alone
cannot compare a video to a caption directly. Two ways to bridge them, both of which need
a set of paired (video, caption) examples (i.e. this is NOT zero-shot):
  * ridge:    fit a linear map video-space -> text-space on the training pairs.
  * relative: describe every item by its cosine similarities to the training items in its
              own space ("relative representations"); no parameters are fitted.
Layers are chosen with mutual k-NN on the training split only, so the held-out
evaluation is not leaked.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from .alignment import layer_sweep, remove_outliers


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def split_indices(n: int, test_frac: float = 0.2, seed: int = 0):
    perm = np.random.default_rng(seed).permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def fit_ridge_map(x_train: np.ndarray, y_train: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """W minimising ||XW - Y||^2 + lam ||W||^2 (dual form when dim > n_train)."""
    x, y = _normalize(x_train), _normalize(y_train)
    n, p = x.shape
    if p > n:
        return x.T @ np.linalg.solve(x @ x.T + lam * np.eye(n), y)
    return np.linalg.solve(x.T @ x + lam * np.eye(p), x.T @ y)


def recall_at_k(sim: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """sim[i, j] = score of video i vs caption j; the correct caption of video i is j == i."""
    ranks = (sim > sim.diagonal()[:, None]).sum(axis=1)  # 0-based rank of the true caption
    return {f"recall@{k}": float((ranks < k).mean()) for k in ks}


def evaluate_retrieval(
    video_feats: np.ndarray, text_feats: np.ndarray, k_align: int = 10,
    test_frac: float = 0.2, lam: float = 1.0, ks: Sequence[int] = (1, 5, 10),
    seed: int = 0, outlier_q: Optional[float] = 0.95,
) -> Dict:
    """Held-out video->caption retrieval on [N, layers, dim] features."""
    v, t = np.asarray(video_feats, np.float32), np.asarray(text_feats, np.float32)
    if outlier_q is not None:
        v, t = remove_outliers(v, outlier_q), remove_outliers(t, outlier_q)
    train, test = split_indices(len(v), test_frac, seed)
    # Layer choice on train only. -1 means "all layers concatenated".
    _, best = layer_sweep(v[train], t[train], k=min(k_align, len(train) - 1), outlier_q=None)
    pick = lambda f, layer: f.reshape(len(f), -1) if layer == -1 else f[:, layer]
    xv, xt = pick(v, best["layer_x"]), pick(t, best["layer_y"])

    w = fit_ridge_map(xv[train], xt[train], lam)
    sim_ridge = _normalize(_normalize(xv[test]) @ w) @ _normalize(xt[test]).T

    rel_v = _normalize(xv[test]) @ _normalize(xv[train]).T
    rel_t = _normalize(xt[test]) @ _normalize(xt[train]).T
    sim_rel = _normalize(rel_v - rel_v.mean(1, keepdims=True)) @ _normalize(rel_t - rel_t.mean(1, keepdims=True)).T

    n_test = len(test)
    return {
        "n_train": int(len(train)), "n_test": int(n_test),
        "vision_layer": best["layer_x"], "text_layer": best["layer_y"],
        "chance": {f"recall@{k}": min(1.0, k / n_test) for k in ks},
        "ridge": recall_at_k(sim_ridge, ks),
        "relative": recall_at_k(sim_rel, ks),
        "test_indices": test.tolist(),
        "_sims": {"ridge": sim_ridge, "relative": sim_rel},
    }


def top_captions(sim_row: np.ndarray, captions: Sequence[str], k: int = 3) -> List[str]:
    return [captions[j] for j in np.argsort(-sim_row)[:k]]


def build_fusion_prompt(retrieved: Sequence[str]) -> str:
    """Prompt asking an LLM to merge retrieved captions into one description."""
    lines = "\n".join(f"- {c}" for c in retrieved)
    return (
        "These captions were retrieved for a video that we cannot show you. "
        "Write one short description of the video that keeps only what the captions agree on. "
        "Do not add details that are not in the captions.\n\nCaptions:\n" + lines + "\n\nDescription:"
    )
