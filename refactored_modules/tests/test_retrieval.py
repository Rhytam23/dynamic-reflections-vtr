import numpy as np
import pytest

from refactored_modules.retrieval import (build_fusion_prompt, evaluate_retrieval, fit_ridge_map,
                                          recall_at_k, split_indices, top_captions)


def _paired(n=300, layers=2, dv=40, dt=60, noise=0.3, seed=0, shared=True):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 12))
    def side(dim, s):
        r = np.random.default_rng(s)
        proj = r.normal(size=(12, dim))
        base = (z if shared else r.normal(size=(n, 12))) @ proj
        return np.stack([base + noise * r.normal(size=(n, dim)) for _ in range(layers)], 1)
    return side(dv, 1), side(dt, 2)


def test_recall_at_k_perfect_and_worst():
    assert recall_at_k(np.eye(5) + 0.0, ks=(1,))["recall@1"] == 1.0
    worst = -np.eye(5)
    assert recall_at_k(worst, ks=(1, 5))["recall@1"] == 0.0
    assert recall_at_k(worst, ks=(5,))["recall@5"] == 1.0


def test_split_is_disjoint_and_complete():
    tr, te = split_indices(50, 0.2)
    assert len(te) == 10 and set(tr).isdisjoint(te) and len(set(tr) | set(te)) == 50


def test_ridge_dual_matches_primal():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(30, 50)), rng.normal(size=(30, 7))
    dual = fit_ridge_map(x, y, 1.0)  # p > n -> dual branch
    xn, yn = x / np.linalg.norm(x, axis=1, keepdims=True), y / np.linalg.norm(y, axis=1, keepdims=True)
    primal = np.linalg.solve(xn.T @ xn + np.eye(50), xn.T @ yn)
    np.testing.assert_allclose(dual, primal, atol=1e-6)


def test_retrieval_beats_chance_on_aligned_data():
    v, t = _paired()
    res = evaluate_retrieval(v, t, k_align=5)
    assert res["ridge"]["recall@5"] > 5 * res["chance"]["recall@5"]
    assert res["relative"]["recall@5"] > 3 * res["chance"]["recall@5"]
    assert res["n_train"] + res["n_test"] == 300


def test_retrieval_near_chance_on_unaligned_data():
    v, t = _paired(shared=False)
    res = evaluate_retrieval(v, t, k_align=5)
    assert res["ridge"]["recall@10"] < 3 * res["chance"]["recall@10"]


def test_prompt_and_top_captions():
    caps = ["a dog runs", "a cat sleeps", "a car drives"]
    assert top_captions(np.array([0.1, 0.9, 0.5]), caps, k=2) == ["a cat sleeps", "a car drives"]
    p = build_fusion_prompt(caps[:2])
    assert "- a dog runs" in p and p.endswith("Description:")
