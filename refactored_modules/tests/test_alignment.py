import numpy as np
import pytest

from refactored_modules.alignment import layer_sweep, mutual_knn, remove_outliers


def _reference_mutual_knn(a, b, topk):
    """Independent, sort-based transcription of the authors' vprh.metrics.mutual_knn."""
    def knn(f):
        f = f / np.linalg.norm(f, axis=1, keepdims=True)
        sim = f @ f.T
        np.fill_diagonal(sim, -1e8)
        return np.argsort(-sim, axis=1)[:, :topk]
    ka, kb = knn(a), knn(b)
    n = ka.shape[0]
    am, bm = np.zeros((n, n)), np.zeros((n, n))
    rows = np.arange(n)[:, None]
    am[rows, ka], bm[rows, kb] = 1.0, 1.0
    return ((am * bm).sum(1) / topk).mean()


def test_matches_authors_algorithm():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(120, 10))
    x = base @ rng.normal(size=(10, 24)) + 0.5 * rng.normal(size=(120, 24))
    y = base @ rng.normal(size=(10, 40)) + 0.5 * rng.normal(size=(120, 40))
    assert mutual_knn(x, y, k=10) == pytest.approx(_reference_mutual_knn(x, y, 10), abs=1e-6)


def test_identical_spaces_score_one():
    x = np.random.default_rng(0).normal(size=(60, 16))
    assert mutual_knn(x, x, k=5) == pytest.approx(1.0)


def test_rotation_invariant_and_different_dims():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert mutual_knn(x, x @ q, k=5) == pytest.approx(1.0)
    proj = rng.normal(size=(8, 32))
    assert mutual_knn(x, x @ proj, k=5) > 0.5  # same structure, different dimensionality


def test_random_spaces_near_chance():
    rng = np.random.default_rng(2)
    n, k = 400, 10
    score = mutual_knn(rng.normal(size=(n, 16)), rng.normal(size=(n, 16)), k=k)
    assert score == pytest.approx(k / (n - 1), abs=0.02)


def test_layer_sweep_finds_planted_pair():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(80, 12))
    fx = rng.normal(size=(80, 3, 12))
    fy = rng.normal(size=(80, 4, 12))
    fx[:, 2], fy[:, 1] = base, base
    scores, best = layer_sweep(fx, fy, k=5, outlier_q=None)
    assert scores.shape == (4, 5)  # +1 row/col for the all-layers concat
    assert (best["layer_x"], best["layer_y"]) == (2, 1)
    assert best["score"] == pytest.approx(1.0)


def test_outlier_clamp_bounds_values():
    rng = np.random.default_rng(4)
    f = rng.normal(size=(20, 2, 500)).astype(np.float32)  # 1000 values per sample
    f[0, 0, 0] = 1e6
    out = remove_outliers(f, q=0.95)
    assert np.abs(out).max() < 10
    assert remove_outliers(f, q=1) is f


def test_bad_inputs():
    x = np.zeros((10, 4))
    with pytest.raises(ValueError):
        mutual_knn(x, np.zeros((9, 4)), k=3)
    with pytest.raises(ValueError):
        mutual_knn(x, x, k=10)
