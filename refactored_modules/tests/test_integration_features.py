"""Everything after extraction, run on real torch .pt files named like the authors' code names them."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from refactored_modules import sweeps
from refactored_modules.features import feature_path, load_features
from refactored_modules.pipeline import run_alignment_pipeline
from refactored_modules.preflight import check_feature_file
from refactored_modules.retrieval import evaluate_retrieval

N, K = 120, 5


def _save(path, feats, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feats": torch.from_numpy(feats), **extra}, path)  # authors also store num_params/good/loss


def _make(latent, dim, layers, signal, seed):
    r = np.random.default_rng(seed)
    proj = np.random.default_rng(seed + 1000).normal(size=(latent.shape[1], dim))
    return np.stack([signal * (latent @ proj) + r.normal(size=(len(latent), dim)) for _ in range(layers)], 1).astype(np.float32)


@pytest.fixture()
def feature_dir(tmp_path):
    latent = np.random.default_rng(0).normal(size=(N, 8))
    d = tmp_path / "results" / "sample"
    # LLM: 4 layers x 24 dims; caption count changes the signal strength
    for nc, sig in ((1, 0.4), (2, 0.7), (4, 1.0)):
        _save(feature_path(d, "pvd", "gemma2-2b-it", "avg", num_captions=nc), _make(latent, 24, 4, sig, 10 + nc), loss=torch.tensor(1.0))
    _save(feature_path(d, "pvd", "gemma2-2b-it", "avg"), _make(latent, 24, 4, 1.0, 99))  # default = all captions
    # vision: 3 layers x 16 dims
    for nf, sig in ((1, 0.4), (2, 0.7), (4, 1.0)):
        _save(feature_path(d, "pvd", "dinov2_large_video", "cls", num_frames=nf), _make(latent, 16, 3, sig, 20 + nf), num_params=1, good=[])
    _save(feature_path(d, "pvd", "dinov2_large_video", "cls"), _make(latent, 16, 3, 1.0, 88))
    return d


def test_load_and_validate_real_pt(feature_dir):
    p = feature_path(feature_dir, "pvd", "gemma2-2b-it", "avg")
    assert load_features(p).dtype == np.float32
    assert check_feature_file(p, N) == (N, 4, 24)


def test_baseline_pipeline_writes_json_with_context(feature_dir, tmp_path):
    res = run_alignment_pipeline(feature_dir, "gemma2-2b-it", "dinov2_large_video", k=K, output_json=tmp_path / "out" / "baseline.json")
    assert res["n"] == N and res["k"] == K and res["chance_level"] == pytest.approx(K / (N - 1))
    assert res["mutual_knn_score"] > 3 * res["chance_level"]
    assert (tmp_path / "out" / "baseline.json").exists() and (tmp_path / "out" / "baseline.layers.npy").exists()


def test_grid_fit_and_caption_alias(feature_dir):
    llm4 = feature_path(feature_dir, "pvd", "gemma2-2b-it", "avg", num_captions=4)
    llm4.unlink()  # pretend the 4-caption run was never done ...
    assert sweeps.alias_full_caption_features(feature_dir, "gemma2-2b-it", 4)  # ... and reuse the default run
    assert llm4.exists()
    res = sweeps.compute_grid(feature_dir, "gemma2-2b-it", "dinov2_large_video", [1, 2, 4], [1, 2, 4], k=K)
    by = {(c["n_frames"], c["n_captions"]): c["score"] for c in res["grid"]}
    assert by[(4, 4)] > by[(1, 1)]
    res = sweeps.fit_grid(res)
    assert res["fit"] is not None and "dinov2_large_video" in sweeps.comparison_table([res])


def test_retrieval_on_loaded_features(feature_dir):
    v = load_features(feature_path(feature_dir, "pvd", "dinov2_large_video", "cls"))
    t = load_features(feature_path(feature_dir, "pvd", "gemma2-2b-it", "avg"))
    ev = evaluate_retrieval(v, t, k_align=K)
    assert ev["ridge"]["recall@5"] > 3 * ev["chance"]["recall@5"]
