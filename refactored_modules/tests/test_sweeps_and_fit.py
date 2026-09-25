import os
import numpy as np
import pytest

from refactored_modules import features, sweeps
from refactored_modules.plots import plot_fit, plot_layer_heatmap, plot_sweeps
from refactored_modules.alignment import layer_sweep
from refactored_modules.scaling_law import fit_scaling_law, predict, saturation_model

# VideoMAEv2 vs Gemma-2 parameters reported in the paper (Sec. 6)
PAPER = dict(S_inf=0.41, C_f=0.15, C_c=0.13, alpha=0.75, beta=1.30)
FRAMES, CAPS = [1, 2, 4, 8, 16, 32, 48], [1, 2, 4, 10]


def _grid(noise=0.0, seed=0):
    nf, nc = np.meshgrid(FRAMES, CAPS)
    nf, nc = nf.ravel().astype(float), nc.ravel().astype(float)
    y = saturation_model((nf, nc), *PAPER.values())
    return nf, nc, y + np.random.default_rng(seed).normal(scale=noise, size=y.shape)


def test_fit_recovers_paper_parameters_noise_free():
    nf, nc, y = _grid()
    fit = fit_scaling_law(nf, nc, y)
    assert fit["r2"] > 0.999
    for name, true in PAPER.items():
        assert fit[name] == pytest.approx(true, rel=0.05), name


def test_fit_with_noise_still_predictive():
    nf, nc, y = _grid(noise=0.005)
    fit = fit_scaling_law(nf, nc, y)
    assert fit["r2"] > 0.95
    assert fit["S_inf"] == pytest.approx(PAPER["S_inf"], abs=0.05)
    assert predict(fit, [16], [4]).shape == (1,)


def test_fit_needs_enough_points():
    with pytest.raises(ValueError):
        fit_scaling_law([1, 2, 1, 2], [1, 1, 2, 2], [0.1, 0.2, 0.2, 0.3])


def test_feature_path_matches_authors_naming():
    p = features.feature_path("results/sample", "pvd", "gemma2-2b-it", "avg")
    assert p.as_posix() == "results/sample/pvd/gemma2-2b-it_pool-avg.pt"
    p = features.feature_path("r", "pvd", "dinov2_large_video", "cls", num_frames=8)
    assert p.name == "dinov2_large_video_pool-cls_numframes-8.pt"
    p = features.feature_path("r", "pvd", "gemma2-2b-it", "avg", num_captions=4)
    assert p.name == "gemma2-2b-it_pool-avg_numcap-4.pt"


def test_extract_features_builds_expected_command(monkeypatch):
    seen = {}
    monkeypatch.setattr(features, "run_streaming", lambda cmd, **kw: seen.update(cmd=cmd, kw=kw))
    features.extract_features("/repo", "llm", llm_names=["gemma2-2b-it"], num_captions=4,
                              feature_dir="out", hf_token="tok")
    assert seen["cmd"][:5] == ["uv", "run", "scripts/main_extract.py", "pvd_sample", "--llm_only"]
    assert ["--llm_names", "gemma2-2b-it"] == seen["cmd"][5:7]
    assert "--num_captions" in seen["cmd"] and "--feature_dir" in seen["cmd"]
    assert seen["kw"]["env"]["HF_TOKEN"] == "tok" and seen["kw"]["cwd"] == "/repo"
    with pytest.raises(ValueError):
        features.extract_features("/repo", "llm")  # would silently fall back to the 9B model


def _fake_features(n=90, layers=3, dim=20, seed=0):
    """More frames/captions => more of the shared latent leaks into each modality."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 8))
    pv, pt = rng.normal(size=(8, dim)), rng.normal(size=(8, dim))
    def make(signal, proj, s):
        r = np.random.default_rng(s)
        return np.stack([signal * (latent @ proj) + r.normal(size=(n, dim)) for _ in range(layers)], 1)
    return make, pv, pt


def test_compute_grid_is_monotone_in_signal(tmp_path):
    make, pv, pt = _fake_features()
    sig_f = {1: 0.3, 4: 0.8, 16: 1.6}
    sig_c = {1: 0.3, 4: 0.8, 10: 1.6}
    store = {}
    for nf, s in sig_f.items():
        store[features.feature_path(tmp_path, "pvd", "vid", "cls", num_frames=nf)] = make(s, pv, nf)
    for nc, s in sig_c.items():
        store[features.feature_path(tmp_path, "pvd", "llm", "avg", num_captions=nc)] = make(s, pt, 100 + nc)
    res = sweeps.compute_grid(tmp_path, "llm", "vid", list(sig_f), list(sig_c), k=5, load=lambda p: store[p])
    assert res["n"] == 90 and res["chance_level"] == pytest.approx(5 / 89)
    by = {(c["n_frames"], c["n_captions"]): c["score"] for c in res["grid"]}
    assert by[(16, 10)] > by[(4, 4)] > by[(1, 1)]
    assert by[(1, 1)] < 0.3  # weak signal stays close to chance level
    res = sweeps.fit_grid(res)  # 3x3 grid is enough to fit
    assert res["fit"] is not None
    table = sweeps.comparison_table([res])
    assert "vid" in table and "best score" in table


def test_plots_write_files(tmp_path):
    make, pv, pt = _fake_features()
    scores, best = layer_sweep(make(1.0, pv, 1), make(1.0, pt, 2), k=5)
    plot_layer_heatmap(scores, best, tmp_path / "heat.png")
    nf, nc, y = _grid()
    result = {"vision": "m", "llm": "l", "n": 100, "k": 10, "chance_level": 0.1,
              "grid": [{"n_frames": int(a), "n_captions": int(b), "score": float(c)} for a, b, c in zip(nf, nc, y)]}
    result = sweeps.fit_grid(result)
    plot_sweeps([result], tmp_path / "sweep.png")
    plot_fit(result, tmp_path / "fit.png")
    for name in ("heat.png", "sweep.png", "fit.png"):
        assert (tmp_path / name).stat().st_size > 1000


def test_cuda_library_env_finds_venv_nvidia_dirs(tmp_path):
    from refactored_modules.utils import cuda_library_env
    lib = tmp_path / ".venv" / "lib" / "python3.13" / "site-packages" / "nvidia" / "npp" / "lib"
    lib.mkdir(parents=True)
    env = cuda_library_env(tmp_path, {"LD_LIBRARY_PATH": "/existing", "X": "1"})
    parts = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert str(lib) in parts and parts[-1] == "/existing" and env["X"] == "1"
