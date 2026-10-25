"""Runs the real code cells of colab_run_pfinal.ipynb with Colab, the network, the GPU and model
extraction faked. Catches wiring mistakes (names, stage dependencies, paths, file naming) that would
otherwise cost a GPU session to discover."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from refactored_modules import colab_inference, features, pipeline, preflight, runner, sweeps, utils

NOTEBOOK = Path(__file__).resolve().parents[2] / "colab_run_pfinal.ipynb"
N_VIDEOS = 60


def _fake_clone(repo_path, persist_dir=None):
    repo = Path(repo_path)
    (repo / "assets").mkdir(parents=True, exist_ok=True)
    (repo / "results").mkdir(exist_ok=True)
    (repo / "data").mkdir(exist_ok=True)
    (repo / "assets" / "pe_video_dataset_1k_rephrased.jsonl").write_text(
        "\n".join(json.dumps({"video_id": f"vid{i}", "rephrased_captions": [f"caption {i}.{j}" for j in range(10)]})
                  for i in range(N_VIDEOS)) + "\n")


def _fake_download(json_path, output_dir, repo_path=None):
    out = Path(repo_path) / output_dir
    out.mkdir(parents=True, exist_ok=True)
    for line in (Path(repo_path) / json_path).read_text().splitlines():
        (out / f"{json.loads(line)['video_id']}.bin").write_bytes(b"x")


def _make_extract(fail_models=()):
    def fake_extract(repo_path, which, config="pvd_sample", llm_names=None, video_names=None, num_captions=-1,
                     num_frames=-1, feature_dir=None, hf_token=None, annotation_path=None):
        repo = Path(repo_path)
        n = len((repo / (annotation_path or "assets/pe_video_dataset_1k_rephrased.jsonl")).read_text().splitlines())
        fdir = repo / (feature_dir or "results/sample")
        latent = np.random.default_rng(0).normal(size=(N_VIDEOS, 8))[:n]

        def save(path, dim, layers, sig, seed):
            proj = np.random.default_rng(seed).normal(size=(8, dim))
            noise = np.random.default_rng(seed + 1)
            feats = np.stack([sig * (latent @ proj) + noise.normal(size=(n, dim)) for _ in range(layers)], 1).astype(np.float32)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                return  # the authors' script skips existing files
            torch.save({"feats": torch.from_numpy(feats)}, path)

        if which == "llm":
            for m in llm_names:
                sig = 1.0 if num_captions == -1 else min(1.0, 0.25 + 0.2 * num_captions)
                save(features.feature_path(fdir, "pvd", m, "avg", num_captions=num_captions), 24, 4, sig, 11)
        else:
            for m in video_names:
                if m in fail_models:
                    raise RuntimeError(f"simulated failure loading {m}")
                pool = "avg" if "videomae" in m else "cls"
                sig = 1.0 if num_frames == -1 else min(1.0, 0.25 + 0.05 * num_frames)
                save(features.feature_path(fdir, "pvd", m, pool, num_frames=num_frames), 16, 3, sig, 22)
    return fake_extract


def run_notebook(monkeypatch, tmp_path, *, gpu=True, hf_error=None, fail_models=()):
    unassigned = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda p: None)
    colab.userdata = types.SimpleNamespace(get=lambda k: "tok")
    colab.runtime = types.SimpleNamespace(unassign=lambda: unassigned.append(True))
    ipy = types.ModuleType("IPython.display")
    ipy.Image, ipy.display = (lambda *a, **k: None), (lambda *a, **k: None)
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: None  # matplotlib probes these when it initialises its backend
    ipython.version_info = (9, 0, 0, "final")
    for name, mod in (("google", types.ModuleType("google")), ("google.colab", colab),
                      ("IPython", ipython), ("IPython.display", ipy)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("GENAI_BASE", str(tmp_path))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: types.SimpleNamespace(returncode=0))  # git clone / pull
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)

    fake_extract = _make_extract(fail_models)
    for mod in (features, sweeps):
        monkeypatch.setattr(mod, "extract_features", fake_extract)
    monkeypatch.setattr(utils, "clone_authors_repo", _fake_clone)
    monkeypatch.setattr(utils, "patch_scripts", lambda repo: None)
    monkeypatch.setattr(utils, "setup_environment", lambda repo, persist=None: None)
    monkeypatch.setattr(pipeline, "download_pvd_dataset", _fake_download)
    monkeypatch.setattr(preflight, "has_gpu", lambda: gpu)
    monkeypatch.setattr(preflight, "check_gpu", lambda: "Fake T4, 15.0 GB")
    monkeypatch.setattr(preflight, "check_disk", lambda p, g: 100.0)

    def hf(token, *a, **k):
        if hf_error:
            raise preflight.PreflightError(hf_error)
        return ["google/gemma-2-2b-it", "facebook/PE-Video"]
    monkeypatch.setattr(preflight, "check_hf_access", hf)
    monkeypatch.setattr(colab_inference, "sample_frames", lambda p, n=5: [np.zeros((8, 8, 3), np.uint8)] * n)
    monkeypatch.setattr(colab_inference, "generate_descriptions", lambda batches, **k: ["fused"] * len(batches))

    ns = {}
    for cell in json.loads(NOTEBOOK.read_text(encoding="utf8"))["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "<notebook cell>", "exec"), ns)
    return ns, unassigned


def test_happy_path_runs_every_stage_and_writes_results(monkeypatch, tmp_path):
    ns, unassigned = run_notebook(monkeypatch, tmp_path)
    run = ns["run"]
    assert set(run.status.values()) == {"ok"}, run.summary()
    res = ns["RES"]
    for name in ("baseline.json", "sweep_dinov2_large_video.json", "sweep_videomaev2_base.json", "comparison.md",
                 "retrieval.json", "baseline.layers.npy"):
        assert (res / name).exists(), name
    assert (res / "figures" / "sweeps.png").exists() and (res / "figures" / "layer_heatmap.png").exists()
    base = json.loads((res / "baseline.json").read_text())
    assert base["n"] == N_VIDEOS and base["mutual_knn_score"] > base["chance_level"]
    sweep = json.loads((res / "sweep_dinov2_large_video.json").read_text())
    assert len(sweep["grid"]) == 5 * 4 and sweep["fit"] is not None
    assert (tmp_path / "drive" / "MyDrive" / "GenAI_Project" / "run_summary.json").exists()
    assert unassigned == [True]  # GPU released at the end


def test_hf_access_failure_aborts_before_anything_expensive(monkeypatch, tmp_path):
    ns, unassigned = run_notebook(monkeypatch, tmp_path, hf_error="google/gemma-2-2b-it: access denied")
    run = ns["run"]
    assert run.status["Preflight: token + disk"] == "FAILED"
    assert all(s == "skipped" for n, s in run.status.items() if n != "Preflight: token + disk" and n != "GPU")
    assert unassigned == [True]  # do not leave the GPU running after a failure
    assert not (ns["REPO"] / "results" / "smoke").exists()  # no model was ever loaded


def test_cpu_runtime_only_prepares_data(monkeypatch, tmp_path):
    ns, unassigned = run_notebook(monkeypatch, tmp_path, gpu=False)
    run = ns["run"]
    assert run.status["PVD videos"] == "ok" and run.status["GPU"] == "skipped"
    assert run.status["Environment"] == "skipped" and run.status["Baseline alignment"] == "skipped"
    assert len(list((ns["REPO"] / "data" / "pvd").glob("*.bin"))) == N_VIDEOS


def test_videomae_failure_does_not_stop_dinov2(monkeypatch, tmp_path):
    ns, _ = run_notebook(monkeypatch, tmp_path, fail_models=("videomaev2_base",))
    run = ns["run"]
    assert run.status["Smoke test: VideoMAEv2"] == "FAILED"
    assert run.status["Sweep: videomaev2_base"] == "skipped"
    for name in ("Smoke test", "Baseline alignment", "Sweep: dinov2_large_video", "Report", "Retrieval"):
        assert run.status[name] == "ok", run.summary()
    assert "dinov2_large_video" in (ns["RES"] / "comparison.md").read_text()
    assert "videomaev2_base" not in (ns["RES"] / "comparison.md").read_text()
