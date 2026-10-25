import json

import numpy as np
import pytest

from refactored_modules import preflight
from refactored_modules.preflight import PreflightError
from refactored_modules.runner import SkipStage, StageRunner


# ---------- runner ----------
def test_runner_records_failures_and_skips_dependents(capsys):
    run = StageRunner()

    @run.stage("a")
    def _():
        return 1

    @run.stage("b")
    def _():
        raise ValueError("boom")

    @run.stage("c", needs=["b"])
    def _():
        raise AssertionError("must not run")

    @run.stage("d", needs=["a"])
    def _():
        return 4

    assert run.status == {"a": "ok", "b": "FAILED", "c": "skipped", "d": "ok"}
    assert "boom" in run.detail["b"] and run.results["d"] == 4
    assert run.aborted is None


def test_critical_failure_aborts_everything_after():
    run = StageRunner()

    @run.stage("setup", critical=True)
    def _():
        raise RuntimeError("no gpu")

    @run.stage("later")
    def _():
        raise AssertionError("must not run")

    assert run.status["later"] == "skipped" and "setup" in run.detail["later"]


def test_skipstage_is_not_a_failure_and_summary_saves(tmp_path):
    run = StageRunner()

    @run.stage("gpu")
    def _():
        raise SkipStage("cpu runtime")

    assert run.status["gpu"] == "skipped"
    run.finish(disconnect=False, summary_path=tmp_path / "s.json")
    assert json.loads((tmp_path / "s.json").read_text())["gpu"]["detail"] == "cpu runtime"


# ---------- preflight ----------
class GatedRepoError(Exception):
    pass


class RepositoryNotFoundError(Exception):
    pass


def test_hf_access_lists_every_problem():
    def head(repo, kind, token):
        if "gemma" in repo:
            raise GatedRepoError("403")
        if "PE-Video" in repo:
            raise RepositoryNotFoundError("404")
    with pytest.raises(PreflightError) as e:
        preflight.check_hf_access("tok", head=head)
    msg = str(e.value)
    assert "gemma-2-2b-it" in msg and "accept the license" in msg and "PE-Video" in msg and "not found" in msg


def test_hf_access_ok_and_missing_token():
    assert preflight.check_hf_access("tok", head=lambda *a: None) == ["google/gemma-2-2b-it", "facebook/PE-Video"]
    with pytest.raises(PreflightError, match="HF_TOKEN"):
        preflight.check_hf_access(None, head=lambda *a: None)


def test_check_gpu_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda _: None)
    assert preflight.has_gpu() is False
    with pytest.raises(PreflightError, match="T4"):
        preflight.check_gpu()


def test_disk_check(tmp_path):
    assert preflight.check_disk(tmp_path, 0.001) > 0
    with pytest.raises(PreflightError):
        preflight.check_disk(tmp_path, 10 ** 9)


def _write_ann(path, n, caps=10):
    path.write_text("\n".join(json.dumps({"video_id": f"v{i}", "rephrased_captions": ["c"] * caps}) for i in range(n)) + "\n")


def test_smoke_annotation_and_video_presence(tmp_path):
    ann = tmp_path / "a.jsonl"
    _write_ann(ann, 12)
    ids = preflight.make_smoke_annotation(ann, tmp_path / "sub" / "smoke.jsonl", n=8)
    assert ids == [f"v{i}" for i in range(8)]
    data = tmp_path / "data"
    data.mkdir()
    for i in range(8):
        (data / f"v{i}.bin").write_bytes(b"x")
    assert preflight.check_videos_present(tmp_path / "sub" / "smoke.jsonl", data) == 8
    (data / "v3.bin").write_bytes(b"")  # empty file counts as missing
    with pytest.raises(PreflightError, match="1 of 8"):
        preflight.check_videos_present(tmp_path / "sub" / "smoke.jsonl", data)


def test_feature_file_checks(tmp_path):
    good = np.ones((8, 3, 4), np.float32)
    assert preflight.check_feature_file("f.pt", 8, load=lambda p: good) == (8, 3, 4)
    with pytest.raises(PreflightError, match="expected 9"):
        preflight.check_feature_file("f.pt", 9, load=lambda p: good)
    bad = good.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(PreflightError, match="NaN"):
        preflight.check_feature_file("f.pt", 8, load=lambda p: bad)


def test_caption_counts_uniform(tmp_path):
    a = tmp_path / "a.jsonl"
    _write_ann(a, 5, caps=10)
    assert preflight.caption_counts_uniform(a) == 10
    a.write_text(json.dumps({"rephrased_captions": ["c"] * 3}) + "\n" + json.dumps({"rephrased_captions": ["c"] * 4}) + "\n")
    assert preflight.caption_counts_uniform(a) is None
