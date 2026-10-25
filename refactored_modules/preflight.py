"""Cheap checks that run BEFORE any long or GPU-heavy step, so a problem costs minutes, not hours."""
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class PreflightError(RuntimeError):
    """A check failed; the message says what to do about it."""


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def check_gpu(min_gb: float = 14.0) -> str:
    """Name and memory of the GPU; raise if there is none or it is smaller than expected."""
    if not has_gpu():
        raise PreflightError("No GPU: Runtime > Change runtime type > T4 GPU.")
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    name, mem_mb = [x.strip() for x in out.split(",")]
    if float(mem_mb) / 1024 < min_gb:
        raise PreflightError(f"GPU {name} has {float(mem_mb) / 1024:.1f} GB, need >= {min_gb} GB (Gemma-2-2B in fp32 needs ~10 GB).")
    return f"{name}, {float(mem_mb) / 1024:.1f} GB"


def check_disk(path: Path, min_gb: float) -> float:
    free = shutil.disk_usage(path).free / 1024 ** 3
    if free < min_gb:
        raise PreflightError(f"Only {free:.0f} GB free on {path}; need >= {min_gb:.0f} GB (PyTorch env + model weights).")
    return free


def _hf_head(repo_id: str, repo_type: str, token: Optional[str]):
    """HEAD request for one file of the repo: fails for gated repos the token cannot access."""
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    files = HfApi(token=token).list_repo_files(repo_id, repo_type=repo_type)
    first = "config.json" if "config.json" in files else files[0]
    return get_hf_file_metadata(hf_hub_url(repo_id, first, repo_type=repo_type), token=token)


def check_hf_access(token: Optional[str], targets: Sequence[Tuple[str, str]] = (("google/gemma-2-2b-it", "model"),
                    ("facebook/PE-Video", "dataset")), head: Callable = _hf_head) -> List[str]:
    """Verify the token can read every (repo_id, repo_type); raise one error listing all fixes."""
    if not token:
        raise PreflightError("No Hugging Face token. Add a Colab secret named HF_TOKEN (key icon, enable notebook access).")
    problems, ok = [], []
    for repo_id, repo_type in targets:
        try:
            head(repo_id, repo_type, token)
            ok.append(repo_id)
        except Exception as e:  # map hub errors to instructions without importing huggingface_hub here
            kind = type(e).__name__
            url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
            if "Gated" in kind:
                problems.append(f"{repo_id}: access denied. Open {url} while logged in as the token's owner and accept the license; "
                                f"also check the token is a valid, unexpired Read token.")
            elif "NotFound" in kind:
                problems.append(f"{repo_id}: not found or no access with this token ({url}).")
            elif "401" in str(e) or "Unauthorized" in str(e):
                problems.append(f"{repo_id}: token rejected (invalid or expired). Create a new read token.")
            else:
                problems.append(f"{repo_id}: {kind}: {str(e)[:160]}")
    if problems:
        raise PreflightError("\n".join(problems))
    return ok


def make_smoke_annotation(src: Path, dst: Path, n: int = 8) -> List[str]:
    """First n lines of the annotation jsonl; returns their video ids."""
    lines = Path(src).read_text().splitlines()[:n]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("\n".join(lines) + "\n")
    return [str(json.loads(l)["video_id"]) for l in lines]


def check_videos_present(annotation: Path, data_dir: Path) -> int:
    """Every video in the annotation file exists (non-empty) in data_dir; returns the count."""
    ids = [str(json.loads(l)["video_id"]) for l in Path(annotation).read_text().splitlines() if l.strip()]
    missing = [i for i in ids if not (Path(data_dir) / f"{i}.bin").exists() or (Path(data_dir) / f"{i}.bin").stat().st_size == 0]
    if missing:
        raise PreflightError(f"{len(missing)} of {len(ids)} videos missing in {data_dir}, e.g. {missing[:3]}. Re-run the download step.")
    return len(ids)


def check_feature_file(path: Path, n: int, load: Optional[Callable] = None) -> Tuple[int, ...]:
    """Feature file exists, has n rows, and contains no NaN/inf; returns its shape."""
    if load is None:
        from .features import load_features as load
    feats = load(path)
    if feats.shape[0] != n:
        raise PreflightError(f"{Path(path).name}: {feats.shape[0]} rows, expected {n}.")
    if not np.isfinite(feats).all():
        raise PreflightError(f"{Path(path).name}: contains NaN/inf (numerical problem in the model run).")
    return tuple(feats.shape)


def caption_counts_uniform(annotation: Path, key: str = "rephrased_captions") -> Optional[int]:
    """Number of captions per video if it is the same for all videos, else None."""
    counts = {len(json.loads(l)[key]) for l in Path(annotation).read_text().splitlines() if l.strip()}
    return counts.pop() if len(counts) == 1 else None
