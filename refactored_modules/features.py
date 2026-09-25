"""Feature file naming, extraction (via the authors' scripts) and loading."""
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .utils import cuda_library_env, run_streaming

logger = logging.getLogger(__name__)


def feature_path(
    feature_dir: Path,
    dataset: str,
    model: str,
    pool: str,
    num_captions: int = -1,
    num_frames: int = -1,
) -> Path:
    """Mirror of `vprh.misc.utils.to_feature_filename` for the pvd config
    (linspace sampling; caption key 'rephrased_captions' adds no suffix)."""
    name = f"{model.replace('/', '_')}_pool-{pool}"
    if num_captions > -1:
        name += f"_numcap-{num_captions}"
    if num_frames > -1:
        name += f"_numframes-{num_frames}"
    return Path(feature_dir) / dataset / f"{name}.pt"


def extract_features(
    repo_path: Path,
    which: str,
    config: str = "pvd_sample",
    llm_names: Optional[Sequence[str]] = None,
    video_names: Optional[Sequence[str]] = None,
    num_captions: int = -1,
    num_frames: int = -1,
    feature_dir: Optional[Path] = None,
    hf_token: Optional[str] = None,
):
    """Run the authors' `scripts/main_extract.py` for LLM ('llm') or video ('video') features.

    The authors' `pvd_sample` config also lists gemma2-9b-it, which does not fit a
    free Colab T4, so the model lists are always overridden explicitly here.
    Extraction skips files that already exist, so this is safe to re-run.
    """
    if which not in ("llm", "video"):
        raise ValueError("which must be 'llm' or 'video'")
    cmd = ["uv", "run", "scripts/main_extract.py", config, f"--{which}_only"]
    if which == "llm":
        if not llm_names:
            raise ValueError("llm_names is required for LLM extraction")
        cmd += ["--llm_names", *llm_names]
        if num_captions > -1:
            cmd += ["--num_captions", str(num_captions)]
    else:
        if not video_names:
            raise ValueError("video_names is required for video extraction")
        cmd += ["--video_model_names", *video_names]
        if num_frames > -1:
            cmd += ["--num_frames", str(num_frames)]
    if feature_dir is not None:
        cmd += ["--feature_dir", str(feature_dir)]
    env = cuda_library_env(repo_path)
    if hf_token:
        env["HF_TOKEN"] = hf_token
    logger.info("Running: %s", " ".join(cmd))
    run_streaming(cmd, cwd=repo_path, env=env)


def load_features(path: Path) -> np.ndarray:
    """Load an authors' .pt file and return its 'feats' as float32 [N, layers, dim]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}. Run extraction first.")
    import torch  # heavy import, only needed on Colab
    data = torch.load(path, map_location="cpu", weights_only=False)
    feats = data["feats"].float().numpy()
    if feats.ndim == 2:  # single-layer models
        feats = feats[:, None, :]
    return feats
