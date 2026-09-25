import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .alignment import layer_sweep
from .features import feature_path, load_features
from .utils import run_streaming

logger = logging.getLogger(__name__)


def download_pvd_dataset(json_path: Path, output_dir: Path, repo_path: Optional[Path] = None):
    """Download the PVD videos listed in `json_path` (run from the authors' repo root)."""
    logger.info(f"Downloading dataset to {output_dir}...")
    # plain Colab python (not uv): avoids the datasets/torchcodec conflict noted in the original notebook
    run_streaming(["pip", "install", "-q", "datasets", "tqdm"])
    run_streaming(
        ["python", "src/vprh/misc/download_pvd.py", "-l", str(json_path), "-o", str(output_dir)],
        cwd=repo_path,
    )
    logger.info("Dataset download complete.")


def run_alignment_pipeline(
    feature_dir: Path,
    llm_name: str,
    vision_name: str,
    llm_pool: str = "avg",
    vision_pool: str = "cls",
    dataset: str = "pvd",
    k: int = 10,
    n_max: Optional[int] = None,
    output_json: Optional[Path] = None,
) -> Dict:
    """Compute the real mutual k-NN alignment between saved LLM and vision features.

    `n_max` keeps only the first n videos (for quick subset runs); always read the
    score together with N, k and the chance level in the returned dict.
    """
    llm_file = feature_path(feature_dir, dataset, llm_name, llm_pool)
    vision_file = feature_path(feature_dir, dataset, vision_name, vision_pool)
    logger.info("LLM features: %s | vision features: %s", llm_file.name, vision_file.name)
    llm, vision = load_features(llm_file), load_features(vision_file)
    if llm.shape[0] != vision.shape[0]:
        raise ValueError(f"Feature count mismatch: LLM {llm.shape} vs vision {vision.shape}")
    if n_max is not None:
        llm, vision = llm[:n_max], vision[:n_max]
    scores, best = layer_sweep(vision, llm, k=k)  # rows = vision layers, cols = LLM layers
    result = {
        "llm": llm_name, "vision": vision_name,
        "n": best["n"], "k": k,
        "best_vision_layer": best["layer_x"], "best_llm_layer": best["layer_y"],
        "mutual_knn_score": best["score"],
        "chance_level": k / (best["n"] - 1),
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2))
        np.save(output_json.with_suffix(".layers.npy"), scores)
    return result
