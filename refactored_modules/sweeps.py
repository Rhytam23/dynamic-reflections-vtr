"""Frames x captions sweeps (paper Fig. 3) on top of the authors' feature extraction."""
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .alignment import layer_sweep
from .features import extract_features, feature_path, load_features
from .scaling_law import fit_scaling_law

logger = logging.getLogger(__name__)


def extract_sweep_features(
    repo_path: Path, feature_dir: Path, llm_name: str, vision_name: str,
    frame_counts: Sequence[int], caption_counts: Sequence[int], hf_token: Optional[str] = None,
):
    """One LLM extraction per caption count and one video extraction per frame count.

    Existing files are skipped by the authors' script, so an interrupted run resumes.
    """
    for nc in caption_counts:
        extract_features(repo_path, "llm", llm_names=[llm_name], num_captions=nc,
                         feature_dir=feature_dir, hf_token=hf_token)
    for nf in frame_counts:
        extract_features(repo_path, "video", video_names=[vision_name], num_frames=nf,
                         feature_dir=feature_dir)


def compute_grid(
    feature_dir: Path, llm_name: str, vision_name: str,
    frame_counts: Sequence[int], caption_counts: Sequence[int],
    k: int = 10, dataset: str = "pvd", llm_pool: str = "avg", vision_pool: str = "cls",
    n_max: Optional[int] = None, load: Callable = load_features,
) -> Dict:
    """Best-layer-pair alignment for every (n_frames, n_captions) combination."""
    llm_feats = {nc: load(feature_path(feature_dir, dataset, llm_name, llm_pool, num_captions=nc))
                 for nc in caption_counts}
    vis_feats = {nf: load(feature_path(feature_dir, dataset, vision_name, vision_pool, num_frames=nf))
                 for nf in frame_counts}
    cells: List[Dict] = []
    n = None
    for nf, v in vis_feats.items():
        for nc, t in llm_feats.items():
            if n_max is not None:
                v, t = v[:n_max], t[:n_max]
            _, best = layer_sweep(v, t, k=k)
            n = best["n"]
            cells.append({"n_frames": nf, "n_captions": nc, "score": best["score"],
                          "vision_layer": best["layer_x"], "llm_layer": best["layer_y"]})
            logger.info("frames=%d captions=%d -> %.4f", nf, nc, best["score"])
    return {"vision": vision_name, "llm": llm_name, "n": n, "k": k,
            "chance_level": k / (n - 1), "grid": cells}


def fit_grid(result: Dict) -> Dict:
    """Attach a scaling-law fit to a grid result (skipped if the grid is too small)."""
    g = result["grid"]
    try:
        result["fit"] = fit_scaling_law([c["n_frames"] for c in g], [c["n_captions"] for c in g],
                                        [c["score"] for c in g])
    except ValueError as e:
        logger.warning("No scaling-law fit for %s: %s", result["vision"], e)
        result["fit"] = None
    return result


def save_result(result: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))


def comparison_table(results: Sequence[Dict]) -> str:
    """Markdown table comparing encoders: best score, and fitted scaling parameters."""
    rows = ["| vision model | N | k | best score | S_inf | C_f | C_c | alpha | beta | R^2 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        best = max(c["score"] for c in r["grid"])
        f = r.get("fit")
        fit_cols = [f"{f[p]:.2f}" for p in ("S_inf", "C_f", "C_c", "alpha", "beta")] + [f"{f['r2']:.3f}"] if f else ["-"] * 6
        rows.append(f"| {r['vision']} | {r['n']} | {r['k']} | {best:.3f} | " + " | ".join(fit_cols) + " |")
    return "\n".join(rows)
