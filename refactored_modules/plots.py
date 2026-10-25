"""Figures for the alignment results. Saves to disk and closes the figure (no duplicate inline display).

Does not force a matplotlib backend, so notebook cells that call plt.show() still display inline on Colab;
headless runs (tests) set MPLBACKEND=Agg.
"""
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .scaling_law import predict


def _finish(fig, path: Optional[Path]):
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def plot_layer_heatmap(scores: np.ndarray, best: Dict, path: Optional[Path] = None):
    """Layer x layer alignment (row/col 0 = all layers concatenated)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(scores, origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="mutual k-NN")
    ax.plot(best["layer_y"] + 1, best["layer_x"] + 1, "r*", markersize=14)
    ax.set_xlabel("text layer (0 = concat)")
    ax.set_ylabel("vision layer (0 = concat)")
    ax.set_title(f"Best {best['score']:.3f} (N={best['n']}, k={best['k']})")
    return _finish(fig, path)


def plot_sweeps(results: Sequence[Dict], path: Optional[Path] = None):
    """Score vs number of frames (left) and captions (right), one line per model, at the max of the other axis."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in results:
        g = r["grid"]
        max_nc, max_nf = max(c["n_captions"] for c in g), max(c["n_frames"] for c in g)
        fr = sorted((c["n_frames"], c["score"]) for c in g if c["n_captions"] == max_nc)
        cp = sorted((c["n_captions"], c["score"]) for c in g if c["n_frames"] == max_nf)
        a1.plot(*zip(*fr), "o-", label=r["vision"])
        a2.plot(*zip(*cp), "o-", label=r["vision"])
    chance = results[0]["chance_level"]
    for ax, xl in ((a1, "number of frames"), (a2, "number of captions")):
        ax.axhline(chance, color="gray", ls=":", label="chance")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xl)
        ax.set_ylabel("mutual k-NN alignment")
        ax.grid(alpha=0.3)
    a1.legend()
    fig.suptitle(f"Test-time scaling (N={results[0]['n']}, k={results[0]['k']}, text: {results[0]['llm']})")
    return _finish(fig, path)


def plot_fit(result: Dict, path: Optional[Path] = None):
    """Measured vs scaling-law predicted score."""
    fit = result.get("fit")
    if not fit:
        raise ValueError("Result has no fit; run fit_grid first")
    g = result["grid"]
    nf = np.array([c["n_frames"] for c in g])
    nc = np.array([c["n_captions"] for c in g])
    y = np.array([c["score"] for c in g])
    pred = predict(fit, nf, nc)
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.scatter(y, pred)
    lim = [min(y.min(), pred.min()) - 0.01, max(y.max(), pred.max()) + 0.01]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("measured alignment")
    ax.set_ylabel("Eq. 2 prediction")
    ax.set_title(f"{result['vision']}: R^2={fit['r2']:.3f}, S_inf={fit['S_inf']:.2f}")
    return _finish(fig, path)
