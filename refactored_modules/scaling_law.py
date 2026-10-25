"""Test-time scaling law (Eq. 2): score(nf, nc) = S_inf - (C_f * nf^-alpha + C_c * nc^-beta)."""
from typing import Dict, Sequence

import numpy as np
from scipy.optimize import curve_fit

PARAM_NAMES = ("S_inf", "C_f", "C_c", "alpha", "beta")


def saturation_model(X, s_inf, c_f, c_c, alpha, beta):
    nf, nc = X
    return s_inf - (c_f * np.power(nf, -alpha) + c_c * np.power(nc, -beta))


def fit_scaling_law(n_frames: Sequence[float], n_captions: Sequence[float], scores: Sequence[float]) -> Dict:
    """Fit Eq. 2 to measured alignment scores; returns parameters and R^2.

    Needs >= 3 distinct frame counts and >= 3 distinct caption counts, otherwise the
    five parameters are not identifiable. Several starting points are tried because the
    exponents make the objective non-convex.
    """
    nf, nc, y = (np.asarray(v, dtype=float) for v in (n_frames, n_captions, scores))
    if not (nf.shape == nc.shape == y.shape):
        raise ValueError("n_frames, n_captions and scores must have the same length")
    if len(set(nf)) < 3 or len(set(nc)) < 3:
        raise ValueError("Need at least 3 distinct frame counts and 3 distinct caption counts")
    lower = [0.0, 0.0, 0.0, 0.05, 0.05]
    upper = [1.5, 5.0, 5.0, 5.0, 5.0]
    best, best_sse = None, np.inf
    for alpha0 in (0.5, 1.0, 2.0):
        for beta0 in (0.5, 1.0, 2.0):
            p0 = [float(y.max()) + 0.05, 0.1, 0.1, alpha0, beta0]
            try:
                popt, _ = curve_fit(saturation_model, (nf, nc), y, p0=p0, bounds=(lower, upper), maxfev=20000)
            except RuntimeError:
                continue
            sse = float(np.sum((y - saturation_model((nf, nc), *popt)) ** 2))
            if sse < best_sse:
                best, best_sse = popt, sse
    if best is None:
        raise RuntimeError("Scaling-law fit did not converge")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - best_sse / ss_tot if ss_tot > 0 else float("nan")
    return {**dict(zip(PARAM_NAMES, map(float, best))), "r2": r2, "n_points": int(len(y))}


def predict(fit: Dict, n_frames, n_captions):
    return saturation_model(
        (np.asarray(n_frames, float), np.asarray(n_captions, float)),
        fit["S_inf"], fit["C_f"], fit["C_c"], fit["alpha"], fit["beta"],
    )
