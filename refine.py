"""Role B (bonus) - precision refinement of the first layer.

Follows the idea of Carlini et al. (CRYPTO 2020), Sec. 5.1: start from points
on the *estimated* hyperplane of neuron j, snap each one to the *true*
critical point with a short single-kink line search, then fit a hyperplane
through the snapped points (SVD, with one round of outlier removal).
The single-kink search below is our own and does not use search.py.
Attack code: only oracle(x), oracle.queries and oracle.input_dim are used.
"""

from __future__ import annotations

import numpy as np

from common import Oracle

SEG_MAX = 1e-2  # half-length of the short search segment
MIN_ANGLE = 0.3  # require |a_j . u| >= MIN_ANGLE for the segment direction
PROBE = 0.25  # slope probes at PROBE * half-length from each end
LINE_TOL = 1e-9  # relative tolerance of the "on the line" check
MAX_SHIFT = 1e-4  # reject a refined row that moved more than this
N_TRIES = 40  # candidate starting points tried per neuron


def _kink_on_segment(
    oracle: Oracle, x: np.ndarray, u: np.ndarray, tau: float
) -> np.ndarray | None:
    """Locate the single kink of g(t) = f(x + t u) on [-tau, tau].

    Fits one line through the two left probes and one through the two right
    probes and intersects them (6 queries including the two mid-checks).
    Returns None if the segment does not look like exactly one kink.
    """
    h = PROBE * tau
    ts = np.array([-tau, -tau + h, tau - h, tau])
    g = np.asarray(oracle(x + ts[:, None] * u))
    sl, sr = (g[1] - g[0]) / h, (g[3] - g[2]) / h
    scale = max(np.max(np.abs(g)), 1.0)
    if abs(sl - sr) * tau < 1e3 * np.finfo(float).eps * scale:
        return None  # no visible kink
    t = (g[2] - sr * ts[2] - g[1] + sl * ts[1]) / (sl - sr)
    if not -tau + h < t < tau - h:
        return None
    # both halves must be straight: check their midpoints lie on the lines
    tm = np.array([(t - tau) / 2, (t + tau) / 2])
    gm = np.asarray(oracle(x + tm[:, None] * u))
    pred = np.array([g[1] + sl * (tm[0] - ts[1]), g[2] + sr * (tm[1] - ts[2])])
    if np.max(np.abs(gm - pred)) > LINE_TOL * scale:
        return None
    return x + t * u


def _fit_plane(Xs: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Unit normal a and bias b minimising sum (a.x + b)^2 (total least squares)."""
    mu = Xs.mean(axis=0)
    a = np.linalg.svd(Xs - mu)[2][-1]
    b = float(-a @ mu)
    return a, b, np.abs(Xs @ a + b)


def refine_neuron(
    oracle: Oracle,
    A1: np.ndarray,
    b1: np.ndarray,
    j: int,
    rng: np.random.Generator,
    n_points: int | None = None,
) -> tuple[np.ndarray, float] | None:
    """Refined (unit row, bias) of neuron j, or None if refinement failed."""
    d1, d0 = A1.shape
    n_points = n_points or 2 * d0
    a, b = A1[j], b1[j]
    pts = []
    for _ in range(N_TRIES):
        if len(pts) == n_points:
            break
        x = rng.normal(size=d0)
        x -= (a @ x + b) * a  # project onto the estimated hyperplane (|a| = 1)
        u = rng.normal(size=d0)
        u /= np.linalg.norm(u)
        if abs(a @ u) < MIN_ANGLE:
            u = u + np.sign(a @ u or 1.0) * a
            u /= np.linalg.norm(u)
        V, s = A1 @ x + b1, np.abs(A1 @ u)
        others = [abs(V[k]) / max(s[k], 1e-12) for k in range(d1) if k != j]
        tau = min(SEG_MAX, 0.5 * min(others)) if others else SEG_MAX
        xs = _kink_on_segment(oracle, x, u, tau)
        if xs is not None:
            pts.append(xs)
    if len(pts) < d0 + 1:
        return None
    Xs = np.array(pts)
    a_new, b_new, res = _fit_plane(Xs)
    keep = res <= 10 * np.median(res) + 1e-15
    if keep.sum() >= d0 + 1 and not keep.all():
        a_new, b_new, _ = _fit_plane(Xs[keep])
    s = 1.0 if a_new @ a >= 0 else -1.0
    a_new, b_new = s * a_new, s * b_new
    if np.linalg.norm(a_new - a) > MAX_SHIFT:
        return None
    return a_new, b_new


def refine_layer1(
    oracle: Oracle, A1_hat: np.ndarray, b1_hat: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Refine every row; rows are returned with unit norm and the same sign.

    A row whose refinement fails keeps its (normalised) original value.
    """
    A1 = np.array(A1_hat, dtype=np.float64)
    b1 = np.array(b1_hat, dtype=np.float64)
    n = np.linalg.norm(A1, axis=1)
    A1, b1 = A1 / n[:, None], b1 / n
    A_out, b_out = A1.copy(), b1.copy()
    for j in range(A1.shape[0]):
        r = refine_neuron(oracle, A1, b1, j, rng)
        if r is not None:
            A_out[j], b_out[j] = r
    return A_out, b_out
