# Role B (bonus): precision refinement of layer 1, idea from Sec 5.1.
# Snap points on the estimated hyperplane to the true one with a short
# single-kink search, then refit the plane with SVD.
from __future__ import annotations

import numpy as np

from common import Oracle

SEG_MAX = 1e-2
MIN_ANGLE = 0.3
PROBE = 0.25
LINE_TOL = 1e-9
MAX_SHIFT = 1e-4
N_TRIES = 40


def find_kink(oracle, x, u, tau):
    # g(t) = f(x + t u) has one kink on [-tau, tau]: intersect the left and
    # right lines (2 probes each), then check both half-midpoints are on them
    h = PROBE * tau
    ts = np.array([-tau, -tau + h, tau - h, tau])
    g = np.asarray(oracle(x + ts[:, None] * u))
    sl, sr = (g[1] - g[0]) / h, (g[3] - g[2]) / h
    sc = max(np.abs(g).max(), 1.0)
    if abs(sl - sr) * tau < 1e3 * np.finfo(float).eps * sc:
        return None
    t = (g[2] - sr * ts[2] - g[1] + sl * ts[1]) / (sl - sr)
    if not -tau + h < t < tau - h:
        return None
    tm = np.array([(t - tau) / 2, (t + tau) / 2])
    gm = np.asarray(oracle(x + tm[:, None] * u))
    pred = [g[1] + sl * (tm[0] - ts[1]), g[2] + sr * (tm[1] - ts[2])]
    return x + t * u if np.abs(gm - pred).max() <= LINE_TOL * sc else None


def fit_plane(X):
    mu = X.mean(axis=0)
    a = np.linalg.svd(X - mu)[2][-1]
    return a, float(-a @ mu), np.abs(X @ a - a @ mu)


def refine_neuron(oracle, A1, b1, j, rng, n_pts=None):
    d1, d0 = A1.shape
    n_pts = n_pts or 2 * d0
    a, b = A1[j], b1[j]
    pts = []
    for _ in range(N_TRIES):
        if len(pts) == n_pts:
            break
        x = rng.normal(size=d0)
        x -= (a @ x + b) * a  # onto the estimated plane (|a| = 1)
        u = rng.normal(size=d0)
        u /= np.linalg.norm(u)
        if abs(a @ u) < MIN_ANGLE:
            u = u + np.sign(a @ u or 1.0) * a
            u /= np.linalg.norm(u)
        V, s = A1 @ x + b1, np.abs(A1 @ u)
        # keep the segment away from the other neurons' planes
        tau = min(SEG_MAX, 0.5 * min(abs(V[k]) / max(s[k], 1e-12) for k in range(d1) if k != j))
        xs = find_kink(oracle, x, u, tau)
        if xs is not None:
            pts.append(xs)
    if len(pts) < d0 + 1:
        return None
    X = np.array(pts)
    a2, b2, res = fit_plane(X)
    keep = res <= 10 * np.median(res) + 1e-15
    if keep.sum() >= d0 + 1 and not keep.all():
        a2, b2, _ = fit_plane(X[keep])
    if a2 @ a < 0:
        a2, b2 = -a2, -b2
    return (a2, b2) if np.linalg.norm(a2 - a) <= MAX_SHIFT else None


def refine_layer1(
    oracle: Oracle, A1_hat: np.ndarray, b1_hat: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(A1_hat, axis=1)
    A1, b1 = A1_hat / n[:, None], b1_hat / n
    A, b = A1.copy(), b1.copy()
    for j in range(len(A1)):
        r = refine_neuron(oracle, A1, b1, j, rng)
        if r is not None:  # failed rows keep the old value
            A[j], b[j] = r
    return A, b
