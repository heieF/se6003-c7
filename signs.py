# Role B: fix the signs of the first-layer rows
# fix_signs: pre-image method (Sec 4.2), fix_signs_lstsq: our least-squares test
from __future__ import annotations

import numpy as np

from common import Oracle

TAU = 1.0


def fix_signs(
    oracle: Oracle, A1_hat: np.ndarray, b1_hat: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    A1, b1 = np.array(A1_hat, dtype=np.float64), np.array(b1_hat, dtype=np.float64)
    d1, d0 = A1.shape
    if d1 > d0:
        raise ValueError("pre-image method needs d1 <= d0")
    P = np.linalg.pinv(A1)
    X = []
    for j in range(d1):
        # hidden target with h_j = 0 and the others away from 0
        h = rng.choice([-1.0, 1.0], d1) * rng.uniform(0.5, 1.0, d1)
        h[j] = 0.0
        x = P @ (h - b1)
        X += [x, x + TAU * P[:, j], x - TAU * P[:, j]]  # only neuron j moves
    y = np.asarray(oracle(np.array(X))).reshape(d1, 3)
    # f unchanged on the '+' side -> row points the wrong way
    s = np.where(np.abs(y[:, 1] - y[:, 0]) < np.abs(y[:, 2] - y[:, 0]), -1.0, 1.0)
    return A1 * s[:, None], b1 * s


def fix_signs_lstsq(oracle, A1_hat, b1_hat, rng, n=60, scale=2.0):
    # ReLU(-z) = ReLU(z) - z, so f = sum w_j ReLU(z_j) + u.x + u0 holds for any
    # signs, and flipped rows leave u = -sum_{j in S} w_j a_j.
    # Solve A1^T beta = -u: beta_j = w_j if flipped else 0. Also gives layer 2.
    A1, b1 = np.array(A1_hat, dtype=np.float64), np.array(b1_hat, dtype=np.float64)
    d1, d0 = A1.shape
    if d1 > d0:
        raise ValueError("needs d1 <= d0")
    X = rng.normal(scale=scale, size=(n, d0))
    M = np.hstack([np.maximum(X @ A1.T + b1, 0), X, np.ones((n, 1))])
    if np.linalg.matrix_rank(M) < d1 + d0 + 1:
        raise np.linalg.LinAlgError("rank deficient, use larger n")
    sol = np.linalg.lstsq(M, np.asarray(oracle(X)), rcond=None)[0]
    w, u, u0 = sol[:d1], sol[d1 : d1 + d0], sol[-1]
    beta = np.linalg.lstsq(A1.T, -u, rcond=None)[0]
    flip = np.abs(beta - w) < np.abs(beta)
    s = np.where(flip, -1.0, 1.0)
    return A1 * s[:, None], b1 * s, w, float(u0 + np.sum(w[flip] * b1[flip]))
