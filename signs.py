"""Role B - recovering the signs of the first-layer rows.

fix_signs:        pre-image method of Carlini et al. (CRYPTO 2020), Sec. 4.2.
fix_signs_lstsq:  our alternative, based on ReLU(-z) = ReLU(z) - z.
Attack code: only oracle(x), oracle.queries and oracle.input_dim are used.
"""

from __future__ import annotations

import numpy as np

from common import Oracle

TAU = 1.0  # size of the step in hidden space for the pre-image method


def fix_signs(
    oracle: Oracle, A1_hat: np.ndarray, b1_hat: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Flip rows (and biases) whose sign is wrong; 3 queries per neuron.

    For neuron j pick a hidden target h with h_j = 0 and |h_k| >= 1/2 for
    k != j, take its pre-image x, and step to x +/- TAU * P[:, j] where
    P = pinv(A1_hat). Only neuron j moves. The side on which f does not
    change is the inactive side; if that is the '+' side the row is flipped.
    Requires d1 <= d0 and A1_hat of full row rank.
    """
    A1, b1 = np.array(A1_hat, dtype=np.float64), np.array(b1_hat, dtype=np.float64)
    d1, d0 = A1.shape
    if d1 > d0:
        raise ValueError("pre-image method needs d1 <= d0")
    P = np.linalg.pinv(A1)
    X = np.empty((3 * d1, d0))
    for j in range(d1):
        h = rng.choice([-1.0, 1.0], d1) * rng.uniform(0.5, 1.0, d1)
        h[j] = 0.0
        x = P @ (h - b1)
        step = TAU * P[:, j]
        X[3 * j : 3 * j + 3] = [x, x + step, x - step]
    y = np.asarray(oracle(X)).reshape(d1, 3)
    plus, minus = np.abs(y[:, 1] - y[:, 0]), np.abs(y[:, 2] - y[:, 0])
    flip = plus < minus  # f unchanged on the '+' side -> sign is wrong
    s = np.where(flip, -1.0, 1.0)
    return A1 * s[:, None], b1 * s


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def fix_signs_lstsq(
    oracle: Oracle,
    A1_hat: np.ndarray,
    b1_hat: np.ndarray,
    rng: np.random.Generator,
    n: int = 60,
    scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Our least-squares sign test; n queries (default 60).

    Fit f(x) ~ sum_j w_j ReLU(z_j) + u.x + u0 with z = A1_hat x + b1_hat.
    Because ReLU(-z) = ReLU(z) - z the fit is exact whatever the signs, and a
    flipped set S leaves u = -sum_{j in S} w_j a_j. Solving A1_hat^T beta = -u
    gives beta_j = w_j on S and 0 elsewhere. As a by-product the output layer
    is recovered too. Returns (A1, b1, A2, b2) with corrected signs.
    """
    A1, b1 = np.array(A1_hat, dtype=np.float64), np.array(b1_hat, dtype=np.float64)
    d1, d0 = A1.shape
    if d1 > d0:
        raise ValueError("least-squares sign test needs d1 <= d0")
    X = rng.normal(scale=scale, size=(n, d0))
    Z = X @ A1.T + b1
    M = np.hstack([_relu(Z), X, np.ones((n, 1))])
    if np.linalg.matrix_rank(M) < d1 + d0 + 1:
        raise np.linalg.LinAlgError("design matrix is rank deficient; use larger n")
    sol = np.linalg.lstsq(M, np.asarray(oracle(X)), rcond=None)[0]
    w, u, u0 = sol[:d1], sol[d1 : d1 + d0], sol[-1]
    beta = np.linalg.lstsq(A1.T, -u, rcond=None)[0]
    flip = np.abs(beta - w) < np.abs(beta)  # beta_j ~ w_j (flipped) vs ~ 0
    s = np.where(flip, -1.0, 1.0)
    b2 = float(u0 + np.sum(w[flip] * b1[flip]))
    return A1 * s[:, None], b1 * s, w, b2
