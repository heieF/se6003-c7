"""Shared base code for SE6003 project C7 (FROZEN - do not modify after 11/10).

Network definition, black-box oracle with query counting, and ground-truth
helpers. Attack code may only use Oracle.__call__, Oracle.queries and
Oracle.input_dim. Ground-truth helpers are for tests and experiments only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    """f(x) = A2 . ReLU(A1 x + b1) + b2 (one hidden layer, scalar output)."""

    A1: np.ndarray  # (d1, d0)
    b1: np.ndarray  # (d1,)
    A2: np.ndarray  # (d1,)
    b2: float

    @property
    def d0(self) -> int:
        return self.A1.shape[1]

    @property
    def d1(self) -> int:
        return self.A1.shape[0]


def forward(p: Params, X: np.ndarray) -> float | np.ndarray:
    """X of shape (d0,) -> float; X of shape (n, d0) -> array of shape (n,)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.maximum(np.atleast_2d(X) @ p.A1.T + p.b1, 0.0) @ p.A2 + p.b2
    return float(y[0]) if X.ndim == 1 else y


def random_network(
    rng: np.random.Generator, d0: int = 10, d1: int = 10, weights: str = "uniform01"
) -> Params:
    """weights: 'uniform01' (task setting), 'uniform_pm1' or 'normal'. Biases ~ U[-1, 1]."""
    if weights == "uniform01":
        A1, A2 = rng.uniform(0, 1, (d1, d0)), rng.uniform(0, 1, d1)
    elif weights == "uniform_pm1":
        A1, A2 = rng.uniform(-1, 1, (d1, d0)), rng.uniform(-1, 1, d1)
    elif weights == "normal":
        A1, A2 = rng.normal(size=(d1, d0)), rng.normal(size=d1)
    else:
        raise ValueError(weights)
    return Params(A1, rng.uniform(-1, 1, d1), A2, float(rng.uniform(-1, 1)))


class Oracle:
    """Black-box access to a secret network; every input point costs one query."""

    def __init__(self, params: Params) -> None:
        self._secret = params
        self._queries = 0

    def __call__(self, X: np.ndarray) -> float | np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        self._queries += 1 if X.ndim == 1 else X.shape[0]
        return forward(self._secret, X)

    @property
    def queries(self) -> int:
        return self._queries

    @property
    def input_dim(self) -> int:
        return self._secret.d0


@dataclass(frozen=True)
class Witness:
    """A point x where exactly one hidden neuron is at its critical point."""

    x: np.ndarray  # (d0,)
    direction: np.ndarray  # (d0,) unit vector of the search line through x
    radius: float  # no other critical point on the line within +-radius


# ---------------- ground-truth helpers: tests / experiments ONLY ----------------


def true_witness(p: Params, j: int, rng: np.random.Generator, max_radius: float = 1.0) -> Witness:
    """Exact witness for neuron j, built from the true parameters."""
    a, V = p.A1[j], lambda z: p.A1 @ z + p.b1
    x0 = rng.normal(size=p.d0)
    x = x0 - (a @ x0 + p.b1[j]) / (a @ a) * a
    while True:
        v = rng.normal(size=p.d0)
        v /= np.linalg.norm(v)
        if abs(a @ v) >= 0.3 * np.linalg.norm(a):
            break
    others = [abs(V(x)[k]) / max(abs(p.A1[k] @ v), 1e-12) for k in range(p.d1) if k != j]
    return Witness(x, v, min(0.5 * min(others), max_radius))


def equivalent_params(p: Params, rng: np.random.Generator) -> Params:
    """Same function: random neuron permutation + positive per-neuron scaling."""
    perm, c = rng.permutation(p.d1), rng.uniform(0.1, 10, p.d1)
    return Params((p.A1 * c[:, None])[perm], (p.b1 * c)[perm], (p.A2 / c)[perm], p.b2)


def max_func_err(p: Params, q: Params, rng: np.random.Generator, n: int = 100_000) -> float:
    """max |f_p(x) - f_q(x)| over n uniform samples from the ball ||x||_2 <= d0."""
    d = p.d0
    U = rng.normal(size=(n, d))
    U *= (d * rng.uniform(size=(n, 1)) ** (1 / d)) / np.linalg.norm(U, axis=1, keepdims=True)
    return float(np.max(np.abs(forward(p, U) - forward(q, U))))
