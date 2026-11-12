"""Role B - helpers for B's tests and experiments (uses ground truth)."""

from __future__ import annotations

import numpy as np

from common import Params


def unit_rows(A1: np.ndarray, b1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(A1, axis=1)
    return A1 / n[:, None], b1 / n


def match_rows(p: Params, A1_hat: np.ndarray) -> np.ndarray:
    """Index of the true neuron each estimated row corresponds to (by |cos|)."""
    T, _ = unit_rows(p.A1, p.b1)
    E, _ = unit_rows(A1_hat, np.zeros(len(A1_hat)))
    return np.argmax(np.abs(E @ T.T), axis=1)


def layer1_error(p: Params, A1_hat: np.ndarray, b1_hat: np.ndarray) -> float:
    """max |theta - theta_hat| over unit-normalised, matched, sign-aligned rows."""
    T, tb = unit_rows(p.A1, p.b1)
    E, eb = unit_rows(A1_hat, b1_hat)
    idx = match_rows(p, A1_hat)
    s = np.sign(np.sum(E * T[idx], axis=1))
    return float(
        max(np.max(np.abs(s[:, None] * E - T[idx])), np.max(np.abs(s * eb - tb[idx])))
    )


def scrambled_layer1(
    p: Params, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True first layer with random positive scaling and random sign flips.

    Returns (A1, b1, flipped) where flipped[j] says row j has the wrong sign.
    """
    c = rng.uniform(0.1, 10, p.d1)
    flipped = rng.random(p.d1) < 0.5
    s = np.where(flipped, -1.0, 1.0) * c
    return p.A1 * s[:, None], p.b1 * s, flipped
