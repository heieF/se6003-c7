"""Role B - first-layer recovery: rows from witnesses, and merging candidates.

Follows Carlini et al. (CRYPTO 2020), Sec. 4.2 ("Extracting the rows of A(1)
up to sign") and Sec. 4.3.1 / 5.4 (deduplicating candidate rows).
Attack code: only oracle(x), oracle.queries and oracle.input_dim are used.
"""

from __future__ import annotations

import numpy as np

from common import Oracle, Witness

# Relative size of the two step lengths w.r.t. the witness radius.
DELTA_FRAC = 0.1  # distance from the witness to the two gradient points
EPS_FRAC = 0.01  # finite-difference step, relative to delta
# Two candidate rows are the same neuron if |cos| exceeds this threshold.
MERGE_COS = 1.0 - 1e-6
# A jump smaller than this (relative to the gradient size) means "no kink".
MIN_REL_JUMP = 1e-9


def gradients(oracle: Oracle, points: np.ndarray, eps: float) -> np.ndarray:
    """Forward-difference gradients at each row of `points` (k, d0).

    Uses k * (d0 + 1) queries, sent to the oracle as one batch.
    """
    k, d0 = points.shape
    probes = points[:, None, :] + eps * np.vstack([np.zeros(d0), np.eye(d0)])[None]
    y = np.asarray(oracle(probes.reshape(-1, d0))).reshape(k, d0 + 1)
    return (y[:, 1:] - y[:, :1]) / eps


def row_jump(
    oracle: Oracle, w: Witness, delta: float, eps: float
) -> np.ndarray:
    """grad f(x + delta v) - grad f(x - delta v)  (2 (d0 + 1) queries)."""
    pts = np.stack([w.x + delta * w.direction, w.x - delta * w.direction])
    g = gradients(oracle, pts, eps)
    return g[0] - g[1]


def row_from_witness(oracle: Oracle, w: Witness) -> tuple[np.ndarray, float] | None:
    """Unit normal of the critical hyperplane through w.x, and its bias.

    The sign of the row is NOT determined (see signs.py). Returns None when no
    gradient jump is visible at w (e.g. the neuron has zero output weight).
    """
    delta = DELTA_FRAC * w.radius
    return row_with_steps(oracle, w, delta, EPS_FRAC * delta)


def row_with_steps(
    oracle: Oracle, w: Witness, delta: float, eps: float
) -> tuple[np.ndarray, float] | None:
    """row_from_witness with explicit step sizes (used by experiments)."""
    jump = row_jump(oracle, w, delta, eps)
    norm = float(np.linalg.norm(jump))
    if not np.isfinite(norm) or norm == 0.0:
        return None
    row = jump / norm
    return row, float(-row @ w.x)


def _cluster(rows: np.ndarray, thr: float) -> list[list[int]]:
    """Greedy clustering of unit vectors by |cos| > thr (order-independent
    for well-separated neurons, which is the case we care about)."""
    clusters: list[list[int]] = []
    reps: list[np.ndarray] = []
    for i, r in enumerate(rows):
        for c, rep in zip(clusters, reps):
            if abs(float(rep @ r)) > thr:
                c.append(i)
                break
        else:
            clusters.append([i])
            reps.append(r)
    return clusters


def merge_neurons(
    cands: list[tuple[np.ndarray, float]], d1: int, min_count: int = 1
) -> tuple[np.ndarray, np.ndarray] | None:
    """Merge candidate (row, bias) pairs into d1 distinct neurons.

    Candidates with |cos| > MERGE_COS are the same neuron: their signs are
    aligned to the first member, rows are averaged and re-normalised, and the
    bias is the median of the aligned biases (robust to a bad witness).

    If more than d1 groups are found, groups seen fewer than `min_count`
    times (and, failing that, the rarest groups) are treated as spurious; if
    the d1 most frequent groups cannot be told apart from the rest, or there
    are fewer than d1 groups, None is returned so the caller searches more.
    """
    cands = [c for c in cands if c is not None]
    if len(cands) < d1:
        return None
    rows = np.array([np.asarray(r, dtype=np.float64) for r, _ in cands])
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    biases = np.array([float(b) for _, b in cands])
    rows_b = biases / np.linalg.norm([np.asarray(r) for r, _ in cands], axis=1)

    groups = []
    for idx in _cluster(rows, MERGE_COS):
        s = np.sign(rows[idx] @ rows[idx[0]])
        row = (s[:, None] * rows[idx]).mean(axis=0)
        row /= np.linalg.norm(row)
        groups.append((len(idx), row, float(np.median(s * rows_b[idx]))))

    groups = [g for g in groups if g[0] >= min_count]
    if len(groups) < d1:
        return None
    groups.sort(key=lambda g: -g[0])
    if len(groups) > d1 and groups[d1 - 1][0] == groups[d1][0]:
        return None  # cannot tell the real neurons from spurious rows
    keep = groups[:d1]
    return np.array([g[1] for g in keep]), np.array([g[2] for g in keep])
