# Role B: recover first-layer rows from witnesses and merge them
# (Carlini et al. CRYPTO'20, Sec 4.2 / 4.3.1 / 5.2)
from __future__ import annotations

import numpy as np

from common import Oracle, Witness

DELTA_FRAC = 0.1  # gradient points at x +- delta*v, delta = 0.1 * radius
EPS_FRAC = 0.01  # finite diff step = 0.01 * delta
MERGE_COS = 1.0 - 1e-6
SAFETY = 0.1
MAX_RETRIES = 3
CONSIST_TOL = 1e-5
COORD_FRAC = 0.01  # step for the Sec 5.2 method


def gradients(oracle, pts, eps, extra=None):
    # forward differences at every row of pts, one batched query
    k, d0 = pts.shape
    X = (pts[:, None, :] + eps * np.vstack([np.zeros(d0), np.eye(d0)])[None]).reshape(-1, d0)
    if extra is not None:
        X = np.vstack([X, extra])
    y = np.asarray(oracle(X))
    base = y[: k * (d0 + 1)].reshape(k, d0 + 1)
    return (base[:, 1:] - base[:, :1]) / eps, base[:, 0], y[k * (d0 + 1) :]


def row_from_witness(oracle: Oracle, w: Witness) -> tuple[np.ndarray, float] | None:
    v, delta = w.direction, DELTA_FRAC * w.radius
    eps = EPS_FRAC * delta
    pts = np.stack([w.x + delta * v, w.x - delta * v])
    g, fb, fx = gradients(oracle, pts, eps, extra=w.x[None])
    # slope jump along v from 3 points; must match J.v unless a probe crossed
    # the neuron's own hyperplane (search line almost parallel to it)
    s_v = (fb[0] - 2 * fx[0] + fb[1]) / delta
    for _ in range(MAX_RETRIES):
        J = g[0] - g[1]
        if abs(J @ v - s_v) <= CONSIST_TOL * np.linalg.norm(J):
            break
        eps = min(0.01 * eps, SAFETY * delta * abs(s_v) / np.abs(J).max())
        g = gradients(oracle, pts, eps)[0]
    J = g[0] - g[1]
    n = np.linalg.norm(J)
    if n == 0:  # A2_j = 0, neuron not visible
        return None
    row = J / n
    return row, float(-row @ w.x)


def row_with_steps(oracle, w, delta, eps):
    # plain version with fixed steps, only used in experiments
    pts = np.stack([w.x + delta * w.direction, w.x - delta * w.direction])
    g = gradients(oracle, pts, eps)[0]
    J = g[0] - g[1]
    n = np.linalg.norm(J)
    return (J / n, float(-(J / n) @ w.x)) if n > 0 else None


def row_from_witness_coord(oracle, w, step_frac=COORD_FRAC):
    # Sec 5.2: step from x along +-e_i and differentiate along e_i -> |a_i| only,
    # then signs from the jump along e_r + e_k (|a_r|+|a_k| if same sign)
    x, d0 = w.x, len(w.x)
    eps = step_frac * w.radius
    E = np.eye(d0)
    y = np.asarray(oracle(np.vstack([x[None], x + eps * E, x - eps * E])))
    f0 = y[0]
    mag = ((y[1 : d0 + 1] - f0) - (f0 - y[d0 + 1 :])) / eps
    r = int(np.argmax(np.abs(mag)))
    oth = [k for k in range(d0) if k != r]
    U = E[r] + E[oth]
    y2 = np.asarray(oracle(np.vstack([x + eps * U, x - eps * U])))
    pair = ((y2[: d0 - 1] - f0) - (f0 - y2[d0 - 1 :])) / eps
    same = np.abs(pair - (mag[r] + mag[oth])) <= np.abs(pair - (mag[r] - mag[oth]))
    row = mag.copy()
    row[oth] = np.where(same, mag[oth], -mag[oth])
    row /= np.linalg.norm(row)
    return row, float(-row @ x)


def merge_neurons(
    cands: list[tuple[np.ndarray, float]], d1: int, min_count: int = 1
) -> tuple[np.ndarray, np.ndarray] | None:
    cands = [c for c in cands if c is not None]
    if len(cands) < d1:
        return None
    raw = np.array([r for r, _ in cands], dtype=np.float64)
    nrm = np.linalg.norm(raw, axis=1)
    rows, bias = raw / nrm[:, None], np.array([b for _, b in cands]) / nrm

    # greedy clustering by |cos|
    groups = []
    for i, r in enumerate(rows):
        for g in groups:
            if abs(rows[g[0]] @ r) > MERGE_COS:
                g.append(i)
                break
        else:
            groups.append([i])

    merged = []
    for g in groups:
        s = np.sign(rows[g] @ rows[g[0]])
        row = (s[:, None] * rows[g]).mean(axis=0)
        # median bias, so one bad witness doesn't matter
        merged.append((len(g), row / np.linalg.norm(row), float(np.median(s * bias[g]))))

    merged = sorted([m for m in merged if m[0] >= min_count], key=lambda m: -m[0])
    if len(merged) < d1:
        return None
    # more groups than neurons -> drop the rare (spurious) ones, unless it's a tie
    if len(merged) > d1 and merged[d1 - 1][0] == merged[d1][0]:
        return None
    return np.array([m[1] for m in merged[:d1]]), np.array([m[2] for m in merged[:d1]])
