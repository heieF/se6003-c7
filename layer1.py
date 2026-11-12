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
# Probes must stay this fraction of the way from the witness's own hyperplane.
SAFETY = 0.1
MAX_RETRIES = 3
# Allowed |J.v - s_v| relative to |J| (rounding is ~1e-8 at worst).
CONSIST_TOL = 1e-5
# Step of the Sec. 5.2 coordinate method, relative to the witness radius.
COORD_FRAC = 0.01


def gradients(
    oracle: Oracle, points: np.ndarray, eps: float, extra: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-difference gradients at each row of `points` (k, d0).

    Uses k * (d0 + 1) queries (plus len(extra)), sent as one batch.
    Returns (gradients (k, d0), f(points) (k,), f(extra)).
    """
    k, d0 = points.shape
    probes = points[:, None, :] + eps * np.vstack([np.zeros(d0), np.eye(d0)])[None]
    X = probes.reshape(-1, d0)
    if extra is not None:
        X = np.vstack([X, extra])
    y = np.asarray(oracle(X))
    base = y[: k * (d0 + 1)].reshape(k, d0 + 1)
    return (base[:, 1:] - base[:, :1]) / eps, base[:, 0], y[k * (d0 + 1) :]


def row_jump(oracle: Oracle, w: Witness, delta: float, eps: float) -> np.ndarray:
    """grad f(x + delta v) - grad f(x - delta v)  (2 (d0 + 1) queries)."""
    pts = np.stack([w.x + delta * w.direction, w.x - delta * w.direction])
    g = gradients(oracle, pts, eps)[0]
    return g[0] - g[1]


def row_from_witness(oracle: Oracle, w: Witness) -> tuple[np.ndarray, float] | None:
    """Unit normal of the critical hyperplane through w.x, and its bias.

    The sign of the row is NOT determined (see signs.py). Returns None when no
    gradient jump is visible at w (e.g. the neuron has zero output weight).

    Normally 2 (d0 + 1) + 1 = 23 queries. The extra query f(x) gives the
    slope jump along v, s_v = (f(x+dv) - 2 f(x) + f(x-dv)) / d, which must
    equal J.v for the gradient jump J. If not, a probe crossed neuron j's own
    hyperplane (the search line is almost parallel to it), and J is
    recomputed (22 more queries) with a step that is safe for that angle.
    """
    v, delta = w.direction, DELTA_FRAC * w.radius
    eps = EPS_FRAC * delta
    pts = np.stack([w.x + delta * v, w.x - delta * v])
    g, base, fx = gradients(oracle, pts, eps, extra=w.x[None])
    s_v = (base[0] - 2 * fx[0] + base[1]) / delta
    for _ in range(MAX_RETRIES):
        jump = g[0] - g[1]
        norm = float(np.linalg.norm(jump))
        if not np.isfinite(norm) or norm == 0.0:
            return None
        if abs(jump @ v - s_v) <= CONSIST_TOL * norm:
            break
        eps = min(0.01 * eps, SAFETY * delta * abs(s_v) / np.max(np.abs(jump)))
        if eps == 0.0:
            return None
        g = gradients(oracle, pts, eps)[0]
    jump = g[0] - g[1]
    norm = float(np.linalg.norm(jump))
    if not np.isfinite(norm) or norm == 0.0:
        return None
    row = jump / norm
    return row, float(-row @ w.x)


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


def row_from_witness_coord(
    oracle: Oracle, w: Witness, step_frac: float = COORD_FRAC
) -> tuple[np.ndarray, float] | None:
    """Alternative row recovery of [14, Sec. 5.2] (bonus, for comparison).

    Steps from the witness itself along +/- e_i and differentiates along the
    same axis: the slope jump of t -> f(x + t e_i) at t = 0 is
    |A2_j| * |A1_ji| up to one common sign, so only magnitudes are learned
    (1 + 2 d0 queries). Column signs relative to a reference coordinate r
    (the largest entry) come from the jump along e_r + e_k, which is
    |a_r| + |a_k| for equal signs and ||a_r| - |a_k|| otherwise
    (2 (d0 - 1) queries). Total 1 + 2 d0 + 2 (d0 - 1) = 39 queries for d0 = 10.
    """
    x, d0 = w.x, len(w.x)
    eps = step_frac * w.radius
    E = np.eye(d0)
    y = np.asarray(oracle(np.vstack([x[None], x + eps * E, x - eps * E])))
    f0, fp, fm = y[0], y[1 : d0 + 1], y[d0 + 1 :]
    mag = ((fp - f0) - (f0 - fm)) / eps  # = c |a_i| with one unknown c != 0
    if not np.all(np.isfinite(mag)) or np.max(np.abs(mag)) == 0.0:
        return None
    r = int(np.argmax(np.abs(mag)))
    others = [k for k in range(d0) if k != r]
    U = E[r] + E[others]  # directions e_r + e_k
    y2 = np.asarray(oracle(np.vstack([x + eps * U, x - eps * U])))
    pair = ((y2[: d0 - 1] - f0) - (f0 - y2[d0 - 1 :])) / eps
    same = np.abs(pair - (mag[r] + mag[others])) <= np.abs(pair - (mag[r] - mag[others]))
    row = mag.copy()
    row[others] = np.where(same, mag[others], -mag[others])
    row /= np.linalg.norm(row)
    return row, float(-row @ x)


def _cluster(rows: np.ndarray, thr: float) -> list[list[int]]:
    """Greedy clustering of unit vectors by |cos| > thr (order-independent
    for well-separated neurons, which is the case we care about)."""
    clusters: list[list[int]] = []
    reps: list[np.ndarray] = []
    for i, r in enumerate(rows):
        for c, rep in zip(clusters, reps, strict=True):
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
