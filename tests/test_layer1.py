# Tests for layer1.py (role B). Fixed seeds; ground truth from common.py.
import numpy as np

from common import Oracle, Witness, random_network, true_witness
from layer1 import merge_neurons, row_from_witness, row_from_witness_coord


def test_rows_from_true_witnesses():
    # All neurons of 100 u01 networks: |cos(row, A1_j)| >= 1 - 1e-10.
    for seed in range(100):
        rng = np.random.default_rng(seed)
        p = random_network(rng)
        oracle = Oracle(p)
        for j in range(p.d1):
            w = true_witness(p, j, rng)
            q0 = oracle.queries
            row, bias = row_from_witness(oracle, w)
            assert oracle.queries - q0 == 2 * (p.d0 + 1) + 1
            a = p.A1[j] / np.linalg.norm(p.A1[j])
            cos = row @ a
            assert abs(cos) >= 1 - 1e-10
            assert abs(np.sign(cos) * bias - p.b1[j] / np.linalg.norm(p.A1[j])) < 1e-6


def test_coordinate_method_sec52():
    # Bonus method of [14, Sec. 5.2]: magnitudes plus column signs.
    for seed in range(50):
        rng = np.random.default_rng(seed)
        p = random_network(rng, weights="uniform_pm1")  # mixed signs in each row
        oracle = Oracle(p)
        for j in range(p.d1):
            q0 = oracle.queries
            row, _ = row_from_witness_coord(oracle, true_witness(p, j, rng))
            assert oracle.queries - q0 == 1 + 2 * p.d0 + 2 * (p.d0 - 1)
            a = p.A1[j] / np.linalg.norm(p.A1[j])
            assert abs(row @ a) >= 1 - 1e-10


def _search_like_witness(p, j, rng, max_angle=None):
    # Witness as a line search returns it: random direction v (possibly almost parallel to the
    # hyperplane), radius = distance to the next kink.
    a = p.A1[j]
    x0 = rng.normal(size=p.d0)
    x = x0 - (a @ x0 + p.b1[j]) / (a @ a) * a
    v = rng.normal(size=p.d0)
    if max_angle is not None:  # force |cos(a, v)| <= max_angle
        u = a / np.linalg.norm(a)
        v -= (v @ u) * u
        v = v / np.linalg.norm(v) + max_angle * rng.uniform(-1, 1) * u
    v /= np.linalg.norm(v)
    V = p.A1 @ x + p.b1
    r = min(abs(V[k]) / max(abs(p.A1[k] @ v), 1e-300) for k in range(p.d1) if k != j)
    return Witness(x, v, min(r, 1.0))


def test_rows_from_nearly_parallel_lines():
    # Search lines almost parallel to the hyperplane (|cos| <= 1e-3): the consistency check must
    # detect probes crossing the hyperplane.
    for seed in range(100):
        rng = np.random.default_rng(seed)
        p = random_network(rng)
        oracle = Oracle(p)
        for j in range(p.d1):
            w = _search_like_witness(p, j, rng, max_angle=1e-3)
            row, _ = row_from_witness(oracle, w)
            a = p.A1[j] / np.linalg.norm(p.A1[j])
            assert abs(row @ a) >= 1 - 1e-10


def _true_cands(p, rng, reps):
    T = p.A1 / np.linalg.norm(p.A1, axis=1, keepdims=True)
    tb = p.b1 / np.linalg.norm(p.A1, axis=1)
    cands = []
    for j in range(p.d1):
        for _ in range(reps):
            s = rng.choice([-1.0, 1.0])
            c = rng.uniform(0.5, 2.0)  # rows need not arrive normalised
            noise = 1e-12 * rng.normal(size=p.d0)
            cands.append((s * c * (T[j] + noise), s * c * tb[j]))
    return cands, T


def _check(p, A, b, T):
    assert A.shape == (p.d1, p.d0) and b.shape == (p.d1,)
    cos = np.abs(A @ T.T)
    assert np.all(np.max(cos, axis=1) > 1 - 1e-10)
    assert sorted(np.argmax(cos, axis=1)) == list(range(p.d1))


def test_merge_duplicates():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        p = random_network(rng)
        cands, T = _true_cands(p, rng, 3)
        rng.shuffle(cands)
        _check(p, *merge_neurons(cands, p.d1), T)


def test_merge_with_spurious_rows():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        p = random_network(rng)
        cands, T = _true_cands(p, rng, 2)
        for _ in range(5):  # rows produced by two kinks mistaken for one
            r = rng.normal(size=p.d0)
            cands.append((r / np.linalg.norm(r), rng.normal()))
        rng.shuffle(cands)
        _check(p, *merge_neurons(cands, p.d1), T)
        _check(p, *merge_neurons(cands, p.d1, min_count=2), T)


def test_merge_too_few_returns_none():
    rng = np.random.default_rng(0)
    p = random_network(rng)
    cands, _ = _true_cands(p, rng, 1)
    assert merge_neurons(cands[:-1], p.d1) is None
    assert merge_neurons(cands, p.d1, min_count=2) is None
