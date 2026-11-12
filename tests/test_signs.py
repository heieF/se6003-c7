# Tests for signs.py (role B). Fixed seeds; ground truth from common.py.
import numpy as np

from bmetrics import scrambled_layer1
from common import Oracle, forward, Params, random_network
from signs import fix_signs, fix_signs_lstsq


def _all_signs_right(p, A1):
    return np.all(np.sum(A1 * p.A1, axis=1) > 0)


def test_preimage_method_100_networks():
    # 100 u01 networks x 10 rows, random positive scaling and sign flips.
    for seed in range(100):
        rng = np.random.default_rng(seed)
        p = random_network(rng)
        oracle = Oracle(p)
        A1, b1, _ = scrambled_layer1(p, rng)
        A1f, b1f = fix_signs(oracle, A1, b1, rng)
        assert oracle.queries == 3 * p.d1
        assert _all_signs_right(p, A1f)
        assert np.allclose(np.abs(b1f), np.abs(b1))


def test_preimage_method_signed_weights():
    for weights in ("uniform_pm1", "normal"):
        for seed in range(50):
            rng = np.random.default_rng(seed)
            p = random_network(rng, weights=weights)
            A1, b1, _ = scrambled_layer1(p, rng)
            assert _all_signs_right(p, fix_signs(Oracle(p), A1, b1, rng)[0])


def test_lstsq_method_and_layer2_by_product():
    for weights in ("uniform01", "uniform_pm1", "normal"):
        for seed in range(50):
            rng = np.random.default_rng(seed)
            p = random_network(rng, weights=weights)
            A1, b1, _ = scrambled_layer1(p, rng)
            A1f, b1f, A2, b2 = fix_signs_lstsq(Oracle(p), A1, b1, rng)
            assert _all_signs_right(p, A1f)
            X = rng.normal(scale=5, size=(1000, p.d0))
            err = np.abs(forward(Params(A1f, b1f, A2, b2), X) - forward(p, X))
            assert err.max() < 1e-9
