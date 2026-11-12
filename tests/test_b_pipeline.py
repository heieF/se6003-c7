# B's modules chained the way attack.py will use them. A's search is replaced by
# a stand-in that gives witnesses on random lines with A's precision (|V| ~ 1e-10),
# and layer 2 comes from fix_signs_lstsq, so this runs without A's code.
import numpy as np

from common import Oracle, Params, Witness, max_func_err, random_network
from layer1 import merge_neurons, row_from_witness
from refine import refine_layer1
from signs import fix_signs, fix_signs_lstsq


def fake_search(p, rng):
    x0 = rng.normal(size=p.d0)
    v = rng.normal(size=p.d0)
    v /= np.linalg.norm(v)
    ts = sorted(t for t in -(p.A1 @ x0 + p.b1) / (p.A1 @ v) if abs(t) < 100)
    out = []
    for t in ts:
        gap = min([abs(t - s) for s in ts if s != t] + [1.0])
        out.append(Witness(x0 + (t + 1e-10 * rng.normal()) * v, v, gap))
    return out


def test_b_pipeline():
    for dist, d0, d1 in [("uniform01", 10, 10), ("normal", 10, 10), ("uniform01", 20, 10)]:
        for seed in range(15):
            rng = np.random.default_rng(seed)
            p = random_network(rng, d0, d1, dist)
            o = Oracle(p)
            cands, m = [], None
            for _ in range(20):
                cands += [row_from_witness(o, w) for w in fake_search(p, rng)]
                m = merge_neurons(cands, d1)
                if m is not None:
                    break
            A, b = refine_layer1(o, *fix_signs(o, *m, rng), rng)
            A, b, A2, b2 = fix_signs_lstsq(o, A, b, rng)
            assert max_func_err(p, Params(A, b, A2, b2), rng, 20_000) < 2**-20
