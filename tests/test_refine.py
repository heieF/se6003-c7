"""Tests for refine.py (role B, bonus)."""

import numpy as np

from bmetrics import layer1_error, unit_rows
from common import Oracle, random_network
from refine import refine_layer1


def test_refine_improves_noisy_layer():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        p = random_network(rng)
        T, tb = unit_rows(p.A1, p.b1)
        A1 = T + 1e-8 * rng.normal(size=T.shape)
        b1 = tb + 1e-8 * rng.normal(size=tb.shape)
        Ar, br = refine_layer1(Oracle(p), A1, b1, rng)
        assert layer1_error(p, Ar, br) < 2.0**-36
