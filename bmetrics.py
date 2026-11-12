# Ground-truth helpers for B's tests and experiments only
import numpy as np


def unit_rows(A1, b1):
    n = np.linalg.norm(A1, axis=1)
    return A1 / n[:, None], b1 / n


def match_rows(p, A1_hat):
    T = unit_rows(p.A1, p.b1)[0]
    E = A1_hat / np.linalg.norm(A1_hat, axis=1, keepdims=True)
    return np.argmax(np.abs(E @ T.T), axis=1)


def layer1_error(p, A1_hat, b1_hat):
    # max |theta - theta_hat| after normalising, matching and sign alignment
    T, tb = unit_rows(p.A1, p.b1)
    E, eb = unit_rows(A1_hat, b1_hat)
    idx = match_rows(p, A1_hat)
    s = np.sign(np.sum(E * T[idx], axis=1))
    return float(max(np.abs(s[:, None] * E - T[idx]).max(), np.abs(s * eb - tb[idx]).max()))


def scrambled_layer1(p, rng):
    # true layer 1 with random positive scaling and random sign flips
    c = rng.uniform(0.1, 10, p.d1)
    flipped = rng.random(p.d1) < 0.5
    s = np.where(flipped, -1.0, 1.0) * c
    return p.A1 * s[:, None], p.b1 * s, flipped
