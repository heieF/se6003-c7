"""Fig. B1: first-layer row error vs. finite-difference step size.

Uses exact witnesses (true_witness) so that only layer1.py is exercised.
Output: results/b_rows.csv, figures/B/b_rows.pdf
"""

import csv

import _path
import numpy as np
from bstyle import BLUE, GRAY, ORANGE, plt

from common import Oracle, random_network, true_witness
from layer1 import DELTA_FRAC, row_from_witness, row_with_steps

N_NETS, SEED = 50, 2026
EPS = np.logspace(-10, -2, 17)


def row_err(p, j, r):
    a = p.A1[j] / np.linalg.norm(p.A1[j])
    if r is None:
        return np.inf
    s = np.sign(r[0] @ a) or 1.0
    return float(np.max(np.abs(s * r[0] - a)))


def main():
    rng = np.random.default_rng(SEED)
    fixed = np.zeros((len(EPS), N_NETS * 10))
    ours = np.zeros(N_NETS * 10)
    k = 0
    for _ in range(N_NETS):
        p = random_network(rng)
        oracle = Oracle(p)
        for j in range(p.d1):
            w = true_witness(p, j, rng)
            delta = DELTA_FRAC * w.radius
            for i, eps in enumerate(EPS):
                fixed[i, k] = row_err(p, j, row_with_steps(oracle, w, delta, eps))
            ours[k] = row_err(p, j, row_from_witness(oracle, w))
            k += 1

    def stats(e):
        return np.median(e), np.percentile(e, 90), np.max(e)

    with open(f"{_path.ROOT}/results/b_rows.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["eps", "median", "p90", "max"])
        for eps, e in zip(EPS, fixed):
            wr.writerow([f"{eps:.3e}", *(f"{v:.3e}" for v in stats(e))])
        wr.writerow(["relative(0.001r)", *(f"{v:.3e}" for v in stats(ours))])

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    med = np.array([stats(e) for e in fixed])
    ax.loglog(EPS, med[:, 0], "o-", color=BLUE, ms=3, label="median, fixed $\\varepsilon$")
    ax.loglog(EPS, med[:, 2], "s--", color=ORANGE, ms=3, label="max, fixed $\\varepsilon$")
    m0, _, m2 = stats(ours)
    ax.axhline(m0, color=BLUE, lw=0.8, ls=":")
    ax.axhline(m2, color=ORANGE, lw=0.8, ls=":")
    ax.text(EPS[-1], m2 * 2.0, "relative $\\varepsilon=10^{-3}r$ (max)", color=GRAY, fontsize=7, ha="right")
    ax.text(EPS[0], m0 * 0.12, "relative $\\varepsilon=10^{-3}r$ (median)", color=GRAY, fontsize=7)
    ax.set_xlabel("finite-difference step $\\varepsilon$")
    ax.set_ylabel("row error $\\|\\hat a_j-a_j\\|_\\infty$")
    ax.set_ylim(top=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{_path.ROOT}/figures/B/b_rows.pdf")
    print(open(f"{_path.ROOT}/results/b_rows.csv").read())


if __name__ == "__main__":
    main()
