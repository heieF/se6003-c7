"""Fig. B2: precision refinement [14, Sec. 5.1] of a noisy first layer.

Input: the true first layer (unit rows) plus i.i.d. Gaussian noise.
Output: results/b_refine.csv, figures/B/b_refine.pdf
"""

import csv

import _path
import numpy as np
from bstyle import BLUE, GRAY, ORANGE, plt

from bmetrics import layer1_error, unit_rows
from common import Oracle, random_network
from refine import refine_layer1

N_NETS, SEED = 100, 2026
NOISE = [1e-6, 1e-8, 1e-10]
DISTS = ["uniform01", "uniform_pm1", "normal"]


def main():
    rng = np.random.default_rng(SEED)
    out, fig_data = [], {}
    for dist in DISTS:
        for noise in NOISE:
            before, after, q = [], [], []
            for _ in range(N_NETS):
                p = random_network(rng, weights=dist)
                T, tb = unit_rows(p.A1, p.b1)
                A1 = T + noise * rng.normal(size=T.shape)
                b1 = tb + noise * rng.normal(size=tb.shape)
                o = Oracle(p)
                Ar, br = refine_layer1(o, A1, b1, rng)
                before.append(layer1_error(p, A1, b1))
                after.append(layer1_error(p, Ar, br))
                q.append(o.queries)
            b, a = np.log2(before), np.log2(after)
            out.append([dist, f"{noise:.0e}", f"{np.median(b):.1f}", f"{np.max(b):.1f}",
                        f"{np.median(a):.1f}", f"{np.max(a):.1f}",
                        f"{np.mean(a <= -40):.2f}", f"{np.mean(q):.0f}"])
            if dist == "uniform01" and noise == 1e-8:
                fig_data = (b, a)
    with open(f"{_path.ROOT}/results/b_refine.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["weights", "noise", "log2_before_med", "log2_before_max",
                     "log2_after_med", "log2_after_max", "frac_after_le_2^-40", "queries"])
        wr.writerows(out)

    b, a = fig_data
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for v, c, lab in [(b, ORANGE, "before refinement"), (a, BLUE, "after refinement")]:
        s = np.sort(v)
        ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=c, label=lab)
    ax.axvline(-40, color=GRAY, lw=0.8, ls="--")
    ax.text(-39.6, 0.08, "$2^{-40}$", color=GRAY, fontsize=7)
    ax.set_xlabel("$\\log_2$ max parameter error of layer 1 (per network)")
    ax.set_ylabel("fraction of networks")
    ax.legend(loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{_path.ROOT}/figures/B/b_refine.pdf")
    print(open(f"{_path.ROOT}/results/b_refine.csv").read())


if __name__ == "__main__":
    main()
