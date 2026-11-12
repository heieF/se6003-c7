"""Table B1: pre-image sign recovery [14, Sec. 4.2] vs. our least-squares test.

Input: the true first layer with random positive row scaling, random sign
flips and optional relative noise on every entry.
Output: results/b_signs.csv, tables/B/b_signs.tex
"""

import csv

import _path
import numpy as np

from bmetrics import scrambled_layer1
from common import Oracle, Params, forward, random_network
from signs import fix_signs, fix_signs_lstsq

N_NETS, SEED = 100, 2026
DISTS = ["uniform01", "uniform_pm1", "normal"]
NOISE = [0.0, 1e-8, 1e-4]
conds = {d: [] for d in DISTS}


FAILS = []  # (weights, noise, method, cond(A1 with unit rows))


def run(dist, noise, rng):
    ok_pre = ok_ls = 0
    q_pre = q_ls = 0
    l2err = []
    for _ in range(N_NETS):
        p = random_network(rng, weights=dist)
        A1, b1, _ = scrambled_layer1(p, rng)
        A1 = A1 * (1 + noise * rng.normal(size=A1.shape))
        b1 = b1 * (1 + noise * rng.normal(size=b1.shape))
        o = Oracle(p)
        A, _ = fix_signs(o, A1, b1, rng)
        q_pre += o.queries
        ok_pre += np.all(np.sum(A * p.A1, axis=1) > 0)
        cond = np.linalg.cond(p.A1 / np.linalg.norm(p.A1, axis=1, keepdims=True))
        conds[dist].append(cond)
        if not np.all(np.sum(A * p.A1, axis=1) > 0):
            FAILS.append((dist, noise, "pre-image", cond))
        o = Oracle(p)
        A, b, A2, b2 = fix_signs_lstsq(o, A1, b1, rng)
        q_ls += o.queries
        ok_ls += np.all(np.sum(A * p.A1, axis=1) > 0)
        if not np.all(np.sum(A * p.A1, axis=1) > 0):
            FAILS.append((dist, noise, "lstsq", cond))
        X = rng.normal(scale=5, size=(10_000, p.d0))
        l2err.append(np.max(np.abs(forward(Params(A, b, A2, b2), X) - forward(p, X))))
    return ok_pre, q_pre / N_NETS, ok_ls, q_ls / N_NETS, float(np.median(l2err))


def main():
    rng = np.random.default_rng(SEED)
    rows = [(d, n, *run(d, n, rng)) for d in DISTS for n in NOISE]
    hdr = ["weights", "noise", "pre_ok", "pre_queries", "ls_ok", "ls_queries", "ls_med_func_err"]
    with open(f"{_path.ROOT}/results/b_signs.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(hdr)
        for r in rows:
            wr.writerow(
                [r[0], f"{r[1]:.0e}", r[2], f"{r[3]:.0f}", r[4], f"{r[5]:.0f}", f"{r[6]:.2e}"]
            )
    with open(f"{_path.ROOT}/results/b_signs_failures.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["weights", "noise", "method", "cond_A1", "median_cond_all_nets"])
        for d, n, m, c in FAILS:
            wr.writerow([d, f"{n:.0e}", m, f"{c:.3g}", f"{np.median(conds[d]):.3g}"])
    name = {"uniform01": "$U[0,1]$", "uniform_pm1": "$U[-1,1]$", "normal": "$\\mathcal N(0,1)$"}

    def lg(v):
        return "$0$" if v == 0 else f"$10^{{{int(np.log10(v))}}}$"

    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{pre-image~\\cite{DBLP:conf/crypto/CarliniJM20}}"
        " & \\multicolumn{3}{c}{least squares (ours)}\\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}",
        "weights & noise & succ. & queries & succ. & queries & med.\\ $\\max|f-\\hat f|$\\\\",
        "\\midrule",
    ]
    for r in rows:
        e = r[6]
        lines.append(
            f"{name[r[0]]} & {lg(r[1])} & {r[2]}/{N_NETS} & {r[3]:.0f} & {r[4]}/{N_NETS} & "
            f"{r[5]:.0f} & $2^{{{np.log2(e):.0f}}}$\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(f"{_path.ROOT}/tables/B/b_signs.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(open(f"{_path.ROOT}/results/b_signs.csv").read())
    print(open(f"{_path.ROOT}/results/b_signs_failures.csv").read())


if __name__ == "__main__":
    main()
