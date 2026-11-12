"""Fig. B1: first-layer row error vs. finite-difference step size.

Uses exact witnesses (true_witness) so that only layer1.py is exercised.
Output: results/b_rows.csv, figures/B/b_rows.pdf
"""

import csv

import _path
import numpy as np
from bstyle import BLUE, GRAY, ORANGE, plt

from common import Oracle, Witness, random_network, true_witness
from layer1 import DELTA_FRAC, row_from_witness, row_from_witness_coord, row_with_steps

N_NETS, SEED = 50, 2026
EPS = np.logspace(-10, -2, 17)


def row_err(p, j, r):
    a = p.A1[j] / np.linalg.norm(p.A1[j])
    if r is None:
        return np.inf
    s = np.sign(r[0] @ a) or 1.0
    return float(np.max(np.abs(s * r[0] - a)))


def search_like_witness(p, j, rng):
    """Witness with a uniformly random search direction, as a line search
    returns it (radius = distance to the next kink along v, capped at 1)."""
    a = p.A1[j]
    x0 = rng.normal(size=p.d0)
    x = x0 - (a @ x0 + p.b1[j]) / (a @ a) * a
    v = rng.normal(size=p.d0)
    v /= np.linalg.norm(v)
    V = p.A1 @ x + p.b1
    r = min(abs(V[k]) / max(abs(p.A1[k] @ v), 1e-300) for k in range(p.d1) if k != j)
    return Witness(x, v, min(r, 1.0))


def line_check(n_nets=300):
    """Random search directions: plain differencing vs. our consistency check."""
    rng = np.random.default_rng(SEED + 1)
    plain_bad = ours_bad = retried = total = queries = 0
    for _ in range(n_nets):
        p = random_network(rng)
        oracle = Oracle(p)
        for j in range(p.d1):
            w = search_like_witness(p, j, rng)
            delta = DELTA_FRAC * w.radius
            plain_bad += row_err(p, j, row_with_steps(oracle, w, delta, 0.01 * delta)) > 1e-5
            q0 = oracle.queries
            ours_bad += row_err(p, j, row_from_witness(oracle, w)) > 1e-5
            q = oracle.queries - q0
            queries += q
            retried += q > 23
            total += 1
    with open(f"{_path.ROOT}/results/b_rows_lines.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["witnesses", "plain_wrong", "checked_wrong", "retried", "mean_queries"])
        wr.writerow([total, plain_bad, ours_bad, retried, f"{queries / total:.2f}"])
    print(open(f"{_path.ROOT}/results/b_rows_lines.csv").read())


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
        for eps, e in zip(EPS, fixed, strict=True):
            wr.writerow([f"{eps:.3e}", *(f"{v:.3e}" for v in stats(e))])
        wr.writerow(["relative(0.001r)", *(f"{v:.3e}" for v in stats(ours))])

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    med = np.array([stats(e) for e in fixed])
    ax.loglog(EPS, med[:, 0], "o-", color=BLUE, ms=3, label="median, fixed $\\varepsilon$")
    ax.loglog(EPS, med[:, 2], "s--", color=ORANGE, ms=3, label="max, fixed $\\varepsilon$")
    m0, _, m2 = stats(ours)
    ax.axhline(m0, color=BLUE, lw=0.8, ls=":")
    ax.axhline(m2, color=ORANGE, lw=0.8, ls=":")
    lab = "relative $\\varepsilon=10^{-3}r$"
    ax.text(EPS[-1], m2 * 2.0, lab + " (max)", color=GRAY, fontsize=7, ha="right")
    ax.text(EPS[0], m0 * 0.12, lab + " (median)", color=GRAY, fontsize=7)
    ax.set_xlabel("finite-difference step $\\varepsilon$")
    ax.set_ylabel("row error $\\|\\hat a_j-a_j\\|_\\infty$")
    ax.set_ylim(top=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{_path.ROOT}/figures/B/b_rows.pdf")
    print(open(f"{_path.ROOT}/results/b_rows.csv").read())


def method_compare(n_nets=100):
    """Our differencing (Alg. 2) vs. the coordinate method of [14, Sec. 5.2]."""
    rng = np.random.default_rng(SEED + 2)
    methods = {"ours": row_from_witness, "sec52": row_from_witness_coord}
    res = {(m, k): ([], []) for m in methods for k in ("exact", "search")}
    for _ in range(n_nets):
        p = random_network(rng)
        oracle = Oracle(p)
        for j in range(p.d1):
            for kind, w in (
                ("exact", true_witness(p, j, rng)),
                ("search", search_like_witness(p, j, rng)),
            ):
                for m, fn in methods.items():
                    q0 = oracle.queries
                    err = row_err(p, j, fn(oracle, w))
                    res[m, kind][0].append(err)
                    res[m, kind][1].append(oracle.queries - q0)
    rows = []
    for (m, kind), (e, q) in res.items():
        e = np.array(e)
        rows.append(
            [
                m,
                kind,
                f"{np.mean(q):.1f}",
                f"{np.median(e):.2e}",
                f"{np.max(e):.2e}",
                int(np.sum(e > 1e-5)),
                len(e),
            ]
        )
    with open(f"{_path.ROOT}/results/b_rows_methods.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["method", "witnesses", "mean_queries", "median_err", "max_err", "wrong", "n"])
        wr.writerows(rows)
    name = {
        "ours": "ours (Alg.~\\ref{alg:B-row})",
        "sec52": "coordinate~\\cite[\\S5.2]{DBLP:conf/crypto/CarliniJM20}",
    }

    def tex(v):
        m, e = f"{float(v):.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"

    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "method & witnesses & queries & median err. & max err. & wrong\\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{name[r[0]]} & {r[1]} & {r[2]} & {tex(r[3])} & {tex(r[4])} & {r[5]}/{r[6]}\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(f"{_path.ROOT}/tables/B/b_rows_methods.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(open(f"{_path.ROOT}/results/b_rows_methods.csv").read())


if __name__ == "__main__":
    main()
    line_check()
    method_compare()
