# SE6003 C7 — Recovering neural-network models

Black-box extraction of a 10-10-1 ReLU network, following Carlini, Jagielski
and Mironov, *Cryptanalytic Extraction of Neural Network Models*, CRYPTO 2020.

This repository currently holds the frozen shared code (`common.py`) and
**role B (first-layer recovery)**. Roles A (`search.py`, `layer2.py`) and C
(`attack.py`, `evaluate.py`, `benchmark.py`) are merged in by their owners.

## Role B modules

| File | Function(s) | Paper section |
|---|---|---|
| `layer1.py` | `row_from_witness`, `merge_neurons`; bonus `row_from_witness_coord` | §4.2, §4.3.1, §5.2 |
| `signs.py` | `fix_signs` (pre-image), `fix_signs_lstsq` (ours) | §4.2 |
| `refine.py` | `refine_layer1` (bonus) | §5.1 |
| `bmetrics.py` | ground-truth helpers for B's tests/experiments only | — |

Attack code uses only `oracle(x)`, `oracle.queries` and `oracle.input_dim`.

## Usage

```bash
make install     # numpy, scipy, matplotlib, pytest
make test        # unit tests (< 5 s)
make bench-b     # role B experiments -> results/, figures/B/, tables/B/
```

| Output | Script | Report |
|---|---|---|
| `results/b_rows*.csv`, `figures/B/b_rows.pdf`, `tables/B/b_rows_methods.tex` | `experiments/b_rows.py` | Fig. B1, Table B2, §3.2 |
| `results/b_signs.csv`, `results/b_signs_failures.csv`, `tables/B/b_signs.tex` | `experiments/b_signs.py` | Table B1, §3.3 |
| `results/b_refine.csv`, `figures/B/b_refine.pdf` | `experiments/b_refine.py` | Fig. B2, §3.5 |

All experiments use fixed seeds (2026).

## Merging into the group repository

Role B owns only these files: `layer1.py`, `signs.py`, `refine.py`, `bmetrics.py`,
`tests/test_layer1.py`, `tests/test_signs.py`, `tests/test_refine.py`,
`tests/test_b_pipeline.py`, `experiments/b_*.py`, `results/b_*.csv`,
`figures/B/`, `tables/B/`, `report/`. `common.py` is the frozen copy (byte-identical).
`conftest.py`, `pytest.ini`, `Makefile`, `README.md`, `requirements.txt` and
`.gitignore` are only here so this repository runs on its own; use C's versions
when merging. B's code only imports `common` and numpy, and only calls
`oracle(x)`; `bmetrics.py` uses ground truth and is imported by tests and
experiments only.

`tests/test_b_pipeline.py` chains B's functions the way `attack.py` will:
`row_from_witness` on every witness, `merge_neurons` until it returns 10 rows,
then `fix_signs` and `refine_layer1`. A's search is replaced by a stand-in. The LaTeX sources of B's report
sections are in `report/`.
