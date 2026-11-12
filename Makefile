PY ?= python3

.PHONY: install test bench-b clean

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

# Role B experiments: regenerate results/b_*.csv, figures/B/*.pdf, tables/B/*.tex
bench-b:
	cd experiments && $(PY) b_rows.py && $(PY) b_signs.py && $(PY) b_refine.py

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
