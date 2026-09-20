PY := ../.venv/bin/python

.PHONY: smoke train serve test

smoke:
	$(PY) -m hydro1.train --smoke

train:
	$(PY) -m hydro1.train

serve:
	$(PY) -m hydro1.server

test:
	$(PY) -m pytest tests/ -q
