PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
.DEFAULT_GOAL := help
.PHONY: help test test-deep lint format benchmark
help:
	@echo 'test | test-deep | lint | format | benchmark'
test:
	$(PYTHON) -m pytest tests -q
test-deep:
	$(PYTHON) -m pytest tests -m slow -q

lint:
	$(RUFF) check .
	$(RUFF) format --check .
format:
	$(RUFF) format .
benchmark:
	$(PYTHON) -m bench.run --depth 3 --repeats 3
