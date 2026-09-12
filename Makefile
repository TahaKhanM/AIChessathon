PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
.DEFAULT_GOAL := help
.PHONY: help test test-deep lint format
help:
	@echo 'test | test-deep | lint | format'
test:
	$(PYTHON) -m pytest tests -q
test-deep:
	$(PYTHON) -m pytest tests/test_movegen.py -m slow -q

lint:
	$(RUFF) check .
	$(RUFF) format --check .
format:
	$(RUFF) format .
