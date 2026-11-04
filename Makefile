# Shortcuts for the common tasks. Uses the active Python (or .venv if present).
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: help install test test-fast lint format app menu all clean

help:
	@echo "make install    install the pinned requirements"
	@echo "make test       run the full test suite"
	@echo "make test-fast  skip tests that need the full Parquet file"
	@echo "make lint       ruff checks (same as CI)"
	@echo "make format     auto-fix import order and simple lint issues"
	@echo "make app        start the Streamlit dashboard"
	@echo "make menu       start the console menu"
	@echo "make all        full pipeline (python main.py --all, 30-60 min)"
	@echo "make clean      remove caches"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

test-fast:
	$(PYTHON) -m pytest -q -m "not slow"

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --fix .

app:
	$(PYTHON) -m streamlit run app.py

menu:
	$(PYTHON) main.py

all:
	$(PYTHON) main.py --all

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
