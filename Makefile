.PHONY: help install install-dev data synthetic train evaluate test lint format app clean

PYTHON ?= python

help:
	@echo "PartSense Platform Reproducibility Suite"
	@echo "----------------------------------------"
	@echo "make install     : Install runtime dependencies"
	@echo "make install-dev : Install development and runtime dependencies"
	@echo "make data        : Ingest UCI AI4I 2020 dataset and engineer physics features"
	@echo "make synthetic   : Generate seeded synthetic warranty & sales aftersales layer (DuckDB)"
	@echo "make train       : Train XGBoost failure classifier, OvR diagnoser, & Isolation Forest"
	@echo "make evaluate    : Evaluate test set, SHAP global/waterfall, & export reports/metrics.json"
	@echo "make test        : Run pytest suite"
	@echo "make lint        : Run ruff lint and format check"
	@echo "make format      : Auto-format code with ruff"
	@echo "make app         : Launch Streamlit web dashboard"
	@echo "make clean       : Clean build, cache, and test artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) -m partsense.data.pipeline --stage data

synthetic:
	$(PYTHON) -m partsense.synthetic

train:
	$(PYTHON) -m partsense.models.train

evaluate:
	$(PYTHON) -m partsense.models.evaluate

test:
	$(PYTHON) -m pytest tests/

lint:
	$(PYTHON) -m ruff check src/ tests/ app/
	$(PYTHON) -m ruff format --check src/ tests/ app/

format:
	$(PYTHON) -m ruff format src/ tests/ app/
	$(PYTHON) -m ruff check --fix src/ tests/ app/

app:
	$(PYTHON) -m streamlit run app/Home.py

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ src/partsense/__pycache__ tests/__pycache__ app/__pycache__
