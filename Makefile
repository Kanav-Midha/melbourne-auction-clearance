.PHONY: help setup data audit baseline tune train test api docker clean all

PY ?= python

help:
	@echo "make setup     install dependencies"
	@echo "make data      build the raw extracts"
	@echo "make audit     run the leakage gate (exits 1 on any finding)"
	@echo "make baseline  fit the logistic-regression reference model"
	@echo "make tune      Optuna hyperparameter search (~3 min)"
	@echo "make train     fit and save the production model"
	@echo "make test      run the test suite"
	@echo "make api       serve the model on :8000"
	@echo "make all       data -> audit -> baseline -> train -> test"

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.data.make_dataset --n-auctions 48000 --seed 42

audit:
	$(PY) -m src.features.leakage_audit

baseline:
	$(PY) -m src.models.baseline --save

tune:
	$(PY) -m src.models.tune --trials 60

train:
	$(PY) -m src.models.train --save

test:
	$(PY) -m pytest

api:
	uvicorn src.api.main:app --reload --port 8000

docker:
	docker build -t melbourne-clearance:latest .
	docker run --rm -p 8000:8000 melbourne-clearance:latest

all: data audit baseline train test

clean:
	rm -rf data/raw/*.csv data/interim/* data/processed/* models/*.joblib
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
