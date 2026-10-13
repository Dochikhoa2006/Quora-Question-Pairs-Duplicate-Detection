.PHONY: install install-semantic install-dev demo train-demo test test-semantic lint security package docker check

install:
	python -m pip install --requirement requirements.txt
	python -m pip install --no-deps --editable .

install-semantic:
	python -m pip install --requirement requirements-semantic.txt
	python -m pip install --no-deps --editable .

install-dev:
	python -m pip install --requirement requirements-dev.txt
	python -m pip install --no-deps --editable .

demo:
	python scripts/make_demo_data.py --output data/demo_train.csv

train-demo: demo
	qqdup train --data data/demo_train.csv --output artifacts/demo --config config/default.json

test:
	pytest --cov=quora_duplicate_detection --cov-report=term-missing

test-semantic:
	QQDUP_RUN_SEMANTIC_INTEGRATION=1 pytest -q -s tests/integration/test_semantic_training.py

lint:
	ruff check .
	ruff format --check .

security:
	bandit -q -r src
	pip-audit --requirement requirements.txt
	pip-audit --requirement requirements-semantic.txt
	pip-audit --requirement requirements-signing.txt

package:
	python -m pip wheel --no-deps --wheel-dir dist .

docker:
	docker build --tag quora-duplicate-detection:local .
	docker run --rm quora-duplicate-detection:local --help

check: lint test security package
