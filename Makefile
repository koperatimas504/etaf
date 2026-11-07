.PHONY: help install dev lint type test docker clean

help:
	@echo "targets: install dev lint type test docker clean"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	black --check .
	isort --check-only .

type:
	mypy

test:
	pytest

docker:
	docker build -t etaf-agent .

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache *.egg-info build dist runs
