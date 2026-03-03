.PHONY: help install dev-install test coverage lint format clean docs build

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install the package
	uv sync

dev-install:  ## Install the package with development dependencies
	uv sync --extra dev --extra docs

test:  ## Run tests
	uv run pytest tests/

coverage:  ## Run tests with coverage report
	uv run pytest --cov=src --cov-report=html --cov-report=term tests/

lint:  ## Run linters (ruff and mypy)
	uv run ruff check src tests
	uv run mypy src

format:  ## Format code with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

clean:  ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docs:  ## Build documentation
	cd docs && uv run sphinx-build -b html source _build/html

build:  ## Build the package
	uv build

.DEFAULT_GOAL := help
