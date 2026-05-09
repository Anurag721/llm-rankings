.PHONY: help ingest validate test serve clean install

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dev dependencies
	pip install -e ".[dev]"

ingest: ## Run the data ingestion pipeline
	python -m scripts.ingest_rankings

ingest-offline: ## Run ingestion using cached data
	python -m scripts.ingest_rankings --offline

validate: ## Validate the generated rankings payload
	python -m scripts.validate_rankings data/rankings.json

test: ## Run the test suite
	pytest -v

test-cov: ## Run tests with coverage
	pytest --cov=scripts --cov-report=term-missing

serve: ## Serve the site locally
	python -m http.server 8000 -d .

clean: ## Remove cache and generated files
	rm -rf data/.cache
	rm -rf __pycache__ scripts/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .coverage htmlcov