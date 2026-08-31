# Prosper Challenge — run everything from the repo root.

VENV := backend/.venv
PYTHON := $(VENV)/bin/python

.PHONY: help install run test test-ui index benchmark evals ui clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

ui: ## Build the builder UI (served at http://localhost:7860/builder)
	cd frontend && npm install && npm run build

install: ## Create the venv and install dependencies
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r backend/requirements.txt

run: ## Run the agent + builder UI (http://localhost:7860/builder)
	$(PYTHON) backend/bot.py

test: ## Run the domain test suite (no API keys, no network)
	$(PYTHON) -m unittest discover -s backend/tests -t backend

test-ui: ## Run the graph-operation tests (Node's built-in runner)
	cd frontend && node --test src/

index: ## Precompute specialty embeddings (one batched embedding call)
	$(PYTHON) backend/scripts/build_specialty_vectors.py

benchmark: ## Measure context cost against the naive full-catalog prompt
	$(PYTHON) backend/evals/benchmark_context.py

evals: ## Run the accuracy scenarios (makes OpenAI calls; --dry-run to preview)
	$(PYTHON) backend/evals/run_evals.py $(ARGS)

clean: ## Remove the venv, node_modules and caches
	rm -rf $(VENV) frontend/node_modules frontend/dist
	find backend -type d -name __pycache__ -prune -exec rm -rf {} +
