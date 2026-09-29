# PRISM - developer entry points.
# Works with GNU make on Linux/macOS and with Git Bash / WSL on Windows.

PY ?= python
CONFIG ?= configs/default.yaml

.PHONY: help install install-dev run run-fast test test-fast lint fmt typecheck \
        api dashboard mlflow docker docker-run clean smoke all

help:                     ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:                  ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

install-dev:              ## Install runtime + dev dependencies and the package itself
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e .

run:                      ## Full pipeline at production scale (~10-20 min)
	$(PY) -m prism.pipelines.run_all --config $(CONFIG) --steps all

run-fast:                 ## Full pipeline at reduced scale (~2-4 min) - start here
	$(PY) -m prism.pipelines.run_all --config configs/fast.yaml --steps all

smoke:                    ## Run every module's built-in self test
	@fail=0; for m in prism.utils.seeds prism.utils.optional prism.utils.logging \
	          prism.utils.validation prism.utils.tracking prism.config prism.data.schema \
	          prism.data.dgp prism.data.messy prism.data.warehouse prism.data.features \
	          prism.data.real prism.models.survival prism.models.clv prism.models.propensity \
	          prism.causal.learners prism.causal.forest prism.causal.survival_uplift \
	          prism.causal.evaluate prism.causal.refute prism.decision.economics \
	          prism.decision.optimize prism.decision.policy_eval prism.decision.fairness \
	          prism.monitoring.drift prism.monitoring.alerts prism.serving.schemas \
	          prism.serving.api prism.pipelines.figures; do \
	  if $(PY) -m $$m >/dev/null 2>&1; then echo "  PASS  $$m"; \
	  else echo "  FAIL  $$m"; fail=1; fi; \
	done; exit $$fail

test:                     ## Full test suite
	$(PY) -m pytest tests -v

test-fast:                ## Test suite excluding slow and network tests
	$(PY) -m pytest tests -v -m "not slow and not network"

lint:                     ## Lint with ruff
	$(PY) -m ruff check prism tests

fmt:                      ## Auto-fix lint issues and format
	$(PY) -m ruff check --fix prism tests
	$(PY) -m ruff format prism tests

typecheck:                ## Static type check
	$(PY) -m mypy prism

api:                      ## Serve the scoring API on :8000
	$(PY) -m uvicorn prism.serving.api:app --host 0.0.0.0 --port 8000 --reload

dashboard:                ## Launch the executive dashboard on :8501
	$(PY) -m streamlit run prism/app/dashboard.py

mlflow:                   ## Browse experiment tracking on :5000
	$(PY) -m mlflow ui --backend-store-uri artifacts/mlruns --port 5000

docker:                   ## Build the container image
	docker build -t prism-retention:latest .

docker-run:               ## Run the full stack (API + dashboard + mlflow)
	docker compose up --build

clean:                    ## Remove generated artifacts and caches
	rm -rf artifacts/data artifacts/models artifacts/reports artifacts/mlruns artifacts/*.duckdb
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

all: install-dev lint test run-fast   ## Install, lint, test, then run the fast pipeline
