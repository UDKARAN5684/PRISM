"""PRISM - Prescriptive Retention Intelligence & Spend Management.

Causal Survival Uplift for subscription retention: estimate the effect of a retention offer
on a customer's expected discounted remaining lifetime, then allocate a fixed budget across
competing offers to maximise incremental profit.

Layers
------
``prism.data``        ground-truth simulator, realism injection, medallion warehouse,
                      point-in-time feature store, real-dataset adapters
``prism.models``      survival (discrete-time hazard, Cox, RSF), CLV (BG/NBD, Gamma-Gamma,
                      DeepCLV), multi-arm propensity with overlap diagnostics
``prism.causal``      S/T/X/DR/R meta-learners, honest causal forest, Causal Survival Uplift,
                      uplift + policy evaluation, refutation and sensitivity analysis
``prism.decision``    unit economics, budget-constrained multi-arm assignment, off-policy
                      evaluation, fairness audit
``prism.monitoring``  covariate/prediction/CATE drift and alerting
``prism.serving``     FastAPI scoring and policy service
``prism.pipelines``   the end-to-end orchestrator

See ``SPEC.md`` for the interface contract and ``docs/METHODOLOGY.md`` for the statistics.
"""

from __future__ import annotations

import os

# Set before anything can import mlflow (prism.utils.optional probes it at import time).
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
# Keep BLAS from oversubscribing cores when joblib already parallelises above it.
os.environ.setdefault("OMP_NUM_THREADS", "4")

__version__ = "1.0.0"
__all__ = ["__version__"]
