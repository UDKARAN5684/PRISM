"""FastAPI scoring and policy service.

Endpoints
---------
``GET  /health``    liveness plus whether a model bundle is actually loaded
``GET  /metadata``  everything a caller needs to build a valid request
``POST /score``     per-customer causal effect of every offer, in months and in currency
``POST /policy``    budget-constrained assignment across a batch
``POST /explain``   per-customer feature attributions for one offer
``GET  /metrics``   Prometheus-format counters and a latency histogram

Design notes
------------
* **The service starts without a model.** A container that crash-loops because an artifact is
  missing is worse than one that reports ``degraded`` and serves ``/health``. Every scoring
  endpoint returns HTTP 503 with an actionable message until a bundle is present.
* **Training and serving share one encoder.** The fitted encoder travels inside the bundle, so
  the design matrix built at serving time has the same columns in the same order as training.
  That is the whole defence against train/serve skew.
* **The bundle is hot-reloadable** via ``POST /reload``, so a new model can be rolled out without
  restarting the process.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.data.schema import ARM_COSTS, ARM_REDEMPTION, FEATURES, N_ARMS, arm_name
from prism.utils.logging import get_logger
from prism.utils.optional import HAS_SHAP, availability

# These must live in MODULE globals, not inside the app factory: `from __future__ import
# annotations` turns every annotation into a string, and FastAPI resolves an endpoint's hints
# against the function's __globals__. Imported locally, the response models would be invisible
# and FastAPI would read the request body as a query parameter.
try:
    from prism.serving.schemas import (
        ArmScore,
        CustomerScore,
        ExplainRequest,
        ExplainResponse,
        FeatureContribution,
        HealthResponse,
        MetadataResponse,
        PolicyArmSummary,
        PolicyRequest,
        PolicyResponse,
        ScoreRequest,
        ScoreResponse,
    )

    _HAS_SCHEMAS = True
except ImportError:  # pragma: no cover - pydantic absent
    _HAS_SCHEMAS = False

log = get_logger("serving.api")

DEFAULT_BUNDLE_PATH = os.environ.get("PRISM_BUNDLE_PATH", "artifacts/models/bundle.joblib")

__all__ = ["app", "ModelBundle", "get_bundle", "METRICS"]


# =====================================================================================
# Lightweight metrics (no prometheus_client dependency)
# =====================================================================================


@dataclass
class _Metrics:
    """In-process request counters and latency buckets, rendered in Prometheus text format."""

    requests: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latencies: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    customers_scored: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, endpoint: str, seconds: float, n: int = 0, error: bool = False) -> None:
        """Record one request."""
        with self._lock:
            self.requests[endpoint] += 1
            if error:
                self.errors[endpoint] += 1
            lat = self.latencies[endpoint]
            lat.append(seconds)
            if len(lat) > 5000:  # bounded memory
                del lat[: len(lat) - 5000]
            self.customers_scored += n

    def render(self) -> str:
        """Return the Prometheus exposition-format text."""
        buckets = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        lines = [
            "# HELP prism_requests_total Total requests by endpoint.",
            "# TYPE prism_requests_total counter",
        ]
        with self._lock:
            for ep, n in sorted(self.requests.items()):
                lines.append(f'prism_requests_total{{endpoint="{ep}"}} {n}')
            lines += ["# HELP prism_errors_total Failed requests by endpoint.",
                      "# TYPE prism_errors_total counter"]
            for ep in sorted(self.requests):
                lines.append(f'prism_errors_total{{endpoint="{ep}"}} {self.errors.get(ep, 0)}')
            lines += ["# HELP prism_request_duration_seconds Request latency.",
                      "# TYPE prism_request_duration_seconds histogram"]
            for ep, lat in sorted(self.latencies.items()):
                arr = np.asarray(lat, dtype=float)
                cum = 0
                for b in buckets:
                    cum = int((arr <= b).sum())
                    lines.append(f'prism_request_duration_seconds_bucket{{endpoint="{ep}",le="{b}"}} {cum}')
                lines.append(f'prism_request_duration_seconds_bucket{{endpoint="{ep}",le="+Inf"}} {arr.size}')
                lines.append(f'prism_request_duration_seconds_sum{{endpoint="{ep}"}} {arr.sum():.6f}')
                lines.append(f'prism_request_duration_seconds_count{{endpoint="{ep}"}} {arr.size}')
            lines += [
                "# HELP prism_customers_scored_total Customers scored since start.",
                "# TYPE prism_customers_scored_total counter",
                f"prism_customers_scored_total {self.customers_scored}",
            ]
        return "\n".join(lines) + "\n"


METRICS = _Metrics()


# =====================================================================================
# Model bundle
# =====================================================================================


class ModelBundle:
    """Everything needed to turn raw customer features into a budget-aware recommendation.

    A bundle is a plain dict persisted with joblib, so it can be inspected without importing
    the serving layer. Expected keys:

    ==================  =========================================================
    ``encoder``         fitted encoder from :func:`prism.data.features.build_design_matrix`
    ``feature_names``   column order of the design matrix
    ``cate_model``      fitted CATE estimator exposing ``predict_cate`` (and optionally
                        ``predict_value_cate`` / ``predict_survival_curves``)
    ``survival_model``  fitted model exposing ``predict_churn_prob`` / ``predict_rmst``
    ``clv``             optional CLV model exposing ``predict``
    ``econ``            :class:`prism.decision.economics.EconomicConfig`
    ``metadata``        dict: version, trained_at, training_window, estimator, metrics
    ==================  =========================================================

    Any of the model slots may be absent; :meth:`score` degrades to whatever is available and
    reports what it could not compute, rather than raising.
    """

    def __init__(self, payload: dict[str, Any], path: str | Path | None = None) -> None:
        self.payload = payload
        self.path = str(path) if path else None
        self.metadata: dict[str, Any] = dict(payload.get("metadata") or {})
        self.version: str = str(self.metadata.get("version", "unknown"))
        self.loaded_at: str = pd.Timestamp.utcnow().isoformat()
        self.encoder = payload.get("encoder")
        self.feature_names: list[str] = list(payload.get("feature_names") or [])
        self.cate_model = payload.get("cate_model")
        self.survival_model = payload.get("survival_model")
        self.clv_model = payload.get("clv")
        self.econ = payload.get("econ")
        if self.econ is None:
            from prism.decision.economics import EconomicConfig

            self.econ = EconomicConfig()

    # -- construction ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path = DEFAULT_BUNDLE_PATH) -> ModelBundle:
        """Load a bundle from disk.

        Raises
        ------
        FileNotFoundError
            If the bundle is absent. Callers should treat this as "degraded", not fatal.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"no model bundle at {p}. Run:  python -m prism.pipelines.run_all "
                f"--config configs/fast.yaml --steps all"
            )
        import joblib

        payload = joblib.load(p)
        if not isinstance(payload, dict):
            raise TypeError(f"bundle at {p} is a {type(payload).__name__}, expected a dict")
        log.info("loaded model bundle %s (version %s)", p, (payload.get("metadata") or {}).get("version"))
        return cls(payload, path=p)

    # -- inference ---------------------------------------------------------------------

    def _design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Encode a raw feature frame with the *training* encoder."""
        from prism.data.features import build_design_matrix

        if self.encoder is None:
            raise RuntimeError("bundle has no fitted encoder; it cannot score safely")
        X, names, _ = build_design_matrix(df, fit=False, encoder=self.encoder)
        if self.feature_names and list(names) != list(self.feature_names):
            raise RuntimeError(
                "serving design matrix does not match training: "
                f"{len(names)} columns vs {len(self.feature_names)} expected"
            )
        return np.asarray(X, dtype=float)

    def score(self, df: pd.DataFrame, include_curves: bool = False) -> pd.DataFrame:
        """Score a batch of customers.

        Parameters
        ----------
        df : pandas.DataFrame
            One row per customer, with a ``customer_id`` column plus the 23 model features.
            Missing features are permitted; the encoder imputes them as it did in training.
        include_curves : bool, default False
            Also return per-arm survival curves as an object column.

        Returns
        -------
        pandas.DataFrame
            ``customer_id``, ``churn_risk``, ``rmst``, ``clv``, and per arm
            ``cate_rmst_<a>``, ``cate_value_<a>``, ``cost_<a>``, ``net_value_<a>``, plus
            ``recommended_arm`` and ``expected_net_value``. The frame carries any degradation
            notes in ``df.attrs["warnings"]``.
        """
        warnings: list[str] = []
        frame = df.copy()
        for col in FEATURES:
            if col not in frame.columns:
                frame[col] = np.nan
                warnings.append(f"feature {col!r} absent from the request; imputed")
        ids = frame["customer_id"].astype(str).to_numpy() if "customer_id" in frame else np.array(
            [f"row_{i}" for i in range(len(frame))]
        )
        n = len(frame)
        X = self._design_matrix(frame)

        # --- baseline risk / lifetime -------------------------------------------------
        churn_risk = np.full(n, np.nan)
        rmst = np.full(n, np.nan)
        if self.survival_model is not None:
            try:
                churn_risk = np.asarray(self.survival_model.predict_churn_prob(X), dtype=float).ravel()
            except Exception as exc:
                warnings.append(f"churn_risk unavailable: {exc}")
            try:
                rmst = np.asarray(self.survival_model.predict_rmst(X), dtype=float).ravel()
            except Exception as exc:
                warnings.append(f"rmst unavailable: {exc}")
        else:
            warnings.append("bundle has no survival model; churn_risk and rmst are null")

        clv = np.full(n, np.nan)
        if self.clv_model is not None:
            try:
                clv = np.asarray(self.clv_model.predict(X), dtype=float).ravel()
            except Exception as exc:
                warnings.append(f"clv unavailable: {exc}")
        elif np.isfinite(rmst).any():
            margin = float(getattr(self.econ, "margin_rate", 0.3))
            aov = pd.to_numeric(frame.get("avg_order_value"), errors="coerce").to_numpy(dtype=float)
            clv = rmst * np.nan_to_num(aov, nan=float(np.nanmedian(aov)) if np.isfinite(aov).any() else 50.0) * margin
            warnings.append("clv approximated from rmst x avg_order_value x margin_rate")

        # --- causal effects -----------------------------------------------------------
        k = N_ARMS - 1
        cate_value = np.zeros((n, k))
        cate_rmst = np.zeros((n, k))
        if self.cate_model is None:
            warnings.append("bundle has no CATE model; all treatment effects are zero")
        else:
            got_value = False
            if hasattr(self.cate_model, "predict_value_cate"):
                try:
                    cate_value = np.asarray(self.cate_model.predict_value_cate(X), dtype=float).reshape(n, -1)
                    got_value = True
                except Exception as exc:
                    warnings.append(f"predict_value_cate failed: {exc}")
            try:
                base = np.asarray(self.cate_model.predict_cate(X), dtype=float).reshape(n, -1)
                if hasattr(self.cate_model, "predict_value_cate"):
                    cate_rmst = base                      # survival uplift: predict_cate is RMST
                    if not got_value:
                        cate_value = base
                else:
                    cate_value = base if not got_value else cate_value
                    cate_rmst = np.full_like(base, np.nan)
            except Exception as exc:
                warnings.append(f"predict_cate failed: {exc}")

            for arr_name, arr in (("cate_value", cate_value), ("cate_rmst", cate_rmst)):
                if arr.shape[1] != k:
                    warnings.append(f"{arr_name} returned {arr.shape[1]} arms, expected {k}; padding")
            cate_value = _fit_width(cate_value, n, k)
            cate_rmst = _fit_width(cate_rmst, n, k)

        # --- economics ----------------------------------------------------------------
        from prism.decision.economics import expected_net_value

        net = np.asarray(expected_net_value(cate_value, self.econ), dtype=float)
        costs = np.asarray(
            [c * r for c, r in zip(getattr(self.econ, "arm_costs", ARM_COSTS),
                                   getattr(self.econ, "offer_redemption_rate", ARM_REDEMPTION))],
            dtype=float,
        )

        out = pd.DataFrame({"customer_id": ids, "churn_risk": churn_risk, "rmst": rmst, "clv": clv})
        for a in range(N_ARMS):
            out[f"cate_rmst_{a}"] = 0.0 if a == 0 else cate_rmst[:, a - 1]
            out[f"cate_value_{a}"] = 0.0 if a == 0 else cate_value[:, a - 1]
            out[f"cost_{a}"] = costs[a] if a < len(costs) else 0.0
            out[f"net_value_{a}"] = net[:, a]

        rec = net.argmax(axis=1)
        rec[net.max(axis=1) <= 0] = 0                 # doing nothing is always available
        out["recommended_arm"] = rec.astype(int)
        out["recommended_arm_name"] = [arm_name(a) for a in rec]
        out["expected_net_value"] = np.maximum(net[np.arange(n), rec], 0.0)

        if include_curves and hasattr(self.cate_model, "predict_survival_curves"):
            try:
                curves = np.asarray(self.cate_model.predict_survival_curves(X), dtype=float)
                out["survival_curves"] = [
                    {arm_name(a): curves[i, a].tolist() for a in range(curves.shape[1])} for i in range(n)
                ]
            except Exception as exc:
                warnings.append(f"survival curves unavailable: {exc}")

        out.attrs["warnings"] = warnings
        return out

    def policy(self, df: pd.DataFrame, budget: float, method: str = "lagrangian") -> pd.DataFrame:
        """Score a batch and solve the budget-constrained assignment over it.

        Returns
        -------
        pandas.DataFrame
            The :meth:`score` frame plus an ``assigned_arm`` column. Allocation diagnostics are
            attached in ``df.attrs["allocation"]``.
        """
        from prism.decision.optimize import greedy_knapsack, lagrangian_allocate, lp_allocate

        scored = self.score(df)
        n = len(scored)
        net = scored[[f"net_value_{a}" for a in range(N_ARMS)]].to_numpy(dtype=float)
        cost = np.tile(
            np.asarray(
                [c * r for c, r in zip(getattr(self.econ, "arm_costs", ARM_COSTS),
                                       getattr(self.econ, "offer_redemption_rate", ARM_REDEMPTION))],
                dtype=float,
            ),
            (n, 1),
        )
        solver = {"greedy": greedy_knapsack, "lagrangian": lagrangian_allocate, "lp": lp_allocate}.get(
            method, lagrangian_allocate
        )
        result = solver(net, cost, float(budget))
        scored["assigned_arm"] = np.asarray(result.assignment, dtype=int)
        scored["assigned_arm_name"] = [arm_name(a) for a in scored["assigned_arm"]]
        scored.attrs["allocation"] = result
        return scored


def _fit_width(arr: np.ndarray, n: int, k: int) -> np.ndarray:
    """Coerce a CATE array to ``(n, k)``, padding with zeros or truncating."""
    arr = np.asarray(arr, dtype=float).reshape(n, -1)
    if arr.shape[1] == k:
        return arr
    out = np.zeros((n, k))
    m = min(k, arr.shape[1])
    out[:, :m] = arr[:, :m]
    return out


# =====================================================================================
# App
# =====================================================================================

_BUNDLE: ModelBundle | None = None
_BUNDLE_ERROR: str | None = None
_BUNDLE_LOCK = threading.Lock()


def get_bundle(path: str | Path = DEFAULT_BUNDLE_PATH, force: bool = False) -> ModelBundle | None:
    """Return the process-wide bundle, loading it on first use.

    Never raises: a missing or broken bundle leaves the service in ``degraded`` state with the
    reason recorded for ``/health``.
    """
    global _BUNDLE, _BUNDLE_ERROR
    with _BUNDLE_LOCK:
        if _BUNDLE is not None and not force:
            return _BUNDLE
        try:
            _BUNDLE = ModelBundle.load(path)
            _BUNDLE_ERROR = None
        except Exception as exc:
            _BUNDLE = None
            _BUNDLE_ERROR = str(exc)
            log.warning("model bundle not loaded: %s", exc)
        return _BUNDLE


def _build_app():
    """Construct the FastAPI application, or return None when FastAPI is not installed."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, PlainTextResponse
    except ImportError:  # pragma: no cover - serving is optional
        log.warning("fastapi is not installed; prism.serving.api.app is None")
        return None

    if not _HAS_SCHEMAS:  # pragma: no cover - pydantic absent
        log.warning("prism.serving.schemas unavailable; prism.serving.api.app is None")
        return None

    application = FastAPI(
        title="PRISM Retention API",
        version="1.0.0",
        description=(
            "Causal Survival Uplift scoring. Returns, per customer, the estimated causal effect "
            "of every retention offer on expected retained months and on discounted lifetime "
            "value, the redemption-adjusted cost, and a budget-aware recommendation."
        ),
        contact={"name": "PRISM"},
        license_info={"name": "MIT"},
    )

    @application.middleware("http")
    async def _timing(request: Request, call_next):
        t0 = time.perf_counter()
        endpoint = request.url.path
        try:
            response = await call_next(request)
        except Exception:
            METRICS.observe(endpoint, time.perf_counter() - t0, error=True)
            raise
        dt = time.perf_counter() - t0
        METRICS.observe(endpoint, dt, error=response.status_code >= 500)
        response.headers["X-Process-Time-ms"] = f"{dt * 1000:.2f}"
        return response

    def _require_bundle() -> ModelBundle:
        bundle = get_bundle()
        if bundle is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "no model bundle loaded. Train one with: python -m prism.pipelines.run_all "
                    f"--config configs/fast.yaml --steps all   (looked in {DEFAULT_BUNDLE_PATH}). "
                    f"Underlying error: {_BUNDLE_ERROR}"
                ),
            )
        return bundle

    def _to_frame(customers) -> pd.DataFrame:
        return pd.DataFrame([c.to_row() for c in customers])

    # -- endpoints ---------------------------------------------------------------------

    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Liveness and model-availability check. Always 200; read ``status``."""
        bundle = get_bundle()
        if bundle is None:
            return HealthResponse(
                status="degraded", model_version=None, loaded_at=None,
                bundle_path=str(DEFAULT_BUNDLE_PATH), detail=_BUNDLE_ERROR,
            )
        return HealthResponse(
            status="ok", model_version=bundle.version, loaded_at=bundle.loaded_at,
            bundle_path=bundle.path, detail=None,
        )

    @application.get("/metadata", response_model=MetadataResponse, tags=["ops"])
    def metadata() -> MetadataResponse:
        """Everything a client needs to construct a valid request."""
        bundle = get_bundle()
        meta = bundle.metadata if bundle else {}
        econ = bundle.econ if bundle else None
        return MetadataResponse(
            model_version=(bundle.version if bundle else None),
            trained_at=meta.get("trained_at"),
            training_window=meta.get("training_window"),
            arm_costs=list(getattr(econ, "arm_costs", ARM_COSTS)),
            estimator=meta.get("estimator"),
            headline_metrics=meta.get("metrics"),
            dependencies=availability(),
        )

    @application.post("/reload", tags=["ops"])
    def reload_bundle() -> dict[str, Any]:
        """Hot-reload the model bundle from disk without restarting the process."""
        bundle = get_bundle(force=True)
        if bundle is None:
            raise HTTPException(status_code=503, detail=_BUNDLE_ERROR or "reload failed")
        return {"status": "reloaded", "model_version": bundle.version, "loaded_at": bundle.loaded_at}

    @application.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    def metrics() -> str:
        """Prometheus exposition format."""
        return METRICS.render()

    @application.post("/score", response_model=ScoreResponse, tags=["scoring"])
    def score(req: ScoreRequest) -> ScoreResponse:
        """Score a batch: the causal effect of every offer, per customer."""
        bundle = _require_bundle()
        t0 = time.perf_counter()
        frame = _to_frame(req.customers)
        try:
            scored = bundle.score(frame, include_curves=req.include_survival_curves)
        except Exception as exc:
            log.exception("scoring failed")
            raise HTTPException(status_code=500, detail=f"scoring failed: {exc}") from exc

        results: list[CustomerScore] = []
        for _, row in scored.iterrows():
            arms = [
                ArmScore(
                    arm=a,
                    arm_name=arm_name(a),
                    cate_rmst=_f_or_none(row.get(f"cate_rmst_{a}")),
                    cate_value=_f(row.get(f"cate_value_{a}")),
                    cost=_f(row.get(f"cost_{a}")),
                    net_value=_f(row.get(f"net_value_{a}")),
                )
                for a in range(N_ARMS)
            ]
            results.append(
                CustomerScore(
                    customer_id=str(row["customer_id"]),
                    churn_risk=float(np.clip(_f(row.get("churn_risk")), 0.0, 1.0)),
                    clv=_f(row.get("clv")),
                    rmst=max(_f(row.get("rmst")), 0.0),
                    arms=arms,
                    recommended_arm=int(row["recommended_arm"]),
                    recommended_arm_name=str(row["recommended_arm_name"]),
                    expected_net_value=max(_f(row.get("expected_net_value")), 0.0),
                    survival_curves=row.get("survival_curves") if "survival_curves" in scored else None,
                )
            )

        METRICS.observe("/score", time.perf_counter() - t0, n=len(results))
        return ScoreResponse(
            scores=results,
            model_version=bundle.version,
            n_scored=len(results),
            latency_ms=(time.perf_counter() - t0) * 1000,
            warnings=list(scored.attrs.get("warnings", []))[:20],
        )

    @application.post("/policy", response_model=PolicyResponse, tags=["scoring"])
    def policy(req: PolicyRequest) -> PolicyResponse:
        """Solve the budget-constrained assignment over a batch."""
        bundle = _require_bundle()
        t0 = time.perf_counter()
        frame = _to_frame(req.customers)
        try:
            scored = bundle.policy(frame, budget=req.budget, method=req.method)
        except Exception as exc:
            log.exception("policy allocation failed")
            raise HTTPException(status_code=500, detail=f"policy allocation failed: {exc}") from exc

        alloc = scored.attrs.get("allocation")
        assignments = dict(zip(scored["customer_id"].astype(str), scored["assigned_arm"].astype(int)))
        n_treated = int((scored["assigned_arm"] > 0).sum())
        total_cost = float(getattr(alloc, "total_cost", 0.0))
        value = float(getattr(alloc, "expected_incremental_value", 0.0))

        per_arm: list[PolicyArmSummary] = []
        for a in range(N_ARMS):
            mask = scored["assigned_arm"] == a
            n_a = int(mask.sum())
            per_arm.append(
                PolicyArmSummary(
                    arm=a,
                    arm_name=arm_name(a),
                    n_assigned=n_a,
                    total_cost=float(scored.loc[mask, f"cost_{a}"].sum()) if n_a else 0.0,
                    total_net_value=float(scored.loc[mask, f"net_value_{a}"].sum()) if n_a else 0.0,
                    mean_net_value=float(scored.loc[mask, f"net_value_{a}"].mean()) if n_a else 0.0,
                )
            )

        fairness = None
        if req.fairness_attribute and req.fairness_attribute in frame.columns:
            try:
                from prism.decision.fairness import policy_fairness_audit

                audit = policy_fairness_audit(
                    scored["assigned_arm"].to_numpy(),
                    frame[req.fairness_attribute],
                    net_value=scored["expected_net_value"].to_numpy(),
                )
                fairness = audit.to_dict(orient="records")
            except Exception as exc:
                log.warning("fairness audit failed: %s", exc)

        lam = getattr(alloc, "dual_lambda", None)
        return PolicyResponse(
            assignments=assignments,
            n_customers=len(scored),
            n_treated=n_treated,
            total_cost=total_cost,
            budget=req.budget,
            budget_utilisation=(total_cost / req.budget) if req.budget > 0 else 0.0,
            expected_incremental_value=value,
            roi=(value / total_cost) if total_cost > 0 else None,
            shadow_price=(float(lam) if lam is not None and np.isfinite(lam) else None),
            per_arm=per_arm,
            method=req.method,
            fairness=fairness,
            model_version=bundle.version,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    @application.post("/explain", response_model=ExplainResponse, tags=["scoring"])
    def explain(req: ExplainRequest) -> ExplainResponse:
        """Per-customer feature attributions for one offer's estimated effect."""
        bundle = _require_bundle()
        frame = _to_frame(req.customers)
        X = bundle._design_matrix(frame)
        names = bundle.feature_names or [f"f{i}" for i in range(X.shape[1])]
        col = req.arm - 1

        def _predict(mat: np.ndarray) -> np.ndarray:
            out = np.asarray(bundle.cate_model.predict_cate(mat), dtype=float).reshape(len(mat), -1)
            return out[:, min(col, out.shape[1] - 1)]

        method = "permutation"
        contributions: np.ndarray
        baseline = float(np.mean(_predict(X))) if len(X) else 0.0

        if HAS_SHAP:
            try:
                import shap

                background = shap.utils.sample(X, min(50, len(X)), random_state=0) if len(X) > 1 else X
                explainer = shap.Explainer(_predict, background)
                contributions = np.asarray(explainer(X).values, dtype=float)
                method = "shap"
            except Exception as exc:
                log.warning("SHAP failed (%s); falling back to permutation attribution", exc)
                contributions = _permutation_contributions(_predict, X)
        else:
            contributions = _permutation_contributions(_predict, X)

        ids = frame["customer_id"].astype(str).tolist()
        explanations: dict[str, list[FeatureContribution]] = {}
        for i, cid in enumerate(ids):
            order = np.argsort(-np.abs(contributions[i]))[: req.top_k]
            explanations[cid] = [
                FeatureContribution(
                    feature=names[j] if j < len(names) else f"f{j}",
                    value=float(X[i, j]),
                    contribution=float(contributions[i, j]),
                )
                for j in order
            ]

        return ExplainResponse(
            explanations=explanations,
            arm=req.arm,
            arm_name=arm_name(req.arm),
            baseline=baseline,
            method=method,
            model_version=bundle.version,
        )

    @application.exception_handler(ValueError)
    async def _value_error(_request, exc: ValueError):  # pragma: no cover - defensive
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return application


def _permutation_contributions(predict, X: np.ndarray) -> np.ndarray:
    """Cheap per-row attribution: the change in prediction when a feature is set to the batch median.

    A documented approximation used when SHAP is unavailable. It is local and additive-ish but
    ignores interactions, which is stated in the response's ``method`` field so nobody mistakes
    it for a Shapley value.
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return np.zeros_like(X)
    base = predict(X)
    med = np.median(X, axis=0)
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, j] = med[j]
        out[:, j] = base - predict(Xp)
    return out


def _f_or_none(v: Any) -> float | None:
    """Coerce to a finite float, or None when the value was never computed.

    Distinct from :func:`_f` on purpose: a missing quantity must surface as ``null``, not as
    ``0.0``. Rendering "not computed" as "no effect" is the kind of quiet mistranslation that
    gets acted on.
    """
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _f(v: Any, default: float = 0.0) -> float:
    """Coerce to a finite float, substituting ``default`` for None/NaN/inf."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


app = _build_app()


if __name__ == "__main__":  # pragma: no cover - smoke test
    if app is None:
        print("fastapi not installed; skipping API smoke test")
        raise SystemExit(0)

    from fastapi.testclient import TestClient

    from prism.serving.schemas import _EXAMPLE_CUSTOMER

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200, h.text
    print("health          :", h.json()["status"])

    m = client.get("/metadata")
    assert m.status_code == 200
    print("metadata arms   :", m.json()["arm_names"])

    payload = {"customers": [_EXAMPLE_CUSTOMER, {**_EXAMPLE_CUSTOMER, "customer_id": "C2", "nps": None}]}
    s = client.post("/score", json=payload)
    # Without a trained bundle this must be a clean 503, never a 500 or a crash.
    assert s.status_code in (200, 503), s.text
    print("score           :", s.status_code, "->", str(s.json())[:110])

    p = client.post("/policy", json={**payload, "budget": 100.0})
    assert p.status_code in (200, 503), p.text
    print("policy          :", p.status_code)

    bad = client.post("/score", json={"customers": [{**_EXAMPLE_CUSTOMER, "nps": 9999}]})
    assert bad.status_code == 422, f"out-of-range nps should be rejected, got {bad.status_code}"
    print("validation      : 422 as expected for nps=9999")

    dupe = client.post("/score", json={"customers": [_EXAMPLE_CUSTOMER, _EXAMPLE_CUSTOMER]})
    assert dupe.status_code == 422, "duplicate customer_id should be rejected"
    print("validation      : 422 as expected for duplicate ids")

    mt = client.get("/metrics")
    assert mt.status_code == 200 and "prism_requests_total" in mt.text
    print("metrics lines   :", len(mt.text.strip().splitlines()))

    schema = client.get("/openapi.json")
    assert schema.status_code == 200 and "/score" in schema.json()["paths"]
    print("openapi paths   :", sorted(schema.json()["paths"]))
    print("api.py OK")
