"""End-to-end pipeline test.

The unit tests prove each module works in isolation; the contract tests prove the modules
agree on interfaces. This proves the whole thing actually composes into a run that produces
the artifacts the README, the dashboard and the serving layer all depend on.

Marked ``slow``: it runs the real pipeline at reduced scale. Skip it with
``pytest -m "not slow"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

#: Artifacts the rest of the project reads. A missing one is a broken promise, not a nuisance.
REQUIRED_REPORTS = [
    "metrics.json",
    "leaderboard.csv",
    "policy_comparison.csv",
    "model_card.md",
    "decision_memo.md",
]

REQUIRED_FIGURES = [
    "qini.png",
    "uplift_deciles.png",
    "survival_curves.png",
    "efficient_frontier.png",
    "policy_comparison.png",
    "segment_heatmap.png",
    "love_plot.png",
    "drift.png",
]


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run the fast pipeline once and return the paths it wrote into."""
    from prism.config import load_config
    from prism.pipelines.run_all import STEPS, Pipeline

    out = tmp_path_factory.mktemp("prism_run")
    cfg = load_config("configs/fast.yaml")
    # Keep it small enough to be a test rather than a job.
    cfg.sim.n_customers = 3000
    cfg.sim.n_periods = 10
    cfg.sim.horizon = 6
    cfg.split.train_end = 5
    cfg.split.valid_end = 7
    cfg.causal.learners = ("t", "dr")
    cfg.causal.refute = False
    cfg.model.survival_epochs = 6
    cfg.model.propensity_splits = 3
    cfg.economics.horizon = 6
    for attr in ("artifacts", "data", "models", "reports", "mlruns"):
        setattr(cfg.paths, attr, str(out / attr))
    cfg.paths.figures = str(out / "figures")
    cfg.paths.warehouse = str(out / "wh.duckdb")
    cfg.paths.ensure()

    pipe = Pipeline(cfg)
    dispatch = {
        "simulate": pipe.step_simulate,
        "ingest": pipe.step_ingest,
        "features": pipe.step_features,
        "train": pipe.step_train,
        "causal": pipe.step_causal,
        "evaluate": pipe.step_evaluate,
        "optimize": pipe.step_optimize,
        "monitor": pipe.step_monitor,
        "report": pipe.step_report,
    }
    for name in STEPS:
        pipe.run_step(name, dispatch[name])
    return pipe, cfg


def test_every_step_succeeded(pipeline_run):
    """A partial run is a failed run: name the steps that broke."""
    pipe, _ = pipeline_run
    failed = [(r.name, r.error) for r in pipe.results if not r.ok]
    assert not failed, "pipeline steps failed:\n" + "\n".join(f"  {n}: {e}" for n, e in failed)


def test_required_reports_exist_and_are_non_empty(pipeline_run):
    """The dashboard and README read these by name."""
    _, cfg = pipeline_run
    reports = Path(cfg.paths.path("reports"))
    missing = [f for f in REQUIRED_REPORTS if not (reports / f).exists()]
    assert not missing, f"missing reports: {missing}"
    empty = [f for f in REQUIRED_REPORTS if (reports / f).stat().st_size < 50]
    assert not empty, f"reports written but effectively empty: {empty}"


def test_required_figures_exist(pipeline_run):
    """Figures are what a reviewer actually looks at first."""
    _, cfg = pipeline_run
    figdir = Path(cfg.paths.path("figures"))
    missing = [f for f in REQUIRED_FIGURES if not (figdir / f).exists()]
    assert not missing, f"missing figures: {missing}"
    tiny = [f for f in REQUIRED_FIGURES if (figdir / f).stat().st_size < 3_000]
    assert not tiny, f"figures look like blank placeholders: {tiny}"


def test_metrics_json_carries_the_headline_claims(pipeline_run):
    """`scripts/check_metrics.py` gates CI on these keys; they must be produced."""
    _, cfg = pipeline_run
    metrics = json.loads((Path(cfg.paths.path("reports")) / "metrics.json").read_text(encoding="utf-8"))

    required = [
        "n_rows_panel",
        "contract_passed",
        "leakage_detected",
        "best_learner",
        "best_learner_pehe",
        "policy_value_optimised",
        "policy_value_highest_risk",
        "policy_uplift_vs_risk_targeting",
    ]
    missing = [k for k in required if k not in metrics]
    assert not missing, f"metrics.json is missing headline keys: {missing}"

    assert metrics["contract_passed"] is True, "the gold panel failed its data contract"
    assert metrics["leakage_detected"] is False, "point-in-time leakage was detected"
    assert metrics["n_rows_panel"] > 0


def test_serving_bundle_loads_and_scores(pipeline_run):
    """The artifact the API depends on must round-trip and actually produce a recommendation."""
    import pandas as pd

    from prism.data.schema import FEATURES
    from prism.serving.api import ModelBundle

    _, cfg = pipeline_run
    path = Path(cfg.paths.path("models")) / "bundle.joblib"
    assert path.exists(), "the pipeline did not write a serving bundle"

    bundle = ModelBundle.load(path)
    assert bundle.encoder is not None, "the bundle carries no fitted encoder; serving would skew"

    row = {"customer_id": "T1"}
    row.update({f: 1.0 for f in FEATURES if f not in ("plan_tier", "channel", "region", "device", "is_autopay", "contract_type")})
    row.update(
        {"plan_tier": "plus", "channel": "paid", "region": "north",
         "device": "ios", "is_autopay": "yes", "contract_type": "monthly"}
    )
    scored = bundle.score(pd.DataFrame([row]))

    assert len(scored) == 1
    assert "recommended_arm" in scored.columns
    assert 0 <= int(scored["recommended_arm"].iloc[0]) < 4
    assert float(scored["expected_net_value"].iloc[0]) >= 0.0, "doing nothing is always available"


def test_api_serves_the_trained_bundle(pipeline_run):
    """With a bundle on disk the API must return 200, not the degraded 503."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import prism.serving.api as api_mod
    from prism.serving.schemas import _EXAMPLE_CUSTOMER

    _, cfg = pipeline_run
    bundle_path = Path(cfg.paths.path("models")) / "bundle.joblib"

    api_mod._BUNDLE = None  # force a reload against this run's bundle
    api_mod.get_bundle(bundle_path, force=True)
    try:
        client = TestClient(api_mod.app)
        health = client.get("/health").json()
        assert health["status"] == "ok", f"API is degraded with a bundle present: {health}"

        resp = client.post("/score", json={"customers": [_EXAMPLE_CUSTOMER]})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["n_scored"] == 1
        arms = body["scores"][0]["arms"]
        assert len(arms) == 4
        assert arms[0]["net_value"] == 0.0, "the do-nothing arm must have exactly zero net value"

        pol = client.post("/policy", json={"customers": [_EXAMPLE_CUSTOMER], "budget": 100.0})
        assert pol.status_code == 200, pol.text
        assert pol.json()["total_cost"] <= 100.0 + 1e-6
    finally:
        api_mod._BUNDLE = None


def test_budget_is_never_exceeded_in_the_real_run(pipeline_run):
    """The single most important operational guarantee."""
    pipe, _ = pipeline_run
    spend = pipe.metrics.get("policy_total_cost")
    budget = pipe.metrics.get("budget")
    assert spend is not None and budget is not None
    assert spend <= budget + 1e-6, f"allocation spent {spend:,.2f} against a budget of {budget:,.2f}"


def test_causal_policy_beats_risk_targeting_in_the_real_run(pipeline_run):
    """The project's headline business claim, on a real end-to-end run.

    Note this uses *estimated* effects, not the oracle, so it is the honest version of the
    claim: the full pipeline, with all its estimation error, still beats the conventional
    churn-model baseline.
    """
    pipe, _ = pipeline_run
    uplift = pipe.metrics.get("policy_uplift_vs_risk_targeting")
    assert uplift is not None, "the comparison was not computed"
    assert uplift > 0, (
        f"the causal policy did not beat risk targeting (gap {uplift:,.2f}). "
        "This is the project's central claim; investigate before shipping."
    )
