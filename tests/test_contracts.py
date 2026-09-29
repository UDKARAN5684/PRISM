"""Interface-conformance tests.

Fourteen modules were built against `SPEC.md` independently. This file is the guard that
they actually compose: every public name the spec promises must exist, be callable, and
accept the parameters the rest of the pipeline passes it.

These tests are deliberately about *shape*, not behaviour — behaviour is covered in
`test_scientific_claims.py` and the per-module suites. A failure here means two modules
disagree about an interface, which is the failure mode parallel development actually has.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

# (module path, [required public names])
SPEC_SURFACE: dict[str, list[str]] = {
    "prism.utils.seeds": ["as_rng", "seed_everything"],
    "prism.utils.optional": ["require", "best_gbm", "HAS_TORCH", "HAS_LIGHTGBM"],
    "prism.utils.logging": ["get_logger"],
    "prism.utils.validation": [
        "Expectation",
        "ValidationResult",
        "DataContract",
        "DataContractError",
        "PANEL_CONTRACT",
    ],
    "prism.utils.tracking": ["get_tracker", "NullTracker", "ExperimentTracker"],
    "prism.config": ["PrismConfig", "load_config"],
    "prism.data.schema": [
        "N_ARMS",
        "ARM_NAMES",
        "ARM_COSTS",
        "NUMERIC_FEATURES",
        "CATEGORICAL_FEATURES",
        "FEATURES",
        "CATEGORICAL_LEVELS",
        "PANEL_COLUMNS",
        "GT_COLUMNS",
        "enforce_schema",
    ],
    "prism.data.dgp": ["DGPConfig", "SimulatedData", "simulate", "N_ARMS", "ARM_NAMES", "ARM_COSTS"],
    "prism.data.messy": ["inject_realism", "clean_panel"],
    "prism.data.warehouse": ["Warehouse"],
    "prism.data.features": [
        "FeatureSpec",
        "PointInTimeFeatureStore",
        "temporal_split",
        "build_design_matrix",
        "assert_no_leakage",
        "customer_summary",
    ],
    "prism.data.real": ["load_real", "map_to_panel"],
    "prism.models.survival": [
        "DiscreteTimeHazardModel",
        "CoxPHModel",
        "RandomSurvivalForestLite",
        "concordance_index",
        "brier_score",
        "integrated_brier_score",
    ],
    "prism.models.clv": ["BGNBD", "GammaGamma", "probabilistic_clv", "DeepCLV", "clv_from_survival"],
    "prism.models.propensity": [
        "PropensityModel",
        "overlap_diagnostics",
        "standardized_mean_differences",
        "trim_by_overlap",
        "effective_sample_size",
    ],
    "prism.causal.learners": ["SLearner", "TLearner", "XLearner", "DRLearner", "RLearner", "make_learner"],
    "prism.causal.forest": ["CausalForest"],
    "prism.causal.survival_uplift": ["CausalSurvivalUplift", "ipcw_weights", "rmst_pseudo_outcome"],
    "prism.causal.evaluate": [
        "qini_curve",
        "qini_score",
        "auuc",
        "uplift_at_k",
        "uplift_by_decile",
        "pehe",
        "eps_ate",
        "policy_value",
        "gates",
        "blp_calibration",
        "compare_learners",
    ],
    "prism.causal.refute": [
        "RefutationResult",
        "placebo_treatment",
        "random_common_cause",
        "subset_refuter",
        "e_value",
        "run_refutation_suite",
    ],
    "prism.decision.economics": ["EconomicConfig", "expected_net_value", "build_decision_frame"],
    "prism.decision.optimize": [
        "AllocationResult",
        "greedy_knapsack",
        "lagrangian_allocate",
        "lp_allocate",
        "efficient_frontier",
        "baseline_policies",
    ],
    "prism.decision.policy_eval": ["evaluate_policy", "compare_policies", "oracle_gap"],
    "prism.decision.fairness": ["policy_fairness_audit", "disparate_impact_ratio", "reweight_for_parity"],
    "prism.monitoring.drift": [
        "psi",
        "ks_statistic",
        "js_divergence",
        "feature_drift_report",
        "cate_stability",
        "performance_decay",
    ],
    "prism.monitoring.alerts": ["Alert", "evaluate_alerts", "render_alerts"],
    "prism.serving.schemas": ["CustomerFeatures", "ScoreRequest", "ScoreResponse", "CustomerScore"],
    "prism.serving.api": ["app", "ModelBundle"],
    "prism.pipelines.run_all": ["main"],
}

#: Required parameters for functions whose call sites live in another module. If one of these
#: is renamed, the pipeline breaks at runtime rather than at import, so they are pinned here.
REQUIRED_PARAMS: dict[str, dict[str, list[str]]] = {
    "prism.data.features": {
        "temporal_split": ["panel", "train_end", "valid_end"],
        "build_design_matrix": ["panel"],
    },
    "prism.causal.evaluate": {
        "pehe": ["true_cate", "est_cate"],
        "qini_score": ["y", "w", "cate"],
        "policy_value": ["y", "w", "policy_arm", "propensity"],
    },
    "prism.decision.economics": {
        "expected_net_value": ["value_cate", "econ"],
    },
    "prism.decision.optimize": {
        "greedy_knapsack": ["net_value", "cost", "budget"],
        "lagrangian_allocate": ["net_value", "cost", "budget"],
        "efficient_frontier": ["net_value", "cost", "budgets"],
    },
    "prism.monitoring.drift": {
        "psi": ["expected", "actual"],
        "cate_stability": ["cate_ref", "cate_cur"],
    },
}

#: Estimators that must all share the CATE learner interface.
CATE_LEARNERS = [
    ("prism.causal.learners", "SLearner"),
    ("prism.causal.learners", "TLearner"),
    ("prism.causal.learners", "XLearner"),
    ("prism.causal.learners", "DRLearner"),
    ("prism.causal.learners", "RLearner"),
    ("prism.causal.forest", "CausalForest"),
]


@pytest.mark.parametrize("module_path", sorted(SPEC_SURFACE))
def test_module_imports_without_side_effects(module_path):
    """Importing any module must not crash, and must not need an optional dependency."""
    importlib.import_module(module_path)


@pytest.mark.parametrize(
    ("module_path", "name"),
    [(m, n) for m, names in SPEC_SURFACE.items() for n in names],
)
def test_spec_name_exists(module_path, name):
    """Every public name promised by SPEC.md must actually be exported."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, name), f"{module_path} does not export {name!r} (required by SPEC.md)"


@pytest.mark.parametrize(
    ("module_path", "func_name", "params"),
    [(m, f, p) for m, funcs in REQUIRED_PARAMS.items() for f, p in funcs.items()],
)
def test_required_parameters_are_accepted(module_path, func_name, params):
    """Cross-module call sites depend on these parameter names; a rename breaks the pipeline."""
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name)
    sig = inspect.signature(fn)
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    for p in params:
        assert p in sig.parameters or accepts_kwargs, (
            f"{module_path}.{func_name} does not accept {p!r}; signature is {sig}"
        )


@pytest.mark.parametrize(("module_path", "cls_name"), CATE_LEARNERS)
def test_cate_learners_share_one_interface(module_path, cls_name):
    """All CATE estimators must be interchangeable: same fit and predict_cate surface.

    The pipeline loops over learners and calls them uniformly, so one divergent signature
    breaks the leaderboard.
    """
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)

    assert hasattr(cls, "fit"), f"{cls_name} has no fit()"
    assert hasattr(cls, "predict_cate"), f"{cls_name} has no predict_cate()"

    fit_sig = inspect.signature(cls.fit)
    for p in ("X", "w", "y"):
        assert p in fit_sig.parameters, f"{cls_name}.fit is missing {p!r}; signature is {fit_sig}"

    predict_sig = inspect.signature(cls.predict_cate)
    assert "X" in predict_sig.parameters, f"{cls_name}.predict_cate is missing 'X'"


def test_make_learner_builds_every_advertised_learner():
    """make_learner is the factory the pipeline uses; every advertised key must work."""
    from prism.causal.learners import make_learner

    for name in ("s", "t", "x", "dr", "r"):
        obj = make_learner(name)
        assert hasattr(obj, "fit") and hasattr(obj, "predict_cate"), f"make_learner({name!r}) is not a CATE learner"


def test_arm_constants_agree_across_modules():
    """A mismatch in arm count or cost between schema, dgp and economics is silently catastrophic."""
    from prism.data import dgp, schema
    from prism.decision.economics import EconomicConfig

    assert dgp.N_ARMS == schema.N_ARMS
    assert tuple(dgp.ARM_NAMES) == tuple(schema.ARM_NAMES)
    assert tuple(dgp.ARM_COSTS) == tuple(schema.ARM_COSTS)

    econ = EconomicConfig()
    assert len(econ.arm_costs) == schema.N_ARMS
    assert tuple(econ.arm_costs) == tuple(schema.ARM_COSTS)
    assert econ.arm_costs[0] == 0.0, "the control arm must be free"


def test_feature_lists_are_consistent():
    """The 23-feature block is referenced by the simulator, the encoder and the API schema."""
    from prism.data.schema import CATEGORICAL_FEATURES, CATEGORICAL_LEVELS, FEATURES, NUMERIC_FEATURES

    assert len(NUMERIC_FEATURES) == 17
    assert len(CATEGORICAL_FEATURES) == 6
    assert tuple(FEATURES) == tuple(NUMERIC_FEATURES) + tuple(CATEGORICAL_FEATURES)
    assert len(set(FEATURES)) == len(FEATURES), "duplicate feature name"
    assert set(CATEGORICAL_LEVELS) == set(CATEGORICAL_FEATURES)


def test_api_schema_covers_every_feature():
    """The serving contract must accept exactly the features the model was trained on."""
    pydantic = pytest.importorskip("pydantic")
    from prism.data.schema import FEATURES
    from prism.serving.schemas import CustomerFeatures

    assert issubclass(CustomerFeatures, pydantic.BaseModel)
    fields = set(CustomerFeatures.model_fields)
    missing = set(FEATURES) - fields
    assert not missing, f"the API schema is missing trained features: {sorted(missing)}"


def test_every_module_has_a_docstring():
    """A module without a docstring is a module nobody will read."""
    undocumented = []
    for module_path in SPEC_SURFACE:
        mod = importlib.import_module(module_path)
        if not (mod.__doc__ or "").strip():
            undocumented.append(module_path)
    assert not undocumented, f"modules missing a docstring: {undocumented}"


@pytest.mark.parametrize("module_path", sorted(SPEC_SURFACE))
def test_public_callables_are_documented(module_path):
    """Public functions and classes defined in the module must carry a docstring."""
    mod = importlib.import_module(module_path)
    undocumented = []
    for name, obj in vars(mod).items():
        if name.startswith("_"):
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if getattr(obj, "__module__", None) != module_path:
            continue  # re-exported from elsewhere
        if not (obj.__doc__ or "").strip():
            undocumented.append(name)
    assert not undocumented, f"{module_path} has undocumented public names: {undocumented}"
