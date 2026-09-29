"""Causal layer: meta-learners, honest causal forest, Causal Survival Uplift,
uplift and policy evaluation, refutation and sensitivity analysis.

Names are resolved lazily (PEP 562) so that importing this package costs nothing until a
specific symbol is used. This keeps ``import prism.data`` fast while still allowing the
convenient ``from prism.data import Warehouse`` form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, str] = {
    "CausalForest": "forest",
    "CausalSurvivalUplift": "survival_uplift",
    "DRLearner": "learners",
    "RLearner": "learners",
    "SLearner": "learners",
    "TLearner": "learners",
    "XLearner": "learners",
    "auuc": "evaluate",
    "blp_calibration": "evaluate",
    "compare_learners": "evaluate",
    "e_value": "refute",
    "eps_ate": "evaluate",
    "gates": "evaluate",
    "make_learner": "learners",
    "pehe": "evaluate",
    "placebo_treatment": "refute",
    "policy_value": "evaluate",
    "qini_curve": "evaluate",
    "qini_score": "evaluate",
    "run_refutation_suite": "refute",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import the submodule that owns ``name`` on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:  # pragma: no cover - static analysers only
    from prism.causal.evaluate import (  # noqa: F401
        auuc,
        blp_calibration,
        compare_learners,
        eps_ate,
        gates,
        pehe,
        policy_value,
        qini_curve,
        qini_score,
    )
    from prism.causal.forest import CausalForest  # noqa: F401
    from prism.causal.learners import DRLearner, RLearner, SLearner, TLearner, XLearner, make_learner  # noqa: F401
    from prism.causal.refute import e_value, placebo_treatment, run_refutation_suite  # noqa: F401
    from prism.causal.survival_uplift import CausalSurvivalUplift  # noqa: F401
