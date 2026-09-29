"""Decision layer: unit economics, budget-constrained multi-arm assignment,
off-policy evaluation and the fairness audit.

Names are resolved lazily (PEP 562) so that importing this package costs nothing until a
specific symbol is used. This keeps ``import prism.data`` fast while still allowing the
convenient ``from prism.data import Warehouse`` form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, str] = {
    "AllocationResult": "optimize",
    "EconomicConfig": "economics",
    "baseline_policies": "optimize",
    "build_decision_frame": "economics",
    "compare_policies": "policy_eval",
    "disparate_impact_ratio": "fairness",
    "efficient_frontier": "optimize",
    "evaluate_policy": "policy_eval",
    "expected_net_value": "economics",
    "greedy_knapsack": "optimize",
    "lagrangian_allocate": "optimize",
    "lp_allocate": "optimize",
    "oracle_gap": "policy_eval",
    "policy_fairness_audit": "fairness",
    "reweight_for_parity": "fairness",
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
    from prism.decision.economics import EconomicConfig, build_decision_frame, expected_net_value  # noqa: F401
    from prism.decision.fairness import disparate_impact_ratio, policy_fairness_audit, reweight_for_parity  # noqa: F401
    from prism.decision.optimize import (  # noqa: F401
        AllocationResult,
        baseline_policies,
        efficient_frontier,
        greedy_knapsack,
        lagrangian_allocate,
        lp_allocate,
    )
    from prism.decision.policy_eval import compare_policies, evaluate_policy, oracle_gap  # noqa: F401
