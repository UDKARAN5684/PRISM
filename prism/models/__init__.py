"""Predictive models: survival, customer lifetime value and multi-arm propensity.

Names are resolved lazily (PEP 562) so that importing this package costs nothing until a
specific symbol is used. This keeps ``import prism.data`` fast while still allowing the
convenient ``from prism.data import Warehouse`` form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, str] = {
    "BGNBD": "clv",
    "CoxPHModel": "survival",
    "DeepCLV": "clv",
    "DiscreteTimeHazardModel": "survival",
    "GammaGamma": "clv",
    "PropensityModel": "propensity",
    "RandomSurvivalForestLite": "survival",
    "clv_from_survival": "clv",
    "concordance_index": "survival",
    "effective_sample_size": "propensity",
    "integrated_brier_score": "survival",
    "overlap_diagnostics": "propensity",
    "probabilistic_clv": "clv",
    "standardized_mean_differences": "propensity",
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
    from prism.models.clv import BGNBD, DeepCLV, GammaGamma, clv_from_survival, probabilistic_clv  # noqa: F401
    from prism.models.propensity import (  # noqa: F401
        PropensityModel,
        effective_sample_size,
        overlap_diagnostics,
        standardized_mean_differences,
    )
    from prism.models.survival import (  # noqa: F401
        CoxPHModel,
        DiscreteTimeHazardModel,
        RandomSurvivalForestLite,
        concordance_index,
        integrated_brier_score,
    )
