"""Monitoring layer: covariate, prediction and CATE drift, plus alerting.

Names are resolved lazily (PEP 562) so that importing this package costs nothing until a
specific symbol is used. This keeps ``import prism.data`` fast while still allowing the
convenient ``from prism.data import Warehouse`` form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, str] = {
    "Alert": "alerts",
    "cate_stability": "drift",
    "evaluate_alerts": "alerts",
    "feature_drift_report": "drift",
    "js_divergence": "drift",
    "ks_statistic": "drift",
    "performance_decay": "drift",
    "psi": "drift",
    "render_alerts": "alerts",
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
    from prism.monitoring.alerts import Alert, evaluate_alerts, render_alerts  # noqa: F401
    from prism.monitoring.drift import (  # noqa: F401
        cate_stability,
        feature_drift_report,
        js_divergence,
        ks_statistic,
        performance_decay,
        psi,
    )
