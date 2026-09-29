"""Data layer: ground-truth simulator, realism injection, medallion warehouse,
point-in-time feature store and real-dataset adapters.

Names are resolved lazily (PEP 562) so that importing this package costs nothing until a
specific symbol is used. This keeps ``import prism.data`` fast while still allowing the
convenient ``from prism.data import Warehouse`` form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, str] = {
    "ARM_COSTS": "schema",
    "ARM_NAMES": "schema",
    "ARM_REDEMPTION": "schema",
    "CATEGORICAL_FEATURES": "schema",
    "CATEGORICAL_LEVELS": "schema",
    "DGPConfig": "dgp",
    "FEATURES": "schema",
    "FeatureSpec": "features",
    "GT_COLUMNS": "schema",
    "NUMERIC_FEATURES": "schema",
    "N_ARMS": "schema",
    "PANEL_COLUMNS": "schema",
    "PointInTimeFeatureStore": "features",
    "SEGMENTS": "schema",
    "SimulatedData": "dgp",
    "Warehouse": "warehouse",
    "arm_name": "schema",
    "assert_no_leakage": "features",
    "build_design_matrix": "features",
    "clean_panel": "messy",
    "customer_summary": "features",
    "describe_datasets": "real",
    "enforce_schema": "schema",
    "inject_realism": "messy",
    "load_real": "real",
    "map_to_panel": "real",
    "simulate": "dgp",
    "temporal_split": "features",
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
    from prism.data.dgp import DGPConfig, SimulatedData, simulate  # noqa: F401
    from prism.data.features import (  # noqa: F401
        FeatureSpec,
        PointInTimeFeatureStore,
        assert_no_leakage,
        build_design_matrix,
        customer_summary,
        temporal_split,
    )
    from prism.data.messy import clean_panel, inject_realism  # noqa: F401
    from prism.data.real import describe_datasets, load_real, map_to_panel  # noqa: F401
    from prism.data.schema import (  # noqa: F401
        ARM_COSTS,
        ARM_NAMES,
        ARM_REDEMPTION,
        CATEGORICAL_FEATURES,
        CATEGORICAL_LEVELS,
        FEATURES,
        GT_COLUMNS,
        N_ARMS,
        NUMERIC_FEATURES,
        PANEL_COLUMNS,
        SEGMENTS,
        arm_name,
        enforce_schema,
    )
    from prism.data.warehouse import Warehouse  # noqa: F401
