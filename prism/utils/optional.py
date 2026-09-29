"""Optional-dependency gatekeeping.

PRISM has a small hard dependency set (numpy/pandas/scipy/sklearn/statsmodels/matplotlib)
and a larger set of accelerators. Nothing in this package may fail at *import* time because
an accelerator is absent; instead we expose ``HAS_*`` flags and a :func:`require` helper
that raises a helpful message only at the point of use.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

__all__ = [
    "HAS_LIGHTGBM",
    "HAS_XGBOOST",
    "HAS_TORCH",
    "HAS_DUCKDB",
    "HAS_SHAP",
    "HAS_LIFELINES",
    "HAS_MLFLOW",
    "HAS_PLOTLY",
    "HAS_OPTUNA",
    "HAS_YAML",
    "HAS_RICH",
    "HAS_FASTAPI",
    "HAS_STREAMLIT",
    "require",
    "best_gbm",
    "availability",
]

_INSTALL_HINTS = {
    "lightgbm": "pip install lightgbm",
    "xgboost": "pip install xgboost",
    "torch": "pip install torch",
    "duckdb": "pip install duckdb",
    "shap": "pip install shap",
    "lifelines": "pip install lifelines",
    "mlflow": "pip install mlflow",
    "plotly": "pip install plotly",
    "optuna": "pip install optuna",
    "yaml": "pip install pyyaml",
    "rich": "pip install rich",
    "fastapi": "pip install 'fastapi[standard]'",
    "streamlit": "pip install streamlit",
}


def _probe(name: str) -> bool:
    """Report whether a package is installed WITHOUT importing it.

    ``import_module`` here would be a serious mistake: this module is imported by almost every
    other one, so probing with a real import would eagerly pull in torch, shap, mlflow,
    lifelines, optuna and streamlit on every ``python -m prism.*`` invocation -- measured at
    roughly 10 seconds of pure startup cost, charged against every smoke test and every CLI
    run. ``find_spec`` answers the same question in microseconds by consulting the import
    machinery's finders without executing the module.

    The tradeoff: a package that is installed but broken (a bad C extension, say) reports
    ``True`` here and only fails later at :func:`require`. That is the right place for it to
    fail, with a real traceback, rather than being silently downgraded at import time.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError, ModuleNotFoundError):
        return False


HAS_LIGHTGBM = _probe("lightgbm")
HAS_XGBOOST = _probe("xgboost")
HAS_TORCH = _probe("torch")
HAS_DUCKDB = _probe("duckdb")
HAS_SHAP = _probe("shap")
HAS_LIFELINES = _probe("lifelines")
HAS_MLFLOW = _probe("mlflow")
HAS_PLOTLY = _probe("plotly")
HAS_OPTUNA = _probe("optuna")
HAS_YAML = _probe("yaml")
HAS_RICH = _probe("rich")
HAS_FASTAPI = _probe("fastapi")
HAS_STREAMLIT = _probe("streamlit")


def require(pkg: str) -> ModuleType:
    """Import an optional package or raise with an actionable install hint.

    Parameters
    ----------
    pkg : str
        Importable module name, e.g. ``"lightgbm"``.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If the package is unavailable, carrying the pip command that would fix it.
    """
    try:
        return importlib.import_module(pkg)
    except ImportError as exc:  # pragma: no cover - environment dependent
        hint = _INSTALL_HINTS.get(pkg, f"pip install {pkg}")
        raise ImportError(f"PRISM needs the optional package '{pkg}' here. Install it with: {hint}") from exc


def best_gbm(task: str = "regression", **kwargs):
    """Return the fastest available gradient-boosting estimator for ``task``.

    Preference order LightGBM > XGBoost > sklearn ``HistGradientBoosting``. All three are
    sklearn-compatible, so downstream code never branches on which one it got.

    Parameters
    ----------
    task : {"regression", "classification"}
        Learning task.
    **kwargs
        Overrides forwarded to the underlying estimator. Recognised portable keys are
        ``n_estimators``, ``learning_rate``, ``max_depth``, ``min_samples_leaf`` and
        ``random_state``; unknown keys are passed through untouched.

    Returns
    -------
    object
        An unfitted sklearn-compatible estimator.
    """
    if task not in {"regression", "classification"}:
        raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")

    n_estimators = kwargs.pop("n_estimators", 300)
    learning_rate = kwargs.pop("learning_rate", 0.05)
    max_depth = kwargs.pop("max_depth", -1)
    min_samples_leaf = kwargs.pop("min_samples_leaf", 20)
    random_state = kwargs.pop("random_state", 0)
    if random_state is not None:
        random_state = int(random_state) % (2**31 - 1)

    if HAS_LIGHTGBM:
        import lightgbm as lgb

        params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_samples_leaf,
            num_leaves=31,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        params.update(kwargs)
        return lgb.LGBMRegressor(**params) if task == "regression" else lgb.LGBMClassifier(**params)

    if HAS_XGBOOST:
        import xgboost as xgb

        params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6 if max_depth in (-1, None) else max_depth,
            min_child_weight=max(1, min_samples_leaf // 4),
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
        params.update(kwargs)
        return xgb.XGBRegressor(**params) if task == "regression" else xgb.XGBClassifier(**params)

    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    params = dict(
        max_iter=n_estimators,
        learning_rate=learning_rate,
        max_depth=None if max_depth in (-1, None) else max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=1.0,
        random_state=random_state,
    )
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params) if task == "regression" else HistGradientBoostingClassifier(**params)


def availability() -> dict[str, bool]:
    """Return the full optional-dependency map, for logging and the ``/metadata`` endpoint."""
    return {
        "lightgbm": HAS_LIGHTGBM,
        "xgboost": HAS_XGBOOST,
        "torch": HAS_TORCH,
        "duckdb": HAS_DUCKDB,
        "shap": HAS_SHAP,
        "lifelines": HAS_LIFELINES,
        "mlflow": HAS_MLFLOW,
        "plotly": HAS_PLOTLY,
        "optuna": HAS_OPTUNA,
        "yaml": HAS_YAML,
        "rich": HAS_RICH,
        "fastapi": HAS_FASTAPI,
        "streamlit": HAS_STREAMLIT,
    }


if __name__ == "__main__":  # pragma: no cover - smoke test
    import numpy as np

    for name, ok in availability().items():
        print(f"  {name:12s} {'yes' if ok else 'no'}")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=300)
    m = best_gbm("regression", n_estimators=40, random_state=0).fit(X, y)
    c = best_gbm("classification", n_estimators=40, random_state=0).fit(X, (y > 0).astype(int))
    print("optional.py OK -> gbm r2 =", round(float(np.corrcoef(m.predict(X), y)[0, 1] ** 2), 4),
          "| clf acc =", round(float((c.predict(X) == (y > 0).astype(int)).mean()), 4))
