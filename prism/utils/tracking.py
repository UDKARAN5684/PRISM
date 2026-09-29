"""Experiment tracking.

A thin wrapper over MLflow that is a no-op when MLflow is absent, so no pipeline step ever
has to branch on whether tracking is available. Also provides :func:`log_figure` and
:func:`log_dataframe` helpers so artifacts land in both MLflow and the repo's
``artifacts/reports`` directory (the repo copy is what the README links to).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from prism.utils.logging import get_logger
from prism.utils.optional import HAS_MLFLOW

__all__ = ["ExperimentTracker", "NullTracker", "get_tracker"]

log = get_logger("tracking")

# MLflow >=3 prints an agent hint on import; silence it for clean pipeline logs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# MLflow >=3.10 refuses the local file store unless this is set. A file store is exactly what a
# portfolio project wants: `mlflow ui --backend-store-uri artifacts/mlruns` with no server to run.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


class NullTracker:
    """Tracking interface that records nothing. Used when MLflow is unavailable."""

    active: bool = False

    def __init__(self, reports_dir: str | Path = "artifacts/reports") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, float] = {}

    @contextmanager
    def run(self, run_name: str, nested: bool = False) -> Iterator[NullTracker]:
        """Context manager for one (no-op) run."""
        log.debug("tracking disabled; run %r not recorded", run_name)
        yield self

    def log_params(self, params: dict[str, Any]) -> None:
        """Record parameters locally so they still reach ``metrics.json``."""
        self._params.update({k: _scalar(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record finite numeric metrics locally; NaN/inf are dropped rather than propagated."""
        self._metrics.update(
            {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and float(v) == float(v)
                and abs(float(v)) != float("inf")
            }
        )

    def log_figure(self, fig: Any, name: str) -> Path:
        """Save a matplotlib figure into the figures directory and return its path."""
        out = Path("docs/figures") / name
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out

    def log_dataframe(self, df: pd.DataFrame, name: str) -> Path:
        """Save a DataFrame as CSV under the reports directory and return its path."""
        out = self.reports_dir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        return out

    def log_dict(self, payload: dict[str, Any], name: str) -> Path:
        """Save a dict as JSON under the reports directory."""
        out = self.reports_dir / name
        out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        return out

    def log_artifact(self, path: str | Path) -> None:
        """No-op; the file already lives in the repo."""

    def set_tags(self, tags: dict[str, Any]) -> None:
        """No-op."""

    @property
    def params(self) -> dict[str, Any]:
        """Parameters recorded so far."""
        return dict(self._params)

    @property
    def metrics(self) -> dict[str, float]:
        """Metrics recorded so far."""
        return dict(self._metrics)


class ExperimentTracker(NullTracker):
    """MLflow-backed tracker with a local mirror of every artifact."""

    active = True

    def __init__(
        self,
        experiment_name: str = "prism-causal-retention",
        tracking_uri: str | Path = "artifacts/mlruns",
        reports_dir: str | Path = "artifacts/reports",
    ) -> None:
        super().__init__(reports_dir)
        import mlflow

        self._mlflow = mlflow
        store = Path(tracking_uri).resolve()
        store.mkdir(parents=True, exist_ok=True)

        # MLflow >= 3.16 has a broken file store on Windows: create_run writes the run
        # directory and then immediately fails to read it back ("Run '...' not found").
        # SQLite is the backend MLflow itself now recommends, it needs no server, and
        # `mlflow ui --backend-store-uri sqlite:///artifacts/mlruns/mlflow.db` reads it.
        self.tracking_uri = f"sqlite:///{(store / 'mlflow.db').as_posix()}"
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name)
        except Exception:  # pragma: no cover - fall back to the file store
            self.tracking_uri = store.as_uri()
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    @contextmanager
    def run(self, run_name: str, nested: bool = False) -> Iterator[ExperimentTracker]:
        """Open an MLflow run; failures degrade to a warning rather than killing the pipeline."""
        try:
            with self._mlflow.start_run(run_name=run_name, nested=nested):
                yield self
        except Exception as exc:  # pragma: no cover - tracking must never break a run
            log.warning("MLflow run %r failed (%s); continuing untracked", run_name, exc)
            yield self

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow (values are stringified; MLflow caps them at 500 chars)."""
        super().log_params(params)
        try:
            self._mlflow.log_params({k: str(_scalar(v))[:480] for k, v in params.items()})
        except Exception as exc:  # pragma: no cover
            log.debug("log_params failed: %s", exc)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics to MLflow, skipping non-finite values."""
        super().log_metrics(metrics, step)
        clean = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) == float(v)
        }
        try:
            self._mlflow.log_metrics(clean, step=step)
        except Exception as exc:  # pragma: no cover
            log.debug("log_metrics failed: %s", exc)

    def log_figure(self, fig: Any, name: str) -> Path:
        """Save the figure locally and attach it to the active MLflow run."""
        out = super().log_figure(fig, name)
        try:
            self._mlflow.log_artifact(str(out), artifact_path="figures")
        except Exception as exc:  # pragma: no cover
            log.debug("log_figure failed: %s", exc)
        return out

    def log_dataframe(self, df: pd.DataFrame, name: str) -> Path:
        """Save the frame locally and attach it to the active MLflow run."""
        out = super().log_dataframe(df, name)
        try:
            self._mlflow.log_artifact(str(out), artifact_path="tables")
        except Exception as exc:  # pragma: no cover
            log.debug("log_dataframe failed: %s", exc)
        return out

    def log_dict(self, payload: dict[str, Any], name: str) -> Path:
        """Save the dict locally and attach it to the active MLflow run."""
        out = super().log_dict(payload, name)
        try:
            self._mlflow.log_artifact(str(out), artifact_path="reports")
        except Exception as exc:  # pragma: no cover
            log.debug("log_dict failed: %s", exc)
        return out

    def log_artifact(self, path: str | Path) -> None:
        """Attach an existing file to the active run."""
        try:
            self._mlflow.log_artifact(str(path))
        except Exception as exc:  # pragma: no cover
            log.debug("log_artifact failed: %s", exc)

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Set MLflow tags on the active run."""
        try:
            self._mlflow.set_tags({k: str(v)[:480] for k, v in tags.items()})
        except Exception as exc:  # pragma: no cover
            log.debug("set_tags failed: %s", exc)


def get_tracker(
    experiment_name: str = "prism-causal-retention",
    tracking_uri: str | Path = "artifacts/mlruns",
    reports_dir: str | Path = "artifacts/reports",
    enabled: bool = True,
) -> NullTracker:
    """Return an :class:`ExperimentTracker` if MLflow is usable, else a :class:`NullTracker`.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment to write into.
    tracking_uri : str or Path
        Local backing store for runs.
    reports_dir : str or Path
        Where the local mirror of tables and JSON lands.
    enabled : bool, default True
        Set False to force the null tracker (used by tests).

    Returns
    -------
    NullTracker
        Either a real tracker or the no-op one; the interface is identical.
    """
    if not enabled or not HAS_MLFLOW:
        return NullTracker(reports_dir)
    try:
        return ExperimentTracker(experiment_name, tracking_uri, reports_dir)
    except Exception as exc:  # pragma: no cover - broken mlflow install
        log.warning("MLflow unavailable (%s); continuing without experiment tracking", exc)
        return NullTracker(reports_dir)


def _scalar(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v)
    return v


def _jsonable(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if f == f and abs(f) != float("inf") else None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
        return None
    return obj


if __name__ == "__main__":  # pragma: no cover - smoke test
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = get_tracker(experiment_name="prism-smoke", reports_dir="artifacts/reports")
    print("tracker:", type(tr).__name__, "| active:", tr.active)
    with tr.run("smoke"):
        tr.log_params({"seed": 7, "learners": ["dr", "r"]})
        tr.log_metrics({"pehe": 1.234, "qini": 0.42, "nan_metric": float("nan")})
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1], [0, 1])
        p1 = tr.log_figure(fig, "_smoke.png")
        plt.close(fig)
        p2 = tr.log_dataframe(pd.DataFrame({"a": [1, 2]}), "_smoke.csv")
        p3 = tr.log_dict({"ok": True, "v": float("inf")}, "_smoke.json")
    for p in (p1, p2, p3):
        assert Path(p).exists(), p
        Path(p).unlink()
    print("metrics kept:", tr.metrics)
    print("tracking.py OK")
