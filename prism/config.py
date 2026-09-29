"""Typed, YAML-backed configuration for the whole PRISM pipeline.

One config object drives simulation, training, causal estimation, budget optimisation,
monitoring and reporting, so a run is fully described by a single file plus a seed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from prism.data.schema import ARM_COSTS, ARM_REDEMPTION

__all__ = [
    "PathsConfig",
    "SimConfig",
    "SplitConfig",
    "ModelConfig",
    "CausalConfig",
    "EconomicsConfig",
    "MonitoringConfig",
    "PrismConfig",
    "load_config",
    "PROJECT_ROOT",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(p: str | Path) -> Path:
    """Resolve ``p`` relative to the project root unless it is already absolute."""
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass
class PathsConfig:
    """Filesystem layout. Relative paths are resolved against the project root."""

    artifacts: str = "artifacts"
    data: str = "artifacts/data"
    models: str = "artifacts/models"
    reports: str = "artifacts/reports"
    figures: str = "docs/figures"
    warehouse: str = "artifacts/warehouse.duckdb"
    mlruns: str = "artifacts/mlruns"

    def ensure(self) -> PathsConfig:
        """Create every directory in this config (idempotent)."""
        for name in ("artifacts", "data", "models", "reports", "figures", "mlruns"):
            _resolve(getattr(self, name)).mkdir(parents=True, exist_ok=True)
        _resolve(self.warehouse).parent.mkdir(parents=True, exist_ok=True)
        return self

    def path(self, name: str) -> Path:
        """Return an absolute :class:`~pathlib.Path` for the named location."""
        return _resolve(getattr(self, name))


@dataclass
class SimConfig:
    """Parameters of the ground-truth data-generating process (see ``prism.data.dgp``)."""

    n_customers: int = 60_000
    n_periods: int = 24
    horizon: int = 12
    origin: str = "2022-01-01"
    monthly_discount_rate: float = 0.01
    rct_fraction: float = 0.15
    confounding_strength: float = 1.0
    hidden_confounder_strength: float = 0.05
    drift_strength: float = 0.6
    seasonality_amplitude: float = 0.25
    missing_rate: float = 0.08
    mnar_strength: float = 0.5
    outlier_rate: float = 0.004
    duplicate_rate: float = 0.002
    random_state: int = 7


@dataclass
class SplitConfig:
    """Temporal split boundaries, expressed as period indices."""

    train_end: int = 13
    valid_end: int = 17


@dataclass
class ModelConfig:
    """Predictive-model hyper-parameters (survival, CLV, propensity)."""

    survival_backend: str = "auto"
    survival_hidden: tuple[int, ...] = (64, 32)
    survival_epochs: int = 40
    survival_lr: float = 1e-3
    clv_backend: str = "auto"
    clv_epochs: int = 40
    propensity_splits: int = 5
    propensity_clip: tuple[float, float] = (0.01, 0.99)
    n_estimators: int = 300
    learning_rate: float = 0.05
    tune: bool = False
    n_trials: int = 20


@dataclass
class CausalConfig:
    """Causal-estimation settings."""

    learners: tuple[str, ...] = ("s", "t", "x", "dr", "r", "forest", "survival")
    n_splits: int = 5
    forest_trees: int = 400
    forest_min_leaf: int = 15
    honest_fraction: float = 0.5
    doubly_robust: bool = True
    outcome: str = "value"  # "value" | "rmst" | "churn"
    refute: bool = True
    refute_sims: int = 10


@dataclass
class EconomicsConfig:
    """Unit economics that convert a causal effect into money."""

    arm_costs: tuple[float, ...] = ARM_COSTS
    offer_redemption_rate: tuple[float, ...] = ARM_REDEMPTION
    margin_rate: float = 0.30
    discount_rate: float = 0.01
    horizon: int = 12
    fixed_campaign_cost: float = 0.0
    budget: float = 150_000.0
    budget_grid: tuple[float, ...] = (0.0, 25_000.0, 50_000.0, 100_000.0, 150_000.0, 250_000.0, 400_000.0, 750_000.0)


@dataclass
class MonitoringConfig:
    """Drift and decay alert thresholds."""

    psi_warn: float = 0.10
    psi_alert: float = 0.25
    ks_alert_p: float = 0.01
    cate_rank_corr_alert: float = 0.70
    sign_flip_alert: float = 0.15


@dataclass
class PrismConfig:
    """Root configuration object."""

    seed: int = 7
    experiment_name: str = "prism-causal-retention"
    paths: PathsConfig = field(default_factory=PathsConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serialisable dict of the whole config."""
        return _to_plain(asdict(self))

    def save(self, path: str | Path) -> None:
        """Write the config to ``.yaml`` or ``.json`` (chosen by suffix)."""
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if p.suffix in {".yaml", ".yml"}:
            from prism.utils.optional import require

            require("yaml").safe_dump(payload, p.open("w", encoding="utf-8"), sort_keys=False)
        else:
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def dgp_config(self):
        """Build a :class:`prism.data.dgp.DGPConfig` from :attr:`sim` plus the global seed."""
        from prism.data.dgp import DGPConfig

        kwargs = asdict(self.sim)
        kwargs["random_state"] = self.seed
        valid = {f.name for f in fields(DGPConfig)}
        return DGPConfig(**{k: v for k, v in kwargs.items() if k in valid})

    def economic_config(self):
        """Build a :class:`prism.decision.economics.EconomicConfig` from :attr:`economics`."""
        from prism.decision.economics import EconomicConfig

        e = self.economics
        return EconomicConfig(
            arm_costs=tuple(e.arm_costs),
            margin_rate=e.margin_rate,
            discount_rate=e.discount_rate,
            horizon=e.horizon,
            offer_redemption_rate=tuple(e.offer_redemption_rate),
            fixed_campaign_cost=e.fixed_campaign_cost,
        )


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _coerce(cls: type, payload: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict."""
    if not is_dataclass(cls):
        return payload
    kwargs: dict[str, Any] = {}
    type_map = {f.name: f for f in fields(cls)}
    for key, value in (payload or {}).items():
        if key not in type_map:
            continue
        f = type_map[key]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[key] = _coerce(f.type, value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    # nested dataclass fields declared as strings (from __future__ annotations)
    for name, f in type_map.items():
        if name in kwargs:
            continue
        if isinstance(f.type, str) and f.type.endswith("Config") and name in (payload or {}):
            sub = globals().get(f.type)
            if sub is not None:
                kwargs[name] = _coerce(sub, payload[name])
    return cls(**kwargs)


_SECTIONS = {
    "paths": PathsConfig,
    "sim": SimConfig,
    "split": SplitConfig,
    "model": ModelConfig,
    "causal": CausalConfig,
    "economics": EconomicsConfig,
    "monitoring": MonitoringConfig,
}


def load_config(path: str | Path | None = None, **overrides: Any) -> PrismConfig:
    """Load a :class:`PrismConfig` from YAML/JSON, applying keyword overrides.

    Parameters
    ----------
    path : str or Path or None
        Config file. ``None`` returns defaults.
    **overrides
        Top-level overrides, e.g. ``seed=1``. Nested sections accept dicts, e.g.
        ``sim={"n_customers": 5000}``.

    Returns
    -------
    PrismConfig
        A fully-populated config with its directories created.
    """
    payload: dict[str, Any] = {}
    if path is not None:
        p = _resolve(path)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            if p.suffix in {".yaml", ".yml"}:
                from prism.utils.optional import require

                payload = require("yaml").safe_load(text) or {}
            else:
                payload = json.loads(text)

    cfg = PrismConfig(
        seed=int(payload.get("seed", 7)),
        experiment_name=payload.get("experiment_name", "prism-causal-retention"),
        **{name: _coerce(cls, payload.get(name, {})) for name, cls in _SECTIONS.items()},
    )

    for key, value in overrides.items():
        if key in _SECTIONS and isinstance(value, dict):
            section = getattr(cfg, key)
            for k, v in value.items():
                if hasattr(section, k):
                    setattr(section, k, v)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise KeyError(f"unknown config key {key!r}")

    # the global seed is authoritative for the simulator
    cfg.sim.random_state = cfg.seed
    cfg.economics.horizon = cfg.sim.horizon
    cfg.economics.discount_rate = cfg.sim.monthly_discount_rate
    cfg.paths.ensure()
    return cfg


if __name__ == "__main__":  # pragma: no cover - smoke test
    cfg = load_config(None, seed=11, sim={"n_customers": 1234})
    assert cfg.seed == 11 and cfg.sim.n_customers == 1234 and cfg.sim.random_state == 11
    out = PROJECT_ROOT / "artifacts" / "reports" / "_config_smoke.json"
    cfg.save(out)
    round_trip = load_config(out)
    assert round_trip.sim.n_customers == 1234
    assert isinstance(round_trip.economics.arm_costs, tuple)
    out.unlink()
    print("sections:", list(_SECTIONS))
    print("budget:", cfg.economics.budget, "| horizon:", cfg.economics.horizon)
    print("config.py OK")
