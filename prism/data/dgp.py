"""Ground-truth data-generating process for PRISM (the scientific core).

This module simulates a multi-arm, confounded, heterogeneous, discrete-time-to-event
subscription business whose **individual treatment effects are known analytically**, so
that every causal estimator downstream can be scored with PEHE against exact targets.

Design summary
--------------
*Per customer* ``i``

- a latent responder segment ``s_i`` drawn from :data:`~prism.data.schema.SEGMENTS`
  (``persuadable``, ``sure_thing``, ``lost_cause``, ``sleeping_dog``);
- a **hidden confounder** ``u_i ~ N(0, 1)`` that shifts *both* the churn hazard and the
  treatment assignment and is **never** exposed as a panel feature (it is published only in
  ``ground_truth.gt_u``). An analyst who conditions on the 23 observed features alone is
  therefore biased, which is exactly what makes ``prism.causal.refute`` meaningful;
- a frailty ``f_i ~ N(0, 0.5)`` (0.5 is the standard deviation);
- static categoricals drawn with non-uniform, *correlated* probabilities from a latent
  affluence factor (tier drives contract type, autopay, channel, device);
- an acquisition period, so cohorts genuinely differ (legacy customers carry pre-existing
  tenure that predates the simulation window).

*Per active period* ``t`` (monthly)

- eight latent AR(1) factors evolve with persistence, a covariate-drift term proportional to
  ``drift_strength * (t / n_periods)`` and a seasonal term of amplitude
  ``seasonality_amplitude``. They drive the transaction/session/ticket intensities from which
  the event log is generated, and the 17 numeric features are then **derived from the event
  history**, which keeps ``events`` internally consistent with ``panel`` (orders drive
  ``monetary_12m`` / ``frequency_12m`` / ``recency_days``, sessions drive ``sessions_30d`` /
  ``days_since_last_session``, tickets drive ``support_tickets_90d``);
- the baseline churn hazard on the logit scale is

  ``eta_it = b0 + g(x_it) + f_i + hidden_confounder_strength * u_i + season_t + concept_drift_t``

  where ``g`` contains squares, ``log1p`` terms and an interaction, so a linear model is
  misspecified by construction. ``h0_it = sigmoid(eta_it)`` keeps the average monthly hazard
  in the 2 %-8 % band;
- the treatment effect of arm ``a`` is an **additive shift on the logit hazard**,
  ``theta_{i,a} = segment_base[s_i][a] + heterogeneity_{i,a}``, so
  ``h_a(t) = sigmoid(eta_it + theta_{i,a})`` stays in ``(0, 1)`` by construction. The
  heterogeneity term is interaction-driven and non-linear (tenure x engagement,
  price change x plan tier, NPS x competitor promo intensity, discount fatigue x value,
  engagement x recency, support load x tenure). Bigger-cost arms are more effective *on
  average* but not uniformly, so the choice of arm genuinely matters per customer.

Counterfactual ground truth (frozen-covariate assumption)
---------------------------------------------------------
At every decision point ``(i, t)`` the counterfactuals are computed **analytically**, never by
resampling, under the documented **frozen-covariate assumption**: the customer's covariates are
held at their period-``t`` values over the whole horizon, while the *time* components
(seasonality, concept drift and the drifting price coefficient) continue to advance. Formally,
with ``h_a(t + j) = sigmoid(eta_it + d_j + theta_{i,a})`` where ``d_j`` is the advance in the
time components between ``t`` and ``t + j`` (``d_0 = 0``):

- ``S_a(k) = prod_{j=0..k-1} (1 - h_a(t + j))`` for ``k = 1..horizon``
- ``gt_churn_p_a = h_a(t)``
- ``gt_rmst_a    = sum_{k=1..horizon} S_a(k)``
- ``gt_value_a   = sum_{k=1..horizon} S_a(k) * monthly_margin_i * (1 + d)^(-k)``
- ``gt_tau_*_a   = gt_*_a - gt_*_0``
- ``gt_best_arm  = argmax_a (gt_value_a - ARM_COSTS[a])`` (control costs 0)
- ``gt_oracle_net_value = max(0, max_{a>=1} (gt_tau_value_a - ARM_COSTS[a]))``

``gt_rmst_*`` and ``gt_value_*`` are therefore **sustained-treatment** contrasts: arm ``a`` is
notionally re-applied at every period of the horizon. The estimable primitive is the
one-period arm-specific hazard, which a per-arm discrete-time hazard model recovers from the
panel and then compounds in exactly the same way (see
:class:`prism.causal.survival_uplift.CausalSurvivalUplift`), so the ground truth is reachable.

One-shot campaign, observed as a panel
--------------------------------------
A customer contributes one row per period **while active**, but the offer is assigned exactly
**once**, at their first panel period, and then persists (a discount stays on the account; a
concierge relationship does not lapse monthly).

This matters more than it looks. An earlier version re-drew the arm independently every
period, which made each customer's realised path a random *sequence* of offers. The analytic
ground truth, however, is defined as a **sustained-arm** contrast,
``S_a(k) = prod_{j<k} (1 - h_a(t+j))`` -- arm ``a`` in force for the whole horizon. Those are
two different estimands, and the sequence version's effect is diluted heavily toward zero
relative to the sustained one, so *no* estimator could recover ``gt_tau_*`` from the realised
data no matter how well specified it was. Assigning once makes the realised outcomes and the
counterfactuals refer to the same intervention, which is what makes PEHE against ground truth
a meaningful score rather than a category error.

Remaining consequence, deliberate:

- past offers feed back into ``discount_depth_hist``, so the feature vector at ``t`` is
  post-treatment with respect to ``t-1`` but strictly pre-treatment with respect to ``t``.

Realised outcomes
-----------------
``churn_next`` is drawn from the *assigned* arm's hazard. The trajectory is simulated ``horizon``
periods beyond the panel window so that every decision point has a fully resolved future:

- ``event_time`` / ``event_observed`` apply the realistic administrative censoring at
  ``min(horizon, n_periods - t)``;
- ``rmst_h = min(periods survived after t, horizon)`` and ``value_h`` (discounted **revenue**
  accrued while alive over the horizon) use the complete simulated trajectory. Both count the
  horizon from ``t + 1``, exactly like ``gt_rmst_*`` / ``gt_value_*``, so a customer with
  ``churn_next == 1`` has ``rmst_h == 0`` and ``value_h == 0``; note ``event_time`` uses the
  duration convention instead and is therefore ``rmst_h + 1`` before censoring. They are
  simulation-only evaluation labels; models must train on ``event_time`` / ``event_observed``.
  Multiply ``value_h`` by ``margin_rate`` to put it on the same footing as ``gt_value_*``,
  which is a **margin**.

Other notes
-----------
- ``missing_rate``, ``mnar_strength``, ``outlier_rate`` and ``duplicate_rate`` are carried on
  :class:`DGPConfig` but are *not* applied here: :func:`simulate` returns the clean gold panel
  and :mod:`prism.data.messy` injects the realism.
- No churn is simulated during the pre-panel burn-in: the customers alive at period 0 are by
  definition the survivors, and the burn-in exists only to give legacy customers a real
  twelve-month event history.
- Every stochastic draw is routed through :func:`prism.utils.seeds.as_rng` /
  :func:`prism.utils.seeds.spawn_rngs`, so the same seed reproduces the run bit-for-bit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.data.schema import (
    ARM_COSTS,
    ARM_NAMES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_LEVELS,
    GT_COLUMNS,
    N_ARMS,
    NUMERIC_FEATURES,
    PANEL_COLUMNS,
    SEGMENT_SHARES,
    SEGMENTS,
    enforce_schema,
)
from prism.utils.logging import get_logger
from prism.utils.seeds import spawn_rngs

__all__ = [
    "N_ARMS",
    "ARM_NAMES",
    "ARM_COSTS",
    "EVENT_TYPES",
    "ORDER_CATEGORIES",
    "PROPENSITY_CLIP",
    "DGPConfig",
    "SimulatedData",
    "simulate",
]

_LOG = get_logger("data.dgp")

# --------------------------------------------------------------------------------------
# Event-log vocabulary
# --------------------------------------------------------------------------------------
#: Event types emitted into the bronze event log.
EVENT_TYPES: tuple[str, ...] = ("order", "session", "support_ticket")

#: Order categories. Index 0 is the recurring subscription charge; 1..7 are add-on purchases.
ORDER_CATEGORIES: tuple[str, ...] = (
    "subscription",
    "addon_storage",
    "addon_seats",
    "support_plan",
    "hardware",
    "training",
    "marketplace",
)
_SESSION_CATEGORIES: tuple[str, ...] = ("app_session", "web_session")
_TICKET_CATEGORIES: tuple[str, ...] = ("billing", "technical", "account", "other")

#: Flat list of every `category` value that can appear in the event log.
_ALL_CATEGORIES: tuple[str, ...] = ORDER_CATEGORIES + _SESSION_CATEGORIES + _TICKET_CATEGORIES
_CAT_OFFSET_SESSION = len(ORDER_CATEGORIES)
_CAT_OFFSET_TICKET = _CAT_OFFSET_SESSION + len(_SESSION_CATEGORIES)
_N_ORDER_CATEGORIES = len(ORDER_CATEGORIES)

#: Overlap band enforced on every ground-truth propensity (SPEC 2.1 property 2).
PROPENSITY_CLIP: tuple[float, float] = (0.02, 0.85)
#: Internal projection band, a hair inside :data:`PROPENSITY_CLIP` so float error cannot escape.
_PROP_LO, _PROP_HI = 0.0205, 0.8495

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)

# --------------------------------------------------------------------------------------
# Latent AR(1) factor bank.  Order: spend/order intensity, basket value, engagement,
# service friction, price pressure, satisfaction, competitor pressure, payment reliability.
# --------------------------------------------------------------------------------------
_LATENT_NAMES: tuple[str, ...] = (
    "order", "spend", "engage", "service", "price", "nps", "promo", "pay",
)
_LATENT_RHO = np.array([0.82, 0.86, 0.78, 0.70, 0.90, 0.84, 0.66, 0.74])
_LATENT_DRIFT = np.array([-0.45, 0.30, -0.70, 0.55, 0.95, -0.60, 0.80, 0.35])
_LATENT_SEAS = np.array([0.35, 0.28, 0.45, 0.15, 0.05, 0.20, 0.70, 0.08])
_N_LATENT = len(_LATENT_NAMES)
_IX_ORDER, _IX_SPEND, _IX_ENGAGE, _IX_SERVICE, _IX_PRICE, _IX_NPS, _IX_PROMO, _IX_PAY = range(_N_LATENT)

# --------------------------------------------------------------------------------------
# Treatment-effect structure (additive shifts on the logit hazard; negative == retains)
# --------------------------------------------------------------------------------------
#: Segment-level base effect per arm. Rows follow :data:`~prism.data.schema.SEGMENTS`.
_SEGMENT_BASE_THETA = np.array(
    [
        [0.00, -0.55, -0.98, -1.34],  # persuadable  : the offer prevents churn
        [0.00, -0.12, -0.20, -0.29],  # sure_thing   : stays anyway, mild help
        [0.00, -0.02, 0.01, 0.04],    # lost_cause   : leaves anyway, ~no effect
        [0.00, 0.36, 0.62, 0.88],     # sleeping_dog : the offer reminds them of the price
    ]
)

#: Coefficients on the seven centred interaction terms, one row per arm (row 0 == control).
#: Deliberately non-proportional across arms so the best arm is customer-specific.
_HET_COEF = np.array(
    [
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.20, -0.30, 0.16, 0.24, -0.09, -0.14, 0.10],   # discount_10
        [-0.34, -0.62, 0.28, 0.50, -0.16, -0.22, 0.18],   # discount_25
        [-0.78, -0.20, 0.44, 0.12, -0.32, -0.50, -0.26],  # concierge
    ]
)
_THETA_CLIP = 2.2

#: Multinomial-logit assignment coefficients, columns
#: ``(intercept, churn-risk, monetary value, tenure, hidden confounder U)``.
_ASSIGN_COEF = np.array(
    [
        [0.00, 0.00, 0.00, 0.00, 0.00],   # control (reference)
        [0.10, 0.85, 0.25, 0.10, 0.45],   # discount_10
        [-0.35, 1.35, 0.60, 0.05, 0.70],  # discount_25
        [-1.15, 1.70, 1.15, 0.40, 0.90],  # concierge
    ]
)

#: Fixed reference means used to centre the hazard index. They are constants (never estimated
#: from the sample) so that the generative model does not depend on the realised draw.
_REF = {
    "log_tenure": 3.10,
    "recency_m": 0.443,
    "sqrt_freq": 3.72,
    "log_monetary": 6.80,
    "engagement": 0.549,
    "sqrt_tickets": 0.285,
    "nps": 22.0,
    "sqrt_payfail": 0.300,
    "price_change": 3.39,
    "promo": 49.2,
    "discount_depth": 0.245,
    "basket_div": 0.494,
    "log_days_no_session": 2.88,
    "log_sessions": 0.757,
}

_PLAN_PRICES = np.array([39.0, 79.0, 149.0, 329.0])


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def _sample_rows(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Draw one categorical index per row of ``probs``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    probs : numpy.ndarray of shape (m, k)
        Row-stochastic probability matrix.

    Returns
    -------
    numpy.ndarray of shape (m,)
        Integer category indices in ``[0, k)``.
    """
    cum = np.cumsum(probs, axis=1)
    cum[:, -1] = 1.0
    u = rng.random(probs.shape[0])
    idx = (u[:, None] >= cum).sum(axis=1)
    return np.minimum(idx, probs.shape[1] - 1).astype(np.int64)


def _softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _project_to_overlap(p: np.ndarray, lo: float = _PROP_LO, hi: float = _PROP_HI) -> np.ndarray:
    """Shrink each row of ``p`` toward uniform until every entry lies in ``[lo, hi]``.

    Shrinking toward the uniform distribution is a convex combination of two points of the
    simplex, so the result still sums to one exactly -- unlike clip-then-renormalise, which can
    push entries back out of the box.

    Parameters
    ----------
    p : numpy.ndarray of shape (m, k)
        Row-stochastic matrix.
    lo, hi : float
        Inclusive bounds; ``lo < 1/k < hi`` must hold.

    Returns
    -------
    numpy.ndarray of shape (m, k)
        Row-stochastic matrix with every entry inside ``[lo, hi]``.
    """
    k = p.shape[1]
    unif = 1.0 / k
    if not (lo < unif < hi):
        raise ValueError(f"overlap box [{lo}, {hi}] does not contain the uniform point {unif}")
    need_lo = np.where(p < lo, (lo - p) / np.maximum(unif - p, 1e-12), 0.0)
    need_hi = np.where(p > hi, (p - hi) / np.maximum(p - unif, 1e-12), 0.0)
    lam = np.clip(np.maximum(need_lo.max(axis=1), need_hi.max(axis=1)), 0.0, 1.0)
    out = (1.0 - lam)[:, None] * p + lam[:, None] * unif
    return out / out.sum(axis=1, keepdims=True)


def _cohort_labels(origin_year: int, origin_month: int, offsets: np.ndarray) -> np.ndarray:
    """Vectorised ``YYYY-MM`` cohort labels ``offsets`` months from the origin month."""
    total = origin_year * 12 + (origin_month - 1) + offsets.astype(np.int64)
    years = total // 12
    months = total % 12 + 1
    ys = np.char.zfill(years.astype(str), 4)
    ms = np.char.zfill(months.astype(str), 2)
    return np.char.add(np.char.add(ys, "-"), ms).astype(object)


# ======================================================================================
# Config
# ======================================================================================
@dataclass
class DGPConfig:
    """Parameters of the ground-truth simulator.

    Parameters
    ----------
    n_customers : int, default 60000
        Number of simulated customers.
    n_periods : int, default 24
        Number of monthly decision periods in the panel (``0 .. n_periods - 1``).
    horizon : int, default 12
        Months used for the RMST / discounted-value outcomes and ground truth.
    origin : str, default "2022-01-01"
        Calendar date of period 0. All timestamps derive from it; no wall clock is read.
    monthly_discount_rate : float, default 0.01
        Monthly discount rate applied to future margin / revenue.
    rct_fraction : float, default 0.15
        Fraction of **customers** randomised into the clean A/B holdout block.
    confounding_strength : float, default 1.0
        Scales the whole assignment index. ``0.0`` collapses to uniform assignment.
    hidden_confounder_strength : float, default 0.05
        Coefficient of the unobserved ``U`` in the hazard. Drives sensitivity analysis.
    drift_strength : float, default 0.6
        Magnitude of covariate drift, concept drift and coefficient drift over the window.
    seasonality_amplitude : float, default 0.25
        Amplitude of the annual seasonal cycle in features and in the hazard.
    missing_rate, mnar_strength, outlier_rate, duplicate_rate : float
        Carried for :mod:`prism.data.messy`; **not** applied by :func:`simulate`.
    random_state : int, default 7
        Master seed. The same value reproduces the run bit-for-bit.
    burn_in_periods : int, default 12
        Pre-panel months simulated so legacy customers own a full 12-month event history.
    legacy_fraction : float, default 0.58
        Fraction of customers already active at period 0 (the rest are acquired later).
    baseline_logit : float, default -3.05
        Intercept ``b0`` of the hazard; tuned so the mean monthly hazard sits near 5 %.
    train_end_period, valid_end_period : int, default 13 / 17
        Default temporal split boundaries written into ``panel.split``.
    emit_events : bool, default True
        Set ``False`` to skip building the (large) bronze event log.
    """

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
    # ---- optional extensions (defaults preserve the contract above) -------------------
    burn_in_periods: int = 12
    legacy_fraction: float = 0.58
    baseline_logit: float = -3.47
    train_end_period: int = 13
    valid_end_period: int = 17
    emit_events: bool = True

    def __post_init__(self) -> None:
        if self.n_customers < 1:
            raise ValueError("n_customers must be >= 1")
        if self.n_periods < 1:
            raise ValueError("n_periods must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not 0.0 <= self.rct_fraction <= 1.0:
            raise ValueError("rct_fraction must lie in [0, 1]")
        if self.confounding_strength < 0.0:
            raise ValueError("confounding_strength must be >= 0")
        if self.monthly_discount_rate <= -1.0:
            raise ValueError("monthly_discount_rate must be > -1")
        self.burn_in_periods = max(1, int(self.burn_in_periods))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of every field."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DGPConfig:
        """Build a config from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})


# ======================================================================================
# Container
# ======================================================================================
@dataclass
class SimulatedData:
    """Bundle returned by :func:`simulate`.

    Attributes
    ----------
    panel : pandas.DataFrame
        The clean gold panel, exactly :data:`~prism.data.schema.PANEL_COLUMNS`.
    customers : pandas.DataFrame
        One row per customer: static attributes plus the latent truth (segment, ``U``, frailty,
        acquisition period, unit economics).
    events : pandas.DataFrame
        Bronze event log: ``customer_id``, ``event_ts``, ``event_type``, ``amount``,
        ``category``, ``n_items``; consistent with the panel features by construction.
    ground_truth : pandas.DataFrame
        Exactly :data:`~prism.data.schema.GT_COLUMNS`, aligned row-for-row with ``panel``.
    config : DGPConfig
        The configuration that produced the bundle.
    """

    panel: pd.DataFrame
    customers: pd.DataFrame
    events: pd.DataFrame
    ground_truth: pd.DataFrame
    config: DGPConfig

    _FILES = {
        "panel": "panel.parquet",
        "customers": "customers.parquet",
        "events": "events.parquet",
        "ground_truth": "ground_truth.parquet",
    }
    _CONFIG_FILE = "dgp_config.json"

    def save(self, dirpath: str | Path) -> None:
        """Write the bundle to ``dirpath`` as parquet plus a JSON config sidecar.

        Parameters
        ----------
        dirpath : str or pathlib.Path
            Destination directory; created if absent.
        """
        out = Path(dirpath)
        out.mkdir(parents=True, exist_ok=True)
        for attr, fname in self._FILES.items():
            frame: pd.DataFrame = getattr(self, attr)
            frame.to_parquet(out / fname, index=False, engine="pyarrow")
        (out / self._CONFIG_FILE).write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, dirpath: str | Path) -> SimulatedData:
        """Read a bundle previously written by :meth:`save`.

        Parameters
        ----------
        dirpath : str or pathlib.Path
            Directory containing the parquet files and ``dgp_config.json``.

        Returns
        -------
        SimulatedData
        """
        src = Path(dirpath)
        frames = {
            attr: pd.read_parquet(src / fname, engine="pyarrow")
            for attr, fname in cls._FILES.items()
        }
        cfg_path = src / cls._CONFIG_FILE
        cfg = DGPConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8"))) if cfg_path.exists() else DGPConfig()
        return cls(config=cfg, **frames)

    def summary(self) -> pd.DataFrame:
        """Return a tidy one-row-per-table description of the bundle."""
        rows = [
            {"table": name, "n_rows": int(len(getattr(self, name))), "n_cols": int(getattr(self, name).shape[1])}
            for name in self._FILES
        ]
        return pd.DataFrame(rows)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<SimulatedData panel={len(self.panel):,}x{self.panel.shape[1]} "
            f"customers={len(self.customers):,} events={len(self.events):,} "
            f"gt={len(self.ground_truth):,}>"
        )


# ======================================================================================
# Stage 1 -- static customer population
# ======================================================================================
def _draw_static_population(cfg: DGPConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw every time-invariant customer attribute.

    Parameters
    ----------
    cfg : DGPConfig
        Simulation configuration.
    rng : numpy.random.Generator
        Dedicated stream for static draws.

    Returns
    -------
    dict of str to numpy.ndarray
        Arrays of length ``cfg.n_customers``: latent segment / confounder / frailty, static
        categorical codes, acquisition timing, unit economics and event intensities.
    """
    n = cfg.n_customers
    T = cfg.n_periods

    # ---- latent truth ----------------------------------------------------------------
    segment = _sample_rows(rng, np.tile(np.asarray(SEGMENT_SHARES, dtype=np.float64), (n, 1)))
    u = rng.normal(0.0, 1.0, size=n)                 # hidden confounder, never a feature
    frailty = rng.normal(0.0, 0.5, size=n)           # 0.5 is the standard deviation
    affluence = rng.normal(0.0, 1.0, size=n)         # drives the correlated categoricals

    # ---- correlated static categoricals ----------------------------------------------
    tier_logit = np.array([0.55, 0.35, -0.10, -1.05])[None, :] + np.outer(
        affluence, np.array([-0.90, -0.20, 0.55, 1.15])
    )
    plan_tier = _sample_rows(rng, _softmax(tier_logit))

    p_annual = _sigmoid(-0.75 + 0.55 * affluence + 0.35 * plan_tier)
    contract_type = (rng.random(n) < p_annual).astype(np.int64)  # 0 == monthly, 1 == annual

    p_autopay = _sigmoid(0.85 + 0.40 * affluence + 0.55 * contract_type)
    is_autopay = (rng.random(n) < p_autopay).astype(np.int64)     # 1 == yes

    channel_rows = np.array(
        [
            [0.42, 0.34, 0.16, 0.08],   # basic
            [0.36, 0.31, 0.22, 0.11],   # plus
            [0.28, 0.24, 0.29, 0.19],   # pro
            [0.18, 0.17, 0.32, 0.33],   # enterprise
        ]
    )
    channel = _sample_rows(rng, channel_rows[plan_tier])

    region_rows = np.array(
        [
            [0.34, 0.24, 0.23, 0.19],   # organic
            [0.28, 0.29, 0.22, 0.21],   # paid
            [0.31, 0.22, 0.27, 0.20],   # referral
            [0.25, 0.26, 0.25, 0.24],   # partner
        ]
    )
    region = _sample_rows(rng, region_rows[channel])

    device_base = np.array(
        [
            [0.34, 0.38, 0.28],   # north
            [0.28, 0.45, 0.27],   # south
            [0.40, 0.32, 0.28],   # east
            [0.31, 0.39, 0.30],   # west
        ]
    )[region]
    device_logit = np.log(device_base) + np.outer(affluence, np.array([0.38, -0.22, -0.10]))
    device = _sample_rows(rng, _softmax(device_logit))

    # ---- acquisition timing / cohorts ------------------------------------------------
    is_legacy = rng.random(n) < cfg.legacy_fraction
    if T < 3:
        is_legacy[:] = True
    tenure0 = np.clip(1.0 + rng.gamma(1.6, 14.0, size=n), 1.0, 96.0).round()
    if T >= 3:
        entry_grid = np.arange(1, T - 1)
        w = np.exp(-0.05 * entry_grid)
        w = w / w.sum()
        entry_new = entry_grid[_sample_rows(rng, np.tile(w, (n, 1)))]
    else:
        entry_new = np.ones(n, dtype=np.int64)
    acq_period = np.where(is_legacy, -tenure0.astype(np.int64), entry_new - 1).astype(np.int64)
    entry_period = np.maximum(acq_period + 1, 0).astype(np.int64)

    # ---- unit economics ---------------------------------------------------------------
    plan_price = _PLAN_PRICES[plan_tier] * np.exp(rng.normal(0.0, 0.18, size=n))
    addon_lambda = np.clip(rng.gamma(1.4, 0.28, size=n), 0.02, 3.0)
    monthly_revenue = plan_price * (1.0 + 0.25 * addon_lambda)
    margin_rate = np.clip(0.18 + 0.30 * rng.beta(2.5, 2.5, size=n), 0.02, 0.97)
    monthly_margin = monthly_revenue * margin_rate

    # ---- per-customer event intensities ------------------------------------------------
    session_lambda = np.clip(rng.gamma(2.2, 0.75, size=n), 0.05, 12.0)
    ticket_lambda = np.clip(0.055 * np.exp(rng.normal(0.0, 0.55, size=n)), 0.0, 1.5)
    payfail_lambda = np.clip(
        0.030 * np.exp(rng.normal(0.0, 0.5, size=n) - 0.60 * is_autopay), 0.0, 0.6
    )
    base_discount = np.clip(rng.beta(2.2, 9.0, size=n), 0.0, 0.6)

    # ---- billing anniversary within the month (keeps order timestamps realistic) --------
    bill_day_frac = rng.random(n)

    # ---- assignment block --------------------------------------------------------------
    in_rct = rng.random(n) < cfg.rct_fraction

    customer_id = np.array([f"C{i:07d}" for i in range(n)], dtype=object)

    return {
        "customer_id": customer_id,
        "segment": segment,
        "u": u,
        "frailty": frailty,
        "affluence": affluence,
        "plan_tier": plan_tier,
        "channel": channel,
        "region": region,
        "device": device,
        "is_autopay": is_autopay,
        "contract_type": contract_type,
        "is_legacy": is_legacy,
        "acq_period": acq_period,
        "entry_period": entry_period,
        "plan_price": plan_price,
        "addon_lambda": addon_lambda,
        "monthly_revenue": monthly_revenue,
        "margin_rate": margin_rate,
        "monthly_margin": monthly_margin,
        "session_lambda": session_lambda,
        "ticket_lambda": ticket_lambda,
        "payfail_lambda": payfail_lambda,
        "base_discount": base_discount,
        "bill_day_frac": bill_day_frac,
        "in_rct": in_rct,
    }


# ======================================================================================
# Stage 2 -- point-in-time feature derivation
# ======================================================================================
def _derive_features(
    state: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    zeff: np.ndarray,
    as_of_day: float,
    slot: int,
    month_phase: float,
    prog: float,
    cfg: DGPConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Build the 17 numeric features for every customer from history strictly before ``t``.

    Event-derived features read the rolling buffers, which at this point hold periods
    ``t-12 .. t-1`` only, so the block is point-in-time safe by construction.

    Parameters
    ----------
    state : dict of str to numpy.ndarray
        Mutable rolling-window buffers (order counts / amounts / category bitmasks / sessions /
        tickets / payment failures) plus last-event day stamps and the current period index.
    static : dict of str to numpy.ndarray
        Output of :func:`_draw_static_population`.
    zeff : numpy.ndarray of shape (n, 8)
        Effective latent factors for this period (AR(1) state + drift + seasonality).
    as_of_day : float
        Decision timestamp expressed in days since ``cfg.origin``.
    slot : int
        Ring-buffer slot index of the current period (``t % 12``).
    month_phase : float
        ``2 * pi * (calendar_month - 1) / 12`` for the current period.
    prog : float
        Normalised progress through the panel window, ``t / (n_periods - 1)``, clipped.
    cfg : DGPConfig
        Simulation configuration.
    rng : numpy.random.Generator
        Stream used for the small observation noise on survey-style features.

    Returns
    -------
    dict of str to numpy.ndarray
        One float64 array of length ``n_customers`` per name in ``NUMERIC_FEATURES``.
    """
    n = cfg.n_customers
    amp = cfg.seasonality_amplitude

    # ---- event-derived ----------------------------------------------------------------
    frequency_12m = state["ord_count"].sum(axis=1).astype(np.float64)
    monetary_12m = state["ord_amount"].sum(axis=1).astype(np.float64)
    denom = np.maximum(frequency_12m, 1.0)
    avg_order_value = np.where(frequency_12m > 0, monetary_12m / denom, 0.0)
    cat_mask = np.bitwise_or.reduce(state["cat_mask"], axis=1)
    n_categories_12m = _POPCOUNT[cat_mask].astype(np.float64)
    basket_diversity = n_categories_12m / float(_N_ORDER_CATEGORIES)

    prev_slot = (slot - 1) % 12
    sessions_30d = state["sess_count"][:, prev_slot].astype(np.float64)
    support_tickets_90d = (
        state["tick_count"][:, prev_slot]
        + state["tick_count"][:, (slot - 2) % 12]
        + state["tick_count"][:, (slot - 3) % 12]
    ).astype(np.float64)
    payment_failures_12m = state["payfail"].sum(axis=1).astype(np.float64)

    recency_days = np.clip(as_of_day - state["last_order_day"], 0.0, 400.0)
    days_since_last_session = np.clip(as_of_day - state["last_session_day"], 0.0, 365.0)

    # ---- tenure counts up ----------------------------------------------------------------
    tenure_months = np.maximum(state["period"] - static["acq_period"], 0).astype(np.float64)

    # ---- latent-driven --------------------------------------------------------------------
    engagement_score = 100.0 * _sigmoid(
        1.05
        + 0.80 * zeff[:, _IX_ENGAGE]
        + 0.45 * np.log1p(sessions_30d)
        - 0.42 * np.log1p(days_since_last_session)
        + 0.05 * np.sqrt(frequency_12m)
    )

    tier_uplift = np.array([0.4, 0.9, 1.6, 2.4])[static["plan_tier"]]
    price_change_pct = np.clip(
        0.9
        + 4.6 * cfg.drift_strength * prog
        + tier_uplift
        + 1.15 * zeff[:, _IX_PRICE]
        + 0.55 * amp * math.sin(month_phase - 0.6),
        -10.0,
        30.0,
    )

    competitor_promo_intensity = np.clip(
        44.0
        + 38.0 * amp * math.sin(month_phase + 1.05)
        + 9.0 * cfg.drift_strength * prog
        + 11.0 * zeff[:, _IX_PROMO],
        0.0,
        100.0,
    )

    seasonality_index = np.clip(
        1.0
        + amp * (0.80 * math.sin(month_phase) + 0.25 * math.cos(2.0 * month_phase))
        + rng.normal(0.0, 0.010, size=n),
        0.40,
        1.80,
    )

    nps = np.clip(
        28.0
        + 20.0 * zeff[:, _IX_NPS]
        - 6.0 * support_tickets_90d
        - 0.55 * (price_change_pct - 3.0)
        - 0.10 * (days_since_last_session - 15.0)
        + 0.12 * (engagement_score - 50.0),
        -100.0,
        100.0,
    )

    discount_depth_hist = np.clip(state["disc_ema"], 0.0, 0.95).astype(np.float64)

    return {
        "tenure_months": tenure_months,
        "recency_days": recency_days,
        "frequency_12m": frequency_12m,
        "monetary_12m": monetary_12m,
        "avg_order_value": avg_order_value,
        "n_categories_12m": n_categories_12m,
        "sessions_30d": sessions_30d,
        "days_since_last_session": days_since_last_session,
        "engagement_score": engagement_score,
        "support_tickets_90d": support_tickets_90d,
        "nps": nps,
        "payment_failures_12m": payment_failures_12m,
        "discount_depth_hist": discount_depth_hist,
        "price_change_pct": price_change_pct,
        "competitor_promo_intensity": competitor_promo_intensity,
        "seasonality_index": seasonality_index,
        "basket_diversity": basket_diversity,
    }


def _risk_index(f: dict[str, np.ndarray], price_beta: float) -> np.ndarray:
    """Non-linear covariate contribution ``g(x)`` to the logit hazard.

    Contains squares, ``log1p`` transforms and a tenure-by-engagement interaction, so an
    additive linear model of the raw features is misspecified by construction.

    Parameters
    ----------
    f : dict of str to numpy.ndarray
        Feature block produced by :func:`_derive_features`.
    price_beta : float
        Time-varying coefficient on ``price_change_pct``; its growth is the *concept* drift.

    Returns
    -------
    numpy.ndarray
        Approximately mean-zero risk index, one entry per customer.
    """
    log_tenure = np.log1p(f["tenure_months"]) - _REF["log_tenure"]
    recency_m = f["recency_days"] / 30.0 - _REF["recency_m"]
    engage = f["engagement_score"] / 100.0 - _REF["engagement"]
    return (
        -0.46 * log_tenure
        + 0.08 * log_tenure**2
        + 0.30 * recency_m
        + 0.16 * recency_m**2
        - 0.20 * (np.sqrt(f["frequency_12m"]) - _REF["sqrt_freq"])
        - 0.18 * (np.log1p(f["monetary_12m"]) - _REF["log_monetary"])
        - 1.05 * engage
        - 0.60 * engage**2
        + 0.55 * (np.sqrt(f["support_tickets_90d"]) - _REF["sqrt_tickets"])
        - 0.010 * (f["nps"] - _REF["nps"])
        + 0.45 * (np.sqrt(f["payment_failures_12m"]) - _REF["sqrt_payfail"])
        + price_beta * (f["price_change_pct"] - _REF["price_change"])
        + 0.011 * (f["competitor_promo_intensity"] - _REF["promo"])
        + 0.85 * (f["discount_depth_hist"] - _REF["discount_depth"])
        - 0.42 * (f["basket_diversity"] - _REF["basket_div"])
        + 0.14 * (np.log1p(f["days_since_last_session"]) - _REF["log_days_no_session"])
        + 0.35 * (f["seasonality_index"] - 1.0)
        - 0.22 * (np.log1p(f["sessions_30d"]) - _REF["log_sessions"])
        + 0.13 * log_tenure * engage
    )


def _treatment_effects(f: dict[str, np.ndarray], static: dict[str, np.ndarray]) -> np.ndarray:
    """Per-customer, per-arm additive shift ``theta`` on the logit hazard.

    ``theta = segment_base[segment][arm] + heterogeneity``, where the heterogeneity block is a
    linear combination of seven **centred interaction terms** with arm-specific, deliberately
    non-proportional coefficients, so the best arm is customer-specific.

    Parameters
    ----------
    f : dict of str to numpy.ndarray
        Feature block for the current period.
    static : dict of str to numpy.ndarray
        Static population arrays (segment, plan tier, autopay, contract type).

    Returns
    -------
    numpy.ndarray of shape (n_customers, N_ARMS)
        ``theta``; column 0 is identically zero (control). Negative means churn-reducing.
    """
    log_tenure_c = np.log1p(f["tenure_months"]) / 3.4 - 1.0
    engage_c = f["engagement_score"] / 100.0 - 0.52
    tier_c = static["plan_tier"].astype(np.float64) - 1.15
    price_c = f["price_change_pct"] / 8.0 - 0.45
    nps_c = f["nps"] / 60.0 - 0.35
    promo_c = f["competitor_promo_intensity"] / 50.0 - 0.95
    value_c = np.log1p(f["monetary_12m"]) / 6.0 - 1.0
    fatigue_c = np.tanh(3.0 * f["discount_depth_hist"] - 0.6)
    recency_c = 1.0 - f["recency_days"] / 22.0
    committed = (static["is_autopay"] * static["contract_type"]).astype(np.float64) - 0.35
    tickets_c = f["support_tickets_90d"] - 0.18

    terms = np.column_stack(
        [
            log_tenure_c * engage_c,   # tenure x engagement
            price_c * tier_c,          # price change x plan tier
            nps_c * promo_c,           # NPS x competitor promo intensity
            fatigue_c * value_c,       # discount fatigue x customer value
            committed,                 # autopay x annual contract
            engage_c * recency_c,      # engagement x recency
            tickets_c * log_tenure_c,  # support load x tenure
        ]
    )
    theta = _SEGMENT_BASE_THETA[static["segment"]] + terms @ _HET_COEF.T
    theta[:, 0] = 0.0
    return np.clip(theta, -_THETA_CLIP, _THETA_CLIP)


def _assignment_propensity(
    eta_observable: np.ndarray,
    f: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    cfg: DGPConfig,
) -> np.ndarray:
    """Multinomial-logit generalised propensity score for the observational block.

    The linear index depends on an **observable** churn-risk score, ``monetary_12m``,
    ``tenure_months`` and the **hidden confounder** ``U``, and the whole index is scaled by
    ``confounding_strength`` so that ``0.0`` collapses to exactly uniform assignment. Rows are
    then projected into the overlap box :data:`PROPENSITY_CLIP` while still summing to one.

    The risk term is deliberately the *observable* part of the logit hazard rather than the
    full latent ``eta``. Two reasons, one practical and one statistical. Practically, a real
    retention team targets on a model score fitted to the data it has, not on a latent truth.
    Statistically, feeding the per-customer frailty into assignment would make it a second
    hidden confounder on top of ``U``, and the combination leaves the treatment effect
    essentially unidentifiable from the panel -- a simulator nobody could analyse correctly is
    not a useful benchmark. ``U`` remains the single documented unobserved confounder, which is
    exactly what the sensitivity analysis in ``prism/causal/refute.py`` is calibrated against.

    Parameters
    ----------
    eta_observable : numpy.ndarray
        Logit-scale churn risk reconstructable from the panel features (no frailty, no ``U``).
    f : dict of str to numpy.ndarray
        Feature block for the current period.
    static : dict of str to numpy.ndarray
        Static population arrays (needs ``u``).
    cfg : DGPConfig
        Simulation configuration.

    Returns
    -------
    numpy.ndarray of shape (n_customers, N_ARMS)
        Row-stochastic propensities, every entry inside :data:`PROPENSITY_CLIP`.
    """
    eta_std = (eta_observable - cfg.baseline_logit) / 1.15
    mon_std = (np.log1p(f["monetary_12m"]) - _REF["log_monetary"]) / 1.10
    ten_std = (np.log1p(f["tenure_months"]) - _REF["log_tenure"]) / 0.95
    design = np.column_stack([np.ones_like(eta_std), eta_std, mon_std, ten_std, static["u"]])
    z = cfg.confounding_strength * (design @ _ASSIGN_COEF.T)
    return _project_to_overlap(_softmax(z))


# ======================================================================================
# Stage 3 -- the simulator
# ======================================================================================
def _calendar(cfg: DGPConfig) -> dict[str, Any]:
    """Build the absolute period grid running from ``-burn_in`` to ``n_periods + horizon - 1``.

    Parameters
    ----------
    cfg : DGPConfig
        Simulation configuration.

    Returns
    -------
    dict of str to Any
        ``dates`` (month starts), ``day_edges`` (days since origin, one extra edge),
        ``month_len``, ``phase`` (annual angle), ``prog`` (normalised progress),
        ``season``, ``cdrift`` and ``price_beta``, each indexed by ``t + burn_in``.
    """
    origin_ts = pd.Timestamp(cfg.origin)
    n_abs = cfg.burn_in_periods + cfg.n_periods + cfg.horizon
    start_ts = origin_ts - pd.DateOffset(months=cfg.burn_in_periods)
    dates_ext = pd.date_range(start_ts, periods=n_abs + 1, freq="MS")
    day_edges = (dates_ext - origin_ts).days.to_numpy().astype(np.float64)
    dates = dates_ext[:-1]
    phase = 2.0 * np.pi * (dates.month.to_numpy() - 1) / 12.0
    abs_periods = np.arange(-cfg.burn_in_periods, cfg.n_periods + cfg.horizon, dtype=np.float64)
    prog = np.clip(abs_periods / float(max(cfg.n_periods - 1, 1)), -0.6, 1.3)
    return {
        "origin_ts": origin_ts,
        "n_abs": n_abs,
        "dates": dates,
        "day_edges": day_edges,
        "month_len": np.diff(day_edges),
        "phase": phase,
        "prog": prog,
        "season": cfg.seasonality_amplitude * (0.90 * np.sin(phase) + 0.30 * np.sin(2.0 * phase)),
        "cdrift": cfg.drift_strength * (0.60 * prog - 0.20),
        "price_beta": 0.030 * (1.0 + 0.90 * cfg.drift_strength * prog),
    }


def _split_bounds(cfg: DGPConfig) -> tuple[int, int]:
    """Return the ``(train_end, valid_end)`` period boundaries for the default temporal split.

    Uses the contracted 13 / 17 rule when the panel is long enough, and proportional
    boundaries otherwise so that short panels still contain all three splits.
    """
    if cfg.n_periods > cfg.valid_end_period + 1:
        return int(cfg.train_end_period), int(cfg.valid_end_period)
    last = cfg.n_periods - 1
    train_end = max(0, int(round(0.60 * last)))
    valid_end = max(train_end, int(round(0.80 * last)))
    if valid_end >= last and last >= 2:
        valid_end = last - 1
    if train_end >= valid_end and valid_end >= 1:
        train_end = valid_end - 1
    return train_end, valid_end


def simulate(config: DGPConfig | None = None) -> SimulatedData:
    """Run the ground-truth simulator and return the full bundle.

    The routine is fully vectorised over customers; the only Python loop is over the
    ``burn_in_periods + n_periods + horizon`` monthly periods, as intended.

    Parameters
    ----------
    config : DGPConfig or None, optional
        Simulation configuration. ``None`` uses :class:`DGPConfig` defaults
        (60 000 customers x 24 periods).

    Returns
    -------
    SimulatedData
        ``panel`` (exactly :data:`~prism.data.schema.PANEL_COLUMNS`), ``customers``,
        ``events`` and ``ground_truth`` (exactly :data:`~prism.data.schema.GT_COLUMNS`).

    Notes
    -----
    See the module docstring for the frozen-covariate assumption behind the analytic
    counterfactuals and for the repeated-decision semantics of the panel.
    """
    cfg = config if config is not None else DGPConfig()
    n, T, H, B = cfg.n_customers, cfg.n_periods, cfg.horizon, cfg.burn_in_periods
    rng_static, rng_dyn, rng_evt, rng_assign, rng_out, rng_feat = spawn_rngs(cfg.random_state, 6)

    static = _draw_static_population(cfg, rng_static)
    cal = _calendar(cfg)
    day_edges, month_len = cal["day_edges"], cal["month_len"]
    season, cdrift, price_beta = cal["season"], cal["cdrift"], cal["price_beta"]
    phase, prog = cal["phase"], cal["prog"]

    disc = (1.0 + cfg.monthly_discount_rate) ** -np.arange(1, H + 1, dtype=np.float64)
    cum_disc = np.concatenate([[0.0], np.cumsum(disc)])
    arm_costs = np.asarray(ARM_COSTS, dtype=np.float64)

    # ---- rolling point-in-time state --------------------------------------------------
    state: dict[str, Any] = {
        "ord_count": np.zeros((n, 12), dtype=np.float32),
        "ord_amount": np.zeros((n, 12), dtype=np.float32),
        "cat_mask": np.zeros((n, 12), dtype=np.uint8),
        "sess_count": np.zeros((n, 12), dtype=np.float32),
        "tick_count": np.zeros((n, 12), dtype=np.float32),
        "payfail": np.zeros((n, 12), dtype=np.float32),
        "last_order_day": np.full(n, -1.0e6, dtype=np.float64),
        "last_session_day": np.full(n, -1.0e6, dtype=np.float64),
        "disc_ema": static["base_discount"].astype(np.float64).copy(),
        "period": -B,
    }
    latent = rng_dyn.normal(size=(n, _N_LATENT))
    sd_innov = np.sqrt(1.0 - _LATENT_RHO**2)

    alive = np.ones(n, dtype=bool)
    churn_period = np.full(n, T + H + 5, dtype=np.int64)

    # ---- accumulators -----------------------------------------------------------------
    acc_idx: list[np.ndarray] = []
    acc_period: list[np.ndarray] = []
    acc_arm: list[np.ndarray] = []
    acc_churn_next: list[np.ndarray] = []
    acc_prop: list[np.ndarray] = []
    acc_gt_churn: list[np.ndarray] = []
    acc_gt_rmst: list[np.ndarray] = []
    acc_gt_value: list[np.ndarray] = []
    acc_feat: dict[str, list[np.ndarray]] = {name: [] for name in NUMERIC_FEATURES}
    ev_owner: list[np.ndarray] = []
    ev_day: list[np.ndarray] = []
    ev_type: list[np.ndarray] = []
    ev_amount: list[np.ndarray] = []
    ev_cat: list[np.ndarray] = []
    ev_items: list[np.ndarray] = []
    hazard_trace: list[float] = []

    # Persistent treatment state: -1 means "not yet reached their first panel period".
    # See the assignment block inside the loop for why the offer is assigned only once.
    assigned_arm = np.full(n, -1, dtype=np.int64)
    assigned_prop = np.full((n, N_ARMS), 1.0 / N_ARMS)

    for ai in range(cal["n_abs"]):
        t = ai - B
        slot = t % 12
        state["period"] = t

        # -- latent AR(1) evolution + covariate drift + seasonality ---------------------
        latent = _LATENT_RHO * latent + sd_innov * rng_dyn.normal(size=(n, _N_LATENT))
        zeff = (
            latent
            + (_LATENT_DRIFT * cfg.drift_strength * prog[ai])[None, :]
            + (_LATENT_SEAS * cfg.seasonality_amplitude * math.sin(phase[ai]))[None, :]
        )

        has_history = alive & (static["acq_period"] <= t)
        active = alive & (static["entry_period"] <= t) & has_history
        if not has_history.any():
            continue

        # -- point-in-time features (history strictly before t) -------------------------
        feats = _derive_features(
            state, static, zeff, float(day_edges[ai]), slot, float(phase[ai]), float(prog[ai]), cfg, rng_feat
        )
        # The observable part of the risk index: everything an analyst could reconstruct from
        # the panel. This -- NOT the latent eta -- is what drives targeting, because a real
        # retention team targets on a model score fitted to observed data, and because routing
        # the frailty into assignment would make it a second, undocumented hidden confounder
        # and leave the treatment effect unidentifiable from the panel alone.
        eta_observable = (
            cfg.baseline_logit
            + _risk_index(feats, float(price_beta[ai]))
            + season[ai]
            + cdrift[ai]
        )
        # The latent hazard adds per-customer frailty (unexplained outcome variance, not a
        # confounder) and the hidden confounder U (which DOES also enter assignment, below).
        eta = (
            eta_observable
            + static["frailty"]
            + cfg.hidden_confounder_strength * static["u"]
        )
        theta = _treatment_effects(feats, static)

        # -- treatment assignment --------------------------------------------------------
        # The offer is assigned ONCE, at the customer's first panel period, and then persists
        # (a discount stays on the account; a concierge relationship does not lapse monthly).
        #
        # This is load-bearing, not cosmetic. The analytic ground truth defines
        #   S_a(k) = prod_{j<k} (1 - h_a(t+j)),
        # i.e. arm `a` in force across the whole horizon. Re-drawing a fresh arm every period
        # would make each customer's realised path a random *sequence* of offers, whose
        # average effect is heavily diluted toward zero relative to that estimand -- so the
        # observable effect and gt_tau_* would measure different things, and no estimator
        # could recover the ground truth. One persistent assignment makes the realised data
        # and the counterfactuals refer to the same intervention.
        if t < 0:
            prop = np.full((n, N_ARMS), 1.0 / N_ARMS)
            arm = np.zeros(n, dtype=np.int64)          # no campaign ran before the panel
        else:
            prop_t = _assignment_propensity(eta_observable, feats, static, cfg)
            prop_t[static["in_rct"]] = 1.0 / N_ARMS    # clean randomised holdout block
            draw = _sample_rows(rng_assign, prop_t)
            newly_eligible = active & (assigned_arm < 0)
            if newly_eligible.any():
                assigned_arm = np.where(newly_eligible, draw, assigned_arm)
                assigned_prop = np.where(newly_eligible[:, None], prop_t, assigned_prop)
            arm = np.where(assigned_arm >= 0, assigned_arm, 0).astype(np.int64)
            prop = assigned_prop

        # -- realised churn under the ASSIGNED arm ---------------------------------------
        if t >= 0:
            h_assigned = _sigmoid(eta + np.take_along_axis(theta, arm[:, None], axis=1)[:, 0])
            churn_now = active & (rng_out.random(n) < h_assigned)
            if t < T and active.any():
                hazard_trace.append(float(h_assigned[active].mean()))
        else:
            churn_now = np.zeros(n, dtype=bool)

        # -- emit the panel row + analytic ground truth ----------------------------------
        if 0 <= t < T:
            idx = np.flatnonzero(active)
            m = idx.size
            adv = (
                (season[ai : ai + H] - season[ai])[None, :]
                + (cdrift[ai : ai + H] - cdrift[ai])[None, :]
                + (price_beta[ai : ai + H] - price_beta[ai])[None, :]
                * (feats["price_change_pct"][idx] - _REF["price_change"])[:, None]
            )
            base = eta[idx][:, None] + adv                      # (m, H) frozen covariates
            margin = static["monthly_margin"][idx]
            gt_churn = np.empty((m, N_ARMS), dtype=np.float64)
            gt_rmst = np.empty((m, N_ARMS), dtype=np.float64)
            gt_value = np.empty((m, N_ARMS), dtype=np.float64)
            for a in range(N_ARMS):
                haz_a = _sigmoid(base + theta[idx, a][:, None])
                surv_a = np.cumprod(1.0 - haz_a, axis=1)        # surv_a[:, k-1] == S_a(k)
                gt_churn[:, a] = haz_a[:, 0]
                gt_rmst[:, a] = surv_a.sum(axis=1)
                gt_value[:, a] = (surv_a @ disc) * margin

            acc_idx.append(idx.astype(np.int32))
            acc_period.append(np.full(m, t, dtype=np.int32))
            acc_arm.append(arm[idx].astype(np.int8))
            acc_churn_next.append(churn_now[idx].astype(np.int8))
            acc_prop.append(prop[idx])
            acc_gt_churn.append(gt_churn)
            acc_gt_rmst.append(gt_rmst)
            acc_gt_value.append(gt_value)
            for name in NUMERIC_FEATURES:
                acc_feat[name].append(feats[name][idx])

        # -- generate this period's events, then roll the buffers forward -----------------
        act = np.flatnonzero(has_history)
        emit = cfg.emit_events and (t < T)
        day0, dlen = float(day_edges[ai]), float(month_len[ai])

        sub_amount = static["plan_price"][act] * np.exp(rng_evt.normal(0.0, 0.05, act.size))
        sub_day = day0 + static["bill_day_frac"][act] * dlen
        lam_add = np.clip(static["addon_lambda"][act] * np.exp(0.45 * zeff[act, _IX_ORDER] - 0.10), 0.0, 8.0)
        n_add = rng_evt.poisson(lam_add)
        add_owner = np.repeat(act, n_add)
        n_add_tot = int(add_owner.size)
        add_amount = (
            static["plan_price"][add_owner]
            * 0.25
            * np.exp(0.30 * zeff[add_owner, _IX_SPEND] + rng_evt.normal(0.0, 0.35, n_add_tot) - 0.115)
        )
        add_day = day0 + rng_evt.random(n_add_tot) * dlen
        add_cat = rng_evt.integers(1, _N_ORDER_CATEGORIES, n_add_tot)

        lam_sess = np.clip(static["session_lambda"][act] * np.exp(0.55 * zeff[act, _IX_ENGAGE] - 0.15), 0.0, 25.0)
        n_sess = rng_evt.poisson(lam_sess)
        sess_owner = np.repeat(act, n_sess)
        sess_day = day0 + rng_evt.random(sess_owner.size) * dlen

        lam_tick = np.clip(static["ticket_lambda"][act] * np.exp(0.70 * zeff[act, _IX_SERVICE]), 0.0, 6.0)
        n_tick = rng_evt.poisson(lam_tick)
        tick_owner = np.repeat(act, n_tick)
        tick_day = day0 + rng_evt.random(tick_owner.size) * dlen

        lam_pay = np.clip(static["payfail_lambda"][act] * np.exp(0.60 * zeff[act, _IX_PAY]), 0.0, 5.0)
        n_pay = rng_evt.poisson(lam_pay)

        # roll the 12-month ring buffers (assignment, so stale slots are overwritten)
        ones = np.ones(act.size, dtype=np.float64)
        ord_count = np.bincount(act, weights=ones, minlength=n)
        ord_amount = np.bincount(act, weights=sub_amount, minlength=n)
        if n_add_tot:
            ord_count += np.bincount(add_owner, weights=np.ones(n_add_tot), minlength=n)
            ord_amount += np.bincount(add_owner, weights=add_amount, minlength=n)
        state["ord_count"][:, slot] = ord_count
        state["ord_amount"][:, slot] = ord_amount
        mask = np.zeros(n, dtype=np.uint8)
        mask[act] = np.uint8(1)                                   # the subscription charge
        if n_add_tot:
            np.bitwise_or.at(mask, add_owner, (np.uint8(1) << add_cat.astype(np.uint8)))
        state["cat_mask"][:, slot] = mask
        state["sess_count"][:, slot] = np.bincount(act, weights=n_sess.astype(np.float64), minlength=n)
        state["tick_count"][:, slot] = np.bincount(act, weights=n_tick.astype(np.float64), minlength=n)
        state["payfail"][:, slot] = np.bincount(act, weights=n_pay.astype(np.float64), minlength=n)
        state["last_order_day"][act] = sub_day
        if n_add_tot:
            np.maximum.at(state["last_order_day"], add_owner, add_day)
        if sess_owner.size:
            np.maximum.at(state["last_session_day"], sess_owner, sess_day)

        # past offers feed back into the observed discount history (documented)
        depth_now = np.clip(
            static["base_discount"] + 0.10 * (arm == 1) + 0.22 * (arm == 2) + 0.03 * zeff[:, _IX_PRICE],
            0.0,
            0.70,
        )
        state["disc_ema"] = np.where(
            has_history, 0.75 * state["disc_ema"] + 0.25 * depth_now, state["disc_ema"]
        )

        if emit:
            ev_owner += [act, add_owner, sess_owner, tick_owner]
            ev_day += [sub_day, add_day, sess_day, tick_day]
            ev_type += [
                np.zeros(act.size, np.uint8),
                np.zeros(n_add_tot, np.uint8),
                np.ones(sess_owner.size, np.uint8),
                np.full(tick_owner.size, 2, np.uint8),
            ]
            ev_amount += [
                sub_amount,
                add_amount,
                np.zeros(sess_owner.size),
                np.zeros(tick_owner.size),
            ]
            ev_cat += [
                np.zeros(act.size, np.uint8),
                add_cat.astype(np.uint8),
                (_CAT_OFFSET_SESSION + (static["device"][sess_owner] == 2).astype(np.uint8)).astype(np.uint8),
                (_CAT_OFFSET_TICKET + rng_evt.integers(0, len(_TICKET_CATEGORIES), tick_owner.size)).astype(np.uint8),
            ]
            ev_items += [
                np.ones(act.size, np.int32),
                (1 + rng_evt.poisson(0.7, n_add_tot)).astype(np.int32),
                (1 + rng_evt.poisson(2.0, sess_owner.size)).astype(np.int32),
                np.ones(tick_owner.size, np.int32),
            ]

        # -- apply churn last: the customer was active for the whole of period t ----------
        if churn_now.any():
            alive[churn_now] = False
            churn_period[churn_now] = t

    _LOG.info(
        "simulated %d periods | mean monthly hazard %.4f | %d alive at end",
        T,
        float(np.mean(hazard_trace)) if hazard_trace else float("nan"),
        int(alive.sum()),
    )
    return _assemble(cfg, static, cal, churn_period, cum_disc, arm_costs, {
        "idx": acc_idx,
        "period": acc_period,
        "arm": acc_arm,
        "churn_next": acc_churn_next,
        "prop": acc_prop,
        "gt_churn": acc_gt_churn,
        "gt_rmst": acc_gt_rmst,
        "gt_value": acc_gt_value,
        "feat": acc_feat,
        "ev_owner": ev_owner,
        "ev_day": ev_day,
        "ev_type": ev_type,
        "ev_amount": ev_amount,
        "ev_cat": ev_cat,
        "ev_items": ev_items,
    })


# ======================================================================================
# Stage 4 -- assembly
# ======================================================================================
def _assemble(
    cfg: DGPConfig,
    static: dict[str, np.ndarray],
    cal: dict[str, Any],
    churn_period: np.ndarray,
    cum_disc: np.ndarray,
    arm_costs: np.ndarray,
    acc: dict[str, Any],
) -> SimulatedData:
    """Turn the per-period accumulators into the four published frames.

    Parameters
    ----------
    cfg : DGPConfig
        Simulation configuration.
    static : dict of str to numpy.ndarray
        Static population arrays.
    cal : dict of str to Any
        Output of :func:`_calendar`.
    churn_period : numpy.ndarray
        Period in which each customer churned (a sentinel beyond the window if never).
    cum_disc : numpy.ndarray
        ``cum_disc[m] = sum_{k=1..m} (1 + d)^-k``, used for the realised discounted revenue.
    arm_costs : numpy.ndarray
        :data:`~prism.data.schema.ARM_COSTS` as an array.
    acc : dict of str to Any
        Per-period accumulator lists collected by :func:`simulate`.

    Returns
    -------
    SimulatedData
    """
    n, T, H, B = cfg.n_customers, cfg.n_periods, cfg.horizon, cfg.burn_in_periods
    origin_ts = cal["origin_ts"]

    cust_idx = np.concatenate(acc["idx"]).astype(np.int64)
    period = np.concatenate(acc["period"]).astype(np.int64)
    arm = np.concatenate(acc["arm"]).astype(np.int64)
    churn_next = np.concatenate(acc["churn_next"]).astype(np.int64)
    prop = np.concatenate(acc["prop"], axis=0)
    gt_churn = np.concatenate(acc["gt_churn"], axis=0)
    gt_rmst = np.concatenate(acc["gt_rmst"], axis=0)
    gt_value = np.concatenate(acc["gt_value"], axis=0)

    # ---- realised, forward-looking outcomes -------------------------------------------
    # `ttc` is the discrete-time *duration*: a customer who churns during period `t` is under
    # observation for exactly one period, so `ttc >= 1`. This is the convention the person-period
    # expansion in `prism.models.survival` expects for `event_time`.
    ttc = np.maximum(churn_period[cust_idx] + 1 - period, 1)
    window = np.minimum(H, T - period)
    event_observed = (ttc <= window).astype(np.int64)
    event_time = np.minimum(ttc, window).astype(np.float64)
    # `rmst_h` / `value_h` count the periods survived *after* the decision point, i.e.
    # `sum_{k=1..H} 1{alive at t + k}` -- which is `ttc - 1`, capped at the horizon. That is the
    # realised analogue of `gt_rmst_a = sum_k S_a(k)` and `gt_value_a = sum_k S_a(k) * margin *
    # (1 + d)^-k`, so the two are on the same scale (using `ttc` here would inflate both labels
    # by one whole period and one discounted month of revenue for every churner).
    alive_months = np.minimum(ttc - 1, H)
    rmst_h = alive_months.astype(np.float64)
    value_h = static["monthly_revenue"][cust_idx] * cum_disc[alive_months]

    # ---- identifiers -------------------------------------------------------------------
    panel_dates = cal["dates"][B : B + T].to_numpy()
    cohort = _cohort_labels(origin_ts.year, origin_ts.month, static["acq_period"])
    train_end, valid_end = _split_bounds(cfg)
    split = np.where(period <= train_end, "train", np.where(period <= valid_end, "valid", "test")).astype(object)
    block = np.where(static["in_rct"][cust_idx], "rct", "observational").astype(object)

    level_arrays = {
        "plan_tier": np.array(CATEGORICAL_LEVELS["plan_tier"], dtype=object),
        "channel": np.array(CATEGORICAL_LEVELS["channel"], dtype=object),
        "region": np.array(CATEGORICAL_LEVELS["region"], dtype=object),
        "device": np.array(CATEGORICAL_LEVELS["device"], dtype=object),
        "is_autopay": np.array(["no", "yes"], dtype=object),
        "contract_type": np.array(CATEGORICAL_LEVELS["contract_type"], dtype=object),
    }

    panel = pd.DataFrame(
        {
            "customer_id": static["customer_id"][cust_idx],
            "period": period,
            "as_of_date": panel_dates[period],
            "cohort": cohort[cust_idx],
            "split": split,
            "assignment_block": block,
            "arm": arm,
            "treated": (arm > 0).astype(np.int64),
            "offer_cost": arm_costs[arm],
            **{name: np.concatenate(acc["feat"][name]) for name in NUMERIC_FEATURES},
            **{col: level_arrays[col][static[col][cust_idx]] for col in CATEGORICAL_FEATURES},
            "churn_next": churn_next,
            "event_time": event_time,
            "event_observed": event_observed,
            "rmst_h": rmst_h,
            "value_h": value_h,
            "margin_rate": static["margin_rate"][cust_idx],
        }
    )
    panel = enforce_schema(panel, strict=True)[list(PANEL_COLUMNS)]

    # ---- ground truth --------------------------------------------------------------------
    tau_churn = gt_churn[:, 1:] - gt_churn[:, :1]
    tau_rmst = gt_rmst[:, 1:] - gt_rmst[:, :1]
    tau_value = gt_value[:, 1:] - gt_value[:, :1]
    gt_best_arm = np.argmax(gt_value - arm_costs[None, :], axis=1).astype(np.int64)
    gt_oracle = np.maximum(0.0, (tau_value - arm_costs[None, 1:]).max(axis=1))

    gt_data: dict[str, np.ndarray] = {
        "customer_id": static["customer_id"][cust_idx],
        "period": period,
        "gt_segment": np.array(SEGMENTS, dtype=object)[static["segment"]][cust_idx],
        "gt_u": static["u"][cust_idx],
    }
    for a in range(N_ARMS):
        gt_data[f"gt_propensity_{a}"] = prop[:, a]
        gt_data[f"gt_churn_p_{a}"] = gt_churn[:, a]
        gt_data[f"gt_rmst_{a}"] = gt_rmst[:, a]
        gt_data[f"gt_value_{a}"] = gt_value[:, a]
    for a in range(1, N_ARMS):
        gt_data[f"gt_tau_churn_{a}"] = tau_churn[:, a - 1]
        gt_data[f"gt_tau_rmst_{a}"] = tau_rmst[:, a - 1]
        gt_data[f"gt_tau_value_{a}"] = tau_value[:, a - 1]
    gt_data["gt_best_arm"] = gt_best_arm
    gt_data["gt_oracle_net_value"] = gt_oracle
    ground_truth = pd.DataFrame(gt_data)[list(GT_COLUMNS)]

    # ---- customers ------------------------------------------------------------------------
    n_rows = np.bincount(cust_idx, minlength=n).astype(np.int64)
    customers = pd.DataFrame(
        {
            "customer_id": static["customer_id"],
            "gt_segment": np.array(SEGMENTS, dtype=object)[static["segment"]],
            "gt_u": static["u"],
            "frailty": static["frailty"],
            "affluence": static["affluence"],
            "acquisition_period": static["acq_period"],
            "entry_period": static["entry_period"],
            "cohort": cohort,
            "is_legacy": static["is_legacy"].astype(np.int64),
            "assignment_block": np.where(static["in_rct"], "rct", "observational").astype(object),
            **{col: level_arrays[col][static[col]] for col in CATEGORICAL_FEATURES},
            "plan_price": static["plan_price"],
            "monthly_revenue": static["monthly_revenue"],
            "margin_rate": static["margin_rate"],
            "monthly_margin": static["monthly_margin"],
            "base_discount": static["base_discount"],
            "session_lambda": static["session_lambda"],
            "ticket_lambda": static["ticket_lambda"],
            "addon_lambda": static["addon_lambda"],
            "churn_period": np.where(churn_period <= T + H - 1, churn_period, -1).astype(np.int64),
            "n_panel_rows": n_rows,
        }
    )

    # ---- events ---------------------------------------------------------------------------
    if acc["ev_owner"]:
        owner = np.concatenate(acc["ev_owner"]).astype(np.int64)
        day = np.concatenate(acc["ev_day"]).astype(np.float64)
        etype = np.concatenate(acc["ev_type"]).astype(np.int64)
        amount = np.concatenate(acc["ev_amount"]).astype(np.float64)
        ecat = np.concatenate(acc["ev_cat"]).astype(np.int64)
        items = np.concatenate(acc["ev_items"]).astype(np.int64)
        order = np.lexsort((day, owner))
        owner, day, etype, amount, ecat, items = (
            owner[order], day[order], etype[order], amount[order], ecat[order], items[order]
        )
        events = pd.DataFrame(
            {
                "customer_id": static["customer_id"][owner],
                "event_ts": origin_ts + pd.to_timedelta(day, unit="D"),
                "event_type": np.array(EVENT_TYPES, dtype=object)[etype],
                "amount": amount,
                "category": np.array(_ALL_CATEGORIES, dtype=object)[ecat],
                "n_items": items,
            }
        )
    else:
        events = pd.DataFrame(
            {
                "customer_id": pd.Series(dtype=object),
                "event_ts": pd.Series(dtype="datetime64[ns]"),
                "event_type": pd.Series(dtype=object),
                "amount": pd.Series(dtype="float64"),
                "category": pd.Series(dtype=object),
                "n_items": pd.Series(dtype="int64"),
            }
        )

    return SimulatedData(
        panel=panel.reset_index(drop=True),
        customers=customers,
        events=events.reset_index(drop=True),
        ground_truth=ground_truth.reset_index(drop=True),
        config=cfg,
    )


if __name__ == "__main__":  # pragma: no cover - smoke test
    import tempfile
    import time
    from dataclasses import replace

    from prism.data.schema import PANEL_DTYPES  # SEGMENT_SHARES is already module-level
    from prism.utils.validation import PANEL_CONTRACT

    t0 = time.perf_counter()
    cfg = DGPConfig(n_customers=2500, n_periods=14, horizon=8, random_state=7)
    cfg_before = cfg.to_dict()
    sim = simulate(cfg)
    elapsed = time.perf_counter() - t0

    panel, gt, ev, cust = sim.panel, sim.ground_truth, sim.events, sim.customers
    H, T = cfg.horizon, cfg.n_periods
    nrow = len(panel)
    arm = panel["arm"].to_numpy()
    per = panel["period"].to_numpy()
    costs = np.asarray(ARM_COSTS, dtype=np.float64)

    # ---- 0. the caller's config is not mutated -------------------------------------------
    assert cfg.to_dict() == cfg_before, "simulate() mutated the caller's DGPConfig"

    # ---- 1. schema / alignment -------------------------------------------------------------
    assert list(panel.columns) == list(PANEL_COLUMNS), "panel columns deviate from PANEL_COLUMNS"
    assert list(gt.columns) == list(GT_COLUMNS), "ground_truth columns deviate from GT_COLUMNS"
    assert len(panel) == len(gt), "panel and ground_truth are not row-aligned"
    assert (panel["customer_id"].to_numpy() == gt["customer_id"].to_numpy()).all()
    assert (panel["period"].to_numpy() == gt["period"].to_numpy()).all()
    # `enforce_schema` renders PANEL_DTYPES' "string" columns as object-of-`str` (SPEC 2.2 says
    # `str`), so compare those by element type and everything else by dtype exactly.
    bad_dtype = {}
    for c in PANEL_COLUMNS:
        if PANEL_DTYPES[c] in ("object", "string"):
            if not panel[c].map(lambda v: isinstance(v, str)).all():
                bad_dtype[c] = "not object-of-str"
        elif str(panel[c].dtype) != PANEL_DTYPES[c]:
            bad_dtype[c] = str(panel[c].dtype)
    assert not bad_dtype, f"dtype drift: {bad_dtype}"
    assert int(panel.isna().sum().sum()) == 0, "the gold panel must have no nulls"
    assert not panel.duplicated(["customer_id", "period"]).any(), "duplicate (customer_id, period)"
    assert panel["period"].between(0, T - 1).all(), "period outside 0..n_periods-1"

    # ---- 2. treatment bookkeeping ------------------------------------------------------------
    assert np.isin(arm, np.arange(N_ARMS)).all(), "arm outside 0..N_ARMS-1"
    assert np.bincount(arm, minlength=N_ARMS).min() > 0, "some arm is never assigned"
    assert (panel["treated"].to_numpy() == (arm > 0)).all(), "treated != (arm > 0)"
    assert np.allclose(panel["offer_cost"], costs[arm]), "offer_cost != ARM_COSTS[arm]"
    assert set(panel["split"].unique()) == {"train", "valid", "test"}, "split must be 3-valued"
    assert set(panel["assignment_block"].unique()) <= {"observational", "rct"}
    lo = panel.groupby("split", observed=True)["period"].min()
    hi = panel.groupby("split", observed=True)["period"].max()
    assert hi["train"] < lo["valid"] <= hi["valid"] < lo["test"], "split is not temporal"
    dmap = panel.drop_duplicates("period").set_index("period")["as_of_date"].sort_index()
    assert dmap.is_monotonic_increasing and dmap.is_unique, "as_of_date is not monotone in period"

    # ---- 3. the panel is a proper per-customer spell --------------------------------------------
    srt = panel.sort_values(["customer_id", "period"], kind="stable")
    cid = srt["customer_id"].to_numpy()
    sper = srt["period"].to_numpy()
    same = cid[1:] == cid[:-1]
    assert (np.diff(sper)[same] == 1).all(), "a customer's periods are not contiguous"
    is_last = np.append(~same, True)
    assert (srt["churn_next"].to_numpy()[~is_last] == 0).all(), "churn_next==1 before a customer's last row"

    # ---- 4. overlap / propensities ------------------------------------------------------------
    prop = gt[[f"gt_propensity_{a}" for a in range(N_ARMS)]].to_numpy()
    lo_p, hi_p = PROPENSITY_CLIP
    assert prop.min() >= lo_p, f"propensity below {lo_p}: {prop.min()}"
    assert prop.max() <= hi_p, f"propensity above {hi_p}: {prop.max()}"
    assert np.allclose(prop.sum(axis=1), 1.0, atol=1e-12), "propensities do not sum to 1"
    rct = (panel["assignment_block"] == "rct").to_numpy()
    assert rct.any() and np.allclose(prop[rct], 1.0 / N_ARMS), "the RCT block is not uniformly randomised"
    assert prop[~rct].std() > 1e-3, "the observational block is not confounded at all"

    # ---- 5. ground-truth probability / survival maths --------------------------------------------
    gch = gt[[f"gt_churn_p_{a}" for a in range(N_ARMS)]].to_numpy()
    grm = gt[[f"gt_rmst_{a}" for a in range(N_ARMS)]].to_numpy()
    gvl = gt[[f"gt_value_{a}" for a in range(N_ARMS)]].to_numpy()
    assert (gch > 0.0).all() and (gch < 1.0).all(), "a counterfactual hazard escaped (0, 1)"
    assert (grm >= 0.0).all() and (grm <= H + 1e-9).all(), "gt_rmst outside [0, horizon]"
    assert (gvl >= 0.0).all(), "gt_value went negative"
    # S(k) = prod(1 - h) is monotone non-increasing by construction; the observable consequences
    # are that RMST cannot exceed the horizon and that value is bracketed by the discount factors
    # -- which is what pins the discount exponent to k = 1..H rather than 0..H-1.
    d = cfg.monthly_discount_rate
    cum_d = np.concatenate([[0.0], np.cumsum((1.0 + d) ** -np.arange(1, H + 1, dtype=np.float64))])
    margin = cust.set_index("customer_id")["monthly_margin"].reindex(panel["customer_id"]).to_numpy()
    for a in range(N_ARMS):
        upper = margin * grm[:, a] * (1.0 + d) ** -1.0
        lower = margin * grm[:, a] * (1.0 + d) ** -float(H)
        assert (gvl[:, a] <= upper + 1e-6).all(), f"gt_value_{a} too large: wrong discount exponent?"
        assert (gvl[:, a] >= lower - 1e-6).all(), f"gt_value_{a} too small: wrong discount exponent?"
        assert (gvl[:, a] <= margin * cum_d[H] + 1e-6).all(), f"gt_value_{a} exceeds the perfect-survival bound"

    # ---- 6. contrast algebra is exact --------------------------------------------------------------
    for a in range(1, N_ARMS):
        assert np.allclose(gt[f"gt_tau_churn_{a}"], gch[:, a] - gch[:, 0]), f"gt_tau_churn_{a}"
        assert np.allclose(gt[f"gt_tau_rmst_{a}"], grm[:, a] - grm[:, 0]), f"gt_tau_rmst_{a}"
        assert np.allclose(gt[f"gt_tau_value_{a}"], gvl[:, a] - gvl[:, 0]), f"gt_tau_value_{a}"
    tau_v = gvl[:, 1:] - gvl[:, :1]
    assert (np.argmax(gvl - costs[None, :], axis=1) == gt["gt_best_arm"].to_numpy()).all(), "gt_best_arm"
    assert np.allclose(
        gt["gt_oracle_net_value"], np.maximum(0.0, (tau_v - costs[None, 1:]).max(axis=1))
    ), "gt_oracle_net_value"
    assert (gt["gt_oracle_net_value"] >= 0.0).all(), "the oracle must be able to do nothing"
    assert gt["gt_best_arm"].nunique() >= 2, "the best arm must be customer-specific"
    assert tau_v.std(axis=0).min() > 0.0, "treatment effects are homogeneous"

    # ---- 7. realised labels are on the SAME scale as the ground truth ------------------------------
    rmst_h = panel["rmst_h"].to_numpy()
    churn_next = panel["churn_next"].to_numpy()
    window = np.minimum(H, T - per)
    assert (rmst_h >= 0.0).all() and (rmst_h <= H).all(), "rmst_h outside [0, horizon]"
    assert ((rmst_h == 0.0).astype(np.int64) == churn_next).all(), (
        "rmst_h must be 0 exactly when the customer churns in the decision period -- an "
        "off-by-one here inflates every realised label by a whole period"
    )
    rev = cust.set_index("customer_id")["monthly_revenue"].reindex(panel["customer_id"]).to_numpy()
    assert np.allclose(panel["value_h"], rev * cum_d[rmst_h.astype(np.int64)]), (
        "value_h != monthly_revenue * sum_{k=1..rmst_h} (1 + d)^-k (wrong discount exponent)"
    )
    assert np.allclose(panel["event_time"], np.minimum(rmst_h + 1.0, window)), (
        "event_time must be the duration convention rmst_h + 1, censored at min(horizon, T - t)"
    )
    assert (panel["event_time"].to_numpy() >= 1.0).all(), "event_time below one period"
    assert np.isin(panel["event_observed"].to_numpy(), (0, 1)).all()
    assert (
        ((panel["event_time"].to_numpy() == 1.0) & (panel["event_observed"].to_numpy() == 1)).astype(np.int64)
        == churn_next
    ).all(), "event_observed disagrees with churn_next"

    # ---- 8. churn_next is calibrated against the assigned arm's counterfactual hazard --------------
    h_assigned = np.take_along_axis(gch, arm[:, None], axis=1)[:, 0]
    se = math.sqrt(h_assigned.mean() * (1.0 - h_assigned.mean()) / nrow)
    assert abs(churn_next.mean() - h_assigned.mean()) < 5.0 * se, (
        f"realised churn {churn_next.mean():.4f} is not drawn from gt_churn_p[assigned arm] "
        f"{h_assigned.mean():.4f}"
    )
    assert 0.015 < h_assigned.mean() < 0.10, f"monthly hazard outside the design band: {h_assigned.mean():.4f}"

    # ---- 9. the four segments and their economics ----------------------------------------------------
    seg = cust["gt_segment"]
    assert set(seg.unique()) == set(SEGMENTS), f"missing segments: {set(SEGMENTS) - set(seg.unique())}"
    shares = seg.value_counts(normalize=True).reindex(list(SEGMENTS)).to_numpy()
    assert np.abs(shares - np.asarray(SEGMENT_SHARES)).max() < 0.05, f"segment shares drifted: {shares}"
    tau_cols = [f"gt_tau_value_{a}" for a in range(1, N_ARMS)]
    by_seg = gt.groupby("gt_segment", observed=True)[tau_cols].mean()
    assert by_seg.loc["persuadable"].mean() > 0.0, "persuadables must gain value from an offer"
    assert by_seg.loc["sleeping_dog"].mean() < 0.0, "sleeping dogs must lose value from an offer"
    ch_seg = gt.groupby("gt_segment", observed=True)[[f"gt_tau_churn_{a}" for a in range(1, N_ARMS)]].mean()
    assert (ch_seg.loc["persuadable"] < 0).all(), "an offer must reduce churn for persuadables"
    assert (ch_seg.loc["sleeping_dog"] > 0).all(), "an offer must increase churn for sleeping dogs"
    assert abs(np.corrcoef(gt["gt_u"].to_numpy()[~rct], panel["treated"].to_numpy()[~rct])[0, 1]) > 0.05, (
        "the hidden confounder does not actually confound assignment"
    )

    # ---- 10. data contract ------------------------------------------------------------------------------
    result = PANEL_CONTRACT.validate(panel)
    if not result.passed:
        print(result.summary().to_string(index=False))
    assert result.passed is True, "PANEL_CONTRACT failed"

    # ---- 11. the event log rebuilds the event-derived features, with NO leakage -------------------------
    assert set(ev["event_type"].unique()) <= set(EVENT_TYPES)
    assert set(ev["category"].unique()) <= set(_ALL_CATEGORIES)
    assert (ev.loc[ev["event_type"] != "order", "amount"] == 0.0).all(), "non-orders must have amount 0"
    assert (ev.loc[ev["event_type"] == "order", "amount"] > 0.0).all(), "orders must have a positive amount"
    assert (ev["n_items"] >= 1).all()
    spine = panel[["customer_id", "as_of_date"]].copy()
    spine["_row"] = np.arange(nrow)
    joined = spine.merge(
        ev.rename(columns={"customer_id": "_cid"}), left_on="customer_id", right_on="_cid", how="left"
    )
    strictly_past = (joined["event_ts"] < joined["as_of_date"]).fillna(False)
    rebuild_corr: dict[str, float] = {}
    for col, etype, win, how in (
        ("monetary_12m", "order", 366, "sum"),
        ("frequency_12m", "order", 366, "count"),
        ("sessions_30d", "session", 31, "count"),
        ("support_tickets_90d", "support_ticket", 92, "count"),
    ):
        sel = joined[
            strictly_past
            & (joined["event_type"] == etype)
            & (joined["event_ts"] >= joined["as_of_date"] - pd.Timedelta(days=win))
        ]
        agg = sel.groupby("_row")["amount"].sum() if how == "sum" else sel.groupby("_row").size()
        got = np.zeros(nrow, dtype=np.float64)
        got[agg.index.to_numpy()] = agg.to_numpy(dtype=np.float64)
        c = float(np.corrcoef(panel[col].to_numpy(), got)[0, 1])
        rebuild_corr[col] = c
        assert c > 0.99, f"{col} does not rebuild point-in-time from the event log (corr={c:.4f})"
    # A genuine leak would let the feature predict NEXT month better than last month's own spend.
    fut = joined[
        (joined["event_type"] == "order")
        & (joined["event_ts"] >= joined["as_of_date"])
        & (joined["event_ts"] < joined["as_of_date"] + pd.Timedelta(days=31))
    ]
    fs = np.zeros(nrow)
    fagg = fut.groupby("_row")["amount"].sum()
    fs[fagg.index.to_numpy()] = fagg.to_numpy(dtype=np.float64)
    prev = joined[
        strictly_past
        & (joined["event_type"] == "order")
        & (joined["event_ts"] >= joined["as_of_date"] - pd.Timedelta(days=31))
    ]
    ps = np.zeros(nrow)
    pagg = prev.groupby("_row")["amount"].sum()
    ps[pagg.index.to_numpy()] = pagg.to_numpy(dtype=np.float64)
    c_feat = float(np.corrcoef(panel["monetary_12m"].to_numpy(), fs)[0, 1])
    c_past = float(np.corrcoef(ps, fs)[0, 1])
    assert c_feat <= c_past + 1e-9, (
        f"leakage: monetary_12m predicts next-month spend ({c_feat:.4f}) better than last "
        f"month's own spend does ({c_past:.4f})"
    )

    # ---- 12. save / load round-trip ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        sim.save(tmp)
        back = SimulatedData.load(tmp)
        assert back.config.to_dict() == cfg.to_dict(), "config did not survive the round-trip"
        for name in ("panel", "customers", "events", "ground_truth"):
            pd.testing.assert_frame_equal(getattr(back, name), getattr(sim, name), check_dtype=False)
        assert list(back.panel.columns) == list(PANEL_COLUMNS)

    # ---- 13. determinism: same seed => bit-identical; different seed => different ---------------------
    small = DGPConfig(n_customers=400, n_periods=6, horizon=4, random_state=3)
    a_run, b_run = simulate(small), simulate(small)
    for name in ("panel", "customers", "events", "ground_truth"):
        pd.testing.assert_frame_equal(getattr(a_run, name), getattr(b_run, name))
    c_run = simulate(replace(small, random_state=4))
    k = min(len(a_run.ground_truth), len(c_run.ground_truth))
    assert not np.allclose(
        a_run.ground_truth["gt_value_3"].to_numpy()[:k], c_run.ground_truth["gt_value_3"].to_numpy()[:k]
    ), "a different seed produced the same data"

    # ---- 14. confounding_strength = 0 collapses to a pure RCT -------------------------------------------
    rand = simulate(DGPConfig(n_customers=400, n_periods=5, horizon=3, confounding_strength=0.0, random_state=2))
    pr0 = rand.ground_truth[[f"gt_propensity_{a}" for a in range(N_ARMS)]].to_numpy()
    assert np.abs(pr0 - 1.0 / N_ARMS).max() < 1e-12, "confounding_strength=0 is not uniform assignment"

    # ---- 15. report ---------------------------------------------------------------------------------------
    table = by_seg.reindex(list(SEGMENTS)).round(2)
    table.columns = [ARM_NAMES[a] for a in range(1, N_ARMS)]
    print("")
    print(f"rows: panel={nrow:,}  gt={len(gt):,}  events={len(ev):,}  customers={len(cust):,}")
    print(f"mean baseline monthly hazard (control) : {gch[:, 0].mean():.4f}")
    print(f"churn_next rate                        : {churn_next.mean():.4f}")
    print(f"arm shares                             : {np.round(np.bincount(arm, minlength=N_ARMS) / nrow, 4)}")
    print(f"rct share of rows                      : {rct.mean():.3f}")
    print(f"propensity range                       : [{prop.min():.4f}, {prop.max():.4f}]")
    print(f"gt_best_arm shares                     : {np.round(np.bincount(gt['gt_best_arm'], minlength=N_ARMS) / nrow, 4)}")
    print(f"mean oracle net value                  : {gt['gt_oracle_net_value'].mean():.2f}")
    print(f"mean rmst_h vs mean gt_rmst_0          : {rmst_h.mean():.3f} vs {grm[:, 0].mean():.3f}")
    print(f"naive churn untreated / treated        : {churn_next[arm == 0].mean():.4f} / {churn_next[arm > 0].mean():.4f}")
    print(f"true mean gt_tau_churn_2               : {gt['gt_tau_churn_2'].mean():+.4f}  (the naive sign is wrong)")
    print("point-in-time rebuild corr from events : " + ", ".join(f"{k2}={v:.4f}" for k2, v in rebuild_corr.items()))
    print(f"splits                                 : {panel['split'].value_counts().to_dict()}")
    print("")
    print("mean gt_tau_value by segment and arm (currency, margin basis)")
    print(table.to_string())
    print("")
    print(sim.summary().to_string(index=False))
    print(
        f"\ndgp.py OK -- 15 invariant blocks asserted in {time.perf_counter() - t0:.1f}s "
        f"(simulate() itself {elapsed:.1f}s)"
    )
