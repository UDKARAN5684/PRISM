"""Realism injection and the repair layer that undoes it.

A simulator that emits a perfect panel proves nothing about a data-science pipeline: the
hard part of retention modelling is that the warehouse hands you *dirt*. This module owns
both halves of that story so they can be tested against each other:

``inject_realism``
    Takes the clean gold panel and degrades it the way a real CRM/billing export degrades
    it -- MAR and MNAR missingness, heavy-tailed sensor/currency outliers, re-delivered
    records (exact and near duplicates), free-text drift in the categorical codes, rows
    that physically landed after the decision timestamp, and a schema wobble where an
    upstream team renames a column part-way through the history.

``clean_panel``
    Repairs every one of those defects, deterministically and *reportably*: it returns a
    tidy audit frame (``step, column, n_affected, action``) so the cleaning is a first-class
    artifact rather than an invisible side effect.

Two invariants hold the pair together:

1. **Injection never touches identity, treatment or outcome.** Only feature columns are
   degraded, so a downstream causal estimate cannot be corrupted by the mess -- the mess
   only makes the *features* harder to trust, which is the point.
2. **Cleaning never imputes.** Missingness is measured and reported but left in place;
   imputation is a modelling decision that belongs to
   :func:`prism.data.features.build_design_matrix`, where it can be fit on train and
   applied to test without leakage.

Notes
-----
:mod:`prism.data.dgp` is imported only under :data:`typing.TYPE_CHECKING`; ``config`` is
read duck-typed via :func:`_cfg`, so this module imports and runs even when the simulator
is unavailable, and accepts any config object exposing the ``DGPConfig`` field names
(``prism.config.SimConfig`` works too).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.special import expit

from prism.data.schema import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_LEVELS,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    OUTCOME_COLUMNS,
    TREATMENT_COLUMNS,
    enforce_schema,
)
from prism.utils.logging import get_logger
from prism.utils.seeds import as_rng
from prism.utils.validation import PANEL_CONTRACT

if TYPE_CHECKING:  # pragma: no cover - typing only, dgp may not exist yet
    from prism.data.dgp import DGPConfig

__all__ = [
    "PLAUSIBLE_RANGES",
    "CATEGORICAL_SYNONYMS",
    "MAR_DRIVERS",
    "MNAR_DIRECTION",
    "OUTLIER_KIND",
    "WOBBLE_RENAMES",
    "WOBBLE_N_PERIODS",
    "LATE_ARRIVING_COLUMN",
    "PROTECTED_COLUMNS",
    "REPORT_COLUMNS",
    "inject_realism",
    "clean_panel",
]

log = get_logger("data.messy")

# --------------------------------------------------------------------------------------
# Domain knowledge -- the constants that make the repair auditable
# --------------------------------------------------------------------------------------

#: Domain-valid range for every numeric feature, as ``(low, high)`` inclusive bounds.
#:
#: These are *plausibility* bounds, not distributional ones: they encode what the business
#: definition of the field permits (a customer cannot have been seen -3 days ago, monetary
#: value over a year cannot be half a million for a consumer subscription), deliberately
#: loose enough that no legitimate observation is ever clipped. Anything outside is a
#: corrupt record, and is winsorized to the bound rather than deleted -- deleting rows
#: would silently change the population the causal estimate is identified on.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "tenure_months": (0.0, 600.0),
    "recency_days": (0.0, 3650.0),
    "frequency_12m": (0.0, 1000.0),
    "monetary_12m": (0.0, 500_000.0),
    "avg_order_value": (0.0, 50_000.0),
    "n_categories_12m": (0.0, 100.0),
    "sessions_30d": (0.0, 3000.0),
    "days_since_last_session": (0.0, 3650.0),
    "engagement_score": (-10.0, 100.0),
    "support_tickets_90d": (0.0, 200.0),
    "nps": (-100.0, 100.0),
    "payment_failures_12m": (0.0, 100.0),
    "discount_depth_hist": (0.0, 5.0),
    "price_change_pct": (-100.0, 100.0),
    "competitor_promo_intensity": (-5.0, 100.0),
    "seasonality_index": (-5.0, 10.0),
    "basket_diversity": (0.0, 50.0),
}

#: Legacy / vendor / free-text codes that must be folded onto :data:`CATEGORICAL_LEVELS`.
#: Keys are already normalised (lower-cased, whitespace and hyphens collapsed to ``_``).
CATEGORICAL_SYNONYMS: dict[str, dict[str, str]] = {
    "plan_tier": {"ent": "enterprise", "enterprise_plus": "enterprise", "basic_v2": "basic", "premium": "pro", "tier_1": "basic"},
    "channel": {"seo": "organic", "organic_search": "organic", "ppc": "paid", "cpc": "paid", "sem": "paid", "affiliate": "partner", "word_of_mouth": "referral", "refer_a_friend": "referral"},
    "region": {"n": "north", "nth": "north", "s": "south", "e": "east", "w": "west", "northern": "north"},
    "device": {"apple": "ios", "iphone": "ios", "ipad": "ios", "ios_app": "ios", "droid": "android", "google_play": "android", "browser": "web", "desktop": "web"},
    "is_autopay": {"y": "yes", "true": "yes", "1": "yes", "t": "yes", "n": "no", "false": "no", "0": "no", "f": "no"},
    "contract_type": {"mo": "monthly", "month": "monthly", "m2m": "monthly", "yearly": "annual", "year": "annual", "12m": "annual", "anual": "annual"},
}

#: Columns whose missingness is **MAR**: driven by *other*, observed features. Recovery is
#: possible in principle, which is exactly why the missing-indicator trick works downstream.
MAR_DRIVERS: dict[str, tuple[str, ...]] = {
    "recency_days": ("engagement_score", "sessions_30d"),
    "avg_order_value": ("frequency_12m", "tenure_months"),
    "basket_diversity": ("n_categories_12m", "frequency_12m"),
    "support_tickets_90d": ("payment_failures_12m", "tenure_months"),
    "is_autopay": ("payment_failures_12m",),
}

#: Columns whose missingness is **MNAR**: driven by the value that is being hidden.
#: ``-1`` means low values disappear (detractors never answer the survey), ``+1`` means
#: high values disappear (enterprise revenue is redacted from the shared extract).
MNAR_DIRECTION: dict[str, float] = {
    "nps": -1.0,
    "engagement_score": -1.0,
    "monetary_12m": +1.0,
}

#: How each corruptible numeric blows up. ``"scale"`` is a heavy-tailed multiplicative
#: fault (unit mix-ups, cents-vs-currency, a bot session storm); ``"negate"`` is a sign
#: fault from a bad date subtraction, which is impossible by construction.
OUTLIER_KIND: dict[str, str] = {
    "monetary_12m": "scale",
    "avg_order_value": "scale",
    "sessions_30d": "scale",
    "support_tickets_90d": "scale",
    "recency_days": "negate",
    "days_since_last_session": "negate",
}

#: The upstream schema wobble: ``canonical -> name it arrives under in the last periods``.
WOBBLE_RENAMES: dict[str, str] = {"nps": "net_promoter_score"}

#: Number of trailing periods affected by the schema wobble.
WOBBLE_N_PERIODS: int = 3

#: Boolean column marking records that physically landed after ``as_of_date``.
LATE_ARRIVING_COLUMN: str = "late_arriving"

#: Numerics eligible for the near-duplicate jitter (values a re-delivery would recompute).
_JITTER_COLUMNS: tuple[str, ...] = (
    "monetary_12m",
    "avg_order_value",
    "engagement_score",
    "sessions_30d",
    "basket_diversity",
)

#: Never degraded by :func:`inject_realism`: identity, treatment and outcome stay pristine.
PROTECTED_COLUMNS: tuple[str, ...] = ID_COLUMNS + TREATMENT_COLUMNS + OUTCOME_COLUMNS

#: Exact column order of the cleaning report.
REPORT_COLUMNS: tuple[str, ...] = ("step", "column", "n_affected", "action")

_KEY: tuple[str, str] = ("customer_id", "period")
_ORDER_COL = "__prism_row_order__"
_TRUTHY = {"1", "true", "t", "y", "yes", "late"}
_FALSY = {"0", "false", "f", "n", "no", "on_time", ""}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _cfg(config: Any, name: str, default: float | int) -> Any:
    """Read ``name`` off a duck-typed config object, falling back to ``default``.

    Parameters
    ----------
    config : object or None
        A ``DGPConfig`` (or anything exposing the same field names, e.g.
        ``prism.config.SimConfig``). ``None`` yields ``default``.
    name : str
        Attribute name.
    default : float or int
        Value used when the attribute is absent or ``None``.

    Returns
    -------
    Any
        The attribute value, or ``default``.
    """
    if config is None:
        return default
    value = getattr(config, name, None)
    return default if value is None else value


def _zscore(values: pd.Series) -> np.ndarray:
    """Standardise a series to mean 0 / sd 1, mapping non-finite entries to 0.

    Parameters
    ----------
    values : pandas.Series
        Any numeric-coercible series.

    Returns
    -------
    numpy.ndarray
        Float64 array of the same length, with zeros where the input was missing.
    """
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    mu = float(x[finite].mean())
    sd = float(x[finite].std())
    sd = sd if sd > 1e-12 else 1.0
    z = np.where(finite, (x - mu) / sd, 0.0)
    return np.clip(z, -6.0, 6.0)


def _calibrated_mask(rng: np.random.Generator, score: np.ndarray, target_rate: float) -> np.ndarray:
    """Draw a Bernoulli mask whose probability follows ``score`` but averages ``target_rate``.

    The intercept of ``sigmoid(score + b)`` is bisected so that the *expected* missing rate
    matches the configured one regardless of how strong the dependence is. This decouples
    "how much is missing" (a config knob) from "what drives it" (the mechanism), which is
    what makes MAR and MNAR comparable at equal severity.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    score : numpy.ndarray
        Per-row log-odds contribution (centred internally).
    target_rate : float
        Desired mean of the resulting mask, in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Boolean mask of the same length as ``score``.
    """
    n = score.shape[0]
    if n == 0 or target_rate <= 0.0:
        return np.zeros(n, dtype=bool)
    if target_rate >= 1.0:
        return np.ones(n, dtype=bool)
    s = np.nan_to_num(score, nan=0.0, posinf=6.0, neginf=-6.0)
    s = s - s.mean()
    # The intercept only needs the *distribution* of scores, so bisect on a deterministic
    # stride of at most 50k points; on a million-row panel that is a 20x saving with no
    # measurable loss of calibration (and no extra randomness).
    probe = s if n <= 50_000 else s[:: (n // 50_000) + 1]
    lo, hi = -40.0, 40.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if float(expit(probe + mid).mean()) < target_rate:
            lo = mid
        else:
            hi = mid
    p = expit(s + 0.5 * (lo + hi))
    return rng.random(n) < p


def _pick_rows(rng: np.random.Generator, eligible: np.ndarray, rate: float) -> np.ndarray:
    """Choose row positions to corrupt at ``rate``, guaranteeing at least one when possible.

    Small smoke panels would otherwise round a 0.4% rate down to zero rows and leave the
    cleaner with nothing to prove, so the count is floored at one whenever ``rate > 0`` and
    at least one row is eligible.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    eligible : numpy.ndarray
        Integer positions that may be corrupted.
    rate : float
        Target fraction of the *eligible* rows.

    Returns
    -------
    numpy.ndarray
        Integer positions, sorted, possibly empty.
    """
    n = int(eligible.shape[0])
    if n == 0 or rate <= 0.0:
        return np.empty(0, dtype=np.int64)
    k = min(n, max(1, int(round(rate * n))))
    return np.sort(rng.choice(eligible, size=k, replace=False))


def _normalise_token(value: Any, column: str) -> Any:
    """Fold one raw categorical value onto its canonical level.

    Lower-cases, strips ASCII and non-breaking whitespace, collapses internal separators to
    ``_``, then applies :data:`CATEGORICAL_SYNONYMS`. Values that still fail to land on a
    declared level become missing rather than silently creating a new category -- an unknown
    code is unknown information, and inventing a level would poison the one-hot encoder.

    Parameters
    ----------
    value : Any
        Raw cell value.
    column : str
        Categorical feature name, used to pick the synonym table and the level set.

    Returns
    -------
    Any
        A canonical level string, or ``None`` when the value is missing/unmappable.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or value is pd.NA:
        return None
    token = str(value).replace("\xa0", " ").replace("\u200b", "").strip().lower()
    token = re.sub(r"[\s\-./]+", "_", token).strip("_")
    token = re.sub(r"_+", "_", token)
    if not token or token in {"na", "nan", "none", "null", "unknown", "?"}:
        return None
    token = CATEGORICAL_SYNONYMS.get(column, {}).get(token, token)
    return token if token in CATEGORICAL_LEVELS.get(column, ()) else None


def _token_key(value: Any) -> str:
    """Render a cell as a comparable string where every flavour of missing collapses to one.

    ``None``, ``numpy.nan`` and ``pandas.NA`` must compare *equal* to each other, otherwise
    the normalisation report counts "missing stayed missing" as a repair.

    Parameters
    ----------
    value : Any
        Cell value.

    Returns
    -------
    str
        ``"\\x00"`` for any missing value, else ``str(value)``.
    """
    if value is None or value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return "\x00"
    return str(value)


def _as_bool(series: pd.Series) -> np.ndarray:
    """Coerce a mixed bool/int/string flag column to a plain boolean array.

    Parameters
    ----------
    series : pandas.Series
        Flag column, possibly holding ``True``/``"Y"``/``1``/``NaN``.

    Returns
    -------
    numpy.ndarray
        Boolean array; anything unrecognised or missing is ``False``.
    """
    def one(v: Any) -> bool:
        if v is None or v is pd.NA or (isinstance(v, float) and not np.isfinite(v)):
            return False
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v) != 0.0
        return str(v).strip().lower() in _TRUTHY

    return series.map(one).to_numpy(dtype=bool)


# --------------------------------------------------------------------------------------
# 1. Realism injection
# --------------------------------------------------------------------------------------
def inject_realism(
    panel: pd.DataFrame,
    config: DGPConfig | None = None,
    random_state: int | np.random.Generator | None = None,
    *,
    dirt_rate: float = 0.12,
    late_arriving_rate: float = 0.02,
    outlier_scale: float = 50.0,
) -> pd.DataFrame:
    """Degrade a clean gold panel into something a real warehouse would hand you.

    Seven independent faults are applied, each to feature columns only. Identity
    (``customer_id``, ``period``, ``as_of_date``, ``cohort``, ``split``,
    ``assignment_block``), treatment (``arm``, ``treated``, ``offer_cost``) and every
    outcome column are left bit-identical, so the ground truth stays trustworthy while the
    *features* become untrustworthy -- the realistic asymmetry.

    1. **MAR missingness** -- for each column in :data:`MAR_DRIVERS`, the probability of
       being hidden is a logistic function of *other* observed features.
    2. **MNAR missingness** -- for each column in :data:`MNAR_DIRECTION`, the probability
       depends on the hidden value itself (detractor ``nps`` vanishes, enterprise
       ``monetary_12m`` is redacted), at strength ``config.mnar_strength``.
    3. **Heavy-tailed outliers** -- at ``config.outlier_rate``, per
       :data:`OUTLIER_KIND`: a Pareto-amplified multiplicative blow-up (``monetary_12m``
       times ~``outlier_scale`` and up) or an impossible negative from a bad date subtraction.
    4. **Duplicates** -- at ``config.duplicate_rate``, half exact re-deliveries and half
       near-duplicates where one numeric was recomputed slightly differently. Copies are
       inserted directly after their source row, as a replayed feed would arrive.
    5. **Categorical dirt** -- mixed case, stray leading/trailing (and non-breaking)
       whitespace, and legacy synonym codes drawn from :data:`CATEGORICAL_SYNONYMS`.
    6. **Late-arriving flag** -- a boolean :data:`LATE_ARRIVING_COLUMN`, concentrated in
       later periods, marking rows that landed *after* the decision timestamp. These are a
       leakage source, not a nuisance: the cleaner must drop them.
    7. **Schema wobble** -- in the last :data:`WOBBLE_N_PERIODS` periods present in the
       frame, ``nps`` arrives under the name ``net_promoter_score`` (see
       :data:`WOBBLE_RENAMES`), so the canonical column is null exactly there.

    Parameters
    ----------
    panel : pandas.DataFrame
        The clean gold panel (see SPEC section 2.2). Never mutated.
    config : DGPConfig or None, optional
        Supplies ``missing_rate``, ``mnar_strength``, ``outlier_rate`` and
        ``duplicate_rate``. Read duck-typed, so ``prism.config.SimConfig`` also works;
        ``None`` uses the ``DGPConfig`` defaults.
    random_state : int or numpy.random.Generator or None, optional
        Seed material; normalised by :func:`prism.utils.seeds.as_rng`. The same seed
        reproduces the mess bit-for-bit.
    dirt_rate : float, default 0.12
        Fraction of rows per categorical column receiving case/whitespace/synonym dirt.
    late_arriving_rate : float, default 0.02
        Overall fraction of rows flagged as late-arriving.
    outlier_scale : float, default 50.0
        Base multiplier for ``"scale"`` outliers before Pareto amplification.

    Returns
    -------
    pandas.DataFrame
        A **new** frame: the panel plus :data:`LATE_ARRIVING_COLUMN` and the wobbled
        column name, with ``attrs["realism_log"]`` holding one dict per applied fault
        (``step``, ``column``, ``n_affected``, ``detail``).

    Raises
    ------
    KeyError
        If ``panel`` lacks the ``(customer_id, period)`` key columns.

    Examples
    --------
    >>> dirty = inject_realism(clean, config, random_state=7)   # doctest: +SKIP
    >>> pd.DataFrame(dirty.attrs["realism_log"]).head()          # doctest: +SKIP
    """
    missing = [c for c in _KEY if c not in panel.columns]
    if missing:
        raise KeyError(f"inject_realism needs the panel key columns; missing: {missing}")

    rng = as_rng(random_state if random_state is not None else _cfg(config, "random_state", 7))
    missing_rate = float(_cfg(config, "missing_rate", 0.08))
    mnar_strength = float(_cfg(config, "mnar_strength", 0.5))
    outlier_rate = float(_cfg(config, "outlier_rate", 0.004))
    duplicate_rate = float(_cfg(config, "duplicate_rate", 0.002))

    out = panel.copy(deep=True)
    out.attrs = {}
    n = len(out)
    realism_log: list[dict[str, Any]] = []

    def note(step: str, column: str, n_affected: int, detail: str) -> None:
        realism_log.append({"step": step, "column": column, "n_affected": int(n_affected), "detail": detail})

    editable = [c for c in out.columns if c not in PROTECTED_COLUMNS]

    # -- 1 & 2. missingness -------------------------------------------------------------
    # Every mask is computed from the *pristine* values first, then applied in one pass, so
    # a driver that is itself about to go missing still drives the mechanism correctly.
    masks: dict[str, np.ndarray] = {}

    for col, drivers in MAR_DRIVERS.items():
        if col not in editable:
            continue
        present = [d for d in drivers if d in out.columns]
        if not present:
            continue
        score = np.zeros(n, dtype="float64")
        for i, drv in enumerate(present):
            score += (1.0 if i % 2 == 0 else -0.6) * _zscore(out[drv])
        masks[col] = _calibrated_mask(rng, 1.1 * score, missing_rate)
        note("mar_missing", col, int(masks[col].sum()), f"P(missing) driven by {present}")

    for col, direction in MNAR_DIRECTION.items():
        if col not in editable:
            continue
        score = direction * (1.0 + 2.0 * mnar_strength) * _zscore(out[col])
        m = _calibrated_mask(rng, score, missing_rate)
        masks[col] = masks[col] | m if col in masks else m
        side = "low" if direction < 0 else "high"
        note("mnar_missing", col, int(m.sum()), f"P(missing) rises with {side} own value, strength={mnar_strength}")

    for col, mask in masks.items():
        out[col] = out[col].mask(mask)

    # -- 3. heavy-tailed outliers -------------------------------------------------------
    for col, kind in OUTLIER_KIND.items():
        if col not in editable:
            continue
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype="float64")
        eligible = np.flatnonzero(np.isfinite(values))
        idx = _pick_rows(rng, eligible, outlier_rate)
        if idx.size == 0:
            continue
        if kind == "scale":
            factor = outlier_scale * (1.0 + rng.pareto(1.2, size=idx.size))
            values[idx] = values[idx] * factor
            detail = f"x{outlier_scale:.0f} Pareto-amplified (max x{factor.max():.0f})"
        else:
            values[idx] = -np.abs(values[idx]) - rng.exponential(10.0, size=idx.size)
            detail = "sign flipped to an impossible negative (bad date subtraction)"
        out[col] = values
        note("outlier", col, int(idx.size), detail)

    # -- 5. categorical dirt (before duplication, so copies inherit it) -----------------
    for col in CATEGORICAL_FEATURES:
        if col not in editable:
            continue
        values_obj = out[col].astype("object").to_numpy(dtype=object).copy()
        present = np.flatnonzero(np.array([isinstance(v, str) for v in values_obj]))
        idx = _pick_rows(rng, present, dirt_rate)
        if idx.size == 0:
            continue
        reverse = {v: k for k, v in CATEGORICAL_SYNONYMS.get(col, {}).items()}
        styles = rng.integers(0, 5, size=idx.size)
        n_syn = 0
        for pos, style in zip(idx.tolist(), styles.tolist()):
            raw = str(values_obj[pos])
            if style == 0:
                dirty = raw.title()
            elif style == 1:
                dirty = raw.upper()
            elif style == 2:
                dirty = f"  {raw} "
            elif style == 3:
                dirty = f"{raw}\u00a0"
            else:
                dirty = reverse.get(raw, raw.capitalize())
                n_syn += int(dirty != raw)
            values_obj[pos] = dirty
        out[col] = values_obj
        note("categorical_dirt", col, int(idx.size), f"mixed case / stray whitespace, {n_syn} legacy synonym codes")

    # -- 6. late-arriving records -------------------------------------------------------
    periods = pd.to_numeric(out["period"], errors="coerce").to_numpy(dtype="float64")
    finite_periods = periods[np.isfinite(periods)]
    if finite_periods.size:
        p_lo, p_hi = float(finite_periods.min()), float(finite_periods.max())
    else:  # empty panel, or a period column that is entirely unparseable
        p_lo = p_hi = 0.0
    recency = (periods - p_lo) / (p_hi - p_lo) if p_hi > p_lo else np.zeros(n)
    late = _calibrated_mask(rng, 2.0 * np.nan_to_num(recency), late_arriving_rate)
    out[LATE_ARRIVING_COLUMN] = late
    note("late_arriving", LATE_ARRIVING_COLUMN, int(late.sum()), "records that landed after as_of_date (leakage source)")

    # -- 4. duplicates ------------------------------------------------------------------
    n_dupes = min(n, int(round(duplicate_rate * n))) if duplicate_rate > 0 else 0
    if duplicate_rate > 0 and n_dupes < 2 and n >= 2:
        n_dupes = 2  # keep both flavours visible on small panels
    if n_dupes:
        order = np.arange(n, dtype="float64")
        src = np.sort(rng.choice(n, size=n_dupes, replace=False))
        copies = out.iloc[src].copy()
        n_near = n_dupes // 2
        jitter_cols = [c for c in _JITTER_COLUMNS if c in editable]
        n_jittered = 0
        if n_near and jitter_cols:
            for k in range(n_near):
                col = str(rng.choice(np.asarray(jitter_cols, dtype=object)))
                value = pd.to_numeric(pd.Series([copies.iloc[k][col]]), errors="coerce").iloc[0]
                if pd.isna(value):
                    continue
                copies.iloc[k, copies.columns.get_loc(col)] = float(value) * (1.0 + 0.03 * rng.normal())
                n_jittered += 1
        out = (
            pd.concat(
                [out.assign(**{_ORDER_COL: order}), copies.assign(**{_ORDER_COL: order[src] + 0.5})],
                ignore_index=True,
            )
            .sort_values(_ORDER_COL, kind="stable")
            .drop(columns=_ORDER_COL)
            .reset_index(drop=True)
        )
        note("duplicate_rows", "|".join(_KEY), n_dupes, f"{n_dupes - n_jittered} exact re-deliveries, {n_jittered} near-dupes with a jittered numeric")

    # -- 7. schema wobble ---------------------------------------------------------------
    for canonical, alias in WOBBLE_RENAMES.items():
        if canonical not in out.columns or canonical in PROTECTED_COLUMNS:
            continue
        per = pd.to_numeric(out["period"], errors="coerce")
        cutoff = float(per.max()) - (WOBBLE_N_PERIODS - 1)
        mask = (per >= cutoff).to_numpy(dtype=bool)
        if not mask.any():
            continue
        out[alias] = out[canonical].where(mask)
        out[canonical] = out[canonical].mask(mask)
        note("schema_wobble", canonical, int(mask.sum()), f"arrives as '{alias}' for period >= {cutoff:g}")

    out = out.reset_index(drop=True)
    out.attrs["realism_log"] = realism_log
    log.debug("inject_realism applied %d faults over %d rows", len(realism_log), len(out))
    return out


# --------------------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------------------
def clean_panel(
    panel: pd.DataFrame,
    *,
    robust_iqr_k: float | None = 30.0,
    min_rows_for_winsorize: int = 50,
    raise_on_contract_failure: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repair an injected-realism panel and return the panel plus an audit trail.

    The steps, in order (each one emits at least one report row):

    ``schema_repair``
        Fold every :data:`WOBBLE_RENAMES` alias back onto its canonical column, filling
        only where the canonical value is absent, then drop the alias.
    ``duplicates``
        Drop rows duplicated on ``(customer_id, period)``, keeping the first. Exact
        re-deliveries and near-duplicates (same key, conflicting values) are counted
        separately; the conflicting keys are flagged in
        ``clean.attrs["near_duplicate_keys"]`` so a data-quality dashboard can chase the
        upstream feed instead of the symptom.
    ``late_arriving``
        Drop rows flagged by :data:`LATE_ARRIVING_COLUMN`. These landed after the decision
        timestamp, so using them would leak the future into a point-in-time feature. They
        go first, before anything is measured or fitted, so no surviving row is repaired
        using a statistic that a future-dated row helped compute.
    ``categorical_normalise``
        Case-fold, strip whitespace and map synonyms onto :data:`CATEGORICAL_LEVELS` via
        :func:`_normalise_token`; unmappable codes become missing.
    ``clip_domain`` / ``winsorize``
        Clip to :data:`PLAUSIBLE_RANGES`, then optionally pull remaining heavy-tail
        survivors to a very wide robust fence (``q25/q75 -/+ k * IQR``, fitted on the
        training fold alone when a ``split`` column is present). Rows are never deleted
        for being extreme -- that would change the estimand's population.
    ``missingness``
        Report the missing rate per column and **leave it missing**. Imputation is fit on
        the training fold inside :func:`prism.data.features.build_design_matrix`; imputing
        here would leak test-fold statistics into training.
    ``schema`` / ``validate``
        :func:`prism.data.schema.enforce_schema`, then
        ``PANEL_CONTRACT.validate(clean).passed``.

    Parameters
    ----------
    panel : pandas.DataFrame
        A dirty (or already clean -- the function is idempotent) panel. Never mutated.
    robust_iqr_k : float or None, default 30.0
        Half-width of the robust winsorization fence in IQRs. Deliberately huge: it only
        catches multiplicative faults that survived the domain clip, never a genuine
        lognormal tail. ``None`` disables the step.
    min_rows_for_winsorize : int, default 50
        Skip the robust fence for columns with fewer non-null values than this.
    raise_on_contract_failure : bool, default False
        If True, re-raise the ``PANEL_CONTRACT`` failure instead of only reporting it.

    Returns
    -------
    clean_panel : pandas.DataFrame
        The repaired panel, schema-enforced, with ``attrs["cleaning_report"]``,
        ``attrs["near_duplicate_keys"]`` and ``attrs["contract_passed"]``.
    cleaning_report : pandas.DataFrame
        Audit trail with exactly the columns ``step``, ``column``, ``n_affected``,
        ``action``.

    Raises
    ------
    KeyError
        If ``panel`` lacks the ``(customer_id, period)`` key columns.
    prism.utils.validation.DataContractError
        Only when ``raise_on_contract_failure`` is True and the contract fails.

    Examples
    --------
    >>> clean, report = clean_panel(inject_realism(gold, cfg, 7))   # doctest: +SKIP
    >>> report.query("step == 'duplicates'")                        # doctest: +SKIP
    """
    key = [c for c in _KEY if c in panel.columns]
    if len(key) != len(_KEY):
        raise KeyError(f"clean_panel needs the panel key columns {_KEY}; got {list(panel.columns)[:8]}...")

    df = panel.copy(deep=True)
    df.attrs = {}
    rows: list[dict[str, Any]] = []

    def record(step: str, column: str, n_affected: int, action: str) -> None:
        rows.append({"step": step, "column": column, "n_affected": int(n_affected), "action": action})

    n_in = len(df)

    # -- schema repair ------------------------------------------------------------------
    for canonical, alias in WOBBLE_RENAMES.items():
        if alias not in df.columns:
            continue
        if canonical not in df.columns:
            df = df.rename(columns={alias: canonical})
            record("schema_repair", canonical, int(df[canonical].notna().sum()), f"renamed '{alias}' -> '{canonical}' (canonical column absent)")
            continue
        fill = df[canonical].isna() & df[alias].notna()
        df.loc[fill, canonical] = df.loc[fill, alias]
        df = df.drop(columns=[alias])
        record("schema_repair", canonical, int(fill.sum()), f"recovered from late-arriving alias '{alias}', alias dropped")

    # -- duplicates ---------------------------------------------------------------------
    dup_key = df.duplicated(subset=key, keep="first")
    dup_full = df.duplicated(keep="first")
    near_mask = dup_key & ~dup_full
    near_keys = df.loc[near_mask, key].astype(object).apply(tuple, axis=1).tolist() if near_mask.any() else []
    n_exact = int((dup_key & dup_full).sum())
    if dup_key.any():
        df = df.loc[~dup_key].reset_index(drop=True)
    record("duplicates", "|".join(key), n_exact, "dropped exact re-delivered rows on (customer_id, period), kept first")
    record("duplicates", "|".join(key), int(near_mask.sum()), "near-duplicates (same key, conflicting values): kept first, keys flagged in attrs['near_duplicate_keys']")

    # -- late-arriving rows --------------------------------------------------------------
    # Dropped *before* any statistic is computed: these rows landed after ``as_of_date``,
    # so they are not part of the point-in-time population and must not be allowed to
    # influence a fitted quantity (the robust fence below) or inflate a repair count for
    # rows that are about to disappear. Running this step last (as it originally did) also
    # broke idempotence: the fence was fitted on the pre-drop population on the first pass
    # and on the post-drop population on the second, so a second call kept moving values.
    if LATE_ARRIVING_COLUMN in df.columns:
        late = _as_bool(df[LATE_ARRIVING_COLUMN])
        df = df.loc[~late].drop(columns=[LATE_ARRIVING_COLUMN]).reset_index(drop=True)
        record("late_arriving", LATE_ARRIVING_COLUMN, int(late.sum()), "rows dropped: they landed after as_of_date and would leak the future into point-in-time features")
    else:
        record("late_arriving", LATE_ARRIVING_COLUMN, 0, "no late-arrival flag present; nothing dropped")

    # -- categorical normalisation ------------------------------------------------------
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        before = df[col].astype("object")
        # Categoricals have O(10) distinct raw spellings even in a million-row panel, so
        # normalise the unique values once and map, rather than per row.
        lut = {raw: _normalise_token(raw, col) for raw in pd.unique(before) if isinstance(raw, str)}
        after = before.map(lambda v, lut=lut: lut.get(v) if isinstance(v, str) else None)
        changed = int((before.map(_token_key) != after.map(_token_key)).sum())
        newly_missing = int((before.notna() & after.isna()).sum())
        df[col] = after
        if changed:
            record("categorical_normalise", col, changed, f"case/whitespace folded and synonyms mapped onto {CATEGORICAL_LEVELS[col]}")
        if newly_missing:
            record("categorical_normalise", col, newly_missing, "unmappable codes set to missing (no new level invented)")

    # -- numeric domain clipping --------------------------------------------------------
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        bad = int(((values < lo) | (values > hi)).sum())
        if bad:
            df[col] = values.clip(lower=lo, upper=hi)
            record("clip_domain", col, bad, f"winsorized to the domain-valid range [{lo:g}, {hi:g}] (rows kept)")
        else:
            df[col] = values

    # -- robust extreme-tail winsorization ----------------------------------------------
    # This is the one *fitted* quantity in the whole function, so it obeys the same rule
    # the missingness step does: the fence is estimated on the training fold only (when a
    # usable `split` column is present) and then applied to every fold. Fitting it on the
    # pooled panel would let test-period quantiles decide how a training row is clipped,
    # which is the leak this module refuses to commit for imputation.
    if robust_iqr_k is not None and robust_iqr_k > 0:
        fit_mask = None
        fence_scope = "all rows"
        if "split" in df.columns:
            train = df["split"].astype("object").map(lambda v: isinstance(v, str) and v.strip().lower() == "train")
            train = train.fillna(False).to_numpy(dtype=bool)
            if int(train.sum()) >= min_rows_for_winsorize:
                fit_mask = train
                fence_scope = "train fold"
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                continue
            values = df[col]
            fit_values = values if fit_mask is None else values[fit_mask]
            if int(fit_values.notna().sum()) < min_rows_for_winsorize:
                continue
            q1, q3 = float(fit_values.quantile(0.25)), float(fit_values.quantile(0.75))
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 0:
                continue
            lo_f, hi_f = q1 - robust_iqr_k * iqr, q3 + robust_iqr_k * iqr
            bad = int(((values < lo_f) | (values > hi_f)).sum())
            if bad:
                df[col] = values.clip(lower=lo_f, upper=hi_f)
                record("winsorize", col, bad, f"extreme tail pulled to q25/q75 +/- {robust_iqr_k:g} IQR ([{lo_f:.4g}, {hi_f:.4g}], fitted on {fence_scope})")

    # -- missingness: measure, never impute ---------------------------------------------
    n_now = max(len(df), 1)
    any_missing = False
    for col in df.columns:
        n_na = int(df[col].isna().sum())
        if n_na:
            any_missing = True
            record("missingness", col, n_na, f"left as missing (rate {n_na / n_now:.2%}); imputation belongs to build_design_matrix")
    if not any_missing:
        record("missingness", "*", 0, "no missing values present")

    # -- schema + contract ---------------------------------------------------------------
    df = enforce_schema(df)
    record("schema", "*", len(df.columns), "enforce_schema applied (canonical dtypes; categoricals stripped and lower-cased)")

    result = PANEL_CONTRACT.validate(df)
    detail = "; ".join(f"{f['column']}:{f['kind']}" for f in result.failures[:5]) or "none"
    record(
        "validate",
        "*",
        int(result.n_errors + result.n_warnings),
        f"PANEL_CONTRACT passed={result.passed} (errors={result.n_errors}, warnings={result.n_warnings}) [{detail}]",
    )
    record("summary", "*", n_in - len(df), f"{n_in} rows in -> {len(df)} rows out")

    report = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
    report["n_affected"] = report["n_affected"].astype("int64")

    df.attrs["cleaning_report"] = report
    df.attrs["near_duplicate_keys"] = near_keys
    df.attrs["contract_passed"] = bool(result.passed)
    df.attrs["contract_failures"] = result.failures

    if not result.passed:
        log.warning("clean_panel: PANEL_CONTRACT still failing with %d error(s)", result.n_errors)
        if raise_on_contract_failure:
            result.raise_for_status()

    return df, report


# --------------------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------------------
def _synthetic_panel(n_customers: int = 100, n_periods: int = 8, random_state: int = 11) -> pd.DataFrame:
    """Build a small, schema-valid panel inline (no dependency on ``prism.data.dgp``).

    Parameters
    ----------
    n_customers : int, default 100
        Number of customers.
    n_periods : int, default 8
        Periods per customer.
    random_state : int, default 11
        Seed material.

    Returns
    -------
    pandas.DataFrame
        A clean panel with exactly the columns of ``PANEL_COLUMNS``.
    """
    from prism.data.schema import ARM_COSTS, PANEL_COLUMNS

    rng = as_rng(random_state)
    cust = np.repeat(np.arange(n_customers), n_periods)
    per = np.tile(np.arange(n_periods), n_customers)
    n = cust.size

    origin = pd.Timestamp("2022-01-01")
    as_of = np.array([origin + pd.DateOffset(months=int(p)) for p in per], dtype="datetime64[ns]")
    acq = rng.integers(0, 24, size=n_customers)
    cohort = np.array([(origin - pd.DateOffset(months=int(a))).strftime("%Y-%m") for a in acq])[cust]

    arm = rng.integers(0, 4, size=n)
    tenure = np.clip(rng.gamma(4.0, 6.0, size=n_customers)[cust] + per, 0.5, 400.0)
    freq = np.clip(rng.poisson(6.0, size=n).astype(float), 0.0, None)
    aov = np.clip(rng.lognormal(4.0, 0.5, size=n), 1.0, None)

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{int(c):05d}" for c in cust],
            "period": per.astype("int64"),
            "as_of_date": as_of,
            "cohort": cohort,
            "split": np.where(per <= 3, "train", np.where(per <= 5, "valid", "test")),
            "assignment_block": np.where(rng.random(n_customers) < 0.15, "rct", "observational")[cust],
            "arm": arm.astype("int64"),
            "treated": (arm > 0).astype("int64"),
            "offer_cost": np.asarray(ARM_COSTS, dtype="float64")[arm],
            "tenure_months": tenure,
            "recency_days": np.clip(rng.exponential(25.0, size=n), 0.0, 900.0),
            "frequency_12m": freq,
            "monetary_12m": freq * aov,
            "avg_order_value": aov,
            "n_categories_12m": np.clip(rng.poisson(3.0, size=n).astype(float), 0.0, None),
            "sessions_30d": np.clip(rng.poisson(9.0, size=n).astype(float), 0.0, None),
            "days_since_last_session": np.clip(rng.exponential(9.0, size=n), 0.0, 400.0),
            "engagement_score": np.clip(rng.beta(2.5, 2.0, size=n) * 10.0, 0.0, 10.0),
            "support_tickets_90d": np.clip(rng.poisson(0.7, size=n).astype(float), 0.0, None),
            "nps": np.clip(rng.normal(7.0, 2.2, size=n).round(), 0.0, 10.0),
            "payment_failures_12m": np.clip(rng.poisson(0.35, size=n).astype(float), 0.0, None),
            "discount_depth_hist": np.clip(rng.beta(2.0, 8.0, size=n), 0.0, 1.0),
            "price_change_pct": rng.normal(1.5, 3.0, size=n),
            "competitor_promo_intensity": np.clip(rng.beta(2.0, 5.0, size=n), 0.0, 1.0),
            "seasonality_index": 1.0 + 0.25 * np.sin(2 * np.pi * per / 12.0),
            "basket_diversity": np.clip(rng.beta(3.0, 3.0, size=n) * 2.0, 0.0, 2.0),
            "plan_tier": rng.choice(np.asarray(CATEGORICAL_LEVELS["plan_tier"], dtype=object), size=n),
            "channel": rng.choice(np.asarray(CATEGORICAL_LEVELS["channel"], dtype=object), size=n),
            "region": rng.choice(np.asarray(CATEGORICAL_LEVELS["region"], dtype=object), size=n_customers)[cust],
            "device": rng.choice(np.asarray(CATEGORICAL_LEVELS["device"], dtype=object), size=n_customers)[cust],
            "is_autopay": rng.choice(np.asarray(CATEGORICAL_LEVELS["is_autopay"], dtype=object), size=n),
            "contract_type": rng.choice(np.asarray(CATEGORICAL_LEVELS["contract_type"], dtype=object), size=n_customers)[cust],
            "churn_next": (rng.random(n) < 0.07).astype("int64"),
            "event_time": np.round(rng.uniform(0.5, 12.0, size=n), 3),
            "event_observed": (rng.random(n) < 0.45).astype("int64"),
            "rmst_h": np.round(rng.uniform(1.0, 12.0, size=n), 3),
            "value_h": np.round(rng.gamma(3.0, 90.0, size=n), 2),
            "margin_rate": np.round(np.clip(rng.normal(0.30, 0.03, size=n), 0.05, 0.95), 4),
        }
    )
    return enforce_schema(frame[list(PANEL_COLUMNS)])


if __name__ == "__main__":  # pragma: no cover - smoke test
    import time

    from prism.config import SimConfig
    from prism.data.schema import PANEL_COLUMNS

    t0 = time.perf_counter()
    gold = _synthetic_panel(n_customers=100, n_periods=8, random_state=11)
    baseline = gold.copy(deep=True)
    assert len(gold) == 800, len(gold)
    assert PANEL_CONTRACT.validate(gold).passed, "the inline synthetic panel is not contract-clean"

    cfg = SimConfig(missing_rate=0.09, mnar_strength=0.6, outlier_rate=0.01, duplicate_rate=0.01, random_state=7)

    dirty = inject_realism(gold, cfg, random_state=7)
    dirty_again = inject_realism(gold, cfg, random_state=7)
    dirty_other = inject_realism(gold, cfg, random_state=8)

    # -- injection contract -------------------------------------------------------------
    assert gold.equals(baseline), "inject_realism mutated its input"
    assert dirty is not gold
    assert dirty.equals(dirty_again), "inject_realism is not deterministic for a fixed seed"
    assert not dirty.equals(dirty_other), "random_state is ignored: two seeds produced the same mess"
    assert len(dirty) > len(gold), "no duplicate rows were injected"
    assert LATE_ARRIVING_COLUMN in dirty.columns
    assert "net_promoter_score" in dirty.columns, "schema wobble was not applied"
    deduped = dirty.drop_duplicates(subset=list(_KEY), keep="first").reset_index(drop=True)
    assert len(deduped) == len(gold)
    for protected in PROTECTED_COLUMNS:
        assert gold[protected].equals(deduped[protected]), f"inject_realism touched protected column {protected}"
    assert dirty["plan_tier"].astype(str).str.contains(r"[A-Z]").any(), "no mixed-case categorical dirt"
    assert not PANEL_CONTRACT.validate(dirty).passed, "the dirty panel unexpectedly satisfies PANEL_CONTRACT"

    log_df = pd.DataFrame(dirty.attrs["realism_log"])
    assert not log_df.empty and set(log_df.columns) == {"step", "column", "n_affected", "detail"}
    assert set(log_df["step"]) >= {
        "mar_missing", "mnar_missing", "outlier", "duplicate_rows",
        "categorical_dirt", "late_arriving", "schema_wobble",
    }, set(log_df["step"])

    # The cleaner must have real work to do: at least one numeric is outside its domain.
    assert any(
        not pd.to_numeric(dirty[c], errors="coerce").dropna().between(lo, hi).all()
        for c, (lo, hi) in PLAUSIBLE_RANGES.items()
    ), "no injected value violates PLAUSIBLE_RANGES, so clip_domain would be untested"

    # -- the missingness mechanisms are the ones advertised ------------------------------
    # MNAR: the hidden value itself must predict its own disappearance, with the sign
    # declared in MNAR_DIRECTION. Wobble-induced nulls in `nps` are value-independent, so
    # they are excluded first -- otherwise they would dilute the signal away.
    wobbled = deduped["net_promoter_score"].notna().to_numpy()
    for col, direction in MNAR_DIRECTION.items():
        hidden = deduped[col].isna().to_numpy()
        if col in WOBBLE_RENAMES:
            hidden = hidden & ~wobbled
        assert hidden.sum() >= 20, f"too few MNAR holes in {col} to test the mechanism"
        g = pd.to_numeric(gold[col], errors="coerce").to_numpy(dtype="float64")
        mu_hidden, mu_seen = float(np.nanmean(g[hidden])), float(np.nanmean(g[~hidden]))
        if direction < 0:
            assert mu_hidden < mu_seen, f"MNAR {col} must hide LOW values ({mu_hidden:.3f} !< {mu_seen:.3f})"
        else:
            assert mu_hidden > mu_seen, f"MNAR {col} must hide HIGH values ({mu_hidden:.3f} !> {mu_seen:.3f})"

    # MAR: missingness must be driven by the declared driver, and driven by it more
    # strongly than by the column's own value -- otherwise the mechanism is really MNAR
    # and the missing-indicator trick downstream would be unjustified.
    for col, drivers in MAR_DRIVERS.items():
        hidden = deduped[col].isna().to_numpy(dtype="float64")
        driver = np.nan_to_num(pd.to_numeric(gold[drivers[0]], errors="coerce").to_numpy(dtype="float64"))
        own = np.nan_to_num(pd.to_numeric(gold[col], errors="coerce").to_numpy(dtype="float64"))
        r_drv = abs(float(np.corrcoef(hidden, driver)[0, 1]))
        r_own = abs(float(np.corrcoef(hidden, own)[0, 1])) if float(np.std(own)) > 0 else 0.0
        assert r_drv > 0.10, f"MAR {col}: missingness barely depends on driver {drivers[0]} (|r|={r_drv:.3f})"
        assert r_drv > r_own, f"MAR {col} behaves like MNAR: |r(driver)|={r_drv:.3f} <= |r(own)|={r_own:.3f}"
        realised = float(hidden.mean())
        assert abs(realised - cfg.missing_rate) < 0.04, f"MAR {col} rate {realised:.3f} far from {cfg.missing_rate}"

    # -- cleaning contract ---------------------------------------------------------------
    clean, report = clean_panel(dirty)
    clean2, _ = clean_panel(clean)  # idempotence

    assert list(report.columns) == list(REPORT_COLUMNS), report.columns.tolist()
    assert len(report) > 0, "empty cleaning report"
    assert report["n_affected"].dtype == np.int64 and bool((report["n_affected"] >= 0).all())
    assert set(report["step"]) >= {
        "schema_repair", "duplicates", "late_arriving",
        "categorical_normalise", "missingness", "schema", "validate",
    }, set(report["step"])
    assert list(clean.columns) == list(PANEL_COLUMNS), "cleaning did not restore the canonical column set"
    assert not clean.duplicated(subset=list(_KEY)).any(), "duplicate keys survived cleaning"
    assert LATE_ARRIVING_COLUMN not in clean.columns and "net_promoter_score" not in clean.columns
    assert clean["nps"].notna().any(), "schema wobble was not repaired"
    assert clean.equals(clean2), "clean_panel is not idempotent"

    # Rows out is exactly rows in minus (deduplicated + late-arriving); nothing else vanished.
    n_dropped = int(report.loc[report["step"].isin(["duplicates", "late_arriving"]), "n_affected"].sum())
    assert len(dirty) - len(clean) == n_dropped, (len(dirty), len(clean), n_dropped)

    for col in CATEGORICAL_FEATURES:
        vals = set(clean[col].dropna().unique())
        assert vals <= set(CATEGORICAL_LEVELS[col]), f"{col} has stray levels {vals - set(CATEGORICAL_LEVELS[col])}"
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        s = clean[col].dropna()
        assert s.between(lo, hi).all(), f"{col} outside PLAUSIBLE_RANGES after cleaning"
    assert clean.isna().any().any(), "cleaning imputed values it should have left missing"

    # -- the repair actually recovers the gold values ------------------------------------
    gi = gold.set_index(list(_KEY))
    ci = clean.set_index(list(_KEY))
    assert ci.index.isin(gi.index).all(), "cleaning invented keys that are not in the gold panel"
    gsub = gi.loc[ci.index]

    for protected in PROTECTED_COLUMNS:
        if protected in ci.columns:
            assert gsub[protected].reset_index(drop=True).equals(ci[protected].reset_index(drop=True)), (
                f"cleaning altered protected column {protected}"
            )

    for col in CATEGORICAL_FEATURES:
        both = gsub[col].notna().to_numpy() & ci[col].notna().to_numpy()
        assert bool((gsub[col].to_numpy()[both] == ci[col].to_numpy()[both]).all()), (
            f"categorical dirt in {col} was not folded back onto the original level"
        )

    # `nps` is wobbled and made MNAR-missing but is never an outlier target, so every
    # surviving value must be bit-identical to gold -- the sharpest test of the repair.
    both_nps = gsub["nps"].notna().to_numpy() & ci["nps"].notna().to_numpy()
    assert np.array_equal(gsub["nps"].to_numpy()[both_nps], ci["nps"].to_numpy()[both_nps]), (
        "the schema-wobble repair did not restore the original nps values"
    )
    assert int(report.loc[report["step"] == "schema_repair", "n_affected"].sum()) > 0

    # Any numeric cell that still differs from gold must be an injected outlier the clipper
    # pulled to a bound -- nothing else is allowed to move a value.
    n_outliers = int(log_df.loc[log_df["step"] == "outlier", "n_affected"].sum())
    n_moved = 0
    for col in NUMERIC_FEATURES:
        both = gsub[col].notna().to_numpy() & ci[col].notna().to_numpy()
        n_moved += int((~np.isclose(
            gsub[col].to_numpy(dtype="float64")[both], ci[col].to_numpy(dtype="float64")[both],
            rtol=1e-9, atol=1e-9,
        )).sum())
    assert 0 < n_moved <= n_outliers, f"{n_moved} numeric cells moved but only {n_outliers} outliers were injected"

    # Near-duplicates: "keep first" must retain the original row, not the re-delivered copy
    # with the recomputed numeric.
    near_keys = clean.attrs["near_duplicate_keys"]
    assert near_keys, "no near-duplicates were flagged, so the near-dupe path is untested"
    stable = [c for c in NUMERIC_FEATURES if c not in OUTLIER_KIND]
    for k in near_keys:
        g_row = gi.loc[k, stable].to_numpy(dtype="float64")
        c_row = ci.loc[k, stable].to_numpy(dtype="float64")
        both = np.isfinite(g_row) & np.isfinite(c_row)
        assert np.allclose(g_row[both], c_row[both]), (
            f"near-duplicate {k} resolved to the jittered copy instead of the original"
        )

    # An already-clean panel must pass through untouched.
    passthrough, _ = clean_panel(gold)
    assert passthrough.equals(gold), "clean_panel is not a no-op on an already-clean panel"

    validation = PANEL_CONTRACT.validate(clean)
    assert validation.passed, validation.summary().to_string(index=False)

    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"smoke test took {elapsed:.1f}s, over the 60 s budget"

    # -- output --------------------------------------------------------------------------
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 78)
    print("=== inject_realism: faults applied ===")
    print(log_df.groupby("step", as_index=False)["n_affected"].sum().to_string(index=False))
    print()
    print(f"=== clean_panel: cleaning report ({len(report)} steps) ===")
    print(report.to_string(index=False))
    print()
    print(
        f"rows  gold={len(gold)}  dirty={len(dirty)}  clean={len(clean)}"
        f" | cols {gold.shape[1]} -> {dirty.shape[1]} -> {clean.shape[1]}"
    )
    print(
        f"missing cells  dirty={int(dirty.isna().to_numpy().sum())}  clean={int(clean.isna().to_numpy().sum())}"
        f" | near-dupe keys flagged={len(near_keys)}"
    )
    print(
        f"repair fidelity  categoricals=exact  nps=exact  numeric cells moved={n_moved}"
        f" (<= {n_outliers} injected outliers)  idempotent=True  clean-panel no-op=True"
    )
    print(f"PANEL_CONTRACT passed={validation.passed} errors={validation.n_errors} warnings={validation.n_warnings}")
    print(f"messy.py OK in {elapsed:.2f}s")
