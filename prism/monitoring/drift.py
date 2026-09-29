"""Drift and decay monitoring for PRISM (SPEC.md section 6, drift half).

Two kinds of monitoring live here, and the distinction is the whole point of the module.

**Predictive monitoring** asks "did the inputs or the score move?" -- that is
:func:`psi`, :func:`ks_statistic`, :func:`js_divergence`, :func:`chi2_categorical`,
:func:`feature_drift_report` and :func:`prediction_drift`.

**Causal monitoring** asks "would the *decision* change?" -- that is :func:`cate_stability`.
A CATE model is never consumed as a number; it is consumed as a ranking and as a sign.
A uniform shift of every treatment effect by +0.5 months changes every prediction and
changes no decision at all. A 2% subgroup whose effect flips from +0.4 to -0.4 barely
moves any distributional statistic, yet it means PRISM is now paying to send retention
offers to sleeping dogs. Distribution tests cannot see the difference; sign-flip rate,
decile migration and top-decile retention can.

Conventions
-----------
All functions take plain ``numpy`` arrays / ``pandas`` objects, never PRISM model objects,
so monitoring has no dependency on how the model was fitted. NaNs are dropped pairwise
(paired functions) or per-sample (two-sample functions), and every function that discards
input logs what it discarded -- silent truncation that reads as full coverage is the one
unforgivable sin (SPEC_PERF.md section 8).

Thresholds
----------
PSI (industry convention, retained here):

===============  ==========================================
``psi < 0.10``   stable -- no action
``0.10 - 0.25``  moderate shift -- investigate ("warn")
``psi > 0.25``   significant shift -- retrain candidate ("alert")
===============  ==========================================

Both hypothesis tests are gated on an effect size before they may raise severity -- the
KS statistic against :data:`KS_MIN_EFFECT`, Cramer's V against :data:`CHI2_MIN_EFFECT`.
A p-value measures "distinguishable from zero at this sample size", which at the panel
sizes of ``SPEC_PERF.md`` section 0 is true of every feature; an alert that always fires
is not an alert.

Defaults are taken from :class:`prism.config.MonitoringConfig` so the pipeline and this
module cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from prism.config import MonitoringConfig
from prism.utils.logging import get_logger
from prism.utils.seeds import as_rng

__all__ = [
    "PSI_WARN",
    "PSI_ALERT",
    "KS_ALERT_P",
    "KS_WARN_P",
    "KS_MIN_EFFECT",
    "CHI2_ALERT_P",
    "CHI2_WARN_P",
    "CHI2_MIN_EFFECT",
    "MISSING_RATE_WARN",
    "CATE_RANK_CORR_ALERT",
    "CATE_RANK_CORR_WARN",
    "SIGN_FLIP_ALERT",
    "SIGN_FLIP_WARN",
    "TOP_DECILE_RETENTION_ALERT",
    "TOP_DECILE_RETENTION_WARN",
    "SEVERITY_ORDER",
    "psi",
    "ks_statistic",
    "js_divergence",
    "chi2_categorical",
    "feature_drift_report",
    "prediction_drift",
    "cate_stability",
    "performance_decay",
]

LOG = get_logger("monitoring.drift")

_MC = MonitoringConfig()

#: PSI below this is "stable".
PSI_WARN: float = _MC.psi_warn          # 0.10
#: PSI above this is "significant" and triggers an alert.
PSI_ALERT: float = _MC.psi_alert        # 0.25
#: Kolmogorov-Smirnov p-value that raises an alert (paired with ``KS_MIN_EFFECT``).
KS_ALERT_P: float = _MC.ks_alert_p      # 0.01
#: Kolmogorov-Smirnov p-value that raises a warning.
KS_WARN_P: float = 0.05
#: Minimum KS statistic before a p-value is allowed to trip anything. At n = 100k a
#: meaningless 0.01 shift is "significant"; significance is not drift.
KS_MIN_EFFECT: float = 0.05
#: Chi-square p-value thresholds for categorical drift.
CHI2_ALERT_P: float = 0.01
CHI2_WARN_P: float = 0.05
#: Minimum Cramer's V before a chi-square p-value is allowed to trip anything -- the
#: categorical twin of ``KS_MIN_EFFECT``. A chi-square p-value is a function of the sample
#: size as much as of the shift: the same 2-percentage-point move in one category share
#: scores p = 3e-06 at n = 20k and p = 1e-33 at n = 100k, while Cramer's V stays at 0.028
#: in both. Grading on the p-value alone therefore marks ~5% of *undrifted* categorical
#: features as warn/alert at every sample size (its nominal type-I rate), and at the panel
#: sizes of SPEC_PERF section 0 (600k-1.1M rows) marks essentially all of them. 0.05 admits
#: a 5-point share move -- which PSI alone misses at psi = 0.018, so the test still earns
#: its place -- and rejects a 2-point one.
CHI2_MIN_EFFECT: float = 0.05
#: Absolute change in a feature's missing rate that is itself a warning.
MISSING_RATE_WARN: float = 0.05

#: CATE rank-correlation floors (below = the ranking the policy consumes has moved).
CATE_RANK_CORR_ALERT: float = _MC.cate_rank_corr_alert   # 0.70
CATE_RANK_CORR_WARN: float = 0.85
#: Share of units whose estimated effect changes sign (treat -> do-not-treat).
SIGN_FLIP_ALERT: float = _MC.sign_flip_alert             # 0.15
SIGN_FLIP_WARN: float = 0.05
#: Share of the reference top decile that must still be in the current top decile.
TOP_DECILE_RETENTION_ALERT: float = 0.50
TOP_DECILE_RETENTION_WARN: float = 0.70

#: Sort order for the ``severity`` column ("alert" first).
SEVERITY_ORDER: dict[str, int] = {"alert": 0, "warn": 1, "ok": 2}

_EPS = 1e-6
_PROB_FLOOR = 1e-12

_REPORT_COLUMNS: tuple[str, ...] = (
    "feature",
    "dtype",
    "kind",
    "n_ref",
    "n_cur",
    "ref_mean",
    "cur_mean",
    "ref_missing_rate",
    "cur_missing_rate",
    "psi",
    "ks_stat",
    "ks_p",
    "js",
    "chi2_stat",
    "chi2_p",
    "chi2_v",
    "severity",
    "reason",
)


# ======================================================================================
# internal helpers
# ======================================================================================
def _as_1d(x: Any) -> np.ndarray:
    """Return ``x`` as a flat array, unwrapping pandas containers first.

    Parameters
    ----------
    x : array-like
        Input sequence, Series, Index or ndarray.

    Returns
    -------
    numpy.ndarray
        One-dimensional view/copy of ``x``.
    """
    if isinstance(x, (pd.Series, pd.Index)):
        x = x.to_numpy()
    return np.asarray(x).ravel()


def _is_numeric(arr: np.ndarray) -> bool:
    """Return True when ``arr`` has a real numeric (non-boolean) dtype."""
    return arr.dtype.kind in "fiu"


def _finite(x: Any, *, label: str = "sample") -> np.ndarray:
    """Coerce to float64 and drop NaN / +-inf, logging anything dropped.

    Parameters
    ----------
    x : array-like
        Input sample.
    label : str, default "sample"
        Name used in the debug line describing the drop.

    Returns
    -------
    numpy.ndarray
        1-D float64 array of finite values (possibly empty).
    """
    arr = np.asarray(_as_1d(x), dtype="float64")
    mask = np.isfinite(arr)
    n_bad = int(arr.size - mask.sum())
    if n_bad:
        LOG.debug("%s: dropped %d non-finite value(s) of %d", label, n_bad, arr.size)
    return arr[mask]


def _value_shares(x: Any) -> pd.Series:
    """Return the category-share (probability) vector of a categorical-like sample.

    Parameters
    ----------
    x : array-like
        Categorical sample; nulls are dropped and values compared as stripped strings.

    Returns
    -------
    pandas.Series
        Shares indexed by category, summing to 1 (empty if the sample is empty).
    """
    s = pd.Series(_as_1d(x)).dropna()
    if s.empty:
        return pd.Series(dtype="float64")
    counts = s.map(lambda v: str(v).strip()).value_counts()
    return counts / float(counts.sum())


def _align_shares(ref: Any, cur: Any) -> tuple[pd.Series, pd.Series]:
    """Align two categorical samples onto the union of their categories, zero-filled.

    Parameters
    ----------
    ref, cur : array-like
        Reference and current categorical samples.

    Returns
    -------
    tuple of pandas.Series
        ``(ref_shares, cur_shares)`` on a common, sorted category index.
    """
    p = _value_shares(ref)
    q = _value_shares(cur)
    cats = sorted(set(p.index) | set(q.index))
    return p.reindex(cats, fill_value=0.0), q.reindex(cats, fill_value=0.0)


def _psi_from_probs(p_ref: np.ndarray, p_cur: np.ndarray, eps: float = _EPS) -> float:
    """PSI between two probability vectors, with empty cells floored at ``eps``.

    Parameters
    ----------
    p_ref, p_cur : numpy.ndarray
        Non-negative probability vectors of equal length.
    eps : float, default 1e-6
        Floor applied to empty cells. Without it an empty bin makes PSI infinite,
        which is never the message you want at 3am.

    Returns
    -------
    float
        ``sum (p_cur - p_ref) * ln(p_cur / p_ref)``, always >= 0.
    """
    a = np.maximum(np.asarray(p_ref, dtype="float64"), eps)
    b = np.maximum(np.asarray(p_cur, dtype="float64"), eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = (b - a) * np.log(b / a)
    return float(np.nansum(contrib))


def _js_from_probs(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    """Jensen-Shannon *divergence* (base 2) between two probability vectors.

    Parameters
    ----------
    p_ref, p_cur : numpy.ndarray
        Non-negative weight vectors of equal length; each is normalised internally.

    Returns
    -------
    float
        Divergence in [0, 1]; ``nan`` if either vector sums to zero.
    """
    p = np.asarray(p_ref, dtype="float64")
    q = np.asarray(p_cur, dtype="float64")
    ps, qs = p.sum(), q.sum()
    if ps <= 0 or qs <= 0:
        return float("nan")
    p, q = p / ps, q / qs
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = np.where(p > 0, p * np.log2(np.maximum(p, _PROB_FLOOR) / np.maximum(m, _PROB_FLOOR)), 0.0)
        tq = np.where(q > 0, q * np.log2(np.maximum(q, _PROB_FLOOR) / np.maximum(m, _PROB_FLOOR)), 0.0)
    return float(np.clip(0.5 * tp.sum() + 0.5 * tq.sum(), 0.0, 1.0))


def _decile_index(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Assign equal-count bin indices ``0..n_bins-1`` by rank.

    Ranking (rather than value binning) keeps the bins equal-sized even when the score is
    heavily tied or skewed, which is what a "decile" means to the campaign operator.

    Parameters
    ----------
    values : numpy.ndarray
        Scores to bin.
    n_bins : int, default 10
        Number of equal-count bins.

    Returns
    -------
    numpy.ndarray
        Integer bin index per element, ``n_bins - 1`` being the top bin.
    """
    n = values.size
    if n == 0:
        return np.empty(0, dtype="int64")
    ranks = stats.rankdata(values, method="ordinal") - 1.0
    idx = np.floor(ranks * n_bins / n).astype("int64")
    return np.clip(idx, 0, n_bins - 1)


def _severity_from_flags(alert: bool, warn: bool) -> str:
    """Collapse two booleans into the ``{alert, warn, ok}`` severity ladder."""
    if alert:
        return "alert"
    return "warn" if warn else "ok"


# ======================================================================================
# 1. two-sample drift statistics
# ======================================================================================
def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

    Bins are the **quantiles of the expected (reference) distribution** -- never equal
    width, and never derived from ``actual``. Deriving edges from the current sample is a
    classic bug: it makes the metric partly blind to the very shift it is meant to catch,
    because the bins move with the data.

    Parameters
    ----------
    expected : numpy.ndarray
        Reference sample (the distribution the model was trained on). Non-numeric input is
        accepted and routed to a category-share PSI.
    actual : numpy.ndarray
        Current sample.
    n_bins : int, default 10
        Number of quantile bins. Reduced automatically (with a log line) when the
        reference sample is too small to support that many, or when the reference has
        fewer distinct values than bins.

    Returns
    -------
    float
        ``sum_b (a_b - e_b) * ln(a_b / e_b)`` over bins, with empty bins floored at 1e-6.
        ``nan`` if either sample is empty.

    Notes
    -----
    Conventional reading of the result:

    * ``psi < 0.10``   -- stable, no action;
    * ``0.10 <= psi <= 0.25`` -- moderate shift, investigate;
    * ``psi > 0.25``   -- significant shift, treat as a retraining trigger.

    The PSI *formula* is symmetric in the two probability vectors, but this function is
    only approximately symmetric in its arguments, because the bin edges come from
    ``expected``: ``psi(a, b)`` and ``psi(b, a)`` agree closely but not exactly. PSI is
    not bounded above.

    Degenerate cases are guarded explicitly. When the reference takes at most ``n_bins``
    distinct values -- a constant column, a count feature such as
    ``payment_failures_12m``, a binary flag -- quantile edges collapse onto the point
    masses and hide real movement, so PSI switches to one cell per distinct reference
    value plus an "other" cell for current values outside the reference support. A
    constant reference therefore scores 0 against an identical constant sample and scores
    large against a sample that has moved off it. Empty input returns ``nan`` rather than
    0, because "no data" is not "no drift".

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> bool(psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.05)
    True
    """
    ref_raw, cur_raw = _as_1d(expected), _as_1d(actual)
    if not (_is_numeric(ref_raw) and _is_numeric(cur_raw)):
        p, q = _align_shares(ref_raw, cur_raw)
        if p.empty or q.empty:
            LOG.warning("psi: empty categorical sample (n_ref=%d, n_cur=%d)", p.size, q.size)
            return float("nan")
        return _psi_from_probs(p.to_numpy(), q.to_numpy())

    ref = _finite(ref_raw, label="psi/expected")
    cur = _finite(cur_raw, label="psi/actual")
    if ref.size == 0 or cur.size == 0:
        LOG.warning("psi: empty sample after dropping non-finite values (n_ref=%d, n_cur=%d)", ref.size, cur.size)
        return float("nan")

    n_bins = int(max(2, n_bins))
    max_bins = max(2, min(n_bins, ref.size // 5)) if ref.size >= 10 else 2
    if max_bins < n_bins:
        LOG.debug("psi: reduced n_bins %d -> %d for a reference of %d rows", n_bins, max_bins, ref.size)
        n_bins = max_bins

    uniq_ref = np.unique(ref)
    if uniq_ref.size <= n_bins:
        # Discrete (or constant) reference: quantile edges would collapse onto the point
        # masses and hide real movement, so use one cell per distinct reference value plus
        # an "other" cell that catches current values outside the reference support.
        LOG.debug("psi: reference has %d distinct value(s) <= n_bins; using discrete cells", uniq_ref.size)
        k = int(uniq_ref.size)
        pos = np.searchsorted(uniq_ref, cur)
        pos_clipped = np.clip(pos, 0, k - 1)
        in_support = uniq_ref[pos_clipped] == cur
        cur_idx = np.where(in_support, pos_clipped, k)
        p_ref = np.bincount(np.searchsorted(uniq_ref, ref), minlength=k + 1).astype("float64") / ref.size
        p_cur = np.bincount(cur_idx, minlength=k + 1).astype("float64") / cur.size
        return _psi_from_probs(p_ref, p_cur)

    edges = np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1))
    interior = np.unique(edges[1:-1])
    ref_idx = np.searchsorted(interior, ref, side="right")
    cur_idx = np.searchsorted(interior, cur, side="right")
    n_cells = int(interior.size) + 1
    p_ref = np.bincount(ref_idx, minlength=n_cells).astype("float64") / ref.size
    p_cur = np.bincount(cur_idx, minlength=n_cells).astype("float64") / cur.size
    return _psi_from_probs(p_ref, p_cur)


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Parameters
    ----------
    expected : numpy.ndarray
        Reference sample.
    actual : numpy.ndarray
        Current sample.

    Returns
    -------
    tuple of (float, float)
        ``(statistic, p_value)``. The statistic is the maximum absolute gap between the
        two empirical CDFs, in [0, 1]; ``(nan, nan)`` if either sample is empty after
        dropping non-finite values.

    Notes
    -----
    At the panel sizes PRISM works with (10**5 rows) the p-value is significant for shifts
    far too small to matter, so :func:`feature_drift_report` requires the *statistic* to
    clear :data:`KS_MIN_EFFECT` before a p-value is allowed to raise severity.
    """
    ref = _finite(expected, label="ks/expected")
    cur = _finite(actual, label="ks/actual")
    if ref.size == 0 or cur.size == 0:
        LOG.warning("ks_statistic: empty sample (n_ref=%d, n_cur=%d)", ref.size, cur.size)
        return float("nan"), float("nan")
    res = stats.ks_2samp(ref, cur)
    return float(res.statistic), float(res.pvalue)


def js_divergence(expected: np.ndarray, actual: np.ndarray, n_bins: int = 20) -> float:
    """Jensen-Shannon divergence (base 2) on a shared histogram grid.

    This is the **divergence**, not the distance: the distance is its square root. In base
    2 the divergence is bounded in [0, 1], where 0 means identical and 1 means disjoint
    support -- which makes it readable as "bits of evidence per observation".

    Parameters
    ----------
    expected : numpy.ndarray
        Reference sample. Non-numeric input is routed to category shares.
    actual : numpy.ndarray
        Current sample.
    n_bins : int, default 20
        Number of equal-width bins spanning the **pooled** range of both samples. A shared
        grid is required: JS between histograms built on different grids is meaningless.

    Returns
    -------
    float
        Divergence in [0, 1]; ``nan`` if either sample is empty.
    """
    ref_raw, cur_raw = _as_1d(expected), _as_1d(actual)
    if not (_is_numeric(ref_raw) and _is_numeric(cur_raw)):
        p, q = _align_shares(ref_raw, cur_raw)
        if p.empty or q.empty:
            LOG.warning("js_divergence: empty categorical sample")
            return float("nan")
        return _js_from_probs(p.to_numpy(), q.to_numpy())

    ref = _finite(ref_raw, label="js/expected")
    cur = _finite(cur_raw, label="js/actual")
    if ref.size == 0 or cur.size == 0:
        LOG.warning("js_divergence: empty sample (n_ref=%d, n_cur=%d)", ref.size, cur.size)
        return float("nan")

    lo = float(min(ref.min(), cur.min()))
    hi = float(max(ref.max(), cur.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, int(max(2, n_bins)) + 1)
    h_ref, _ = np.histogram(ref, bins=edges)
    h_cur, _ = np.histogram(cur, bins=edges)
    return _js_from_probs(h_ref.astype("float64"), h_cur.astype("float64"))


def chi2_categorical(expected: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """Chi-square test comparing reference and current category distributions.

    The two category sets are aligned on their **union** and zero-filled first, so a level
    that appears only in production (a new plan tier, a new device) is tested rather than
    silently dropped -- a new level is exactly the drift you most want to see.

    Parameters
    ----------
    expected : pandas.Series
        Reference categorical sample. Nulls are dropped.
    actual : pandas.Series
        Current categorical sample. Nulls are dropped.

    Returns
    -------
    tuple of (float, float)
        ``(chi2_statistic, p_value)``. ``(0.0, 1.0)`` when fewer than two categories
        survive alignment (nothing can differ), and ``(nan, nan)`` when either sample is
        empty.

    Notes
    -----
    The chi-square approximation is unreliable when expected cell counts fall below 5;
    such cases are logged as a warning and the p-value should then be read with suspicion
    (prefer the PSI on frequencies, which does not rely on an asymptotic approximation).

    The p-value on its own is **not** a drift measure: it grows without bound in the
    sample size, so at panel scale it is significant for shifts far too small to act on.
    Pair it with an effect size -- Cramer's V, which for this 2 x k table is
    ``sqrt(chi2_stat / (n_ref + n_cur))``. :func:`feature_drift_report` does exactly that,
    reports V as ``chi2_v``, and refuses to escalate severity below
    :data:`CHI2_MIN_EFFECT`.
    Categories are compared as stripped strings so ``"Pro"`` and ``"pro "`` do not read as
    drift -- :func:`prism.data.schema.enforce_schema` normalises upstream, but monitoring
    must be robust to a raw feed.
    """
    s_ref = pd.Series(_as_1d(expected)).dropna()
    s_cur = pd.Series(_as_1d(actual)).dropna()
    if s_ref.empty or s_cur.empty:
        LOG.warning("chi2_categorical: empty sample (n_ref=%d, n_cur=%d)", s_ref.size, s_cur.size)
        return float("nan"), float("nan")

    c_ref = s_ref.map(lambda v: str(v).strip()).value_counts()
    c_cur = s_cur.map(lambda v: str(v).strip()).value_counts()
    cats = sorted(set(c_ref.index) | set(c_cur.index))
    table = np.vstack(
        [
            c_ref.reindex(cats, fill_value=0).to_numpy(dtype="float64"),
            c_cur.reindex(cats, fill_value=0).to_numpy(dtype="float64"),
        ]
    )
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 0.0, 1.0

    chi2_stat, p_value, _dof, expected_counts = stats.chi2_contingency(table, correction=False)
    n_small = int((expected_counts < 5).sum())
    if n_small:
        LOG.warning(
            "chi2_categorical: %d of %d expected cell counts are below 5 -- the chi-square "
            "approximation is unreliable here; prefer the PSI on frequencies",
            n_small,
            int(expected_counts.size),
        )
    return float(chi2_stat), float(p_value)


# ======================================================================================
# 2. feature-level report
# ======================================================================================
def _numeric_row(name: str, ref: pd.Series, cur: pd.Series, n_bins: int) -> dict[str, Any]:
    """Build one numeric feature's drift row (PSI / KS / JS).

    Parameters
    ----------
    name : str
        Feature name.
    ref, cur : pandas.Series
        Reference and current columns.
    n_bins : int
        Quantile bins for PSI.

    Returns
    -------
    dict
        Partial report row; severity and reason are added by :func:`_grade_row`.
    """
    r = _finite(ref, label=f"{name}/ref")
    c = _finite(cur, label=f"{name}/cur")
    ks_stat, ks_p = ks_statistic(r, c)
    return {
        "feature": name,
        "dtype": str(ref.dtype),
        "kind": "numeric",
        "n_ref": int(r.size),
        "n_cur": int(c.size),
        "ref_mean": float(r.mean()) if r.size else float("nan"),
        "cur_mean": float(c.mean()) if c.size else float("nan"),
        "ref_missing_rate": float(pd.isna(ref).mean()) if len(ref) else float("nan"),
        "cur_missing_rate": float(pd.isna(cur).mean()) if len(cur) else float("nan"),
        "psi": psi(r, c, n_bins=n_bins),
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "js": js_divergence(r, c),
        "chi2_stat": float("nan"),
        "chi2_p": float("nan"),
        "chi2_v": float("nan"),
    }


def _categorical_row(name: str, ref: pd.Series, cur: pd.Series) -> dict[str, Any]:
    """Build one categorical feature's drift row (chi-square + PSI/JS on frequencies).

    Parameters
    ----------
    name : str
        Feature name.
    ref, cur : pandas.Series
        Reference and current columns.

    Returns
    -------
    dict
        Partial report row; ``ks_stat``/``ks_p`` are ``nan`` because the KS test is only
        defined for an ordered variable.
    """
    p, q = _align_shares(ref, cur)
    chi2_stat, chi2_p = chi2_categorical(ref, cur)
    has_cats = not p.empty and not q.empty
    n_ref, n_cur = int(ref.notna().sum()), int(cur.notna().sum())
    # Cramer's V. The table is always 2 x k, so min(rows - 1, cols - 1) == 1 and
    # V reduces to sqrt(chi2 / N). Unlike the p-value this does not grow with N.
    n_tot = n_ref + n_cur
    chi2_v = (
        float(np.clip(np.sqrt(max(float(chi2_stat), 0.0) / n_tot), 0.0, 1.0))
        if n_tot > 0 and np.isfinite(chi2_stat)
        else float("nan")
    )
    return {
        "feature": name,
        "dtype": str(ref.dtype),
        "kind": "categorical",
        "n_ref": n_ref,
        "n_cur": n_cur,
        "ref_mean": float("nan"),
        "cur_mean": float("nan"),
        "ref_missing_rate": float(pd.isna(ref).mean()) if len(ref) else float("nan"),
        "cur_missing_rate": float(pd.isna(cur).mean()) if len(cur) else float("nan"),
        "psi": _psi_from_probs(p.to_numpy(), q.to_numpy()) if has_cats else float("nan"),
        "ks_stat": float("nan"),
        "ks_p": float("nan"),
        "js": _js_from_probs(p.to_numpy(), q.to_numpy()) if has_cats else float("nan"),
        "chi2_stat": float(chi2_stat),
        "chi2_p": float(chi2_p),
        "chi2_v": chi2_v,
    }


def _grade_row(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(severity, reason)`` for one drift row using the documented thresholds.

    Parameters
    ----------
    row : mapping
        A row carrying any of ``psi``, ``ks_stat``/``ks_p``, ``chi2_p``/``chi2_v`` and the
        two missing-rate keys. Absent or non-finite entries are skipped.

    Notes
    -----
    Both hypothesis tests are gated on an effect size, and for the same reason: a p-value
    answers "is the shift distinguishable from zero at this sample size", which at 10**5
    rows is "yes" for every feature. A KS p-value escalates only once ``ks_stat`` clears
    :data:`KS_MIN_EFFECT`, and a chi-square p-value only once Cramer's V clears
    :data:`CHI2_MIN_EFFECT`. PSI, which is an effect size already, needs no such gate.

    Returns
    -------
    tuple of (str, str)
        Severity in ``{"ok", "warn", "alert"}`` and a human-readable reason naming every
        test that tripped.
    """
    reasons: list[str] = []
    alert = warn = False

    val = float(row.get("psi", float("nan")))
    if np.isfinite(val):
        if val > PSI_ALERT:
            alert = True
            reasons.append(f"psi={val:.3f}>{PSI_ALERT:g}")
        elif val >= PSI_WARN:
            warn = True
            reasons.append(f"psi={val:.3f}>={PSI_WARN:g}")

    ks_stat = float(row.get("ks_stat", float("nan")))
    ks_p = float(row.get("ks_p", float("nan")))
    if np.isfinite(ks_stat) and np.isfinite(ks_p) and ks_stat >= KS_MIN_EFFECT:
        if ks_p < KS_ALERT_P:
            alert = True
            reasons.append(f"ks={ks_stat:.3f} (p={ks_p:.1e})")
        elif ks_p < KS_WARN_P:
            warn = True
            reasons.append(f"ks={ks_stat:.3f} (p={ks_p:.1e})")

    chi2_p = float(row.get("chi2_p", float("nan")))
    chi2_v = float(row.get("chi2_v", float("nan")))
    if np.isfinite(chi2_p) and np.isfinite(chi2_v) and chi2_v >= CHI2_MIN_EFFECT:
        if chi2_p < CHI2_ALERT_P:
            alert = True
            reasons.append(f"chi2 V={chi2_v:.3f} (p={chi2_p:.1e})")
        elif chi2_p < CHI2_WARN_P:
            warn = True
            reasons.append(f"chi2 V={chi2_v:.3f} (p={chi2_p:.1e})")

    d_miss = abs(float(row.get("cur_missing_rate", np.nan)) - float(row.get("ref_missing_rate", np.nan)))
    if np.isfinite(d_miss) and d_miss > MISSING_RATE_WARN:
        warn = True
        reasons.append(f"missing_rate shift {d_miss:+.3f}")

    severity = _severity_from_flags(alert, warn)
    return severity, "; ".join(reasons) if reasons else "no test tripped"


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-feature drift report comparing a reference window to a current window.

    Numeric features are routed to :func:`psi` / :func:`ks_statistic` /
    :func:`js_divergence`; categorical features to :func:`chi2_categorical` plus PSI and JS
    computed on category frequencies (the only sensible PSI for an unordered variable).

    Parameters
    ----------
    reference : pandas.DataFrame
        Reference window -- typically the training slice of the gold panel.
    current : pandas.DataFrame
        Current window -- typically the most recent scoring periods.
    features : sequence of str, optional
        Columns to test. Defaults to the columns present in **both** frames, in reference
        order. Columns requested but missing from either frame are skipped with a warning.
    n_bins : int, default 10
        Quantile bins for the numeric PSI.

    Returns
    -------
    pandas.DataFrame
        One row per feature with columns ``feature, dtype, kind, n_ref, n_cur, ref_mean,
        cur_mean, ref_missing_rate, cur_missing_rate, psi, ks_stat, ks_p, js, chi2_stat,
        chi2_p, chi2_v, severity, reason``. Sorted by ``severity`` (alert, warn, ok) then
        ``psi`` descending, with a fresh ``RangeIndex``.

    Notes
    -----
    Severity is deliberately effect-size aware, on **both** branches. A KS p-value only
    escalates once the KS statistic clears :data:`KS_MIN_EFFECT`, and a chi-square p-value
    only once Cramer's V (``chi2_v``) clears :data:`CHI2_MIN_EFFECT`, because at 10**5 rows
    every feature is "significantly" different from every other one. Gating only the
    numeric branch would leave the categorical one marking roughly 5% of *undrifted*
    features as warn or alert at any sample size, and nearly all of them at panel scale.
    ``chi2_v`` is reported alongside ``chi2_p`` so the gate is auditable rather than
    implicit; unlike the p-value it is stable in the sample size, which is what makes it
    usable as a threshold at all.

    PSI is the primary signal, the hypothesis tests are corroboration that catches shifts
    PSI is too coarse to see (a 5-point move in one category share is psi = 0.018 but
    V = 0.065), and ``reason`` names exactly which of them tripped, with its effect size,
    so the on-call reader does not have to re-derive it from the numbers.
    """
    if features is None:
        cols = [c for c in reference.columns if c in current.columns]
    else:
        cols, missing = [], []
        for c in features:
            (cols if (c in reference.columns and c in current.columns) else missing).append(c)
        if missing:
            LOG.warning("feature_drift_report: skipping %d column(s) absent from one frame: %s", len(missing), missing)
    if not cols:
        LOG.warning("feature_drift_report: no shared columns between reference and current")
        return pd.DataFrame(columns=list(_REPORT_COLUMNS))

    rows: list[dict[str, Any]] = []
    for col in cols:
        ref_s, cur_s = reference[col], current[col]
        numeric = (
            pd.api.types.is_numeric_dtype(ref_s)
            and pd.api.types.is_numeric_dtype(cur_s)
            and not pd.api.types.is_bool_dtype(ref_s)
            and not pd.api.types.is_bool_dtype(cur_s)
        )
        row = _numeric_row(col, ref_s, cur_s, n_bins) if numeric else _categorical_row(col, ref_s, cur_s)
        row["severity"], row["reason"] = _grade_row(row)
        rows.append(row)

    out = pd.DataFrame(rows)[list(_REPORT_COLUMNS)]
    out["_sev"] = out["severity"].map(SEVERITY_ORDER).fillna(3).astype("int64")
    out = out.sort_values(["_sev", "psi"], ascending=[True, False], na_position="last").drop(columns="_sev")
    LOG.info(
        "feature_drift_report: %d feature(s) -> %d alert, %d warn",
        len(out),
        int((out["severity"] == "alert").sum()),
        int((out["severity"] == "warn").sum()),
    )
    return out.reset_index(drop=True)


# ======================================================================================
# 3. score drift
# ======================================================================================
def prediction_drift(ref_pred: np.ndarray, cur_pred: np.ndarray) -> dict[str, Any]:
    """Drift of a model's output distribution between two scoring windows.

    Parameters
    ----------
    ref_pred : numpy.ndarray
        Reference scores (the distribution observed at validation time).
    cur_pred : numpy.ndarray
        Current scores. The two samples need not be paired or of equal length.

    Returns
    -------
    dict
        ``n_ref``, ``n_cur``, ``psi``, ``ks_stat``, ``ks_p``, ``js``, ``mean_ref``,
        ``mean_cur``, ``mean_shift``, ``sd_ref``, ``sd_cur``, ``sd_ratio``, ``p90_ref``,
        ``p90_cur``, ``top_decile_threshold_shift``, ``top_decile_threshold_shift_pct``,
        ``severity``, ``reason``.

    Notes
    -----
    The top-decile threshold is reported because that is the number an operator actually
    consumes: a score is used to *rank for targeting*, and the campaign treats everyone
    above the 90th percentile. A score distribution can drift a great deal in the middle
    without changing who gets an offer, and can drift very little overall while moving the
    cut-off -- and therefore the budget and the eligible population -- substantially.
    ``sd_ratio`` (< 1 means the score has compressed) is the companion diagnostic: a
    compressed score may still rank correctly, but its magnitudes no longer support an
    expected-value decision.
    """
    ref = _finite(ref_pred, label="prediction_drift/ref")
    cur = _finite(cur_pred, label="prediction_drift/cur")
    out: dict[str, Any] = {"n_ref": int(ref.size), "n_cur": int(cur.size)}
    if ref.size == 0 or cur.size == 0:
        LOG.warning("prediction_drift: empty sample (n_ref=%d, n_cur=%d)", ref.size, cur.size)
        for key in (
            "psi", "ks_stat", "ks_p", "js", "mean_ref", "mean_cur", "mean_shift", "sd_ref",
            "sd_cur", "sd_ratio", "p90_ref", "p90_cur", "top_decile_threshold_shift",
            "top_decile_threshold_shift_pct",
        ):
            out[key] = float("nan")
        out["severity"], out["reason"] = "alert", "empty prediction sample"
        return out

    ks_stat, ks_p = ks_statistic(ref, cur)
    sd_ref = float(ref.std(ddof=1)) if ref.size > 1 else 0.0
    sd_cur = float(cur.std(ddof=1)) if cur.size > 1 else 0.0
    p90_ref, p90_cur = float(np.quantile(ref, 0.9)), float(np.quantile(cur, 0.9))
    denom = abs(p90_ref) if abs(p90_ref) > _EPS else float("nan")
    out.update(
        {
            "psi": psi(ref, cur),
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "js": js_divergence(ref, cur),
            "mean_ref": float(ref.mean()),
            "mean_cur": float(cur.mean()),
            "mean_shift": float(cur.mean() - ref.mean()),
            "sd_ref": sd_ref,
            "sd_cur": sd_cur,
            "sd_ratio": float(sd_cur / sd_ref) if sd_ref > _EPS else float("nan"),
            "p90_ref": p90_ref,
            "p90_cur": p90_cur,
            "top_decile_threshold_shift": float(p90_cur - p90_ref),
            "top_decile_threshold_shift_pct": float((p90_cur - p90_ref) / denom),
        }
    )
    out["severity"], out["reason"] = _grade_row(out)
    return out


# ======================================================================================
# 4. CATE stability -- causal-specific monitoring
# ======================================================================================
def _cate_stability_1d(ref: np.ndarray, cur: np.ndarray, n_bins: int) -> dict[str, Any]:
    """Stability metrics for one arm's paired CATE vectors (already NaN-aligned).

    Parameters
    ----------
    ref, cur : numpy.ndarray
        Finite, equal-length, positionally paired CATE estimates.
    n_bins : int
        Number of quantile bins for the migration matrix.

    Returns
    -------
    dict
        The 1-D payload documented on :func:`cate_stability`.
    """
    n = int(ref.size)
    if n < 2:
        LOG.warning("cate_stability: fewer than 2 paired finite units (n=%d)", n)
        empty_mig = pd.DataFrame(
            np.zeros((n_bins, n_bins), dtype="int64"),
            index=pd.Index([f"ref_d{i + 1}" for i in range(n_bins)], name="reference_decile"),
            columns=pd.Index([f"cur_d{i + 1}" for i in range(n_bins)], name="current_decile"),
        )
        nan = float("nan")
        return {
            "n": n,
            "spearman_rank_corr": nan, "kendall_tau": nan, "pearson_corr": nan,
            "sign_flip_rate": nan, "n_sign_compared": 0, "n_sign_flips": 0,
            "flips_pos_to_neg": 0, "flips_neg_to_pos": 0,
            "decile_migration": empty_mig, "off_diagonal_mass": nan,
            "top_decile_retention": nan, "mean_abs_shift": nan, "mean_shift": nan,
            "ate_ref": nan, "ate_cur": nan, "ate_shift": nan,
            "severity": "alert", "reason": "insufficient paired observations",
        }

    if float(np.ptp(ref)) <= 0.0 or float(np.ptp(cur)) <= 0.0:
        spearman = kendall = pearson = float("nan")
        LOG.warning("cate_stability: a CATE vector is constant; rank correlations are undefined")
    else:
        spearman = float(stats.spearmanr(ref, cur).statistic)
        kendall = float(stats.kendalltau(ref, cur).statistic)
        pearson = float(np.corrcoef(ref, cur)[0, 1])

    # --- sign flips: an exact zero is "no recommendation", never a reversed one --------
    both_nonzero = (ref != 0.0) & (cur != 0.0)
    n_cmp = int(both_nonzero.sum())
    flip_mask = both_nonzero & (np.sign(ref) != np.sign(cur))
    n_flip = int(flip_mask.sum())
    sign_flip_rate = float(n_flip / n_cmp) if n_cmp else float("nan")

    # --- decile migration ---------------------------------------------------------------
    d_ref, d_cur = _decile_index(ref, n_bins), _decile_index(cur, n_bins)
    mig = np.zeros((n_bins, n_bins), dtype="int64")
    np.add.at(mig, (d_ref, d_cur), 1)
    migration = pd.DataFrame(
        mig,
        index=pd.Index([f"ref_d{i + 1}" for i in range(n_bins)], name="reference_decile"),
        columns=pd.Index([f"cur_d{i + 1}" for i in range(n_bins)], name="current_decile"),
    )
    off_diag = float((n - np.trace(mig)) / n)
    top = n_bins - 1
    n_top_ref = int((d_ref == top).sum())
    top_ret = float(((d_ref == top) & (d_cur == top)).sum() / n_top_ref) if n_top_ref else float("nan")

    ate_ref, ate_cur = float(ref.mean()), float(cur.mean())
    out: dict[str, Any] = {
        "n": n,
        "spearman_rank_corr": spearman,
        "kendall_tau": kendall,
        "pearson_corr": pearson,
        "sign_flip_rate": sign_flip_rate,
        "n_sign_compared": n_cmp,
        "n_sign_flips": n_flip,
        "flips_pos_to_neg": int((flip_mask & (ref > 0)).sum()),
        "flips_neg_to_pos": int((flip_mask & (ref < 0)).sum()),
        "decile_migration": migration,
        "off_diagonal_mass": off_diag,
        "top_decile_retention": top_ret,
        "mean_abs_shift": float(np.mean(np.abs(cur - ref))),
        "mean_shift": float(np.mean(cur - ref)),
        "ate_ref": ate_ref,
        "ate_cur": ate_cur,
        "ate_shift": float(ate_cur - ate_ref),
    }

    reasons: list[str] = []
    alert = warn = False
    if np.isfinite(spearman):
        if spearman < CATE_RANK_CORR_ALERT:
            alert = True
            reasons.append(f"spearman={spearman:.3f}<{CATE_RANK_CORR_ALERT:g}")
        elif spearman < CATE_RANK_CORR_WARN:
            warn = True
            reasons.append(f"spearman={spearman:.3f}<{CATE_RANK_CORR_WARN:g}")
    else:
        warn = True
        reasons.append("rank correlation undefined (constant CATE)")
    if np.isfinite(sign_flip_rate):
        if sign_flip_rate > SIGN_FLIP_ALERT:
            alert = True
            reasons.append(f"sign_flip_rate={sign_flip_rate:.3f}>{SIGN_FLIP_ALERT:g}")
        elif sign_flip_rate > SIGN_FLIP_WARN:
            warn = True
            reasons.append(f"sign_flip_rate={sign_flip_rate:.3f}>{SIGN_FLIP_WARN:g}")
    if np.isfinite(top_ret):
        if top_ret < TOP_DECILE_RETENTION_ALERT:
            alert = True
            reasons.append(f"top_decile_retention={top_ret:.3f}<{TOP_DECILE_RETENTION_ALERT:g}")
        elif top_ret < TOP_DECILE_RETENTION_WARN:
            warn = True
            reasons.append(f"top_decile_retention={top_ret:.3f}<{TOP_DECILE_RETENTION_WARN:g}")

    out["severity"] = _severity_from_flags(alert, warn)
    out["reason"] = "; ".join(reasons) if reasons else "stable"
    return out


def cate_stability(cate_ref: np.ndarray, cate_cur: np.ndarray, *, n_bins: int = 10) -> dict[str, Any]:
    """Stability of a CATE model's *decisions* between two scoring windows.

    Monitoring a causal model is not monitoring a predictive one.

    A churn classifier is judged on whether its score is still accurate. A CATE model is
    never consumed as a number: it is consumed as (a) a **sign** -- treat or do not treat
    -- and (b) a **ranking** -- who gets the budget first. Two consequences follow, and
    they are why this function reports what it reports.

    1. **Sign flips are the dangerous failure.** A unit whose estimated effect moves from
       +0.4 to -0.4 months is not "slightly less accurate": PRISM's recommendation for that
       unit has inverted. In the PRISM data-generating process the ``sleeping_dog``
       segment has a genuinely negative effect -- the offer reminds them of the price and
       pushes them out -- so a sign flip means spending budget to *cause* churn, which is
       strictly worse than doing nothing. That failure is invisible to RMSE, to AUC, and
       to PSI on the score, all of which see only a small numeric move. Exact zeros are
       excluded from the rate: a zero is "no recommendation", not a reversed one.
    2. **Decile migration is the budget failure.** Under a fixed budget only the top
       deciles are ever treated, so what matters is whether the *same units* are still up
       there. A ``top_decile_retention`` of 0.45 means more than half the treated
       population has silently changed even when the mean effect, the ATE and the score
       distribution all look untouched. Conversely, rescaling every effect by 1.5 moves
       every prediction and trips every distributional test while changing no decision at
       all -- exactly the false alarm a score-level monitor raises and this one does not.
       (A uniform *additive* shift is the instructive middle case: the ranking is
       untouched, so the budget still goes to the same units, but the treat/do-not-treat
       boundary genuinely moves for units near zero. The sign-flip rate reports that, and
       should.)

    In short: rank correlation, sign-flip rate and decile migration measure *policy
    churn*; accuracy measures something the business never buys.

    Parameters
    ----------
    cate_ref : numpy.ndarray
        Reference CATE estimates, shape ``(n,)`` or ``(n, n_arms - 1)``.
    cate_cur : numpy.ndarray
        Current CATE estimates for the **same units in the same order**, same shape as
        ``cate_ref``. This is a paired comparison, not a two-sample test.
    n_bins : int, default 10
        Number of quantile bins for the migration matrix (10 = deciles).

    Returns
    -------
    dict
        For 1-D input: ``n``, ``spearman_rank_corr``, ``kendall_tau``, ``pearson_corr``,
        ``sign_flip_rate``, ``n_sign_compared``, ``n_sign_flips``, ``flips_pos_to_neg``,
        ``flips_neg_to_pos``, ``decile_migration`` (an ``n_bins x n_bins``
        ``pandas.DataFrame`` of unit counts whose rows are reference deciles and whose row
        sums are the reference decile sizes), ``off_diagonal_mass``,
        ``top_decile_retention``, ``mean_abs_shift``, ``mean_shift``, ``ate_ref``,
        ``ate_cur``, ``ate_shift``, ``severity``, ``reason``.

        For 2-D input the same keys are returned **per arm** under ``per_arm``
        (``{arm_index: dict}``, arm 1 being the first non-control arm), plus
        ``per_arm_summary`` (a tidy ``pandas.DataFrame``), an overall summary under the
        same scalar key names (worst arm for the correlations and for retention;
        unit-weighted pooled values for the rates and shifts), ``n_arms``,
        ``recommended_arm_agreement`` and ``recommended_arm_migration``.
        ``decile_migration`` at the top level is arm 1's matrix, for a uniform 1-D/2-D
        caller.

    Notes
    -----
    ``recommended_arm_agreement`` is the multi-arm generalisation of the sign-flip rate:
    each unit's recommendation is ``argmax_a tau_a`` when any effect is positive and 0
    ("do nothing") otherwise, and the metric is the share of units whose recommendation is
    unchanged. It is the single number closest to "would the campaign actually differ?".

    Severity thresholds (documented, and sourced from
    :class:`prism.config.MonitoringConfig` where that class defines them):

    ========================  ==========  ==========
    metric                    warn        alert
    ========================  ==========  ==========
    ``spearman_rank_corr``    ``< 0.85``  ``< 0.70``
    ``sign_flip_rate``        ``> 0.05``  ``> 0.15``
    ``top_decile_retention``  ``< 0.70``  ``< 0.50``
    ========================  ==========  ==========

    In the 2-D case the overall severity is the worst arm's, and is raised to at least
    ``warn`` when fewer than 80% of units keep their recommended arm.

    Raises
    ------
    ValueError
        If the two inputs do not have the same shape, or have more than 2 dimensions.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> tau = rng.normal(size=1000)
    >>> out = cate_stability(tau, 1.5 * tau)       # rescaled, so no decision changes
    >>> out["severity"], round(out["top_decile_retention"], 3)
    ('ok', 1.0)
    """
    ref = np.asarray(cate_ref, dtype="float64")
    cur = np.asarray(cate_cur, dtype="float64")
    if ref.ndim == 2 and ref.shape[1] == 1 and cur.ndim == 1:
        ref = ref.ravel()
    if cur.ndim == 2 and cur.shape[1] == 1 and ref.ndim == 1:
        cur = cur.ravel()
    if ref.shape != cur.shape:
        raise ValueError(f"cate_ref and cate_cur must have the same shape, got {ref.shape} and {cur.shape}")
    if ref.ndim > 2:
        raise ValueError(f"cate arrays must be 1-D or 2-D, got {ref.ndim}-D")
    n_bins = int(max(2, n_bins))

    if ref.ndim == 1:
        keep = np.isfinite(ref) & np.isfinite(cur)
        n_dropped = int(keep.size - keep.sum())
        if n_dropped:
            LOG.warning("cate_stability: dropped %d of %d unit(s) with non-finite CATEs", n_dropped, keep.size)
        return _cate_stability_1d(ref[keep], cur[keep], n_bins)

    # ----------------------------------------------------------------- multi-arm (2-D) --
    n_units, n_eff_arms = ref.shape
    per_arm: dict[int, dict[str, Any]] = {}
    for j in range(n_eff_arms):
        keep = np.isfinite(ref[:, j]) & np.isfinite(cur[:, j])
        n_dropped = int(keep.size - keep.sum())
        if n_dropped:
            LOG.warning("cate_stability[arm %d]: dropped %d of %d unit(s)", j + 1, n_dropped, keep.size)
        per_arm[j + 1] = _cate_stability_1d(ref[keep, j], cur[keep, j], n_bins)

    scalar_keys = (
        "n", "spearman_rank_corr", "kendall_tau", "pearson_corr", "sign_flip_rate",
        "n_sign_compared", "n_sign_flips", "top_decile_retention", "off_diagonal_mass",
        "mean_abs_shift", "mean_shift", "ate_ref", "ate_cur", "ate_shift", "severity",
        "reason",
    )
    summary = pd.DataFrame(
        [{"arm": a, **{k: d[k] for k in scalar_keys}} for a, d in per_arm.items()]
    ).set_index("arm")

    # Recommended-arm agreement: argmax with an explicit do-nothing option (arm 0).
    row_ok = np.isfinite(ref).all(axis=1) & np.isfinite(cur).all(axis=1)
    n_rec = int(row_ok.sum())
    if n_rec:
        r_ok, c_ok = ref[row_ok], cur[row_ok]
        rec_ref = np.where(r_ok.max(axis=1) > 0, r_ok.argmax(axis=1) + 1, 0)
        rec_cur = np.where(c_ok.max(axis=1) > 0, c_ok.argmax(axis=1) + 1, 0)
        agreement = float((rec_ref == rec_cur).mean())
        rec_mig = pd.crosstab(
            pd.Series(rec_ref, name="reference_arm"),
            pd.Series(rec_cur, name="current_arm"),
        ).reindex(index=range(n_eff_arms + 1), columns=range(n_eff_arms + 1), fill_value=0)
    else:
        agreement = float("nan")
        rec_mig = pd.DataFrame(
            np.zeros((n_eff_arms + 1, n_eff_arms + 1), dtype="int64"),
            index=pd.Index(range(n_eff_arms + 1), name="reference_arm"),
            columns=pd.Index(range(n_eff_arms + 1), name="current_arm"),
        )

    def _worst(key: str) -> float:
        vals = [float(d[key]) for d in per_arm.values() if np.isfinite(d[key])]
        return float(min(vals)) if vals else float("nan")

    def _avg(key: str) -> float:
        vals = [float(d[key]) for d in per_arm.values() if np.isfinite(d[key])]
        return float(np.mean(vals)) if vals else float("nan")

    tot_cmp = int(sum(d["n_sign_compared"] for d in per_arm.values()))
    tot_flip = int(sum(d["n_sign_flips"] for d in per_arm.values()))
    overall_sev = min((d["severity"] for d in per_arm.values()), key=lambda s: SEVERITY_ORDER.get(s, 3))
    if overall_sev == "ok" and np.isfinite(agreement) and agreement < 0.80:
        overall_sev = "warn"

    return {
        "n": int(n_units),
        "n_arms": int(n_eff_arms),
        "per_arm": per_arm,
        "per_arm_summary": summary,
        "spearman_rank_corr": _worst("spearman_rank_corr"),
        "kendall_tau": _worst("kendall_tau"),
        "pearson_corr": _worst("pearson_corr"),
        "sign_flip_rate": float(tot_flip / tot_cmp) if tot_cmp else float("nan"),
        "n_sign_compared": tot_cmp,
        "n_sign_flips": tot_flip,
        "top_decile_retention": _worst("top_decile_retention"),
        "off_diagonal_mass": _avg("off_diagonal_mass"),
        "mean_abs_shift": _avg("mean_abs_shift"),
        "mean_shift": _avg("mean_shift"),
        "ate_ref": _avg("ate_ref"),
        "ate_cur": _avg("ate_cur"),
        "ate_shift": _avg("ate_shift"),
        "decile_migration": per_arm[1]["decile_migration"],
        "recommended_arm_agreement": agreement,
        "recommended_arm_migration": rec_mig,
        "severity": overall_sev,
        "reason": "; ".join(f"arm{a}: {d['reason']}" for a, d in per_arm.items()),
    }


# ======================================================================================
# 5. performance decay over time
# ======================================================================================
def _calibration_slope(score: np.ndarray, outcome: np.ndarray, binary: bool) -> float:
    """Slope of outcome on score; 1.0 means perfectly calibrated.

    For a binary outcome whose scores lie strictly inside (0, 1) the slope is the
    coefficient from a logistic regression of the outcome on ``logit(score)``; otherwise
    it is the ordinary least-squares slope of outcome on score.

    Parameters
    ----------
    score : numpy.ndarray
        Model score for one period.
    outcome : numpy.ndarray
        Realised outcome for the same rows.
    binary : bool
        Whether the outcome is binary.

    Returns
    -------
    float
        The slope, or ``nan`` when it is not identified (constant score, one outcome
        class, fewer than 5 rows, or a failed fit).
    """
    if score.size < 5:
        return float("nan")
    if binary and bool(np.all((score > 0.0) & (score < 1.0))):
        p = np.clip(score, 1e-6, 1.0 - 1e-6)
        z = np.log(p / (1.0 - p))
        if float(np.ptp(z)) <= 0.0 or np.unique(outcome).size < 2:
            return float("nan")
        try:
            import statsmodels.api as sm

            fit = sm.GLM(outcome, sm.add_constant(z), family=sm.families.Binomial()).fit(maxiter=100)
            return float(np.asarray(fit.params)[1])
        except Exception as exc:  # pragma: no cover - separation / singular design
            LOG.debug("calibration_slope: logistic fit failed (%s); falling back to OLS", exc)
    if float(np.ptp(score)) <= 0.0:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.polyfit(score, outcome, 1)[0]
    return float(slope)


def performance_decay(
    panel: pd.DataFrame,
    model_scores: np.ndarray | pd.Series | str,
    outcome_col: str,
    by: str = "period",
    *,
    decay_threshold: float = 0.10,
    min_rows: int = 30,
) -> pd.DataFrame:
    """Track model performance period by period and flag the onset of decay.

    Parameters
    ----------
    panel : pandas.DataFrame
        Scored rows. Must contain ``by`` and ``outcome_col``.
    model_scores : numpy.ndarray or pandas.Series or str
        Model output, positionally aligned to ``panel`` rows, or the name of a column of
        ``panel``.
    outcome_col : str
        Realised outcome. Treated as binary when it takes exactly the values {0, 1}.
    by : str, default "period"
        Grouping column, normally the panel period index.
    decay_threshold : float, default 0.10
        Relative drop in *skill* against the first usable period that counts as decay,
        i.e. 10 percent.
    min_rows : int, default 30
        Periods with fewer usable rows get ``nan`` metrics and are excluded from decay
        detection; the exclusion is logged rather than silent.

    Returns
    -------
    pandas.DataFrame
        One row per period, sorted ascending by ``by``, with columns ``<by>, n,
        mean_score, mean_outcome, auc, correlation, calibration_slope, skill,
        skill_rel_drop, decayed, first_decay``.

    Raises
    ------
    KeyError
        If ``by``, ``outcome_col`` or a named score column is not in ``panel``.
    ValueError
        If ``model_scores`` does not have one entry per panel row.

    Notes
    -----
    ``skill`` is the quantity decay is measured on. For a binary outcome it is
    ``auc - 0.5`` -- the *excess over random*, not the AUC itself -- because a fall from
    0.80 to 0.70 destroys a third of the model's discriminatory value while looking like a
    12 percent dip. For a continuous outcome it is ``abs(correlation)``.

    ``first_decay`` is ``True`` on at most one row: the earliest period whose skill is more
    than ``decay_threshold`` below the first usable period's skill. Later periods keep
    ``decayed=True`` but ``first_decay=False``, so an alerting layer reports an onset date
    rather than a stream of repeats.

    ``correlation`` and ``calibration_slope`` separate the two failure modes that a mean
    hides: a model that still ranks but has lost its level (slope away from 1, correlation
    intact) versus one that has stopped ranking at all (correlation collapses).

    Every row that does not reach the output is logged: rows with a non-finite score or
    outcome, and rows whose ``by`` key is null (those belong to no period at all and would
    otherwise vanish into pandas' default ``groupby(dropna=True)``).

    Decay is measured *relative to a baseline that had skill*. If the first usable period
    scores at or below chance (``skill <= 0``) there is nothing to decay from, so
    ``skill_rel_drop`` is ``nan``, nothing is flagged, and a warning says why: a model
    that never ranked is a different finding from one that stopped ranking, and reporting
    it as decay would raise an alert on the wrong problem. A merely small positive
    baseline still reports, with a warning that the ratio will read as an extreme
    percentage.
    """
    if by not in panel.columns:
        raise KeyError(f"performance_decay: grouping column {by!r} not in panel")
    if outcome_col not in panel.columns:
        raise KeyError(f"performance_decay: outcome column {outcome_col!r} not in panel")

    if isinstance(model_scores, str):
        if model_scores not in panel.columns:
            raise KeyError(f"performance_decay: score column {model_scores!r} not in panel")
        scores = panel[model_scores].to_numpy(dtype="float64")
    else:
        scores = np.asarray(_as_1d(model_scores), dtype="float64")
    if scores.size != len(panel):
        raise ValueError(f"model_scores has {scores.size} rows but panel has {len(panel)}")

    out_cols = [
        by, "n", "mean_score", "mean_outcome", "auc", "correlation", "calibration_slope",
        "skill", "skill_rel_drop", "decayed", "first_decay",
    ]
    df = pd.DataFrame(
        {
            "grp": panel[by].to_numpy(),
            "score": scores,
            "outcome": pd.to_numeric(panel[outcome_col], errors="coerce").to_numpy(dtype="float64"),
        }
    )
    n_before = len(df)
    n_null_grp = int(pd.isna(df["grp"]).sum())
    if n_null_grp:
        # pandas groupby silently drops null keys; those rows would then be missing from
        # the report with nothing to show for them (SPEC_PERF section 8).
        LOG.warning(
            "performance_decay: dropped %d row(s) of %d whose %r key is null -- they are "
            "in no period and appear in no output row",
            n_null_grp,
            n_before,
            by,
        )
        df = df[df["grp"].notna()]
    df = df[np.isfinite(df["score"].to_numpy()) & np.isfinite(df["outcome"].to_numpy())]
    if len(df) < n_before - n_null_grp:
        LOG.warning(
            "performance_decay: dropped %d row(s) with a non-finite score or outcome",
            n_before - n_null_grp - len(df),
        )
    if df.empty:
        LOG.warning("performance_decay: no usable rows")
        return pd.DataFrame(columns=out_cols)

    uniq = np.unique(df["outcome"].to_numpy())
    binary = uniq.size == 2 and set(uniq.tolist()) <= {0.0, 1.0}
    LOG.info("performance_decay: outcome %r treated as %s", outcome_col, "binary" if binary else "continuous")

    rows: list[dict[str, Any]] = []
    n_small = 0
    for key, g in df.groupby("grp", sort=True):
        s = g["score"].to_numpy()
        y = g["outcome"].to_numpy()
        rec: dict[str, Any] = {
            by: key,
            "n": int(len(g)),
            "mean_score": float(s.mean()),
            "mean_outcome": float(y.mean()),
            "auc": float("nan"),
            "correlation": float("nan"),
            "calibration_slope": float("nan"),
        }
        if len(g) < min_rows:
            n_small += 1
            rows.append(rec)
            continue
        if binary and np.unique(y).size == 2:
            from sklearn.metrics import roc_auc_score

            rec["auc"] = float(roc_auc_score(y, s))
        if float(np.ptp(s)) > 0.0 and float(np.ptp(y)) > 0.0:
            rec["correlation"] = float(np.corrcoef(s, y)[0, 1])
        rec["calibration_slope"] = _calibration_slope(s, y, binary)
        rows.append(rec)

    if n_small:
        LOG.warning(
            "performance_decay: %d period(s) had fewer than %d usable rows -> their metrics are "
            "nan and they are excluded from decay detection",
            n_small,
            min_rows,
        )

    out = pd.DataFrame(rows).sort_values(by).reset_index(drop=True)
    out["skill"] = (out["auc"] - 0.5) if binary else out["correlation"].abs()

    valid = out["skill"].notna()
    if not bool(valid.any()):
        LOG.warning("performance_decay: no period produced a usable skill value")
        out["skill_rel_drop"] = np.nan
        out["decayed"] = False
        out["first_decay"] = False
        return out[out_cols]

    base = float(out.loc[valid, "skill"].iloc[0])
    # A relative drop is only meaningful against a baseline that had skill to lose. At
    # base == 0 the ratio is undefined; just above it, it explodes (a baseline AUC of
    # 0.4885 gives skill -0.0115 and turns an ordinary later period into a -3278% "drop").
    # Either way the number must not be handed over as though it meant something.
    if base <= 1e-12:
        # No skill at the baseline means there is nothing to decay *from*. Dividing by
        # |base| here would manufacture a decay flag out of noise: a baseline AUC of
        # 0.466 (skill -0.034) followed by 0.458 (skill -0.042) reads as a 22.9% "drop"
        # and blocks a deployment gate, when the honest finding is that the model never
        # ranked better than chance. That is a different alert, raised on the raw skill.
        LOG.warning(
            "performance_decay: baseline skill is %.4f (the first usable %s ranks no "
            "better than chance) -- there is no skill to lose, so skill_rel_drop is nan "
            "and decay detection is disabled for this run; read the raw skill column, and "
            "note that a model this weak is a separate problem from a decaying one",
            base,
            by,
        )
        out["skill_rel_drop"] = np.nan
        out["decayed"] = False
        out["first_decay"] = False
        return out[out_cols]
    if base < 0.05:
        LOG.warning(
            "performance_decay: baseline skill is only %.4f -- skill_rel_drop is a ratio "
            "against near-zero and will read as an extreme percentage; use the raw skill "
            "column instead",
            base,
        )
    out["skill_rel_drop"] = (base - out["skill"]) / base
    out["decayed"] = (out["skill_rel_drop"] > decay_threshold).fillna(False).astype(bool)
    out["first_decay"] = False
    hits = np.flatnonzero(out["decayed"].to_numpy())
    if hits.size:
        i = int(hits[0])
        out.loc[out.index[i], "first_decay"] = True
        LOG.info(
            "performance_decay: first decay at %s=%s (skill %.4f vs baseline %.4f, drop %.1f%%)",
            by,
            out.loc[out.index[i], by],
            float(out.loc[out.index[i], "skill"]),
            base,
            100.0 * float(out.loc[out.index[i], "skill_rel_drop"]),
        )
    return out[out_cols]


# ======================================================================================
# smoke test
# ======================================================================================
def _smoke_frames(rng: np.random.Generator, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a reference/current pair in which exactly one numeric feature moved by 1 sd.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator.
    n : int
        Rows per frame.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(reference, current)``; ``price_change_pct`` is shifted by one standard
        deviation in ``current`` and ``plan_tier``'s mix is re-weighted; everything else
        is drawn from the same distribution in both frames.
    """

    def block(shift_sd: float, tier_p: Sequence[float]) -> pd.DataFrame:
        sd = 0.05
        return pd.DataFrame(
            {
                "tenure_months": rng.normal(24.0, 6.0, n),
                "engagement_score": rng.normal(0.0, 1.0, n),
                "monetary_12m": rng.gamma(3.0, 40.0, n),
                "price_change_pct": rng.normal(shift_sd * sd, sd, n),
                "plan_tier": rng.choice(["basic", "plus", "pro", "enterprise"], n, p=list(tier_p)),
                "region": rng.choice(["north", "south", "east", "west"], n),
            }
        )

    reference = block(0.0, (0.40, 0.30, 0.20, 0.10))
    current = block(1.0, (0.22, 0.28, 0.30, 0.20))
    return reference, current


if __name__ == "__main__":  # pragma: no cover - smoke test
    import time

    t_start = time.perf_counter()
    rng_main = as_rng(7)
    N = 8000

    # ---- 1. the statistics themselves -------------------------------------------------
    x = rng_main.normal(size=N)
    assert abs(psi(x, x)) < 1e-9, psi(x, x)
    assert abs(js_divergence(x, x)) < 1e-9
    ks_s, ks_p_val = ks_statistic(x, x)
    assert ks_s == 0.0 and ks_p_val > 0.99
    y_draw = rng_main.normal(size=N)
    # bins come from the reference, so PSI is only approximately symmetric
    assert abs(psi(x, y_draw) - psi(y_draw, x)) < 0.01, (psi(x, y_draw), psi(y_draw, x))
    assert psi(x, y_draw) < PSI_WARN, psi(x, y_draw)
    assert psi(x, x + 1.0) > PSI_ALERT, psi(x, x + 1.0)
    assert 0.0 <= js_divergence(x, x + 1.0) <= 1.0
    assert js_divergence(np.zeros(500), np.full(500, 50.0)) > 0.99, "disjoint support -> ~1 bit"
    # degenerate / empty inputs are guarded, never a crash
    assert psi(np.ones(100), np.ones(100)) == 0.0
    assert psi(np.ones(100), np.concatenate([np.ones(50), np.full(50, 9.0)])) > PSI_ALERT
    assert np.isnan(psi(np.array([]), x))
    assert all(np.isnan(v) for v in ks_statistic(np.array([]), x))

    # ---- 2. feature drift report ------------------------------------------------------
    ref_df, cur_df = _smoke_frames(rng_main, N)
    report = feature_drift_report(ref_df, cur_df)
    psi_by_feature = report.set_index("feature")["psi"].to_dict()
    assert psi_by_feature["price_change_pct"] > PSI_ALERT, psi_by_feature
    for stable_feature in ("tenure_months", "engagement_score", "monetary_12m", "region"):
        assert psi_by_feature[stable_feature] < PSI_WARN, (stable_feature, psi_by_feature[stable_feature])
    assert list(report.columns) == list(_REPORT_COLUMNS)
    assert report["severity"].map(SEVERITY_ORDER).is_monotonic_increasing, "must sort by severity"
    alerting = set(report.loc[report["severity"] == "alert", "feature"])
    assert alerting == {"price_change_pct", "plan_tier"}, alerting
    assert report.loc[report["severity"] == "alert", "psi"].is_monotonic_decreasing
    assert report["reason"].str.contains("psi=").any()
    assert report.loc[report["feature"] == "region", "severity"].item() == "ok"
    assert report.loc[report["feature"] == "plan_tier", "kind"].item() == "categorical"
    assert report.loc[report["feature"] == "plan_tier", "ks_stat"].isna().all()
    assert report.loc[report["kind"] == "numeric", "chi2_v"].isna().all()
    chi2_stat_v, chi2_p_v = chi2_categorical(ref_df["plan_tier"], cur_df["plan_tier"])
    assert chi2_stat_v > 0.0 and chi2_p_v < 1e-6, (chi2_stat_v, chi2_p_v)
    assert chi2_categorical(ref_df["region"], cur_df["region"])[1] > CHI2_ALERT_P
    # a category seen only in production must be tested, not dropped
    new_level = pd.Series(["basic"] * 90 + ["brand_new_tier"] * 10)
    assert chi2_categorical(ref_df["plan_tier"], new_level)[1] < 1e-6

    # ---- 2b. both severity branches are effect-size gated, not p-value gated ----------
    # This is a property, not a lucky seed: a p-value grows with n, an effect size does
    # not, so grading on p alone marks undrifted features as drifted at panel scale.
    SHARES = [0.40, 0.30, 0.20, 0.10]

    def _cat_frame(rng_: np.random.Generator, n_rows: int, shares: Sequence[float]) -> pd.DataFrame:
        return pd.DataFrame({"tier": rng_.choice(list("abcd"), n_rows, p=list(shares))})

    null_rows = []
    for n_rows in (20_000, 100_000):
        nul_ref = _cat_frame(rng_main, n_rows, SHARES)
        nul_cur = _cat_frame(rng_main, n_rows, SHARES)          # SAME distribution
        null_rows.append(feature_drift_report(nul_ref, nul_cur).iloc[0])
        assert null_rows[-1]["severity"] == "ok", null_rows[-1].to_dict()
        assert null_rows[-1]["chi2_v"] < CHI2_MIN_EFFECT, null_rows[-1]["chi2_v"]
    # a 2-point share move: hugely "significant" at 100k, still not worth an alert
    tiny_cur = _cat_frame(rng_main, 100_000, [0.38, 0.30, 0.22, 0.10])
    tiny = feature_drift_report(_cat_frame(rng_main, 100_000, SHARES), tiny_cur).iloc[0]
    assert tiny["chi2_p"] < 1e-10, tiny["chi2_p"]               # p says "certain drift"
    assert tiny["chi2_v"] < CHI2_MIN_EFFECT, tiny["chi2_v"]     # V says "0.03 of nothing"
    assert tiny["psi"] < PSI_WARN and tiny["severity"] == "ok", tiny.to_dict()
    small_cur = _cat_frame(rng_main, 20_000, [0.38, 0.30, 0.22, 0.10])
    small = feature_drift_report(_cat_frame(rng_main, 20_000, SHARES), small_cur).iloc[0]
    assert small["chi2_v"] < CHI2_MIN_EFFECT and small["severity"] == "ok", small.to_dict()
    # Under the p-value-only rule these two graded differently (warn at 20k, alert at
    # 100k) for the *same* shift. That is the bug: significance tracks n, drift does not.
    assert small["chi2_p"] < CHI2_WARN_P and tiny["chi2_p"] < CHI2_ALERT_P

    # The same 8-point shift measured at 20k and at 100k: V agrees to ~1%, the p-value
    # does not agree to within 30 orders of magnitude. That gap is the whole argument.
    real_ref_s, real_cur_s = SHARES, [0.32, 0.30, 0.28, 0.10]
    real = feature_drift_report(_cat_frame(rng_main, 20_000, real_ref_s), _cat_frame(rng_main, 20_000, real_cur_s)).iloc[0]
    real_big = feature_drift_report(_cat_frame(rng_main, 100_000, real_ref_s), _cat_frame(rng_main, 100_000, real_cur_s)).iloc[0]
    assert abs(real["chi2_v"] - real_big["chi2_v"]) < 0.01, (real["chi2_v"], real_big["chi2_v"])
    assert real["chi2_p"] > 1e30 * real_big["chi2_p"], (real["chi2_p"], real_big["chi2_p"])
    # and that shift IS worth acting on, so it must still escalate -- via chi2 alone,
    # because PSI is too coarse to see it. This is what the test is *for*.
    for row in (real, real_big):
        assert row["chi2_v"] >= CHI2_MIN_EFFECT, row["chi2_v"]
        assert row["psi"] < PSI_WARN, row["psi"]
        assert row["severity"] == "alert" and "chi2 V=" in row["reason"], row.to_dict()

    # ---- 3. prediction drift ----------------------------------------------------------
    ref_pred = rng_main.beta(2.0, 5.0, N)
    cur_pred = np.clip(ref_pred + 0.08, 0.0, 1.0)
    pdrift = prediction_drift(ref_pred, cur_pred)
    assert pdrift["mean_shift"] > 0.05
    assert pdrift["top_decile_threshold_shift"] > 0.0, pdrift["top_decile_threshold_shift"]
    assert 0.5 < pdrift["sd_ratio"] < 1.5
    assert pdrift["severity"] in {"warn", "alert"}
    assert prediction_drift(ref_pred, ref_pred)["severity"] == "ok"

    # ---- 4. cate_stability recovers a planted sign-flip rate --------------------------
    INJECTED = 0.12
    cate_ref = rng_main.normal(0.0, 1.0, N)
    flipped = rng_main.random(N) < INJECTED
    cate_cur = np.where(flipped, -cate_ref, cate_ref) + rng_main.normal(0.0, 0.01, N)
    stab = cate_stability(cate_ref, cate_cur)
    assert abs(stab["sign_flip_rate"] - INJECTED) < 0.02, stab["sign_flip_rate"]
    assert stab["n_sign_flips"] >= int(flipped.sum() * 0.95)
    assert stab["severity"] == "warn", (stab["severity"], stab["reason"])
    mig = stab["decile_migration"]
    assert mig.shape == (10, 10)
    assert int(mig.to_numpy().sum()) == N, "migration matrix must account for every unit"
    assert (mig.sum(axis=1).to_numpy() == np.bincount(_decile_index(cate_ref), minlength=10)).all()
    assert (mig.sum(axis=0).to_numpy() == np.bincount(_decile_index(cate_cur), minlength=10)).all()
    assert 0.0 <= stab["top_decile_retention"] <= 1.0
    assert stab["spearman_rank_corr"] > 0.5

    # a pure RESCALING moves every prediction but changes no decision: must stay "ok"
    rescaled = cate_stability(cate_ref, 1.5 * cate_ref)
    assert rescaled["spearman_rank_corr"] > 0.999999
    assert rescaled["top_decile_retention"] == 1.0
    assert rescaled["n_sign_flips"] == 0
    assert rescaled["severity"] == "ok", rescaled["reason"]
    assert psi(cate_ref, 1.5 * cate_ref) > PSI_WARN, "score-level PSI does fire -- that is the point"
    # a uniform ADDITIVE shift keeps the ranking but genuinely moves the treat boundary
    shifted = cate_stability(cate_ref, cate_ref + 0.5)
    assert shifted["spearman_rank_corr"] > 0.999999
    assert shifted["top_decile_retention"] == 1.0
    assert abs(shifted["ate_shift"] - 0.5) < 1e-9
    assert shifted["severity"] != "ok" and "sign_flip_rate" in shifted["reason"], shifted["reason"]
    assert "spearman" not in shifted["reason"] and "top_decile" not in shifted["reason"]
    # exact zeros are never sign flips
    z_ref = np.array([0.0, 1.0, -1.0, 0.0, 2.0, -2.0])
    z_cur = np.array([0.0, -1.0, -1.0, 3.0, 2.0, -2.0])
    z_out = cate_stability(z_ref, z_cur, n_bins=2)
    assert z_out["n_sign_compared"] == 4 and z_out["n_sign_flips"] == 1, z_out
    # exhaustive: all 9 (sign(ref), sign(cur)) combinations in one vector, counted by hand
    combos = [(a, b) for a in (-1.0, 0.0, 1.0) for b in (-1.0, 0.0, 1.0)]
    e_out = cate_stability(
        np.array([a for a, _ in combos]), np.array([b for _, b in combos]), n_bins=2
    )
    assert e_out["n_sign_compared"] == sum(1 for a, b in combos if a and b) == 4
    assert e_out["n_sign_flips"] == sum(1 for a, b in combos if a and b and a != b) == 2
    assert e_out["flips_pos_to_neg"] == 1 and e_out["flips_neg_to_pos"] == 1, e_out
    # deterministic under a fixed seed
    assert cate_stability(cate_ref, cate_cur)["sign_flip_rate"] == stab["sign_flip_rate"]

    # multi-arm (2-D) path
    cate_ref2 = rng_main.normal(0.0, 1.0, (N, 3))
    cate_cur2 = cate_ref2 + rng_main.normal(0.0, 0.10, (N, 3))
    cate_cur2[:, 2] = -cate_ref2[:, 2]          # arm 3 inverted -> must alert
    stab2 = cate_stability(cate_ref2, cate_cur2)
    assert stab2["n_arms"] == 3 and set(stab2["per_arm"]) == {1, 2, 3}
    assert stab2["per_arm"][3]["sign_flip_rate"] > 0.95
    assert stab2["per_arm"][1]["severity"] == "ok", stab2["per_arm"][1]["reason"]
    assert stab2["per_arm"][2]["severity"] == "ok", stab2["per_arm"][2]["reason"]
    assert stab2["per_arm"][3]["severity"] == "alert"
    assert stab2["severity"] == "alert", "the worst arm must set the overall severity"
    assert 0.0 <= stab2["recommended_arm_agreement"] < 1.0
    assert int(stab2["recommended_arm_migration"].to_numpy().sum()) == N
    assert len(stab2["per_arm_summary"]) == 3
    # brute force the multi-arm decision metric on small instances (n <= 12, 4 arms):
    # enumerate each unit's recommendation in plain Python and compare, cell by cell.
    for _ in range(300):
        n_small = int(rng_main.integers(2, 13))
        grid = [-1.0, -0.5, 0.0, 0.5, 1.0]
        b_ref = rng_main.choice(grid, size=(n_small, 3))
        b_cur = rng_main.choice(grid, size=(n_small, 3))
        b_out = cate_stability(b_ref, b_cur)
        brute_ref = [0 if max(r) <= 0 else int(np.argmax(r)) + 1 for r in b_ref]
        brute_cur = [0 if max(r) <= 0 else int(np.argmax(r)) + 1 for r in b_cur]
        want = float(np.mean([a == b for a, b in zip(brute_ref, brute_cur)]))
        assert abs(b_out["recommended_arm_agreement"] - want) < 1e-12, (b_out, want)
        hand = np.zeros((4, 4), dtype="int64")
        for a, b in zip(brute_ref, brute_cur):
            hand[a, b] += 1
        assert (b_out["recommended_arm_migration"].to_numpy() == hand).all()
    # a unit whose every effect is <= 0 must recommend "do nothing" (arm 0), not argmax
    nonpos = np.array([[0.0, 0.0, 0.0], [-1.0, -2.0, -3.0], [0.0, -1.0, 0.0]])
    assert int(cate_stability(nonpos, nonpos)["recommended_arm_migration"].loc[0, 0]) == 3
    try:
        cate_stability(cate_ref, cate_ref[:10])
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("cate_stability must reject mismatched shapes")

    # ---- 5. performance decay ---------------------------------------------------------
    N_PER, N_PERIODS = 900, 6
    frames = []
    for t in range(N_PERIODS):
        signal = max(0.0, 1.6 - 0.32 * t)        # the score progressively stops predicting
        u = rng_main.normal(size=N_PER)
        prob = 1.0 / (1.0 + np.exp(-(signal * u)))
        frames.append(
            pd.DataFrame(
                {
                    "period": t,
                    "score": 1.0 / (1.0 + np.exp(-1.6 * u)),
                    "churn_next": (rng_main.random(N_PER) < prob).astype("int64"),
                }
            )
        )
    pan = pd.concat(frames, ignore_index=True)
    decay = performance_decay(pan, pan["score"].to_numpy(), "churn_next", by="period")
    assert len(decay) == N_PERIODS and (decay["n"] == N_PER).all()
    assert decay["auc"].iloc[0] > decay["auc"].iloc[-1], decay[["period", "auc"]]
    assert int(decay["first_decay"].sum()) == 1, decay[["period", "skill_rel_drop", "decayed"]]
    first_idx = int(np.flatnonzero(decay["first_decay"].to_numpy())[0])
    assert first_idx > 0 and bool(decay["decayed"].iloc[-1])
    assert decay["calibration_slope"].notna().all()
    assert decay["skill_rel_drop"].iloc[0] == 0.0
    # the column-name form of model_scores must agree exactly
    by_name = performance_decay(pan, "score", "churn_next")
    assert np.array_equal(by_name["auc"].to_numpy(), decay["auc"].to_numpy())
    # AUC must be the real thing, not an approximation of it
    from sklearn.metrics import roc_auc_score as _auc_check

    assert np.array_equal(
        decay["auc"].to_numpy(),
        np.array([_auc_check(g["churn_next"], g["score"]) for _, g in pan.groupby("period")]),
    )
    # no row may vanish: a null period key is dropped by pandas, so it must be announced
    pan_null = pd.concat([pan, pan.head(50).assign(period=np.nan)], ignore_index=True)
    decay_null = performance_decay(pan_null, "score", "churn_next")
    assert int(decay_null["n"].sum()) == len(pan), int(decay_null["n"].sum())
    assert np.array_equal(decay_null["auc"].to_numpy(), decay["auc"].to_numpy())
    # a model that never had skill cannot "decay": no baseline, no false alarm, no crash
    noise = pd.DataFrame(
        {
            "period": np.repeat(np.arange(4), 400),
            "score": rng_main.random(1600),
            "churn_next": rng_main.integers(0, 2, 1600),
        }
    )
    noise_decay = performance_decay(noise, "score", "churn_next")
    assert len(noise_decay) == 4 and not noise_decay["first_decay"].any()
    # ... and it must not manufacture one either: with no baseline skill there is no
    # ratio to take, so the column is nan rather than a number read off noise
    assert noise_decay["skill_rel_drop"].isna().all() or float(noise_decay["skill"].iloc[0]) > 0.0
    chance = pd.DataFrame(
        {
            "period": np.repeat(np.arange(3), 400),
            "score": rng_main.random(1200),
            "churn_next": rng_main.integers(0, 2, 1200),
        }
    )
    chance.loc[chance.index[:400], "churn_next"] = (chance.loc[chance.index[:400], "score"] < 0.5).astype("int64")
    chance_decay = performance_decay(chance, "score", "churn_next")  # baseline AUC < 0.5
    if float(chance_decay["skill"].iloc[0]) <= 0.0:
        assert chance_decay["skill_rel_drop"].isna().all(), chance_decay
        assert not chance_decay["decayed"].any(), "a model that never ranked has not decayed"
    # A stationary, genuinely skilful, perfectly calibrated model must not be flagged:
    # no false positive to match the true positive above. 6000 rows/period keeps the
    # sampling error on AUC near 0.7pp, so the 10% relative gate sits ~4 sd away -- at
    # 600 rows/period this assertion fails on noise alone, which is itself worth knowing.
    N_FLAT = 6000
    u_flat = rng_main.normal(size=(4, N_FLAT))
    p_flat = 1.0 / (1.0 + np.exp(-2.0 * u_flat))
    flat = pd.concat(
        [
            pd.DataFrame(
                {
                    "period": t,
                    "score": p_flat[t],
                    "churn_next": (rng_main.random(N_FLAT) < p_flat[t]).astype("int64"),
                }
            )
            for t in range(4)
        ],
        ignore_index=True,
    )
    flat_decay = performance_decay(flat, "score", "churn_next")
    assert not flat_decay["decayed"].any(), flat_decay[["period", "skill", "skill_rel_drop"]]
    # score == P(outcome), so the calibration slope is identified and must be ~1
    assert abs(float(flat_decay["calibration_slope"].mean()) - 1.0) < 0.15, flat_decay["calibration_slope"]

    # ---- printed result ----------------------------------------------------------------
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    fmt = lambda v: f"{v:.4g}"  # noqa: E731 - display only
    print("\n=== feature_drift_report (sorted by severity, then psi desc) ===")
    print(
        report[
            ["feature", "kind", "psi", "ks_stat", "ks_p", "js", "chi2_p", "chi2_v", "severity", "reason"]
        ].to_string(index=False, float_format=fmt)
    )
    print("\n=== effect-size gate on the categorical branch (chi2_v = Cramer's V) ===")
    print(f"  {'category shift':<16} {'n':>8} {'chi2_p':>12} {'V':>8} {'psi':>9}  severity")
    for label, row in (("none (null)", null_rows[1]), ("2 points", small), ("2 points", tiny),
                       ("8 points", real), ("8 points", real_big)):
        print(f"  {label:<16} {int(row['n_cur']):>8} {row['chi2_p']:>12.1e} {row['chi2_v']:>8.4f} "
              f"{row['psi']:>9.5f}  {row['severity']}")
    print(f"  the p-value for one fixed shift moves {real['chi2_p'] / max(real_big['chi2_p'], 1e-300):.0e}x "
          f"with n; V moves {abs(real['chi2_v'] - real_big['chi2_v']):.4f}.")
    print("  a p-value-only rule grades the 2-point shift warn at 20k and alert at 100k, and")
    print("  marks ~5% of undrifted features at any n. PSI alone misses the 8-point shift entirely.")
    print("\n=== prediction_drift (churn score, +0.08 shift) ===")
    for k, v in pdrift.items():
        print(f"  {k:<32} {v if isinstance(v, str) else fmt(v)}")
    print(f"\n=== cate_stability (1-D; injected sign-flip rate = {INJECTED:.0%}) ===")
    for k, v in stab.items():
        if isinstance(v, pd.DataFrame):
            continue
        print(f"  {k:<32} {v if isinstance(v, str) else fmt(v)}")
    print("\n  decile_migration (rows = reference decile, cols = current decile):")
    print(stab["decile_migration"].to_string())
    print("\n=== cate_stability (2-D, 3 arms; arm 3 inverted) ===")
    print(
        stab2["per_arm_summary"][
            ["spearman_rank_corr", "sign_flip_rate", "top_decile_retention", "ate_shift", "severity"]
        ].to_string(float_format=fmt)
    )
    print(f"  overall severity            = {stab2['severity']}")
    print(f"  recommended_arm_agreement   = {stab2['recommended_arm_agreement']:.4f}")
    print("\n=== performance_decay ===")
    print(decay.to_string(index=False, float_format=fmt))
    print(f"\ndrift.py OK in {time.perf_counter() - t_start:.2f}s")
