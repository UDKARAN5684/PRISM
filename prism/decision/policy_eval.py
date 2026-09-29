"""Off-policy evaluation: what *would* a policy have earned?

The allocator in :mod:`prism.decision.optimize` produces an assignment ``pi(X_i)`` -- one arm
per customer. The logged data was collected under a *different* (confounded) logging policy
that chose ``A_i`` with probability ``e_{A_i}(X_i)``. Off-policy evaluation answers the only
question a budget owner actually cares about: **what would the new policy have been worth on
this data, and how sure are we?**

The three estimators
--------------------
Let ``pi(X_i)`` be the arm the evaluated policy chooses, ``A_i`` the logged arm,
``e_a(X_i) = P(A = a | X_i)`` the logging propensity, ``mu_a(X_i) = E[Y | X_i, A = a]`` an
outcome model, and ``Y_i`` the realised outcome (higher is better)::

    w_i    = 1{A_i == pi(X_i)} / e_{pi(X_i)}(X_i)                 importance weight

    IPW    V_hat = (1/n) sum_i  w_i * Y_i
    SNIPW  V_hat = ( sum_i w_i * Y_i ) / ( sum_i w_i )
    DR     V_hat = (1/n) sum_i [ mu_{pi(X_i)}(X_i)
                                 + w_i * ( Y_i - mu_{pi(X_i)}(X_i) ) ]

* **IPW** is unbiased when the propensities are correct, but its variance is driven by the
  *level* of ``Y`` and blows up when some ``e`` is small.
* **SNIPW** divides by the realised sum of weights instead of ``n``. That is the Hajek
  estimator: slightly biased in finite samples, much lower variance, and -- unlike IPW --
  guaranteed to land inside the range of the observed outcomes, so it can never return a
  value no customer could have produced.
* **DR** (AIPW) is the default. It is consistent if *either* the propensity model or the
  outcome model is right, and the outcome model absorbs the level of ``Y`` so only the
  residual is re-weighted. That is why its standard error is typically a fraction of IPW's.

**When ``mu_hat`` is None there is no outcome model, so DR is not computable.** This module
then falls back to SNIPW and *says so*: the ``method`` column reads
``"dr[fallback=snipw]"`` (:data:`DR_FALLBACK_METHOD`) rather than ``"dr"``, and a warning is
logged. A silent fallback would let a reader believe a doubly-robust number was produced
when it was not.

Honest uncertainty
------------------
Two standard errors are reported for every row and they are meant to be compared:

* ``se_analytic`` -- ``sd(psi_i) / sqrt(n)`` from the per-row influence contributions
  ``psi_i`` (for SNIPW, ``psi_i = w_i (Y_i - V) / mean(w)``, the delta-method linearisation
  of the ratio).
* ``se_boot`` -- the standard deviation of ``n_boot`` nonparametric bootstrap replicates.

If the two disagree badly, the asymptotics have not kicked in (usually a handful of enormous
weights) and neither interval should be believed. Reporting both is cheap; reporting one is
an invitation to be wrong quietly.

Per `SPEC_PERF.md` section 8 the bootstrap resamples the **per-row contribution vectors**,
never the model. 500 resamples of an n-vector is milliseconds; refitting a CATE model 500
times is minutes. Every policy and every method in one :func:`compare_policies` call shares
the *same* resample draws, which is what makes the paired comparison below valid.

Whether to believe the number at all
------------------------------------
A policy value driven by three rows carrying weight 400 is not evidence. Every row therefore
carries ``ess`` (Kish effective sample size ``(sum w)^2 / sum w^2``), ``max_weight`` (the
largest importance weight **before** clipping), ``n_clipped`` and a ``reliable`` boolean set
by this explicit rule -- see :data:`RELIABILITY_RULE`::

    reliable = (ess / n > 0.05)
               and (max_weight < 50)
               and (n_matched >= 30)
               and (n_clipped <= 0.01 * n_matched)

``max_weight`` is deliberately the *pre-clipping* maximum: clipping bounds the variance but
does not make the data informative, and a tight cap must not be able to launder a violent
weight distribution into a green flag. The last clause catches exactly that case -- heavy
clipping trades variance for bias, so it is reported rather than hidden.

References
----------
Horvitz & Thompson (1952); Hajek (1971); Robins, Rotnitzky & Zhao (1994);
Dudik, Langford & Li (2011), *Doubly Robust Policy Evaluation and Learning*;
Swaminathan & Joachims (2015) on self-normalisation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from prism.data.schema import ARM_COSTS, N_ARMS
from prism.utils.logging import get_logger
from prism.utils.seeds import as_rng

__all__ = [
    "evaluate_policy",
    "compare_policies",
    "oracle_gap",
    "METHODS",
    "DEFAULT_WEIGHT_CAP",
    "DR_FALLBACK_METHOD",
    "RELIABILITY_RULE",
    "EVALUATE_POLICY_COLUMNS",
    "COMPARE_POLICIES_COLUMNS",
]

LOG = get_logger("decision.policy_eval")

#: The estimators reported, in order. One row per method per policy.
METHODS: tuple[str, str, str] = ("ipw", "snipw", "dr")

#: Default importance-weight cap. Weights above this are truncated and counted.
DEFAULT_WEIGHT_CAP: float = 100.0

#: The ``method`` label used when ``mu_hat`` is absent and DR degrades to SNIPW.
DR_FALLBACK_METHOD: str = "dr[fallback=snipw]"

# --- the reliability rule, as named constants so the report can quote it ---------------
RELIABILITY_MIN_ESS_FRACTION: float = 0.05
RELIABILITY_MAX_WEIGHT: float = 50.0
RELIABILITY_MIN_MATCHED: int = 30
RELIABILITY_MAX_CLIP_FRACTION: float = 0.01

RELIABILITY_RULE: str = (
    f"reliable = (ess/n > {RELIABILITY_MIN_ESS_FRACTION:g}) and (max_weight < {RELIABILITY_MAX_WEIGHT:g}) and (n_matched >= {RELIABILITY_MIN_MATCHED:d}) "
    f"and (n_clipped <= {RELIABILITY_MAX_CLIP_FRACTION:g} * n_matched); max_weight is the PRE-clipping maximum, so a "
    "tight cap cannot launder a heavy-tailed weight distribution."
)

#: Column order returned by :func:`evaluate_policy`.
EVALUATE_POLICY_COLUMNS: tuple[str, ...] = (
    "method",
    "value",
    "se_analytic",
    "se_boot",
    "ci_low",
    "ci_high",
    "n",
    "n_matched",
    "ess",
    "max_weight",
    "n_clipped",
    "reliable",
)

#: Column order returned by :func:`compare_policies`.
COMPARE_POLICIES_COLUMNS: tuple[str, ...] = (
    "policy",
    "method",
    "value",
    "se_analytic",
    "se_boot",
    "ci_low",
    "ci_high",
    "diff_vs_baseline",
    "diff_ci_low",
    "diff_ci_high",
    "beats_baseline",
    "n",
    "n_treated",
    "total_cost",
    "n_matched",
    "ess",
    "max_weight",
    "n_clipped",
    "reliable",
)

_ARM_COL = "arm"
_EPS = 1e-12
#: Bootstrap index draws are generated in chunks of this many *elements* so that memory is
#: bounded and the RNG stream depends only on (n, n_boot, seed) -- never on the chunking.
_BOOT_INDEX_BUDGET: int = 2_000_000


# =======================================================================================
# internals
# =======================================================================================
@dataclass(frozen=True)
class _Contributions:
    """Per-row contribution vectors and weight diagnostics for one policy.

    Attributes
    ----------
    name : str
        Policy label, used only for messages.
    wy, w : numpy.ndarray
        ``w_i * Y_i`` and ``w_i``. IPW is ``mean(wy)``; SNIPW is ``sum(wy) / sum(w)``.
    dr : numpy.ndarray or None
        The DR per-row contribution, or ``None`` when ``mu_hat`` was not supplied.
    n, n_matched, n_clipped : int
        Rows, rows where ``A_i == pi(X_i)``, and rows whose raw weight exceeded the cap.
    ess, max_weight : float
        Kish effective sample size and the largest *pre-clipping* importance weight.
    reliable : bool
        :data:`RELIABILITY_RULE` applied to the four diagnostics above.
    """

    name: str
    wy: np.ndarray
    w: np.ndarray
    dr: np.ndarray | None
    n: int
    n_matched: int
    n_clipped: int
    ess: float
    max_weight: float
    reliable: bool


def _as_1d_int(a: Any, n: int, label: str) -> np.ndarray:
    """Coerce ``a`` to an ``(n,)`` int64 array, raising a specific error if it cannot be.

    Arm indices are *labels*, not quantities. A float array is accepted only when every
    entry is exactly integral: ``astype("int64")`` truncates towards zero, so an unrounded
    score such as ``1.9`` would silently become arm 1 and the whole evaluation would score
    a policy the caller never asked for. That failure is invisible in the output, so it is
    rejected at the door instead.
    """
    arr = np.asarray(a)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{label} has length {arr.shape[0]} but panel_test has {n} rows")
    as_float = arr.astype("float64")
    if not np.all(np.isfinite(as_float)):
        raise ValueError(f"{label} contains non-finite entries")
    if not np.all(as_float == np.rint(as_float)):
        bad = float(as_float[as_float != np.rint(as_float)][0])
        raise ValueError(
            f"{label} contains non-integral arm indices (e.g. {bad!r}); arm indices are "
            "labels, and casting to int would silently truncate them to a different policy"
        )
    out = as_float.astype("int64", copy=False)
    if np.any(out < 0):
        raise ValueError(f"{label} contains negative arm indices")
    return out


def _check_weight_cap(weight_cap: float) -> float:
    """Validate the importance-weight cap.

    A non-positive cap is not a conservative setting, it is a broken estimator: every weight
    collapses to the cap, so IPW returns ``cap * mean(Y)`` -- exactly ``0.0`` with a
    ``0.0`` standard error at ``cap = 0``, and a *sign-flipped* estimate at ``cap < 0``.
    Both come out of the formula looking like ordinary, tightly-estimated numbers, which is
    the one thing this module refuses to emit.
    """
    cap = float(weight_cap)
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError(
            f"weight_cap must be finite and strictly positive, got {weight_cap!r}; a "
            "non-positive cap zeroes or sign-flips every importance weight and would "
            "report the result as a confidently estimated policy value"
        )
    return cap


def _check_alpha(alpha: float) -> float:
    """Validate the two-sided level.

    Outside ``(0, 1)`` the percentile call either raises from deep inside numpy or -- worse,
    for ``alpha > 1`` -- returns ``ci_low > ci_high``: an inverted interval that every
    downstream ``ci_low > 0`` test reads as a perfectly ordinary result.
    """
    a = float(alpha)
    if not np.isfinite(a) or not (0.0 < a < 1.0):
        raise ValueError(f"alpha must lie strictly inside (0, 1), got {alpha!r}")
    return a


def _std(x: np.ndarray) -> float:
    """Sample standard deviation that returns NaN (not a warning) for n < 2."""
    x = np.asarray(x, dtype="float64")
    finite = x[np.isfinite(x)]
    if finite.size < 2:
        return float("nan")
    return float(np.std(finite, ddof=1))


def _policy_contributions(
    y: np.ndarray,
    arm: np.ndarray,
    policy_arm: np.ndarray,
    propensity: np.ndarray,
    mu_hat: np.ndarray | None,
    weight_cap: float,
    name: str,
) -> _Contributions:
    """Build the per-row contribution vectors for one policy.

    Parameters
    ----------
    y : numpy.ndarray
        ``(n,)`` logged outcomes, higher is better.
    arm : numpy.ndarray
        ``(n,)`` logged arm ``A_i``.
    policy_arm : numpy.ndarray
        ``(n,)`` arm chosen by the evaluated policy, ``pi(X_i)``.
    propensity : numpy.ndarray
        ``(n, n_arms)`` logging propensities, or ``(n,)`` propensities *of the logged arm*.
        The 1-D form is sufficient because ``e_{pi(X_i)}`` is only ever multiplied by
        ``1{A_i == pi(X_i)}``, and on those rows ``pi(X_i) == A_i``.
    mu_hat : numpy.ndarray or None
        ``(n, n_arms)`` outcome-model predictions, or ``None``.
    weight_cap : float
        Importance weights are truncated at this value.
    name : str
        Policy label used in log messages.

    Returns
    -------
    _Contributions
    """
    n = y.shape[0]
    rows = np.arange(n)
    match = arm == policy_arm
    n_matched = int(match.sum())

    if propensity.ndim == 2:
        e_pi = propensity[rows, policy_arm].astype("float64", copy=True)
    else:
        e_pi = propensity.astype("float64", copy=True)

    valid_e = np.isfinite(e_pi) & (e_pi > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_e = np.where(valid_e, 1.0 / np.where(valid_e, e_pi, 1.0), np.inf)
    raw_w = np.where(match, inv_e, 0.0)

    n_bad_e = int(np.sum(match & ~valid_e))
    if n_bad_e:
        LOG.warning(
            "policy %r: %d matched rows have a non-positive or non-finite propensity; "
            "their weights are truncated to the cap (%.1f)",
            name,
            n_bad_e,
            weight_cap,
        )

    max_weight = float(raw_w.max()) if n else 0.0
    n_clipped = int(np.sum(raw_w > weight_cap))
    w = np.minimum(raw_w, float(weight_cap))
    if n_clipped:
        LOG.warning(
            "policy %r: clipped %d/%d matched rows at weight %.1f (raw max %.1f) -- "
            "this trades variance for bias and is reported in n_clipped",
            name,
            n_clipped,
            max(n_matched, 1),
            weight_cap,
            max_weight,
        )

    sw = float(w.sum())
    sw2 = float(np.square(w).sum())
    ess = (sw * sw / sw2) if sw2 > 0 else 0.0

    wy = w * y

    dr: np.ndarray | None = None
    if mu_hat is not None:
        mu_pi = mu_hat[rows, policy_arm].astype("float64", copy=False)
        dr = mu_pi + w * (y - mu_pi)

    reliable = bool(
        n > 0
        and (ess / n) > RELIABILITY_MIN_ESS_FRACTION
        and max_weight < RELIABILITY_MAX_WEIGHT
        and n_matched >= RELIABILITY_MIN_MATCHED
        and n_clipped <= RELIABILITY_MAX_CLIP_FRACTION * n_matched
    )

    return _Contributions(
        name=name,
        wy=wy,
        w=w,
        dr=dr,
        n=n,
        n_matched=n_matched,
        n_clipped=n_clipped,
        ess=float(ess),
        max_weight=max_weight,
        reliable=reliable,
    )


def _assemble_columns(contribs: Sequence[_Contributions]) -> tuple[np.ndarray, list[dict[str, int | None]]]:
    """Stack every policy's contribution vectors into one ``(n, m)`` matrix.

    One matrix means one bootstrap pass shared by every policy and every method, which is
    both fast and -- crucially -- what makes the paired policy differences valid.

    Returns
    -------
    cols : numpy.ndarray
        ``(n, m)`` float64 matrix of per-row contributions.
    specs : list of dict
        For each policy, the column index of ``"wy"``, ``"w"`` and ``"dr"`` (``None`` when
        no outcome model was supplied).
    """
    columns: list[np.ndarray] = []
    specs: list[dict[str, int | None]] = []
    for c in contribs:
        spec: dict[str, int | None] = {"wy": len(columns)}
        columns.append(c.wy)
        spec["w"] = len(columns)
        columns.append(c.w)
        if c.dr is not None:
            spec["dr"] = len(columns)
            columns.append(c.dr)
        else:
            spec["dr"] = None
        specs.append(spec)
    cols = np.column_stack(columns).astype("float64", copy=False)
    return cols, specs


def _bootstrap_means(cols: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Nonparametric bootstrap of the column means of ``cols``.

    Resamples row indices with replacement and returns the resampled column means. Nothing
    is refitted -- this is the cheap, correct bootstrap mandated by ``SPEC_PERF.md`` s8.
    Implemented as counts-times-matrix so the inner loop is BLAS, not Python.

    Parameters
    ----------
    cols : numpy.ndarray
        ``(n, m)`` per-row contributions.
    n_boot : int
        Number of resamples.
    rng : numpy.random.Generator
        Seeded generator. The draw sequence depends only on ``(n, n_boot, seed)``, never on
        the internal chunk size, so results are reproducible across call sites.

    Returns
    -------
    numpy.ndarray
        ``(n_boot, m)`` resampled column means.
    """
    n, m = cols.shape
    out = np.empty((max(int(n_boot), 0), m), dtype="float64")
    if n == 0 or n_boot <= 0:
        return out
    chunk = int(np.clip(_BOOT_INDEX_BUDGET // max(n, 1), 1, n_boot))
    inv_n = 1.0 / n
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        counts = np.empty((b, n), dtype="float64")
        for r in range(b):
            counts[r] = np.bincount(idx[r], minlength=n)
        out[start : start + b] = (counts @ cols) * inv_n
    return out


def _method_values(means: np.ndarray, spec: Mapping[str, int | None]) -> dict[str, np.ndarray]:
    """Map column means to the three estimator values.

    Works both on the point estimate (``means`` of shape ``(m,)``) and on the bootstrap
    replicates (shape ``(n_boot, m)``), because every reduction is applied on the last axis.

    Parameters
    ----------
    means : numpy.ndarray
        Column means, last axis indexed by the column ids in ``spec``.
    spec : Mapping
        Column-index map produced by :func:`_assemble_columns`.

    Returns
    -------
    dict
        ``{"ipw": ..., "snipw": ..., "dr": ...}``. When no outcome model was supplied the
        ``"dr"`` entry *is* the SNIPW value; the caller relabels the method accordingly.
    """
    wy = means[..., int(spec["wy"])]
    w = means[..., int(spec["w"])]
    with np.errstate(divide="ignore", invalid="ignore"):
        snipw = np.where(w > _EPS, wy / np.where(w > _EPS, w, 1.0), np.nan)
    out = {"ipw": wy, "snipw": snipw}
    out["dr"] = means[..., int(spec["dr"])] if spec["dr"] is not None else snipw
    return out


def _analytic_ses(contrib: _Contributions, snipw_value: float) -> dict[str, float]:
    """Closed-form standard errors from the per-row influence contributions.

    ``se = sd(psi_i) / sqrt(n)`` with ``psi_i = w_i Y_i`` for IPW, ``psi_i = mu + w(Y-mu)``
    for DR, and the delta-method linearisation ``psi_i = (w_i Y_i - V w_i) / mean(w)`` for
    the SNIPW ratio.

    Parameters
    ----------
    contrib : _Contributions
        Per-row contributions for one policy.
    snipw_value : float
        The SNIPW point estimate ``V``, needed to centre the ratio influence function.

    Returns
    -------
    dict
        Standard errors keyed by method name.
    """
    n = contrib.n
    root_n = np.sqrt(n) if n > 0 else np.nan
    se_ipw = _std(contrib.wy) / root_n

    mean_w = float(contrib.w.mean()) if n else 0.0
    if mean_w > _EPS and np.isfinite(snipw_value):
        psi_sn = (contrib.wy - snipw_value * contrib.w) / mean_w
        se_snipw = _std(psi_sn) / root_n
    else:
        se_snipw = float("nan")

    se_dr = (_std(contrib.dr) / root_n) if contrib.dr is not None else se_snipw
    return {"ipw": float(se_ipw), "snipw": float(se_snipw), "dr": float(se_dr)}


def _interval(
    point: float,
    boot: np.ndarray | None,
    se_analytic: float,
    alpha: float,
) -> tuple[float, float, float]:
    """Return ``(se_boot, ci_low, ci_high)``.

    Uses the percentile bootstrap when replicates are available (it needs no symmetry
    assumption, which matters because IPW is right-skewed), otherwise a normal interval
    built from the analytic standard error.
    """
    lo_q, hi_q = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    if boot is not None and boot.size >= 2 and np.isfinite(boot).sum() >= 2:
        finite = boot[np.isfinite(boot)]
        se_boot = float(np.std(finite, ddof=1))
        return se_boot, float(np.percentile(finite, lo_q)), float(np.percentile(finite, hi_q))
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha / 2.0))
    if not np.isfinite(se_analytic) or not np.isfinite(point):
        return float("nan"), float("nan"), float("nan")
    return float("nan"), point - z * se_analytic, point + z * se_analytic


def _method_label(method: str, has_mu: bool) -> str:
    """Return the ``method`` column value, flagging a DR fallback explicitly."""
    if method == "dr" and not has_mu:
        return DR_FALLBACK_METHOD
    return method


def _is_unidentified(contrib: _Contributions, method: str) -> bool:
    """True when this method's estimand is not identified on this data.

    With ``n_matched == 0`` the logging policy never played the evaluated action on any row,
    so no re-weighting estimator has anything to re-weight. The IPW formula would still
    *evaluate* to exactly ``0`` with a zero standard error, which is a confident statement
    about a quantity the data cannot speak to -- so those cells are returned as NaN instead.
    DR survives only when an outcome model is present, and is then pure model extrapolation.
    """
    if contrib.n_matched > 0:
        return False
    return method != "dr" or contrib.dr is None


def _validate_inputs(
    panel_test: pd.DataFrame,
    outcome_col: str,
    propensity: Any,
    mu_hat: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int, bool]:
    """Validate and unpack the shared inputs of the two evaluation entry points.

    Returns
    -------
    y, arm, propensity, mu_hat, n_arms, n_arms_is_bound
        ``n_arms_is_bound`` is True when the arm count is pinned by the shape of
        ``propensity`` or ``mu_hat`` -- only then is an out-of-range policy arm an error
        rather than simply an arm the logging policy never played.
    """
    if not isinstance(panel_test, pd.DataFrame):
        raise TypeError(f"panel_test must be a DataFrame, got {type(panel_test).__name__}")
    n = len(panel_test)
    if n == 0:
        raise ValueError("panel_test is empty; there is nothing to evaluate")
    missing = [c for c in (_ARM_COL, outcome_col) if c not in panel_test.columns]
    if missing:
        raise KeyError(
            f"panel_test is missing required column(s) {missing}; "
            f"off-policy evaluation needs the logged arm column {_ARM_COL!r} and {outcome_col!r}"
        )

    y = pd.to_numeric(panel_test[outcome_col], errors="coerce").to_numpy(dtype="float64")
    if not np.all(np.isfinite(y)):
        raise ValueError(
            f"outcome column {outcome_col!r} has {int((~np.isfinite(y)).sum())} non-finite "
            "values; impute or drop them before evaluating a policy"
        )
    arm = _as_1d_int(panel_test[_ARM_COL].to_numpy(), n, "panel_test['arm']")

    prop = np.asarray(propensity, dtype="float64")
    n_arms_is_bound = False
    if prop.ndim == 1:
        if prop.shape[0] != n:
            raise ValueError(f"propensity has length {prop.shape[0]} but panel_test has {n} rows")
        n_arms = int(max(arm.max(initial=0), N_ARMS - 1)) + 1
    elif prop.ndim == 2:
        if prop.shape[0] != n:
            raise ValueError(f"propensity has {prop.shape[0]} rows but panel_test has {n}")
        n_arms = int(prop.shape[1])
        n_arms_is_bound = True
        row_sums = prop.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=0.05):
            LOG.warning(
                "propensity rows do not sum to 1 (min %.3f, max %.3f); treating them as "
                "generalised propensity scores anyway",
                float(row_sums.min()),
                float(row_sums.max()),
            )
    else:
        raise ValueError(f"propensity must be 1-D or 2-D, got shape {prop.shape}")

    mu: np.ndarray | None = None
    if mu_hat is not None:
        mu = np.asarray(mu_hat, dtype="float64")
        if mu.ndim != 2 or mu.shape[0] != n:
            raise ValueError(f"mu_hat must have shape (n, n_arms) = ({n}, {n_arms}), got {mu.shape}")
        if mu.shape[1] < n_arms:
            raise ValueError(f"mu_hat has {mu.shape[1]} arm columns but {n_arms} arms are in play")
        if not np.all(np.isfinite(mu)):
            raise ValueError("mu_hat contains non-finite predictions")
        if not n_arms_is_bound:
            n_arms = int(mu.shape[1])
            n_arms_is_bound = True
    else:
        LOG.warning(
            "mu_hat is None: the doubly-robust estimator is not computable, so the 'dr' row "
            "falls back to SNIPW and is labelled %r",
            DR_FALLBACK_METHOD,
        )
    return y, arm, prop, mu, n_arms, n_arms_is_bound


def _cost_matrix(costs: Any, n: int, n_arms: int) -> np.ndarray | None:
    """Resolve ``costs`` into an ``(n, n_arms)`` cost matrix, or ``None``.

    Accepts, in order of preference:

    * ``None`` -- no cost accounting, ``total_cost`` comes back NaN;
    * an ``(n, n_arms)`` matrix of per-customer per-arm costs;
    * a length-``n_arms`` sequence of per-arm costs;
    * any object exposing ``arm_costs`` (e.g. ``EconomicConfig``). If it also exposes
      ``offer_redemption_rate`` the *expected* cost ``cost_a * redemption_a`` is used, since
      an unredeemed offer costs nothing.
    """
    if costs is None:
        return None
    if hasattr(costs, "arm_costs"):
        vec = np.asarray(costs.arm_costs, dtype="float64")
        red = getattr(costs, "offer_redemption_rate", None)
        if red is not None:
            red_v = np.asarray(red, dtype="float64")
            if red_v.shape == vec.shape:
                vec = vec * red_v
                LOG.info(
                    "compare_policies: charging redemption-weighted expected costs "
                    "(cost_a * redemption_a). oracle_gap charges FULL arm costs unless "
                    "use_redemption=True -- keep one convention across a report."
                )
    else:
        vec = np.asarray(costs, dtype="float64")
    if vec.ndim == 2:
        if vec.shape[0] != n:
            raise ValueError(f"cost matrix has {vec.shape[0]} rows but panel_test has {n}")
        return vec
    if vec.ndim != 1:
        raise ValueError(f"costs must be 1-D or 2-D, got shape {vec.shape}")
    if vec.shape[0] < n_arms:
        raise ValueError(f"costs has {vec.shape[0]} entries but {n_arms} arms are in play")
    return np.tile(vec[:n_arms], (n, 1))


# =======================================================================================
# public API
# =======================================================================================
def evaluate_policy(
    panel_test: pd.DataFrame,
    assignment: np.ndarray,
    propensity: np.ndarray,
    outcome_col: str,
    mu_hat: np.ndarray | None = None,
    n_boot: int = 500,
    random_state: int | np.random.Generator | None = None,
    *,
    weight_cap: float = DEFAULT_WEIGHT_CAP,
    alpha: float = 0.05,
    policy_name: str = "evaluated",
) -> pd.DataFrame:
    """Estimate the value of ``assignment`` from logged data by IPW, SNIPW and DR.

    All three estimators answer ``E[Y(pi(X))]`` -- the mean outcome the evaluated policy
    would have produced -- from data logged under a different policy. See the module
    docstring for the estimating equations.

    Parameters
    ----------
    panel_test : pandas.DataFrame
        Logged test data. Must contain the logged arm column ``"arm"`` and ``outcome_col``.
        Row order defines the alignment of every array argument.
    assignment : numpy.ndarray
        ``(n,)`` integer arm chosen by the evaluated policy, ``pi(X_i)``; ``0`` = no offer.
    propensity : numpy.ndarray
        ``(n, n_arms)`` logging propensities ``e_a(X_i)``, or ``(n,)`` propensities of the
        *logged* arm (sufficient, because the propensity only ever multiplies the match
        indicator). Rows need not sum to one; a warning is logged if they do not.
    outcome_col : str
        Column of ``panel_test`` holding the logged outcome ``Y_i``; higher is better.
    mu_hat : numpy.ndarray or None, default None
        ``(n, n_arms)`` outcome-model predictions ``mu_a(X_i)``, ideally cross-fitted. When
        ``None`` the DR row falls back to SNIPW and its ``method`` reads
        ``"dr[fallback=snipw]"`` -- never silently.
    n_boot : int, default 500
        Bootstrap resamples of the per-row contributions. ``0`` disables the bootstrap and
        the intervals become normal intervals from ``se_analytic``.
    random_state : int or numpy.random.Generator or None, default None
        Seed for the bootstrap; normalised with :func:`prism.utils.seeds.as_rng`.
    weight_cap : float, default 100.0
        Importance weights are truncated here; ``n_clipped`` reports how many rows hit it.
    alpha : float, default 0.05
        Two-sided level; the interval is the ``(alpha/2, 1-alpha/2)`` bootstrap percentile.
    policy_name : str, default "evaluated"
        Label used in log messages only.

    Returns
    -------
    pandas.DataFrame
        One row per method (``ipw``, ``snipw``, ``dr``) with columns
        ``method, value, se_analytic, se_boot, ci_low, ci_high, n, n_matched, ess,
        max_weight, n_clipped, reliable``. ``max_weight`` is the pre-clipping maximum and
        ``reliable`` applies :data:`RELIABILITY_RULE`.

    Raises
    ------
    KeyError
        If ``panel_test`` lacks ``"arm"`` or ``outcome_col``.
    ValueError
        On shape mismatches, non-finite or non-integral arm indices, non-finite
        outcomes/predictions, ``weight_cap <= 0`` or ``alpha`` outside ``(0, 1)``.

    Notes
    -----
    ``se_analytic`` and ``se_boot`` are both reported on purpose. Agreement is evidence the
    normal approximation holds; disagreement of more than ~25% usually means a few rows with
    enormous weights dominate the estimate, in which case ``reliable`` is typically already
    ``False``.

    If the policy picks an action the logging policy never played (``n_matched == 0``) the
    re-weighting estimand is not identified, and the IPW/SNIPW cells come back as ``NaN``
    rather than the ``0.0 +/- 0.0`` the formula would mechanically produce. DR survives when
    an outcome model is supplied, but is then pure extrapolation from that model.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> arm = rng.integers(0, 2, n)
    >>> df = pd.DataFrame({"arm": arm, "y": rng.normal(size=n) + arm})
    >>> e = np.full((n, 2), 0.5)
    >>> mu = np.column_stack([np.zeros(n), np.ones(n)])
    >>> out = evaluate_policy(df, np.ones(n, int), e, "y", mu_hat=mu, n_boot=50, random_state=0)
    >>> list(out["method"])
    ['ipw', 'snipw', 'dr']
    >>> bool(abs(out.set_index("method").loc["dr", "value"] - 1.0) < 0.15)
    True

    Omitting ``mu_hat`` logs a warning and relabels the third row
    ``"dr[fallback=snipw]"`` (:data:`DR_FALLBACK_METHOD`); downstream code filtering on
    ``method == "dr"`` would then get an empty frame, so filter with
    ``method.str.startswith("dr")`` or always pass an outcome model.
    """
    weight_cap = _check_weight_cap(weight_cap)
    alpha = _check_alpha(alpha)
    y, arm, prop, mu, n_arms, bounded = _validate_inputs(panel_test, outcome_col, propensity, mu_hat)
    n = y.shape[0]
    pi = _as_1d_int(assignment, n, "assignment")
    if bounded and pi.max(initial=0) >= n_arms:
        raise ValueError(f"assignment contains arm {int(pi.max())} but only {n_arms} arms exist")

    contrib = _policy_contributions(y, arm, pi, prop, mu, weight_cap, policy_name)
    if contrib.n_matched == 0:
        LOG.warning(
            "policy %r matches the logged arm on 0/%d rows: IPW/SNIPW are undefined and DR "
            "collapses to the outcome model alone",
            policy_name,
            n,
        )

    cols, specs = _assemble_columns([contrib])
    point = _method_values(cols.mean(axis=0), specs[0])
    ses = _analytic_ses(contrib, float(point["snipw"]))

    rng = as_rng(random_state)
    boot_means = _bootstrap_means(cols, int(n_boot), rng)
    boot_vals = _method_values(boot_means, specs[0]) if boot_means.size else None

    has_mu = contrib.dr is not None
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        if _is_unidentified(contrib, method):
            value = se_a = se_b = lo = hi = float("nan")
        else:
            value = float(point[method])
            se_a = float(ses[method])
            replicates = None if boot_vals is None else np.asarray(boot_vals[method], dtype="float64")
            se_b, lo, hi = _interval(value, replicates, se_a, alpha)
        rows.append(
            {
                "method": _method_label(method, has_mu),
                "value": value,
                "se_analytic": se_a,
                "se_boot": se_b,
                "ci_low": lo,
                "ci_high": hi,
                "n": int(contrib.n),
                "n_matched": int(contrib.n_matched),
                "ess": float(contrib.ess),
                "max_weight": float(contrib.max_weight),
                "n_clipped": int(contrib.n_clipped),
                "reliable": bool(contrib.reliable),
            }
        )
    return pd.DataFrame(rows, columns=list(EVALUATE_POLICY_COLUMNS))


def compare_policies(
    policies: Mapping[str, np.ndarray],
    panel_test: pd.DataFrame,
    propensity: np.ndarray,
    outcome_col: str,
    mu_hat: np.ndarray | None = None,
    **kw: Any,
) -> pd.DataFrame:
    """Score several policies on the same logged data, with *paired* differences.

    Every policy is evaluated with the same three estimators and -- importantly -- the same
    bootstrap resamples. That lets the difference against a baseline policy be bootstrapped
    *pairwise*: for resample ``b``, ``d_b = V_policy(b) - V_baseline(b)``. Because the two
    policies share rows, ``Var(d)`` is far smaller than ``Var(V_policy) + Var(V_baseline)``,
    and the pairing is what ``beats_baseline`` is built on (``diff_ci_low > 0``).

    The conventional alternative -- declaring a winner when two marginal confidence intervals
    do not overlap -- is *conservative and inferior*: non-overlap is a sufficient but far
    from necessary condition for a significant difference, so it throws away real wins,
    and it ignores the positive correlation between two policies scored on the same rows.
    Both marginal intervals are still reported (``ci_low``/``ci_high``) so a reader can see
    them, but the verdict comes from the paired interval.

    Parameters
    ----------
    policies : Mapping[str, numpy.ndarray]
        ``{name: assignment}``; each assignment is ``(n,)`` integer arms aligned to
        ``panel_test``.
    panel_test : pandas.DataFrame
        Logged test data with the ``"arm"`` column and ``outcome_col``.
    propensity : numpy.ndarray
        ``(n, n_arms)`` logging propensities (or ``(n,)`` for the logged arm).
    outcome_col : str
        Logged outcome column, higher is better.
    mu_hat : numpy.ndarray or None, default None
        ``(n, n_arms)`` outcome model. ``None`` degrades DR to SNIPW with a flagged label.
    **kw
        ``baseline`` (str, default ``"treat_none"``) -- the policy every difference is taken
        against. If it is not in ``policies`` an all-control assignment is synthesised for
        the comparison and logged; no extra row is added to the output.
        ``costs`` -- per-arm costs, an ``(n, n_arms)`` matrix, or an object with
        ``arm_costs`` (see :func:`_cost_matrix`); drives ``total_cost``, NaN if omitted.
        ``n_boot`` (int, default 500), ``random_state``, ``weight_cap``
        (default :data:`DEFAULT_WEIGHT_CAP`, must be finite and > 0) and ``alpha``
        (default 0.05, must lie in ``(0, 1)``) behave exactly as in
        :func:`evaluate_policy`. Unknown keys are ignored with a warning.

        Note that ``costs`` given as an object exposing ``offer_redemption_rate`` yields
        *expected* (redemption-weighted) costs here, whereas :func:`oracle_gap` charges the
        full arm cost unless ``use_redemption=True``. Mixing the two conventions in one
        report makes a policy look cheaper than the oracle it is graded against; the
        redemption path logs at INFO so the choice is visible.

    Returns
    -------
    pandas.DataFrame
        Tidy, one row per ``(policy, method)``, columns
        ``policy, method, value, se_analytic, se_boot, ci_low, ci_high, diff_vs_baseline,
        diff_ci_low, diff_ci_high, beats_baseline, n, n_treated, total_cost, n_matched,
        ess, max_weight, n_clipped, reliable``. Sorted by policy (input order) then method.

    Notes
    -----
    ``diff_vs_baseline`` is a difference in *policy value per customer*. Multiply by ``n`` for
    the campaign-level number, and subtract ``total_cost`` differences if the outcome is
    gross rather than net of the offer cost.
    """
    baseline: str = kw.pop("baseline", "treat_none")
    costs = kw.pop("costs", None)
    n_boot = int(kw.pop("n_boot", 500))
    random_state = kw.pop("random_state", None)
    weight_cap = _check_weight_cap(kw.pop("weight_cap", DEFAULT_WEIGHT_CAP))
    alpha = _check_alpha(kw.pop("alpha", 0.05))
    if kw:
        LOG.warning("compare_policies ignoring unknown keyword(s): %s", sorted(kw))

    if not isinstance(policies, Mapping) or len(policies) == 0:
        raise ValueError("policies must be a non-empty mapping {name: assignment}")

    y, arm, prop, mu, n_arms, bounded = _validate_inputs(panel_test, outcome_col, propensity, mu_hat)
    n = y.shape[0]

    named: dict[str, np.ndarray] = {}
    for name, assignment in policies.items():
        pi = _as_1d_int(assignment, n, f"policies[{name!r}]")
        if bounded and pi.max(initial=0) >= n_arms:
            raise ValueError(f"policies[{name!r}] contains arm {int(pi.max())} but only {n_arms} arms exist")
        n_arms = max(n_arms, int(pi.max(initial=0)) + 1)
        named[str(name)] = pi

    reported = list(named)
    if baseline not in named:
        if baseline == "treat_none":
            LOG.info(
                "baseline %r absent from policies; synthesising an all-control assignment for "
                "the paired comparison (it is not added to the output)",
                baseline,
            )
            named[baseline] = np.zeros(n, dtype="int64")
        else:
            raise KeyError(f"baseline policy {baseline!r} is not in policies ({sorted(named)})")

    order = reported + ([baseline] if baseline not in reported else [])
    contribs = [_policy_contributions(y, arm, named[k], prop, mu, weight_cap, k) for k in order]
    cols, specs = _assemble_columns(contribs)

    mean_cols = cols.mean(axis=0)
    point = {k: _method_values(mean_cols, spec) for k, spec in zip(order, specs)}

    rng = as_rng(random_state)
    boot_means = _bootstrap_means(cols, n_boot, rng)
    boot = (
        {k: _method_values(boot_means, spec) for k, spec in zip(order, specs)}
        if boot_means.size
        else None
    )

    cost_mat = _cost_matrix(costs, n, n_arms)
    has_mu = mu is not None
    lo_q, hi_q = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)

    base_contrib = contribs[order.index(baseline)]
    rows: list[dict[str, Any]] = []
    for key, contrib in zip(order, contribs):
        if key not in reported:
            continue
        pi = named[key]
        ses = _analytic_ses(contrib, float(point[key]["snipw"]))
        n_treated = int(np.sum(pi > 0))
        total_cost = float(cost_mat[np.arange(n), pi].sum()) if cost_mat is not None else float("nan")
        is_baseline = key == baseline

        for method in METHODS:
            unidentified = _is_unidentified(contrib, method) or _is_unidentified(base_contrib, method)
            if _is_unidentified(contrib, method):
                value = se_a = se_b = ci_lo = ci_hi = float("nan")
                reps = None
            else:
                value = float(point[key][method])
                se_a = float(ses[method])
                reps = None if boot is None else np.asarray(boot[key][method], dtype="float64")
                se_b, ci_lo, ci_hi = _interval(value, reps, se_a, alpha)

            if unidentified:
                diff = d_lo = d_hi = float("nan")
                beats = False
            elif is_baseline:
                diff, d_lo, d_hi = 0.0, 0.0, 0.0
                beats = False
            else:
                diff = value - float(point[baseline][method])
                if boot is not None:
                    paired = reps - np.asarray(boot[baseline][method], dtype="float64")
                    finite = paired[np.isfinite(paired)]
                    if finite.size >= 2:
                        d_lo = float(np.percentile(finite, lo_q))
                        d_hi = float(np.percentile(finite, hi_q))
                    else:
                        d_lo = d_hi = float("nan")
                else:
                    d_lo = d_hi = float("nan")
                beats = bool(np.isfinite(d_lo) and d_lo > 0.0)

            rows.append(
                {
                    "policy": key,
                    "method": _method_label(method, has_mu),
                    "value": value,
                    "se_analytic": se_a,
                    "se_boot": se_b,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "diff_vs_baseline": diff,
                    "diff_ci_low": d_lo,
                    "diff_ci_high": d_hi,
                    "beats_baseline": beats,
                    "n": int(contrib.n),
                    "n_treated": n_treated,
                    "total_cost": total_cost,
                    "n_matched": int(contrib.n_matched),
                    "ess": float(contrib.ess),
                    "max_weight": float(contrib.max_weight),
                    "n_clipped": int(contrib.n_clipped),
                    "reliable": bool(contrib.reliable),
                }
            )
    return pd.DataFrame(rows, columns=list(COMPARE_POLICIES_COLUMNS))


def oracle_gap(
    assignment: np.ndarray,
    ground_truth: pd.DataFrame,
    econ: Any,
    *,
    use_redemption: bool = False,
) -> dict[str, Any]:
    """Score an assignment against the simulator's *known* optimal assignment.

    This is the metric no real project can report. Because ``prism.data.dgp`` computes every
    counterfactual value analytically, the realised net value of a policy can be evaluated
    under the true effects rather than estimated::

        net_a(i)    = gt_tau_value_a(i) - cost_a          for a >= 1,   net_0(i) = 0
        realised(i) = net_{assignment_i}(i)
        oracle(i)   = gt_oracle_net_value(i) = max(0, max_a net_a(i))
        regret(i)   = oracle(i) - realised(i)             (always >= 0)

    Parameters
    ----------
    assignment : numpy.ndarray
        ``(n,)`` chosen arm per row, positionally aligned to ``ground_truth``.
    ground_truth : pandas.DataFrame
        Ground-truth frame (see ``SPEC.md`` s2.3). Uses ``gt_tau_value_1..K`` when present,
        otherwise reconstructs them from ``gt_value_0..K``; uses ``gt_oracle_net_value`` and
        ``gt_best_arm`` when present and recomputes them otherwise.
    econ : object or sequence or None
        Anything exposing ``arm_costs`` (e.g. ``prism.decision.economics.EconomicConfig``), a
        plain per-arm cost sequence, or ``None`` to use
        :data:`prism.data.schema.ARM_COSTS`.
    use_redemption : bool, default False
        If True and ``econ`` exposes ``offer_redemption_rate``, charge the *expected* cost
        ``cost_a * redemption_a``. Off by default on purpose: ``gt_oracle_net_value`` is
        defined in the DGP with full undiscounted costs, and mixing the two conventions
        inflates ``pct_of_oracle_captured`` by making the evaluated policy look cheaper than
        the oracle it is measured against.

    Returns
    -------
    dict
        ``available`` (bool) plus ``n``, ``n_treated``, ``total_cost``,
        ``realised_net_value``, ``realised_net_value_mean``, ``oracle_net_value``,
        ``oracle_net_value_mean``, ``pct_of_oracle_captured``, ``regret_total``,
        ``regret_mean``, ``pct_of_rows_matching_gt_best_arm``, and the provenance keys
        ``oracle_source`` / ``best_arm_source`` / ``reason``.

        If the ground-truth columns are absent the same keys come back as ``NaN`` with
        ``available=False`` and a human-readable ``reason``, so a pipeline running on real
        data (where no ground truth exists) degrades instead of crashing.

    Raises
    ------
    ValueError
        Only if ``assignment`` and ``ground_truth`` have different lengths, which is a
        caller bug rather than missing data.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> gt = pd.DataFrame({"gt_tau_value_1": [40.0, 5.0]})
    >>> res = oracle_gap(np.array([1, 0]), gt, None)
    >>> bool(res["available"]), round(res["pct_of_oracle_captured"], 3)
    (True, 1.0)
    """
    pi = np.asarray(assignment).reshape(-1).astype("int64", copy=False)
    n = int(pi.shape[0])

    def _unavailable(reason: str) -> dict[str, Any]:
        LOG.warning("oracle_gap unavailable: %s", reason)
        nan = float("nan")
        return {
            "available": False,
            "reason": reason,
            "n": n,
            "n_treated": int(np.sum(pi > 0)) if n else 0,
            "total_cost": nan,
            "realised_net_value": nan,
            "realised_net_value_mean": nan,
            "oracle_net_value": nan,
            "oracle_net_value_mean": nan,
            "pct_of_oracle_captured": nan,
            "regret_total": nan,
            "regret_mean": nan,
            "pct_of_rows_matching_gt_best_arm": nan,
            "oracle_source": None,
            "best_arm_source": None,
        }

    if not isinstance(ground_truth, pd.DataFrame):
        return _unavailable(f"ground_truth is {type(ground_truth).__name__}, not a DataFrame")
    if len(ground_truth) != n:
        raise ValueError(
            f"assignment has {n} rows but ground_truth has {len(ground_truth)}; "
            "they must be positionally aligned"
        )
    if n == 0:
        return _unavailable("ground_truth is empty")

    cols = set(ground_truth.columns)
    tau_cols = sorted(c for c in cols if c.startswith("gt_tau_value_"))
    val_cols = sorted(c for c in cols if c.startswith("gt_value_"))

    if tau_cols:
        n_arms = len(tau_cols) + 1
        tau = np.zeros((n, n_arms), dtype="float64")
        for a in range(1, n_arms):
            name = f"gt_tau_value_{a}"
            if name not in cols:
                return _unavailable(f"expected ground-truth column {name!r} is missing")
            tau[:, a] = pd.to_numeric(ground_truth[name], errors="coerce").to_numpy("float64")
    elif len(val_cols) >= 2:
        n_arms = len(val_cols)
        base = pd.to_numeric(ground_truth["gt_value_0"], errors="coerce").to_numpy("float64")
        tau = np.zeros((n, n_arms), dtype="float64")
        for a in range(1, n_arms):
            tau[:, a] = pd.to_numeric(ground_truth[f"gt_value_{a}"], errors="coerce").to_numpy("float64") - base
        LOG.info("oracle_gap: reconstructed gt_tau_value_* from gt_value_* columns")
    else:
        return _unavailable("no gt_tau_value_* or gt_value_* columns in ground_truth")

    if not np.all(np.isfinite(tau)):
        return _unavailable("ground-truth value columns contain non-finite entries")
    if pi.max(initial=0) >= n_arms:
        raise ValueError(f"assignment contains arm {int(pi.max())} but ground truth covers {n_arms} arms")

    # --- cost vector, matching the DGP's own convention by default ---------------------
    red: Any = None
    if econ is None:
        cost_vec = np.asarray(ARM_COSTS, dtype="float64")
    elif hasattr(econ, "arm_costs"):
        cost_vec = np.asarray(econ.arm_costs, dtype="float64")
        red = getattr(econ, "offer_redemption_rate", None)
    elif isinstance(econ, Mapping) and "arm_costs" in econ:
        cost_vec = np.asarray(econ["arm_costs"], dtype="float64")
        red = econ.get("offer_redemption_rate")
    else:
        cost_vec = np.asarray(econ, dtype="float64").reshape(-1)
    if use_redemption and red is not None:
        red_v = np.asarray(red, dtype="float64")
        if red_v.shape == cost_vec.shape:
            cost_vec = cost_vec * red_v
        else:
            LOG.warning("offer_redemption_rate shape %s != arm_costs shape %s; ignored",
                        red_v.shape, cost_vec.shape)
    elif use_redemption:
        LOG.warning("use_redemption=True but econ exposes no offer_redemption_rate; "
                    "charging full arm costs")
    if cost_vec.shape[0] < n_arms:
        raise ValueError(f"econ supplies {cost_vec.shape[0]} arm costs but {n_arms} arms are in play")
    cost_vec = cost_vec[:n_arms].copy()
    cost_vec[0] = 0.0

    net = tau - cost_vec[None, :]
    net[:, 0] = 0.0
    rows = np.arange(n)
    realised = net[rows, pi]

    if "gt_oracle_net_value" in cols:
        oracle = pd.to_numeric(ground_truth["gt_oracle_net_value"], errors="coerce").to_numpy("float64")
        oracle_source = "gt_oracle_net_value"
        if not np.all(np.isfinite(oracle)):
            LOG.warning("gt_oracle_net_value has non-finite entries; recomputing from gt_tau_value_*")
            oracle = np.maximum(net.max(axis=1), 0.0)
            oracle_source = "recomputed"
    else:
        oracle = np.maximum(net.max(axis=1), 0.0)
        oracle_source = "recomputed"

    if "gt_best_arm" in cols:
        best = pd.to_numeric(ground_truth["gt_best_arm"], errors="coerce").to_numpy("float64")
        best_arm = np.where(np.isfinite(best), best, -1).astype("int64")
        best_arm_source = "gt_best_arm"
    else:
        best_arm = np.argmax(net, axis=1).astype("int64")
        best_arm[net.max(axis=1) <= 0.0] = 0
        best_arm_source = "recomputed"

    realised_total = float(realised.sum())
    oracle_total = float(oracle.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = realised_total / oracle_total if abs(oracle_total) > _EPS else float("nan")
    if not np.isfinite(pct):
        LOG.warning("oracle net value is ~0 for this sample; pct_of_oracle_captured is undefined")

    regret = oracle - realised
    total_cost = float(cost_vec[pi].sum())

    return {
        "available": True,
        "reason": "",
        "n": n,
        "n_treated": int(np.sum(pi > 0)),
        "total_cost": total_cost,
        "realised_net_value": realised_total,
        "realised_net_value_mean": float(realised.mean()),
        "oracle_net_value": oracle_total,
        "oracle_net_value_mean": float(oracle.mean()),
        "pct_of_oracle_captured": float(pct),
        "regret_total": float(regret.sum()),
        "regret_mean": float(regret.mean()),
        "pct_of_rows_matching_gt_best_arm": float(np.mean(pi == best_arm)),
        "oracle_source": oracle_source,
        "best_arm_source": best_arm_source,
    }


# =======================================================================================
# smoke test: a logged bandit whose true policy value is known analytically
# =======================================================================================
def _simulate_logged_bandit(
    n: int = 20_000,
    n_arms: int = 4,
    confounding: float = 1.1,
    noise: float = 3.0,
    prop_floor: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> dict[str, np.ndarray]:
    """Simulate a confounded multi-arm logged bandit with all potential outcomes known.

    Every potential outcome ``Y_i(a)`` is drawn, then exactly one is revealed under a known
    logging propensity that depends on the same covariates as the outcome level. The true
    value of any policy is therefore available in closed form as
    ``mean_i mu_true(X_i, pi(X_i))``, which is what the smoke test grades against.

    Parameters
    ----------
    n : int, default 20000
        Rows.
    n_arms : int, default 4
        Number of arms (arm 0 = control).
    confounding : float, default 1.1
        Scale of the covariate dependence of the logging policy. ``0`` gives an RCT.
    noise : float, default 3.0
        Outcome noise standard deviation.
    prop_floor : float, default 0.05
        Propensities are floored here and renormalised, guaranteeing overlap.
    random_state : int or numpy.random.Generator or None, default 0
        Seed.

    Returns
    -------
    dict
        ``X, mu_true, y_all, propensity, arm, y`` as numpy arrays.
    """
    rng = as_rng(random_state)
    x0, x1, x2, x3 = (rng.normal(size=n) for _ in range(4))
    X = np.column_stack([x0, x1, x2, x3])

    base = 30.0 + 8.0 * x0 + 4.0 * x1 - 5.0 * x2 + 2.0 * x0 * x1
    tau = np.zeros((n, n_arms), dtype="float64")
    if n_arms > 1:
        tau[:, 1] = 3.0 + 2.0 * x1 - 1.0 * x0
    if n_arms > 2:
        tau[:, 2] = 5.0 - 3.0 * x2 + 2.0 * np.maximum(x3, 0.0)
    if n_arms > 3:
        tau[:, 3] = 1.0 + 4.0 * x2 - 2.0 * x1
    mu_true = base[:, None] + tau

    y_all = mu_true + rng.normal(scale=noise, size=(n, n_arms))

    # Logging policy: treats high-value, low-risk customers -- exactly the customers with a
    # high outcome level, so the naive mean of logged outcomes is badly confounded.
    score = np.zeros((n, n_arms), dtype="float64")
    if n_arms > 1:
        score[:, 1] = confounding * (0.8 * x0 + 0.5 * x1)
    if n_arms > 2:
        score[:, 2] = confounding * (1.0 * x0 - 0.6 * x2)
    if n_arms > 3:
        score[:, 3] = confounding * (1.2 * x0 + 0.7 * x3)
    score -= score.max(axis=1, keepdims=True)
    p = np.exp(score)
    p /= p.sum(axis=1, keepdims=True)
    p = np.maximum(p, prop_floor)
    p /= p.sum(axis=1, keepdims=True)

    u = rng.random(n)
    arm = (p.cumsum(axis=1) < u[:, None]).sum(axis=1).clip(0, n_arms - 1).astype("int64")
    y = y_all[np.arange(n), arm]
    return {"X": X, "mu_true": mu_true, "y_all": y_all, "propensity": p, "arm": arm, "y": y}


def _crossfit_outcome_model(
    X: np.ndarray, arm: np.ndarray, y: np.ndarray, n_arms: int, seed: int = 0
) -> np.ndarray:
    """Return a 2-fold cross-fitted ``(n, n_arms)`` outcome model ``mu_a(X)``.

    Cross-fitting matters: an in-sample ``mu_hat`` correlates with the residual it is meant
    to correct, which quietly re-introduces the bias DR exists to remove.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import StratifiedKFold

    n = X.shape[0]
    onehot = np.eye(n_arms)[arm]
    design = np.hstack([X, onehot])
    mu = np.zeros((n, n_arms), dtype="float64")
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    for tr, te in skf.split(design, arm):
        model = HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.1, min_samples_leaf=40, random_state=seed
        )
        model.fit(design[tr], y[tr])
        for a in range(n_arms):
            counterfactual = np.hstack([X[te], np.tile(np.eye(n_arms)[a], (len(te), 1))])
            mu[te, a] = model.predict(counterfactual)
    return mu


if __name__ == "__main__":  # pragma: no cover - smoke test
    import time

    t_start = time.perf_counter()
    np.set_printoptions(suppress=True)
    pd.set_option("display.width", 200)

    N, K, SEED = 20_000, 4, 11
    sim = _simulate_logged_bandit(n=N, n_arms=K, random_state=SEED)
    X, mu_true, y_all = sim["X"], sim["mu_true"], sim["y_all"]
    e, arm, y = sim["propensity"], sim["arm"], sim["y"]
    panel = pd.DataFrame({"arm": arm, "value_h": y})

    # The policy under evaluation: the oracle arm by true conditional mean. Its value is
    # known exactly, which is what makes this test a test and not a demo.
    pi_star = np.argmax(mu_true, axis=1).astype("int64")
    truth = float(mu_true[np.arange(N), pi_star].mean())

    mu_hat = _crossfit_outcome_model(X, arm, y, K, seed=SEED)

    print("=" * 96)
    print("PRISM off-policy evaluation -- logged bandit with known ground truth")
    print("=" * 96)
    print(
        f"n={N:,}  arms={K}  logging propensity in [{e.min():.3f}, {e.max():.3f}]  "
        f"logged-arm share matching pi: {np.mean(arm == pi_star):.3f}"
    )
    print(f"policy under evaluation: oracle argmax_a mu_true(X, a)   TRUE VALUE = {truth:.4f}")
    print(f"reliability rule: {RELIABILITY_RULE}")
    print()

    res = evaluate_policy(panel, pi_star, e, "value_h", mu_hat=mu_hat, n_boot=500, random_state=SEED)

    # The naive comparator every dashboard reports: the mean logged outcome among customers
    # who happened to receive the arm the policy would have chosen.
    match = arm == pi_star
    naive = float(y[match].mean())
    naive_se = float(np.std(y[match], ddof=1) / np.sqrt(match.sum()))

    print(f"{'estimator':<22}{'value':>10}{'se_analytic':>13}{'se_boot':>10}{'ci_low':>10}"
          f"{'ci_high':>10}{'bias':>9}{'|bias|/se':>11}")
    print("-" * 96)
    print(f"{'truth (analytic)':<22}{truth:>10.3f}{'-':>13}{'-':>10}{'-':>10}{'-':>10}{'-':>9}{'-':>11}")
    print(f"{'naive logged mean':<22}{naive:>10.3f}{naive_se:>13.3f}{'-':>10}"
          f"{naive - 1.96 * naive_se:>10.3f}{naive + 1.96 * naive_se:>10.3f}"
          f"{naive - truth:>+9.3f}{abs(naive - truth) / naive_se:>11.1f}")
    z_scores: dict[str, float] = {}
    for _, r in res.iterrows():
        z = abs(r["value"] - truth) / r["se_analytic"] if r["se_analytic"] > 0 else np.nan
        z_scores[str(r["method"])] = float(z)
        print(f"{str(r['method']):<22}{r['value']:>10.3f}{r['se_analytic']:>13.3f}{r['se_boot']:>10.3f}"
              f"{r['ci_low']:>10.3f}{r['ci_high']:>10.3f}{r['value'] - truth:>+9.3f}{z:>11.1f}")
    print("-" * 96)
    r0 = res.iloc[0]
    print(f"diagnostics: n={int(r0['n']):,}  n_matched={int(r0['n_matched']):,}  "
          f"ess={r0['ess']:.0f} ({r0['ess'] / r0['n']:.1%} of n)  max_weight={r0['max_weight']:.2f}  "
          f"n_clipped={int(r0['n_clipped'])}  reliable={bool(r0['reliable'])}")
    print()

    # ---------------------------------------------------------------- assertions --------
    ipw, snipw, dr = (res.set_index("method").loc[m] for m in METHODS)

    assert list(res.columns) == list(EVALUATE_POLICY_COLUMNS), res.columns
    assert len(res) == 3 and list(res["method"]) == list(METHODS)

    # 1. THE KEY PROPERTY: the causal estimators recover the known truth, the naive mean does not.
    for name in METHODS:
        assert z_scores[name] < 3.0, f"{name} is {z_scores[name]:.2f} SE from the truth"
    naive_z = abs(naive - truth) / naive_se
    assert naive_z > 10.0, f"naive estimator is not detectably biased (z={naive_z:.2f})"
    assert abs(naive - truth) > 8.0 * abs(dr["value"] - truth), "DR did not beat the naive mean"

    # 2. Variance ordering: self-normalisation and the outcome model both buy precision.
    assert snipw["se_analytic"] < ipw["se_analytic"], "SNIPW should be tighter than IPW"
    assert dr["se_analytic"] < ipw["se_analytic"], "DR should be tighter than IPW"

    # 3. The two standard errors must agree; if they do not, neither interval is trustworthy.
    for name, row in (("ipw", ipw), ("snipw", snipw), ("dr", dr)):
        ratio = row["se_boot"] / row["se_analytic"]
        assert 0.75 < ratio < 1.35, f"{name}: se_boot/se_analytic = {ratio:.3f}"
        assert row["ci_low"] < row["value"] < row["ci_high"]
        assert row["ci_low"] < truth < row["ci_high"], f"{name} CI missed the truth"

    # 4. Diagnostics behave, and clipping is reported rather than hidden.
    assert 0 < ipw["ess"] <= ipw["n_matched"] and bool(ipw["reliable"])
    assert int(ipw["n_clipped"]) == 0 and ipw["max_weight"] < DEFAULT_WEIGHT_CAP
    clipped = evaluate_policy(panel, pi_star, e, "value_h", mu_hat=mu_hat, n_boot=50,
                              random_state=SEED, weight_cap=2.0)
    assert int(clipped.iloc[0]["n_clipped"]) > 0 and not bool(clipped.iloc[0]["reliable"])
    assert clipped.iloc[0]["max_weight"] == ipw["max_weight"], "clipping must not launder max_weight"

    # 5. The DR fallback is announced, never silent.
    nomu = evaluate_policy(panel, pi_star, e, "value_h", mu_hat=None, n_boot=50, random_state=SEED)
    assert list(nomu["method"]) == ["ipw", "snipw", DR_FALLBACK_METHOD], list(nomu["method"])
    assert np.isclose(nomu.iloc[2]["value"], nomu.iloc[1]["value"])

    # 5b. Zero overlap is *unidentified*, not zero. Only the overlap structure matters here,
    #     so the logged arms are collapsed so that the top arm is never played.
    no_overlap = pd.DataFrame({"arm": np.minimum(arm, K - 2), "value_h": y})
    unid = evaluate_policy(no_overlap, np.full(N, K - 1), e, "value_h", mu_hat=mu_hat,
                           n_boot=20, random_state=SEED)
    assert int(unid.iloc[0]["n_matched"]) == 0
    assert np.isnan(unid.iloc[0]["value"]) and np.isnan(unid.iloc[1]["value"])
    assert np.isfinite(unid.iloc[2]["value"]) and not bool(unid.iloc[2]["reliable"])

    # 6. Determinism under a fixed seed.
    again = evaluate_policy(panel, pi_star, e, "value_h", mu_hat=mu_hat, n_boot=500, random_state=SEED)
    pd.testing.assert_frame_equal(res, again)

    # 7. A 1-D propensity of the logged arm is equivalent to the full matrix -- exactly,
    #    not approximately, so a caller who only stored e_{A_i} loses nothing.
    one_d = evaluate_policy(panel, pi_star, e[np.arange(N), arm], "value_h", mu_hat=mu_hat,
                            n_boot=500, random_state=SEED)
    pd.testing.assert_frame_equal(one_d, res)

    # 8. BRUTE FORCE. Everything above tests the module against itself. This tests the
    #    vectorised code against the textbook formula written out as a Python loop on an
    #    instance small enough to check by hand (n=12, 4 arms, every constant policy).
    brng = np.random.default_rng(3)
    bn, bK = 12, 4
    b_arm = brng.integers(0, bK, bn)
    b_e = brng.uniform(0.15, 0.6, size=(bn, bK))
    b_e /= b_e.sum(1, keepdims=True)
    b_y = brng.normal(10.0, 3.0, bn)
    b_mu = brng.normal(10.0, 2.0, size=(bn, bK))
    b_df = pd.DataFrame({"arm": b_arm, "y": b_y})
    for b_pi in [brng.integers(0, bK, bn)] + [np.full(bn, a) for a in range(bK)]:
        bw = np.array([min(1.0 / b_e[i, b_pi[i]], DEFAULT_WEIGHT_CAP) if b_arm[i] == b_pi[i]
                       else 0.0 for i in range(bn)])
        want_ipw = sum(bw[i] * b_y[i] for i in range(bn)) / bn
        want_sn = sum(bw[i] * b_y[i] for i in range(bn)) / sum(bw[i] for i in range(bn))
        want_dr = sum(b_mu[i, b_pi[i]] + bw[i] * (b_y[i] - b_mu[i, b_pi[i]])
                      for i in range(bn)) / bn
        want_se_ipw = float(np.std(bw * b_y, ddof=1) / np.sqrt(bn))
        got = evaluate_policy(b_df, b_pi, b_e, "y", mu_hat=b_mu, n_boot=0).set_index("method")
        for meth, want in (("ipw", want_ipw), ("snipw", want_sn), ("dr", want_dr)):
            assert abs(float(got.loc[meth, "value"]) - want) < 1e-12, (meth, want)
        assert abs(float(got.loc["ipw", "se_analytic"]) - want_se_ipw) < 1e-12
        assert int(got.iloc[0]["n_matched"]) == int(np.sum(b_arm == b_pi))
        assert abs(float(got.iloc[0]["ess"]) - bw.sum() ** 2 / np.square(bw).sum()) < 1e-12

    # 9. REPEATED SAMPLING. One z-score can pass by luck; bias is a property of the
    #    sampling distribution, so replicate and compare the mean bias to its own Monte
    #    Carlo standard error. mu_true is used as the outcome model here on purpose --
    #    with a correct mu, DR must be essentially exact.
    R_REP, N_REP = 25, 4_000
    biases = {m: [] for m in METHODS}
    for rep in range(R_REP):
        s = _simulate_logged_bandit(n=N_REP, n_arms=K, random_state=1_000 + rep)
        s_pi = np.argmax(s["mu_true"], axis=1)
        s_truth = float(s["mu_true"][np.arange(N_REP), s_pi].mean())
        s_res = evaluate_policy(pd.DataFrame({"arm": s["arm"], "value_h": s["y"]}), s_pi,
                                s["propensity"], "value_h", mu_hat=s["mu_true"], n_boot=0)
        for _, rr in s_res.iterrows():
            biases[str(rr["method"])].append(float(rr["value"]) - s_truth)
    print(f"repeated sampling: {R_REP} independent replications at n={N_REP:,}")
    print(f"{'method':<10}{'mean bias':>12}{'mc se':>10}{'z':>8}{'emp sd':>10}")
    for m in METHODS:
        b = np.array(biases[m])
        mc_se = float(b.std(ddof=1) / np.sqrt(R_REP))
        z_rep = float(b.mean() / mc_se)
        print(f"{m:<10}{b.mean():>+12.4f}{mc_se:>10.4f}{z_rep:>8.2f}{b.std(ddof=1):>10.4f}")
        assert abs(z_rep) < 4.0, f"{m} is biased across replications (z={z_rep:.2f})"
    print()

    # 10. Settings that would produce a confident lie are refused, not accommodated.
    #     weight_cap <= 0 zeroes (or sign-flips) every weight; alpha outside (0, 1) yields
    #     a zero-width or *inverted* interval that reads as an ordinary result downstream.
    for bad_kw in ({"weight_cap": 0.0}, {"weight_cap": -1.0}, {"weight_cap": float("inf")},
                   {"alpha": 0.0}, {"alpha": 1.0}, {"alpha": 1.5}, {"alpha": -0.1}):
        try:
            evaluate_policy(panel, pi_star, e, "value_h", mu_hat=mu_hat, n_boot=0, **bad_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"evaluate_policy accepted {bad_kw}")
        try:
            compare_policies({"a": pi_star}, panel, e, "value_h", mu_hat=mu_hat, n_boot=0, **bad_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"compare_policies accepted {bad_kw}")
    #     A float assignment is fine when it is integral and refused when it is not: an
    #     unrounded argmax score of 1.9 would silently be truncated to arm 1.
    pd.testing.assert_frame_equal(
        evaluate_policy(panel, pi_star.astype("float64"), e, "value_h", mu_hat=mu_hat, n_boot=0),
        evaluate_policy(panel, pi_star, e, "value_h", mu_hat=mu_hat, n_boot=0),
    )
    for bad_pi in (np.full(N, 1.9), np.full(N, -1)):
        try:
            evaluate_policy(panel, bad_pi, e, "value_h", mu_hat=mu_hat, n_boot=0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"evaluate_policy accepted assignment {bad_pi[0]!r}")

    # 11. n_boot=0 takes the normal-interval path: se_boot is NaN (not a fake 0) and the
    #     interval is exactly the analytic one.
    nb0 = evaluate_policy(panel, pi_star, e, "value_h", mu_hat=mu_hat, n_boot=0).set_index("method")
    assert bool(nb0["se_boot"].isna().all()), "n_boot=0 must not invent a bootstrap SE"
    from scipy.stats import norm as _norm
    z95 = float(_norm.ppf(0.975))
    for m in METHODS:
        r = nb0.loc[m]
        assert np.isclose(r["ci_low"], r["value"] - z95 * r["se_analytic"])
        assert np.isclose(r["ci_high"], r["value"] + z95 * r["se_analytic"])

    # ------------------------------------------------------- compare_policies -----------
    rng = as_rng(SEED + 1)
    policies = {
        "treat_none": np.zeros(N, dtype="int64"),
        "treat_all_discount_10": np.ones(N, dtype="int64"),
        "random": rng.integers(0, K, N),
        "oracle_cate": pi_star,
    }
    cmp_df = compare_policies(policies, panel, e, "value_h", mu_hat=mu_hat, n_boot=500,
                              random_state=SEED, costs=ARM_COSTS[:K], baseline="treat_none")
    show = cmp_df[cmp_df["method"] == "dr"]
    print("compare_policies (DR rows; paired bootstrap difference vs treat_none)")
    print(f"{'policy':<24}{'value':>9}{'ci_low':>9}{'ci_high':>9}{'diff':>9}{'diff_lo':>9}"
          f"{'diff_hi':>9}{'beats':>7}{'n_treated':>11}{'total_cost':>12}")
    print("-" * 96)
    for _, r in show.iterrows():
        print(f"{r['policy']:<24}{r['value']:>9.3f}{r['ci_low']:>9.3f}{r['ci_high']:>9.3f}"
              f"{r['diff_vs_baseline']:>+9.3f}{r['diff_ci_low']:>+9.3f}{r['diff_ci_high']:>+9.3f}"
              f"{str(bool(r['beats_baseline'])):>7}{int(r['n_treated']):>11,}{r['total_cost']:>12,.0f}")
    print("-" * 96)

    assert list(cmp_df.columns) == list(COMPARE_POLICIES_COLUMNS)
    assert set(cmp_df["policy"]) == set(policies)
    assert len(cmp_df) == len(policies) * 3
    base_row = show[show["policy"] == "treat_none"].iloc[0]
    assert base_row["diff_vs_baseline"] == 0.0 and not bool(base_row["beats_baseline"])
    oracle_row = show[show["policy"] == "oracle_cate"].iloc[0]
    assert bool(oracle_row["beats_baseline"]), "the oracle policy must beat treat_none"
    assert oracle_row["diff_ci_low"] > 0.0 and oracle_row["diff_ci_high"] > 0.0
    # Not merely close: compare_policies shares the resample draws with evaluate_policy, so
    # every reported quantity must agree to the last bit or the two tables in one report
    # would disagree about the same policy.
    for col in ("value", "se_analytic", "se_boot", "ci_low", "ci_high"):
        assert float(oracle_row[col]) == float(dr[col]), (col, oracle_row[col], dr[col])
    assert base_row["total_cost"] == 0.0
    assert np.isclose(
        show[show["policy"] == "treat_all_discount_10"].iloc[0]["total_cost"], N * ARM_COSTS[1]
    )
    # The paired interval is tighter than the naive difference of two marginal intervals.
    marginal_width = (oracle_row["ci_high"] - oracle_row["ci_low"]) + (
        base_row["ci_high"] - base_row["ci_low"]
    )
    paired_width = oracle_row["diff_ci_high"] - oracle_row["diff_ci_low"]
    assert paired_width < marginal_width, (paired_width, marginal_width)

    # The paired interval must cover the *known* true difference in policy value, and a
    # policy that is byte-for-byte the baseline must produce an exact zero -- never a
    # spurious win. An estimator that counts exact zeros as wins is worse than useless.
    true_diff = truth - float(mu_true[:, 0].mean())
    assert oracle_row["diff_ci_low"] < true_diff < oracle_row["diff_ci_high"], (
        true_diff, oracle_row["diff_ci_low"], oracle_row["diff_ci_high"])
    null_cmp = compare_policies({"treat_none": policies["treat_none"],
                                 "same_as_baseline": policies["treat_none"].copy()},
                                panel, e, "value_h", mu_hat=mu_hat, n_boot=200,
                                random_state=SEED, baseline="treat_none")
    null_row = null_cmp[(null_cmp.policy == "same_as_baseline") & (null_cmp.method == "dr")].iloc[0]
    assert null_row["diff_vs_baseline"] == 0.0
    assert null_row["diff_ci_low"] == 0.0 and null_row["diff_ci_high"] == 0.0
    assert not bool(null_row["beats_baseline"]), "an exactly-zero difference is not a win"
    print(f"paired diff CI width {paired_width:.3f} vs sum of marginal CI widths {marginal_width:.3f} "
          "-- non-overlapping-CI tests are conservative and would discard real wins")
    print()

    # ------------------------------------------------------------- oracle_gap -----------
    SCALE = 8.0
    tau_value = (mu_true - mu_true[:, [0]]) * SCALE
    net_true = tau_value - np.asarray(ARM_COSTS[:K])[None, :]
    net_true[:, 0] = 0.0
    gt = pd.DataFrame({f"gt_tau_value_{a}": tau_value[:, a] for a in range(1, K)})
    gt["gt_value_0"] = 0.0
    gt["gt_best_arm"] = np.where(net_true.max(axis=1) > 0, net_true.argmax(axis=1), 0)
    gt["gt_oracle_net_value"] = np.maximum(net_true.max(axis=1), 0.0)

    best_policy = gt["gt_best_arm"].to_numpy("int64")
    gap_best = oracle_gap(best_policy, gt, None)
    gap_rand = oracle_gap(policies["random"], gt, None)
    gap_none = oracle_gap(policies["treat_none"], gt, None)

    print("oracle_gap (true effects, ARM_COSTS charged in full)")
    print(f"{'assignment':<18}{'n_treated':>11}{'realised':>13}{'oracle':>13}{'%oracle':>10}"
          f"{'regret_mean':>13}{'%match_best':>13}")
    print("-" * 96)
    for label, g in (("gt_best_arm", gap_best), ("random", gap_rand), ("treat_none", gap_none)):
        print(f"{label:<18}{g['n_treated']:>11,}{g['realised_net_value']:>13,.0f}"
              f"{g['oracle_net_value']:>13,.0f}{g['pct_of_oracle_captured']:>10.1%}"
              f"{g['regret_mean']:>13.2f}{g['pct_of_rows_matching_gt_best_arm']:>13.1%}")
    print("-" * 96)

    # The oracle must genuinely decline to treat some customers, otherwise the "floored at
    # zero / do nothing" branch of the objective is never exercised.
    assert 0 < gap_best["n_treated"] < N, gap_best["n_treated"]
    assert gap_best["available"] and np.isclose(gap_best["pct_of_oracle_captured"], 1.0)
    assert abs(gap_best["regret_total"]) < 1e-8 and gap_best["pct_of_rows_matching_gt_best_arm"] == 1.0
    assert gap_rand["pct_of_oracle_captured"] < gap_best["pct_of_oracle_captured"]
    assert gap_rand["regret_total"] > 0 and gap_none["realised_net_value"] == 0.0
    assert gap_none["pct_of_oracle_captured"] == 0.0
    assert gap_best["oracle_source"] == "gt_oracle_net_value"

    # The three provenance branches must agree: supplied gt_oracle_net_value, the oracle
    # recomputed from gt_tau_value_*, and the oracle reconstructed from absolute
    # gt_value_* columns are three routes to one number. If they disagree, the pipeline's
    # oracle depends on which columns dgp.py happened to write.
    gap_recomputed = oracle_gap(best_policy, gt.drop(columns=["gt_oracle_net_value"]), None)
    assert gap_recomputed["oracle_source"] == "recomputed"
    assert np.isclose(gap_recomputed["oracle_net_value"], gap_best["oracle_net_value"])
    gt_abs = pd.DataFrame({f"gt_value_{a}": tau_value[:, a] for a in range(K)})
    gap_abs = oracle_gap(best_policy, gt_abs, None)
    assert gap_abs["available"] and gap_abs["best_arm_source"] == "recomputed"
    assert np.isclose(gap_abs["oracle_net_value"], gap_best["oracle_net_value"])
    assert np.isclose(gap_abs["realised_net_value"], gap_best["realised_net_value"])

    # Do-nothing is pinned to exactly zero net value, whatever the cost vector says, and
    # the oracle is floored at zero, so regret can never be negative.
    assert gap_none["realised_net_value"] == 0.0 and gap_none["total_cost"] == 0.0
    for g in (gap_best, gap_rand, gap_none):
        assert g["regret_total"] >= -1e-9 and g["oracle_net_value"] >= g["realised_net_value"] - 1e-9

    # use_redemption is an accounting switch on the *policy's* cost only; it must lower the
    # bill and it must not touch the oracle it is graded against.
    class _Econ:
        arm_costs = tuple(ARM_COSTS[:K])
        offer_redemption_rate = (0.0, 0.62, 0.71, 0.48)
    gap_full = oracle_gap(best_policy, gt, _Econ())
    gap_red = oracle_gap(best_policy, gt, _Econ(), use_redemption=True)
    assert np.isclose(gap_full["total_cost"], gap_best["total_cost"])
    assert gap_red["total_cost"] < gap_full["total_cost"]
    assert np.isclose(gap_red["oracle_net_value"], gap_full["oracle_net_value"]), (
        "the oracle must not be re-priced by the evaluated policy's cost convention")
    # A mapping econ honours the same switch as an object econ.
    gap_map = oracle_gap(best_policy, gt, {"arm_costs": list(_Econ.arm_costs),
                                           "offer_redemption_rate": list(_Econ.offer_redemption_rate)},
                         use_redemption=True)
    assert np.isclose(gap_map["total_cost"], gap_red["total_cost"])
    print(f"cost conventions: full {gap_full['total_cost']:,.0f} vs redemption-weighted "
          f"{gap_red['total_cost']:,.0f} -- the oracle is unchanged at "
          f"{gap_full['oracle_net_value']:,.0f} either way")

    # Real data has no ground truth: degrade to NaNs with available=False, never raise.
    gap_missing = oracle_gap(best_policy, gt[["gt_value_0"]], None)
    assert not gap_missing["available"] and np.isnan(gap_missing["pct_of_oracle_captured"])
    assert gap_missing["reason"] and set(gap_best) == set(gap_missing)
    print(f"no ground-truth columns -> available={gap_missing['available']}, "
          f"reason={gap_missing['reason']!r} (NaNs, not an exception)")

    elapsed = time.perf_counter() - t_start
    print()
    print(f"naive bias {naive - truth:+.3f} ({naive_z:.0f} SE) vs DR bias {dr['value'] - truth:+.3f} "
          f"({z_scores['dr']:.1f} SE) -- off-policy evaluation is the difference between a "
          "policy that looks good and one that is good")
    print(f"policy_eval.py OK  ({elapsed:.1f}s, {len(res) + len(cmp_df)} estimates, "
          f"n_boot=500 on influence-function values only)")
