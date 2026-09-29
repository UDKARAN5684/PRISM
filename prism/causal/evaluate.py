"""Evaluating a causal model when the target is never observed.

The problem
-----------
Supervised learning has an easy answer to "is this model any good?": hold out rows, compare
``y_hat`` with ``y``.  Causal inference has no such luxury.  The estimand is

    tau_i = Y_i(1) - Y_i(0)

and exactly one of those two potential outcomes is ever realised, so the label for the thing
being predicted **does not exist in any dataset, ever**.  Everything in this module is a way of
answering the question anyway, from data in which the answer is missing by construction.

Four families, and what each one can and cannot see
---------------------------------------------------
============================  ==============================  ====================================
family                        what it measures                blind spot
============================  ==============================  ====================================
Qini / AUUC / TOC / deciles   *ranking* quality               calibration; confounded arm mixes
GATES                         group effects **with CIs**      the shape inside a group
BLP calibration               the *scale* of the spread       which particular rows are wrong
Policy value                  currency, with a CI             needs overlap to be believable
============================  ==============================  ====================================

They are complementary, and the module is built so that the four disagree loudly when a model is
bad in only one of the four ways.  The classic trap -- a score that has merely rediscovered the
*level* of the outcome rather than the *effect* of the offer -- moves the Qini coefficient a long
way from zero in whichever direction the confounding happens to run, and is caught immediately by
families 2 and 3.  See :func:`gates`, and the smoke test that measures it.

And, uniquely in PRISM: ground truth
------------------------------------
The simulator in :mod:`prism.data.dgp` computes ``tau_i`` analytically, so :func:`pehe` and
:func:`eps_ate` can score an estimator against the thing no real project can observe.  That is
the point of building a DGP instead of downloading a CSV: the estimator is *proven* correct on
data where the answer is known before it is trusted on data where it is not.

Conventions used throughout
---------------------------
``y``
    Outcome, float, **higher is better**.  Continuous or binary; nothing here assumes binary.
``w``
    Arm index, int, ``0`` = control.  The classic Qini geometry is two-armed, so functions that
    need a two-arm contrast binarise to ``(w > 0)`` and **log that they did it** (never
    silently).  Per-arm numbers come from one call per arm on the rows with ``w in {0, a}``.
``cate``
    Predicted effect.  ``(n,)`` or ``(n, n_arms - 1)`` as returned by every PRISM CATE learner.
    A 2-D input is reduced to a 1-D *targeting score* by taking the best arm's effect (``max``
    across columns), which is the quantity an allocator actually ranks on; pass ``arm=a`` to
    score one specific arm's column instead.
``propensity``
    ``(n, n_arms)`` generalised propensity from :class:`prism.models.propensity.PropensityModel`.
    A 1-D vector is accepted where it is unambiguous; each function documents its own reading.

Two orderings, on purpose
-------------------------
:func:`uplift_by_decile` numbers **decile 1 = highest** predicted effect (so a good model's
``observed_uplift`` *decreases* down the table), because that is the convention the retention
report and ``prism.pipelines.figures.plot_uplift_deciles`` use.  :func:`gates` numbers
**group 1 = lowest** predicted effect (so a good model's ``gate`` *increases*), which is the
Chernozhukov-Demirer-Duflo-Fernandez-Val convention and the one ``plot_gates`` draws.  Both are
stated again in their own docstrings, because getting this backwards silently inverts a
conclusion.

Determinism
-----------
Every ranking uses a **stable** descending sort, so ties are broken by original row order and two
runs on the same frame give bit-identical curves.  That does mean a tie-heavy score (a shallow
tree, say) inherits the row order of the input; if that matters, average over permutations.  The
only stochastic component in the module is the bootstrap in :func:`policy_value`, which is seeded
through :func:`prism.utils.seeds.as_rng`.

Performance
-----------
Everything here is O(n log n) vectorised numpy on the decision-point sample (``n <= ~120k`` per
`SPEC_PERF.md` section 1).  Per `SPEC_PERF.md` section 8 the bootstrap resamples **per-row
influence contributions**, never a model, and every function that thins or truncates its output
logs exactly what it dropped.

References
----------
Radcliffe, N. & Surry, P. (2011). *Real-World Uplift Modelling with Significance-Based Uplift Trees.*
Gutierrez, P. & Gerardy, J-Y. (2017). *Causal inference and uplift modelling: a review.* PMLR.
Chernozhukov, V., Demirer, M., Duflo, E. & Fernandez-Val, I. (2018). *Generic machine learning
inference on heterogeneous treatment effects in randomized experiments.* NBER WP 24678.
Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E. & Wager, S. (2021). *Evaluating treatment
prioritization rules via rank-weighted average treatment effects.* (TOC / AUTOC.)
Dudik, M., Langford, J. & Li, L. (2011). *Doubly robust policy evaluation and learning.* ICML.
Hirano, K. & Imbens, G. (2001). *Estimation of causal effects using propensity score weighting.*
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from prism.data.schema import ARM_NAMES
from prism.utils.logging import get_logger
from prism.utils.seeds import as_rng

__all__ = [
    "qini_curve",
    "qini_score",
    "auuc",
    "uplift_at_k",
    "uplift_by_decile",
    "pehe",
    "eps_ate",
    "policy_value",
    "policy_risk",
    "gates",
    "blp_calibration",
    "targeting_operator_characteristic",
    "compare_learners",
    "QINI_CURVE_COLUMNS",
    "DECILE_COLUMNS",
    "GATES_COLUMNS",
    "BLP_COLUMNS",
    "TOC_COLUMNS",
    "LEADERBOARD_COLUMNS",
    "POLICY_VALUE_METHODS",
    "DR_FALLBACK_METHOD",
]

LOG = get_logger("causal.evaluate")

#: Guard applied to any denominator that is a probability.
_PROB_EPS: float = 1e-6
#: Guard applied to any other denominator.
_EPS: float = 1e-12

#: Trapezoid rule under whichever name this numpy exposes (``trapz`` was renamed in numpy 2).
_trapz = getattr(np, "trapezoid", np.trapz)

#: Estimators offered by :func:`policy_value`, in increasing order of sophistication.
POLICY_VALUE_METHODS: tuple[str, str, str] = ("ipw", "snipw", "dr")

#: The ``method`` string returned when ``method="dr"`` is asked for without an outcome model.
#: A silent downgrade would let a reader believe a doubly-robust number was produced when it was
#: not, so the degradation is written into the result itself.
DR_FALLBACK_METHOD: str = "dr[fallback=snipw]"

#: Columns of :func:`qini_curve`, in order. The first five are the contract; the rest are the raw
#: cumulative counts, kept so a reader can audit the arithmetic without re-deriving it.
QINI_CURVE_COLUMNS: tuple[str, ...] = (
    "frac_targeted",
    "n_targeted",
    "incremental",
    "random_incremental",
    "qini",
    "n_treated",
    "n_control",
    "cum_y_treated",
    "cum_y_control",
)

#: Columns of :func:`uplift_by_decile`, in order.
DECILE_COLUMNS: tuple[str, ...] = (
    "decile",
    "n",
    "n_treated",
    "n_control",
    "mean_y_treated",
    "mean_y_control",
    "observed_uplift",
    "predicted_uplift",
    "se",
    "ci_low",
    "ci_high",
)

#: Columns of :func:`gates`, in order.
GATES_COLUMNS: tuple[str, ...] = (
    "group",
    "n",
    "mean_predicted_cate",
    "gate",
    "se",
    "ci_low",
    "ci_high",
    "p_value",
    "top_minus_bottom",
    "top_minus_bottom_se",
    "top_minus_bottom_p",
)

#: Columns of :func:`blp_calibration`, in order.
BLP_COLUMNS: tuple[str, ...] = (
    "term",
    "parameter",
    "coefficient",
    "se",
    "t",
    "p_value",
    "ci_low",
    "ci_high",
)

#: Columns of :func:`targeting_operator_characteristic`, in order.
TOC_COLUMNS: tuple[str, ...] = (
    "frac_targeted",
    "n_targeted",
    "n_treated",
    "n_control",
    "ate_top",
    "ate_overall",
    "toc",
    "se_ate_top",
)

#: Columns of :func:`compare_learners`, in order. The schema is *stable*: a metric that could not
#: be computed from the inputs given is ``NaN``, never a missing column, so
#: ``artifacts/reports/leaderboard.csv`` keeps the same header on every run.
LEADERBOARD_COLUMNS: tuple[str, ...] = (
    "learner",
    "rank",
    "policy_value",
    "policy_value_se",
    "qini",
    "auuc",
    "uplift_at_20pct",
    "pehe",
    "eps_ate",
    "spearman_true",
    "blp_slope",
    "gates_p",
    "n",
    "mean_cate",
    "sd_cate",
)


# ======================================================================================
# validation and shared primitives
# ======================================================================================
def _as_float_1d(a: Any, name: str) -> np.ndarray:
    """Coerce ``a`` to a finite 1-D float64 array or raise with a precise message."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (or (n, 1)); got shape {np.shape(a)}")
    if arr.size and not np.all(np.isfinite(arr)):
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{name} contains {n_bad} non-finite value(s); clean or drop them first")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _binarise_arm(
    w: Any, *, context: str, require_both: bool = True
) -> tuple[np.ndarray, np.ndarray, int]:
    """Map an arm-index vector onto the two-arm ``(w > 0)`` contrast uplift geometry needs.

    Parameters
    ----------
    w : array-like
        Arm indices, integer-valued, ``0`` = control.
    context : str
        Caller name, used in the log line.
    require_both : bool, default True
        Raise if either the treated or the control side is empty.

    Returns
    -------
    treated : numpy.ndarray
        ``(n,)`` float64 indicator, 1.0 where ``w > 0``.
    arms : numpy.ndarray
        ``(n,)`` int64 original arm indices.
    n_levels : int
        Number of distinct arms seen.

    Notes
    -----
    The classic Qini statistic is defined for one treatment against one control.  With ``K > 2``
    arms the collapse to "any offer versus none" is a real modelling choice -- it averages over a
    treatment mix the analyst did not pick -- so it is **logged every time** rather than performed
    quietly.
    """
    raw = np.asarray(w)
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw[:, 0]
    raw = raw.ravel()
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{context}: w must be numeric arm indices, got dtype {raw.dtype}")
    as_f = raw.astype(np.float64)
    if as_f.size and not np.all(np.isfinite(as_f)):
        raise ValueError(f"{context}: w contains non-finite values")
    arms = np.rint(as_f).astype(np.int64)
    if as_f.size and float(np.max(np.abs(as_f - arms))) > 1e-9:
        raise ValueError(f"{context}: w must hold integer arm indices (0 = control)")
    if arms.size and int(arms.min()) < 0:
        raise ValueError(f"{context}: arm indices must be >= 0; got min {int(arms.min())}")

    levels = np.unique(arms)
    n_levels = int(levels.size)
    if n_levels > 2:
        names = ", ".join(
            f"{int(a)}={ARM_NAMES[int(a)]}" if 0 <= int(a) < len(ARM_NAMES) else str(int(a))
            for a in levels
        )
        LOG.info(
            "%s: w has %d arm levels (%s); binarising to (w > 0) for the two-arm uplift geometry. "
            "Per-arm numbers need one call per arm on rows with w in {0, a}.",
            context,
            n_levels,
            names,
        )
    treated = (arms > 0).astype(np.float64)
    if require_both:
        n_t = float(treated.sum())
        if n_t == 0.0 or n_t == float(treated.size):
            raise ValueError(
                f"{context}: need both treated and control rows; got {int(n_t)} treated out of "
                f"{treated.size}"
            )
    return treated, arms, n_levels


def _score_vector(cate: Any, arm: int | None, *, context: str) -> np.ndarray:
    """Reduce a CATE array to the 1-D targeting score a ranking metric consumes.

    Parameters
    ----------
    cate : array-like
        ``(n,)``, ``(n, 1)`` or ``(n, n_arms - 1)``. Column ``j`` holds arm ``j + 1``.
    arm : int or None
        Score arm ``arm`` only (1-based, matching the PRISM arm index).  ``None`` with a
        multi-column input takes ``max`` across arms -- the value of the *best available offer*,
        which is what a budget allocator ranks on.
    context : str
        Caller name, used in the log line.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` float64.
    """
    c = np.asarray(cate, dtype=np.float64)
    if c.ndim == 1:
        if arm is not None and int(arm) != 1:
            raise ValueError(f"{context}: cate is 1-D so only arm=1 is meaningful; got arm={arm}")
        s = c
    elif c.ndim == 2:
        if c.shape[1] == 0:
            raise ValueError(f"{context}: cate has zero columns")
        if arm is not None:
            j = int(arm) - 1
            if not 0 <= j < c.shape[1]:
                raise ValueError(
                    f"{context}: arm={arm} is out of range for a cate with {c.shape[1]} "
                    f"non-control column(s) (valid arms are 1..{c.shape[1]})"
                )
            s = c[:, j]
        elif c.shape[1] == 1:
            s = c[:, 0]
        else:
            s = c.max(axis=1)
            LOG.info(
                "%s: reduced a (%d, %d) CATE to a 1-D targeting score with max across arms "
                "(the best-offer value an allocator ranks on); pass arm=a to score one arm.",
                context,
                c.shape[0],
                c.shape[1],
            )
    else:
        raise ValueError(f"{context}: cate must be 1-D or 2-D; got shape {c.shape}")
    return _as_float_1d(s, f"{context}: cate")


def _prepare(
    y: Any, w: Any, cate: Any, arm: int | None, context: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and align ``(y, w, cate)`` for a two-arm ranking metric."""
    y_arr = _as_float_1d(y, f"{context}: y")
    treated, _, _ = _binarise_arm(w, context=context)
    score = _score_vector(cate, arm, context=context)
    n = y_arr.size
    if n == 0:
        raise ValueError(f"{context}: empty input")
    if treated.size != n or score.size != n:
        raise ValueError(
            f"{context}: length mismatch -- y={n}, w={treated.size}, cate={score.size}"
        )
    return y_arr, treated, score


def _check_alpha(alpha: float, context: str) -> float:
    """Validate a two-sided confidence level, so a bad one raises instead of returning ``nan``."""
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"{context}: alpha must lie strictly in (0, 1); got {alpha}")
    return a


def _check_clip(clip: tuple[float, float], context: str) -> tuple[float, float]:
    """Validate a propensity clip range."""
    low, high = float(clip[0]), float(clip[1])
    if not 0.0 < low < high < 1.0:
        raise ValueError(f"{context}: clip must satisfy 0 < low < high < 1; got {clip}")
    return low, high


def _rank_desc(score: np.ndarray) -> np.ndarray:
    """Descending order of ``score`` with a stable (original row order) tie-break."""
    return np.argsort(-score, kind="stable")


def _cumulative(
    y: np.ndarray, treated: np.ndarray, order: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prefix sums along a ranking, with a leading zero row for the empty prefix.

    Returns
    -------
    tuple of numpy.ndarray
        ``(n_t, n_c, y_t, y_c)``, each of length ``n + 1``; element ``k`` describes the top-``k``
        prefix of the ranking.
    """
    ys = y[order]
    ts = treated[order]
    zero = np.zeros(1, dtype=np.float64)
    n_t = np.concatenate((zero, np.cumsum(ts)))
    n_c = np.concatenate((zero, np.cumsum(1.0 - ts)))
    y_t = np.concatenate((zero, np.cumsum(ys * ts)))
    y_c = np.concatenate((zero, np.cumsum(ys * (1.0 - ts))))
    return n_t, n_c, y_t, y_c


def _qini_values(
    n_t: np.ndarray, n_c: np.ndarray, y_t: np.ndarray, y_c: np.ndarray
) -> np.ndarray:
    """Qini value ``Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)``, zero where ``N_c(k) == 0``.

    The head of a ranking can easily hold treated rows only; the ratio is then undefined and the
    convention (Radcliffe & Surry) is ``Q = 0`` there.  It is a *convention*, not a measurement:
    the treated outcome in that stretch is discarded because there is no control group to net it
    against.  A curve with a long flat zero head is telling you the ranking is correlated with
    assignment, which is itself worth knowing.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_c = np.where(n_c > 0.0, n_c, 1.0)
        q = y_t - y_c * (n_t / safe_c)
    return np.where(n_c > 0.0, q, 0.0)


def _uplift_values(
    n_t: np.ndarray, n_c: np.ndarray, y_t: np.ndarray, y_c: np.ndarray
) -> np.ndarray:
    """Uplift-curve value ``U(k) = (Y_t(k)/N_t(k) - Y_c(k)/N_c(k)) * k``, zero where undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        m_t = np.where(n_t > 0.0, y_t / np.where(n_t > 0.0, n_t, 1.0), 0.0)
        m_c = np.where(n_c > 0.0, y_c / np.where(n_c > 0.0, n_c, 1.0), 0.0)
    ok = (n_t > 0.0) & (n_c > 0.0)
    return np.where(ok, (m_t - m_c) * (n_t + n_c), 0.0)


def _oracle_order(y: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """The reference ("perfect model") ranking used to normalise :func:`qini_score` / :func:`auuc`.

    Construction
    ------------
    Within the treated arm order by **descending** ``y``; within the control arm order by
    **ascending** ``y``; then interleave the two lists *in proportion to the arm shares* by giving
    row ``j`` of an arm of size ``m`` the position ``(j + 0.5) / m`` and sorting on that.

    Why this, and not "all treated responders first"
    ------------------------------------------------
    Radcliffe's classical perfect curve (treated responders, then neutrals, then control
    responders) is defined for a **binary** outcome.  With a continuous, strictly positive ``y``
    it degenerates: every treated row outranks every control row, ``N_c(k) = 0`` over the whole
    first stretch of the curve, :func:`_qini_values` returns zero there by convention, and the
    "perfect" area becomes an artefact of that convention rather than a bound.

    Proportional interleaving pins the treated/control mix at the global ratio for every prefix.
    Holding that mix fixed, the prefix sums are extremal -- ``Y_t(k)`` as large and ``Y_c(k)`` as
    small as any ordering can make them -- so the curve is the best attainable *at the population
    arm mix*.

    Honest caveat
    -------------
    It is an upper **reference**, not a supremum over all orderings: a ranking that also skews the
    arm mix can in principle exceed it, which shows up as a normalised score above 1.  That is
    diagnostic in itself -- a score above 1 means the ranking is correlated with *assignment*, not
    only with effect, and the Qini count correction has not fully absorbed it.
    """
    idx_t = np.flatnonzero(treated > 0.5)
    idx_c = np.flatnonzero(treated <= 0.5)
    ord_t = idx_t[np.argsort(-y[idx_t], kind="stable")]
    ord_c = idx_c[np.argsort(y[idx_c], kind="stable")]
    pos_t = (np.arange(ord_t.size, dtype=np.float64) + 0.5) / max(ord_t.size, 1)
    pos_c = (np.arange(ord_c.size, dtype=np.float64) + 0.5) / max(ord_c.size, 1)
    merged = np.concatenate((ord_t, ord_c))
    positions = np.concatenate((pos_t, pos_c))
    return merged[np.argsort(positions, kind="stable")]


def _area_between(frac: np.ndarray, curve: np.ndarray) -> float:
    """Trapezoidal area between ``curve`` and the chord from ``(0, 0)`` to ``(1, curve[-1])``."""
    return float(_trapz(curve - curve[-1] * frac, frac))


def _thin(n_rows: int, max_points: int | None, context: str) -> np.ndarray | None:
    """Evenly spaced row selector that always keeps both endpoints, or ``None`` for "keep all"."""
    if max_points is None or n_rows <= int(max_points):
        return None
    keep = np.unique(np.linspace(0, n_rows - 1, int(max_points)).round().astype(np.int64))
    LOG.info(
        "%s: thinned the returned curve from %d to %d points (max_points=%d); scalar summaries "
        "are computed on the full curve before thinning.",
        context,
        n_rows,
        int(keep.size),
        int(max_points),
    )
    return keep


def _equal_bins(n: int, n_bins: int, context: str) -> list[np.ndarray]:
    """Split ``0..n-1`` into ``n_bins`` contiguous, near-equal blocks.

    ``np.array_split`` on the *ranked positions* is used rather than ``pd.qcut`` on the score:
    quantile cutting collapses bins when the score has heavy ties (a shallow tree predicts a
    handful of distinct values), which silently changes the number of groups a caller asked for.
    Equal-count blocks always return exactly ``n_bins`` groups; the price is that a tie can
    straddle a boundary, which is resolved by the stable sort and is therefore deterministic.
    """
    k = int(n_bins)
    if k < 2:
        raise ValueError(f"{context}: need at least 2 bins, got {n_bins}")
    if k > n:
        LOG.info("%s: reduced n_bins from %d to %d (only %d rows available).", context, k, n, n)
        k = n
    return np.array_split(np.arange(n, dtype=np.int64), k)


def _welch_se(y: np.ndarray, treated: np.ndarray, idx: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return ``(n_t, n_c, mean_t, mean_c, se)`` for the difference in means inside ``idx``."""
    sel = treated[idx] > 0.5
    y_t = y[idx][sel]
    y_c = y[idx][~sel]
    n_t, n_c = float(y_t.size), float(y_c.size)
    mean_t = float(y_t.mean()) if n_t else np.nan
    mean_c = float(y_c.mean()) if n_c else np.nan
    var_t = float(y_t.var(ddof=1)) if n_t > 1 else np.nan
    var_c = float(y_c.var(ddof=1)) if n_c > 1 else np.nan
    if np.isnan(var_t) or np.isnan(var_c):
        se = np.nan
    else:
        se = float(np.sqrt(var_t / max(n_t, 1.0) + var_c / max(n_c, 1.0)))
    return n_t, n_c, mean_t, mean_c, se


def _binary_propensity(
    propensity: Any,
    treated: np.ndarray,
    *,
    clip: tuple[float, float],
    context: str,
) -> np.ndarray:
    """Return ``P(W = 1 | X)`` for the binarised contrast, clipped and reported.

    Parameters
    ----------
    propensity : array-like
        ``(n, n_arms)`` generalised propensity -- ``P(W = 1 | X)`` is then ``1 - e_0(X)``, which
        is the right reading for "any offer versus none" -- or ``(n,)``, read directly as
        ``P(W = 1 | X)``.
    treated : numpy.ndarray
        ``(n,)`` binarised arm indicator, used only for the length check.
    clip : tuple of float
        ``(low, high)`` bounds. Clipping is logged when it binds, because a propensity that needs
        clipping is an overlap failure and the estimate downstream rests on it.
    context : str
        Caller name for messages.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` float64 in ``[low, high]``.
    """
    p = np.asarray(propensity, dtype=np.float64)
    n = treated.size
    if p.ndim == 2:
        if p.shape[0] != n:
            raise ValueError(f"{context}: propensity has {p.shape[0]} rows but w has {n}")
        if p.shape[1] < 2:
            raise ValueError(f"{context}: a 2-D propensity needs at least 2 arms")
        e1 = 1.0 - p[:, 0]
    elif p.ndim == 1:
        if p.size != n:
            raise ValueError(f"{context}: propensity has {p.size} entries but w has {n}")
        e1 = p.copy()
    else:
        raise ValueError(f"{context}: propensity must be (n,) or (n, n_arms); got {p.shape}")

    if not np.all(np.isfinite(e1)):
        raise ValueError(f"{context}: propensity contains non-finite values")
    e_min, e_max = float(e1.min()), float(e1.max())
    if e_min < 0.0 or e_max > 1.0:
        raise ValueError(
            f"{context}: propensity must be a probability in [0, 1]; got the range "
            f"[{e_min:.6g}, {e_max:.6g}]"
        )
    # Calibration check. Any propensity worth using satisfies E[e(X)] = P(W = 1), so a mean that
    # is nowhere near the realised treated share means the wrong quantity was passed. The way
    # that happens in practice is a *single arm's* column of an (n, n_arms) generalised
    # propensity being handed to a function whose contrast is "any offer versus none": with four
    # arms P(W = 1) is roughly a third of P(W > 0), the Hajek weights are then wrong by that
    # factor, and nothing else in the call would ever complain.
    share = float(treated.mean()) if n else float("nan")
    mean_e = float(e1.mean())
    tol = max(0.05, 3.0 * float(np.sqrt(max(share * (1.0 - share), 0.0) / max(n, 1))))
    if n and np.isfinite(share) and abs(mean_e - share) > tol:
        LOG.warning(
            "%s: the propensity averages %.3f but %.3f of the rows are treated (tolerance "
            "%.3f). A calibrated propensity has mean equal to the treated share, so this is "
            "probably the wrong vector -- most often one arm's column of an (n, n_arms) "
            "generalised propensity where P(W > 0) = 1 - e_0(X) was meant. Pass the full "
            "(n, n_arms) matrix and it will be read correctly.",
            context,
            mean_e,
            share,
            tol,
        )
    low, high = float(clip[0]), float(clip[1])
    n_bind = int(((e1 < low) | (e1 > high)).sum())
    if n_bind:
        LOG.warning(
            "%s: clipped %d/%d treatment propensities into [%.3g, %.3g] -- that many rows have "
            "near-deterministic assignment, so the estimate there is extrapolation, not overlap.",
            context,
            n_bind,
            n,
            low,
            high,
        )
    return np.clip(e1, low, high)


def _aipw_scores(
    y: np.ndarray,
    treated: np.ndarray,
    e1: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> np.ndarray:
    """AIPW (doubly-robust) pseudo-outcome for the binary contrast.

    ESTIMATOR MATHS
    ---------------
    ::

        psi_i = mu1(X_i) - mu0(X_i)
                + W_i / e(X_i)         * (Y_i - mu1(X_i))
                - (1 - W_i) / (1 - e(X_i)) * (Y_i - mu0(X_i))

    ``E[psi | X] = tau(X)`` when **either** the outcome models ``mu_a`` or the propensity ``e``
    is correct, which is the double-robustness property.  ``psi`` is also the efficient influence
    function of the ATE, so ``sd(psi) / sqrt(n)`` is the semiparametrically efficient standard
    error -- that is why every interval in this module is built from ``psi`` rather than from a
    naive difference in means.
    """
    e1 = np.clip(e1, _PROB_EPS, 1.0 - _PROB_EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_t = treated / e1 * (y - mu1)
        corr_c = (1.0 - treated) / (1.0 - e1) * (y - mu0)
    return (mu1 - mu0) + corr_t - corr_c


def _hajek_means(
    y: np.ndarray, treated: np.ndarray, e1: np.ndarray, idx: np.ndarray
) -> tuple[float, float]:
    """Hajek (self-normalised IPW) estimates of ``E[Y(1)]`` and ``E[Y(0)]`` inside ``idx``.

    Used as the *group-constant outcome model* when no ``mu_hat`` is supplied.  The choice is not
    arbitrary: with these particular constants the AIPW score of :func:`_aipw_scores` satisfies
    ``sum_i (Y_i - mu1) * W_i / e_i = 0`` exactly over the block, so the mean of ``psi`` collapses
    algebraically to the Hajek difference while ``sd(psi)/sqrt(n)`` still delivers the correct
    linearised standard error.  The constant then acts as a variance-reducing control variate
    rather than as a bias correction, and consistency rests on the propensity alone.
    """
    ysub = y[idx]
    tsub = treated[idx]
    esub = np.clip(e1[idx], _PROB_EPS, 1.0 - _PROB_EPS)
    wt = tsub / esub
    wc = (1.0 - tsub) / (1.0 - esub)
    sum_t, sum_c = float(wt.sum()), float(wc.sum())
    mu1 = float((wt * ysub).sum() / sum_t) if sum_t > _EPS else float("nan")
    mu0 = float((wc * ysub).sum() / sum_c) if sum_c > _EPS else float("nan")
    return mu0, mu1


def _outcome_model_columns(
    mu_hat: Any, n: int, arm: int | None, context: str
) -> tuple[np.ndarray, np.ndarray]:
    """Split a user-supplied ``mu_hat`` into ``(mu0, mu1)`` for the binary contrast."""
    m = np.asarray(mu_hat, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(
            f"{context}: mu_hat must be (n, 2) as [control, treated] or (n, n_arms); got {m.shape}"
        )
    if m.shape[0] != n:
        raise ValueError(f"{context}: mu_hat has {m.shape[0]} rows but the data has {n}")
    if m.shape[1] == 2:
        return _as_float_1d(m[:, 0], "mu_hat[:, 0]"), _as_float_1d(m[:, 1], "mu_hat[:, 1]")
    if arm is None:
        raise ValueError(
            f"{context}: mu_hat has {m.shape[1]} arm columns, so the treated arm is ambiguous "
            "under the (w > 0) binarisation -- pass arm=a to pick one, or pass an (n, 2) "
            "[control, treated] matrix built for the contrast you mean."
        )
    j = int(arm)
    if not 1 <= j < m.shape[1]:
        raise ValueError(f"{context}: arm={arm} is out of range for mu_hat with {m.shape[1]} arms")
    return _as_float_1d(m[:, 0], "mu_hat[:, 0]"), _as_float_1d(m[:, j], f"mu_hat[:, {j}]")


# ======================================================================================
# 1. ranking metrics -- Qini, AUUC, uplift curves
# ======================================================================================
def qini_curve(
    y: np.ndarray,
    w: np.ndarray,
    cate: np.ndarray,
    treatment_cost: float = 0.0,
    *,
    arm: int | None = None,
    max_points: int | None = None,
) -> pd.DataFrame:
    """Cumulative incremental outcome as the ranking is walked from the top down.

    ESTIMATOR MATHS
    ---------------
    Sort the rows by ``cate`` descending.  For the prefix of size ``k`` let ``N_t(k)``, ``N_c(k)``
    be the treated and control counts inside it and ``Y_t(k)``, ``Y_c(k)`` the corresponding
    outcome sums.  The Qini value is::

        Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)                              (Radcliffe 2007)

    The factor ``N_t(k) / N_c(k)`` rescales the control sum to the treated head-count, so ``Q(k)``
    reads as "extra outcome units produced by treating the top ``k``, over and above what those
    same customers would have produced untreated".  It corrects for the treated/control **count**
    imbalance inside the prefix -- and only for that; see the warning below.

    When ``treatment_cost != 0`` the curve becomes a **net-value curve**::

        Q_net(k) = Q(k) - treatment_cost * N_t(k)

    i.e. every treated customer in the prefix is charged the marginal cost of the offer, so the
    peak of the curve is the profit-maximising campaign depth rather than the volume-maximising
    one.  Those two depths are rarely the same, and the difference is the whole argument for
    budget-constrained targeting.  ``treatment_cost`` is in the same units as ``y``.

    The **random line** is the chord from ``(0, 0)`` to ``(1, Q_net(n))``.  Random targeting of a
    fraction ``q`` earns ``q`` of the total incremental value and pays ``q`` of the total cost, so
    the chord is the right baseline for the cost-adjusted curve too.

    Parameters
    ----------
    y : numpy.ndarray
        ``(n,)`` outcome, higher is better.
    w : numpy.ndarray
        ``(n,)`` arm index, ``0`` = control.  More than two levels is binarised to ``(w > 0)``
        and logged -- the classic Qini geometry is two-armed.
    cate : numpy.ndarray
        ``(n,)`` or ``(n, n_arms - 1)`` predicted effects; a 2-D input is reduced with ``max``
        across arms unless ``arm`` is given.
    treatment_cost : float, default 0.0
        Marginal cost charged per treated customer in the prefix, in outcome units.
    arm : int, optional
        Score only this arm's column of a 2-D ``cate`` (1-based).
    max_points : int, optional
        Thin the returned frame to at most this many evenly spaced points (endpoints always
        kept).  Only affects the returned frame, never a computed scalar, and the thinning is
        logged.

    Returns
    -------
    pandas.DataFrame
        ``n + 1`` rows (the leading row is the empty prefix) with columns
        :data:`QINI_CURVE_COLUMNS`:

        ``frac_targeted``
            ``k / n``.
        ``n_targeted``
            ``k``.
        ``incremental``
            ``Q_net(k)`` -- the Qini value, net of ``treatment_cost``.
        ``random_incremental``
            The chord, i.e. what random targeting of the same depth would have earned.
        ``qini``
            ``incremental - random_incremental``, the vertical gap that :func:`qini_score`
            integrates.
        ``n_treated``, ``n_control``, ``cum_y_treated``, ``cum_y_control``
            The raw prefix sums, so the arithmetic above can be audited from the frame.

        ``frame.attrs`` carries ``treatment_cost``, ``n``, ``n_treated_total``,
        ``n_control_total`` and ``ate_overall``.

    Warnings
    --------
    The Qini correction adjusts for the **count** imbalance in the prefix, not for any difference
    in the *kind* of customer on each side of it.  Under confounded assignment the control rows in
    the head of the ranking are not exchangeable with the treated ones, and the curve then mixes
    real uplift with selection.  :func:`gates` and :func:`blp_calibration` use the propensity
    score and do not have this weakness; a big Qini with a flat GATES means the Qini is selection.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    >>> w = np.array([1, 1, 0, 0, 1, 0])
    >>> cate = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    >>> curve = qini_curve(y, w, cate)
    >>> list(curve.columns[:5])
    ['frac_targeted', 'n_targeted', 'incremental', 'random_incremental', 'qini']
    >>> float(curve["incremental"].iloc[0])
    0.0
    """
    y_arr, treated, score = _prepare(y, w, cate, arm, "qini_curve")
    n = y_arr.size
    cost = float(treatment_cost)

    order = _rank_desc(score)
    n_t, n_c, y_t, y_c = _cumulative(y_arr, treated, order)
    q_net = _qini_values(n_t, n_c, y_t, y_c) - cost * n_t
    frac = np.arange(n + 1, dtype=np.float64) / float(n)
    random_line = q_net[-1] * frac

    frame = pd.DataFrame(
        {
            "frac_targeted": frac,
            "n_targeted": np.arange(n + 1, dtype=np.int64),
            "incremental": q_net,
            "random_incremental": random_line,
            "qini": q_net - random_line,
            "n_treated": n_t.astype(np.int64),
            "n_control": n_c.astype(np.int64),
            "cum_y_treated": y_t,
            "cum_y_control": y_c,
        }
    )[list(QINI_CURVE_COLUMNS)]

    n_tt = float(treated.sum())
    n_cc = float(n - n_tt)
    ate = float(y_arr[treated > 0.5].mean() - y_arr[treated <= 0.5].mean())
    frame.attrs.update(
        {
            "treatment_cost": cost,
            "n": int(n),
            "n_treated_total": int(n_tt),
            "n_control_total": int(n_cc),
            "ate_overall": ate,
            "net_value_curve": cost != 0.0,
        }
    )
    keep = _thin(n + 1, max_points, "qini_curve")
    if keep is not None:
        attrs = dict(frame.attrs)
        frame = frame.iloc[keep].reset_index(drop=True)
        frame.attrs.update(attrs)
    return frame


def _curve_areas(
    y: np.ndarray,
    treated: np.ndarray,
    score: np.ndarray,
    *,
    kind: Literal["qini", "uplift"],
    treatment_cost: float,
) -> tuple[float, float, float]:
    """Return ``(area_model, area_perfect, endpoint)`` for one of the two curve families.

    ``area`` means *between the curve and its random chord* for ``kind="qini"`` and *under the
    curve* for ``kind="uplift"``; that difference is exactly the difference between the two public
    scores and is documented on each of them.
    """
    n = y.size
    frac = np.arange(n + 1, dtype=np.float64) / float(n)
    values = _qini_values if kind == "qini" else _uplift_values

    def _one(order: np.ndarray) -> np.ndarray:
        n_t, n_c, y_t, y_c = _cumulative(y, treated, order)
        return values(n_t, n_c, y_t, y_c) - treatment_cost * n_t

    model = _one(_rank_desc(score))
    perfect = _one(_oracle_order(y, treated))
    if kind == "qini":
        return _area_between(frac, model), _area_between(frac, perfect), float(model[-1])
    return float(_trapz(model, frac)), float(_trapz(perfect, frac)), float(model[-1])


def qini_score(
    y: np.ndarray,
    w: np.ndarray,
    cate: np.ndarray,
    *,
    treatment_cost: float = 0.0,
    normalize: bool = True,
    arm: int | None = None,
) -> float:
    """Normalised area between the Qini curve and the random line -- the Qini coefficient.

    ESTIMATOR MATHS
    ---------------
    With ``Q(k)`` the Qini curve of :func:`qini_curve`, ``R(k) = Q(n) * k/n`` its random chord and
    ``q = k/n`` the targeting depth::

        A_model   = integral_0^1 [ Q(qn) - R(qn) ] dq            (trapezoid over all n+1 prefixes)
        A_perfect = the same area for the reference ranking of :func:`_oracle_order`
        qini_score = A_model / A_perfect                          (normalize=True, the default)

    THE NORMALISATION, EXPLICITLY
    -----------------------------
    ``A_model`` alone is in *outcome units times customers*: it grows with ``n`` and with the
    scale of ``y``, so it cannot be compared between a 5k-row experiment and a 100k-row one, or
    between a churn-flag outcome and a currency outcome.  Dividing by the area of the best
    attainable ranking on the *same rows* removes both, leaving a dimensionless number:

    * ``1``      -- ranks as well as the reference ordering can on this data;
    * ``0``      -- indistinguishable from random targeting;
    * ``< 0``    -- worse than random, i.e. the ranking is inverted (treating the top of the list
      destroys value: exactly what a model that has locked on to sleeping dogs looks like);
    * ``> 1``    -- see :func:`_oracle_order`; the ranking is exploiting the treated/control mix
      rather than the effect, which is a red flag, not a triumph.

    The reference ranking is built from the **realised** outcomes, so the denominator contains the
    outcome noise.  Two datasets with the same true uplift but different noise therefore get
    different denominators, and the comparability claim is "across datasets of broadly similar
    signal-to-noise", not "universally".  This is inherent to every perfect-curve normalisation in
    the uplift literature, and stating it is the difference between a metric and a number.

    Parameters
    ----------
    y, w, cate
        As in :func:`qini_curve`.
    treatment_cost : float, keyword-only, default 0.0
        Charge the offer cost, turning the curve into a net-value curve.  Note the denominator is
        computed with the *same* cost, so the score stays on the same 0-1 reading.
    normalize : bool, keyword-only, default True
        ``False`` returns the raw ``A_model`` in outcome-units-times-customers.  Useful for
        summing across segments; useless for comparing across datasets.
    arm : int, optional
        Score only this arm's column of a 2-D ``cate``.

    Returns
    -------
    float
        The Qini coefficient (or the raw area when ``normalize=False``).

    Notes
    -----
    This measures **ranking**, nothing else.  A model can score 0.9 here and still claim effects
    three times too large; that is what :func:`blp_calibration` is for.
    """
    y_arr, treated, score = _prepare(y, w, cate, arm, "qini_score")
    area, area_perfect, _ = _curve_areas(
        y_arr, treated, score, kind="qini", treatment_cost=float(treatment_cost)
    )
    if not normalize:
        return float(area)
    # ``A_perfect >= 0`` is guaranteed, not hoped for: at the fixed proportional arm mix the
    # reference curve is (concave increasing treated sums) - (ratio x convex control sums) minus
    # a term linear in k, hence concave and through the origin, so it never falls below its own
    # chord.  The guard below therefore only ever fires on a degenerate (constant-y) input; it is
    # written as "not strictly positive" rather than "close to zero" so that a floating-point
    # sliver below zero can never flip the sign of the score.
    if not (area_perfect > _EPS):
        LOG.warning(
            "qini_score: the reference (perfect) curve has zero area, so there is no separable "
            "uplift signal in these outcomes to normalise against; returning 0.0. Check that y "
            "is not constant and that both arms are populated."
        )
        return 0.0
    return float(area / area_perfect)


def auuc(
    y: np.ndarray,
    w: np.ndarray,
    cate: np.ndarray,
    normalize: bool = True,
    *,
    treatment_cost: float = 0.0,
    arm: int | None = None,
) -> float:
    """Area under the uplift curve.

    ESTIMATOR MATHS
    ---------------
    The uplift curve is the *difference in means* form of the cumulative gain::

        U(k) = ( Y_t(k)/N_t(k) - Y_c(k)/N_c(k) ) * k          (0 where either arm is empty)
        AUUC_raw = integral_0^1 U(qn) dq

    and with ``normalize=True`` it is divided by the same integral taken over the reference
    ranking of :func:`_oracle_order`::

        auuc = AUUC_raw / AUUC_perfect

    Qini versus AUUC, since they are constantly confused
    ----------------------------------------------------
    Both walk the same ranking.  They differ in how they handle the arm imbalance and in what they
    are centred on:

    * :func:`qini_score` rescales the *control sum* by the count ratio and is measured **against
      the random line**, so a random ranking scores ``0``.
    * ``auuc`` compares *means*, which is more stable when the arm mix drifts along the ranking,
      and is measured **from zero**.  A random ranking therefore scores a positive number
      (roughly ``0.5 * ATE * n / AUUC_perfect``) whenever the ATE is positive.  A positive AUUC is
      not evidence of a good ranking; only a *comparison* between models on the same data is.

    Use ``qini_score`` for "is this better than not targeting"; use ``auuc`` as a
    mean-based cross-check that is less sensitive to an unbalanced head.

    When the normalisation is not available
    ---------------------------------------
    ``qini_score`` measures its area **against the chord**, and the reference curve is concave
    through the origin at a fixed arm mix, so ``A_perfect >= 0`` always: that denominator is
    safe.  ``auuc`` measures its area **from zero**, and no such guarantee exists.  When the
    overall effect is negative the reference area shrinks, crosses zero and goes negative, and a
    ratio taken across that crossing is worse than useless -- it inverts the sign and explodes
    the magnitude, so a treatment that is getting steadily *more* harmful can produce a large
    positive "score".  ``normalize=True`` therefore returns ``nan`` (with a warning) whenever the
    reference area is not strictly positive, rather than reporting the ratio.  Use
    ``normalize=False`` for the raw signed area, or :func:`qini_score`, in that regime.

    Parameters
    ----------
    y, w, cate
        As in :func:`qini_curve`.
    normalize : bool, default True
        Divide by the reference ranking's area.  ``False`` returns outcome-units-times-customers.
    treatment_cost : float, keyword-only, default 0.0
        Charge the offer cost per treated customer in the prefix, as in :func:`qini_curve`.
    arm : int, optional
        Score only this arm's column of a 2-D ``cate``.

    Returns
    -------
    float
        The (optionally normalised) area under the uplift curve.
    """
    y_arr, treated, score = _prepare(y, w, cate, arm, "auuc")
    area, area_perfect, _ = _curve_areas(
        y_arr, treated, score, kind="uplift", treatment_cost=float(treatment_cost)
    )
    if not normalize:
        return float(area)
    # The reference area is NOT guaranteed positive here, and that is the one trap in this
    # function -- see the "When the normalisation is not available" section of the docstring.
    if not (area_perfect > _EPS):
        LOG.warning(
            "auuc: the reference curve's area is %.6g, which is not positive, so the normalised "
            "AUUC is not defined (dividing by it would invert the sign and explode the "
            "magnitude). Returning nan. This happens when the overall effect is negative enough "
            "that even the best ranking cannot accumulate positive area under the uplift curve; "
            "use normalize=False for the raw area, or qini_score, whose reference area is "
            "provably non-negative.",
            area_perfect,
        )
        return float("nan")
    return float(area / area_perfect)


def uplift_at_k(
    y: np.ndarray,
    w: np.ndarray,
    cate: np.ndarray,
    k: float = 0.2,
    *,
    kind: Literal["uplift", "qini"] = "uplift",
    treatment_cost: float = 0.0,
    arm: int | None = None,
) -> float:
    """Estimated incremental outcome from treating the top ``k`` of the ranking.

    ESTIMATOR MATHS
    ---------------
    With ``m = ceil(k * n)`` the campaign size and the prefix sums of :func:`qini_curve`::

        kind="uplift"  ->  ( Y_t(m)/N_t(m) - Y_c(m)/N_c(m) ) * m  - treatment_cost * N_t(m)
        kind="qini"    ->    Y_t(m) - Y_c(m) * N_t(m)/N_c(m)      - treatment_cost * N_t(m)

    Both answer "how many extra outcome units does treating these ``m`` customers buy?", and both
    are in **outcome units**, not per customer -- divide by ``m`` for the per-customer figure.
    The ``uplift`` form estimates the effect from the difference in *means* and is the one to
    quote in a campaign plan; the ``qini`` form is the count-rescaled version, kept so the number
    matches the ``incremental`` column of :func:`qini_curve` exactly.

    Parameters
    ----------
    y, w, cate
        As in :func:`qini_curve`.
    k : float, default 0.2
        Targeting depth.  ``0 < k <= 1`` is a **fraction** of the population; ``k > 1`` is an
        **absolute head-count** (rounded, capped at ``n``).
    kind : {"uplift", "qini"}, keyword-only, default "uplift"
        Which of the two prefix statistics above to return.
    treatment_cost : float, keyword-only, default 0.0
        Charged per treated customer inside the prefix, making the result a net value.
    arm : int, optional
        Score only this arm's column of a 2-D ``cate``.

    Returns
    -------
    float
        Incremental outcome from the top-``k`` campaign.  ``nan`` if the prefix contains only one
        arm, which is a genuine "cannot be estimated" rather than a zero.
    """
    if kind not in {"uplift", "qini"}:
        raise ValueError(f"uplift_at_k: kind must be 'uplift' or 'qini'; got {kind!r}")
    y_arr, treated, score = _prepare(y, w, cate, arm, "uplift_at_k")
    n = y_arr.size
    kf = float(k)
    if kf <= 0:
        raise ValueError(f"uplift_at_k: k must be positive; got {k}")
    m = int(np.ceil(kf * n)) if kf <= 1.0 else int(round(kf))
    m = int(np.clip(m, 1, n))
    if kf > 1.0 and int(round(kf)) > n:
        LOG.info("uplift_at_k: requested head-count %d exceeds n=%d; capped at n.", int(round(kf)), n)

    order = _rank_desc(score)
    n_t, n_c, y_t, y_c = _cumulative(y_arr, treated, order)
    if n_t[m] == 0.0 or n_c[m] == 0.0:
        LOG.info(
            "uplift_at_k: the top %d rows contain %d treated and %d control, so the incremental "
            "outcome is not identified there; returning nan.",
            m,
            int(n_t[m]),
            int(n_c[m]),
        )
        return float("nan")
    values = _uplift_values if kind == "uplift" else _qini_values
    val = float(values(n_t[m : m + 1], n_c[m : m + 1], y_t[m : m + 1], y_c[m : m + 1])[0])
    return val - float(treatment_cost) * float(n_t[m])


def uplift_by_decile(
    y: np.ndarray,
    w: np.ndarray,
    cate: np.ndarray,
    n_bins: int = 10,
    *,
    arm: int | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Observed versus predicted uplift within equal-size bins of the predicted effect.

    This is the eyeball test.  A model with real signal produces ``observed_uplift`` that falls
    monotonically down the table, and ``predicted_uplift`` that tracks it rather than merely
    correlating with it: parallel-but-offset lines mean the ranking is right and the level is
    wrong, a fan shape means the spread is wrong.  Both are ordinary, both are invisible in a
    single Qini number, and :func:`blp_calibration` puts a standard error on the second one.

    ORDERING (read this)
    --------------------
    **Decile 1 is the highest** predicted effect, so a good model's ``observed_uplift``
    *decreases* down the frame.  This matches ``prism.pipelines.figures.plot_uplift_deciles``
    ("1 = highest") and the retention report.  :func:`gates` uses the opposite convention
    (group 1 = lowest, effects increase) to match the CDDF literature.

    ESTIMATOR MATHS
    ---------------
    Inside bin ``g``::

        observed_uplift_g = mean(Y | W = 1, g) - mean(Y | W = 0, g)
        se_g              = sqrt( s2_t/n_t + s2_c/n_c )               (Welch, unpooled)
        ci                = observed_uplift_g -/+ z_{1-alpha/2} * se_g

    The difference in means is **unadjusted**, so it is unbiased inside a bin only if assignment
    is (conditionally) random there.  On confounded data read the bin effects as descriptive and
    take the adjusted numbers from :func:`gates`.

    Parameters
    ----------
    y, w, cate
        As in :func:`qini_curve`.
    n_bins : int, default 10
        Number of equal-size bins.  Reduced to ``n`` with a log line if there are fewer rows.
    arm : int, optional
        Score only this arm's column of a 2-D ``cate``.
    alpha : float, keyword-only, default 0.05
        Two-sided level for ``ci_low`` / ``ci_high``.

    Returns
    -------
    pandas.DataFrame
        One row per bin with columns :data:`DECILE_COLUMNS`.  ``se`` is the standard error of
        ``observed_uplift`` (named ``se`` because that is what the plotting layer looks for), and
        is ``nan`` for a bin with fewer than two rows in either arm.  ``frame.attrs`` carries
        ``spearman_decile_uplift`` (rank correlation between bin index and observed uplift --
        strongly negative is good, given the ordering above), ``monotone_decreasing``,
        ``ate_overall`` and ``ordering``.
    """
    alpha = _check_alpha(alpha, "uplift_by_decile")
    y_arr, treated, score = _prepare(y, w, cate, arm, "uplift_by_decile")
    n = y_arr.size
    order = _rank_desc(score)
    blocks = _equal_bins(n, n_bins, "uplift_by_decile")
    z = float(stats.norm.ppf(1.0 - float(alpha) / 2.0))

    rows: list[dict[str, Any]] = []
    for g, block in enumerate(blocks, start=1):
        idx = order[block]
        n_t, n_c, mean_t, mean_c, se = _welch_se(y_arr, treated, idx)
        obs = mean_t - mean_c if (n_t and n_c) else np.nan
        rows.append(
            {
                "decile": g,
                "n": int(idx.size),
                "n_treated": int(n_t),
                "n_control": int(n_c),
                "mean_y_treated": mean_t,
                "mean_y_control": mean_c,
                "observed_uplift": obs,
                "predicted_uplift": float(score[idx].mean()),
                "se": se,
                "ci_low": obs - z * se if np.isfinite(se) else np.nan,
                "ci_high": obs + z * se if np.isfinite(se) else np.nan,
            }
        )

    frame = pd.DataFrame(rows)[list(DECILE_COLUMNS)]
    obs_vals = frame["observed_uplift"].to_numpy(dtype=float)
    finite = np.isfinite(obs_vals)
    if finite.sum() >= 3:
        rho = float(stats.spearmanr(frame["decile"].to_numpy()[finite], obs_vals[finite]).statistic)
    else:
        rho = float("nan")
    frame.attrs.update(
        {
            "ordering": "decile 1 = highest predicted effect; observed_uplift should DECREASE",
            "spearman_decile_uplift": rho,
            "monotone_decreasing": bool(finite.all() and np.all(np.diff(obs_vals) <= 0)),
            "ate_overall": float(
                y_arr[treated > 0.5].mean() - y_arr[treated <= 0.5].mean()
            ),
            "alpha": float(alpha),
            "adjusted": False,
        }
    )
    return frame


def targeting_operator_characteristic(
    y: np.ndarray,
    w: np.ndarray,
    cate: np.ndarray,
    *,
    n_points: int = 100,
    arm: int | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """The TOC curve: how much better is the treated-effect among the top ``q`` than overall?

    ESTIMATOR MATHS
    ---------------
    For a targeting depth ``q`` in ``(0, 1]``, with ``m = round(q n)`` and ``S(m)`` the top-``m``
    prefix of the ranking::

        ATE_top(q) = mean(Y | W = 1, S(m)) - mean(Y | W = 0, S(m))
        ATE_all    = mean(Y | W = 1)       - mean(Y | W = 0)
        TOC(q)     = ATE_top(q) - ATE_all
        AUTOC      = integral_0^1 TOC(q) dq

    TOC is the Qini curve's *per-customer* sibling (Yadlowsky et al. 2021): where the Qini curve
    accumulates total value and therefore always rises with ``q`` when the ATE is positive, the
    TOC divides by the head-count and answers the sharper question -- **is the top of my list
    actually more responsive than the average customer?**  A model with no heterogeneity gives a
    TOC that is flat at zero even though its Qini curve climbs steadily, which makes the TOC the
    better picture to put in front of someone deciding whether to target at all.

    ``TOC(1) = 0`` identically, since targeting everyone *is* the average.  ``AUTOC`` is the
    rank-weighted ATE and is positive exactly when the ranking concentrates effect at the top.

    Parameters
    ----------
    y, w, cate
        As in :func:`qini_curve`.
    n_points : int, keyword-only, default 100
        Number of depths on the grid ``q = 1/n_points, ..., 1``.  Capped at ``n``.
    arm : int, optional
        Score only this arm's column of a 2-D ``cate``.
    alpha : float, keyword-only, default 0.05
        Reserved for the interval reported in ``attrs``; the per-row ``se_ate_top`` is two-sided.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`TOC_COLUMNS`.  ``se_ate_top`` is the Welch standard error of ``ate_top``
        alone; the standard error of ``toc`` itself is **not** the naive combination, because the
        prefix sample is nested inside the full sample, so it is deliberately not reported.
        ``frame.attrs`` carries ``autoc``, ``ate_overall`` and ``n_undefined`` (depths at which
        one arm was empty, reported as ``nan``).
    """
    alpha = _check_alpha(alpha, "targeting_operator_characteristic")
    y_arr, treated, score = _prepare(y, w, cate, arm, "targeting_operator_characteristic")
    n = y_arr.size
    pts = int(np.clip(int(n_points), 1, n))
    if pts != int(n_points):
        LOG.info(
            "targeting_operator_characteristic: reduced the grid from %d to %d depths (n=%d).",
            int(n_points),
            pts,
            n,
        )
    order = _rank_desc(score)
    n_t, n_c, y_t, y_c = _cumulative(y_arr, treated, order)
    # Prefix sums of squares as well, so the Welch standard error at every depth is a vectorised
    # expression rather than a Python loop over the grid (SPEC_PERF section 9).
    zero = np.zeros(1, dtype=np.float64)
    sq = (y_arr * y_arr)[order]
    ts = treated[order]
    y2_t = np.concatenate((zero, np.cumsum(sq * ts)))
    y2_c = np.concatenate((zero, np.cumsum(sq * (1.0 - ts))))

    q_grid = (np.arange(1, pts + 1, dtype=np.float64)) / float(pts)
    m_grid = np.clip(np.round(q_grid * n).astype(np.int64), 1, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_t = np.where(n_t > 0, y_t / np.where(n_t > 0, n_t, 1.0), np.nan)
        mean_c = np.where(n_c > 0, y_c / np.where(n_c > 0, n_c, 1.0), np.nan)
    ate_top = np.where((n_t[m_grid] > 0) & (n_c[m_grid] > 0), mean_t[m_grid] - mean_c[m_grid], np.nan)
    ate_all = float(mean_t[n] - mean_c[n])

    with np.errstate(divide="ignore", invalid="ignore"):
        nt_g, nc_g = n_t[m_grid], n_c[m_grid]
        # unbiased variance of the mean: (sum y^2 - (sum y)^2 / k) / (k * (k - 1))
        var_mt = np.where(
            nt_g > 1, (y2_t[m_grid] - y_t[m_grid] ** 2 / np.maximum(nt_g, 1.0)) / (nt_g * (nt_g - 1.0)), np.nan
        )
        var_mc = np.where(
            nc_g > 1, (y2_c[m_grid] - y_c[m_grid] ** 2 / np.maximum(nc_g, 1.0)) / (nc_g * (nc_g - 1.0)), np.nan
        )
    se = np.sqrt(np.clip(var_mt + var_mc, 0.0, None))

    frame = pd.DataFrame(
        {
            "frac_targeted": m_grid.astype(np.float64) / float(n),
            "n_targeted": m_grid,
            "n_treated": n_t[m_grid].astype(np.int64),
            "n_control": n_c[m_grid].astype(np.int64),
            "ate_top": ate_top,
            "ate_overall": np.full(pts, ate_all),
            "toc": ate_top - ate_all,
            "se_ate_top": se,
        }
    )[list(TOC_COLUMNS)]

    valid = np.isfinite(frame["toc"].to_numpy(dtype=float))
    n_bad = int((~valid).sum())
    if n_bad:
        LOG.info(
            "targeting_operator_characteristic: %d/%d depths had an empty arm and are nan; AUTOC "
            "is integrated over the remaining %d.",
            n_bad,
            pts,
            int(valid.sum()),
        )
    autoc = (
        float(_trapz(frame["toc"].to_numpy(dtype=float)[valid], frame["frac_targeted"].to_numpy()[valid]))
        if valid.sum() >= 2
        else float("nan")
    )
    frame.attrs.update(
        {
            "autoc": autoc,
            "ate_overall": ate_all,
            "n_undefined": n_bad,
            "alpha": float(alpha),
        }
    )
    return frame


# ======================================================================================
# 2. ground-truth metrics -- only available against the simulator
# ======================================================================================
def _align_cate(true_cate: Any, est_cate: Any, context: str) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a pair of CATE arrays to the same shape, squeezing degenerate second axes."""
    a = np.asarray(true_cate, dtype=np.float64)
    b = np.asarray(est_cate, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if b.ndim == 2 and b.shape[1] == 1:
        b = b[:, 0]
    if a.ndim > 2 or b.ndim > 2:
        raise ValueError(f"{context}: inputs must be 1-D or 2-D; got {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise ValueError(
            f"{context}: shape mismatch -- true_cate {a.shape} vs est_cate {b.shape}. For a "
            "multi-arm model both must be (n, n_arms - 1) with arms in the same column order."
        )
    if a.size == 0:
        raise ValueError(f"{context}: empty input")
    for arr, name in ((a, "true_cate"), (b, "est_cate")):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{context}: {name} contains non-finite values")
    return a, b


def pehe(true_cate: np.ndarray, est_cate: np.ndarray) -> float:
    """Precision in Estimating Heterogeneous Effects -- the RMSE of the treatment effect.

    ESTIMATOR MATHS
    ---------------
    ::

        PEHE = sqrt( mean_i ( tau_i - tau_hat_i )^2 )

    This is the metric no real project can compute, because ``tau_i`` is never observed.  PRISM
    can, because :mod:`prism.data.dgp` derives ``tau_i`` analytically from the hazard path.  It is
    the strictest score in the module: unlike Qini it punishes a wrong *level* and a wrong
    *spread* as well as a wrong *order*, so a model can be the Qini champion and the PEHE loser.

    Shapes
    ------
    1-D ``(n,)`` for a single contrast, or 2-D ``(n, n_arms - 1)`` for a multi-arm learner.  A 2-D
    input **pools across arms**: the mean is taken over all ``n * (n_arms - 1)`` cells, so an arm
    that is estimated badly is diluted by arms that are estimated well.  For a per-arm breakdown
    call ``pehe(true[:, j], est[:, j])`` in a loop.  ``(n, 1)`` is squeezed to ``(n,)`` so the two
    conventions agree.

    Parameters
    ----------
    true_cate : numpy.ndarray
        Ground-truth effects, ``(n,)`` or ``(n, n_arms - 1)``.
    est_cate : numpy.ndarray
        Estimated effects, same shape.

    Returns
    -------
    float
        Root mean squared error, in outcome units.  Exactly ``0.0`` for identical inputs.

    Examples
    --------
    >>> import numpy as np
    >>> tau = np.array([1.0, -2.0, 0.5])
    >>> pehe(tau, tau)
    0.0
    >>> round(pehe(tau, tau + 1.0), 6)
    1.0
    """
    a, b = _align_cate(true_cate, est_cate, "pehe")
    diff = a - b
    return float(np.sqrt(np.mean(diff * diff)))


def eps_ate(
    true_cate: np.ndarray,
    est_cate: np.ndarray,
    *,
    reduce: Literal["pool", "mean_abs", "max"] = "pool",
) -> float:
    """Absolute error of the average treatment effect.

    ESTIMATOR MATHS
    ---------------
    ::

        eps_ATE = | mean_i(tau_i) - mean_i(tau_hat_i) |

    PEHE and eps_ATE fail in different ways and both are needed.  A model that is wildly wrong row
    by row but unbiased on average has a large PEHE and a tiny eps_ATE -- fine for sizing a budget,
    useless for choosing who to spend it on.  A model that is uniformly shifted has a small PEHE
    relative to its bias and a large eps_ATE -- good targeting, wrong ROI.

    Parameters
    ----------
    true_cate, est_cate : numpy.ndarray
        ``(n,)`` or ``(n, n_arms - 1)``, same shape.
    reduce : {"pool", "mean_abs", "max"}, keyword-only, default "pool"
        How a 2-D input is collapsed.  ``"pool"`` is the literal formula above over all cells and
        therefore lets an over-estimate on one arm **cancel** an under-estimate on another;
        ``"mean_abs"`` and ``"max"`` take the per-arm absolute errors first and do not cancel.
        Ignored for 1-D input, where all three agree.

    Returns
    -------
    float
        Absolute ATE error, in outcome units.
    """
    if reduce not in {"pool", "mean_abs", "max"}:
        raise ValueError(f"eps_ate: reduce must be 'pool', 'mean_abs' or 'max'; got {reduce!r}")
    a, b = _align_cate(true_cate, est_cate, "eps_ate")
    if a.ndim == 1 or reduce == "pool":
        return float(abs(a.mean() - b.mean()))
    per_arm = np.abs(a.mean(axis=0) - b.mean(axis=0))
    return float(per_arm.mean()) if reduce == "mean_abs" else float(per_arm.max())


def policy_risk(true_outcomes_by_arm: np.ndarray, policy_arm: np.ndarray) -> float:
    """Mean regret of a policy against the oracle that always picks the best arm.

    ESTIMATOR MATHS
    ---------------
    ::

        regret_i    = max_a Y_i(a) - Y_i(pi(X_i))
        policy_risk = mean_i regret_i

    Zero exactly for the oracle; positive for everything else; in outcome units, so it can be read
    as "currency left on the table per customer".  Like :func:`pehe` this needs the full
    counterfactual table and is therefore a simulator-only metric -- the real-data equivalent is
    :func:`policy_value`, which estimates one policy's value rather than the gap to the best.

    Note that it charges nothing for the offers: ``true_outcomes_by_arm`` should already be net of
    cost if the comparison is meant to be economic (``gt_value_a - ARM_COSTS[a]``).  Pass raw
    outcomes and the oracle will happily "choose" the most expensive arm for everyone.

    Parameters
    ----------
    true_outcomes_by_arm : numpy.ndarray
        ``(n, n_arms)`` potential outcomes, column ``a`` = ``Y_i(a)``, higher is better.
    policy_arm : numpy.ndarray
        ``(n,)`` integer arm chosen by the policy.

    Returns
    -------
    float
        Mean regret.
    """
    yy = np.asarray(true_outcomes_by_arm, dtype=np.float64)
    if yy.ndim != 2:
        raise ValueError(f"policy_risk: true_outcomes_by_arm must be (n, n_arms); got {yy.shape}")
    if not np.all(np.isfinite(yy)):
        raise ValueError("policy_risk: true_outcomes_by_arm contains non-finite values")
    n, k = yy.shape
    pi = np.asarray(policy_arm).ravel()
    if pi.size != n:
        raise ValueError(f"policy_risk: policy_arm has {pi.size} entries but there are {n} rows")
    pi = np.rint(np.asarray(pi, dtype=np.float64)).astype(np.int64)
    if n and (int(pi.min()) < 0 or int(pi.max()) >= k):
        raise ValueError(f"policy_risk: policy_arm must lie in [0, {k - 1}]")
    chosen = yy[np.arange(n), pi]
    return float(np.mean(yy.max(axis=1) - chosen))


# ======================================================================================
# 3. off-policy value
# ======================================================================================
def _bootstrap_column_means(
    cols: np.ndarray, n_boot: int, rng: np.random.Generator, *, budget: int = 4_000_000
) -> np.ndarray:
    """Nonparametric bootstrap of the column means of ``cols``, in memory-bounded chunks.

    Per `SPEC_PERF.md` section 8 the resampling is over **per-row influence contributions**: a
    replicate is a reweighted mean of rows that already exist, and no model is ever refitted.

    Implementation notes, because the obvious ways are both slow at ``n = 120k``:

    * a replicate is materialised as a **count vector** (``bincount`` of ``n`` uniform draws) and
      the whole chunk is reduced with one ``counts @ cols`` BLAS call.  Fancy-indexing
      ``cols[idx]`` per replicate instead costs a full ``(n, m)`` gather each time and measured
      ~9x slower; ``rng.multinomial`` draws the same counts directly but is O(n) *binomials* per
      replicate and measured ~5x slower.
    * one draw call per replicate, in order, so the random stream -- and therefore the answer --
      does not depend on ``budget`` or on how the work happened to be chunked.

    Peak memory is ``chunk * n`` floats regardless of how many columns are carried.
    """
    n, m = cols.shape
    chunk = int(max(1, min(n_boot, budget // max(n, 1))))
    out = np.empty((n_boot, m), dtype=np.float64)
    counts = np.empty((chunk, n), dtype=np.float64)
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        for r in range(size):
            counts[r] = np.bincount(rng.integers(0, n, size=n), minlength=n)
        out[done : done + size] = (counts[:size] @ cols) / float(n)
        done += size
    return out


def policy_value(
    y: np.ndarray,
    w: np.ndarray,
    policy_arm: np.ndarray,
    propensity: np.ndarray,
    *,
    method: Literal["ipw", "snipw", "dr"] = "dr",
    mu_hat: np.ndarray | None = None,
    n_boot: int = 500,
    random_state: int | np.random.Generator | None = None,
    alpha: float = 0.05,
    weight_cap: float | None = None,
) -> dict[str, Any]:
    """Off-policy value of an assignment rule, with a bootstrap confidence interval.

    The logged data was collected under a confounded logging policy that chose ``A_i`` with
    probability ``e_{A_i}(X_i)``.  A *different* policy ``pi`` was never run.  This estimates what
    it would have earned, which is the only number a budget owner cares about.

    ESTIMATOR MATHS (METHODOLOGY section 5.4)
    -----------------------------------------
    With the importance weight ``rho_i = 1{A_i = pi(X_i)} / e_{pi(X_i)}(X_i)``::

        IPW    V = (1/n) * sum_i rho_i * Y_i
        SNIPW  V = ( sum_i rho_i * Y_i ) / ( sum_i rho_i )
        DR     V = (1/n) * sum_i [ mu_{pi(X_i)}(X_i) + rho_i * ( Y_i - mu_{pi(X_i)}(X_i) ) ]

    * **IPW** is unbiased given correct propensities, and its variance is driven by the *level* of
      ``Y``: one matched row with ``e = 0.02`` contributes fifty times its own outcome.
    * **SNIPW** (Hajek) divides by the realised weight sum. Slightly biased in finite samples,
      much lower variance, and guaranteed to land inside the range of the observed outcomes -- it
      can never return a value no customer could have produced.
    * **DR** (AIPW) is the default and is consistent if *either* the propensity or the outcome
      model is right.  The outcome model absorbs the level of ``Y`` so only the residual is
      re-weighted, which is why its standard error is typically a fraction of IPW's.

    **If ``method="dr"`` and ``mu_hat is None`` there is no outcome model, so DR is not
    computable.** The estimator falls back to SNIPW and the returned ``method`` reads
    ``"dr[fallback=snipw]"`` (:data:`DR_FALLBACK_METHOD`), with a warning logged.  It never
    degrades silently.

    Uncertainty
    -----------
    ``n_boot`` nonparametric bootstrap replicates of the **per-row influence contributions**
    (`SPEC_PERF.md` section 8 -- nothing is refitted inside the loop).  ``se`` is the standard
    deviation of the replicates and the interval is the percentile interval, which for SNIPW
    respects the ratio structure that a normal approximation would linearise away.  An analytic
    influence-function standard error is returned alongside as ``se_analytic``; if the two
    disagree badly the asymptotics have not arrived, usually because a handful of rows carry
    enormous weight, and neither interval should be believed.  ``n_boot=0`` skips the bootstrap
    and returns the analytic standard error with a normal interval.

    Parameters
    ----------
    y : numpy.ndarray
        ``(n,)`` realised outcome, higher is better.
    w : numpy.ndarray
        ``(n,)`` logged arm index.  **Not** binarised here -- the policy is evaluated arm by arm,
        so a multi-arm policy is handled natively.
    policy_arm : numpy.ndarray
        ``(n,)`` arm the evaluated policy would choose.
    propensity : numpy.ndarray
        ``(n, n_arms)`` logging propensities, or ``(n,)`` read as ``e_{A_i}(X_i)``, the propensity
        of the arm actually received.  The 1-D form is sufficient because rows where the policy
        disagrees with the log get weight zero regardless of what their propensity was.
    method : {"ipw", "snipw", "dr"}, keyword-only, default "dr"
        Estimator, as above.
    mu_hat : numpy.ndarray, optional
        ``(n, n_arms)`` outcome model, or ``(n,)`` already evaluated at the policy's arm.
        Required for a genuine DR estimate.
    n_boot : int, keyword-only, default 500
        Bootstrap replicates. ``0`` disables the bootstrap.
    random_state : int or numpy.random.Generator, optional
        Seed for the bootstrap; same seed gives bit-identical output.
    alpha : float, keyword-only, default 0.05
        Two-sided level.
    weight_cap : float, optional
        Truncate importance weights at this value.  Trades variance for bias; the number of
        truncated rows is logged and returned.  ``None`` (the default) leaves them alone, because
        a silent cap can launder a broken propensity model into a tight interval.

    Returns
    -------
    dict
        ``value``, ``se``, ``ci_low``, ``ci_high``, ``method``, ``n`` (the contract), plus
        ``se_analytic``, ``se_boot``, ``n_matched`` (rows where the policy agrees with the log),
        ``match_rate``, ``ess`` (Kish effective sample size of the weights), ``max_weight``,
        ``n_clipped``, ``n_boot``, ``alpha`` and ``fallback``.

    Warnings
    --------
    Off-policy value is only believable where the logging policy had some chance of taking the
    evaluated action.  ``ess`` far below ``n`` means the estimate rests on a handful of rows; the
    overlap diagnostics in :func:`prism.models.propensity.overlap_diagnostics` bound where these
    numbers can be trusted, and the report should say so.
    """
    y_arr = _as_float_1d(y, "policy_value: y")
    n = y_arr.size
    if n == 0:
        raise ValueError("policy_value: empty input")
    logged = np.rint(_as_float_1d(w, "policy_value: w")).astype(np.int64)
    pi = np.rint(_as_float_1d(policy_arm, "policy_value: policy_arm")).astype(np.int64)
    if logged.size != n or pi.size != n:
        raise ValueError(
            f"policy_value: length mismatch -- y={n}, w={logged.size}, policy_arm={pi.size}"
        )
    if int(logged.min()) < 0 or int(pi.min()) < 0:
        raise ValueError("policy_value: arm indices must be non-negative (0 = control)")
    if str(method) not in POLICY_VALUE_METHODS:
        raise ValueError(f"policy_value: method must be one of {POLICY_VALUE_METHODS}; got {method!r}")
    alpha = _check_alpha(alpha, "policy_value")
    if int(n_boot) < 0:
        raise ValueError(f"policy_value: n_boot must be >= 0; got {n_boot}")

    p = np.asarray(propensity, dtype=np.float64)
    if p.ndim == 2:
        if p.shape[0] != n:
            raise ValueError(f"policy_value: propensity has {p.shape[0]} rows but y has {n}")
        if int(pi.max()) >= p.shape[1]:
            raise ValueError(
                f"policy_value: policy_arm reaches {int(pi.max())} but propensity has only "
                f"{p.shape[1]} arm columns"
            )
        e_pi = p[np.arange(n), pi]
    elif p.ndim == 1:
        if p.size != n:
            raise ValueError(f"policy_value: propensity has {p.size} entries but y has {n}")
        e_pi = p.copy()
    else:
        raise ValueError(f"policy_value: propensity must be (n,) or (n, n_arms); got {p.shape}")
    if not np.all(np.isfinite(e_pi)):
        raise ValueError("policy_value: propensity contains non-finite values")
    # A propensity is a probability. Values outside [0, 1] used to be absorbed by the clip below
    # -- a negative one became 1e-6 and produced an importance weight of a million, i.e. a
    # silently enormous number where the input was simply invalid. Refuse them instead.
    e_min, e_max = float(e_pi.min()), float(e_pi.max())
    if e_min < 0.0 or e_max > 1.0:
        raise ValueError(
            f"policy_value: propensity must be a probability in [0, 1]; got the range "
            f"[{e_min:.6g}, {e_max:.6g}]. A generalised propensity passed as (n, n_arms) should "
            "have rows that sum to 1; a 1-D vector is read as the propensity of the arm the "
            "policy would take."
        )

    match = (logged == pi).astype(np.float64)
    n_matched = int(match.sum())
    if n_matched == 0:
        raise ValueError(
            "policy_value: the evaluated policy never agrees with the logged arm, so its value is "
            "not identified from this data at all (every importance weight is zero)."
        )

    n_positivity = int(((e_pi < _PROB_EPS) & (match > 0.0)).sum())
    if n_positivity:
        LOG.warning(
            "policy_value: %d row(s) the policy matches have a logging propensity below %.0e, so "
            "each contributes an importance weight above %.0e and the estimate is effectively a "
            "few rows. This is a positivity violation, not a small numerical detail; check ess "
            "and max_weight in the result before quoting the value.",
            n_positivity,
            _PROB_EPS,
            1.0 / _PROB_EPS,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = match / np.clip(e_pi, _PROB_EPS, None)
    max_weight = float(rho.max())
    n_clipped = 0
    if weight_cap is not None:
        cap = float(weight_cap)
        n_clipped = int((rho > cap).sum())
        if n_clipped:
            LOG.warning(
                "policy_value: truncated %d/%d importance weights at %.4g (max was %.4g); this "
                "trades variance for bias and the bias is not in the interval.",
                n_clipped,
                n_matched,
                cap,
                max_weight,
            )
        rho = np.minimum(rho, cap)

    used = str(method)
    fallback = False
    if used == "dr" and mu_hat is None:
        LOG.warning(
            "policy_value: method='dr' needs mu_hat and none was given; falling back to SNIPW and "
            "reporting method=%r. This is not a doubly-robust estimate.",
            DR_FALLBACK_METHOD,
        )
        used = "snipw"
        fallback = True

    mu_pi = np.zeros(n, dtype=np.float64)
    if mu_hat is not None:
        m = np.asarray(mu_hat, dtype=np.float64)
        if m.ndim == 2:
            if m.shape[0] != n:
                raise ValueError(f"policy_value: mu_hat has {m.shape[0]} rows but y has {n}")
            if int(pi.max()) >= m.shape[1]:
                raise ValueError("policy_value: policy_arm indexes past mu_hat's arm columns")
            mu_pi = m[np.arange(n), pi]
        elif m.ndim == 1:
            if m.size != n:
                raise ValueError(f"policy_value: mu_hat has {m.size} entries but y has {n}")
            mu_pi = m.copy()
        else:
            raise ValueError(f"policy_value: mu_hat must be (n,) or (n, n_arms); got {m.shape}")
        if not np.all(np.isfinite(mu_pi)):
            raise ValueError("policy_value: mu_hat contains non-finite values")

    psi_ipw = rho * y_arr
    psi_dr = mu_pi + rho * (y_arr - mu_pi)
    num = rho * y_arr
    den = rho

    mean_den = float(den.mean())
    if used == "ipw":
        value = float(psi_ipw.mean())
        infl = psi_ipw
    elif used == "snipw":
        if mean_den <= _EPS:
            raise ValueError("policy_value: SNIPW denominator is zero; no matched rows carry weight")
        value = float(num.mean() / mean_den)
        infl = (den * (y_arr - value)) / mean_den + value
    else:
        value = float(psi_dr.mean())
        infl = psi_dr
    se_analytic = float(np.std(infl, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    se_boot = float("nan")
    if int(n_boot) > 0:
        rng = as_rng(random_state)
        cols = np.column_stack([psi_ipw, num, den, psi_dr])
        means = _bootstrap_column_means(cols, int(n_boot), rng)
        if used == "ipw":
            reps = means[:, 0]
        elif used == "snipw":
            with np.errstate(divide="ignore", invalid="ignore"):
                reps = np.where(np.abs(means[:, 2]) > _EPS, means[:, 1] / means[:, 2], np.nan)
        else:
            reps = means[:, 3]
        good = np.isfinite(reps)
        if int(good.sum()) < 2:
            LOG.warning("policy_value: the bootstrap produced no usable replicates; using the "
                        "analytic standard error and a normal interval instead.")
            ci_low, ci_high = _normal_ci(value, se_analytic, alpha)
            se = se_analytic
        else:
            reps = reps[good]
            se_boot = float(np.std(reps, ddof=1))
            ci_low = float(np.quantile(reps, float(alpha) / 2.0))
            ci_high = float(np.quantile(reps, 1.0 - float(alpha) / 2.0))
            se = se_boot
    else:
        ci_low, ci_high = _normal_ci(value, se_analytic, alpha)
        se = se_analytic

    sum_rho = float(rho.sum())
    ess = float(sum_rho * sum_rho / max(float((rho * rho).sum()), _EPS))
    return {
        "value": float(value),
        "se": float(se),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "method": DR_FALLBACK_METHOD if fallback else used,
        "n": int(n),
        "se_analytic": se_analytic,
        "se_boot": se_boot,
        "n_matched": n_matched,
        "match_rate": float(n_matched) / float(n),
        "ess": ess,
        "max_weight": max_weight,
        "n_clipped": n_clipped,
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "fallback": fallback,
    }


def _normal_ci(value: float, se: float, alpha: float) -> tuple[float, float]:
    """Symmetric normal interval, degrading to ``(nan, nan)`` when the SE is unusable."""
    if not np.isfinite(se):
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(1.0 - float(alpha) / 2.0))
    return float(value - z * se), float(value + z * se)


# ======================================================================================
# 4. adjusted heterogeneity tests -- GATES and the BLP
# ======================================================================================
def gates(
    y: np.ndarray,
    w: np.ndarray,
    cate: np.ndarray,
    propensity: np.ndarray,
    n_groups: int = 5,
    *,
    mu_hat: np.ndarray | None = None,
    arm: int | None = None,
    alpha: float = 0.05,
    clip: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """Sorted Group Average Treatment Effects (Chernozhukov-Demirer-Duflo-Fernandez-Val).

    Sort the rows into ``n_groups`` equal-size groups by predicted effect and estimate the **true**
    ATE inside each group with a doubly-robust estimator and a confidence interval.  If the model
    has found real heterogeneity the group effects rise monotonically and the top-minus-bottom
    difference is significantly positive.  If it has not, the groups are flat and the difference is
    noise -- and that verdict comes with a p-value rather than an opinion about a chart.

    WHY THIS CATCHES WHAT QINI CANNOT
    ---------------------------------
    Consider a model that has quietly learned the **outcome level** ``E[Y | X]`` instead of the
    **effect** ``E[Y(1) - Y(0) | X]`` -- the single most common failure in uplift modelling,
    because the level is easy to predict and the effect is not.  The Qini curve cannot tell you
    that this has happened:

    1. The Qini statistic is an *unadjusted* contrast.  Its value at depth ``k`` is
       ``N_t(k) * (mean_t(k) - mean_c(k))``: it rescales the control sum by the treated/control
       **count** ratio in the prefix and by nothing else.  Under confounded assignment -- the
       whole point of PRISM's observational block -- neither factor is clean.  The count ratio
       drifts along a level-sorted ranking because the level is correlated with the propensity,
       and the difference in means inside the prefix mixes the treatment effect with the baseline
       gap between the rows that happened to be treated there and the rows that were not.  There
       is no propensity score anywhere in the Qini computation to remove either.
    2. The size of the resulting artefact is not small and its **sign is not fixed**.  The smoke
       test at the bottom of this module builds a score from features that are provably unrelated
       to the treatment effect and measures Qini coefficients of ``-0.12`` and ``-0.42`` for it
       (against ``+0.01`` for an honestly uninformative score), one of them ranked below a model
       whose sign has been deliberately inverted.  A Qini coefficient computed on confounded data
       is therefore not readable as "fraction of the achievable uplift captured".

    GATES removes both problems.  It is built on the AIPW score (:func:`_aipw_scores`), which
    divides by the propensity, so selection into treatment is adjusted away and the arm mix cannot
    leak in; and it asks a *difference* question -- is ``gate_top`` bigger than ``gate_bottom``?
    -- so any common level shift cancels.  A model that has rediscovered the outcome level
    therefore produces groups that differ enormously in mean ``y`` and not at all in ``gate``, and
    the top-minus-bottom p-value lands where it should.  That is the test, and it is the one to
    run before believing a Qini number.

    ORDERING (read this)
    --------------------
    **Group 1 is the lowest** predicted effect, so a good model's ``gate`` *increases* down the
    frame, matching the CDDF papers and ``prism.pipelines.figures.plot_gates``.
    :func:`uplift_by_decile` uses the opposite convention.

    ESTIMATOR MATHS
    ---------------
    Inside group ``g``::

        psi_i = mu1(X_i) - mu0(X_i)
                + W_i/e(X_i)         * (Y_i - mu1(X_i))
                - (1-W_i)/(1-e(X_i)) * (Y_i - mu0(X_i))

        gate_g = mean_{i in g} psi_i
        se_g   = sd_{i in g}(psi_i) / sqrt(n_g)
        ci     = gate_g -/+ z_{1-alpha/2} * se_g

    ``psi`` is the efficient influence function of the ATE, so ``se_g`` is the efficient standard
    error rather than a plug-in.  Groups are disjoint, hence independent, so::

        diff   = gate_G - gate_1
        se_diff = sqrt( se_G^2 + se_1^2 )
        p       = 2 * (1 - Phi(|diff| / se_diff))

    **The outcome model.** If ``mu_hat`` is supplied the full AIPW score above is used and the
    estimator is doubly robust.  If it is not, ``mu1`` and ``mu0`` are fitted *inside each group*
    as Hajek-weighted constants (:func:`_hajek_means`); the score then reduces algebraically to
    the within-group self-normalised IPW difference, with ``sd(psi)/sqrt(n_g)`` delivering the
    correct linearised standard error.  In that mode the constant is a variance-reducing control
    variate and consistency rests on the propensity alone -- so it is *singly* robust, and
    ``frame.attrs["outcome_model"]`` says which of the two produced the numbers.

    Parameters
    ----------
    y : numpy.ndarray
        ``(n,)`` outcome, higher is better.
    w : numpy.ndarray
        ``(n,)`` arm index; more than two levels is binarised to ``(w > 0)`` and logged.
    cate : numpy.ndarray
        ``(n,)`` or ``(n, n_arms - 1)`` predicted effects used only to *sort*, never to estimate.
    propensity : numpy.ndarray
        ``(n, n_arms)`` generalised propensity (``P(W=1|X)`` is taken as ``1 - e_0``) or ``(n,)``
        read directly as ``P(W = 1 | X)``.
    n_groups : int, default 5
        Number of equal-size groups.
    mu_hat : numpy.ndarray, optional
        ``(n, 2)`` as ``[control, treated]``, or ``(n, n_arms)`` together with ``arm``.
    arm : int, optional
        Use only this arm's column of a 2-D ``cate`` to *sort* the groups, and read ``mu_hat``'s
        arm-``arm`` column as the treated outcome model.  It does **not** change the contrast,
        which is always ``(w > 0)`` against control; for a clean arm-``a``-versus-control GATES,
        subset to the rows with ``w in {0, a}`` before calling.
    alpha : float, keyword-only, default 0.05
        Two-sided level for the intervals and the p-values.
    clip : tuple of float, keyword-only, default (0.01, 0.99)
        Propensity clip. Clipping is logged when it binds.

    Returns
    -------
    pandas.DataFrame
        One row per group, columns :data:`GATES_COLUMNS`.  ``top_minus_bottom``,
        ``top_minus_bottom_se`` and ``top_minus_bottom_p`` are constant columns repeating the
        headline test -- duplicated from ``frame.attrs`` on purpose, because ``attrs`` do not
        survive ``to_csv`` and this is the number the report quotes.  ``frame.attrs`` additionally
        carries ``p_value_one_sided`` (the directional test that the top group really is higher),
        ``z``, ``monotone_increasing``, ``spearman_group_gate``, ``outcome_model``, ``n_groups``,
        ``n_groups_unidentified``, ``ate_aipw``, ``ate_aipw_identified_rows`` and ``alpha``.

    Unidentified groups
    -------------------
    A group that contains **only treated** or **only control** rows has no within-group contrast,
    and without ``mu_hat`` there is nothing to stand in for the missing arm.  Such a group is
    reported as ``nan`` (gate, se, interval and p-value), ``n_groups_unidentified`` counts them,
    and the ``nan`` deliberately propagates into ``top_minus_bottom`` and its p-value: a headline
    heterogeneity test must not read as significant on the strength of a group that was never
    estimated.  ``ate_aipw`` is ``nan`` in that case too, with the average over the identified
    rows -- a different estimand -- reported separately as ``ate_aipw_identified_rows``.

    It is worth knowing *when* this fires: a ranking strongly correlated with assignment, which
    is exactly the level-only score this function exists to catch, combined with many groups or
    weak overlap.  Fewer groups, an overlap trim, or a supplied ``mu_hat`` all resolve it.
    """
    alpha = _check_alpha(alpha, "gates")
    clip = _check_clip(clip, "gates")
    y_arr, treated, score = _prepare(y, w, cate, arm, "gates")
    n = y_arr.size
    e1 = _binary_propensity(propensity, treated, clip=clip, context="gates")

    if mu_hat is not None:
        mu0_all, mu1_all = _outcome_model_columns(mu_hat, n, arm, "gates")
        psi = _aipw_scores(y_arr, treated, e1, mu0_all, mu1_all)
        model_label = "aipw[user mu_hat] -- doubly robust"
    else:
        psi = np.empty(n, dtype=np.float64)
        model_label = "aipw[group-constant Hajek mu] -- consistent via the propensity alone"

    order = np.argsort(score, kind="stable")  # ascending: group 1 = lowest predicted effect
    blocks = _equal_bins(n, n_groups, "gates")
    z = float(stats.norm.ppf(1.0 - float(alpha) / 2.0))

    rows: list[dict[str, Any]] = []
    n_unidentified = 0
    for g, block in enumerate(blocks, start=1):
        idx = order[block]
        n_t_g = float(treated[idx].sum())
        n_c_g = float(idx.size) - n_t_g
        single_armed = n_t_g == 0.0 or n_c_g == 0.0
        if mu_hat is None:
            mu0_g, mu1_g = _hajek_means(y_arr, treated, e1, idx)
            if single_armed or not (np.isfinite(mu0_g) and np.isfinite(mu1_g)):
                # Both arm means are needed and one of them does not exist.  Substituting a
                # placeholder (a zero, say) would return a large, tiny-standard-error number that
                # reads as a measurement; it is not one.  The group is reported as nan instead,
                # and the nan propagates into top-minus-bottom so the headline test cannot be
                # "passed" on the strength of a group that was never identified.
                LOG.warning(
                    "gates: group %d holds %d treated and %d control rows, so its effect is not "
                    "identified; the group and the top-minus-bottom test are reported as nan. "
                    "This happens when the ranking is strongly correlated with assignment; use "
                    "fewer groups, trim on overlap, or pass mu_hat.",
                    g,
                    int(n_t_g),
                    int(n_c_g),
                )
                psi[idx] = np.nan
                n_unidentified += 1
            else:
                psi[idx] = _aipw_scores(
                    y_arr[idx],
                    treated[idx],
                    e1[idx],
                    np.full(idx.size, mu0_g),
                    np.full(idx.size, mu1_g),
                )
        elif single_armed:
            # With a user outcome model the score is still defined, but the missing arm's value
            # in this group comes entirely from mu_hat's extrapolation, not from any observation.
            LOG.warning(
                "gates: group %d holds %d treated and %d control rows; its gate rests entirely "
                "on mu_hat's extrapolation for the empty arm, not on observed outcomes.",
                g,
                int(n_t_g),
                int(n_c_g),
            )
        block_psi = psi[idx]
        identified = bool(idx.size > 1 and np.all(np.isfinite(block_psi)))
        gate = float(block_psi.mean()) if identified else float("nan")
        se = (
            float(np.std(block_psi, ddof=1) / np.sqrt(idx.size)) if identified else float("nan")
        )
        p_g = (
            float(2.0 * stats.norm.sf(abs(gate) / se))
            if np.isfinite(se) and se > _EPS
            else float("nan")
        )
        rows.append(
            {
                "group": g,
                "n": int(idx.size),
                "mean_predicted_cate": float(score[idx].mean()),
                "gate": gate,
                "se": se,
                "ci_low": gate - z * se if np.isfinite(se) else np.nan,
                "ci_high": gate + z * se if np.isfinite(se) else np.nan,
                "p_value": p_g,
            }
        )

    frame = pd.DataFrame(rows)
    top, bottom = frame.iloc[-1], frame.iloc[0]
    diff = float(top["gate"] - bottom["gate"])
    se_diff = float(np.sqrt(float(top["se"]) ** 2 + float(bottom["se"]) ** 2))
    if np.isfinite(se_diff) and se_diff > _EPS:
        z_stat = diff / se_diff
        p_two = float(2.0 * stats.norm.sf(abs(z_stat)))
        p_one = float(stats.norm.sf(z_stat))
    else:
        z_stat, p_two, p_one = float("nan"), float("nan"), float("nan")

    frame["top_minus_bottom"] = diff
    frame["top_minus_bottom_se"] = se_diff
    frame["top_minus_bottom_p"] = p_two
    frame = frame[list(GATES_COLUMNS)]

    gate_vals = frame["gate"].to_numpy(dtype=float)
    finite = np.isfinite(gate_vals)
    rho = (
        float(stats.spearmanr(frame["group"].to_numpy()[finite], gate_vals[finite]).statistic)
        if finite.sum() >= 3
        else float("nan")
    )
    frame.attrs.update(
        {
            "ordering": "group 1 = lowest predicted effect; gate should INCREASE",
            "top_minus_bottom": diff,
            "top_minus_bottom_se": se_diff,
            "z": z_stat,
            "p_value": p_two,
            "p_value_one_sided": p_one,
            "monotone_increasing": bool(finite.all() and np.all(np.diff(gate_vals) >= 0)),
            "spearman_group_gate": rho,
            "outcome_model": model_label,
            "n_groups": int(len(frame)),
            "n_groups_unidentified": int(n_unidentified),
            "alpha": float(alpha),
            # nan rather than a nanmean when a group could not be identified: averaging over the
            # surviving rows silently changes the estimand to "the ATE among the groups that
            # happened to have both arms". The identified-rows figure is kept alongside, named.
            "ate_aipw": float(psi.mean()) if n_unidentified == 0 else float("nan"),
            "ate_aipw_identified_rows": (
                float(np.nanmean(psi)) if np.any(np.isfinite(psi)) else float("nan")
            ),
        }
    )
    return frame


def blp_calibration(
    y: np.ndarray,
    w: np.ndarray,
    cate: np.ndarray,
    propensity: np.ndarray,
    *,
    mu_hat: np.ndarray | None = None,
    arm: int | None = None,
    alpha: float = 0.05,
    clip: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """Best Linear Predictor of the true effect given the predicted one (CDDF 2018).

    ESTIMATOR MATHS
    ---------------
    Form the AIPW pseudo-outcome ``psi_i`` (:func:`_aipw_scores`), which satisfies
    ``E[psi | X] = tau(X)``, and run the ordinary least squares regression::

        psi_i = beta1 + beta2 * ( tau_hat_i - mean(tau_hat) ) + error_i

    with heteroskedasticity-consistent (HC1) standard errors, because ``psi`` is wildly
    heteroskedastic by construction -- its variance scales with ``1/e(X)``.

        beta1  =  E[tau]                      the ATE. The regressor is centred, so the intercept
                                              is exactly the AIPW ATE.
        beta2  =  Cov(tau, tau_hat) / Var(tau_hat)
                                              the CALIBRATION SLOPE.

    HOW TO READ beta2
    -----------------
    ``beta2 ~ 1``
        Calibrated.  A predicted effect one unit larger really does correspond to a true effect
        one unit larger.  Budgets built on these numbers will size correctly.
    ``beta2 << 1`` (the common failure)
        The **spread of the predictions is exaggerated**.  The model says "these customers are
        worth 30 and those are worth 2" when the truth is 12 and 7.  The *ranking* can still be
        excellent, so Qini, AUUC and TOC all look fine -- they only ever see the order.  The
        damage is economic: every ROI, every payback period and every budget sized off the
        predicted effect is wrong, and a knapsack solver fed inflated spreads over-concentrates
        spend on a head that does not deliver.  An unregularised T-learner and any model scored on
        rank metrics alone will land here.
    ``beta2 ~ 0``
        No usable information: the predictions are unrelated to the truth (and the confidence
        interval will say so).
    ``beta2 < 0``
        Systematically inverted -- treating the top of the ranking destroys value.
    ``beta2 > 1``
        Under-dispersed, the classic signature of a heavily regularised S-learner whose predicted
        effects are all shrunk toward the ATE.

    A useful corollary: shrinking a noisy but well-ranked score toward its mean by the right
    factor turns ``beta2 = 0.3`` into ``beta2 = 1`` without changing the ranking or the Qini at
    all.  Calibration and accuracy are different properties, and this is the only function in the
    module that can tell them apart.

    **The outcome model.** As in :func:`gates`: with ``mu_hat`` the score is the full AIPW score
    and doubly robust; without it, ``mu0`` and ``mu1`` are global Hajek-weighted constants and
    consistency rests on the propensity.  ``frame.attrs["outcome_model"]`` records which.

    Parameters
    ----------
    y, w, cate, propensity
        As in :func:`gates`.
    mu_hat : numpy.ndarray, optional
        ``(n, 2)`` as ``[control, treated]``, or ``(n, n_arms)`` together with ``arm``.
    arm : int, optional
        As in :func:`gates`: it selects the ``cate`` column and the ``mu_hat`` column, not the
        contrast, which is always ``(w > 0)`` against control.
    alpha : float, keyword-only, default 0.05
        Two-sided level.
    clip : tuple of float, keyword-only, default (0.01, 0.99)
        Propensity clip, logged when it binds.

    Returns
    -------
    pandas.DataFrame
        Two rows -- ``beta1`` (the ATE) and ``beta2`` (the calibration slope) -- with columns
        :data:`BLP_COLUMNS`.  ``frame.attrs`` carries ``r_squared``, ``n``, ``outcome_model``,
        ``sd_cate``, ``heterogeneity_detected`` (is ``beta2`` significantly above zero?) and
        ``calibrated`` (does the ``beta2`` interval contain 1?).

    Notes
    -----
    ``beta2`` is undefined when the predicted effects are constant; the function then returns
    ``nan`` for that row and logs, rather than dividing by a zero variance.
    """
    alpha = _check_alpha(alpha, "blp_calibration")
    clip = _check_clip(clip, "blp_calibration")
    y_arr, treated, score = _prepare(y, w, cate, arm, "blp_calibration")
    n = y_arr.size
    e1 = _binary_propensity(propensity, treated, clip=clip, context="blp_calibration")

    if mu_hat is not None:
        mu0, mu1 = _outcome_model_columns(mu_hat, n, arm, "blp_calibration")
        model_label = "aipw[user mu_hat] -- doubly robust"
    else:
        idx_all = np.arange(n, dtype=np.int64)
        m0, m1 = _hajek_means(y_arr, treated, e1, idx_all)
        if not (np.isfinite(m0) and np.isfinite(m1)):
            raise ValueError("blp_calibration: one arm carries no weight; cannot form the score")
        mu0 = np.full(n, m0)
        mu1 = np.full(n, m1)
        model_label = "aipw[global Hajek mu] -- consistent via the propensity alone"

    psi = _aipw_scores(y_arr, treated, e1, mu0, mu1)
    sd_score = float(score.std(ddof=1)) if n > 1 else 0.0
    centred = score - float(score.mean())
    design = np.column_stack([np.ones(n), centred])
    k = design.shape[1]
    if n <= k:
        raise ValueError(f"blp_calibration: need more than {k} rows; got {n}")

    degenerate = sd_score <= _EPS
    if degenerate:
        LOG.warning(
            "blp_calibration: the predicted effects are constant (sd=%.3g), so the calibration "
            "slope is not identified; beta2 is reported as nan.",
            sd_score,
        )

    xtx = design.T @ design
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (design.T @ psi)
    resid = psi - design @ beta
    meat = (design * (resid**2)[:, None]).T @ design
    cov = xtx_inv @ meat @ xtx_inv * (float(n) / float(n - k))  # HC1 small-sample correction
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > _EPS, beta / se, np.nan)
    df = n - k
    pvals = 2.0 * stats.t.sf(np.abs(tstat), df=df)
    tcrit = float(stats.t.ppf(1.0 - float(alpha) / 2.0, df=df))

    if degenerate:
        beta = np.array([beta[0], np.nan])
        se = np.array([se[0], np.nan])
        tstat = np.array([tstat[0], np.nan])
        pvals = np.array([pvals[0], np.nan])

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((psi - psi.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > _EPS else float("nan")

    frame = pd.DataFrame(
        {
            "term": ["beta1", "beta2"],
            "parameter": ["ate", "calibration_slope"],
            "coefficient": beta.astype(float),
            "se": se.astype(float),
            "t": np.asarray(tstat, dtype=float),
            "p_value": np.asarray(pvals, dtype=float),
            "ci_low": beta - tcrit * se,
            "ci_high": beta + tcrit * se,
        }
    )[list(BLP_COLUMNS)]

    slope, slope_se = float(frame.loc[1, "coefficient"]), float(frame.loc[1, "se"])
    frame.attrs.update(
        {
            "r_squared": r2,
            "n": int(n),
            "outcome_model": model_label,
            "sd_cate": sd_score,
            "se_type": "HC1 (heteroskedasticity-consistent)",
            "heterogeneity_detected": bool(
                np.isfinite(slope) and np.isfinite(slope_se) and slope - tcrit * slope_se > 0.0
            ),
            "calibrated": bool(
                np.isfinite(slope)
                and np.isfinite(slope_se)
                and (slope - tcrit * slope_se) <= 1.0 <= (slope + tcrit * slope_se)
            ),
            "alpha": float(alpha),
        }
    )
    return frame


# ======================================================================================
# 5. the leaderboard
# ======================================================================================
def compare_learners(
    results: Mapping[str, np.ndarray],
    true_cate: np.ndarray | None = None,
    y: np.ndarray | None = None,
    w: np.ndarray | None = None,
    *,
    propensity: np.ndarray | None = None,
    treatment_cost: float = 0.0,
    arm: int | None = None,
    k: float = 0.2,
    n_groups: int = 5,
    n_boot: int = 200,
    random_state: int | np.random.Generator | None = 0,
) -> pd.DataFrame:
    """Tidy leaderboard: one row per learner, one column per metric that the inputs support.

    Which metrics appear depends entirely on what is passed, and the function computes *every*
    metric the inputs allow rather than making the caller guess:

    ==========================  ====================================================
    inputs supplied             metrics filled in
    ==========================  ====================================================
    ``results`` only            ``n``, ``mean_cate``, ``sd_cate``
    ``+ true_cate``             ``pehe``, ``eps_ate``, ``spearman_true``
    ``+ y, w``                  ``qini``, ``auuc``, ``uplift_at_20pct``
    ``+ y, w, propensity``      ``blp_slope``, ``gates_p``, ``policy_value``(+ ``se``)
    ==========================  ====================================================

    The column set is **always** :data:`LEADERBOARD_COLUMNS`; a metric that could not be computed
    is ``NaN``, never a missing column, so ``artifacts/reports/leaderboard.csv`` has a stable
    header across runs and can be concatenated over time.

    ORDERING (documented, as required)
    ----------------------------------
    Rows are sorted by the **most decision-relevant metric available**, in this precedence:

    1. ``policy_value`` descending -- currency.  This is the only metric in the frame denominated
       in the thing the business maximises, so when it exists it wins.
    2. ``qini`` descending -- the best ranking-quality proxy obtainable from ``(y, w, cate)``
       alone, and ranking is what a budget-constrained allocator consumes.
    3. ``pehe`` ascending -- ground-truth accuracy.  It ranks *below* the decision metrics on
       purpose: a model can win on PEHE and still lose money if it is accurate on the customers
       nobody will be able to afford to treat.
    4. ``eps_ate`` ascending.
    5. Learner name, ascending -- the deterministic tie-break, also applied within ties above.

    ``frame.attrs["sorted_by"]`` names the metric that actually decided the order, and the
    ``rank`` column materialises it so the ordering survives a ``to_csv``.

    The ``policy_value`` column
    ---------------------------
    Evaluated for the concrete policy **"treat where the predicted effect exceeds
    ``treatment_cost``, otherwise do nothing"**, which is the natural unconstrained decision rule
    implied by a CATE model, using SNIPW on the binarised arm.  It is *not* the budget-constrained
    number: that comes from :mod:`prism.decision.optimize` scored with
    :func:`prism.decision.policy_eval.evaluate_policy`, which also takes an outcome model and can
    therefore be genuinely doubly robust.  This column exists so the leaderboard is sorted by
    something economic rather than by something merely statistical.

    Parameters
    ----------
    results : mapping of str to numpy.ndarray
        Learner name to predicted CATE, ``(n,)`` or ``(n, n_arms - 1)``.  All entries must have
        the same number of rows.
    true_cate : numpy.ndarray, optional
        Ground truth, enabling ``pehe`` / ``eps_ate`` / ``spearman_true``.  Must match each
        learner's shape (see :func:`pehe`).
    y, w : numpy.ndarray, optional
        Outcome and arm, enabling the ranking metrics.  Both or neither.
    propensity : numpy.ndarray, keyword-only, optional
        ``(n, n_arms)`` or ``(n,)``, enabling the adjusted metrics and the policy value.
    treatment_cost : float, keyword-only, default 0.0
        Passed through to the Qini/AUUC/uplift metrics and used as the treat/do-not-treat
        threshold for the policy value.
    arm : int, optional
        Score only this arm's column of 2-D predictions.
    k : float, keyword-only, default 0.2
        Depth for the ``uplift_at_20pct`` column (renamed in the column header only if you change
        it; the column name is fixed for schema stability, so a non-default ``k`` is recorded in
        ``frame.attrs["uplift_at_k"]``).
    n_groups : int, keyword-only, default 5
        Groups for the ``gates_p`` column.
    n_boot : int, keyword-only, default 200
        Bootstrap replicates for ``policy_value_se``.  Lower than :func:`policy_value`'s default
        because this runs once per learner.
    random_state : int or numpy.random.Generator, keyword-only, default 0
        Seed for those bootstraps.  Each learner is given its own child stream, so the leaderboard
        is reproducible and adding a learner does not perturb the others.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`LEADERBOARD_COLUMNS`, sorted as documented above.  ``frame.attrs`` carries
        ``sorted_by``, ``ordering_rule``, ``n_learners``, ``uplift_at_k`` and ``metrics_available``.

    Raises
    ------
    ValueError
        If ``results`` is empty, if the learners disagree about ``n``, or if exactly one of
        ``y`` / ``w`` is given.
    """
    if not isinstance(results, Mapping) or len(results) == 0:
        raise ValueError("compare_learners: results must be a non-empty mapping name -> cate array")
    if (y is None) != (w is None):
        raise ValueError("compare_learners: pass both y and w, or neither")

    names = list(results.keys())
    n_rows = {name: int(np.asarray(results[name]).shape[0]) for name in names}
    if len(set(n_rows.values())) > 1:
        raise ValueError(f"compare_learners: learners disagree about n -- {n_rows}")

    has_truth = true_cate is not None
    has_obs = y is not None and w is not None
    has_prop = has_obs and propensity is not None
    # One deterministic child stream per learner, in insertion order, so the leaderboard is
    # reproducible and the bootstrap of one learner does not depend on how many others there are.
    child_seeds = as_rng(random_state).integers(0, 2**32 - 1, size=len(names))
    rngs = {name: np.random.default_rng(int(s)) for name, s in zip(names, child_seeds)}

    rows: list[dict[str, Any]] = []
    for name in names:
        cate = np.asarray(results[name], dtype=np.float64)
        score = _score_vector(cate, arm, context=f"compare_learners[{name}]")
        row: dict[str, Any] = {
            "learner": str(name),
            "rank": np.nan,
            "policy_value": np.nan,
            "policy_value_se": np.nan,
            "qini": np.nan,
            "auuc": np.nan,
            "uplift_at_20pct": np.nan,
            "pehe": np.nan,
            "eps_ate": np.nan,
            "spearman_true": np.nan,
            "blp_slope": np.nan,
            "gates_p": np.nan,
            "n": int(score.size),
            "mean_cate": float(score.mean()),
            "sd_cate": float(score.std(ddof=1)) if score.size > 1 else 0.0,
        }

        if has_truth:
            row["pehe"] = pehe(true_cate, cate)
            row["eps_ate"] = eps_ate(true_cate, cate)
            truth_score = _score_vector(true_cate, arm, context="compare_learners[true_cate]")
            if truth_score.std() > _EPS and score.std() > _EPS:
                row["spearman_true"] = float(stats.spearmanr(truth_score, score).statistic)

        if has_obs:
            row["qini"] = qini_score(y, w, score, treatment_cost=treatment_cost)
            row["auuc"] = auuc(y, w, score, treatment_cost=treatment_cost)
            row["uplift_at_20pct"] = uplift_at_k(y, w, score, k, treatment_cost=treatment_cost)

        if has_prop:
            blp = blp_calibration(y, w, score, propensity)
            row["blp_slope"] = float(blp.loc[1, "coefficient"])
            gate_frame = gates(y, w, score, propensity, n_groups=n_groups)
            row["gates_p"] = float(gate_frame.attrs["p_value"])

            treated_flag, _, _ = _binarise_arm(w, context=f"compare_learners[{name}]")
            e1 = _binary_propensity(
                propensity, treated_flag, clip=(0.01, 0.99), context=f"compare_learners[{name}]"
            )
            policy = (score > float(treatment_cost)).astype(np.int64)
            prop2 = np.column_stack([1.0 - e1, e1])
            try:
                pv = policy_value(
                    y,
                    treated_flag.astype(np.int64),
                    policy,
                    prop2,
                    method="snipw",
                    n_boot=int(n_boot),
                    random_state=rngs[name],
                )
                row["policy_value"] = pv["value"]
                row["policy_value_se"] = pv["se"]
            except ValueError as exc:  # degenerate policy (treats nobody the log ever treated)
                LOG.info("compare_learners[%s]: policy value not identified (%s).", name, exc)

        rows.append(row)

    frame = pd.DataFrame(rows)[list(LEADERBOARD_COLUMNS)]

    sort_plan: list[tuple[str, bool]] = [
        ("policy_value", False),
        ("qini", False),
        ("pehe", True),
        ("eps_ate", True),
    ]
    sorted_by = "learner"
    ascending = True
    for col, asc in sort_plan:
        if frame[col].notna().any():
            sorted_by, ascending = col, asc
            break
    frame = frame.sort_values(
        [sorted_by, "learner"], ascending=[ascending, True], kind="mergesort"
    ).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1, dtype=np.int64)

    frame.attrs.update(
        {
            "sorted_by": sorted_by,
            "ordering_rule": (
                "policy_value desc > qini desc > pehe asc > eps_ate asc > learner asc; the first "
                "metric with any non-null value decides, ties broken by learner name"
            ),
            "n_learners": int(len(frame)),
            "uplift_at_k": float(k),
            "treatment_cost": float(treatment_cost),
            "metrics_available": {
                "ground_truth": bool(has_truth),
                "observational": bool(has_obs),
                "propensity": bool(has_prop),
            },
        }
    )
    return frame


# ======================================================================================
# smoke test -- real statistical assertions against a KNOWN truth
# ======================================================================================
def _simulate_evaluation_data(n: int = 12_000, random_state: int = 20_240_917) -> dict[str, Any]:
    """Two-arm confounded data with a known ``tau`` and both potential outcomes retained.

    The design is deliberately adversarial to naive evaluation:

    * assignment depends on ``x0`` alone, so the data is **confounded** but overlapping
      (propensities live in ``[0.12, 0.88]``) and the naive difference in means is badly biased;
    * the outcome **level** ``mu0`` is driven by ``x0`` and ``x3``; the effect ``tau`` is driven
      by ``x1``, ``x2`` and their interaction.  The two feature sets are **disjoint**, so a score
      built from the outcome level carries *provably zero* information about the effect -- and
      yet, because the head of a level-sorted ranking has a different arm mix from the tail, an
      unadjusted Qini still reports a large number for it.  That is the failure :func:`gates` is
      claimed to catch, and the smoke test checks the claim on both signs of the artefact;
    * ``tau`` is heterogeneous and **negative for a fifth of the rows** (sleeping dogs), so sign
      errors are punished rather than averaged away;
    * the two potential outcomes share their noise draw, so the finite-sample truth of any policy
      is exact rather than approximate.
    """
    rng = as_rng(random_state)
    x = rng.normal(size=(n, 4))
    e1 = np.clip(1.0 / (1.0 + np.exp(-(1.2 * x[:, 0]))), 0.12, 0.88)
    treated = (rng.random(n) < e1).astype(np.int64)
    mu0 = 20.0 + 3.0 * x[:, 0] + 1.5 * x[:, 3]
    tau = 2.0 + 2.5 * x[:, 1] - 1.2 * x[:, 2] + 1.0 * x[:, 1] * x[:, 2]
    noise = rng.normal(scale=2.0, size=n)
    y0 = mu0 + noise
    y1 = mu0 + tau + noise
    y = np.where(treated == 1, y1, y0)

    sd_tau = float(tau.std())
    good = tau + rng.normal(scale=0.5 * sd_tau, size=n)
    noisy = tau + rng.normal(scale=2.0 * sd_tau, size=n)
    useless = rng.normal(size=n)
    # Two scores built from the outcome LEVEL only. tau depends on x1/x2 and mu0 on x0/x3, so
    # both carry exactly zero effect information; they differ only by a sign flip.
    level_only = mu0.copy()
    level_inverted = -mu0
    # Shrinking a noisy score by cov(tau, s)/var(s) leaves the ranking untouched and makes the
    # BLP slope exactly 1 -- calibration and accuracy are different properties.
    lam = float(np.cov(tau, good, ddof=1)[0, 1] / np.var(good, ddof=1))
    calibrated = float(tau.mean()) + lam * (good - float(good.mean()))
    inflated = float(calibrated.mean()) + 3.0 * (calibrated - float(calibrated.mean()))

    return {
        "n": n,
        "x": x,
        "y": y,
        "y0": y0,
        "y1": y1,
        "w": treated,
        "e1": e1,
        "propensity": np.column_stack([1.0 - e1, e1]),
        "mu_true": np.column_stack([mu0, mu0 + tau]),
        "tau": tau,
        "sd_tau": sd_tau,
        "shrinkage": lam,
        "cate": {
            "perfect": tau,
            "good": good,
            "calibrated": calibrated,
            "noisy": noisy,
            "level_only": level_only,
            "level_inverted": level_inverted,
            "random": useless,
            "reversed": -good,
            "inflated_3x": inflated,
        },
    }


def _show(frame: pd.DataFrame, fmt: str = "{:.4f}") -> None:
    """Print a frame without the index, in a fixed ASCII width."""
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        print(frame.to_string(index=False, float_format=lambda v: fmt.format(v)))


if __name__ == "__main__":  # pragma: no cover - smoke test
    import time

    from prism.utils.logging import set_level

    set_level("WARNING")  # keep the binarisation/clip notices out of the table output
    t_start = time.perf_counter()
    print("=" * 96)
    print("prism.causal.evaluate -- smoke test against a KNOWN ground truth")
    print("=" * 96)

    D = _simulate_evaluation_data()
    y_, w_, tau_, prop_, mu_ = D["y"], D["w"], D["tau"], D["propensity"], D["mu_true"]
    n_ = D["n"]
    print(
        f"n={n_:,}  treated={int(w_.sum()):,}  true ATE={tau_.mean():.3f}  sd(tau)={D['sd_tau']:.3f}  "
        f"negative-effect rows={float((tau_ < 0).mean()):.1%}  e in [{D['e1'].min():.2f}, {D['e1'].max():.2f}]"
    )
    print()

    # ---------------------------------------------------------------- 1. ranking metrics -----
    order = [
        "perfect", "good", "calibrated", "noisy", "level_only", "level_inverted",
        "random", "reversed",
    ]
    rank_rows = []
    for key in order:
        s = D["cate"][key]
        rank_rows.append(
            {
                "cate": key,
                "qini": qini_score(y_, w_, s),
                "auuc": auuc(y_, w_, s),
                "uplift@20%": uplift_at_k(y_, w_, s, 0.2),
                "autoc": targeting_operator_characteristic(y_, w_, s).attrs["autoc"],
            }
        )
    rank_tbl = pd.DataFrame(rank_rows)
    print("ranking metrics (qini normalised by the proportional-mix reference curve)")
    _show(rank_tbl)
    print()

    q = {r["cate"]: r["qini"] for r in rank_rows}
    a = {r["cate"]: r["auuc"] for r in rank_rows}
    assert q["good"] > q["noisy"] > q["random"], (q["good"], q["noisy"], q["random"])
    assert abs(q["random"]) < 0.05, f"random qini should sit on the random line, got {q['random']}"
    assert q["perfect"] > max(q["good"], q["noisy"], q["calibrated"], q["random"]), q
    assert q["reversed"] < 0.0, f"an inverted ranking must score below random, got {q['reversed']}"
    assert a["good"] > a["noisy"] > a["random"], (a["good"], a["noisy"], a["random"])
    assert q["perfect"] <= 1.0 + 1e-9, "no ranking may beat the reference curve at the same arm mix"
    # The two level-only scores are a sign flip of each other and carry PROVABLY zero effect
    # information (mu0 uses x0/x3, tau uses x1/x2). An unadjusted contrast still hands them a
    # large number of either sign, purely from the arm mix along the ranking. This is the trap.
    assert abs(q["level_only"]) > 5.0 * abs(q["random"]), (q["level_only"], q["random"])
    assert abs(q["level_inverted"]) > 5.0 * abs(q["random"])
    assert q["level_inverted"] < q["reversed"], (q["level_inverted"], q["reversed"])
    print(
        f"the two level-only scores carry ZERO effect information (mu0 uses x0/x3, tau uses "
        f"x1/x2) yet score {q['level_only']:+.3f} and {q['level_inverted']:+.3f} -- against "
        f"{q['random']:+.3f} for an honestly uninformative score. The second is even ranked BELOW "
        f"the actively inverted model ({q['reversed']:+.3f}). An unadjusted contrast is measuring "
        f"the arm mix along the ranking, not the uplift."
    )
    print()

    # zero-control head: the first prefixes hold treated rows only
    tiny = qini_curve(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 1, 0, 0]), np.array([4.0, 3.0, 2.0, 1.0])
    )
    assert float(tiny.loc[0, "incremental"]) == 0.0
    assert float(tiny.loc[1, "incremental"]) == 0.0 and int(tiny.loc[1, "n_control"]) == 0
    assert float(tiny.loc[2, "incremental"]) == 0.0 and int(tiny.loc[2, "n_control"]) == 0
    assert np.isclose(float(tiny.loc[3, "incremental"]), 3.0 - 3.0 * 2.0 / 1.0)
    assert np.isfinite(tiny["incremental"]).all(), "N_c == 0 must not produce inf/nan"

    # cost turns the curve into a net-value curve whose peak is the profitable depth
    net = qini_curve(y_, w_, D["cate"]["good"], treatment_cost=1.0)
    gross = qini_curve(y_, w_, D["cate"]["good"], treatment_cost=0.0)
    assert net.attrs["net_value_curve"] and not gross.attrs["net_value_curve"]
    peak_net = int(net["incremental"].idxmax())
    assert peak_net < int(gross["incremental"].idxmax()), "charging cost must shorten the campaign"
    print(
        f"campaign depth at the peak of the curve: {gross['frac_targeted'][int(gross['incremental'].idxmax())]:.0%} "
        f"gross -> {net['frac_targeted'][peak_net]:.0%} once each offer costs 1.0 outcome unit"
    )

    # multi-arm input is binarised and logged, not silently mishandled
    rng_arm = np.random.default_rng(3)
    w_multi = np.where(w_ == 1, rng_arm.integers(1, 4, size=n_), 0)
    assert np.isclose(qini_score(y_, w_multi, D["cate"]["good"]), q["good"])
    cate_2d = np.column_stack([D["cate"]["good"] - 1.0, D["cate"]["good"], D["cate"]["good"] - 2.0])
    assert np.isclose(qini_score(y_, w_multi, cate_2d), q["good"])  # max across arms == good
    assert np.isclose(qini_score(y_, w_multi, cate_2d, arm=2), q["good"])
    print("multi-arm w and (n, n_arms-1) cate collapse to the documented two-arm contrast")
    print()

    # ---------------------------------------------------------------- 2. ground truth --------
    assert pehe(tau_, tau_) == 0.0, "PEHE of an exact estimate must be exactly zero"
    assert eps_ate(tau_, tau_) == 0.0
    assert np.isclose(pehe(tau_, tau_ + 1.0), 1.0) and np.isclose(eps_ate(tau_, tau_ + 1.0), 1.0)
    two_d_true = np.column_stack([tau_, tau_])
    two_d_est = np.column_stack([tau_ + 1.0, tau_ - 1.0])
    assert np.isclose(pehe(two_d_true, two_d_est), 1.0), "2-D PEHE pools across arms"
    assert np.isclose(eps_ate(two_d_true, two_d_est), 0.0), "pooled eps_ATE lets arms cancel"
    assert np.isclose(eps_ate(two_d_true, two_d_est, reduce="mean_abs"), 1.0), "mean_abs does not"
    pehe_good = pehe(tau_, D["cate"]["good"])
    pehe_noisy = pehe(tau_, D["cate"]["noisy"])
    pehe_rand = pehe(tau_, D["cate"]["random"])
    assert 0.0 == pehe(tau_, tau_) < pehe_good < pehe_noisy
    assert pehe_good < pehe_rand
    print(
        f"pehe: perfect={pehe(tau_, tau_):.4f}  good={pehe_good:.3f}  noisy={pehe_noisy:.3f}  "
        f"random={pehe_rand:.3f}"
    )
    # PEHE is a squared-error loss, so it rewards shrinkage: the zero-information N(0,1) score
    # beats the informative-but-over-dispersed one. Exactly why the leaderboard reports PEHE,
    # a ranking metric and a policy value side by side instead of crowning one of them.
    assert pehe_rand < pehe_noisy and q["noisy"] > q["random"], (pehe_rand, pehe_noisy)
    print(
        f"  note: the uninformative score has the BETTER pehe ({pehe_rand:.2f} < {pehe_noisy:.2f}) "
        f"because squared error punishes variance, while qini ranks it far behind "
        f"({q['random']:+.3f} vs {q['noisy']:+.3f}) -- no single metric is sufficient"
    )

    # policy_risk against the oracle over both potential outcomes
    po = np.column_stack([D["y0"], D["y1"]])
    oracle_pi = (po[:, 1] > po[:, 0]).astype(np.int64)
    assert policy_risk(po, oracle_pi) == 0.0, "the oracle has zero regret by construction"
    risk_all = policy_risk(po, np.ones(n_, dtype=np.int64))
    assert np.isclose(risk_all, float(np.maximum(-tau_, 0.0).mean()))
    assert risk_all > 0.0 and policy_risk(po, np.zeros(n_, dtype=np.int64)) > 0.0
    print(
        f"policy_risk: oracle=0.0000  treat_all={risk_all:.4f}  "
        f"treat_none={policy_risk(po, np.zeros(n_, dtype=np.int64)):.4f} (outcome units per customer)"
    )
    print()

    # ---------------------------------------------------------------- 3. deciles + TOC -------
    dec = uplift_by_decile(y_, w_, D["cate"]["good"])
    print("uplift by decile of predicted effect (decile 1 = HIGHEST predicted effect)")
    _show(dec[["decile", "n", "n_treated", "n_control", "observed_uplift", "predicted_uplift", "se"]])
    rho_dec = float(dec.attrs["spearman_decile_uplift"])
    assert rho_dec < -0.8, f"observed uplift must fall from decile 1 downward; rho={rho_dec}"
    assert float(dec.loc[0, "observed_uplift"]) > float(dec.loc[9, "observed_uplift"])
    assert (dec["n_treated"] > 0).all() and (dec["n_control"] > 0).all()
    dec_rand = uplift_by_decile(y_, w_, D["cate"]["random"])
    assert abs(float(dec_rand.attrs["spearman_decile_uplift"])) < 0.8
    print(f"spearman(decile, observed_uplift): good={rho_dec:+.3f}  "
          f"random={float(dec_rand.attrs['spearman_decile_uplift']):+.3f}")

    toc_good = targeting_operator_characteristic(y_, w_, D["cate"]["good"])
    assert np.isclose(float(toc_good["toc"].iloc[-1]), 0.0), "TOC(1) is identically zero"
    assert float(toc_good["toc"].iloc[9]) > 0.0, "the top decile must out-respond the average"
    assert toc_good.attrs["autoc"] > targeting_operator_characteristic(
        y_, w_, D["cate"]["random"]
    ).attrs["autoc"]
    print()

    # ---------------------------------------------------------------- 4. GATES ---------------
    g_good = gates(y_, w_, D["cate"]["good"], prop_, n_groups=5)
    g_rand = gates(y_, w_, D["cate"]["random"], prop_, n_groups=5)
    g_level = gates(y_, w_, D["cate"]["level_only"], prop_, n_groups=5)
    g_level_inv = gates(y_, w_, D["cate"]["level_inverted"], prop_, n_groups=5)
    print("GATES on the GOOD estimate (group 1 = LOWEST predicted effect; AIPW, Hajek group mu)")
    _show(g_good[["group", "n", "mean_predicted_cate", "gate", "se", "ci_low", "ci_high", "p_value"]])
    print(
        f"  top - bottom = {g_good.attrs['top_minus_bottom']:+.3f} "
        f"(se {g_good.attrs['top_minus_bottom_se']:.3f}, z {g_good.attrs['z']:.1f}, "
        f"p {g_good.attrs['p_value']:.2e})   monotone={g_good.attrs['monotone_increasing']}"
    )
    print()
    print("GATES on the RANDOM estimate -- the same table, no signal")
    _show(g_rand[["group", "n", "mean_predicted_cate", "gate", "se", "ci_low", "ci_high", "p_value"]])
    print(
        f"  top - bottom = {g_rand.attrs['top_minus_bottom']:+.3f} "
        f"(se {g_rand.attrs['top_minus_bottom_se']:.3f}, z {g_rand.attrs['z']:.1f}, "
        f"p {g_rand.attrs['p_value']:.3f})   monotone={g_rand.attrs['monotone_increasing']}"
    )
    print()

    assert g_good.attrs["monotone_increasing"], g_good["gate"].tolist()
    assert g_good.attrs["p_value"] < 1e-4, g_good.attrs["p_value"]
    assert g_good.attrs["top_minus_bottom"] > 0.0
    assert not g_rand.attrs["monotone_increasing"] or abs(g_rand.attrs["z"]) < 2.0
    assert g_rand.attrs["p_value"] > 0.05, g_rand.attrs["p_value"]
    assert abs(g_rand.attrs["z"]) < 2.0, g_rand.attrs["z"]
    assert np.isclose(float(g_good["top_minus_bottom_p"].iloc[0]), g_good.attrs["p_value"])
    # the headline claim of the module, tested rather than asserted in prose:
    assert g_level.attrs["p_value"] > 0.05, g_level.attrs["p_value"]
    assert g_level_inv.attrs["p_value"] > 0.05, g_level_inv.attrs["p_value"]
    assert np.isclose(g_level.attrs["p_value"], g_level_inv.attrs["p_value"])
    print(
        f"THE TEST: Qini gives the level-only score {q['level_only']:+.3f} and its sign flip "
        f"{q['level_inverted']:+.3f} -- two very different verdicts on the same zero information. "
        f"GATES returns p={g_level.attrs['p_value']:.3f} for both: no heterogeneity, correctly, "
        f"because the propensity is in the estimator and the arm mix cannot leak into it."
    )
    # AIPW recovers the true ATE; the unadjusted difference in means does not.
    ate_aipw = float(g_good.attrs["ate_aipw"])
    ate_naive = float(y_[w_ == 1].mean() - y_[w_ == 0].mean())
    assert abs(ate_aipw - float(tau_.mean())) < abs(ate_naive - float(tau_.mean()))
    print(
        f"ATE: truth={tau_.mean():.3f}  AIPW={ate_aipw:.3f}  naive difference in means="
        f"{ate_naive:.3f} (confounding, uncorrected)"
    )
    print()

    # ---------------------------------------------------------------- 5. BLP ----------------
    blp_rows = []
    for key in ["perfect", "calibrated", "good", "noisy", "inflated_3x", "random"]:
        b = blp_calibration(y_, w_, D["cate"][key], prop_)
        blp_rows.append(
            {
                "cate": key,
                "beta1_ate": float(b.loc[0, "coefficient"]),
                "beta2_slope": float(b.loc[1, "coefficient"]),
                "se": float(b.loc[1, "se"]),
                "t": float(b.loc[1, "t"]),
                "ci_low": float(b.loc[1, "ci_low"]),
                "ci_high": float(b.loc[1, "ci_high"]),
                "calibrated": bool(b.attrs["calibrated"]),
            }
        )
    blp_tbl = pd.DataFrame(blp_rows)
    print("BLP calibration -- beta1 is the ATE, beta2 is the calibration slope (1.0 = calibrated)")
    _show(blp_tbl)
    blp_map = {r["cate"]: r for r in blp_rows}
    print(
        f"  'calibrated' is 'good' shrunk by {D['shrinkage']:.3f} toward its mean: identical "
        f"ranking (qini {q['calibrated']:.3f} vs {q['good']:.3f}), slope moved "
        f"{blp_map['good']['beta2_slope']:.2f} -> {blp_map['calibrated']['beta2_slope']:.2f}"
    )
    print()

    assert abs(blp_map["perfect"]["beta2_slope"] - 1.0) < 0.15, blp_map["perfect"]
    assert abs(blp_map["calibrated"]["beta2_slope"] - 1.0) < 0.20, blp_map["calibrated"]
    assert blp_map["calibrated"]["calibrated"], "the shrunk estimate's CI must contain 1"
    assert blp_map["inflated_3x"]["beta2_slope"] < 0.6, blp_map["inflated_3x"]
    assert np.isclose(
        blp_map["inflated_3x"]["beta2_slope"], blp_map["calibrated"]["beta2_slope"] / 3.0, atol=1e-6
    ), "inflating the spread 3x must divide the slope by exactly 3"
    assert not blp_map["inflated_3x"]["calibrated"]
    assert blp_map["good"]["beta2_slope"] < 1.0, "an unshrunk noisy score is over-dispersed"
    assert abs(blp_map["random"]["beta2_slope"]) < 3.0 * blp_map["random"]["se"], blp_map["random"]
    for r in blp_rows:
        assert abs(r["beta1_ate"] - float(tau_.mean())) < 0.6, r  # beta1 is the AIPW ATE
    # Qini cannot see the inflation at all -- that is the point of running both.
    assert np.isclose(q["calibrated"], qini_score(y_, w_, D["cate"]["inflated_3x"]), atol=1e-12)
    print(
        "inflating the spread 3x leaves the Qini bit-identical "
        f"({q['calibrated']:.6f}) and cuts the BLP slope to "
        f"{blp_map['inflated_3x']['beta2_slope']:.3f} -- Qini is blind to calibration"
    )
    print()

    # ---------------------------------------------------------------- 6. policy value --------
    policy = (tau_ > 0.0).astype(np.int64)
    v_true = float(np.where(policy == 1, D["y1"], D["y0"]).mean())
    pv_rows = []
    for meth, mu_arg in (("ipw", None), ("snipw", None), ("dr", mu_)):
        res = policy_value(
            y_, w_, policy, prop_, method=meth, mu_hat=mu_arg, n_boot=500, random_state=11
        )
        pv_rows.append(
            {
                "method": res["method"],
                "value": res["value"],
                "truth": v_true,
                "error": res["value"] - v_true,
                "se": res["se"],
                "z": (res["value"] - v_true) / res["se"],
                "ci_low": res["ci_low"],
                "ci_high": res["ci_high"],
                "ess": res["ess"],
            }
        )
    pv_tbl = pd.DataFrame(pv_rows)
    print(f"policy value of the known policy 'treat where tau > 0'   (finite-sample truth {v_true:.4f})")
    _show(pv_tbl)
    for r in pv_rows:
        assert abs(r["z"]) < 3.0, f"{r['method']} is {r['z']:.2f} SE from the known truth"
        assert r["ci_low"] <= v_true <= r["ci_high"], r
    dr_row = pv_rows[2]
    assert dr_row["se"] < pv_rows[0]["se"], "DR must be more precise than raw IPW"
    print(
        f"  DR standard error {dr_row['se']:.4f} vs IPW {pv_rows[0]['se']:.4f} "
        f"({pv_rows[0]['se'] / dr_row['se']:.1f}x tighter, from absorbing the outcome level)"
    )

    # the DR fallback must announce itself
    fb = policy_value(y_, w_, policy, prop_, method="dr", mu_hat=None, n_boot=50, random_state=1)
    assert fb["method"] == DR_FALLBACK_METHOD and fb["fallback"] is True, fb["method"]
    plain = policy_value(y_, w_, policy, prop_, method="snipw", n_boot=50, random_state=1)
    assert np.isclose(fb["value"], plain["value"]) and fb["method"] != plain["method"]
    print(f"  method='dr' without mu_hat returns method={fb['method']!r}, never a silent downgrade")

    # 1-D propensity (own-arm) agrees with the full matrix, since unmatched rows carry weight 0
    own = prop_[np.arange(n_), w_]
    pv_1d = policy_value(y_, w_, policy, own, method="dr", mu_hat=mu_, n_boot=0)
    pv_2d = policy_value(y_, w_, policy, prop_, method="dr", mu_hat=mu_, n_boot=0)
    assert np.isclose(pv_1d["value"], pv_2d["value"]), (pv_1d["value"], pv_2d["value"])

    # determinism
    r1 = policy_value(y_, w_, policy, prop_, method="dr", mu_hat=mu_, n_boot=200, random_state=99)
    r2 = policy_value(y_, w_, policy, prop_, method="dr", mu_hat=mu_, n_boot=200, random_state=99)
    assert (r1["value"], r1["se"], r1["ci_low"], r1["ci_high"]) == (
        r2["value"], r2["se"], r2["ci_low"], r2["ci_high"]
    ), "same seed must give bit-identical output"
    print("  two runs at the same seed are bit-identical")
    print()

    # ---------------------------------------------------------------- 7. leaderboard ---------
    board = compare_learners(
        {
            k: D["cate"][k]
            for k in ["perfect", "good", "calibrated", "noisy", "level_only", "random"]
        },
        true_cate=tau_,
        y=y_,
        w=w_,
        propensity=prop_,
        n_boot=200,
        random_state=5,
    )
    print(f"learner leaderboard (sorted by {board.attrs['sorted_by']})")
    _show(
        board[
            ["rank", "learner", "policy_value", "qini", "auuc", "pehe", "eps_ate",
             "spearman_true", "blp_slope", "gates_p"]
        ]
    )
    print(f"  ordering rule: {board.attrs['ordering_rule']}")
    print()

    assert list(board.columns) == list(LEADERBOARD_COLUMNS)
    assert len(board) == 6 and board["rank"].tolist() == [1, 2, 3, 4, 5, 6]
    assert board["policy_value"].notna().all() and board.attrs["sorted_by"] == "policy_value"
    assert board["policy_value"].is_monotonic_decreasing
    by_name = board.set_index("learner")
    assert by_name.loc["perfect", "pehe"] == 0.0 and by_name.loc["perfect", "eps_ate"] == 0.0
    assert by_name.loc["good", "pehe"] < by_name.loc["noisy", "pehe"]
    assert by_name.loc["good", "qini"] > by_name.loc["noisy", "qini"] > by_name.loc["random", "qini"]
    assert by_name.loc["random", "gates_p"] > 0.05 and by_name.loc["good", "gates_p"] < 1e-4
    assert by_name.loc["perfect", "spearman_true"] == 1.0
    # the trap is visible in the leaderboard itself: a large-magnitude qini next to a null GATES
    assert abs(by_name.loc["level_only", "qini"]) > 0.05
    assert by_name.loc["level_only", "gates_p"] > 0.05
    assert abs(by_name.loc["level_only", "spearman_true"]) < 0.05
    assert by_name.loc["perfect", "policy_value"] == board["policy_value"].max()
    # stable schema when only part of the evidence is available
    thin_board = compare_learners({k: D["cate"][k] for k in ["good", "random"]})
    assert list(thin_board.columns) == list(LEADERBOARD_COLUMNS)
    assert thin_board[["pehe", "qini", "policy_value"]].isna().all().all()
    assert thin_board.attrs["sorted_by"] == "learner"
    print("leaderboard keeps a stable schema when true_cate / y / w / propensity are absent")

    elapsed = time.perf_counter() - t_start
    print()
    print(
        "Qini ranks, GATES proves, BLP calibrates, policy value prices -- a model needs all four, "
        "and the level-only column above is why."
    )
    print(f"evaluate.py OK  ({elapsed:.1f}s, n={n_:,}, 13 public functions, {len(board)} learners scored)")
