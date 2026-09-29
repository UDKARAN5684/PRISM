"""Refutation and sensitivity analysis: the module whose job is to break the finding.

A validation metric measures *fit*. It says nothing about whether the number is causal. Every
estimator in :mod:`prism.causal` will happily return a confident treatment effect from data where
the treatment does nothing, provided the confounding is arranged helpfully enough. This module is
the adversary: it perturbs the data in ways that should destroy a genuine effect, perturbs it in
ways that should *not*, and -- where perturbation is not enough -- quantifies exactly how strong an
unmeasured confounder would have to be before the conclusion flips.

Two distinct families live here, and conflating them is a common error.

**1. Falsification tests** (:func:`placebo_treatment`, :func:`random_common_cause`,
:func:`subset_refuter`, :func:`add_unobserved_confounder`). These re-run the *whole estimation
pipeline* on modified data and compare the answer. They can only ever find a problem; passing all
four is weak evidence, because the modifications are ones the analyst chose. Their real value is
catching the embarrassing failures -- a pipeline that "finds" an effect after the treatment has
been permuted is broken, and this is how you learn that before a reviewer does.

**2. Sensitivity analyses** (:func:`e_value`, :func:`rosenbaum_bounds`,
:func:`sensitivity_contour`). These accept that unconfoundedness is unverifiable and ask the only
answerable question instead: *how strong would the violation have to be?* They do not refit
anything expensive; they map the observed estimate onto the space of possible hidden biases.

Estimator maths
---------------
Write ``tau_a = E[Y(a) - Y(0)]`` for the ATE of arm ``a`` against control, and ``tau_hat`` for the
sample mean of a fitted learner's CATE column,

.. math:: \\hat{\\tau}_a = \\frac{1}{n} \\sum_i \\hat{\\tau}_a(X_i).

Every falsification test in this module is a comparison of ``tau_hat`` computed on the original
data against ``tau_hat`` recomputed after an intervention on the data:

===========================  ==============================================  =====================
test                         intervention                                    passes when
===========================  ==============================================  =====================
placebo treatment            ``W -> pi(W)``, a uniform random permutation     effect collapses to 0
random common cause          ``X -> [X, Z]``, ``Z ~ N(0,1)`` independent      estimate is unchanged
subset                       ``(X,W,Y) -> `` random ``f * n`` subsample       estimate is stable
unobserved confounder        ``Y -> Y + beta_u U``, ``U`` correlated with W   shift is as predicted
===========================  ==============================================  =====================

The two sensitivity analyses rest on published closed forms.

*E-value* (VanderWeele & Ding, 2017). For an effect on the risk-ratio scale ``RR >= 1``, the
minimum joint confounder-treatment and confounder-outcome risk ratio that could produce ``RR`` from
a true null is

.. math:: E = RR + \\sqrt{RR (RR - 1)}.

For ``RR < 1`` the reciprocal is taken first. ``E = 1`` means no confounding is needed at all.

*Rosenbaum bounds* (Rosenbaum, 2002). In a matched pair, hidden bias of magnitude ``Gamma``
means two units with identical observed covariates may differ in their odds of treatment by up to a
factor ``Gamma``. The sign of each pair difference is then Bernoulli with success probability
bounded in ``[1/(1+Gamma), Gamma/(1+Gamma)]``, which bounds the Wilcoxon signed-rank statistic
``W = sum_{d_i > 0} rank(|d_i|)`` between two binomial-mixture null distributions and so bounds the
p-value in an interval ``[p_lower, p_upper]``.

*Omitted variable bias contour* (Cinelli & Hazlett, 2020; Austen plot). In a partially linear model
the bias from omitting a confounder ``U`` is a function of two partial ``R^2`` values only:

.. math::

   |\\text{bias}| = \\hat{se}(\\hat\\tau) \\, \\sqrt{df} \\,
                    \\sqrt{\\frac{R^2_{Y \\sim U \\mid D, X} \\; R^2_{D \\sim U \\mid X}}
                                 {1 - R^2_{D \\sim U \\mid X}}}.

Setting both partial ``R^2`` equal and solving for the value that drives the estimate to zero gives
the *robustness value*

.. math:: RV = \\tfrac{1}{2}\\left(\\sqrt{f^4 + 4 f^2} - f^2\\right), \\qquad f = |t| / \\sqrt{df},

and the same algebra, applied to each observed covariate's own partial ``R^2``, yields the multiple
``k`` by which a confounder would have to exceed that covariate to erase the effect. That multiple
is the number that makes the contour plot an argument rather than a decoration.

What this module deliberately does not claim
--------------------------------------------
- Passing every test is **not** evidence of a causal effect. It is the absence of four specific
  kinds of evidence against one. The docstrings say so individually as well.
- The omitted-variable-bias formula is exact for a *linear* model. PRISM's estimates come from
  gradient-boosted learners, so :func:`sensitivity_contour` applies it as a linear approximation
  around the partialling-out (Robinson / DML) representation of the estimate, and says so in the
  returned frame's ``attrs`` and in every log line. A sensitivity analysis that hides its own
  approximation is worse than none.
- The E-value is defined on the **risk-ratio scale**. A difference-scale estimate is converted with
  a published approximation, and :func:`e_value` reports loudly which scale it assumed and that the
  conversion is approximate. It will not quietly return an authoritative-looking number computed on
  the wrong scale.

Performance
-----------
``SPEC_PERF`` section 8 is binding here: refutation refits models, so defaults are small
(``n_sim <= 20``), the built-in default estimator is a deliberately cheap 100-tree T-learner
(:func:`cheap_cate_factory`), every refuter logs exactly how many simulations it ran, and
:func:`run_refutation_suite` fits the original model **once** and passes the value down rather than
letting each refuter refit it.

Public surface
--------------
:class:`RefutationResult`, :func:`placebo_treatment`, :func:`random_common_cause`,
:func:`subset_refuter`, :func:`add_unobserved_confounder`, :func:`e_value`,
:func:`rosenbaum_bounds`, :func:`sensitivity_contour`, :func:`run_refutation_suite`,
:func:`cheap_cate_factory`.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from prism.data.schema import ARM_NAMES, FEATURES, N_ARMS
from prism.utils.logging import get_logger
from prism.utils.optional import HAS_LIGHTGBM, HAS_XGBOOST, best_gbm
from prism.utils.seeds import as_rng, spawn_rngs

__all__ = [
    "RefutationResult",
    "placebo_treatment",
    "random_common_cause",
    "subset_refuter",
    "add_unobserved_confounder",
    "e_value",
    "rosenbaum_bounds",
    "sensitivity_contour",
    "run_refutation_suite",
    "cheap_cate_factory",
]

LOGGER = get_logger("causal.refute")

#: Guard for every denominator in this module (SPEC_PERF section 9).
_EPS: float = 1e-12
#: Partial R-squared values are clipped below 1 so ``1 - R2`` never vanishes.
_R2_MAX: float = 0.999
#: Default relative tolerance for "the estimate barely moved" (:func:`random_common_cause`).
_DEFAULT_REL_TOL: float = 0.10
#: Minimum rows an arm needs before :class:`_CheapTLearner` fits it its own model.
_MIN_ROWS_PER_ARM: int = 10
#: Threads per gradient-boosting fit inside this module.
#:
#: SPEC_PERF section 9 defaults ``n_jobs`` to ``-1``, and that is right for one large fit. It is
#: wrong here. Refutation's unit of work is a *small* model fit repeated dozens of times, and at
#: that size the boosting library spends more time standing up and tearing down its thread pool
#: than it does boosting: measured at 0.45 s per fit with ``n_jobs=-1`` against 0.046 s with
#: ``n_jobs=1`` on 3000 x 5 with 100 trees, a 10x difference. The parallel dimension that pays
#: here is the simulation loop, not the tree builder. Every entry point exposes ``n_jobs`` so a
#: caller with genuinely large refutation data can override it.
_GBM_N_JOBS: int = 1

#: Column order of the frame returned by :func:`run_refutation_suite`.
SUITE_COLUMNS: tuple[str, ...] = (
    "name",
    "original",
    "refuted",
    "p_value",
    "passed",
    "detail",
    "n_sim",
    "seconds",
)

#: A zero-argument callable returning a fresh, unfitted CATE learner.
EstimatorFactory = Callable[[], Any]


def _gbm_accepts_n_jobs() -> bool:
    """Whether the backend :func:`~prism.utils.optional.best_gbm` will pick takes ``n_jobs``.

    LightGBM and XGBoost do; the scikit-learn ``HistGradientBoosting`` fallback does not, and
    ``best_gbm`` forwards unknown keys straight to the constructor, so passing ``n_jobs``
    blindly would raise ``TypeError`` on a machine without either accelerator.
    """
    return bool(HAS_LIGHTGBM or HAS_XGBOOST)


def _cheap_gbm(task: str, *, seed: int, n_estimators: int, learning_rate: float,
               max_depth: int, n_jobs: int, min_samples_leaf: int = 10) -> Any:
    """``best_gbm`` with the thread count applied only when the backend understands it."""
    kw: dict[str, Any] = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "random_state": seed,
    }
    if _gbm_accepts_n_jobs():
        kw["n_jobs"] = int(n_jobs)
    return best_gbm(task, **kw)


# ======================================================================================
# result container
# ======================================================================================
@dataclass
class RefutationResult:
    """One refutation or sensitivity test, in the form the leaderboard consumes.

    Parameters
    ----------
    name : str
        Test identifier, e.g. ``"placebo_treatment"``.
    original : float
        The ATE estimated on the unmodified data.
    refuted : float
        The comparison quantity produced by the test. For the falsification tests this is the ATE
        after the intervention; for the sensitivity analyses it is the headline sensitivity
        statistic (E-value, robustness value, or the Rosenbaum breakdown ``Gamma``). The ``detail``
        string always says which, because a column that means different things in different rows is
        a trap.
    p_value : float or None
        Test-specific p-value, or ``None`` where the test does not define one. Where present, the
        conventional reading holds: **small means the original estimate is unusual under the
        test's null**, which for a placebo test is the *good* outcome.
    passed : bool
        ``True`` when the finding survived this test. Each function's docstring states its own
        pass criterion explicitly; none of them are universal conventions.
    detail : str
        Human-readable evidence -- the numbers a reviewer would ask for next.
    n_sim : int, default 0
        Number of model refits actually performed (SPEC_PERF section 8 requires this be reported).
    seconds : float, default 0.0
        Wall-clock cost of the test.
    extra : dict, optional
        Machine-readable extras, e.g. the placebo draws or the predicted bias. Never required by
        the suite; present so callers can assert on exact numbers instead of parsing ``detail``.
    """

    name: str
    original: float
    refuted: float
    p_value: float | None
    passed: bool
    detail: str
    n_sim: int = 0
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Return the tidy-row form used by :func:`run_refutation_suite`.

        Returns
        -------
        dict
            Keys exactly :data:`SUITE_COLUMNS`. ``extra`` is intentionally omitted so the frame
            stays CSV-writable.
        """
        return {
            "name": self.name,
            "original": float(self.original),
            "refuted": float(self.refuted),
            "p_value": None if self.p_value is None else float(self.p_value),
            "passed": bool(self.passed),
            "detail": str(self.detail),
            "n_sim": int(self.n_sim),
            "seconds": round(float(self.seconds), 3),
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        p = "n/a" if self.p_value is None else f"{self.p_value:.4f}"
        flag = "PASS" if self.passed else "FAIL"
        return f"[{flag}] {self.name}: {self.original:+.4f} -> {self.refuted:+.4f} (p={p}) | {self.detail}"


# ======================================================================================
# internal data helpers
# ======================================================================================
def _to_frame(X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    """Wrap ``X`` in a DataFrame without copying when it already is one."""
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D; got shape {arr.shape}")
    return pd.DataFrame(arr, columns=[f"x{j}" for j in range(arr.shape[1])])


def _design(
    X: np.ndarray | pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Numeric design matrix for this module's *internal* estimators only.

    One-hot encodes object/categorical/bool columns and median-imputes numerics, so the built-in
    cheap learner and the OLS benchmark regressions can run on a raw panel slice. Learners supplied
    by the caller through ``estimator_factory`` never see this -- they receive ``X`` untouched.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Covariates.
    columns : sequence of str, optional
        Column layout to conform to (from a previous call), so train and predict align.

    Returns
    -------
    (numpy.ndarray, list of str)
        ``(n, p)`` float64 matrix and its column names.
    """
    df = _to_frame(X).copy()
    cat_cols = [
        c
        for c in df.columns
        if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == bool
    ]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False, dtype=np.float64)
    df = df.apply(pd.to_numeric, errors="coerce")
    med = df.median(numeric_only=True)
    df = df.fillna(med).fillna(0.0)
    if columns is not None:
        df = df.reindex(columns=list(columns), fill_value=0.0)
    arr = np.ascontiguousarray(df.to_numpy(dtype=np.float64))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, list(df.columns)


def _take_rows(X: np.ndarray | pd.DataFrame, idx: np.ndarray) -> np.ndarray | pd.DataFrame:
    """Row-subset ``X`` preserving its container type."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx]
    return np.asarray(X)[idx]


def _append_column(
    X: np.ndarray | pd.DataFrame, col: np.ndarray, name: str
) -> np.ndarray | pd.DataFrame:
    """Append one numeric column to ``X`` preserving its container type."""
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        out[name] = np.asarray(col, dtype=np.float64)
        return out
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.column_stack([arr, np.asarray(col, dtype=np.float64)])


def _coerce(
    X: np.ndarray | pd.DataFrame, w: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray | pd.DataFrame, np.ndarray, np.ndarray]:
    """Validate shapes and normalise ``w`` to int64 and ``y`` to float64."""
    w_arr = np.asarray(w).ravel()
    if w_arr.dtype == bool:
        w_arr = w_arr.astype(np.int64)
    w_arr = np.rint(np.asarray(w_arr, dtype=np.float64)).astype(np.int64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = len(y_arr)
    if len(w_arr) != n:
        raise ValueError(f"w and y must be the same length; got {len(w_arr)} and {n}")
    n_x = X.shape[0] if hasattr(X, "shape") else len(X)
    if n_x != n:
        raise ValueError(f"X has {n_x} rows but y has {n}")
    if w_arr.min() < 0:
        raise ValueError("w must be non-negative arm indices with 0 = control")
    return X, w_arr, y_arr


def _feature_names(X: np.ndarray | pd.DataFrame, feature_names: Sequence[str] | None) -> list[str]:
    """Best available column names for ``X``."""
    if feature_names is not None:
        return [str(c) for c in feature_names]
    if isinstance(X, pd.DataFrame):
        return [str(c) for c in X.columns]
    p = int(np.asarray(X).reshape(len(X), -1).shape[1])
    return [f"x{j}" for j in range(p)]


def _ate_from_learner(model: Any, X: np.ndarray | pd.DataFrame, target_arm: int) -> float:
    """Mean predicted CATE for ``target_arm`` -- this module's definition of "the ATE".

    Parameters
    ----------
    model : object
        A fitted learner exposing ``predict_cate(X) -> (n, n_arms - 1)``.
    X : numpy.ndarray or pandas.DataFrame
        Rows to average the CATE over. This is the *target population*: every refuter states which
        population it uses, because averaging over a different one is a different estimand.
    target_arm : int
        Arm index (>= 1); column ``target_arm - 1`` of ``predict_cate``.

    Returns
    -------
    float
        ``mean_i tau_hat_{target_arm}(X_i)``.
    """
    if target_arm < 1:
        raise ValueError(f"target_arm must be >= 1 (0 is control); got {target_arm}")
    fn = getattr(model, "predict_cate", None) or getattr(model, "predict", None)
    if fn is None:
        raise TypeError(f"{type(model).__name__} exposes neither predict_cate nor predict")
    arr = np.asarray(fn(X), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"predict_cate must return a 2-D array; got shape {arr.shape}")
    col = target_arm - 1
    if col >= arr.shape[1]:
        raise ValueError(
            f"target_arm={target_arm} needs column {col} but predict_cate returned "
            f"{arr.shape[1]} column(s)"
        )
    val = float(np.nanmean(arr[:, col]))
    if not np.isfinite(val):
        raise ValueError("the learner produced a non-finite CATE")
    return val


def _fit_and_ate(
    estimator_factory: EstimatorFactory,
    X: np.ndarray | pd.DataFrame,
    w: np.ndarray,
    y: np.ndarray,
    *,
    target_arm: int = 1,
    predict_X: np.ndarray | pd.DataFrame | None = None,
    propensity: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Build a fresh learner, fit it, and return its mean CATE.

    The learner is created by ``estimator_factory()`` on every call, which is the whole point: a
    refuter that reused a fitted model would be measuring nothing.
    """
    model = estimator_factory()
    kw: dict[str, Any] = {}
    if propensity is not None:
        kw["propensity"] = propensity
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight
    try:
        model.fit(X, w, y, **kw)
    except TypeError:
        # A learner that does not accept the keyword-only extras still satisfies the
        # (X, w, y) core of the BaseCATELearner contract.
        model.fit(X, w, y)
    return _ate_from_learner(model, X if predict_X is None else predict_X, target_arm)


def _central_mass(
    mean: float,
    sd: float,
    q_lo: float,
    q_hi: float,
    *,
    n_sim: int,
    alpha: float,
    interval_method: str,
    caller: str,
) -> tuple[float, float, str]:
    """Central ``1 - alpha`` mass of a simulation distribution, robust at small ``S``.

    Estimator maths
    ---------------
    The obvious choice -- the empirical quantiles ``[q_{alpha/2}, q_{1-alpha/2}]`` of the draws --
    degenerates when a refuter can only afford ten or twenty simulations: at ``S = 10`` those
    quantiles *are* the minimum and maximum, so "inside the central mass" becomes "inside the
    observed range", which a value drawn from the very same distribution falls outside with
    probability ``2 / (S + 1)``, about 18%. A criterion with an 18% false-alarm rate built into
    its arithmetic is not a criterion.

    The default instead uses the Student **prediction interval for one further draw**,

    .. math:: \\bar{x} \\;\\pm\\; t_{S-1,\\,1-\\alpha/2}\\; s \\sqrt{1 + \\tfrac{1}{S}},

    which has the nominal coverage at any ``S`` because it borrows strength from the mean and sd
    of all the draws rather than from two order statistics.

    Parameters
    ----------
    mean, sd : float
        Mean and sample standard deviation (``ddof=1``) of the simulation draws.
    q_lo, q_hi : float
        The empirical quantiles, used when ``interval_method`` selects them.
    n_sim : int, keyword-only
        Number of draws ``S``.
    alpha : float, keyword-only
        One minus the coverage.
    interval_method : {"auto", "prediction", "quantile"}, keyword-only
        ``"auto"`` picks ``"prediction"`` when ``n_sim < 40``.
    caller : str, keyword-only
        Name used in the warning when a degenerate ``n_sim`` forces a fallback.

    Returns
    -------
    (float, float, str)
        Lower bound, upper bound, and the method actually used.
    """
    if interval_method not in {"auto", "prediction", "quantile"}:
        raise ValueError(
            f"interval_method must be auto/prediction/quantile; got {interval_method!r}"
        )
    method = interval_method
    if method == "auto":
        method = "prediction" if n_sim < 40 else "quantile"
    if method == "prediction":
        if n_sim < 2:
            LOGGER.warning(
                "%s: n_sim=%d leaves no spread to build a prediction interval from, falling "
                "back to the (degenerate) empirical quantiles", caller, n_sim,
            )
            return float(q_lo), float(q_hi), "quantile"
        half = float(
            stats.t.ppf(1.0 - alpha / 2.0, df=n_sim - 1) * sd * np.sqrt(1.0 + 1.0 / n_sim)
        )
        return float(mean - half), float(mean + half), "prediction"
    return float(q_lo), float(q_hi), "quantile"


def _rel_change(new: float, old: float, *, scale: float) -> float:
    """Relative movement of an estimate, with an absolute floor so a near-null original is safe.

    ``|new - old| / max(|old|, floor)`` where ``floor = 0.05 * scale`` and ``scale`` is typically
    ``sd(y)``. Without the floor a true effect of 0.001 would make any movement look catastrophic,
    which is how a refuter turns into a random number generator.
    """
    floor = max(0.05 * float(scale), _EPS)
    return float(abs(new - old) / max(abs(old), floor))


# ======================================================================================
# the deliberately cheap default learner (SPEC_PERF section 8)
# ======================================================================================
class _CheapTLearner:
    """A 100-tree T-learner: the cheap estimator refutation is allowed to refit dozens of times.

    ``SPEC_PERF`` section 8 requires that the estimator used *inside* refutation be cheap -- "a
    T-learner with 100 trees, not the full forest". This class is exactly that, and it is
    deliberately self-contained so :mod:`prism.causal.refute` can be run and smoke-tested without
    importing the learner zoo (and so a change to a production learner can never silently change
    what the refuters mean).

    Estimator maths
    ---------------
    Fit one outcome model per arm on that arm's rows only,

    .. math:: \\hat\\mu_a(x) = \\widehat{E}[Y \\mid X = x, A = a],

    and contrast against control:

    .. math:: \\hat\\tau_a(x) = \\hat\\mu_a(x) - \\hat\\mu_0(x).

    The T-learner is not doubly robust and it is not efficient. It is unbiased for the ATE whenever
    the per-arm regressions are consistent, it costs ``K`` gradient-boosting fits, and its errors do
    not depend on a propensity model -- which makes it a defensible neutral yardstick for a test
    whose subject is the *data*, not the estimator.

    Parameters
    ----------
    n_arms : int, optional
        Number of arms including control. Inferred from ``w`` at fit time when ``None``.
    n_estimators : int, default 100
        Trees per arm model.
    learning_rate : float, default 0.1
        Boosting rate; paired with 100 trees this is a deliberately under-tuned, fast setting.
    max_depth : int, default 4
        Depth cap, to keep a refit in the tens of milliseconds.
    n_jobs : int, default 1
        Threads per boosting fit. One is the right default at refutation sizes; see
        :data:`_GBM_N_JOBS` for the measurement behind that.
    random_state : int or numpy.random.Generator or None, optional
        Seed. Fixed seeds make the whole refutation suite bit-reproducible.

    Attributes
    ----------
    n_arms : int
        Part of the ``BaseCATELearner`` contract.
    """

    def __init__(
        self,
        n_arms: int | None = None,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        n_jobs: int = _GBM_N_JOBS,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_arms: int = int(n_arms) if n_arms is not None else int(N_ARMS)
        self._n_arms_given = n_arms is not None
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.n_jobs = int(n_jobs)
        self.random_state = random_state
        # Drawn once, at construction, so the per-arm seeds do not depend on the order in which
        # the arm models happen to be fitted.
        self._base_seed = int(as_rng(random_state).integers(0, 2**31 - 1))
        self.models_: dict[int, Any] = {}
        self.pooled_: Any = None
        self.columns_: list[str] | None = None
        self.fallback_arms_: list[int] = []

    def _new_model(self, offset: int) -> Any:
        seed = int((self._base_seed + offset) % (2**31 - 1))
        return _cheap_gbm(
            "regression",
            seed=seed,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
        )

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        w: np.ndarray,
        y: np.ndarray,
        *,
        propensity: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> _CheapTLearner:
        """Fit one outcome model per arm.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Covariates.
        w : numpy.ndarray
            Arm index, 0 = control.
        y : numpy.ndarray
            Outcome, higher is better.
        propensity : numpy.ndarray, optional
            Accepted for interface compatibility and ignored: a T-learner uses no propensity.
        sample_weight : numpy.ndarray, optional
            Per-row weights forwarded to each arm model.

        Returns
        -------
        _CheapTLearner
        """
        _, w_arr, y_arr = _coerce(X, w, y)
        Z, cols = _design(X)
        self.columns_ = cols
        if not self._n_arms_given:
            self.n_arms = int(max(int(w_arr.max()) + 1, 2))
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).ravel()

        self.pooled_ = self._new_model(0)
        self.pooled_.fit(Z, y_arr, **({} if sw is None else {"sample_weight": sw}))

        self.models_ = {}
        self.fallback_arms_ = []
        for a in range(self.n_arms):
            m = w_arr == a
            if int(m.sum()) < _MIN_ROWS_PER_ARM:
                self.fallback_arms_.append(a)
                self.models_[a] = self.pooled_
                continue
            mod = self._new_model(a + 1)
            mod.fit(Z[m], y_arr[m], **({} if sw is None else {"sample_weight": sw[m]}))
            self.models_[a] = mod
        if self.fallback_arms_:
            LOGGER.warning(
                "_CheapTLearner: arms %s had < %d rows and fall back to the pooled model "
                "(their CATE is therefore shrunk toward 0)",
                self.fallback_arms_,
                _MIN_ROWS_PER_ARM,
            )
        return self

    def predict_cate(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return ``(n, n_arms - 1)`` CATE estimates, one column per non-control arm."""
        if self.columns_ is None:
            raise RuntimeError("_CheapTLearner must be fitted before predict_cate")
        Z, _ = _design(X, columns=self.columns_)
        mu0 = np.asarray(self.models_[0].predict(Z), dtype=np.float64)
        out = np.empty((Z.shape[0], self.n_arms - 1), dtype=np.float64)
        for a in range(1, self.n_arms):
            out[:, a - 1] = np.asarray(self.models_[a].predict(Z), dtype=np.float64) - mu0
        return out

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Alias of :meth:`predict_cate`, per the ``BaseCATELearner`` contract."""
        return self.predict_cate(X)


def cheap_cate_factory(
    n_arms: int | None = None,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 4,
    n_jobs: int = _GBM_N_JOBS,
    random_state: int | np.random.Generator | None = 0,
) -> EstimatorFactory:
    """Build the zero-argument factory the refuters expect, wrapping the cheap T-learner.

    An ``estimator_factory`` must return a **fresh, unfitted** learner every call. This helper is
    the reference implementation and the default used by :func:`run_refutation_suite` when the
    caller has nothing cheaper to offer.

    Parameters
    ----------
    n_arms : int, optional
        Arms including control; inferred from the data when ``None``.
    n_estimators, learning_rate, max_depth, n_jobs : optional
        Forwarded to :class:`_CheapTLearner`.
    random_state : int or numpy.random.Generator or None, default 0
        Seed for every learner the factory produces. A **fixed integer** is required for the
        refutation suite to be bit-reproducible; ``None`` makes each refit independent and the
        results irreproducible, which the refuters cannot detect or warn about.

    Returns
    -------
    callable
        Zero-argument callable returning an unfitted learner.

    Examples
    --------
    >>> factory = cheap_cate_factory(n_arms=2, random_state=7)
    >>> learner = factory()
    >>> learner.n_arms
    2
    """

    def _factory() -> _CheapTLearner:
        return _CheapTLearner(
            n_arms,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    return _factory


# ======================================================================================
# partialling-out (Robinson / DML) reference fit -- supplies se and dof for sensitivity
# ======================================================================================
def _partial_linear_ate(
    X: np.ndarray | pd.DataFrame,
    d: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 3,
    n_jobs: int = _GBM_N_JOBS,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, float]:
    """Cross-fitted partialling-out ATE with a heteroskedasticity-robust standard error.

    Estimator maths
    ---------------
    Robinson's (1988) partially linear model ``Y = tau D + g(X) + eps``. With out-of-fold nuisance
    fits ``m_hat(x) = E[Y|X=x]`` and ``e_hat(x) = P(D=1|X=x)``, residualise

    .. math:: \\tilde{Y} = Y - \\hat{m}(X), \\qquad \\tilde{D} = D - \\hat{e}(X),

    then

    .. math:: \\hat\\tau = \\frac{\\sum_i \\tilde{D}_i \\tilde{Y}_i}{\\sum_i \\tilde{D}_i^2},
              \\qquad
              \\widehat{se}(\\hat\\tau) =
              \\frac{\\sqrt{\\sum_i \\tilde{D}_i^2 \\hat\\varepsilon_i^2}}{\\sum_i \\tilde{D}_i^2},

    the Eicker-Huber-White sandwich for the residualised regression, where
    ``eps_hat = Y_res - tau_hat * D_res``. This is the Neyman-orthogonal score, so first-stage
    machine-learning error enters only at second order.

    Why it exists here
    ------------------
    :func:`sensitivity_contour` and the E-value need ``se(tau_hat)`` and residual degrees of
    freedom, which a black-box CATE learner does not expose. This function supplies them from the
    *linear approximation* in which the Cinelli-Hazlett omitted-variable-bias formula is exact.
    Reported degrees of freedom are the linear-model count ``n - p - 2``; with boosted nuisances
    the effective count is unknown, but the bias formula depends on ``se * sqrt(df)``, which is
    close to ``sigma / sd(D_res)`` and hence nearly invariant to the choice.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Covariates.
    d : numpy.ndarray
        Binary treatment indicator.
    y : numpy.ndarray
        Outcome.
    n_splits : int, default 3
        Cross-fitting folds; kept small because refutation is the expensive caller.
    n_jobs : int, default 1
        Threads per boosting fit; see :data:`_GBM_N_JOBS`.
    random_state : int or numpy.random.Generator or None, optional
        Seed.

    Returns
    -------
    dict
        ``{"estimate", "se", "dof", "t_value", "ci_low", "ci_high", "n", "n_splits"}``.
    """
    Z, cols = _design(X)
    d_arr = np.asarray(d, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = len(y_arr)
    d_int = np.rint(d_arr).astype(np.int64)
    counts = np.bincount(d_int, minlength=2)
    if counts.min() < 2:
        raise ValueError(f"partialling-out needs both treatment levels; got counts {counts.tolist()}")

    k = int(max(2, min(int(n_splits), int(counts.min()), 5)))
    if k != int(n_splits):
        LOGGER.warning("_partial_linear_ate reduced n_splits %d -> %d (rarest level has %d rows)",
                       int(n_splits), k, int(counts.min()))
    seed = int(as_rng(random_state).integers(0, 2**31 - 1))

    m_hat = np.zeros(n, dtype=np.float64)
    e_hat = np.full(n, float(d_arr.mean()), dtype=np.float64)
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in folds.split(Z, d_int):
        out_m = _cheap_gbm("regression", seed=seed, n_estimators=150, learning_rate=0.08,
                           max_depth=4, n_jobs=n_jobs)
        out_m.fit(Z[tr], y_arr[tr])
        m_hat[te] = np.asarray(out_m.predict(Z[te]), dtype=np.float64)
        if len(np.unique(d_int[tr])) > 1:
            out_e = _cheap_gbm("classification", seed=seed, n_estimators=150, learning_rate=0.08,
                               max_depth=4, n_jobs=n_jobs)
            out_e.fit(Z[tr], d_int[tr])
            e_hat[te] = np.asarray(out_e.predict_proba(Z[te])[:, 1], dtype=np.float64)

    e_hat = np.clip(e_hat, 0.01, 0.99)
    y_res = y_arr - m_hat
    d_res = d_arr - e_hat
    den = float(d_res @ d_res)
    if den < _EPS:
        raise ValueError("treatment has no residual variation after partialling out X (no overlap)")
    tau = float(d_res @ y_res / den)
    eps = y_res - tau * d_res
    with np.errstate(invalid="ignore", divide="ignore"):
        se = float(np.sqrt(float(np.sum((d_res**2) * (eps**2)))) / den)
    se = float(max(se, _EPS))
    dof = int(max(n - len(cols) - 2, 1))
    t_value = tau / se
    return {
        "estimate": tau,
        "se": se,
        "dof": float(dof),
        "t_value": float(t_value),
        "ci_low": float(tau - 1.959963985 * se),
        "ci_high": float(tau + 1.959963985 * se),
        "n": float(n),
        "n_splits": float(k),
    }


# ======================================================================================
# 1. falsification: placebo treatment
# ======================================================================================
def placebo_treatment(
    estimator_factory: EstimatorFactory,
    X: np.ndarray | pd.DataFrame,
    w: np.ndarray,
    y: np.ndarray,
    n_sim: int = 20,
    random_state: int | np.random.Generator | None = None,
    *,
    target_arm: int = 1,
    alpha: float = 0.05,
    interval_method: str = "auto",
    original: float | None = None,
) -> RefutationResult:
    """Permute the treatment and check that the effect dies with it.

    This is the single most informative refutation, because it tests the *pipeline* rather than
    the data. Replace the realised arm vector with a uniform random permutation of itself. The
    permuted arm is, by construction, independent of ``X``, of ``Y``, and of every unobserved
    cause. Any remaining "effect" is therefore not an effect: it is leakage, a bug, or the
    estimator's own optimism.

    Estimator maths
    ---------------
    Let ``pi`` be a uniform random permutation of ``{1..n}`` and ``W^pi_i = W_{pi(i)}``. The arm
    marginals ``P(A = a)`` are preserved exactly (it is a permutation, not a resample), so the
    per-arm sample sizes -- and hence the estimator's variance -- are unchanged. Under the
    permutation null

    .. math:: \\tau^{\\pi}_a = E[Y(a) - Y(0)] = 0 \\quad \\text{for every } a,

    so the placebo draws ``{tau_hat^{(s)}}`` for ``s = 1..S`` sample the estimator's *null
    distribution* at this sample size. Two summaries are reported:

    .. math::

       p = \\frac{1}{S} \\sum_{s=1}^{S}
           \\mathbf{1}\\{ |\\hat\\tau^{(s)}| \\ge |\\hat\\tau^{\\mathrm{obs}}| \\},

    the share of placebo runs that reach the observed magnitude, and the central
    ``1 - alpha`` interval of the placebo draws.

    Pass criterion
    --------------
    ``passed`` is ``True`` when the original estimate lies **outside** the central
    ``1 - alpha`` mass of the placebo distribution. A real effect sits far outside; an artefact
    sits inside. Note the direction: for this test a **small** ``p_value`` is the good outcome,
    because it means a permuted treatment essentially never reproduces the finding.

    How "central mass" is measured, and why it is not the empirical quantiles
    ------------------------------------------------------------------------
    The obvious choice, ``[q_{alpha/2}, q_{1-alpha/2}]`` of the draws, is a bad one at the
    sample sizes refutation can afford. With ``S = 20`` draws the 2.5th and 97.5th empirical
    percentiles *are* essentially the minimum and maximum, so the criterion degenerates into
    "did the observed value land inside the range of 20 draws" -- which, under a true null, it
    fails with probability ``2/(S+1) ~ 10%`` no matter how good the estimator is. A refuter with
    a 10% false-alarm rate baked into its arithmetic is not a refuter.

    The default (``interval_method="auto"``, which selects ``"prediction"`` whenever
    ``n_sim < 40``) instead uses the **Student prediction interval for one further draw**:

    .. math::

       \\bar{\\tau}^{\\pi} \\;\\pm\\; t_{S-1,\\,1-\\alpha/2}\\;
       s^{\\pi}\\sqrt{1 + \\tfrac{1}{S}} .

    That is exactly the right object: under the null, the observed estimate *is* one further
    draw from the placebo distribution, so the interval has the nominal ``1 - alpha`` coverage
    at any ``S``, and it borrows strength from the placebo mean and sd rather than from two
    order statistics. The empirical quantiles are still computed and reported in ``extra`` and
    ``detail`` for comparison. Pass ``interval_method="quantile"`` to score on them instead.

    Parameters
    ----------
    estimator_factory : callable
        Zero-argument callable returning a **fresh unfitted** CATE learner. Called ``n_sim + 1``
        times (``n_sim`` times if ``original`` is supplied).
    X : numpy.ndarray or pandas.DataFrame
        Covariates, passed to the learner untouched.
    w : numpy.ndarray
        Arm index, 0 = control.
    y : numpy.ndarray
        Outcome.
    n_sim : int, default 20
        Number of permutations. SPEC_PERF section 8 caps the default at 20 because each one is a
        full refit; the count actually run is logged and returned in ``n_sim``.
    random_state : int or numpy.random.Generator or None, optional
        Seed. Independent child streams are spawned per simulation with
        :func:`prism.utils.seeds.spawn_rngs`, so the result does not depend on iteration order.
    target_arm : int, keyword-only, default 1
        Which non-control arm's CATE column to average.
    alpha : float, keyword-only, default 0.05
        Width of the placebo "central mass" interval used by the pass criterion.
    interval_method : {"auto", "prediction", "quantile"}, keyword-only, default "auto"
        How the central mass is measured. ``"auto"`` uses the Student prediction interval when
        ``n_sim < 40`` and the empirical quantiles otherwise; see above for why.
    original : float, keyword-only, optional
        Precomputed ATE on unmodified data. :func:`run_refutation_suite` passes this so the
        original model is fit once for the whole suite rather than once per refuter.

    Returns
    -------
    RefutationResult
        ``refuted`` is the **mean** placebo ATE (which should be near 0). ``extra`` carries the
        full draw vector under ``"placebo_ates"``, plus ``"placebo_mean"``, ``"placebo_sd"``,
        ``"ci_low"``, ``"ci_high"`` (the interval actually scored on),
        ``"quantile_low"``/``"quantile_high"``, ``"z_score"`` and ``"p_value_add_one"``.

    Notes
    -----
    With ``n_sim = 20`` the p-value is granular to 0.05 and the quantile interval is effectively
    the observed range; both are reported for honesty rather than precision. Raise ``n_sim`` if
    the p-value itself, rather than the pass/fail, is going to be quoted.

    A conservative variant ``(r + 1) / (S + 1)``, which never returns exactly zero and is the
    correct form for a Monte-Carlo permutation p-value, is returned in
    ``extra["p_value_add_one"]``; ``p_value`` itself is the plain share, as specified.

    Passing this test is not evidence of a causal effect. It only rules out the specific failure
    mode in which the estimate survives the destruction of the treatment.
    """
    t0 = time.perf_counter()
    X, w_arr, y_arr = _coerce(X, w, y)
    n_sim = int(max(1, n_sim))
    if len(np.unique(w_arr)) < 2:
        raise ValueError("placebo_treatment needs at least two distinct arms in w")

    obs = float(original) if original is not None else _fit_and_ate(
        estimator_factory, X, w_arr, y_arr, target_arm=target_arm
    )

    rngs = spawn_rngs(random_state, n_sim)
    draws = np.empty(n_sim, dtype=np.float64)
    for s, rng in enumerate(rngs):
        w_perm = w_arr[rng.permutation(len(w_arr))]
        draws[s] = _fit_and_ate(estimator_factory, X, w_perm, y_arr, target_arm=target_arm)

    mean = float(draws.mean())
    sd = float(draws.std(ddof=1)) if n_sim > 1 else 0.0
    q_lo, q_hi = (float(v) for v in np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0]))

    lo, hi, method = _central_mass(mean, sd, q_lo, q_hi, n_sim=n_sim, alpha=alpha,
                                   interval_method=interval_method, caller="placebo_treatment")

    n_ge = int(np.sum(np.abs(draws) >= abs(obs)))
    p_value = float(n_ge / n_sim)
    p_add_one = float((n_ge + 1) / (n_sim + 1))
    passed = bool(obs < lo or obs > hi)
    z_score = float((obs - mean) / sd) if sd > _EPS else float("inf") if abs(obs - mean) > _EPS else 0.0

    LOGGER.info(
        "placebo_treatment: ran %d simulations (%d model refits) | original=%+.5f "
        "placebo mean=%+.5f sd=%.5f | central %.0f%% by %s interval = [%+.5f, %+.5f] "
        "(empirical quantiles [%+.5f, %+.5f]) | z=%.2f p=%.4f",
        n_sim, n_sim + (0 if original is not None else 1), obs, mean, sd,
        100 * (1 - alpha), method, lo, hi, q_lo, q_hi, z_score, p_value,
    )
    arm_label = ARM_NAMES[target_arm] if target_arm < len(ARM_NAMES) else f"arm{target_arm}"
    detail = (
        f"{n_sim} permutations of w, contrast arm {target_arm} ({arm_label}) vs control; "
        f"placebo ATE mean={mean:+.5f} sd={sd:.5f}; central {100 * (1 - alpha):.0f}% by the "
        f"{method} interval = [{lo:+.5f}, {hi:+.5f}] (empirical quantiles "
        f"[{q_lo:+.5f}, {q_hi:+.5f}], uninformative at this S); original {obs:+.5f} lies "
        f"{'OUTSIDE' if passed else 'INSIDE'} it at z={z_score:+.2f}. "
        f"p = share of placebo |ATE| >= |original| = {n_ge}/{n_sim} = {p_value:.4f} "
        f"(add-one variant {p_add_one:.4f}); low p is the good outcome here."
    )
    return RefutationResult(
        name="placebo_treatment",
        original=obs,
        refuted=mean,
        p_value=p_value,
        passed=passed,
        detail=detail,
        n_sim=n_sim,
        seconds=time.perf_counter() - t0,
        extra={
            "placebo_ates": draws,
            "placebo_mean": mean,
            "placebo_sd": sd,
            "ci_low": lo,
            "ci_high": hi,
            "quantile_low": q_lo,
            "quantile_high": q_hi,
            "interval_method": method,
            "z_score": z_score,
            "p_value_add_one": p_add_one,
            "alpha": float(alpha),
        },
    )


# ======================================================================================
# 2. falsification: random common cause
# ======================================================================================
def random_common_cause(
    estimator_factory: EstimatorFactory,
    X: np.ndarray | pd.DataFrame,
    w: np.ndarray,
    y: np.ndarray,
    n_sim: int = 10,
    random_state: int | np.random.Generator | None = None,
    *,
    target_arm: int = 1,
    tolerance: float = _DEFAULT_REL_TOL,
    original: float | None = None,
) -> RefutationResult:
    """Add an irrelevant covariate and check the estimate does **not** move.

    The mirror image of the placebo test. Append a column of independent standard normal noise to
    ``X``. It is by construction not a confounder, not a mediator and not a moderator, so the
    identified estimand is unchanged and the estimate should move only by the estimator's own
    numerical jitter. A large move means the learner is chasing noise -- over-fitting, an unstable
    split, or a nuisance model with too much capacity for the sample.

    Estimator maths
    ---------------
    Draw ``Z ~ N(0, 1)`` independently of ``(X, W, Y)`` and refit on ``[X, Z]``. Because ``Z`` is
    independent of everything,

    .. math:: E[Y(a) - Y(0) \\mid X, Z] = E[Y(a) - Y(0) \\mid X],

    so the target is untouched. The reported statistic is the mean relative movement

    .. math:: \\bar{\\delta} = \\frac{1}{S} \\sum_s
              \\frac{|\\hat\\tau^{(s)} - \\hat\\tau^{\\mathrm{obs}}|}
                    {\\max(|\\hat\\tau^{\\mathrm{obs}}|,\\; 0.05\\,\\mathrm{sd}(Y))}.

    Pass criterion
    --------------
    ``passed`` is ``True`` when the mean relative change is below ``tolerance``
    (**default 10%**). The denominator carries an absolute floor of ``0.05 * sd(y)``: when the
    original estimate is itself near zero a relative criterion is meaningless and would fail at
    random, so the test becomes an absolute one at that scale. Whether the floor was active is
    stated in ``detail`` and in ``extra["floor_active"]`` -- an undocumented fallback is how a
    refuter turns into a rubber stamp.

    ``p_value`` is ``None``: there is no null hypothesis here, only a tolerance.

    Parameters
    ----------
    estimator_factory : callable
        Zero-argument callable returning a fresh unfitted CATE learner.
    X : numpy.ndarray or pandas.DataFrame
        Covariates. A DataFrame gains a column named ``random_common_cause``; an ndarray gains a
        trailing column.
    w, y : numpy.ndarray
        Arm index and outcome.
    n_sim : int, default 10
        Number of independent noise columns tried. Each is a refit.
    random_state : int or numpy.random.Generator or None, optional
        Seed.
    target_arm : int, keyword-only, default 1
        CATE column to average.
    tolerance : float, keyword-only, default 0.10
        Relative-change tolerance defining "barely moves".
    original : float, keyword-only, optional
        Precomputed ATE on unmodified data.

    Returns
    -------
    RefutationResult
        ``refuted`` is the mean ATE across the ``n_sim`` refits. ``extra`` carries ``"ates"``,
        ``"rel_changes"``, ``"mean_rel_change"``, ``"max_rel_change"``, ``"floor"``,
        ``"floor_active"`` and ``"tolerance"``.

    Notes
    -----
    A pass here is a *stability* statement about the estimator, not a causal claim. It is the
    weakest of the four falsification tests and should never be quoted on its own.
    """
    t0 = time.perf_counter()
    X, w_arr, y_arr = _coerce(X, w, y)
    n_sim = int(max(1, n_sim))
    obs = float(original) if original is not None else _fit_and_ate(
        estimator_factory, X, w_arr, y_arr, target_arm=target_arm
    )
    y_sd = float(np.std(y_arr)) if len(y_arr) > 1 else 1.0
    floor = max(0.05 * y_sd, _EPS)

    rngs = spawn_rngs(random_state, n_sim)
    draws = np.empty(n_sim, dtype=np.float64)
    for s, rng in enumerate(rngs):
        z = rng.standard_normal(len(y_arr))
        X_aug = _append_column(X, z, "random_common_cause")
        draws[s] = _fit_and_ate(estimator_factory, X_aug, w_arr, y_arr, target_arm=target_arm)

    rel = np.array([_rel_change(float(v), obs, scale=y_sd) for v in draws], dtype=np.float64)
    mean_rel = float(rel.mean())
    max_rel = float(rel.max())
    mean_ate = float(draws.mean())
    passed = bool(mean_rel < float(tolerance))
    used_floor = bool(abs(obs) < floor)

    LOGGER.info(
        "random_common_cause: ran %d simulations (%d model refits) | original=%+.5f "
        "mean refit=%+.5f mean rel change=%.4f (tol %.2f) max=%.4f%s",
        n_sim, n_sim + (0 if original is not None else 1), obs, mean_ate, mean_rel,
        float(tolerance), max_rel, "  [absolute floor active]" if used_floor else "",
    )
    detail = (
        f"{n_sim} independent N(0,1) columns appended and refit; ATE {obs:+.5f} -> "
        f"{mean_ate:+.5f} (range [{draws.min():+.5f}, {draws.max():+.5f}]). "
        f"Mean relative change {mean_rel:.2%} vs tolerance {float(tolerance):.0%} "
        f"(max {max_rel:.2%}). Denominator = max(|original|, 0.05*sd(y)={floor:.5f})"
        + (" -- the absolute floor is ACTIVE because the original estimate is near zero, so this "
           "is an absolute, not a relative, criterion." if used_floor else ".")
    )
    return RefutationResult(
        name="random_common_cause",
        original=obs,
        refuted=mean_ate,
        p_value=None,
        passed=passed,
        detail=detail,
        n_sim=n_sim,
        seconds=time.perf_counter() - t0,
        extra={
            "ates": draws,
            "rel_changes": rel,
            "mean_rel_change": mean_rel,
            "max_rel_change": max_rel,
            "floor": float(floor),
            "floor_active": used_floor,
            "tolerance": float(tolerance),
        },
    )


# ======================================================================================
# 3. falsification: subset stability
# ======================================================================================
def subset_refuter(
    estimator_factory: EstimatorFactory,
    X: np.ndarray | pd.DataFrame,
    w: np.ndarray,
    y: np.ndarray,
    fraction: float = 0.7,
    n_sim: int = 10,
    random_state: int | np.random.Generator | None = None,
    *,
    target_arm: int = 1,
    alpha: float = 0.10,
    interval_method: str = "auto",
    predict_on: str = "subset",
    original: float | None = None,
) -> RefutationResult:
    """Refit on random subsets and check the answer does not depend on which rows you kept.

    A finding driven by a handful of influential rows, or by one segment that happens to dominate
    the sample, will wobble badly under resampling. This refuter quantifies that wobble.

    Estimator maths
    ---------------
    For ``s = 1..S`` draw a subsample ``I_s`` of size ``round(f n)`` **without replacement**,
    refit, and record ``tau_hat^{(s)}``. Report the spread of the draws and whether the original
    estimate lies inside their central ``1 - alpha`` interval.

    Honest reading of the spread
    ----------------------------
    ``sd(tau_hat^{(s)})`` is **not** a standard error. Two things break that interpretation, and
    both are reported:

    1. The subsamples are nested inside the original data, so each draw is positively correlated
       with the original estimate. That shrinks the apparent spread.
    2. A subsample of size ``f n`` has sampling variance roughly ``1 / f`` times the full-sample
       variance, which inflates it. The crude rescaling ``sd * sqrt(f)`` puts the figure back on a
       full-sample scale and is reported as ``sd_rescaled`` -- indicative only.

    What it *is* is an honest rough stability measure: how much the number moves when 30% of the
    evidence is removed at random.

    Pass criterion
    --------------
    ``passed`` is ``True`` when the original estimate lies inside the central ``1 - alpha`` mass
    of the subset draws. As in :func:`placebo_treatment`, that mass is measured by the Student
    **prediction interval**

    .. math:: \\bar{\\tau}^{f} \\;\\pm\\; t_{S-1,\\,1-\\alpha/2}\\; s^{f}\\sqrt{1 + 1/S}

    rather than by the empirical quantiles, which at ``S = 10`` are just the two extreme order
    statistics and would reject a perfectly stable estimate about ``2/(S+1) ~ 18%`` of the time.
    ``interval_method="quantile"`` restores the naive version for comparison, and both intervals
    are always reported.

    The prediction interval is deliberately **conservative** here. It treats the original as one
    further draw from the subset distribution, but the original is a *full-sample* estimate: it
    has smaller sampling variance than an ``f n``-row subsample and is positively correlated with
    every one of them. So the interval is wider than a correctly-calibrated one would be, and the
    test errs toward declaring stability. That is the right direction for this particular
    question -- it will not cry instability at noise that is merely the price of using 70% of the
    data -- but it means a pass here is weak evidence and the **sd is the informative output, not
    the boolean**. Said plainly rather than buried, because a test that cannot fail is worse than
    no test.

    ``p_value`` is ``None``. A "position" p-value could be computed, but it would read backwards
    relative to the placebo test's p-value in the same table, and that is exactly the kind of
    silent sign-flip a reader gets wrong.

    Parameters
    ----------
    estimator_factory : callable
        Zero-argument callable returning a fresh unfitted CATE learner.
    X, w, y : array-like
        Covariates, arm index and outcome.
    fraction : float, default 0.7
        Share of rows kept in each subsample; must lie in ``(0, 1]``.
    n_sim : int, default 10
        Number of subsamples. Each is a refit.
    random_state : int or numpy.random.Generator or None, optional
        Seed.
    target_arm : int, keyword-only, default 1
        CATE column to average.
    alpha : float, keyword-only, default 0.10
        Width of the central interval used by the pass criterion.
    interval_method : {"auto", "prediction", "quantile"}, keyword-only, default "auto"
        How that interval is built. ``"auto"`` uses the Student prediction interval when
        ``n_sim < 40`` and the empirical quantiles otherwise; see above.
    predict_on : {"subset", "full"}, keyword-only, default "subset"
        Population the refit CATE is averaged over. ``"subset"`` answers "would I have reached
        this conclusion from 70% of my data?", and so mixes estimation noise with a genuinely
        different covariate mix. ``"full"`` holds the target population fixed and isolates
        estimation instability alone.
    original : float, keyword-only, optional
        Precomputed ATE on unmodified data.

    Returns
    -------
    RefutationResult
        ``refuted`` is the mean subset ATE; ``extra`` carries ``"ates"``, ``"sd"``,
        ``"sd_rescaled"``, ``"ci_low"``, ``"ci_high"``, ``"fraction"``, ``"n_kept"`` and
        ``"percentile"``.
    """
    t0 = time.perf_counter()
    X, w_arr, y_arr = _coerce(X, w, y)
    n = len(y_arr)
    f = float(fraction)
    if not 0.0 < f <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1]; got {f}")
    if predict_on not in {"subset", "full"}:
        raise ValueError(f"predict_on must be 'subset' or 'full'; got {predict_on!r}")
    n_sim = int(max(1, n_sim))
    m = int(max(2, round(f * n)))

    obs = float(original) if original is not None else _fit_and_ate(
        estimator_factory, X, w_arr, y_arr, target_arm=target_arm
    )

    rngs = spawn_rngs(random_state, n_sim)
    draws = np.empty(n_sim, dtype=np.float64)
    for s, rng in enumerate(rngs):
        idx = np.sort(rng.choice(n, size=m, replace=False))
        draws[s] = _fit_and_ate(
            estimator_factory,
            _take_rows(X, idx),
            w_arr[idx],
            y_arr[idx],
            target_arm=target_arm,
            predict_X=None if predict_on == "subset" else X,
        )

    mean = float(draws.mean())
    sd = float(draws.std(ddof=1)) if n_sim > 1 else 0.0
    sd_rescaled = float(sd * np.sqrt(f))
    q_lo, q_hi = (float(v) for v in np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0]))
    lo, hi, method = _central_mass(mean, sd, q_lo, q_hi, n_sim=n_sim, alpha=alpha,
                                   interval_method=interval_method, caller="subset_refuter")
    passed = bool(lo <= obs <= hi)
    pct = float(np.mean(draws <= obs))
    z_score = float((obs - mean) / sd) if sd > _EPS else 0.0

    LOGGER.info(
        "subset_refuter: ran %d simulations (%d model refits) at fraction=%.2f (%d of %d rows), "
        "predict_on=%s | original=%+.5f subset mean=%+.5f sd=%.5f | central %.0f%% by %s "
        "interval = [%+.5f, %+.5f] (empirical quantiles [%+.5f, %+.5f]) z=%+.2f",
        n_sim, n_sim + (0 if original is not None else 1), f, m, n, predict_on,
        obs, mean, sd, 100 * (1 - alpha), method, lo, hi, q_lo, q_hi, z_score,
    )
    detail = (
        f"{n_sim} subsamples of {m}/{n} rows ({f:.0%}) without replacement, CATE averaged over "
        f"the {predict_on}; ATE {obs:+.5f} vs subset mean {mean:+.5f}, sd {sd:.5f} "
        f"(rescaled to full-sample scale by sqrt(f): {sd_rescaled:.5f} -- indicative only, "
        f"subsamples are nested so this is a stability measure, not a standard error). "
        f"Central {100 * (1 - alpha):.0f}% by the {method} interval = [{lo:+.5f}, {hi:+.5f}] "
        f"(empirical quantiles [{q_lo:+.5f}, {q_hi:+.5f}], uninformative at this S); original "
        f"sits at the {pct:.0%} percentile, z={z_score:+.2f}, and is "
        f"{'INSIDE' if passed else 'OUTSIDE'}. The interval is conservative (the original is a "
        f"full-sample estimate, not a subsample draw), so read the sd, not the boolean."
    )
    return RefutationResult(
        name="subset_refuter",
        original=obs,
        refuted=mean,
        p_value=None,
        passed=passed,
        detail=detail,
        n_sim=n_sim,
        seconds=time.perf_counter() - t0,
        extra={
            "ates": draws,
            "sd": sd,
            "sd_rescaled": sd_rescaled,
            "ci_low": lo,
            "ci_high": hi,
            "quantile_low": q_lo,
            "quantile_high": q_hi,
            "interval_method": method,
            "z_score": z_score,
            "fraction": f,
            "n_kept": int(m),
            "percentile": pct,
            "predict_on": predict_on,
        },
    )


# ======================================================================================
# 4. falsification: inject a confounder of known strength
# ======================================================================================
def add_unobserved_confounder(
    estimator_factory: EstimatorFactory,
    X: np.ndarray | pd.DataFrame,
    w: np.ndarray,
    y: np.ndarray,
    confounding_strength: float,
    random_state: int | np.random.Generator | None = None,
    *,
    target_arm: int = 1,
    tolerance: float = 0.35,
    original: float | None = None,
) -> RefutationResult:
    """Inject a synthetic unmeasured confounder ``U`` of known strength and measure the damage.

    The exact injection model
    -------------------------
    A sensitivity result without its injection model is uninterpretable, so here it is in full.
    Let ``s = confounding_strength >= 0``, let ``D_i = 1{W_i = target_arm}`` be the treatment
    indicator with ``p = mean(D)`` and ``sd(D) = sqrt(p (1 - p))``, and let
    ``C_i = (D_i - p) / sd(D)`` be the standardised treatment. Then

    .. math::

       U_i = s\\,C_i + Z_i, \\qquad Z_i \\sim N(0, 1)
       \\ \\text{i.i.d., independent of } (X, W, Y),

    .. math::

       \\tilde{Y}_i = Y_i + \\beta_u U_i, \\qquad \\beta_u = s \\cdot \\mathrm{sd}(Y).

    ``U`` is **never added to** ``X``. That is the entire point: it is unobserved, the estimator
    cannot adjust for it, and the bias it induces is the quantity of interest.

    Properties of this construction, all of them checkable:

    - ``var(U) = 1 + s^2`` and ``corr(U, D) = s / sqrt(1 + s^2)``, both reported in ``extra``.
    - ``U`` is independent of ``X``, so conditioning on ``X`` cannot remove its effect.
    - Because ``E[U | D = 1] - E[U | D = 0] = s / sd(D)``, the bias induced in **any** estimator
      that correctly adjusts for ``X`` has the closed form

      .. math::

         \\Delta \\;=\\; \\beta_u \\left( E[U \\mid D{=}1] - E[U \\mid D{=}0] \\right)
                 \\;=\\; \\frac{s^2\\,\\mathrm{sd}(Y)}{\\mathrm{sd}(D)},

      exact in expectation and reported as ``predicted_shift``. Comparing it against the shift
      actually observed is a self-test of the refuter, and of the learner.

    Why the treatment is not re-drawn
    ---------------------------------
    A tempting alternative is to re-draw ``W`` from a ``U``-shifted propensity. It is wrong here.
    The outcomes in hand are the outcomes realised under the arms actually given; re-assigning
    arms without re-generating the corresponding potential outcomes silently destroys the effect
    and turns this into a second, badly specified placebo test. Instead we draw ``U`` from the
    conditional law ``U | W`` implied by a structural model in which ``U`` shifts the treatment
    odds -- observationally equivalent for the bias calculation, and it changes only the
    unobserved cause.

    Pass criterion
    --------------
    ``passed`` is ``True`` when the **qualitative conclusion survives**: the biased estimate keeps
    the sign of the original. Whether the *mechanism* behaved as theory predicts is a separate
    question, reported as ``extra["bias_as_predicted"]`` and in ``detail`` -- a mismatch there
    indicts the refuter or the learner, not the finding, and folding the two into one boolean
    would mislead.

    Parameters
    ----------
    estimator_factory : callable
        Zero-argument callable returning a fresh unfitted CATE learner. Called once (twice if
        ``original`` is not supplied).
    X, w, y : array-like
        Covariates, arm index and outcome.
    confounding_strength : float
        ``s`` above, dimensionless and non-negative. ``0`` injects nothing; ``0.35`` matches
        ``DGPConfig.hidden_confounder_strength``; ``1.0`` is a confounder worth one standard
        deviation of the outcome.
    random_state : int or numpy.random.Generator or None, optional
        Seed.
    target_arm : int, keyword-only, default 1
        Arm whose contrast against control defines ``D``.
    tolerance : float, keyword-only, default 0.35
        Relative agreement required between observed and predicted shift before
        ``bias_as_predicted`` is set, with an absolute floor of ``0.02 * sd(y)``.
    original : float, keyword-only, optional
        Precomputed ATE on unmodified data.

    Returns
    -------
    RefutationResult
        ``refuted`` is the ATE after injection. ``extra`` carries ``"shift"``,
        ``"predicted_shift"``, ``"gap"``, ``"bias_as_predicted"``, ``"corr_u_d"``, ``"beta_u"``,
        ``"sign_kept"`` and ``"confounding_strength"``.
    """
    t0 = time.perf_counter()
    X, w_arr, y_arr = _coerce(X, w, y)
    s = float(confounding_strength)
    if s < 0:
        raise ValueError(f"confounding_strength must be >= 0; got {s}")
    rng = as_rng(random_state)

    d = (w_arr == int(target_arm)).astype(np.float64)
    p = float(d.mean())
    if not 0.0 < p < 1.0:
        raise ValueError(f"target_arm={target_arm} must be present and not universal; share={p:.4f}")
    sd_d = float(np.sqrt(p * (1.0 - p)))
    y_sd = float(np.std(y_arr))
    if y_sd < _EPS:
        raise ValueError("y has no variance; an outcome-scale confounder cannot be defined")

    c = (d - p) / sd_d
    u = s * c + rng.standard_normal(len(y_arr))
    beta_u = s * y_sd
    y_conf = y_arr + beta_u * u

    obs = float(original) if original is not None else _fit_and_ate(
        estimator_factory, X, w_arr, y_arr, target_arm=target_arm
    )
    biased = _fit_and_ate(estimator_factory, X, w_arr, y_conf, target_arm=target_arm)

    shift = float(biased - obs)
    predicted = float(s * s * y_sd / sd_d)
    abs_floor = 0.02 * y_sd
    gap = float(abs(shift - predicted))
    as_predicted = bool(gap <= max(float(tolerance) * abs(predicted), abs_floor))
    corr_ud = float(s / np.sqrt(1.0 + s * s)) if s > 0 else 0.0
    sign_kept = bool(np.sign(biased) == np.sign(obs) or abs(obs) < _EPS)

    LOGGER.info(
        "add_unobserved_confounder: ran 1 simulation (%d model refits) at strength s=%.3f "
        "(corr(U,D)=%.3f, beta_u=%.4f) | ATE %+.5f -> %+.5f, shift %+.5f vs predicted %+.5f",
        1 + (0 if original is not None else 1), s, corr_ud, beta_u, obs, biased, shift, predicted,
    )
    detail = (
        f"U = {s:g}*std(D) + N(0,1), y' = y + {beta_u:.4f}*U, U withheld from X "
        f"(corr(U,D)={corr_ud:.3f}, var(U)={1 + s * s:.3f}). "
        f"ATE {obs:+.5f} -> {biased:+.5f}, shift {shift:+.5f}; closed-form prediction "
        f"s^2*sd(y)/sd(D) = {predicted:+.5f}, gap {gap:.5f} -> mechanism "
        f"{'behaves as predicted' if as_predicted else 'DOES NOT match theory (suspect the learner)'}. "
        f"Sign of the effect {'survives' if sign_kept else 'FLIPS'} at this confounding strength."
    )
    return RefutationResult(
        name=f"add_unobserved_confounder(s={s:g})",
        original=obs,
        refuted=biased,
        p_value=None,
        passed=sign_kept,
        detail=detail,
        n_sim=1,
        seconds=time.perf_counter() - t0,
        extra={
            "confounding_strength": s,
            "shift": shift,
            "predicted_shift": predicted,
            "gap": gap,
            "bias_as_predicted": as_predicted,
            "corr_u_d": corr_ud,
            "beta_u": float(beta_u),
            "sign_kept": sign_kept,
            "sd_y": y_sd,
            "sd_d": sd_d,
        },
    )


# ======================================================================================
# 5. sensitivity: the E-value (VanderWeele & Ding, 2017)
# ======================================================================================
#: Accepted ``scale`` spellings, mapped to the canonical name.
_SCALE_ALIASES: dict[str, str] = {
    "auto": "auto",
    "rr": "rr", "risk_ratio": "rr", "riskratio": "rr", "ratio": "rr",
    "or": "or", "odds_ratio": "or", "oddsratio": "or",
    "hr": "hr", "hazard_ratio": "hr", "hazardratio": "hr",
    "smd": "smd", "standardized_mean_difference": "smd", "cohens_d": "smd", "d": "smd",
    "rd": "rd", "risk_difference": "rd",
    "difference": "difference", "diff": "difference", "ate": "difference", "linear": "difference",
}


def _rr_to_e_value(rr: float) -> float:
    """E-value for one risk ratio: ``E = RR + sqrt(RR (RR - 1))``, reciprocating when ``RR < 1``.

    Parameters
    ----------
    rr : float
        Risk ratio, strictly positive.

    Returns
    -------
    float
        The E-value, always ``>= 1``. ``nan`` if ``rr <= 0`` (not a risk ratio).
    """
    r = float(rr)
    if not np.isfinite(r) or r <= 0.0:
        return float("nan")
    if r < 1.0:
        r = 1.0 / r
    return float(r + np.sqrt(r * (r - 1.0)))


def e_value(
    estimate: float,
    ci_low: float,
    ci_high: float,
    *,
    scale: str = "auto",
    outcome_sd: float | None = None,
    baseline_risk: float | None = None,
    rare_outcome: bool = False,
) -> dict[str, Any]:
    """E-value: how strong an unmeasured confounder would have to be to explain the result away.

    **SCALE WARNING, STATED FIRST BECAUSE IT IS THE THING REVIEWERS CATCH.** The E-value is
    defined for an effect measured on the **risk-ratio scale**. It is *not* defined for a
    difference-scale estimate such as "this offer is worth +18.4 currency units of discounted
    margin". If you hand this function a difference, it will convert it using a published
    approximation and then tell you, loudly and in three places -- the ``scale_assumed`` key,
    the ``approximate`` flag and the ``warning`` string -- exactly what it assumed and that the
    number is approximate. It will never quietly return an authoritative-looking E-value computed
    on the wrong scale.

    Estimator maths
    ---------------
    For a risk ratio ``RR >= 1``, VanderWeele & Ding (2017) show that the minimum strength of
    association -- on the risk-ratio scale, with *both* the treatment and the outcome, above and
    beyond the measured covariates -- that an unmeasured confounder would need in order to fully
    explain away an observed ``RR`` is

    .. math:: E = RR + \\sqrt{RR\\,(RR - 1)}.

    It is the value of ``g`` at which the bound
    ``RR_{\\text{true}} \\ge RR_{\\text{obs}} / B``, with
    ``B = g^2 / (2g - 1)`` the Ding-VanderWeele bounding factor, first admits ``RR_true = 1``.
    For ``RR < 1`` the reciprocal is taken first, so the E-value is symmetric in
    ``log RR``. ``E = 1`` means no confounding at all is needed. ``E = 1.5`` is weak, ``E = 3``
    is substantial, and the only way to read it is against what the *measured* covariates
    achieve -- which is what :func:`sensitivity_contour` benchmarks.

    Which confidence bound
    ----------------------
    ``e_value_ci`` is computed on the **confidence bound nearest the null** (``RR = 1``): the
    lower bound when ``RR > 1``, the upper bound when ``RR < 1``. That is the conservative
    choice -- it asks how strong a confounder would have to be to move the *weakest* value the
    data are consistent with to the null. When the interval already contains ``RR = 1`` no
    confounding is needed at all and ``e_value_ci`` is ``1.0``, with ``bound_used`` saying so.

    Scale conversions (all approximate, all named in the output)
    -----------------------------------------------------------
    =============  ======================================================  ==========================
    ``scale``      transformation to RR                                    source
    =============  ======================================================  ==========================
    ``"rr"``       identity                                                --
    ``"or"``       ``RR = sqrt(OR)``, or ``RR = OR`` if ``rare_outcome``    VanderWeele & Ding (2017)
    ``"hr"``       ``RR = (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR))``       VanderWeele & Ding (2017)
    ``"smd"``      ``RR = exp(0.91 * d)``, ``d`` a standardised difference  Chinn (2000); VW&D (2017)
    ``"rd"``       ``RR = (p0 + RD) / p0`` with ``p0 = baseline_risk``      exact for a binary Y
    ``"difference"``  ``rd`` if ``baseline_risk`` given, else ``smd``       --
    =============  ======================================================  ==========================

    Every one of these is monotone increasing, so the confidence bounds map across in order.

    ``scale="auto"`` can only *rule out* the risk-ratio scale, never confirm it: if any of the
    three inputs is non-positive it cannot be a risk ratio, so the difference scale is assumed.
    If all three are positive the risk-ratio scale is assumed **and a warning is emitted**,
    because a positive difference-scale ATE looks exactly the same. Pass ``scale`` explicitly in
    production. :func:`run_refutation_suite` does.

    Parameters
    ----------
    estimate : float
        Point estimate on the scale given by ``scale``.
    ci_low, ci_high : float
        Confidence limits on the same scale. Swapped automatically if reversed.
    scale : str, keyword-only, default "auto"
        One of ``"auto"``, ``"rr"``, ``"or"``, ``"hr"``, ``"smd"``, ``"rd"``, ``"difference"``
        (aliases such as ``"risk_ratio"``, ``"ate"``, ``"cohens_d"`` are accepted).
    outcome_sd : float, keyword-only, optional
        Standard deviation of the outcome, used to standardise a difference-scale estimate before
        the ``exp(0.91 d)`` step. If omitted on the difference scale, the estimate is assumed to
        be **already standardised**, and that assumption is reported.
    baseline_risk : float, keyword-only, optional
        Control-arm risk ``p0`` in ``(0, 1)``. When supplied with a difference-scale estimate the
        exact risk-difference transformation is used instead of the standardised approximation.
    rare_outcome : bool, keyword-only, default False
        For ``scale="or"`` only: under the rare-outcome assumption ``RR ~ OR``; otherwise
        ``RR ~ sqrt(OR)``.

    Returns
    -------
    dict
        ``e_value_point`` : float
            E-value for the point estimate.
        ``e_value_ci`` : float
            E-value for the confidence bound nearest the null; ``1.0`` when the interval already
            covers the null.
        ``interpretation`` : str
            A sentence a reviewer can paste into a paper, including the scale caveat.
        Plus ``rr_point``, ``rr_ci_low``, ``rr_ci_high``, ``rr_bound_used``, ``bound_used``,
        ``scale_input``, ``scale_assumed``, ``scale_inferred``, ``converted``, ``conversion``,
        ``approximate``, ``null_value`` and ``warning``.

    Raises
    ------
    ValueError
        If ``scale`` is unknown, or ``baseline_risk`` is outside ``(0, 1)``.

    Examples
    --------
    >>> res = e_value(2.0, 1.5, 2.7, scale="rr")
    >>> round(res["e_value_point"], 3)
    3.414
    >>> res["bound_used"]
    'lower'
    >>> e_value(0.5, 0.35, 0.8, scale="rr")["e_value_point"] == res["e_value_point"]
    True

    References
    ----------
    VanderWeele, T. J. & Ding, P. (2017). Sensitivity analysis in observational research:
    introducing the E-value. *Annals of Internal Medicine* 167(4), 268-274.
    Chinn, S. (2000). A simple method for converting an odds ratio to effect size for use in
    meta-analysis. *Statistics in Medicine* 19, 3127-3131.
    """
    key = _SCALE_ALIASES.get(str(scale).strip().lower())
    if key is None:
        raise ValueError(
            f"unknown scale {scale!r}; expected one of {sorted(set(_SCALE_ALIASES))}"
        )
    est, lo_in, hi_in = float(estimate), float(ci_low), float(ci_high)
    if lo_in > hi_in:
        LOGGER.warning("e_value: ci_low %.6g > ci_high %.6g, swapping", lo_in, hi_in)
        lo_in, hi_in = hi_in, lo_in

    warning = ""
    inferred = False
    if key == "auto":
        inferred = True
        if min(est, lo_in, hi_in) > 0.0:
            key = "rr"
            warning = (
                "SCALE NOT SPECIFIED. All three inputs are positive, so the RISK-RATIO scale was "
                "assumed. If these are difference-scale numbers (an ATE in currency, months or "
                "probability points) this E-value is MEANINGLESS -- re-call with "
                "scale='difference' and outcome_sd= or baseline_risk=."
            )
        else:
            key = "difference"
            warning = (
                "SCALE NOT SPECIFIED. At least one input is <= 0, which a risk ratio cannot be, "
                "so the DIFFERENCE scale was assumed and converted approximately."
            )
        LOGGER.warning("e_value: %s", warning)

    if baseline_risk is not None and not 0.0 < float(baseline_risk) < 1.0:
        raise ValueError(f"baseline_risk must lie in (0, 1); got {baseline_risk}")

    # ---- transform to the risk-ratio scale -------------------------------------------
    if key == "rr":
        conv = "identity (already a risk ratio)"
        approximate = False
        scale_assumed = "risk ratio"
        to_rr = lambda v: float(v)  # noqa: E731
    elif key == "or":
        if rare_outcome:
            conv = "RR = OR (rare-outcome approximation)"
            to_rr = lambda v: float(v)  # noqa: E731
        else:
            conv = "RR = sqrt(OR) (VanderWeele & Ding 2017, common outcome)"
            to_rr = lambda v: float(np.sqrt(max(float(v), _EPS)))  # noqa: E731
        approximate = True
        scale_assumed = "odds ratio -> risk ratio"
    elif key == "hr":
        conv = "RR = (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR)) (VanderWeele & Ding 2017)"
        approximate = True
        scale_assumed = "hazard ratio -> risk ratio"

        def to_rr(v: float) -> float:
            h = max(float(v), _EPS)
            num = 1.0 - 0.5 ** np.sqrt(h)
            den = 1.0 - 0.5 ** np.sqrt(1.0 / h)
            return float(num / max(den, _EPS))
    elif key == "rd" or (key == "difference" and baseline_risk is not None):
        p0 = float(baseline_risk) if baseline_risk is not None else float("nan")
        if not np.isfinite(p0):
            raise ValueError("scale='rd' requires baseline_risk (the control-arm risk p0)")
        conv = f"RR = (p0 + RD) / p0 with p0 = {p0:.4f} (exact for a binary outcome)"
        approximate = False
        scale_assumed = "risk difference -> risk ratio"
        to_rr = lambda v: float(max(p0 + float(v), _EPS) / p0)  # noqa: E731
    else:  # "smd" or bare "difference"
        approximate = True
        if outcome_sd is not None:
            sd = float(outcome_sd)
            if sd <= 0:
                raise ValueError(f"outcome_sd must be > 0; got {sd}")
            conv = (
                f"d = estimate / sd(Y) with sd(Y) = {sd:.6g}, then "
                "RR = exp(0.91 * d) (Chinn 2000 / VanderWeele & Ding 2017) -- APPROXIMATE"
            )
            scale_assumed = "difference standardised by outcome_sd -> risk ratio"
            to_rr = lambda v: float(np.exp(0.91 * (float(v) / sd)))  # noqa: E731
        else:
            conv = (
                "estimate TREATED AS AN ALREADY-STANDARDISED mean difference (no outcome_sd "
                "given), then RR = exp(0.91 * d) (Chinn 2000 / VanderWeele & Ding 2017) "
                "-- APPROXIMATE"
            )
            scale_assumed = "assumed already-standardised difference -> risk ratio"
            to_rr = lambda v: float(np.exp(0.91 * float(v)))  # noqa: E731
            extra_warn = (
                "outcome_sd was not supplied, so the estimate was treated as ALREADY standardised "
                "(in units of sd(Y)). If it is in raw units the E-value is on the wrong scale."
            )
            warning = f"{warning} {extra_warn}".strip()
            LOGGER.warning("e_value: %s", extra_warn)

    rr = to_rr(est)
    rr_lo, rr_hi = to_rr(lo_in), to_rr(hi_in)
    if rr_lo > rr_hi:
        rr_lo, rr_hi = rr_hi, rr_lo
    converted = key != "rr"

    e_point = _rr_to_e_value(rr)
    if rr_lo <= 1.0 <= rr_hi:
        e_ci = 1.0
        bound_used = "none (the confidence interval already covers the null RR = 1)"
        rr_bound = 1.0
    elif rr > 1.0:
        bound_used = "lower"
        rr_bound = rr_lo
        e_ci = _rr_to_e_value(rr_lo)
    else:
        bound_used = "upper"
        rr_bound = rr_hi
        e_ci = _rr_to_e_value(rr_hi)

    if converted:
        scale_line = (
            f"Scale: input was {key!r}; {conv}. The E-value is defined on the risk-ratio scale, "
            f"so this conversion is {'exact' if not approximate else 'APPROXIMATE'} and the "
            f"number inherits its error."
        )
    else:
        scale_line = "Scale: the input was already a risk ratio, so no conversion was applied."

    interpretation = (
        f"An unmeasured confounder would have to be associated with both the treatment and the "
        f"outcome by a risk ratio of at least {e_point:.2f} each, above and beyond the measured "
        f"covariates, to explain away the point estimate (RR = {rr:.3f}). "
        + (
            "The confidence interval already includes the null, so no confounding is "
            "required to explain it (E-value for the interval = 1.00). "
            if bound_used.startswith("none")
            else f"To move the {bound_used} confidence bound (the one NEAREST THE NULL, "
            f"RR = {rr_bound:.3f}) to the null, a joint association of {e_ci:.2f} would suffice. "
        )
        + f"Weaker joint confounding could not do it. {scale_line}"
        + (f" WARNING: {warning}" if warning else "")
    )

    return {
        "e_value_point": float(e_point),
        "e_value_ci": float(e_ci),
        "interpretation": interpretation,
        "rr_point": float(rr),
        "rr_ci_low": float(rr_lo),
        "rr_ci_high": float(rr_hi),
        "rr_bound_used": float(rr_bound),
        "bound_used": bound_used,
        "scale_input": str(scale),
        "scale_assumed": scale_assumed,
        "scale_inferred": bool(inferred),
        "converted": bool(converted),
        "conversion": conv,
        "approximate": bool(approximate),
        "null_value": 1.0,
        "estimate": est,
        "ci_low": lo_in,
        "ci_high": hi_in,
        "warning": warning,
    }


# ======================================================================================
# 6. sensitivity: Rosenbaum bounds
# ======================================================================================
def _match_pairs(
    y: np.ndarray,
    w: np.ndarray,
    *,
    target_arm: int,
    score: np.ndarray | None,
    pair_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Build matched treated/control pairs, returning ``(idx_treated, idx_control, method)``.

    Three routes, in order of statistical quality:

    1. ``pair_ids`` -- the design already matched the data. Each id must contain exactly one
       treated and one control row; other groups are dropped and counted.
    2. ``score`` -- one-dimensional matching on a supplied propensity or prognostic score. Both
       arms are sorted by the score and paired in rank order, which **minimises the total
       absolute score gap** in one dimension (sorting is optimal for 1-D assignment when the two
       sets are the same size), so no greedy approximation is needed. The larger arm is thinned
       first by evenly spaced ranks so both sets cover the same score range.
    3. Neither -- fall back to rank pairing on the observed outcome. This does **not** control
       confounding; see the caller's docstring.
    """
    w = np.asarray(w).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    t_all = np.flatnonzero(w == int(target_arm))
    c_all = np.flatnonzero(w == 0)
    if len(t_all) == 0 or len(c_all) == 0:
        raise ValueError(
            f"rosenbaum_bounds needs both control (w == 0) and treated (w == {target_arm}) rows; "
            f"found {len(c_all)} and {len(t_all)}"
        )

    if pair_ids is not None:
        ids = np.asarray(pair_ids).ravel()
        if len(ids) != len(w):
            raise ValueError("pair_ids must have one entry per row")
        order = np.argsort(ids, kind="stable")
        ti: list[int] = []
        ci: list[int] = []
        dropped = 0
        start = 0
        ids_sorted = ids[order]
        while start < len(order):
            stop = start + 1
            while stop < len(order) and ids_sorted[stop] == ids_sorted[start]:
                stop += 1
            grp = order[start:stop]
            gt = [int(i) for i in grp if w[i] == int(target_arm)]
            gc = [int(i) for i in grp if w[i] == 0]
            if len(gt) == 1 and len(gc) == 1:
                ti.append(gt[0])
                ci.append(gc[0])
            else:
                dropped += 1
            start = stop
        if not ti:
            raise ValueError("pair_ids produced no usable 1-treated / 1-control pairs")
        if dropped:
            LOGGER.warning("rosenbaum_bounds: dropped %d pair id(s) that were not 1:1", dropped)
        return np.asarray(ti), np.asarray(ci), f"supplied pair_ids ({len(ti)} pairs, {dropped} dropped)"

    if score is not None:
        s = np.asarray(score, dtype=np.float64).ravel()
        if len(s) != len(w):
            raise ValueError("score must have one entry per row")
        method = "rank pairing on the supplied matching score (1-D optimal)"
    else:
        s = y
        method = (
            "FALLBACK rank pairing on the observed outcome -- the input was not matched and no "
            "score was given; this is descriptive only and controls no confounding"
        )
        LOGGER.warning(
            "rosenbaum_bounds: no pair_ids and no score supplied, so pairs were formed by "
            "rank-matching on the OBSERVED OUTCOME. This is a descriptive fallback: it assumes "
            "unconfoundedness rather than enforcing it, and the resulting bounds are not a "
            "design-based sensitivity analysis. Pass pair_ids= (matched data) or score= "
            "(a propensity or prognostic score) for a valid one."
        )

    t_sorted = t_all[np.argsort(s[t_all], kind="stable")]
    c_sorted = c_all[np.argsort(s[c_all], kind="stable")]
    m = int(min(len(t_sorted), len(c_sorted)))
    if len(t_sorted) > m:
        t_sorted = t_sorted[np.linspace(0, len(t_sorted) - 1, m).round().astype(int)]
    if len(c_sorted) > m:
        c_sorted = c_sorted[np.linspace(0, len(c_sorted) - 1, m).round().astype(int)]
    return t_sorted, c_sorted, f"{method} ({m} pairs)"


def rosenbaum_bounds(
    y: np.ndarray,
    w: np.ndarray,
    gammas: Sequence[float] = (1.0, 1.2, 1.5, 2.0, 3.0),
    *,
    target_arm: int = 1,
    score: np.ndarray | None = None,
    pair_ids: np.ndarray | None = None,
    alternative: str = "auto",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Rosenbaum bounds: how much hidden bias would it take to overturn the significance?

    The matched-pairs assumption, stated up front
    ---------------------------------------------
    Rosenbaum's sensitivity analysis is **defined on matched pairs**. Each pair contains one
    treated and one control unit with (as far as the design can manage) identical observed
    covariates, so that the only remaining source of difference in treatment odds is the
    unobserved. ``Gamma`` then bounds exactly that residual: two units in the same pair may
    differ in their odds of treatment by at most a factor ``Gamma``.

    If the input is **not matched**, this function pairs it for you, and says so explicitly in
    the ``matching`` column and in ``df.attrs["matching"]``:

    - ``pair_ids`` supplied -> those pairs are used (the correct input);
    - ``score`` supplied -> both arms are sorted by the score and paired in rank order, which is
      the optimal 1-D matching; pass a propensity score or a prognostic (predicted-outcome)
      score here;
    - **neither** -> pairs are formed by rank-matching on the *observed outcome*, with a loud
      warning logged. That fallback assumes unconfoundedness rather than enforcing it, so the
      resulting ``Gamma`` is descriptive, not design-based. It is offered because a bounds table
      with a caveat is more useful than a crash, not because it is valid.

    Estimator maths
    ---------------
    Let ``d_i = Y_{t(i)} - Y_{c(i)}`` be the ``I`` pair differences, drop the exact zeros, and let
    ``r_i = rank(|d_i|)`` (mid-ranks for ties). The Wilcoxon signed-rank statistic is

    .. math:: W = \\sum_{i : d_i > 0} r_i .

    Under the sharp null of no effect **and no hidden bias**, ``sign(d_i)`` is a fair coin, so
    ``E[W] = \\tfrac12 \\sum_i r_i`` and ``var(W) = \\tfrac14 \\sum_i r_i^2``. Under hidden bias of
    magnitude ``Gamma``, Rosenbaum (2002, ch. 4) shows the success probability of each sign is
    confined to

    .. math:: \\frac{1}{1 + \\Gamma} \\;\\le\\; \\pi_i \\;\\le\\; \\frac{\\Gamma}{1 + \\Gamma},

    which sandwiches the null distribution of ``W`` between two sums of independent weighted
    Bernoullis. With ``p^{+} = Gamma/(1+Gamma)`` and ``p^{-} = 1/(1+Gamma)``,

    .. math::

       z^{\\pm} = \\frac{W - p^{\\pm}\\sum_i r_i}
                        {\\sqrt{p^{\\pm}(1 - p^{\\pm})\\sum_i r_i^2}}, \\qquad
       p_{\\text{upper}} = 1 - \\Phi(z^{+}), \\qquad
       p_{\\text{lower}} = 1 - \\Phi(z^{-}).

    ``p_upper`` is the **least favourable** (largest possible) p-value at that ``Gamma`` and is
    the one the conclusion must survive; ``p_lower`` is the most favourable. At ``Gamma = 1`` the
    two coincide and equal the ordinary one-sided signed-rank p-value (normal approximation).

    The breakdown point ``Gamma*`` -- the largest tabulated ``Gamma`` at which ``p_upper`` is
    still below ``alpha`` -- is placed in ``df.attrs["gamma_star"]``. Read it as: "a hidden
    confounder would have to raise the odds of treatment by more than ``Gamma*`` to 1 within a
    matched pair before this result stops being significant."

    Parameters
    ----------
    y : numpy.ndarray
        Outcome, one row per unit.
    w : numpy.ndarray
        Arm index, 0 = control.
    gammas : sequence of float, default (1.0, 1.2, 1.5, 2.0, 3.0)
        Hidden-bias magnitudes, each ``>= 1``. Sorted ascending internally; duplicates removed.
    target_arm : int, keyword-only, default 1
        Treated arm to contrast against control. Rows in other arms are ignored.
    score : numpy.ndarray, keyword-only, optional
        Matching score (propensity or predicted outcome), one entry per row.
    pair_ids : numpy.ndarray, keyword-only, optional
        Pre-matched pair identifiers, one entry per row. Takes precedence over ``score``.
    alternative : {"auto", "greater", "less", "two-sided"}, keyword-only, default "auto"
        Rosenbaum bounds are inherently one-sided. ``"auto"`` picks the direction from the sign of
        the mean pair difference -- convenient, but it spends a look at the data, so for a
        pre-registered analysis state the direction. ``"two-sided"`` doubles the p-values
        (capped at 1).
    alpha : float, keyword-only, default 0.05
        Significance threshold behind ``still_significant``.

    Returns
    -------
    pandas.DataFrame
        One row per ``gamma`` with columns ``gamma``, ``p_lower``, ``p_upper``,
        ``still_significant``, plus ``n_pairs``, ``statistic``, ``alternative`` and ``matching``.
        ``df.attrs`` carries ``gamma_star``, ``matching``, ``n_pairs``, ``n_zero_dropped``,
        ``mean_difference``, ``hodges_lehmann``, ``hodges_lehmann_method``, ``alpha``, and the
        realised pairing itself -- ``pair_index_treated``, ``pair_index_control`` and
        ``pair_differences`` (signed in the *tested* direction, before zero differences are
        dropped) -- so a caller can audit or re-test the pairs this function chose.

    Raises
    ------
    ValueError
        If any ``gamma < 1``, if an arm is missing, or if every pair difference is zero.

    References
    ----------
    Rosenbaum, P. R. (2002). *Observational Studies*, 2nd ed., chapter 4. Springer.
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    w_arr = np.asarray(w).ravel()
    if len(y_arr) != len(w_arr):
        raise ValueError(f"y and w must be the same length; got {len(y_arr)} and {len(w_arr)}")
    if alternative not in {"auto", "greater", "less", "two-sided"}:
        raise ValueError(f"alternative must be auto/greater/less/two-sided; got {alternative!r}")

    g_arr = np.unique(np.asarray(list(gammas), dtype=np.float64))
    if g_arr.size == 0:
        raise ValueError("gammas must not be empty")
    if float(g_arr.min()) < 1.0:
        raise ValueError(f"every gamma must be >= 1 (1 = no hidden bias); got min {g_arr.min()}")

    it, ic, matching = _match_pairs(
        y_arr, w_arr, target_arm=target_arm, score=score, pair_ids=pair_ids
    )
    diff = y_arr[it] - y_arr[ic]
    n_all = int(diff.size)
    mean_diff = float(diff.mean())

    direction = alternative
    if alternative == "auto":
        direction = "greater" if mean_diff >= 0 else "less"
    if direction == "less":
        diff = -diff

    nz = diff != 0.0
    n_zero = int((~nz).sum())
    d = diff[nz]
    if d.size == 0:
        raise ValueError("every pair difference is exactly zero; the signed-rank test is undefined")
    ranks = stats.rankdata(np.abs(d))
    s1 = float(ranks.sum())
    s2 = float((ranks**2).sum())
    stat = float(ranks[d > 0].sum())
    i_pairs = int(d.size)
    # Hodges-Lehmann point estimate: median of the Walsh averages, on the tested direction.
    # The Walsh matrix is O(I^2); above the cap fall back to the median difference (a consistent
    # but less efficient location estimate) rather than allocating gigabytes silently.
    _WALSH_CAP = 4000
    if i_pairs <= _WALSH_CAP:
        walsh = (d[:, None] + d[None, :]) / 2.0
        hl = float(np.median(walsh[np.triu_indices(i_pairs)]))
        hl_method = "Hodges-Lehmann (median Walsh average)"
    else:
        hl = float(np.median(d))
        hl_method = f"median pair difference (I={i_pairs} > {_WALSH_CAP}, Walsh matrix skipped)"
        LOGGER.info(
            "rosenbaum_bounds: %d pairs exceeds the Walsh-average cap of %d, reporting the "
            "median pair difference instead of the Hodges-Lehmann estimate",
            i_pairs, _WALSH_CAP,
        )

    rows: list[dict[str, Any]] = []
    for g in g_arr:
        p_plus = float(g / (1.0 + g))
        p_minus = float(1.0 / (1.0 + g))
        with np.errstate(divide="ignore", invalid="ignore"):
            z_up = (stat - p_plus * s1) / np.sqrt(max(p_plus * (1.0 - p_plus) * s2, _EPS))
            z_lo = (stat - p_minus * s1) / np.sqrt(max(p_minus * (1.0 - p_minus) * s2, _EPS))
        p_upper = float(stats.norm.sf(z_up))
        p_lower = float(stats.norm.sf(z_lo))
        if alternative == "two-sided":
            p_upper = float(min(1.0, 2.0 * p_upper))
            p_lower = float(min(1.0, 2.0 * p_lower))
        rows.append(
            {
                "gamma": float(g),
                "p_lower": p_lower,
                "p_upper": p_upper,
                "still_significant": bool(p_upper < float(alpha)),
                "n_pairs": i_pairs,
                "statistic": stat,
                "alternative": direction if alternative != "two-sided" else "two-sided",
                "matching": matching,
            }
        )

    out = pd.DataFrame(rows)
    sig = out.loc[out["still_significant"], "gamma"]
    gamma_star = float(sig.max()) if len(sig) else float("nan")
    out.attrs.update(
        {
            "gamma_star": gamma_star,
            "matching": matching,
            "n_pairs": i_pairs,
            "n_pairs_before_zero_drop": n_all,
            "n_zero_dropped": n_zero,
            "mean_difference": mean_diff,
            "pair_index_treated": it,
            "pair_index_control": ic,
            "pair_differences": diff,
            "hodges_lehmann": hl if direction == "greater" else -hl,
            "hodges_lehmann_method": hl_method,
            "alpha": float(alpha),
            "alternative": direction if alternative != "two-sided" else "two-sided",
            "note": (
                "p_upper is the worst-case (least favourable) p-value at that Gamma and is the "
                "one a conclusion must survive. Normal approximation, no continuity correction, "
                "matching R's rbounds::psens."
            ),
        }
    )
    LOGGER.info(
        "rosenbaum_bounds: %d pairs (%d zero differences dropped) via %s | W=%.1f, "
        "mean pair difference %+.5f, direction=%s, gamma* = %s",
        i_pairs, n_zero, matching, stat, mean_diff, out.attrs["alternative"],
        "none (not significant at Gamma=1)" if not np.isfinite(gamma_star) else f"{gamma_star:g}",
    )
    return out


# ======================================================================================
# 7. sensitivity: Austen / Cinelli-Hazlett contour with covariate benchmarks
# ======================================================================================
def _ols_partial_r2(Z: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, int]:
    """Partial ``R^2`` of every column of ``Z`` with ``target``, from one OLS fit.

    Estimator maths
    ---------------
    In a linear regression of ``target`` on an intercept and all columns of ``Z``, the partial
    ``R^2`` of regressor ``j`` -- the share of the variance in ``target`` left unexplained by the
    *other* regressors that ``j`` then explains -- is a pure function of its t-statistic:

    .. math:: R^2_{\\text{target} \\sim Z_j \\mid Z_{-j}} = \\frac{t_j^2}{t_j^2 + df}.

    That identity is what makes benchmarking cheap: ``p`` partial ``R^2`` values come from one
    least-squares solve instead of ``2p`` auxiliary regressions. It is the same route
    ``sensemakr`` takes.

    Parameters
    ----------
    Z : numpy.ndarray
        ``(n, p)`` regressors, no intercept column (one is added).
    target : numpy.ndarray
        ``(n,)`` dependent variable.

    Returns
    -------
    (numpy.ndarray, int)
        ``(p,)`` partial R-squared values in column order, and the residual degrees of freedom.
    """
    n = Z.shape[0]
    A = np.column_stack([np.ones(n, dtype=np.float64), np.asarray(Z, dtype=np.float64)])
    t_vec = np.asarray(target, dtype=np.float64).ravel()
    beta, *_ = np.linalg.lstsq(A, t_vec, rcond=None)
    resid = t_vec - A @ beta
    dof = int(max(n - A.shape[1], 1))
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(A.T @ A)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), _EPS))
        t_stat = beta / se
    t2 = np.nan_to_num(t_stat[1:] ** 2, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(t2 / (t2 + dof), 0.0, _R2_MAX), dof


def _ovb_bias(r2_y: np.ndarray, r2_w: np.ndarray, se: float, dof: float) -> np.ndarray:
    """Cinelli-Hazlett omitted-variable-bias magnitude on the ``(R2_y, R2_w)`` grid."""
    ry = np.clip(np.asarray(r2_y, dtype=np.float64), 0.0, _R2_MAX)
    rw = np.clip(np.asarray(r2_w, dtype=np.float64), 0.0, _R2_MAX)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(float(se) * np.sqrt(float(dof)) * np.sqrt(ry * rw / (1.0 - rw)))


def sensitivity_contour(
    estimator_factory: EstimatorFactory,
    X: np.ndarray | pd.DataFrame,
    w: np.ndarray,
    y: np.ndarray,
    r2_y_grid: np.ndarray | Sequence[float] | None = None,
    r2_w_grid: np.ndarray | Sequence[float] | None = None,
    *,
    target_arm: int = 1,
    feature_names: Sequence[str] | None = None,
    estimate: float | None = None,
    se: float | None = None,
    dof: float | None = None,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Austen / Cinelli-Hazlett sensitivity grid, benchmarked against the observed covariates.

    The E-value asks one question with one number. This asks the same question as a surface: for
    every combination of *how much of the outcome* and *how much of the treatment* a hypothetical
    unmeasured confounder would explain, how far would the estimate move, and where does it hit
    zero? The contour at zero is the frontier a confounder must cross to destroy the finding.

    A contour plot alone is decoration. What makes it an argument is the **benchmark**: each
    observed covariate is placed on the same axes at its *own* partial ``R^2`` values, so the
    reader can see whether the required confounder is plausible. That is what licenses the
    sentence "you would need a confounder twice as strong as ``tenure_months`` to erase this
    effect", and the multiple is computed exactly, in closed form, per feature.

    Estimator maths
    ---------------
    In the partially linear model ``Y = tau D + g(X) + eps`` with an omitted ``U``, Cinelli &
    Hazlett (2020) show the bias depends on the confounder only through two partial ``R^2``
    values -- with the outcome, ``R^2_{Y~U|D,X}``, and with the treatment, ``R^2_{D~U|X}``:

    .. math::

       |\\widehat{\\text{bias}}| = \\widehat{se}(\\hat\\tau)\\;\\sqrt{df}\\;
          \\sqrt{\\frac{R^2_{Y \\sim U \\mid D, X}\\; R^2_{D \\sim U \\mid X}}
                       {1 - R^2_{D \\sim U \\mid X}}}.

    The **direction** of the bias is not identified, so the adversarial direction is assumed
    throughout: ``bias = sign(tau_hat) * |bias|`` and ``adjusted_estimate = tau_hat - bias``, so
    the surface always moves the estimate *toward* the null and ``nullified`` marks
    ``|bias| >= |tau_hat|``.

    Two derived quantities are computed in closed form:

    *Robustness value.* Setting both partial ``R^2`` to a common ``RV`` and solving
    ``|bias| = |tau_hat|`` gives

    .. math:: RV = \\tfrac12\\left(\\sqrt{f^4 + 4f^2} - f^2\\right), \\qquad f = |t| / \\sqrt{df}.

    A confounder explaining less than ``RV`` of both the treatment and the residual outcome
    cannot overturn the result.

    *Benchmark multiple.* For observed covariate ``j`` with partial ``R^2`` values
    ``(a_j, b_j) = (R^2_{Y \\sim X_j}, R^2_{D \\sim X_j})``, the factor ``k`` by which a
    confounder must exceed ``j`` **on both axes** to nullify the estimate solves
    ``k^2 a_j b_j / (1 - k b_j) = f^2``, i.e.

    .. math:: k_j = \\frac{-f^2 b_j + \\sqrt{f^4 b_j^2 + 4 a_j b_j f^2}}{2 a_j b_j}.

    ``k_j = 1`` exactly when a copy of covariate ``j`` would erase the effect; ``k_j = 2`` is the
    "twice as strong as ``tenure_months``" sentence.

    The approximation, stated plainly
    ---------------------------------
    The formula above is **exact for a linear model**. PRISM's point estimate comes from a
    machine-learning CATE learner, so this function:

    1. takes the point estimate from ``estimator_factory`` (or from ``estimate=``), and
    2. takes ``se`` and ``df`` from the cross-fitted **partialling-out (Robinson / DML)**
       representation of the same contrast -- the linear approximation in which the bias formula
       holds -- unless the caller supplies them.

    The mixture is deliberate and is recorded in ``df.attrs["approximation"]``, in every log
    line, and here. Treat the contour as a well-calibrated order-of-magnitude statement about
    confounding strength, not as a second decimal place.

    Parameters
    ----------
    estimator_factory : callable
        Zero-argument callable returning a fresh unfitted CATE learner. Called once, and not at
        all if ``estimate`` is supplied.
    X : numpy.ndarray or pandas.DataFrame
        Covariates. Categorical columns are one-hot encoded for the benchmark regressions only.
    w : numpy.ndarray
        Arm index, 0 = control. Only rows with ``w in {0, target_arm}`` enter the analysis.
    y : numpy.ndarray
        Outcome.
    r2_y_grid, r2_w_grid : array-like, optional
        Grid values in ``[0, 1)``. The default is 13 points from 0 to
        ``max(0.30, 1.5 * strongest benchmark)``, capped at 0.9, so the plot always contains the
        observed covariates instead of cropping them out of frame.
    target_arm : int, keyword-only, default 1
        Arm contrasted against control.
    feature_names : sequence of str, keyword-only, optional
        Names for benchmark rows; taken from a DataFrame's columns otherwise.
    estimate, se, dof : float, keyword-only, optional
        Override the point estimate, its standard error and the residual degrees of freedom.
        :func:`run_refutation_suite` passes all three so the partialling-out fit is shared.
    random_state : int or numpy.random.Generator or None, keyword-only, optional
        Seed for the cross-fitted nuisance models.

    Returns
    -------
    pandas.DataFrame
        Tidy, one row per grid point **and** one per observed covariate:

        =====================  ====================================================
        column                 meaning
        =====================  ====================================================
        ``r2_y``               partial R^2 of the confounder with the outcome
        ``r2_w``               partial R^2 of the confounder with the treatment
        ``bias``               signed bias, in outcome units, adversarial direction
        ``adjusted_estimate``  ``estimate - bias``
        ``nullified``          ``True`` when the adjusted estimate crosses the null
        ``benchmark``          ``True`` for the observed-covariate rows
        ``feature``            covariate name on benchmark rows, ``pd.NA`` on grid rows
        ``nullify_multiple``   ``k_j`` above, on benchmark rows only
        =====================  ====================================================

        ``df.attrs`` carries ``estimate``, ``se``, ``dof``, ``t_value``, ``robustness_value``,
        ``robustness_value_alpha``, ``n``, ``approximation`` and ``estimate_partialled``.

    References
    ----------
    Cinelli, C. & Hazlett, C. (2020). Making sense of sensitivity: extending omitted variable
    bias. *JRSS-B* 82(1), 39-67.
    Veitch, V. & Zaveri, A. (2020). Sense and sensitivity analysis (the Austen plot).
    """
    X, w_arr, y_arr = _coerce(X, w, y)
    keep = np.flatnonzero((w_arr == 0) | (w_arr == int(target_arm)))
    if len(keep) < 20:
        raise ValueError(
            f"sensitivity_contour needs control and arm-{target_arm} rows; found {len(keep)}"
        )
    if len(keep) < len(w_arr):
        LOGGER.info(
            "sensitivity_contour: restricted to the arm-%d vs control contrast, keeping %d of "
            "%d rows (%.1f%%); other arms are not part of this estimand",
            target_arm, len(keep), len(w_arr), 100.0 * len(keep) / len(w_arr),
        )
    Xk = _take_rows(X, keep)
    yk = y_arr[keep]
    dk = (w_arr[keep] == int(target_arm)).astype(np.float64)

    names = _feature_names(X, feature_names)
    Z, z_cols = _design(Xk)
    if len(names) != Z.shape[1]:
        names = z_cols  # one-hot expansion changed the width; use the expanded names

    # ---- point estimate, standard error, degrees of freedom ---------------------------
    plm: dict[str, float] | None = None
    if se is None or dof is None:
        plm = _partial_linear_ate(Xk, dk, yk, random_state=random_state)
    se_val = float(se) if se is not None else float(plm["se"])  # type: ignore[index]
    dof_val = float(dof) if dof is not None else float(plm["dof"])  # type: ignore[index]

    if estimate is not None:
        tau = float(estimate)
        est_source = "supplied by the caller"
    else:
        tau = _fit_and_ate(estimator_factory, Xk, w_arr[keep], yk, target_arm=target_arm)
        est_source = "mean CATE from estimator_factory"
    sign = 1.0 if tau >= 0 else -1.0
    f_stat = abs(tau) / max(se_val, _EPS) / np.sqrt(max(dof_val, 1.0))
    rv = float(0.5 * (np.sqrt(f_stat**4 + 4.0 * f_stat**2) - f_stat**2))
    f_alpha = max(abs(tau) - 1.959963985 * se_val, 0.0) / max(se_val, _EPS) / np.sqrt(max(dof_val, 1.0))
    rv_alpha = float(0.5 * (np.sqrt(f_alpha**4 + 4.0 * f_alpha**2) - f_alpha**2))

    # ---- benchmark the observed covariates -------------------------------------------
    r2_y_bench, dof_y = _ols_partial_r2(np.column_stack([Z, dk]), yk)
    r2_y_bench = r2_y_bench[: Z.shape[1]]  # drop the treatment column's own entry
    r2_w_bench, _ = _ols_partial_r2(Z, dk)

    a = np.clip(r2_y_bench, 0.0, _R2_MAX)
    b = np.clip(r2_w_bench, 0.0, _R2_MAX)
    ab = a * b
    with np.errstate(invalid="ignore", divide="ignore"):
        k_mult = np.where(
            ab > _EPS,
            (-(f_stat**2) * b + np.sqrt(f_stat**4 * b**2 + 4.0 * ab * f_stat**2)) / (2.0 * ab),
            np.inf,
        )
    k_mult = np.where(np.isfinite(k_mult), k_mult, np.inf)

    # ---- grid, widened so the benchmarks are actually on the plot ---------------------
    widest = float(max(np.max(a) if a.size else 0.0, np.max(b) if b.size else 0.0))
    top = float(min(0.90, max(0.30, 1.5 * widest)))
    gy = np.linspace(0.0, top, 13) if r2_y_grid is None else np.asarray(r2_y_grid, dtype=np.float64)
    gw = np.linspace(0.0, top, 13) if r2_w_grid is None else np.asarray(r2_w_grid, dtype=np.float64)
    gy = np.clip(np.unique(gy), 0.0, _R2_MAX)
    gw = np.clip(np.unique(gw), 0.0, _R2_MAX)
    GY, GW = np.meshgrid(gy, gw, indexing="ij")
    grid_bias = sign * _ovb_bias(GY.ravel(), GW.ravel(), se_val, dof_val)

    grid = pd.DataFrame(
        {
            "r2_y": GY.ravel(),
            "r2_w": GW.ravel(),
            "bias": grid_bias,
            "adjusted_estimate": tau - grid_bias,
            "nullified": np.abs(grid_bias) >= abs(tau),
            "benchmark": False,
            "feature": pd.NA,
            "nullify_multiple": np.nan,
        }
    )
    bench_bias = sign * _ovb_bias(a, b, se_val, dof_val)
    bench = pd.DataFrame(
        {
            "r2_y": a,
            "r2_w": b,
            "bias": bench_bias,
            "adjusted_estimate": tau - bench_bias,
            "nullified": np.abs(bench_bias) >= abs(tau),
            "benchmark": True,
            "feature": list(names),
            "nullify_multiple": k_mult,
        }
    )
    out = pd.concat([grid, bench], ignore_index=True)
    out["nullified"] = out["nullified"].astype(bool)
    out["benchmark"] = out["benchmark"].astype(bool)

    approximation = (
        "The Cinelli-Hazlett bias formula is exact for a LINEAR model. The point estimate here "
        f"is {est_source}, while se and df come from the cross-fitted partialling-out "
        "(Robinson / DML) representation of the same contrast. Read the contour as an "
        "order-of-magnitude statement about confounding strength, not a second decimal place."
    )
    out.attrs.update(
        {
            "estimate": float(tau),
            "estimate_source": est_source,
            "estimate_partialled": float(plm["estimate"]) if plm is not None else float("nan"),
            "se": se_val,
            "dof": dof_val,
            "t_value": float(tau / max(se_val, _EPS)),
            "f_stat": float(f_stat),
            "robustness_value": rv,
            "robustness_value_alpha": rv_alpha,
            "n": int(len(keep)),
            "target_arm": int(target_arm),
            "approximation": approximation,
            "bias_direction": "adversarial (always toward the null); the sign is not identified",
        }
    )

    finite_k = k_mult[np.isfinite(k_mult)]
    worst = int(np.argmin(k_mult)) if finite_k.size else -1
    LOGGER.info(
        "sensitivity_contour: estimate=%+.5f se=%.5f df=%.0f t=%.2f | robustness value "
        "RV(q=1)=%.4f, RV at alpha=0.05 = %.4f | %d grid points x %d benchmarks%s",
        tau, se_val, dof_val, tau / max(se_val, _EPS), rv, rv_alpha,
        len(grid), len(bench),
        ""
        if worst < 0
        else (
            f" | most threatening covariate: {names[worst]} "
            f"(R2_y={a[worst]:.4f}, R2_w={b[worst]:.4f}), a confounder would need to be "
            f"{k_mult[worst]:.2f}x as strong to nullify"
        ),
    )
    LOGGER.info("sensitivity_contour: %s", approximation)
    return out


# ======================================================================================
# 8. the suite
# ======================================================================================
def run_refutation_suite(
    estimator_factory: EstimatorFactory | None = None,
    X: np.ndarray | pd.DataFrame | None = None,
    w: np.ndarray | None = None,
    y: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
    *,
    target_arm: int = 1,
    n_sim: int | None = None,
    fraction: float = 0.7,
    confounding_strengths: Sequence[float] = (0.25, 0.5),
    gammas: Sequence[float] = (1.0, 1.2, 1.5, 2.0, 3.0),
    tests: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run every applicable refutation and sensitivity test and return one tidy row per test.

    This is what the pipeline calls and what lands in ``artifacts/reports/refutation.csv``. It is
    written so that **one broken refuter cannot take the report down**: every test runs inside its
    own ``try``/``except``, and a failure becomes a row with ``passed=False`` and the exception in
    ``detail`` rather than an exception out of this function.

    What it runs
    ------------
    ===========================  ==================================================================
    row                          ``refuted`` column holds
    ===========================  ==================================================================
    ``placebo_treatment``        mean ATE under permuted treatment (should be ~0)
    ``random_common_cause``      mean ATE after adding an irrelevant covariate (should not move)
    ``subset_refuter``           mean ATE across ``f``-subsamples
    ``add_unobserved_confounder(s=...)``  ATE after injecting a confounder of that strength
    ``e_value``                  the E-value of the point estimate
    ``rosenbaum_bounds``         ``Gamma*``, the breakdown point of the significance (passes
                                 only when ``Gamma* > 1``, i.e. the result survives *some*
                                 hidden bias, not merely the assumption of none)
    ``sensitivity_contour``      the robustness value ``RV(q=1)``
    ===========================  ==================================================================

    Because ``refuted`` means different things in different rows, every ``detail`` string names
    the quantity. A single column of heterogeneous numbers is a reporting hazard; labelling it is
    the cheapest possible fix.

    Efficiency
    ----------
    SPEC_PERF section 8: the original model is fit **once** here and passed to each refuter as
    ``original=``, and the cross-fitted partialling-out fit (which supplies the standard error
    for the E-value CI and for the contour) is also computed once and shared. Total refits are
    logged at the end.

    Parameters
    ----------
    estimator_factory : callable, optional
        Zero-argument callable returning a fresh unfitted CATE learner. Defaults to
        :func:`cheap_cate_factory` seeded from ``random_state`` -- a 100-tree T-learner, as
        SPEC_PERF section 8 requires of anything refitted this many times.
    X : numpy.ndarray or pandas.DataFrame
        Covariates.
    w : numpy.ndarray
        Arm index, 0 = control.
    y : numpy.ndarray
        Outcome, higher is better.
    random_state : int or numpy.random.Generator or None, optional
        Seed. Every test gets its own child stream, so adding or removing a test does not change
        the others' draws.
    target_arm : int, keyword-only, default 1
        Non-control arm to analyse.
    n_sim : int, keyword-only, optional
        Overrides the per-test simulation counts (placebo 20, common cause 10, subset 10).
    fraction : float, keyword-only, default 0.7
        Subsample share for :func:`subset_refuter`.
    confounding_strengths : sequence of float, keyword-only, default (0.25, 0.5)
        One :func:`add_unobserved_confounder` row per value.
    gammas : sequence of float, keyword-only, default (1.0, 1.2, 1.5, 2.0, 3.0)
        Hidden-bias grid for :func:`rosenbaum_bounds`.
    tests : sequence of str, keyword-only, optional
        Subset of ``{"placebo", "random_common_cause", "subset", "unobserved_confounder",
        "e_value", "rosenbaum", "contour"}``. Default runs all of them.
    feature_names : sequence of str, keyword-only, optional
        Passed through to :func:`sensitivity_contour` for the benchmark labels.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`SUITE_COLUMNS`: ``name``, ``original``, ``refuted``, ``p_value``,
        ``passed``, ``detail``, ``n_sim``, ``seconds``. ``df.attrs`` carries ``original_ate``,
        ``n_refits``, ``n_failed``, ``target_arm`` and, when available, the partialling-out
        summary and the contour's robustness value.

    Notes
    -----
    A clean sweep of this table is **not** proof of causality. It is the absence of several
    specific kinds of evidence against it, plus two quantified statements about how much hidden
    confounding the conclusion could absorb. The decision memo should quote the E-value and the
    benchmark multiple, not the count of green ticks.
    """
    t_suite = time.perf_counter()
    if X is None or w is None or y is None:
        raise ValueError(
            "run_refutation_suite requires X, w and y; only estimator_factory is optional "
            "(it defaults to the cheap 100-tree T-learner)"
        )
    if estimator_factory is None:
        estimator_factory = cheap_cate_factory(
            random_state=int(as_rng(random_state).integers(0, 2**31 - 1))
        )
    X, w_arr, y_arr = _coerce(X, w, y)
    all_tests = ("placebo", "random_common_cause", "subset", "unobserved_confounder",
                 "e_value", "rosenbaum", "contour")
    chosen = tuple(all_tests) if tests is None else tuple(t for t in all_tests if t in set(tests))
    if tests is not None and len(chosen) != len(set(tests)):
        unknown = sorted(set(tests) - set(all_tests))
        if unknown:
            LOGGER.warning("run_refutation_suite: ignoring unknown test name(s) %s", unknown)

    streams = spawn_rngs(random_state, 8)
    y_sd = float(np.std(y_arr))
    rows: list[dict[str, Any]] = []
    attrs: dict[str, Any] = {"target_arm": int(target_arm), "n": int(len(y_arr))}
    n_refits = 0
    n_failed = 0

    def _record(name: str, exc: BaseException, original: float) -> None:
        nonlocal n_failed
        n_failed += 1
        msg = f"ERROR: {type(exc).__name__}: {exc}"
        LOGGER.error("run_refutation_suite: %s failed -- %s", name, msg)
        LOGGER.debug("%s", traceback.format_exc())
        rows.append(
            RefutationResult(
                name=name,
                original=original,
                refuted=float("nan"),
                p_value=None,
                passed=False,
                detail=msg[:600],
                n_sim=0,
                seconds=0.0,
            ).to_row()
        )

    # ---- the one shared original fit --------------------------------------------------
    try:
        original = _fit_and_ate(estimator_factory, X, w_arr, y_arr, target_arm=target_arm)
        n_refits += 1
    except Exception as exc:  # noqa: BLE001 - the suite must never raise
        LOGGER.error("run_refutation_suite: the original fit failed -- %s", exc)
        for name in chosen:
            _record(name, exc, float("nan"))
        out = pd.DataFrame(rows, columns=list(SUITE_COLUMNS))
        out.attrs.update({**attrs, "original_ate": float("nan"), "n_refits": 0,
                          "n_failed": n_failed})
        return out
    attrs["original_ate"] = float(original)
    LOGGER.info(
        "run_refutation_suite: original ATE for arm %d vs control = %+.5f on n=%d "
        "(1 shared fit; refuters receive it rather than refitting)",
        target_arm, original, len(y_arr),
    )

    # ---- shared partialling-out fit (se for the E-value CI and the contour) -----------
    plm: dict[str, float] | None = None
    try:
        d_all = (w_arr == int(target_arm)).astype(np.float64)
        mask = np.flatnonzero((w_arr == 0) | (w_arr == int(target_arm)))
        plm = _partial_linear_ate(
            _take_rows(X, mask), d_all[mask], y_arr[mask], random_state=streams[7]
        )
        attrs["partial_linear"] = dict(plm)
        LOGGER.info(
            "run_refutation_suite: partialling-out reference fit tau=%+.5f se=%.5f "
            "(95%% CI [%+.5f, %+.5f]) -- shared by the E-value and the contour",
            plm["estimate"], plm["se"], plm["ci_low"], plm["ci_high"],
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("run_refutation_suite: partialling-out fit unavailable (%s)", exc)

    # ---- falsification tests -----------------------------------------------------------
    if "placebo" in chosen:
        k = int(n_sim) if n_sim is not None else 20
        try:
            res = placebo_treatment(estimator_factory, X, w_arr, y_arr, n_sim=k,
                                    random_state=streams[0], target_arm=target_arm,
                                    original=original)
            n_refits += res.n_sim
            rows.append(res.to_row())
        except Exception as exc:  # noqa: BLE001
            _record("placebo_treatment", exc, original)

    if "random_common_cause" in chosen:
        k = int(n_sim) if n_sim is not None else 10
        try:
            res = random_common_cause(estimator_factory, X, w_arr, y_arr, n_sim=k,
                                      random_state=streams[1], target_arm=target_arm,
                                      original=original)
            n_refits += res.n_sim
            rows.append(res.to_row())
        except Exception as exc:  # noqa: BLE001
            _record("random_common_cause", exc, original)

    if "subset" in chosen:
        k = int(n_sim) if n_sim is not None else 10
        try:
            res = subset_refuter(estimator_factory, X, w_arr, y_arr, fraction=fraction, n_sim=k,
                                 random_state=streams[2], target_arm=target_arm,
                                 original=original)
            n_refits += res.n_sim
            rows.append(res.to_row())
        except Exception as exc:  # noqa: BLE001
            _record("subset_refuter", exc, original)

    if "unobserved_confounder" in chosen:
        for j, s in enumerate(confounding_strengths):
            try:
                res = add_unobserved_confounder(
                    estimator_factory, X, w_arr, y_arr, float(s),
                    random_state=spawn_rngs(streams[3], len(confounding_strengths))[j],
                    target_arm=target_arm, original=original,
                )
                n_refits += res.n_sim
                rows.append(res.to_row())
            except Exception as exc:  # noqa: BLE001
                _record(f"add_unobserved_confounder(s={float(s):g})", exc, original)

    # ---- sensitivity analyses ----------------------------------------------------------
    if "e_value" in chosen:
        try:
            if plm is None:
                raise RuntimeError("no standard error available (the partialling-out fit failed)")
            ev = e_value(
                original,
                original - 1.959963985 * plm["se"],
                original + 1.959963985 * plm["se"],
                scale="difference",
                outcome_sd=y_sd,
            )
            passed = bool(ev["e_value_ci"] > 1.0)
            detail = (
                f"refuted column = E-value of the point estimate. "
                f"point {ev['e_value_point']:.3f}, CI bound nearest the null "
                f"{ev['e_value_ci']:.3f} ({ev['bound_used']}). {ev['conversion']}. "
                f"Scale assumed: {ev['scale_assumed']}. {ev['interpretation'][:260]}"
            )
            rows.append(
                RefutationResult(
                    name="e_value", original=original, refuted=float(ev["e_value_point"]),
                    p_value=None, passed=passed, detail=detail, n_sim=0, seconds=0.0,
                ).to_row()
            )
            attrs["e_value"] = {k2: v for k2, v in ev.items() if k2 != "interpretation"}
        except Exception as exc:  # noqa: BLE001
            _record("e_value", exc, original)

    if "rosenbaum" in chosen:
        try:
            Z, _ = _design(X)
            prog = _cheap_gbm(
                "regression", seed=int(streams[4].integers(0, 2**31 - 1)),
                n_estimators=100, learning_rate=0.1, max_depth=4, n_jobs=_GBM_N_JOBS,
            )
            ctrl = w_arr == 0
            prog.fit(Z[ctrl], y_arr[ctrl])
            score = np.asarray(prog.predict(Z), dtype=np.float64)
            rb = rosenbaum_bounds(y_arr, w_arr, gammas, target_arm=target_arm, score=score)
            gstar = float(rb.attrs["gamma_star"])
            p1 = float(rb.loc[rb["gamma"] == rb["gamma"].min(), "p_upper"].iloc[0])
            # "Passed" means the result survives SOME hidden bias, not merely that it is
            # significant when you assume there is none. A finding that evaporates at
            # Gamma = 1.2 has not passed a sensitivity analysis. (If the caller tabulated no
            # Gamma above 1 there is nothing to survive, so fall back to significance at 1.)
            g_max = float(rb["gamma"].max())
            passed = bool(np.isfinite(gstar) and (gstar > 1.0 if g_max > 1.0 else True))
            detail = (
                f"refuted column = Gamma*, the largest tabulated hidden bias at which the "
                f"result stays significant at alpha={rb.attrs['alpha']:g}. "
                f"{rb.attrs['n_pairs']} pairs via {rb.attrs['matching']}; pairs matched on a "
                f"prognostic score fit on CONTROLS only, so the score is not contaminated by the "
                f"treatment. one-sided '{rb.attrs['alternative']}'; p at Gamma=1 is {p1:.4g}; "
                f"Hodges-Lehmann effect {rb.attrs['hodges_lehmann']:+.4f}. "
                + (f"Gamma* = {gstar:g}, so the finding survives hidden bias of that magnitude."
                   if np.isfinite(gstar) and gstar > 1.0
                   else "Gamma* = 1: significant only under the assumption of NO hidden bias, "
                        "which is exactly the assumption this analysis exists to question."
                   if np.isfinite(gstar)
                   else "Not significant even at Gamma=1, so no breakdown point exists.")
            )
            rows.append(
                RefutationResult(
                    name="rosenbaum_bounds", original=original,
                    refuted=gstar if np.isfinite(gstar) else 1.0,
                    p_value=p1, passed=passed, detail=detail, n_sim=0, seconds=0.0,
                ).to_row()
            )
            attrs["rosenbaum"] = dict(rb.attrs)
        except Exception as exc:  # noqa: BLE001
            _record("rosenbaum_bounds", exc, original)

    if "contour" in chosen:
        try:
            cont = sensitivity_contour(
                estimator_factory, X, w_arr, y_arr, target_arm=target_arm,
                feature_names=feature_names, estimate=original,
                se=None if plm is None else plm["se"],
                dof=None if plm is None else plm["dof"],
                random_state=streams[5],
            )
            rv = float(cont.attrs["robustness_value"])
            bench = cont.loc[cont["benchmark"]]
            if len(bench):
                worst = bench.loc[bench["nullify_multiple"].idxmin()]
                k_worst = float(worst["nullify_multiple"])
                bench_txt = (
                    f"strongest observed covariate is {worst['feature']} "
                    f"(partial R2 with y={float(worst['r2_y']):.4f}, with w="
                    f"{float(worst['r2_w']):.4f}); a confounder would need to be "
                    f"{k_worst:.2f}x as strong as it, on both axes, to erase the effect."
                )
                passed = bool(k_worst > 1.0)
            else:
                k_worst = float("nan")
                bench_txt = "no covariate benchmarks were computable."
                passed = bool(rv > 0.0)
            detail = (
                f"refuted column = robustness value RV(q=1)={rv:.4f}: a confounder explaining "
                f"less than {rv:.1%} of BOTH the residual outcome and the treatment cannot "
                f"overturn the estimate (RV at alpha=0.05 is "
                f"{float(cont.attrs['robustness_value_alpha']):.4f}). {bench_txt} "
                f"Linear (Cinelli-Hazlett) approximation around the partialling-out fit."
            )
            rows.append(
                RefutationResult(
                    name="sensitivity_contour", original=original, refuted=rv,
                    p_value=None, passed=passed, detail=detail, n_sim=0, seconds=0.0,
                ).to_row()
            )
            attrs["contour"] = {
                "robustness_value": rv,
                "robustness_value_alpha": float(cont.attrs["robustness_value_alpha"]),
                "min_nullify_multiple": k_worst,
                "n_benchmarks": int(len(bench)),
            }
            if plm is None:
                n_refits += 1
        except Exception as exc:  # noqa: BLE001
            _record("sensitivity_contour", exc, original)

    out = pd.DataFrame(rows, columns=list(SUITE_COLUMNS))
    attrs.update({"n_refits": int(n_refits), "n_failed": int(n_failed),
                  "seconds": round(time.perf_counter() - t_suite, 3)})
    out.attrs.update(attrs)
    LOGGER.info(
        "run_refutation_suite: %d tests, %d passed, %d errored | %d model refits in total "
        "| %.2fs",
        len(out), int(out["passed"].sum()), n_failed, n_refits, time.perf_counter() - t_suite,
    )
    return out


# ======================================================================================
# smoke test
# ======================================================================================
def _simulate_refutation_data(
    n: int = 3000,
    effect: float = 1.0,
    *,
    confounding: float = 0.7,
    noise: float = 1.0,
    random_state: int | np.random.Generator | None = 11,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    """Confounded two-arm data with an exactly known average treatment effect.

    The generating process, written out because the smoke test asserts against it:

    .. code-block:: text

        X_j          ~ N(0, 1), j = 1..5, named after real PRISM features
        logit P(W=1) = c * ( 0.9 x1 - 0.7 x2 + 0.5 x3 )        <- confounded assignment
        mu0(x)       = x1 + 0.7 x2 - 0.5 x3 + 0.3 x1 x2        <- non-linear baseline
        tau(x)       = effect * (1 + 0.5 x4)                   <- heterogeneous effect
        Y            = mu0(x) + W tau(x) + noise * N(0, 1)

    Because ``E[x4] = 0`` the true ATE is ``effect`` up to sampling error, and the realised
    sample ATE ``mean(tau(X_i))`` is returned exactly. The same three covariates drive both
    assignment and the outcome, so a learner that ignores ``X`` is badly biased and a learner
    that adjusts for it is not -- which is what makes the null panel a meaningful test.

    Parameters
    ----------
    n : int, default 3000
        Rows.
    effect : float, default 1.0
        Average treatment effect. ``0.0`` builds the null panel.
    confounding : float, keyword-only, default 0.7
        Strength of the covariate-driven selection into treatment.
    noise : float, keyword-only, default 1.0
        Outcome noise standard deviation.
    random_state : int or numpy.random.Generator or None, keyword-only, default 11
        Seed.

    Returns
    -------
    (pandas.DataFrame, numpy.ndarray, numpy.ndarray, float)
        ``(X, w, y, true_ate)``.
    """
    rng = as_rng(random_state)
    cols = list(FEATURES[:5])
    Z = rng.standard_normal((n, 5))
    logit = confounding * (0.9 * Z[:, 0] - 0.7 * Z[:, 1] + 0.5 * Z[:, 2])
    w = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int64)
    mu0 = Z[:, 0] + 0.7 * Z[:, 1] - 0.5 * Z[:, 2] + 0.3 * Z[:, 0] * Z[:, 1]
    tau = float(effect) * (1.0 + 0.5 * Z[:, 3])
    y = mu0 + w * tau + float(noise) * rng.standard_normal(n)
    return pd.DataFrame(Z, columns=cols), w, y, float(tau.mean())


def _check(rows: list[dict[str, Any]], name: str, ok: bool, detail: str) -> bool:
    """Append a pass/fail row to the smoke-test table."""
    rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    return bool(ok)


class _BrokenLearner:
    """A learner that always raises, to prove the suite degrades instead of exploding."""

    n_arms = 2

    def fit(self, X: Any, w: Any, y: Any, **kw: Any) -> None:  # noqa: D102
        raise RuntimeError("deliberately broken learner")

    def predict_cate(self, X: Any) -> np.ndarray:  # noqa: D102
        raise RuntimeError("deliberately broken learner")


if __name__ == "__main__":  # pragma: no cover - smoke test
    t_start = time.perf_counter()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)

    # Simulation counts are kept deliberately low: SPEC_PERF section 8 caps refutation work,
    # and this whole file must run in under 60 s. The suites exist to exercise the tidy-frame
    # contract and to be printed; the sharp statistical assertions use the standalone calls
    # below, which get the larger counts where the extra draws actually buy something.
    N, CONF, SEED = 3000, 0.7, 11
    NSIM_SUITE, NSIM_PLACEBO, NSIM_STABILITY = 6, 10, 8
    FEATS = list(FEATURES[:5])
    checks: list[dict[str, Any]] = []
    ok = True

    # Two panels: identical covariates, identical confounding, only the effect differs.
    X_real, w_real, y_real, true_real = _simulate_refutation_data(
        N, effect=1.0, confounding=CONF, random_state=SEED
    )
    X_null, w_null, y_null, true_null = _simulate_refutation_data(
        N, effect=0.0, confounding=CONF, random_state=SEED
    )
    factory = cheap_cate_factory(n_arms=2, random_state=7)

    print("=" * 108)
    print("PRISM refutation suite -- smoke test")
    print("=" * 108)
    print(
        f"Two panels, n={N} x {len(FEATS)} features, confounding={CONF}, seed={SEED}.\n"
        f"  REAL panel: known sample ATE = {true_real:+.5f}   (heterogeneous tau = 1 + 0.5*x4)\n"
        f"  NULL panel: known sample ATE = {true_null:+.5f}   (identical DGP, effect switched off)\n"
        f"Both are confounded through {FEATS[0]}, {FEATS[1]} and {FEATS[2]}, so an unadjusted\n"
        f"comparison is biased on both panels and only adjustment can tell them apart."
    )

    # --- 1. the estimator recovers the known ATE ------------------------------------
    ate_real = _fit_and_ate(factory, X_real, w_real, y_real, target_arm=1)
    ate_null = _fit_and_ate(factory, X_null, w_null, y_null, target_arm=1)
    ok &= _check(
        checks, "known truth: real ATE recovered", abs(ate_real - true_real) < 0.15,
        f"true {true_real:+.4f} vs estimated {ate_real:+.4f} (error {ate_real - true_real:+.4f})",
    )
    ok &= _check(
        checks, "known truth: null ATE is ~0", abs(ate_null) < 0.15 and abs(ate_null) < 0.25 * abs(ate_real),
        f"true {true_null:+.4f} vs estimated {ate_null:+.4f} "
        f"({abs(ate_null) / max(abs(ate_real), _EPS):.1%} of the real effect)",
    )

    # --- 2. the two suites ------------------------------------------------------------
    suite_real = run_refutation_suite(factory, X_real, w_real, y_real, random_state=5,
                                      n_sim=NSIM_SUITE, feature_names=FEATS)
    suite_null = run_refutation_suite(factory, X_null, w_null, y_null, random_state=5,
                                      n_sim=NSIM_SUITE, feature_names=FEATS)
    ok &= _check(
        checks, "suite schema", tuple(suite_real.columns) == SUITE_COLUMNS,
        f"columns {list(suite_real.columns)}",
    )
    ok &= _check(
        checks, "suite never raised", int(suite_real.attrs["n_failed"]) == 0
        and int(suite_null.attrs["n_failed"]) == 0,
        f"real {len(suite_real)} rows / {suite_real.attrs['n_refits']} refits, "
        f"null {len(suite_null)} rows / {suite_null.attrs['n_refits']} refits, 0 errors",
    )

    # --- 3. placebo behaves in OPPOSITE ways on the two panels -----------------------
    pl_real = placebo_treatment(factory, X_real, w_real, y_real, n_sim=NSIM_PLACEBO,
                                random_state=3, original=ate_real)
    pl_null = placebo_treatment(factory, X_null, w_null, y_null, n_sim=NSIM_PLACEBO,
                                random_state=3, original=ate_null)
    ok &= _check(
        checks, "placebo: permuted w cannot reproduce a real effect",
        pl_real.passed and pl_real.p_value == 0.0
        and abs(pl_real.extra["placebo_mean"]) < 0.1 * abs(ate_real),
        f"original {ate_real:+.4f} vs placebo mean {pl_real.extra['placebo_mean']:+.4f} "
        f"(sd {pl_real.extra['placebo_sd']:.4f}, z={pl_real.extra['z_score']:+.1f}, "
        f"p={pl_real.p_value:.3f})",
    )
    ok &= _check(
        checks, "placebo is NOT a rubber stamp: null panel is not flagged",
        (not pl_null.passed) and pl_null.p_value > 0.10,
        f"original {ate_null:+.4f} sits INSIDE the placebo mass "
        f"[{pl_null.extra['ci_low']:+.4f}, {pl_null.extra['ci_high']:+.4f}] at "
        f"z={pl_null.extra['z_score']:+.2f}, p={pl_null.p_value:.3f} -> passed={pl_null.passed}",
    )
    ok &= _check(
        checks, "placebo separates the panels by orders of magnitude",
        abs(pl_real.extra["z_score"]) > 5.0 * max(abs(pl_null.extra["z_score"]), 1.0),
        f"|z| real {abs(pl_real.extra['z_score']):.1f} vs null {abs(pl_null.extra['z_score']):.2f}",
    )

    # --- 4. random common cause must NOT move the estimate ---------------------------
    rcc = random_common_cause(factory, X_real, w_real, y_real, n_sim=NSIM_STABILITY,
                              random_state=4, original=ate_real)
    ok &= _check(
        checks, "random common cause barely moves the estimate",
        rcc.passed and rcc.extra["mean_rel_change"] < _DEFAULT_REL_TOL,
        f"ATE {ate_real:+.4f} -> {rcc.refuted:+.4f}; mean relative change "
        f"{rcc.extra['mean_rel_change']:.3%} (max {rcc.extra['max_rel_change']:.3%}) "
        f"vs tolerance {rcc.extra['tolerance']:.0%}",
    )

    # --- 5. subset estimates straddle the original -----------------------------------
    sub = subset_refuter(factory, X_real, w_real, y_real, fraction=0.7,
                         n_sim=NSIM_STABILITY, random_state=6, original=ate_real)
    draws = sub.extra["ates"]
    ok &= _check(
        checks, "subset estimates straddle the original",
        bool(draws.min() < ate_real < draws.max()) and sub.passed,
        f"{NSIM_STABILITY} subsets span [{draws.min():+.4f}, {draws.max():+.4f}] around {ate_real:+.4f}; "
        f"sd {sub.extra['sd']:.4f} (rescaled {sub.extra['sd_rescaled']:.4f}), "
        f"original at the {sub.extra['percentile']:.0%} percentile",
    )

    # --- 6. the injected confounder biases the estimate by EXACTLY the predicted amount
    conf_rows = []
    conf_ok = True
    for s in (0.20, 0.35, 0.50):
        r = add_unobserved_confounder(factory, X_real, w_real, y_real, s,
                                      random_state=8, original=ate_real)
        conf_ok &= bool(r.extra["bias_as_predicted"])
        conf_rows.append(
            f"s={s:.2f}: shift {r.extra['shift']:+.4f} vs closed form "
            f"{r.extra['predicted_shift']:+.4f} (gap {r.extra['gap']:.4f})"
        )
    ok &= _check(
        checks, "known truth: injected bias == s^2*sd(y)/sd(D)", conf_ok, "; ".join(conf_rows)
    )

    # --- 7. E-value: monotone away from the null, symmetric in log RR ----------------
    ladder = [e_value(v, v - 0.25, v + 0.25, scale="difference", outcome_sd=float(np.std(y_real)))
              for v in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)]
    e_points = [r["e_value_point"] for r in ladder]
    ok &= _check(
        checks, "E-value rises monotonically away from the null",
        all(b > a for a, b in zip(e_points, e_points[1:], strict=False)) and abs(e_points[0] - 1.0) < 1e-12,
        "estimate 0.00->2.00 gives E = " + ", ".join(f"{v:.3f}" for v in e_points),
    )
    hi_rr, lo_rr = e_value(2.0, 1.5, 2.7, scale="rr"), e_value(0.5, 1 / 2.7, 1 / 1.5, scale="rr")
    ok &= _check(
        checks, "E-value is symmetric in log RR (reciprocal handling)",
        abs(hi_rr["e_value_point"] - lo_rr["e_value_point"]) < 1e-9
        and abs(hi_rr["e_value_point"] - (2.0 + np.sqrt(2.0))) < 1e-9,
        f"RR=2 -> {hi_rr['e_value_point']:.6f}, RR=0.5 -> {lo_rr['e_value_point']:.6f}, "
        f"closed form 2+sqrt(2) = {2 + np.sqrt(2):.6f}",
    )
    crosses = e_value(1.2, 0.8, 1.9, scale="rr")
    ok &= _check(
        checks, "E-value for a CI covering the null is exactly 1",
        crosses["e_value_ci"] == 1.0 and crosses["bound_used"].startswith("none"),
        f"RR=1.2 CI [0.8, 1.9] -> E_point {crosses['e_value_point']:.3f}, "
        f"E_ci {crosses['e_value_ci']:.3f}, bound '{crosses['bound_used']}'",
    )
    ok &= _check(
        checks, "E-value picks the CI bound NEAREST the null",
        e_value(2.0, 1.5, 2.7, scale="rr")["bound_used"] == "lower"
        and e_value(0.5, 0.3, 0.7, scale="rr")["bound_used"] == "upper",
        "RR>1 -> lower bound; RR<1 -> upper bound",
    )
    diff_ev = e_value(1.0, 0.8, 1.2, scale="difference", outcome_sd=1.8)
    auto_ev = e_value(1.0, 0.8, 1.2)
    ok &= _check(
        checks, "E-value declares the scale it assumed, loudly",
        diff_ev["converted"] and diff_ev["approximate"]
        and "difference" in diff_ev["scale_assumed"]
        and bool(auto_ev["scale_inferred"]) and bool(auto_ev["warning"])
        and "risk ratio" in auto_ev["scale_assumed"],
        f"explicit difference -> '{diff_ev['scale_assumed']}', approximate="
        f"{diff_ev['approximate']}; auto on positive inputs -> "
        f"'{auto_ev['scale_assumed']}' WITH a warning of {len(auto_ev['warning'])} chars",
    )

    # --- 8. Rosenbaum bounds ----------------------------------------------------------
    prog_score = np.asarray(
        _cheap_gbm("regression", seed=5, n_estimators=100, learning_rate=0.1, max_depth=4,
                   n_jobs=_GBM_N_JOBS).fit(_design(X_real)[0][w_real == 0], y_real[w_real == 0])
        .predict(_design(X_real)[0]),
        dtype=np.float64,
    )
    rb = rosenbaum_bounds(y_real, w_real, (1.0, 1.2, 1.5, 2.0, 3.0, 5.0),
                          score=prog_score, alternative="greater")
    ok &= _check(
        checks, "Rosenbaum: p_lower <= p_upper, both monotone in Gamma",
        bool((rb["p_lower"] <= rb["p_upper"] + 1e-12).all())
        and bool((np.diff(rb["p_upper"].to_numpy()) >= -1e-12).all())
        and bool((np.diff(rb["p_lower"].to_numpy()) <= 1e-12).all()),
        "p_upper " + " -> ".join(f"{v:.3g}" for v in rb["p_upper"]),
    )
    ok &= _check(
        checks, "Rosenbaum: bounds coincide at Gamma = 1 (no hidden bias)",
        abs(float(rb.loc[0, "p_lower"]) - float(rb.loc[0, "p_upper"])) < 1e-12,
        f"p_lower = p_upper = {float(rb.loc[0, 'p_upper']):.3g} at Gamma=1",
    )
    # Cross-validate the Gamma = 1 row against scipy's independent signed-rank implementation.
    sci_p = float(
        stats.wilcoxon(rb.attrs["pair_differences"], alternative="greater",
                       method="approx", correction=False).pvalue
    )
    ok &= _check(
        checks, "Rosenbaum at Gamma=1 matches scipy.stats.wilcoxon",
        abs(float(rb.loc[0, "p_upper"]) - sci_p) < 1e-6,
        f"ours {float(rb.loc[0, 'p_upper']):.6g} vs scipy {sci_p:.6g} "
        f"(independent implementation, normal approximation, no continuity correction)",
    )
    ok &= _check(
        checks, "Rosenbaum: real effect survives strong hidden bias",
        np.isfinite(float(rb.attrs["gamma_star"])) and float(rb.attrs["gamma_star"]) >= 2.0,
        f"significant up to Gamma* = {rb.attrs['gamma_star']:g}; "
        f"{rb.attrs['n_pairs']} pairs, Hodges-Lehmann {rb.attrs['hodges_lehmann']:+.4f} "
        f"(true ATE {true_real:+.4f})",
    )
    rb_null = rosenbaum_bounds(y_null, w_null, (1.0, 1.5, 2.0), score=prog_score,
                               alternative="greater")
    ok &= _check(
        checks, "Rosenbaum is not a rubber stamp either: weaker Gamma* on the null panel",
        (not np.isfinite(float(rb_null.attrs["gamma_star"])))
        or float(rb_null.attrs["gamma_star"]) < float(rb.attrs["gamma_star"]),
        f"null Gamma* = {rb_null.attrs['gamma_star']} vs real {rb.attrs['gamma_star']}",
    )

    # --- 9. sensitivity contour and its benchmarks ------------------------------------
    cont = sensitivity_contour(factory, X_real, w_real, y_real, feature_names=FEATS,
                               random_state=9)
    bench = cont.loc[cont["benchmark"]]
    grid = cont.loc[~cont["benchmark"]]
    origin = grid.loc[(grid["r2_y"] == 0.0) & (grid["r2_w"] == 0.0)]
    ok &= _check(
        checks, "contour includes a benchmark row per real covariate",
        list(bench["feature"]) == FEATS and len(bench) == len(FEATS),
        "benchmarks: " + ", ".join(
            f"{r.feature}(R2y={r.r2_y:.3f}, R2w={r.r2_w:.3f}, needs {r.nullify_multiple:.2f}x)"
            for r in bench.itertuples()
        ),
    )
    ok &= _check(
        checks, "contour is zero-bias at the origin and nullifies in the corner",
        len(origin) == 1 and abs(float(origin["bias"].iloc[0])) < 1e-12
        and abs(float(origin["adjusted_estimate"].iloc[0]) - cont.attrs["estimate"]) < 1e-12
        and bool(grid["nullified"].any()) and not bool(origin["nullified"].iloc[0]),
        f"origin bias {float(origin['bias'].iloc[0]):.2e}, "
        f"{int(grid['nullified'].sum())}/{len(grid)} grid points nullify the estimate",
    )
    piv = grid.pivot_table(index="r2_y", columns="r2_w", values="bias", aggfunc="mean").abs()
    ok &= _check(
        checks, "|bias| increases in both partial R-squared directions",
        bool((np.diff(piv.to_numpy(), axis=0) >= -1e-12).all())
        and bool((np.diff(piv.to_numpy(), axis=1) >= -1e-12).all()),
        f"monotone over a {piv.shape[0]}x{piv.shape[1]} grid up to r2={grid['r2_y'].max():.3f}",
    )
    rv = float(cont.attrs["robustness_value"])
    ok &= _check(
        checks, "robustness value reproduces the bias formula exactly",
        0.0 < rv < 1.0
        and abs(float(_ovb_bias(np.array([rv]), np.array([rv]),
                                cont.attrs["se"], cont.attrs["dof"])[0])
                - abs(cont.attrs["estimate"])) < 1e-9,
        f"RV(q=1) = {rv:.4f}; bias at (RV, RV) equals |estimate| = "
        f"{abs(cont.attrs['estimate']):.5f} to 1e-9",
    )
    worst = bench.loc[bench["nullify_multiple"].idxmin()]
    ok &= _check(
        checks, "benchmark multiple is self-consistent",
        abs(float(_ovb_bias(np.array([worst.r2_y * worst.nullify_multiple]),
                            np.array([worst.r2_w * worst.nullify_multiple]),
                            cont.attrs["se"], cont.attrs["dof"])[0])
            - abs(cont.attrs["estimate"])) < 1e-8,
        f"a confounder {float(worst['nullify_multiple']):.2f}x as strong as "
        f"{worst['feature']} lands exactly on the nullification contour",
    )

    # --- 10. determinism ---------------------------------------------------------------
    a = placebo_treatment(cheap_cate_factory(n_arms=2, random_state=7), X_real, w_real, y_real,
                          n_sim=3, random_state=3)
    b = placebo_treatment(cheap_cate_factory(n_arms=2, random_state=7), X_real, w_real, y_real,
                          n_sim=3, random_state=3)
    ok &= _check(
        checks, "bit-identical under the same seed",
        bool(np.array_equal(a.extra["placebo_ates"], b.extra["placebo_ates"]))
        and a.original == b.original and a.p_value == b.p_value,
        f"max abs diff = {np.max(np.abs(a.extra['placebo_ates'] - b.extra['placebo_ates'])):.2e}",
    )

    # --- 11. a broken refuter must not take the report down ----------------------------
    broken = run_refutation_suite(lambda: _BrokenLearner(), X_real, w_real, y_real,
                                  random_state=1, n_sim=2)
    ok &= _check(
        checks, "suite degrades gracefully when the estimator raises",
        len(broken) > 0 and not bool(broken["passed"].any())
        and bool(broken["detail"].str.startswith("ERROR").all()),
        f"{len(broken)} rows, all passed=False, all details start with ERROR "
        f"(e.g. {broken['detail'].iloc[0][:60]}...)",
    )

    # --- output --------------------------------------------------------------------------
    show = ["name", "original", "refuted", "p_value", "passed", "n_sim", "seconds"]
    print("\n" + "-" * 108)
    print(f"REFUTATION SUITE -- REAL-EFFECT PANEL (true ATE = {true_real:+.4f})")
    print("-" * 108)
    print(suite_real[show].round(4).to_string(index=False))
    print("\n" + "-" * 108)
    print(f"REFUTATION SUITE -- NULL PANEL (true ATE = {true_null:+.4f})")
    print("-" * 108)
    print(suite_null[show].round(4).to_string(index=False))

    side = suite_real[["name", "original", "refuted", "p_value", "passed"]].merge(
        suite_null[["name", "original", "refuted", "p_value", "passed"]],
        on="name", how="outer", suffixes=("_real", "_null"),
    )
    print("\n" + "-" * 108)
    print("SIDE BY SIDE -- the two panels differ only in whether the treatment does anything")
    print("-" * 108)
    print(side.round(4).to_string(index=False))

    print("\n--- sensitivity contour, observed-covariate benchmarks (real panel) ---")
    print(
        bench[["feature", "r2_y", "r2_w", "bias", "adjusted_estimate", "nullified",
               "nullify_multiple"]].round(5).to_string(index=False)
    )
    print(
        f"robustness value RV(q=1) = {rv:.4f} | RV at alpha=0.05 = "
        f"{float(cont.attrs['robustness_value_alpha']):.4f} | "
        f"headline: a confounder would need to be {float(worst['nullify_multiple']):.2f}x as "
        f"strong as {worst['feature']} to erase this effect."
    )

    print("\n--- Rosenbaum bounds (real panel, prognostic-score pairs) ---")
    print(rb[["gamma", "p_lower", "p_upper", "still_significant", "n_pairs"]].to_string(index=False))
    print(f"matching: {rb.attrs['matching']}")

    print("\n--- checks ---")
    table = pd.DataFrame(checks)
    print(table.to_string(index=False))
    elapsed = time.perf_counter() - t_start
    n_pass = int((table["status"] == "PASS").sum())
    print(
        f"\nrefute.py {'OK' if ok else 'FAILED'} -> {n_pass}/{len(table)} checks passed in "
        f"{elapsed:.1f}s | real-panel refits {suite_real.attrs['n_refits']}, "
        f"null-panel refits {suite_null.attrs['n_refits']}"
    )
    if not ok:
        raise SystemExit(1)
    if elapsed > 60.0:
        raise SystemExit(f"smoke test took {elapsed:.1f}s, over the 60s budget")
