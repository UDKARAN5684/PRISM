"""Multi-arm CATE meta-learners: S, T, X, DR and R.

This module is the entry point to PRISM's causal layer.  It estimates, for every non-control
arm ``a`` and every customer ``x``, the conditional average treatment effect

.. math:: \\tau_a(x) = E[\\, Y(a) - Y(0) \\mid X = x \\,]

from an observational log in which the arm was **not** randomised.  Five estimators are
implemented directly on top of scikit-learn rather than imported from a causal-ML package, so
that the cross-fitting, the pseudo-outcomes and the propensity handling are all visible and
auditable.  Everything here follows the notation of `SPEC.md` section 4 and the maths of
`docs/METHODOLOGY.md` section 4.1.

Estimands and identification
----------------------------
``tau_a(x)`` is identified from observational data under consistency/SUTVA, positivity
(``0 < e_a(x) < 1`` for every arm) and unconfoundedness (``{Y(0), ..., Y(K-1)} _||_ A | X``).
Nothing in this module can test the third assumption -- that is what
:mod:`prism.causal.refute` is for -- but two things here make its failure *visible*:

* every learner exposes :attr:`BaseCATELearner.nuisance_`, which reports how good the nuisance
  models actually are (out-of-fold outcome ``R^2``, propensity log-loss versus the marginal,
  worst inverse-propensity weight, effective sample size);
* the doubly-robust learners log a warning when the outcome model is close to useless, because a
  DR point estimate computed from a nuisance model with ``R^2 ~= 0`` is a number that looks
  authoritative and is not.

The five learners
-----------------
================ ========================================================= ======================
learner          construction                                              main failure mode
================ ========================================================= ======================
:class:`SLearner` one model ``mu(X, A)``; contrast its own predictions      shrinks tau toward 0
:class:`TLearner` one model per arm; subtract                               variance adds up
:class:`XLearner` impute unit-level effects, blend by propensity            two-stage error
:class:`DRLearner` cross-fitted AIPW pseudo-outcome, regressed on X         needs overlap
:class:`RLearner` Robinson residualisation, weighted least squares          needs a flexible m_hat
================ ========================================================= ======================

``DRLearner`` and ``RLearner`` are Neyman-orthogonal: a first-order error in either nuisance does
not propagate to first order in ``tau_hat``.  That is why they are the defaults, and why they are
expected to top the leaderboard in the smoke test at the bottom of this file.

Shared interface (SPEC.md section 4)
------------------------------------
Every learner implements exactly::

    fit(X, w, y, *, propensity=None, sample_weight=None) -> Self
    predict_cate(X) -> np.ndarray          # (n, n_arms - 1), one column per non-control arm
    predict(X) -> np.ndarray               # alias of predict_cate
    n_arms: int

The pipeline loops over learners uniformly, so the signature is a contract, not a suggestion.

Multi-arm handling
------------------
``w`` is an integer arm index with ``0`` meaning control.  Column ``j`` of ``predict_cate`` is
``tau_{j+1}``.  Learners that are natively binary (X and R) are fitted once per non-control arm
on the rows with ``w in {0, a}``; the multi-arm nuisances they need on that subsample are
*derived* from the cached ``K``-arm nuisances rather than refitted (see :class:`RLearner`).

Performance (SPEC_PERF.md section 6)
------------------------------------
Nuisances are fitted **once per fold** and the out-of-fold predictions are cached in an
``(n, K)`` array which every arm then reuses.  With ``K`` arms and ``n_splits`` folds the cost is
``n_splits * (K + 1)`` model fits, not ``n_splits * K * (K - 1)``.  Folds come from
:class:`~sklearn.model_selection.StratifiedKFold` on the arm label; if a fold would miss an arm,
``n_splits`` is reduced automatically and the reduction is logged (silent truncation is the one
unforgivable sin -- SPEC_PERF section 8).

The fold loop is deliberately **serial**: the inner gradient-boosting models are already
multi-threaded (``n_jobs=-1``), so wrapping the folds in ``joblib.Parallel`` would oversubscribe
the machine and, worse, make the result depend on thread scheduling.  Keeping it serial is what
makes two runs with the same seed bit-identical (SPEC_PERF section 10).

Determinism
-----------
Every stochastic choice is seeded from ``random_state`` through
:func:`prism.utils.seeds.as_rng` and :func:`prism.utils.seeds.spawn_rngs`, which gives four
independent streams (folds / outcome models / propensity / effect models) so that adding a fold
does not silently reseed the effect models.

Examples
--------
>>> import numpy as np
>>> from prism.causal.learners import make_learner
>>> X, w, y, tau, _ = _simulate_confounded_multiarm(n=2000, n_arms=3, random_state=0)
>>> dr = make_learner("dr", n_arms=3, n_splits=3, n_estimators=60, random_state=0)
>>> _ = dr.fit(X, w, y)
>>> dr.predict_cate(X).shape
(2000, 2)
"""

from __future__ import annotations

import contextlib
import inspect
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold

from prism.data.schema import ARM_NAMES, N_ARMS
from prism.utils.logging import get_logger
from prism.utils.optional import best_gbm
from prism.utils.seeds import as_rng, spawn_rngs

__all__ = [
    "BaseCATELearner",
    "SLearner",
    "TLearner",
    "XLearner",
    "DRLearner",
    "RLearner",
    "make_learner",
    "LEARNER_NAMES",
]

LOGGER = get_logger("causal.learners")

#: Canonical short names accepted by :func:`make_learner`, in leaderboard order.
LEARNER_NAMES: tuple[str, ...] = ("s", "t", "x", "dr", "r", "forest", "survival")

#: Probabilities are clipped to this before any ``log`` (SPEC_PERF section 9).
_LOG_EPS: float = 1e-6
#: Guard used when a variance or a denominator could be exactly zero.
_TINY: float = 1e-12


# ======================================================================================
# small, dependency-free helpers
# ======================================================================================
def _arm_label(a: int, n_arms: int) -> str:
    """Human-readable name for arm ``a``.

    Parameters
    ----------
    a : int
        Arm index.
    n_arms : int
        Number of arms in the current problem. The canonical
        :data:`prism.data.schema.ARM_NAMES` are used only when they line up.

    Returns
    -------
    str
    """
    if n_arms == N_ARMS and 0 <= a < len(ARM_NAMES):
        return ARM_NAMES[a]
    return f"arm_{a}"


def _expand_frame(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One-hot expand the non-numeric columns of a frame into a float64 matrix.

    Parameters
    ----------
    frame : pandas.DataFrame
        Covariates. ``object`` / ``category`` / ``string`` / ``bool`` columns are expanded with
        :func:`pandas.get_dummies`; numeric columns pass through untouched (``NaN`` included --
        every gradient-boosting backend PRISM uses handles missing values natively).

    Returns
    -------
    tuple of (numpy.ndarray, list of str)
        ``(matrix, column_names)``.
    """
    cat = [
        c
        for c in frame.columns
        if pd.api.types.is_bool_dtype(frame[c]) or not pd.api.types.is_numeric_dtype(frame[c])
    ]
    expanded = pd.get_dummies(frame, columns=cat, dummy_na=False, dtype=np.float64) if cat else frame
    names = [str(c) for c in expanded.columns]
    return np.asarray(expanded.to_numpy(dtype=np.float64), dtype=np.float64), names


def _seeded(estimator: Any, seed: int) -> Any:
    """Set ``random_state`` (or ``seed``) on an estimator when it has one.

    Parameters
    ----------
    estimator : object
        An unfitted sklearn-compatible estimator.
    seed : int
        Seed value, reduced into the int32 range that LightGBM/XGBoost accept.

    Returns
    -------
    object
        The same estimator, mutated in place and returned for chaining.
    """
    seed = int(seed) % (2**31 - 1)
    try:
        params = estimator.get_params(deep=False)
    except Exception:  # pragma: no cover - non-sklearn estimator
        return estimator
    for key in ("random_state", "seed"):
        if key in params:
            with contextlib.suppress(Exception):  # pragma: no cover - non-conforming estimator
                estimator.set_params(**{key: seed})
            break
    return estimator


class _SeedStream:
    """A reproducible stream of integer seeds drawn in call order.

    Parameters
    ----------
    random_state : int or numpy.random.Generator or None
        Parent seed material, normalised with :func:`prism.utils.seeds.as_rng`.

    Notes
    -----
    Call order is the only thing that determines the sequence, and the fold loops in this module
    are serial, so the sequence is stable for a given dataset and configuration.
    """

    def __init__(self, random_state: int | np.random.Generator | None) -> None:
        self._rng = as_rng(random_state)

    def __call__(self) -> int:
        """Return the next seed."""
        return int(self._rng.integers(0, 2**31 - 1))

    @property
    def rng(self) -> np.random.Generator:
        """The underlying generator, for helpers that need a ``Generator`` rather than an int."""
        return self._rng


def _accepts_sample_weight(estimator: Any) -> bool:
    """True when ``estimator.fit`` takes a ``sample_weight`` argument."""
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic estimator
        return False


def _fit_model(estimator: Any, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> Any:
    """Fit ``estimator``, forwarding ``sample_weight`` when the estimator supports it.

    Parameters
    ----------
    estimator : object
        Unfitted sklearn-compatible estimator.
    X : numpy.ndarray
        Design matrix, ``(m, p)``.
    y : numpy.ndarray
        Target, ``(m,)``.
    sample_weight : numpy.ndarray or None
        Per-row weights, ``(m,)``. ``None`` means unweighted.

    Returns
    -------
    object
        The fitted estimator.

    Notes
    -----
    Weights are never silently dropped: if the estimator cannot take them, a warning is logged
    naming the estimator class.
    """
    if sample_weight is None:
        return estimator.fit(X, y)
    if _accepts_sample_weight(estimator):
        return estimator.fit(X, y, sample_weight=sample_weight)
    LOGGER.warning(
        "%s.fit() does not accept sample_weight; the weights you passed are being IGNORED by this model",
        type(estimator).__name__,
    )
    return estimator.fit(X, y)


def _clip_simplex(proba: np.ndarray, low: float, high: float) -> np.ndarray:
    """Project each row of ``proba`` into ``[low, high]^K`` while keeping it on the simplex.

    Delegates to :func:`prism.models.propensity.clip_renormalize`, which solves
    ``sum_a clip(s * p_a, low, high) = 1`` per row by bisection.  The import is lazy and a
    plain clip-then-renormalise is used if that module is unavailable, so this file never fails
    to import because of a problem elsewhere.

    Parameters
    ----------
    proba : numpy.ndarray
        ``(n, K)`` probabilities.
    low, high : float
        Clip bounds.

    Returns
    -------
    numpy.ndarray
        ``(n, K)`` float64.
    """
    p = np.asarray(proba, dtype=np.float64)
    try:
        from prism.models.propensity import clip_renormalize

        return np.asarray(clip_renormalize(p, float(low), float(high)), dtype=np.float64)
    except Exception as exc:  # pragma: no cover - only when propensity.py is broken
        LOGGER.warning("clip_renormalize unavailable (%s); using the approximate clip+renormalise", exc)
        q = np.clip(np.nan_to_num(p, nan=1.0 / max(p.shape[1], 1)), low, high)
        total = q.sum(axis=1, keepdims=True)
        return np.clip(q / np.where(total > _TINY, total, 1.0), low, high)


def _expand_proba(estimator: Any, X: np.ndarray, n_arms: int) -> np.ndarray:
    """Score a classifier into a full ``(m, n_arms)`` matrix, filling absent classes with zero."""
    proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    classes = np.asarray(getattr(estimator, "classes_", np.arange(proba.shape[1]))).astype(np.int64)
    out = np.zeros((X.shape[0], int(n_arms)), dtype=np.float64)
    for j, c in enumerate(classes):
        if 0 <= int(c) < int(n_arms):
            out[:, int(c)] = proba[:, j]
    return out


def _weights_or_ones(sample_weight: np.ndarray | None, n: int) -> np.ndarray:
    """Return ``sample_weight`` or a vector of ones, for weighted summary statistics."""
    if sample_weight is None:
        return np.ones(n, dtype=np.float64)
    return np.asarray(sample_weight, dtype=np.float64)


def _weighted_r2(y: np.ndarray, pred: np.ndarray | None, sample_weight: np.ndarray | None) -> float:
    """Weighted coefficient of determination ``1 - SSE / SST``.

    Parameters
    ----------
    y : numpy.ndarray
        Observed outcome, ``(n,)``.
    pred : numpy.ndarray or None
        Predictions, ``(n,)``. ``None`` returns ``nan``.
    sample_weight : numpy.ndarray or None
        Row weights.

    Returns
    -------
    float
        ``nan`` when the weighted variance of ``y`` is zero.
    """
    if pred is None:
        return float("nan")
    sw = _weights_or_ones(sample_weight, y.size)
    mean = float(np.average(y, weights=sw))
    sst = float(np.sum(sw * (y - mean) ** 2))
    sse = float(np.sum(sw * (y - np.asarray(pred, dtype=np.float64)) ** 2))
    return float(1.0 - sse / sst) if sst > _TINY else float("nan")


def _make_folds(
    w: np.ndarray,
    n_splits: int,
    rng: np.random.Generator,
    *,
    context: str = "",
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, list[str]]:
    """Build stratified cross-fitting folds, shrinking ``n_splits`` when an arm is too rare.

    A fold that contains zero rows of some arm makes the per-arm outcome model undefined on that
    fold, so ``n_splits`` is reduced to the size of the rarest arm (never below 2) and the change
    is logged.  When even that is impossible (an arm with a single row) the splitter falls back to
    unstratified :class:`~sklearn.model_selection.KFold` and says so.

    Parameters
    ----------
    w : numpy.ndarray
        Integer arm labels, ``(n,)``.
    n_splits : int
        Requested number of folds.
    rng : numpy.random.Generator
        Source of the shuffle seed.
    context : str, optional
        Label used in the log message, e.g. ``"DRLearner"``.

    Returns
    -------
    tuple
        ``(folds, n_splits_used, warnings)`` where ``folds`` is a list of
        ``(train_index, test_index)`` int arrays.
    """
    n = int(w.size)
    warnings_: list[str] = []
    requested = int(max(2, n_splits))
    counts = np.bincount(w.astype(np.int64), minlength=1)
    present = counts[counts > 0]
    min_count = int(present.min()) if present.size else 0
    used = int(min(requested, max(2, n)))

    if min_count < used:
        new = int(max(2, min(used, min_count)))
        msg = (
            f"{context or 'cross-fitting'}: rarest arm has {min_count} rows, which cannot fill "
            f"{used} stratified folds; reducing n_splits {used} -> {new}"
        )
        LOGGER.warning(msg)
        warnings_.append(msg)
        used = new

    seed = int(rng.integers(0, 2**31 - 1))
    if min_count >= 2:
        splitter = StratifiedKFold(n_splits=used, shuffle=True, random_state=seed)
        folds = [(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)) for tr, te in splitter.split(np.zeros(n), w)]
    else:
        msg = (
            f"{context or 'cross-fitting'}: an arm has fewer than 2 rows, so the arm label cannot be "
            f"stratified; falling back to unstratified KFold with {used} folds"
        )
        LOGGER.warning(msg)
        warnings_.append(msg)
        splitter = KFold(n_splits=used, shuffle=True, random_state=seed)
        folds = [(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)) for tr, te in splitter.split(np.zeros(n))]
    return folds, used, warnings_


# ======================================================================================
# nuisance estimation -- fitted once per fold, cached, reused by every arm
# ======================================================================================
def _fit_per_arm(
    factory: Callable[[int], Any],
    X: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    *,
    n_arms: int,
    min_arm_rows: int,
    seeds: _SeedStream,
    context: str = "",
) -> tuple[dict[int, Any], set[int], list[str]]:
    """Fit ``mu_a(x) = E[Y | X = x, A = a]`` on the rows of each arm.

    Parameters
    ----------
    factory : callable
        ``factory(seed) -> unfitted regressor``.
    X, w, y : numpy.ndarray
        Design matrix ``(n, p)``, arm labels ``(n,)``, outcome ``(n,)``.
    sample_weight : numpy.ndarray or None
        Row weights.
    n_arms : int
        Number of arms.
    min_arm_rows : int
        An arm with fewer rows than this falls back to a **pooled** model fitted on all rows
        (arm ignored), which is biased but finite -- far better than a model fitted on 3 rows.
        Every fallback is logged.
    seeds : _SeedStream
        Deterministic seed source.
    context : str, optional
        Label for log messages.

    Returns
    -------
    tuple
        ``(models, fallback_arms, warnings)``.
    """
    models: dict[int, Any] = {}
    fallbacks: set[int] = set()
    warnings_: list[str] = []
    pooled: Any = None
    counts = np.bincount(w, minlength=n_arms)
    for a in range(n_arms):
        idx = np.flatnonzero(w == a)
        if idx.size < int(min_arm_rows):
            if pooled is None:
                pooled = _fit_model(factory(seeds()), X, y, sample_weight)
            models[a] = pooled
            fallbacks.add(a)
            msg = (
                f"{context or 'per-arm outcome model'}: arm {a} ({_arm_label(a, n_arms)}) has only "
                f"{int(counts[a])} rows (< min_arm_rows={int(min_arm_rows)}); falling back to the pooled "
                f"outcome model -- its contrast for this arm is shrunk toward zero"
            )
            LOGGER.warning(msg)
            warnings_.append(msg)
        else:
            sw = None if sample_weight is None else sample_weight[idx]
            models[a] = _fit_model(factory(seeds()), X[idx], y[idx], sw)
    return models, fallbacks, warnings_


def _predict_per_arm(models: dict[int, Any], X: np.ndarray, n_arms: int) -> np.ndarray:
    """Score every per-arm outcome model on every row.

    Returns
    -------
    numpy.ndarray
        ``(m, n_arms)`` matrix of ``mu_a(x)``.
    """
    out = np.zeros((X.shape[0], int(n_arms)), dtype=np.float64)
    for a in range(int(n_arms)):
        model = models.get(a)
        if model is not None:
            out[:, a] = np.asarray(model.predict(X), dtype=np.float64).ravel()
    return out


def _oof_per_arm_regression(
    factory: Callable[[int], Any],
    X: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_arms: int,
    min_arm_rows: int,
    seeds: _SeedStream,
    context: str = "",
) -> tuple[np.ndarray, set[int], list[str]]:
    """Cross-fitted ``mu_a(x)`` for every arm, cached in one ``(n, K)`` array.

    This is the cache SPEC_PERF section 6 mandates.  Inside each fold every arm's outcome model
    is fitted **once** on the training part and scored on **all** held-out rows -- including rows
    that received a different arm, because ``mu_a`` must be evaluated counterfactually.  The
    result is ``n_splits * K`` fits total, and every downstream contrast (arm 1 vs 0, arm 2 vs 0,
    ...) reads the same cached matrix instead of refitting.

    Parameters
    ----------
    factory : callable
        ``factory(seed) -> unfitted regressor``.
    X, w, y : numpy.ndarray
        Design matrix, arm labels, outcome.
    sample_weight : numpy.ndarray or None
        Row weights.
    folds : sequence of (ndarray, ndarray)
        ``(train_index, test_index)`` pairs from :func:`_make_folds`.
    n_arms : int
        Number of arms.
    min_arm_rows : int
        Per-fold fallback threshold; see :func:`_fit_per_arm`.
    seeds : _SeedStream
        Deterministic seed source.
    context : str, optional
        Label for log messages.

    Returns
    -------
    tuple
        ``(mu_oof, fallback_arms, warnings)`` with ``mu_oof`` of shape ``(n, n_arms)``.
    """
    n = X.shape[0]
    mu = np.zeros((n, int(n_arms)), dtype=np.float64)
    fallbacks: set[int] = set()
    warnings_: list[str] = []
    for k, (tr, te) in enumerate(folds):
        pooled: Any = None
        for a in range(int(n_arms)):
            sel = tr[w[tr] == a]
            if sel.size < int(min_arm_rows):
                if pooled is None:
                    sw_tr = None if sample_weight is None else sample_weight[tr]
                    pooled = _fit_model(factory(seeds()), X[tr], y[tr], sw_tr)
                model = pooled
                if a not in fallbacks:
                    msg = (
                        f"{context or 'cross-fitted outcome model'}: arm {a} ({_arm_label(a, n_arms)}) has "
                        f"{int(sel.size)} training rows in fold {k} (< min_arm_rows={int(min_arm_rows)}); "
                        f"using the pooled outcome model for that fold"
                    )
                    LOGGER.warning(msg)
                    warnings_.append(msg)
                fallbacks.add(a)
            else:
                sw_sel = None if sample_weight is None else sample_weight[sel]
                model = _fit_model(factory(seeds()), X[sel], y[sel], sw_sel)
            mu[te, a] = np.asarray(model.predict(X[te]), dtype=np.float64).ravel()
    return mu, fallbacks, warnings_


def _oof_regression(
    factory: Callable[[int], Any],
    Z: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    seeds: _SeedStream,
) -> np.ndarray:
    """Plain out-of-fold predictions of a regressor on a fixed design ``Z``.

    Used by :class:`SLearner` to score its own pooled model honestly for the ``nuisance_``
    diagnostic (the S-learner's "nuisance" *is* its outcome model).

    Returns
    -------
    numpy.ndarray
        ``(n,)`` out-of-fold predictions.
    """
    out = np.zeros(Z.shape[0], dtype=np.float64)
    for tr, te in folds:
        sw = None if sample_weight is None else sample_weight[tr]
        model = _fit_model(factory(seeds()), Z[tr], y[tr], sw)
        out[te] = np.asarray(model.predict(Z[te]), dtype=np.float64).ravel()
    return out


class _FullPropensity:
    """Thin ``predict_proba`` wrapper around a plain multiclass classifier.

    Exists so that the internal fallback path and :class:`prism.models.propensity.PropensityModel`
    present the same interface to :class:`XLearner`, which needs ``e_a(x)`` for rows it has never
    seen.

    Parameters
    ----------
    clf : object
        A fitted sklearn-compatible classifier.
    n_arms : int
        Number of arms (columns of the output).
    clip : tuple of (float, float)
        Probability clip applied with :func:`_clip_simplex`.
    """

    def __init__(self, clf: Any, n_arms: int, clip: tuple[float, float]) -> None:
        self.clf = clf
        self.n_arms = int(n_arms)
        self.clip = (float(clip[0]), float(clip[1]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``(m, n_arms)`` clipped, row-stochastic propensities."""
        return _clip_simplex(_expand_proba(self.clf, np.asarray(X, dtype=np.float64), self.n_arms), *self.clip)


def _propensity_factory(
    base_estimator: Any,
    n_estimators: int,
    learning_rate: float,
) -> Callable[[int], Any]:
    """Return ``factory(seed) -> unfitted multiclass classifier`` for the fallback path."""

    def factory(seed: int) -> Any:
        if base_estimator is not None:
            return _seeded(clone(base_estimator), seed)
        return best_gbm(
            "classification",
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            min_samples_leaf=40,
            random_state=int(seed),
        )

    return factory


def _fallback_propensity_oof(
    X: np.ndarray,
    w: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_arms: int,
    clip: tuple[float, float],
    factory: Callable[[int], Any],
    seeds: _SeedStream,
    need_model: bool,
) -> tuple[np.ndarray, Any, str]:
    """Cross-fit a multiclass propensity model inside this module.

    Only reached when :mod:`prism.models.propensity` cannot be imported or its
    :class:`~prism.models.propensity.PropensityModel` raises.  It is deliberately plain: one
    gradient-boosted classifier per fold, no calibration.  The calibrated, diagnosed
    implementation lives in ``prism.models.propensity`` and is preferred whenever it works.

    Returns
    -------
    tuple
        ``(e_oof, model_or_None, source)``.
    """
    n = X.shape[0]
    e = np.full((n, int(n_arms)), 1.0 / float(n_arms), dtype=np.float64)
    for tr, te in folds:
        if np.unique(w[tr]).size < 2:  # pragma: no cover - degenerate design
            continue
        clf = factory(seeds())
        clf.fit(X[tr], w[tr])
        e[te] = _expand_proba(clf, X[te], n_arms)
    e = _clip_simplex(e, *clip)
    model = None
    if need_model:
        clf = factory(seeds())
        clf.fit(X, w)
        model = _FullPropensity(clf, n_arms, clip)
    return e, model, "internal_fallback"


def _oof_propensity(
    X: np.ndarray,
    w: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_arms: int,
    n_splits: int,
    clip: tuple[float, float],
    base_estimator: Any,
    n_estimators: int,
    learning_rate: float,
    seeds: _SeedStream,
    supplied: np.ndarray | None,
    need_model: bool,
) -> tuple[np.ndarray, Any, str, list[str]]:
    """Obtain out-of-fold generalised propensities ``e_a(x)``.

    Three paths, in order of preference:

    1. ``supplied`` -- the caller passed an ``(n, n_arms)`` array (e.g. the design probabilities
       of the RCT block, or ground truth in a simulation). It is clipped onto the simplex and
       used as is; no model is fitted unless ``need_model`` is set.
    2. :class:`prism.models.propensity.PropensityModel` -- cross-fitted and calibrated. Imported
       **lazily inside this function** so that a problem in that module cannot stop
       ``prism.causal.learners`` from importing.
    3. the internal fallback in :func:`_fallback_propensity_oof`, with a logged warning.

    Parameters
    ----------
    X, w : numpy.ndarray
        Design matrix and arm labels.
    folds : sequence of (ndarray, ndarray)
        Folds used by the fallback path (``PropensityModel`` builds its own, which is fine: the
        AIPW pseudo-outcome only requires each nuisance to be honest for row ``i``, not that the
        two nuisances share a partition).
    n_arms, n_splits : int
        Problem size and requested folds.
    clip : tuple of (float, float)
        Propensity clip. The inverse of the lower bound is the worst weight the AIPW correction
        can produce, so this is the main variance control.
    base_estimator : object or None
        Override for the propensity learner.
    n_estimators, learning_rate : int, float
        Defaults forwarded to :func:`prism.utils.optional.best_gbm`.
    seeds : _SeedStream
        Deterministic seed source.
    supplied : numpy.ndarray or None
        Caller-provided propensities.
    need_model : bool
        Whether a model able to score **new** rows is required (X-learner needs one).

    Returns
    -------
    tuple
        ``(e_oof, model_or_None, source, warnings)``.
    """
    n = X.shape[0]
    warnings_: list[str] = []
    factory = _propensity_factory(base_estimator, n_estimators, learning_rate)

    if supplied is not None:
        e = np.asarray(supplied, dtype=np.float64)
        if e.ndim != 2 or e.shape != (n, int(n_arms)):
            raise ValueError(f"propensity must have shape {(n, int(n_arms))}; got {e.shape}")
        if not np.isfinite(e).all():
            raise ValueError("propensity contains non-finite values")
        if (e < 0).any():
            raise ValueError("propensity contains negative values")
        e = _clip_simplex(e, *clip)
        model = None
        if need_model:
            clf = factory(seeds())
            clf.fit(X, w)
            model = _FullPropensity(clf, n_arms, clip)
            msg = (
                "propensity was supplied for the training rows, but this learner also needs e_a(x) for "
                "unseen rows at predict time; one full-data classifier was fitted for that purpose only"
            )
            LOGGER.info(msg)
            warnings_.append(msg)
        return e, model, "supplied", warnings_

    try:
        from prism.models.propensity import PropensityModel
    except Exception as exc:  # pragma: no cover - only when propensity.py is broken
        msg = f"prism.models.propensity could not be imported ({exc}); using the internal propensity fallback"
        LOGGER.warning(msg)
        warnings_.append(msg)
        e, model, source = _fallback_propensity_oof(
            X, w, folds, n_arms=n_arms, clip=clip, factory=factory, seeds=seeds, need_model=need_model
        )
        return e, model, source, warnings_

    try:
        model = PropensityModel(
            n_arms=int(n_arms),
            base_estimator=base_estimator,
            n_splits=int(n_splits),
            calibrate=True,
            clip=(float(clip[0]), float(clip[1])),
            random_state=seeds(),
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
        )
        model.fit(X, w)
        return np.asarray(model.oof_proba(), dtype=np.float64), model, "PropensityModel", warnings_
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"PropensityModel.fit failed ({exc}); using the internal propensity fallback"
        LOGGER.warning(msg)
        warnings_.append(msg)
        e, mdl, source = _fallback_propensity_oof(
            X, w, folds, n_arms=n_arms, clip=clip, factory=factory, seeds=seeds, need_model=need_model
        )
        return e, mdl, source, warnings_


def _nuisance_summary(
    *,
    learner: str,
    y: np.ndarray,
    w: np.ndarray,
    sample_weight: np.ndarray | None,
    n_arms: int,
    n_splits: int,
    crossfit: bool,
    mu_oof: np.ndarray | None = None,
    outcome_oof: np.ndarray | None = None,
    e_oof: np.ndarray | None = None,
    propensity_source: str = "none",
    extra: dict[str, Any] | None = None,
    warnings_: Iterable[str] = (),
) -> dict[str, Any]:
    """Assemble the ``nuisance_`` diagnostic dictionary.

    The point of this object is to let a user answer, in one look, the question *"is this DR
    estimate resting on anything?"*.  A doubly-robust estimator is consistent if **either** the
    outcome model or the propensity model is right; it is not consistent if both are noise, and
    the estimate looks exactly the same either way.  So:

    ``outcome_r2_oof``
        Out-of-fold ``R^2`` of the realised-arm outcome prediction ``mu_{A_i}(X_i)`` against
        ``Y_i``.  This is the honest predictive accuracy of the outcome nuisance.  Near zero
        means the AIPW correction term is carrying the whole estimate and the effective sample
        size below is the real sample size.
    ``propensity_logloss`` vs ``propensity_logloss_marginal``
        Out-of-fold multiclass log-loss of ``e(x)`` against the log-loss of simply predicting the
        marginal arm shares.  ``propensity_logloss_skill = 1 - ratio`` is the fraction of the
        baseline log-loss removed; at or below zero the propensity model has found no confounding
        signal at all, which either means the design was (conditionally) randomised or means the
        model failed.
    ``max_inverse_propensity`` and ``effective_sample_size``
        ``1 / e_{A_i}(X_i)`` is the weight the AIPW correction applies to row ``i``.  Its maximum
        is the single most fragile number in the estimate, and
        ``ESS = (sum w)^2 / sum w^2`` is the honest denominator once those weights are in play.

    Parameters
    ----------
    learner : str
        Class name, for display.
    y, w : numpy.ndarray
        Outcome and arm labels.
    sample_weight : numpy.ndarray or None
        Row weights.
    n_arms, n_splits : int
        Problem size and folds actually used.
    crossfit : bool
        Whether the numbers below came from the estimator's own cross-fitting (``True``) or from
        an extra diagnostic-only pass (``False``, used by the S and T learners).
    mu_oof : numpy.ndarray, optional
        ``(n, n_arms)`` cross-fitted outcome predictions.
    outcome_oof : numpy.ndarray, optional
        ``(n,)`` cross-fitted prediction of ``Y`` used when the learner has a single pooled
        outcome model (the S-learner). Overrides the ``mu_oof`` route for ``outcome_r2_oof``.
    e_oof : numpy.ndarray, optional
        ``(n, n_arms)`` cross-fitted propensities.
    propensity_source : str
        Where ``e_oof`` came from.
    extra : dict, optional
        Learner-specific entries merged into the result.
    warnings_ : iterable of str
        Messages collected during fitting.

    Returns
    -------
    dict
        Flat, JSON-friendly diagnostics.
    """
    n = int(y.size)
    sw = _weights_or_ones(sample_weight, n)
    out: dict[str, Any] = {
        "learner": learner,
        "n": n,
        "n_arms": int(n_arms),
        "n_splits": int(n_splits),
        "crossfit": bool(crossfit),
        "propensity_source": propensity_source,
    }

    if outcome_oof is not None:
        out["outcome_r2_oof"] = _weighted_r2(y, outcome_oof, sw)
        out["outcome_r2_marginal_oof"] = out["outcome_r2_oof"]
        out["outcome_r2_oof_per_arm"] = {
            int(a): _weighted_r2(y[w == a], np.asarray(outcome_oof)[w == a], sw[w == a])
            for a in range(int(n_arms))
            if int(np.sum(w == a)) > 2
        }
    elif mu_oof is not None:
        realised = mu_oof[np.arange(n), w]
        out["outcome_r2_oof"] = _weighted_r2(y, realised, sw)
        out["outcome_r2_oof_per_arm"] = {
            int(a): _weighted_r2(y[w == a], mu_oof[w == a, a], sw[w == a])
            for a in range(int(n_arms))
            if int(np.sum(w == a)) > 2
        }
        if e_oof is not None:
            # Implied marginal outcome regression m(x) = sum_a e_a(x) mu_a(x). Free: no extra fit.
            out["outcome_r2_marginal_oof"] = _weighted_r2(y, np.sum(e_oof * mu_oof, axis=1), sw)
    else:
        out["outcome_r2_oof"] = float("nan")
        out["outcome_r2_oof_per_arm"] = {}

    if e_oof is not None:
        realised_e = np.clip(e_oof[np.arange(n), w], _LOG_EPS, 1.0 - _LOG_EPS)
        shares = np.bincount(w, weights=sw, minlength=int(n_arms))
        shares = np.clip(shares / max(float(sw.sum()), _TINY), _LOG_EPS, 1.0)
        out["propensity_logloss"] = float(-np.average(np.log(realised_e), weights=sw))
        out["propensity_logloss_marginal"] = float(-np.average(np.log(shares[w]), weights=sw))
        denom = out["propensity_logloss_marginal"]
        out["propensity_logloss_skill"] = float(1.0 - out["propensity_logloss"] / denom) if denom > _TINY else float("nan")
        out["propensity_min"] = float(e_oof.min())
        out["propensity_max"] = float(e_oof.max())
        inv = 1.0 / realised_e
        out["max_inverse_propensity"] = float(inv.max())
        ipw = sw * inv
        out["effective_sample_size"] = float(ipw.sum() ** 2 / max(float(np.sum(ipw**2)), _TINY))
        out["ess_fraction"] = float(out["effective_sample_size"] / n)
    else:
        out["propensity_logloss"] = float("nan")
        out["propensity_logloss_marginal"] = float("nan")
        out["propensity_logloss_skill"] = float("nan")
        out["max_inverse_propensity"] = float("nan")
        out["effective_sample_size"] = float("nan")
        out["ess_fraction"] = float("nan")

    r2 = out.get("outcome_r2_oof", float("nan"))
    if np.isfinite(r2):
        out["outcome_flag"] = "weak" if r2 < 0.05 else ("thin" if r2 < 0.20 else "ok")
        if r2 < 0.05:
            LOGGER.warning(
                "%s: the out-of-fold outcome model has R2 = %.4f. A doubly-robust estimate built on a "
                "useless outcome model is a red flag -- it degenerates to plain IPW and inherits its "
                "variance. Check nuisance_['effective_sample_size'].",
                learner,
                r2,
            )
    else:
        out["outcome_flag"] = "unknown"

    skill = out.get("propensity_logloss_skill", float("nan"))
    if np.isfinite(skill):
        out["propensity_flag"] = "no_signal" if skill <= 0.005 else ("weak" if skill < 0.02 else "ok")
    else:
        out["propensity_flag"] = "unknown"

    if extra:
        out.update(extra)
    out["warnings"] = list(warnings_)
    return out


# ======================================================================================
# base class
# ======================================================================================
class BaseCATELearner(BaseEstimator):
    """Common interface, validation and nuisance bookkeeping for every CATE meta-learner.

    Subclasses implement two private hooks, ``_fit`` and ``_predict_cate``; everything visible to
    the pipeline -- input validation, the design matrix, the fitted-state guard, the
    ``(n, n_arms - 1)`` output contract, ``predict`` aliasing and :attr:`nuisance_` -- lives here
    so that all five learners behave identically.

    Parameters
    ----------
    base_outcome : object, optional
        Unfitted sklearn-compatible **regressor** used for the outcome nuisances
        ``mu_a(x) = E[Y | X = x, A = a]``. Default: :func:`prism.utils.optional.best_gbm`
        (LightGBM > XGBoost > ``HistGradientBoosting``).
    base_effect : object, optional
        Unfitted regressor used for the **second stage** (the model of ``tau`` itself: the
        imputed effects of the X-learner, the AIPW pseudo-outcome of the DR-learner, the
        Robinson pseudo-outcome of the R-learner). Defaults to a slightly more regularised
        gradient-boosting model than ``base_outcome``, because a pseudo-outcome is far noisier
        than an outcome and the effect surface is usually smoother than the outcome surface.
    base_propensity : object, optional
        Unfitted sklearn-compatible **classifier** for ``e_a(x) = P(A = a | X = x)``.
    n_splits : int, default 5
        Folds used for cross-fitting the nuisances (DR, R, and the X-learner's stage 1).
        Reduced automatically, with a warning, if an arm is too rare to stratify.
    random_state : int or numpy.random.Generator or None, optional
        Seed material. The same seed gives bit-identical predictions.
    n_arms : int, keyword-only, default ``prism.data.schema.N_ARMS``
        Number of arms including control. ``predict_cate`` returns ``n_arms - 1`` columns, so
        this must match the pipeline's arm space even if some arm is unobserved in a given
        training sample.
    propensity_clip : tuple of (float, float), keyword-only, default (0.01, 0.99)
        Propensities are projected into this box (and back onto the simplex) before any division.
        ``1 / propensity_clip[0]`` bounds the worst AIPW weight.
    min_arm_rows : int, keyword-only, default 50
        Below this many rows an arm's outcome model falls back to the pooled model, with a log
        message.
    nuisance_diagnostics : bool, keyword-only, default True
        For learners that do not otherwise need cross-fitting (S and T), run a small extra
        out-of-fold pass purely to populate :attr:`nuisance_`. Set ``False`` to skip those fits.
    diagnostic_splits : int, keyword-only, default 3
        Number of folds used by that diagnostic-only pass.
    n_estimators : int, keyword-only, default 300
        Trees in the default gradient-boosting nuisance models.
    learning_rate : float, keyword-only, default 0.05
        Learning rate of the default nuisance models.
    effect_max_depth : int, keyword-only, optional
        Maximum depth of the default **second-stage** model (``-1`` for unlimited). ``None``
        takes the per-learner default; see the note below.
    effect_min_samples_leaf : int, keyword-only, optional
        Minimum leaf size of the default second-stage model. ``None`` resolves at fit time to
        ``max(floor, n // divisor)`` using the per-learner constants below, so the smoothing
        scales with the sample rather than being a fixed number that is far too loose at
        ``n = 100k`` and far too tight at ``n = 2k``.
    **kw
        Ignored, recorded on ``extra_kwargs_``. Present so that
        ``make_learner(name, **shared_kwargs)`` can pass one dictionary to every learner without
        the caller having to know which knobs each one reads.

    Attributes
    ----------
    n_arms : int
        Number of arms, including control.
    nuisance_ : dict
        Nuisance diagnostics; see :func:`_nuisance_summary`.
    ate_ : numpy.ndarray
        ``(n_arms - 1,)`` sample average of the fitted CATE on the training rows.
    n_splits_ : int
        Folds actually used after any automatic reduction.
    fit_seconds_ : float
        Wall-clock fit time.

    Notes
    -----
    **Why the second stage is regularised harder than the first.** The outcome nuisance
    ``mu_a(x)`` is fitted against ``Y``, whose conditional signal-to-noise ratio in retention
    data is high -- an out-of-fold ``R^2`` of 0.8 is ordinary. The *second* stage is fitted
    against a pseudo-outcome, and pseudo-outcomes are brutal: the AIPW ``psi`` divides residuals
    by propensities, so ``sd(psi)`` routinely exceeds ``sd(tau)`` by an order of magnitude, and
    the best achievable ``R^2`` on that target is a couple of percent. A gradient-boosting model
    tuned for the first problem will cheerfully fit noise on the second, producing a CATE whose
    dispersion exceeds the truth's -- a model that is *worse than predicting the constant ATE*
    while looking impressively heterogeneous. Measured on the simulation in this module's
    ``__main__``, the same DR-learner scores PEHE 4.18 with an unregularised second stage and
    0.86 with a regularised one; nothing else changed. So the second-stage default here is a
    shallow, large-leaf smoother, with ``XLearner`` allowed more capacity because its imputed
    effects ``Y - mu_0(X)`` are far less noisy than an inverse-propensity-weighted residual.

    ``BaseCATELearner`` itself is not usable: :meth:`_fit` raises. Instantiate a subclass.
    """

    #: Set by subclasses that consume a propensity model (used only for messages).
    _uses_propensity: bool = True
    #: Default ``max_depth`` of the second-stage model (``-1`` means unlimited).
    _EFFECT_DEPTH: int = -1
    #: Default second-stage leaf size is ``max(_EFFECT_LEAF_FLOOR, n // _EFFECT_LEAF_DIVISOR)``.
    _EFFECT_LEAF_FLOOR: int = 40
    _EFFECT_LEAF_DIVISOR: int = 150

    def __init__(
        self,
        base_outcome: Any = None,
        base_effect: Any = None,
        base_propensity: Any = None,
        n_splits: int = 5,
        random_state: int | np.random.Generator | None = None,
        *,
        n_arms: int = N_ARMS,
        propensity_clip: tuple[float, float] = (0.01, 0.99),
        min_arm_rows: int = 50,
        nuisance_diagnostics: bool = True,
        diagnostic_splits: int = 3,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        effect_max_depth: int | None = None,
        effect_min_samples_leaf: int | None = None,
        **kw: Any,
    ) -> None:
        if int(n_arms) < 2:
            raise ValueError(f"n_arms must be at least 2 (control plus one offer); got {n_arms}")
        clip = tuple(propensity_clip)  # tuple(t) is t for a tuple, which keeps sklearn's clone happy
        if len(clip) != 2 or not (0.0 < float(clip[0]) < float(clip[1]) <= 1.0):
            raise ValueError(f"propensity_clip must be (low, high) with 0 < low < high <= 1; got {propensity_clip!r}")
        if int(n_arms) * float(clip[0]) > 1.0:
            raise ValueError(
                f"propensity_clip low={clip[0]} is impossible for {n_arms} arms "
                f"(n_arms * low must not exceed 1)"
            )
        self.base_outcome = base_outcome
        self.base_effect = base_effect
        self.base_propensity = base_propensity
        self.n_splits = int(n_splits)
        self.random_state = random_state
        self.n_arms = int(n_arms)
        self.propensity_clip = clip
        self.min_arm_rows = int(min_arm_rows)
        self.nuisance_diagnostics = bool(nuisance_diagnostics)
        self.diagnostic_splits = int(diagnostic_splits)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.effect_max_depth = effect_max_depth
        self.effect_min_samples_leaf = effect_min_samples_leaf
        self.extra_kwargs_ = dict(kw)
        if kw:
            LOGGER.debug("%s ignoring unrecognised kwargs %s", type(self).__name__, sorted(kw))

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Constructor parameters, including the base ones a subclass absorbs into ``**kw``.

        :class:`XLearner` and :class:`RLearner` declare their own extra keywords and forward the
        shared ones through ``**kw``.  scikit-learn's default introspection only sees a class's
        own signature, so :func:`sklearn.base.clone` on one of those would quietly drop
        ``n_arms``, ``propensity_clip`` and the rest and hand back a learner configured for a
        different problem.  Merging in :class:`BaseCATELearner`'s own parameter list fixes that,
        which matters because ``prism.causal.refute`` clones estimators for every simulation.

        Returns
        -------
        list of str
            Sorted parameter names.
        """
        own = set(super()._get_param_names())
        shared = {
            name
            for name, param in inspect.signature(BaseCATELearner.__init__).parameters.items()
            if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
        }
        return sorted(own | shared)

    # ------------------------------------------------------------------ factories
    def _outcome_factory(self) -> Callable[[int], Any]:
        """Return ``factory(seed) -> unfitted outcome regressor``."""

        def factory(seed: int) -> Any:
            if self.base_outcome is not None:
                return _seeded(clone(self.base_outcome), seed)
            return best_gbm(
                "regression",
                n_estimators=int(self.n_estimators),
                learning_rate=float(self.learning_rate),
                min_samples_leaf=20,
                random_state=int(seed),
            )

        return factory

    def _effect_leaf_size(self) -> int:
        """Resolve the second-stage minimum leaf size for the current training sample."""
        if self.effect_min_samples_leaf is not None:
            return int(self.effect_min_samples_leaf)
        n = int(getattr(self, "_n_train_", 0))
        return int(max(self._EFFECT_LEAF_FLOOR, n // self._EFFECT_LEAF_DIVISOR))

    def _effect_factory(self) -> Callable[[int], Any]:
        """Return ``factory(seed) -> unfitted second-stage (effect) regressor``.

        The default is deliberately a **smoother**, not a flexible learner; see the note in the
        class docstring on why the pseudo-outcome's signal-to-noise ratio forces this.
        """
        depth = self._EFFECT_DEPTH if self.effect_max_depth is None else int(self.effect_max_depth)
        leaf = self._effect_leaf_size()

        def factory(seed: int) -> Any:
            if self.base_effect is not None:
                return _seeded(clone(self.base_effect), seed)
            return best_gbm(
                "regression",
                n_estimators=int(self.n_estimators),
                learning_rate=float(self.learning_rate),
                max_depth=int(depth),
                min_samples_leaf=int(leaf),
                random_state=int(seed),
            )

        return factory

    # ------------------------------------------------------------------ plumbing
    def _design(self, X: np.ndarray | pd.DataFrame, *, fit: bool) -> np.ndarray:
        """Coerce ``X`` to the float64 design matrix this learner was fitted on.

        A :class:`~pandas.DataFrame` has its non-numeric columns one-hot expanded; the expanded
        column order is recorded at fit time and any later frame is reindexed onto it (missing
        levels become zero columns), so a category unseen at training time cannot silently shift
        every downstream column.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Covariates.
        fit : bool
            ``True`` records the layout; ``False`` conforms to the recorded one.

        Returns
        -------
        numpy.ndarray
            ``(n, p)`` float64.
        """
        if isinstance(X, pd.DataFrame):
            matrix, names = _expand_frame(X)
            if fit:
                self.feature_names_in_ = np.asarray(names, dtype=object)
                self._design_names_ = names
            elif names != getattr(self, "_design_names_", names):
                frame = pd.DataFrame(matrix, columns=names).reindex(
                    columns=self._design_names_, fill_value=0.0
                )
                matrix = np.asarray(frame.to_numpy(dtype=np.float64), dtype=np.float64)
        else:
            matrix = np.asarray(X, dtype=np.float64)
            if matrix.ndim == 1:
                matrix = matrix.reshape(-1, 1)
            if fit:
                self._design_names_ = [f"x{j}" for j in range(matrix.shape[1])]
            elif matrix.shape[1] != len(self._design_names_):
                raise ValueError(
                    f"X has {matrix.shape[1]} columns but this learner was fitted on "
                    f"{len(self._design_names_)}"
                )
        if fit:
            self.n_features_in_ = int(matrix.shape[1])
        # NaN is allowed on purpose -- every gradient-boosting backend PRISM uses learns a
        # default direction for missing values, which is strictly better than imputing here.
        # Infinities are not: they silently destroy a split search.
        if np.isinf(matrix).any():
            raise ValueError("X contains infinite values; clean or clip them before fitting")
        return matrix

    def _validate(
        self,
        X: np.ndarray | pd.DataFrame,
        w: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Validate and normalise the training inputs.

        Returns
        -------
        tuple
            ``(X_matrix, w_int64, y_float64, sample_weight_or_None)``. Weights are rescaled to
            mean 1 so that regularisation strengths expressed per unit weight (LightGBM's
            ``min_child_weight``, XGBoost's default) mean the same thing whatever scale the
            caller used; the rescaling is a positive constant and changes no weighted estimate.
        """
        Xm = self._design(X, fit=True)
        n = Xm.shape[0]

        w_arr = np.asarray(w).ravel()
        if w_arr.size != n:
            raise ValueError(f"w has length {w_arr.size} but X has {n} rows")
        if not np.all(np.isfinite(w_arr.astype(np.float64))):
            raise ValueError("w contains non-finite values")
        w_int = np.rint(w_arr.astype(np.float64)).astype(np.int64)
        if not np.allclose(w_int, w_arr.astype(np.float64)):
            raise ValueError("w must contain integer arm indices")
        if w_int.min() < 0:
            raise ValueError(f"w contains negative arm indices (min {int(w_int.min())})")
        if w_int.max() >= self.n_arms:
            raise ValueError(
                f"w contains arm {int(w_int.max())} but n_arms={self.n_arms}; "
                f"construct the learner with n_arms={int(w_int.max()) + 1}"
            )

        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if y_arr.size != n:
            raise ValueError(f"y has length {y_arr.size} but X has {n} rows")
        if not np.isfinite(y_arr).all():
            raise ValueError("y contains non-finite values")

        sw: np.ndarray | None = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()
            if sw.size != n:
                raise ValueError(f"sample_weight has length {sw.size} but X has {n} rows")
            if not np.isfinite(sw).all():
                raise ValueError("sample_weight contains non-finite values")
            if (sw < 0).any():
                raise ValueError("sample_weight contains negative values")
            total = float(sw.sum())
            if total <= _TINY:
                raise ValueError("sample_weight sums to zero")
            sw = sw * (n / total)

        counts = np.bincount(w_int, minlength=self.n_arms)
        if counts[0] == 0:
            raise ValueError("no control rows (w == 0); a contrast against control is undefined")
        missing = [a for a in range(self.n_arms) if counts[a] == 0]
        if missing:
            LOGGER.warning(
                "%s: arms %s have no rows in this sample; their CATE column will come from the "
                "pooled-model fallback and should be read as 'no evidence', not 'no effect'",
                type(self).__name__,
                missing,
            )
        self.arm_counts_ = counts.astype(np.int64)
        self.arm_shares_ = counts.astype(np.float64) / float(n)
        self._n_train_ = int(n)
        return Xm, w_int, y_arr, sw

    def _seed_streams(self) -> tuple[_SeedStream, _SeedStream, _SeedStream, _SeedStream]:
        """Four independent, reproducible seed streams: folds, outcome, propensity, effect.

        Independent streams (via :func:`prism.utils.seeds.spawn_rngs`) mean that changing
        ``n_splits`` reseeds the fold assignment without also reseeding the effect models, which
        makes ablations interpretable.
        """
        children = spawn_rngs(self.random_state, 4)
        return (_SeedStream(children[0]), _SeedStream(children[1]), _SeedStream(children[2]), _SeedStream(children[3]))

    # ------------------------------------------------------------------ public API
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        w: np.ndarray,
        y: np.ndarray,
        *,
        propensity: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Fit the learner.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Covariates, ``(n, p)``.
        w : numpy.ndarray
            Integer arm indices in ``[0, n_arms)``, ``(n,)``. ``0`` is control.
        y : numpy.ndarray
            Outcome, ``(n,)``, higher is better.
        propensity : numpy.ndarray, optional, keyword-only
            ``(n, n_arms)`` generalised propensity score for the training rows. When ``None`` a
            :class:`prism.models.propensity.PropensityModel` is fitted internally (imported
            lazily, inside this call).
        sample_weight : numpy.ndarray, optional, keyword-only
            Non-negative row weights, ``(n,)``. Forwarded to every nuisance and second-stage fit
            and to every weighted diagnostic; never silently dropped.

        Returns
        -------
        Self
            The fitted learner.
        """
        t0 = time.perf_counter()
        Xm, w_int, y_arr, sw = self._validate(X, w, y, sample_weight)
        self.n_splits_ = int(self.n_splits)
        self.nuisance_ = {}
        self._fit(Xm, w_int, y_arr, sw, propensity)
        self.is_fitted_ = True
        cate = np.nan_to_num(
            np.asarray(self._predict_cate(Xm), dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        weights = _weights_or_ones(sw, Xm.shape[0])
        self.ate_ = np.asarray(np.average(cate, axis=0, weights=weights), dtype=np.float64)
        self.fit_seconds_ = float(time.perf_counter() - t0)
        LOGGER.info(
            "%s fitted on n=%d, K=%d in %.2fs | ATE %s",
            type(self).__name__,
            Xm.shape[0],
            self.n_arms,
            self.fit_seconds_,
            np.round(self.ate_, 4).tolist(),
        )
        return self

    def predict_cate(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Conditional average treatment effect of each non-control arm versus control.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Covariates, ``(m, p)``, with the same columns used at fit time.

        Returns
        -------
        numpy.ndarray
            ``(m, n_arms - 1)`` float64. Column ``j`` is ``tau_{j+1}(x)``, the effect of arm
            ``j + 1`` against control on the same scale as ``y``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit(X, w, y) first")
        Xm = self._design(X, fit=False)
        out = np.asarray(self._predict_cate(Xm), dtype=np.float64)
        expected = (Xm.shape[0], self.n_arms - 1)
        if out.shape != expected:
            raise AssertionError(f"{type(self).__name__}.predict_cate returned {out.shape}, expected {expected}")
        if not np.isfinite(out).all():
            n_bad = int((~np.isfinite(out)).sum())
            LOGGER.warning(
                "%s.predict_cate produced %d non-finite value(s); they are replaced by 0.0 "
                "(check nuisance_ -- this usually means a degenerate propensity)",
                type(self).__name__,
                n_bad,
            )
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Alias of :meth:`predict_cate`, required by the SPEC section 4 interface."""
        return self.predict_cate(X)

    def nuisance_frame(self) -> pd.DataFrame:
        """Return :attr:`nuisance_` as a tidy two-column frame for printing or logging.

        Returns
        -------
        pandas.DataFrame
            Columns ``metric`` and ``value``; nested entries are flattened with dotted keys and
            the warning list is collapsed to a count plus the first message.
        """
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit(X, w, y) first")
        rows: list[dict[str, Any]] = []
        for key, value in self.nuisance_.items():
            if key == "warnings":
                rows.append({"metric": "warnings.count", "value": len(value)})
                for i, msg in enumerate(value[:3]):
                    rows.append({"metric": f"warnings.{i}", "value": msg})
            elif isinstance(value, dict):
                for sub, sub_value in value.items():
                    rows.append({"metric": f"{key}.{sub}", "value": sub_value})
            else:
                rows.append({"metric": key, "value": value})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ hooks
    def _fit(
        self,
        X: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        propensity: np.ndarray | None,
    ) -> None:
        """Subclass hook: fit on the validated design matrix."""
        raise NotImplementedError("BaseCATELearner is abstract; use SLearner/TLearner/XLearner/DRLearner/RLearner")

    def _predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Subclass hook: predict on the validated design matrix."""
        raise NotImplementedError

    # ------------------------------------------------------------------ shared internals
    def _diagnostic_pass(
        self,
        X: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        streams: tuple[_SeedStream, _SeedStream, _SeedStream, _SeedStream],
        *,
        propensity: np.ndarray | None,
        outcome_design: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run the extra out-of-fold pass that populates :attr:`nuisance_` for S and T.

        S and T do not need cross-fitting to *estimate* anything -- that is precisely why they are
        cheap -- but a learner whose nuisance quality is invisible cannot be compared honestly
        against DR and R, which report theirs. So a small ``diagnostic_splits``-fold pass is run
        purely for measurement, and it is clearly marked ``crossfit=False`` in the output so
        nobody mistakes it for part of the estimate.

        Parameters
        ----------
        X, w, y : numpy.ndarray
            Validated training arrays.
        sample_weight : numpy.ndarray or None
            Row weights.
        streams : tuple of _SeedStream
            ``(fold, outcome, propensity, effect)`` seed streams.
        propensity : numpy.ndarray or None
            Caller-supplied propensities, if any.
        outcome_design : numpy.ndarray, optional
            When given (the S-learner passes ``[X | one-hot(A)]``), the outcome diagnostic scores
            that single pooled model out-of-fold instead of per-arm models.

        Returns
        -------
        dict
            The ``nuisance_`` dictionary.
        """
        fold_rng, mu_seeds, prop_seeds, _ = streams
        if not self.nuisance_diagnostics:
            return _nuisance_summary(
                learner=type(self).__name__,
                y=y,
                w=w,
                sample_weight=sample_weight,
                n_arms=self.n_arms,
                n_splits=0,
                crossfit=False,
                extra={"note": "nuisance_diagnostics=False; no out-of-fold diagnostic pass was run"},
                warnings_=getattr(self, "_warnings", []),
            )

        splits = int(max(2, min(self.diagnostic_splits, self.n_splits)))
        folds, used, fold_warn = _make_folds(w, splits, fold_rng.rng, context=f"{type(self).__name__} diagnostics")
        warnings_ = list(getattr(self, "_warnings", [])) + fold_warn

        mu_oof: np.ndarray | None = None
        outcome_oof: np.ndarray | None = None
        if outcome_design is not None:
            outcome_oof = _oof_regression(
                self._outcome_factory(), outcome_design, y, sample_weight, folds, seeds=mu_seeds
            )
        else:
            mu_oof, _, mu_warn = _oof_per_arm_regression(
                self._outcome_factory(),
                X,
                w,
                y,
                sample_weight,
                folds,
                n_arms=self.n_arms,
                min_arm_rows=self.min_arm_rows,
                seeds=mu_seeds,
                context=f"{type(self).__name__} diagnostics",
            )
            warnings_ += mu_warn

        e_oof, _, source, prop_warn = _oof_propensity(
            X,
            w,
            folds,
            n_arms=self.n_arms,
            n_splits=used,
            clip=self.propensity_clip,
            base_estimator=self.base_propensity,
            n_estimators=int(self.n_estimators),
            learning_rate=float(max(self.learning_rate, 0.05)),
            seeds=prop_seeds,
            supplied=propensity,
            need_model=False,
        )
        warnings_ += prop_warn

        return _nuisance_summary(
            learner=type(self).__name__,
            y=y,
            w=w,
            sample_weight=sample_weight,
            n_arms=self.n_arms,
            n_splits=used,
            crossfit=False,
            mu_oof=mu_oof,
            outcome_oof=outcome_oof,
            e_oof=e_oof,
            propensity_source=source,
            extra={"note": "diagnostic-only cross-fitting; these models do not enter the estimate"},
            warnings_=warnings_,
        )


# ======================================================================================
# S-learner
# ======================================================================================
class SLearner(BaseCATELearner):
    """Single-model ("S" for *single*) learner: treat the arm as just another feature.

    Estimator maths
    ---------------
    Fit one regression on the concatenation of the covariates and a one-hot encoding of the arm::

        Z_i   = [ X_i , 1{A_i = 0}, 1{A_i = 1}, ..., 1{A_i = K-1} ]
        mu    = argmin_f  sum_i sw_i ( Y_i - f(Z_i) )^2

    and contrast the model against itself with the arm block overwritten::

        tau_a(x) = mu( [x, e_a] ) - mu( [x, e_0] )

    where ``e_a`` is the ``a``-th standard basis vector, so the only thing that changes between
    the two calls is the arm block.  The estimator is a *plug-in* on a single conditional-mean
    model; it is neither doubly robust nor Neyman-orthogonal.

    Why it is biased toward zero
    ----------------------------
    This is the one thing to know about the S-learner.  The treatment indicator enters as one
    column among ``p + K``, and every regularised learner -- shrinkage in a linear model, a split
    budget in a tree ensemble, early stopping -- allocates capacity to the features that reduce
    loss fastest.  When ``Var(mu_0(X))`` is large relative to ``Var(tau(X))``, as it almost always
    is in retention data (baseline value varies far more than the *effect* of a discount), the
    fitting procedure spends its splits on the covariates and may never split on the arm column at
    all.  The contrast is then identically zero, and the model reports "this offer does nothing"
    with perfect confidence.  Nothing in the fit warns you: the outcome ``R^2`` can be excellent
    while the estimated effect is pure shrinkage.  Kuenzel et al. (2019) make the same point; it
    is why the S-learner is included here as a *baseline* rather than a candidate.

    The mitigation used here is the full ``K``-column one-hot rather than a single ``treated``
    flag, which gives the tree a clean, low-cardinality split for each arm, plus the honest
    out-of-fold outcome ``R^2`` in :attr:`nuisance_` so the user can see what the model is doing.

    Parameters
    ----------
    See :class:`BaseCATELearner`.

    Attributes
    ----------
    model_ : object
        The single fitted outcome model.

    References
    ----------
    Kuenzel, S., Sekhon, J., Bickel, P. and Yu, B. (2019). *Metalearners for estimating
    heterogeneous treatment effects using machine learning.* PNAS 116(10).
    """

    _uses_propensity = False

    def _s_design(self, X: np.ndarray, arm: np.ndarray | int) -> np.ndarray:
        """Concatenate ``X`` with a ``K``-column one-hot of the arm.

        Parameters
        ----------
        X : numpy.ndarray
            ``(m, p)`` design matrix.
        arm : numpy.ndarray or int
            Per-row arm labels, or a single arm applied to every row (used to build the
            counterfactual designs at predict time).

        Returns
        -------
        numpy.ndarray
            ``(m, p + n_arms)`` float64.
        """
        m = X.shape[0]
        onehot = np.zeros((m, self.n_arms), dtype=np.float64)
        if np.isscalar(arm):
            onehot[:, int(arm)] = 1.0
        else:
            onehot[np.arange(m), np.asarray(arm, dtype=np.int64)] = 1.0
        return np.hstack([X, onehot])

    def _fit(
        self,
        X: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        propensity: np.ndarray | None,
    ) -> None:
        streams = self._seed_streams()
        _, mu_seeds, _, _ = streams
        design = self._s_design(X, w)
        self.model_ = _fit_model(self._outcome_factory()(mu_seeds()), design, y, sample_weight)
        self._warnings: list[str] = []
        self.nuisance_ = self._diagnostic_pass(
            X, w, y, sample_weight, streams, propensity=propensity, outcome_design=design
        )
        self.n_splits_ = int(self.nuisance_.get("n_splits", 0))

    def _predict_cate(self, X: np.ndarray) -> np.ndarray:
        base = np.asarray(self.model_.predict(self._s_design(X, 0)), dtype=np.float64).ravel()
        out = np.empty((X.shape[0], self.n_arms - 1), dtype=np.float64)
        for a in range(1, self.n_arms):
            out[:, a - 1] = np.asarray(self.model_.predict(self._s_design(X, a)), dtype=np.float64).ravel() - base
        return out


# ======================================================================================
# T-learner
# ======================================================================================
class TLearner(BaseCATELearner):
    """Two-model ("T" for *two*) learner: one outcome model per arm, then subtract.

    Estimator maths
    ---------------
    For each arm ``a`` fit a separate regression on that arm's rows only::

        mu_a = argmin_f  sum_{i : A_i = a} sw_i ( Y_i - f(X_i) )^2       a = 0, ..., K-1

    and contrast::

        tau_a(x) = mu_a(x) - mu_0(x)

    Strength and weakness
    ---------------------
    Because the arm never competes with the covariates for model capacity, the T-learner cannot
    shrink the contrast toward zero the way the S-learner does: each ``mu_a`` is free to have its
    own shape.  The price is variance.  The error of ``tau_a`` is the *sum* of two independent
    estimation errors, and each is estimated on only the rows of its own arm, so with ``K`` arms
    each model sees roughly ``n / K`` rows.  When arms are imbalanced the rare arm's model
    dominates the error, and the contrast inherits the sum of two smoothing biases that need not
    cancel: if ``mu_0`` and ``mu_a`` are smoothed differently (different effective degrees of
    freedom on different sample sizes), their difference has structure that is an artefact of the
    fitting, not of the treatment.  This is the classic reason the T-learner looks heterogeneous
    when the truth is constant.

    Rare-arm guard
    --------------
    An arm with fewer than ``min_arm_rows`` rows gets the **pooled** outcome model -- one model
    fitted on all rows, ignoring the arm -- instead of a model fitted on a handful of rows.  Read
    the resulting column carefully: it is ``E[Y | X] - mu_0(X)``, a *borrow-strength* estimate
    that answers "what does the average arm do here?", not "what does this arm do here".  It is
    deliberately **not** zero, because zero would assert no effect, which is a stronger claim than
    the data supports; but it is equally not evidence about this particular arm.  Every fallback
    is logged by arm name and row count and recorded in ``nuisance_["warnings"]`` and
    :attr:`fallback_arms_`, and any two arms that fall back share the identical column -- which is
    the clearest possible signal that neither column is arm-specific.

    Parameters
    ----------
    See :class:`BaseCATELearner`.

    Attributes
    ----------
    models_ : dict of {int: object}
        Fitted ``mu_a`` for each arm.
    fallback_arms_ : list of int
        Arms that fell back to the pooled model.
    """

    _uses_propensity = False

    def _fit(
        self,
        X: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        propensity: np.ndarray | None,
    ) -> None:
        streams = self._seed_streams()
        _, mu_seeds, _, _ = streams
        self.models_, fallbacks, warns = _fit_per_arm(
            self._outcome_factory(),
            X,
            w,
            y,
            sample_weight,
            n_arms=self.n_arms,
            min_arm_rows=self.min_arm_rows,
            seeds=mu_seeds,
            context="TLearner",
        )
        self.fallback_arms_ = sorted(fallbacks)
        self._warnings = warns
        self.nuisance_ = self._diagnostic_pass(X, w, y, sample_weight, streams, propensity=propensity)
        self.nuisance_["fallback_arms"] = list(self.fallback_arms_)
        self.n_splits_ = int(self.nuisance_.get("n_splits", 0))

    def _predict_cate(self, X: np.ndarray) -> np.ndarray:
        mu = _predict_per_arm(self.models_, X, self.n_arms)
        return np.asarray(mu[:, 1:] - mu[:, [0]], dtype=np.float64)


# ======================================================================================
# X-learner
# ======================================================================================
class XLearner(BaseCATELearner):
    """Imputed-effect ("X" for *cross*) learner with a propensity-weighted blend.

    Estimator maths
    ---------------
    **Stage 1** -- outcome models, exactly as in the T-learner::

        mu_0(x) = E[Y | X = x, A = 0]        mu_a(x) = E[Y | X = x, A = a]

    **Stage 2** -- impute each unit's individual effect using the *other* arm's model, so every
    row contributes one pseudo-effect::

        treated rows (A_i = a):   D_i^(1) = Y_i - mu_0(X_i)
        control rows (A_i = 0):   D_i^(0) = mu_a(X_i) - Y_i

    and regress each on ``X``, giving two competing estimates of the same object::

        tau_a^(1) = E[ D^(1) | X ]   fitted on the arm-a rows
        tau_a^(0) = E[ D^(0) | X ]   fitted on the control rows

    **Stage 3** -- blend them with a weight ``g(x) in [0, 1]``::

        tau_a(x) = g(x) * tau_a^(0)(x) + (1 - g(x)) * tau_a^(1)(x),      g(x) = e_a(x)

    In the multi-arm case the relevant propensity is the *conditional* one on the two-arm
    subproblem, ``g(x) = e_a(x) / (e_0(x) + e_a(x))``, which reduces to ``e_a(x)`` when ``K = 2``.

    Why the weights are that way round
    ----------------------------------
    At first reading the blend looks backwards: when arm ``a`` is common (``e_a -> 1``) the weight
    ``g`` on ``tau_a^(0)`` -- the model fitted on the *scarce* control rows -- goes to one.  The
    resolution is that the accuracy of ``tau_a^(j)`` is not governed by how many rows it was
    fitted on, but by **which outcome model its pseudo-outcome leans on**:

    * ``D^(0) = mu_a(X) - Y`` uses the observed control outcome ``Y`` directly and the *estimated*
      ``mu_a``. When arm ``a`` is abundant, ``mu_a`` is estimated well, so ``D^(0)`` is nearly
      noise-free and ``tau_a^(0)`` is the reliable branch.
    * ``D^(1) = Y - mu_0(X)`` leans on ``mu_0``, which is estimated from the scarce control rows
      and is therefore the *bad* model in that regime; ``tau_a^(1)`` inherits its error.

    So putting weight ``e_a`` on ``tau_a^(0)`` puts the weight on the branch whose imputation
    comes from the arm with more data -- the arm with more data does get more weight, just through
    its outcome model rather than through its row count.  Kuenzel et al. (2019, Section 3.3) give
    the same argument in the limiting cases ``g = 1`` and ``g = 0``.  This is what makes the
    X-learner the right choice when the arms are badly imbalanced, which is the normal state of a
    retention log where 90 percent of customers get no offer.

    Honest stage 1
    --------------
    ``D_i`` compares an *observed* outcome against a *predicted* one for the same row.  If
    ``mu_0`` was fitted on row ``i``, its prediction there is shrunk toward ``Y_i``, so ``D_i`` is
    biased toward zero and stage 2 learns an attenuated effect.  With ``crossfit_stage1=True``
    (the default) the stage-1 predictions used in the imputation are out-of-fold, which removes
    that bias at no extra cost beyond the folds themselves.  Set it to ``False`` for the textbook
    in-sample construction.

    Parameters
    ----------
    crossfit_stage1 : bool, keyword-only, default True
        Use out-of-fold stage-1 predictions when forming the imputed effects.
    blend : {"propensity", "constant"}, keyword-only, default "propensity"
        ``"propensity"`` uses ``g(x) = e_a(x) / (e_0(x) + e_a(x))``; ``"constant"`` uses the
        marginal arm shares, which is the right thing to do when there is no usable propensity
        model and the alternative is extrapolating one.
    Other parameters
        See :class:`BaseCATELearner`.

    Attributes
    ----------
    tau_treated_ , tau_control_ : dict of {int: object or None}
        The stage-2 models ``tau_a^(1)`` and ``tau_a^(0)``.
    propensity_model_ : object or None
        Model used to obtain ``g(x)`` for unseen rows.

    References
    ----------
    Kuenzel, S., Sekhon, J., Bickel, P. and Yu, B. (2019). PNAS 116(10).
    """

    # The X-learner's second-stage target is an imputed effect, Y - mu_0(X) or mu_a(X) - Y.
    # No division by a propensity is involved, so its noise is of the order of the outcome
    # noise rather than an order of magnitude above it, and the second stage can afford real
    # capacity. Contrast DRLearner / RLearner below.
    _EFFECT_DEPTH: int = -1
    _EFFECT_LEAF_FLOOR: int = 40
    _EFFECT_LEAF_DIVISOR: int = 150

    def __init__(
        self,
        base_outcome: Any = None,
        base_effect: Any = None,
        base_propensity: Any = None,
        n_splits: int = 5,
        random_state: int | np.random.Generator | None = None,
        *,
        crossfit_stage1: bool = True,
        blend: str = "propensity",
        **kw: Any,
    ) -> None:
        super().__init__(base_outcome, base_effect, base_propensity, n_splits, random_state, **kw)
        if blend not in {"propensity", "constant"}:
            raise ValueError(f"blend must be 'propensity' or 'constant'; got {blend!r}")
        self.crossfit_stage1 = bool(crossfit_stage1)
        self.blend = str(blend)

    def _fit(
        self,
        X: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        propensity: np.ndarray | None,
    ) -> None:
        fold_rng, mu_seeds, prop_seeds, eff_seeds = self._seed_streams()
        folds, used, warnings_ = _make_folds(w, self.n_splits, fold_rng.rng, context="XLearner")
        self.n_splits_ = used

        # ---- stage 1: mu_a(x), cross-fitted so the imputed effects are not shrunk ----------
        if self.crossfit_stage1:
            mu, fallbacks, mu_warn = _oof_per_arm_regression(
                self._outcome_factory(),
                X,
                w,
                y,
                sample_weight,
                folds,
                n_arms=self.n_arms,
                min_arm_rows=self.min_arm_rows,
                seeds=mu_seeds,
                context="XLearner stage 1",
            )
        else:
            models, fallbacks, mu_warn = _fit_per_arm(
                self._outcome_factory(),
                X,
                w,
                y,
                sample_weight,
                n_arms=self.n_arms,
                min_arm_rows=self.min_arm_rows,
                seeds=mu_seeds,
                context="XLearner stage 1",
            )
            mu = _predict_per_arm(models, X, self.n_arms)
        warnings_ += mu_warn
        self.stage1_fallback_arms_ = sorted(fallbacks)

        # ---- propensity: needed at predict time, so a scoring model is required -------------
        need_model = self.blend == "propensity"
        e_oof, prop_model, source, prop_warn = _oof_propensity(
            X,
            w,
            folds,
            n_arms=self.n_arms,
            n_splits=used,
            clip=self.propensity_clip,
            base_estimator=self.base_propensity,
            n_estimators=int(self.n_estimators),
            learning_rate=float(max(self.learning_rate, 0.05)),
            seeds=prop_seeds,
            supplied=propensity,
            need_model=need_model,
        )
        warnings_ += prop_warn
        self.propensity_model_ = prop_model

        # ---- stage 2: regress the imputed effects ------------------------------------------
        effect_factory = self._effect_factory()
        control_idx = np.flatnonzero(w == 0)
        self.tau_treated_: dict[int, Any] = {}
        self.tau_control_: dict[int, Any] = {}
        for a in range(1, self.n_arms):
            treated_idx = np.flatnonzero(w == a)

            if treated_idx.size >= self.min_arm_rows:
                d_treated = y[treated_idx] - mu[treated_idx, 0]
                sw_t = None if sample_weight is None else sample_weight[treated_idx]
                self.tau_treated_[a] = _fit_model(effect_factory(eff_seeds()), X[treated_idx], d_treated, sw_t)
            else:
                self.tau_treated_[a] = None
                msg = (
                    f"XLearner: arm {a} ({_arm_label(a, self.n_arms)}) has {int(treated_idx.size)} rows "
                    f"(< min_arm_rows={self.min_arm_rows}); tau_a^(1) is skipped and the blend falls back "
                    f"to tau_a^(0) alone"
                )
                LOGGER.warning(msg)
                warnings_.append(msg)

            if control_idx.size >= self.min_arm_rows:
                d_control = mu[control_idx, a] - y[control_idx]
                sw_c = None if sample_weight is None else sample_weight[control_idx]
                self.tau_control_[a] = _fit_model(effect_factory(eff_seeds()), X[control_idx], d_control, sw_c)
            else:
                self.tau_control_[a] = None
                msg = (
                    f"XLearner: only {int(control_idx.size)} control rows (< min_arm_rows="
                    f"{self.min_arm_rows}); tau_a^(0) is skipped for arm {a}"
                )
                LOGGER.warning(msg)
                warnings_.append(msg)

            if self.tau_treated_[a] is None and self.tau_control_[a] is None:
                msg = f"XLearner: arm {a} has neither stage-2 model; its CATE column is identically zero"
                LOGGER.warning(msg)
                warnings_.append(msg)

        self.nuisance_ = _nuisance_summary(
            learner=type(self).__name__,
            y=y,
            w=w,
            sample_weight=sample_weight,
            n_arms=self.n_arms,
            n_splits=used,
            crossfit=bool(self.crossfit_stage1),
            mu_oof=mu,
            e_oof=e_oof,
            propensity_source=source,
            extra={
                "blend": self.blend,
                "crossfit_stage1": bool(self.crossfit_stage1),
                "stage1_fallback_arms": list(self.stage1_fallback_arms_),
            },
            warnings_=warnings_,
        )

    def _blend_weight(self, X: np.ndarray, a: int) -> np.ndarray:
        """Return ``g(x) = e_a(x) / (e_0(x) + e_a(x))`` for the rows of ``X``."""
        if self.blend == "constant" or self.propensity_model_ is None:
            share = float(self.arm_shares_[a])
            denom = float(self.arm_shares_[0]) + share
            g = share / denom if denom > _TINY else 0.5
            return np.full(X.shape[0], g, dtype=np.float64)
        e = _clip_simplex(
            np.asarray(self.propensity_model_.predict_proba(X), dtype=np.float64), *self.propensity_clip
        )
        denom = e[:, 0] + e[:, a]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(denom > _TINY, e[:, a] / np.maximum(denom, _TINY), 0.5)
        return np.clip(np.nan_to_num(g, nan=0.5), 0.0, 1.0)

    def _predict_cate(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], self.n_arms - 1), dtype=np.float64)
        for a in range(1, self.n_arms):
            m_treated = self.tau_treated_.get(a)
            m_control = self.tau_control_.get(a)
            if m_treated is None and m_control is None:
                continue
            if m_treated is None:
                out[:, a - 1] = np.asarray(m_control.predict(X), dtype=np.float64).ravel()
                continue
            if m_control is None:
                out[:, a - 1] = np.asarray(m_treated.predict(X), dtype=np.float64).ravel()
                continue
            g = self._blend_weight(X, a)
            tau0 = np.asarray(m_control.predict(X), dtype=np.float64).ravel()
            tau1 = np.asarray(m_treated.predict(X), dtype=np.float64).ravel()
            out[:, a - 1] = g * tau0 + (1.0 - g) * tau1
        return out


# ======================================================================================
# DR-learner
# ======================================================================================
class DRLearner(BaseCATELearner):
    """Doubly-robust learner: cross-fitted AIPW pseudo-outcome, then a regression on ``X``.

    Estimator maths
    ---------------
    For arm ``a`` versus control, form the augmented inverse-probability-weighted (AIPW)
    pseudo-outcome for every row ``i``::

        psi_i^a = mu_a(X_i) - mu_0(X_i)
                  + 1{A_i = a} / e_a(X_i) * ( Y_i - mu_a(X_i) )
                  - 1{A_i = 0} / e_0(X_i) * ( Y_i - mu_0(X_i) )

    and regress it on the covariates::

        tau_a = argmin_f  sum_i sw_i ( psi_i^a - f(X_i) )^2

    Every row contributes: rows in other arms contribute only the plug-in difference
    ``mu_a - mu_0``, rows in arm ``a`` or in control additionally contribute their residual,
    reweighted by the inverse probability of having landed there.

    Why it is doubly robust
    -----------------------
    Write ``mu`` and ``e`` for the true functions and ``mu_hat``, ``e_hat`` for the estimates.
    Taking the conditional expectation of ``psi`` given ``X = x``,

        E[psi | X = x] = tau_a(x)
                         + ( e_a(x)/e_hat_a(x) - 1 ) ( mu_a(x) - mu_hat_a(x) )
                         - ( e_0(x)/e_hat_0(x) - 1 ) ( mu_0(x) - mu_hat_0(x) )

    The bias is a **product** of the two nuisance errors.  If the propensity is right the first
    factor of each term is zero; if the outcome models are right the second factor is.  Either
    one suffices -- that is the double robustness -- and when both are merely consistent at
    ``o(n^{-1/4})`` the product is ``o(n^{-1/2})`` and drops out of the asymptotics entirely.
    This product structure is Neyman orthogonality, and it is the reason DR is the default.

    Honest cross-fitting is mandatory
    ---------------------------------
    The derivation above assumes ``mu_hat`` and ``e_hat`` are independent of row ``i``.  If they
    were fitted on row ``i``, then ``Y_i - mu_hat_a(X_i)`` is an in-sample residual, which is
    systematically too small, so the correction term is attenuated and ``psi`` collapses back
    onto the (biased) plug-in ``mu_hat_a - mu_hat_0``.  Empirically this shows up as a CATE that
    is shrunk toward zero with confidence intervals that are too narrow -- the worst combination.
    So the nuisances here are fitted on ``K - 1`` folds and the pseudo-outcome is formed only on
    the held-out fold, with :class:`~sklearn.model_selection.StratifiedKFold` on the arm label so
    that no fold is missing an arm.  Following SPEC_PERF section 6 each nuisance is fitted **once
    per fold** and cached in an ``(n, K)`` array; the arms reuse the cache rather than triggering
    a refit apiece.

    Overlap is the cost
    -------------------
    ``1 / e_a(x)`` is unbounded as overlap fails.  Propensities are projected into
    ``propensity_clip`` first, so the worst weight is ``1 / propensity_clip[0]``, and
    ``nuisance_["max_inverse_propensity"]`` and ``nuisance_["effective_sample_size"]`` report how
    close the estimate came to resting on a handful of rows.

    Parameters
    ----------
    See :class:`BaseCATELearner`.

    Attributes
    ----------
    models_ : dict of {int: object}
        Second-stage regressions, one per non-control arm.
    pseudo_outcome_ : numpy.ndarray
        ``(n, n_arms - 1)`` matrix of ``psi``, kept because it is directly reusable: its column
        mean is the efficient AIPW estimate of the ATE, and it is the input that
        ``prism.causal.evaluate`` GATES / BLP calibration expect.
    ate_aipw_ : numpy.ndarray
        ``(n_arms - 1,)`` column means of ``pseudo_outcome_`` -- the semiparametrically efficient
        ATE, which is generally a better ATE than the mean of the regressed CATE.
    ate_aipw_se_ : numpy.ndarray
        Influence-function standard errors of ``ate_aipw_``.

    References
    ----------
    Robins, J., Rotnitzky, A. and Zhao, L. (1994). JASA 89(427).
    Chernozhukov, V. et al. (2018). *Double/debiased machine learning.* Econometrics Journal 21(1).
    Kennedy, E. (2023). *Towards optimal doubly robust estimation of heterogeneous causal effects.*
    """

    # psi divides residuals by propensities, so sd(psi) is typically several times sd(Y) and an
    # order of magnitude above sd(tau). The second stage is therefore a shallow, large-leaf
    # smoother by default; see the note in BaseCATELearner.
    _EFFECT_DEPTH: int = 3
    _EFFECT_LEAF_FLOOR: int = 100
    _EFFECT_LEAF_DIVISOR: int = 25

    def _fit(
        self,
        X: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        propensity: np.ndarray | None,
    ) -> None:
        fold_rng, mu_seeds, prop_seeds, eff_seeds = self._seed_streams()
        folds, used, warnings_ = _make_folds(w, self.n_splits, fold_rng.rng, context="DRLearner")
        self.n_splits_ = used

        mu, fallbacks, mu_warn = _oof_per_arm_regression(
            self._outcome_factory(),
            X,
            w,
            y,
            sample_weight,
            folds,
            n_arms=self.n_arms,
            min_arm_rows=self.min_arm_rows,
            seeds=mu_seeds,
            context="DRLearner",
        )
        warnings_ += mu_warn

        e_oof, _, source, prop_warn = _oof_propensity(
            X,
            w,
            folds,
            n_arms=self.n_arms,
            n_splits=used,
            clip=self.propensity_clip,
            base_estimator=self.base_propensity,
            n_estimators=int(self.n_estimators),
            learning_rate=float(max(self.learning_rate, 0.05)),
            seeds=prop_seeds,
            supplied=propensity,
            need_model=False,
        )
        warnings_ += prop_warn
        e = _clip_simplex(e_oof, *self.propensity_clip)

        n = X.shape[0]
        sw_stats = _weights_or_ones(sample_weight, n)
        is_control = (w == 0).astype(np.float64)
        resid_0 = y - mu[:, 0]
        control_term = is_control * resid_0 / e[:, 0]

        effect_factory = self._effect_factory()
        psi = np.empty((n, self.n_arms - 1), dtype=np.float64)
        self.models_: dict[int, Any] = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for a in range(1, self.n_arms):
                is_a = (w == a).astype(np.float64)
                psi_a = (mu[:, a] - mu[:, 0]) + is_a * (y - mu[:, a]) / e[:, a] - control_term
                psi_a = np.nan_to_num(psi_a, nan=0.0, posinf=0.0, neginf=0.0)
                psi[:, a - 1] = psi_a
                self.models_[a] = _fit_model(effect_factory(eff_seeds()), X, psi_a, sample_weight)

        self.pseudo_outcome_ = psi
        self.ate_aipw_ = np.asarray(np.average(psi, axis=0, weights=sw_stats), dtype=np.float64)
        centered = psi - self.ate_aipw_[None, :]
        self.ate_aipw_se_ = np.sqrt(
            np.average(centered**2, axis=0, weights=sw_stats) * float(np.sum(sw_stats**2)) / (float(sw_stats.sum()) ** 2)
        )

        self.nuisance_ = _nuisance_summary(
            learner=type(self).__name__,
            y=y,
            w=w,
            sample_weight=sample_weight,
            n_arms=self.n_arms,
            n_splits=used,
            crossfit=True,
            mu_oof=mu,
            e_oof=e,
            propensity_source=source,
            extra={
                "outcome_fallback_arms": sorted(fallbacks),
                "pseudo_outcome_sd": {int(a + 1): float(psi[:, a].std()) for a in range(self.n_arms - 1)},
                "ate_aipw": {int(a + 1): float(self.ate_aipw_[a]) for a in range(self.n_arms - 1)},
                "ate_aipw_se": {int(a + 1): float(self.ate_aipw_se_[a]) for a in range(self.n_arms - 1)},
            },
            warnings_=warnings_,
        )

    def _predict_cate(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((X.shape[0], self.n_arms - 1), dtype=np.float64)
        for a in range(1, self.n_arms):
            out[:, a - 1] = np.asarray(self.models_[a].predict(X), dtype=np.float64).ravel()
        return out


# ======================================================================================
# R-learner
# ======================================================================================
class RLearner(BaseCATELearner):
    """Robinson-residualisation learner (Nie and Wager's *quasi-oracle* R-learner).

    Estimator maths
    ---------------
    Start from Robinson's (1988) partially linear decomposition, applied conditionally.  With
    ``m(x) = E[Y | X = x]`` and ``e(x) = P(A = a | X = x)`` on the two-arm subproblem::

        Y_i - m(X_i) = tau_a(X_i) * ( 1{A_i = a} - e(X_i) ) + eps_i,      E[eps | X, A] = 0

    so ``tau_a`` solves the residual-on-residual least-squares problem::

        tau_a = argmin_f  sum_i sw_i ( ( Y_i - m_hat(X_i) ) - f(X_i) ( 1{A_i=a} - e_hat(X_i) ) )^2

    That objective is not something a standard regressor can be handed directly, but it is
    algebraically identical to a **weighted** regression.  Writing ``Y_res = Y - m_hat`` and
    ``W_res = 1{A=a} - e_hat``::

        sum_i ( Y_res_i - f(X_i) W_res_i )^2
          = sum_i W_res_i^2 * ( Y_res_i / W_res_i - f(X_i) )^2

    so fitting any weighted least-squares learner with

        target  = Y_res / W_res        weight = sw * W_res^2

    minimises exactly the R-loss.  That is what is implemented here, which is why an off-the-shelf
    gradient-boosting regressor can solve a causal estimating equation.

    Why it is orthogonal
    --------------------
    The R-loss is a Neyman-orthogonal moment: its derivative with respect to ``m`` and ``e``,
    evaluated at the truth, is zero, so an ``o(n^{-1/4})`` error in either nuisance leaves
    ``tau_hat`` unaffected at first order.  As with DR this only holds when ``m_hat`` and
    ``e_hat`` are **out-of-fold**; in-sample residualisation drives ``Y_res`` toward zero and
    shrinks the estimated effect.

    Clipping the denominator by magnitude, not by absolute value
    ------------------------------------------------------------
    ``Y_res / W_res`` is undefined where ``W_res = 0``, so the denominator is floored::

        W_safe = sign(W_res) * max( |W_res|, residual_clip ),    sign(0) := +1

    Note what is *not* done: ``W_res`` is never replaced by ``|W_res|`` or by
    ``max(W_res, eps)``.  ``W_res`` is negative for every control row (it equals ``-e_hat``), so
    clipping on the signed value alone would flip the sign of the pseudo-outcome for half the
    sample and the estimator would converge to something with no causal meaning.  Only the
    magnitude is floored; the sign is carried through untouched.

    The clip is also nearly inert by construction.  The **weight** keeps the unclipped
    ``W_res^2``, so a row whose treatment was almost perfectly predictable (``|W_res| ~ 0``)
    enters the objective with weight ``~ 0`` no matter how large its ratio became; the clip only
    prevents an intermediate overflow.  With propensities projected into ``[lo, hi]`` one also has
    ``|W_res| >= min(lo, 1 - hi)`` before the clip is ever reached, so with the default
    ``(0.01, 0.99)`` and ``residual_clip = 1e-3`` the clip binds on no rows at all; it is a
    numerical guard, not a modelling choice.  ``nuisance_["residual_clip_fraction"]`` reports the
    share of rows on which it actually bound.

    Multi-arm nuisances, derived rather than refitted
    -------------------------------------------------
    For ``K > 2`` the R-learner is fitted once per non-control arm on the rows with
    ``A in {0, a}``.  On that subsample the required nuisances are *not* the pooled ``E[Y | X]``
    and ``P(A = a | X)`` -- they are conditional on being in the subsample.  Both follow in closed
    form from the cached ``K``-arm nuisances, with no extra model fits (SPEC_PERF section 6)::

        e_{0a}(x) = e_a(x) / ( e_0(x) + e_a(x) )
        m_{0a}(x) = ( e_0(x) mu_0(x) + e_a(x) mu_a(x) ) / ( e_0(x) + e_a(x) )

    the second being just the law of total expectation over the two arms in the subsample.  When
    ``K = 2`` these collapse to ``e_1(x)`` and ``E[Y | X = x]``, recovering the textbook
    R-learner exactly.  The trade-off is honest and is why the alternative is offered: the derived
    ``m_{0a}`` inherits error from the propensity model as well as from the outcome models, which
    slightly weakens the "R only needs a good ``m_hat``" story.  Set
    ``pairwise_nuisance="refit"`` to cross-fit ``m`` and ``e`` separately on each ``{0, a}``
    subsample instead, at a cost of ``(K - 1) * n_splits * 2`` extra fits.

    Parameters
    ----------
    residual_clip : float, keyword-only, default 1e-3
        Magnitude floor applied to ``W_res`` in the denominator only.
    pairwise_nuisance : {"derived", "refit"}, keyword-only, default "derived"
        How the two-arm nuisances are obtained; see above.
    Other parameters
        See :class:`BaseCATELearner`.

    Attributes
    ----------
    models_ : dict of {int: object}
        The weighted second-stage regressions, one per non-control arm.
    residual_diagnostics_ : pandas.DataFrame
        Per-arm ``n``, mean ``|W_res|``, share of rows where the clip bound, and the out-of-fold
        ``R^2`` of the pairwise outcome nuisance ``m_{0a}``.

    References
    ----------
    Robinson, P. (1988). *Root-N-consistent semiparametric regression.* Econometrica 56(4).
    Nie, X. and Wager, S. (2021). *Quasi-oracle estimation of heterogeneous treatment effects.*
    Biometrika 108(2).
    """

    # The R pseudo-outcome divides by W_res (typically ~0.3, so a ~3x amplification of the
    # outcome residual) and is fitted under weights that concentrate on a subset of rows. Same
    # regime as the DR-learner: the second stage must be a smoother.
    _EFFECT_DEPTH: int = 3
    _EFFECT_LEAF_FLOOR: int = 100
    _EFFECT_LEAF_DIVISOR: int = 25

    def __init__(
        self,
        base_outcome: Any = None,
        base_effect: Any = None,
        base_propensity: Any = None,
        n_splits: int = 5,
        random_state: int | np.random.Generator | None = None,
        *,
        residual_clip: float = 1e-3,
        pairwise_nuisance: str = "derived",
        **kw: Any,
    ) -> None:
        super().__init__(base_outcome, base_effect, base_propensity, n_splits, random_state, **kw)
        if float(residual_clip) <= 0.0:
            raise ValueError(f"residual_clip must be positive; got {residual_clip}")
        if pairwise_nuisance not in {"derived", "refit"}:
            raise ValueError(f"pairwise_nuisance must be 'derived' or 'refit'; got {pairwise_nuisance!r}")
        self.residual_clip = float(residual_clip)
        self.pairwise_nuisance = str(pairwise_nuisance)

    def _pairwise_refit(
        self,
        X: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        idx: np.ndarray,
        arm: int,
        streams: tuple[_SeedStream, _SeedStream, _SeedStream],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Cross-fit ``m_{0a}`` and ``e_{0a}`` directly on the ``{0, a}`` subsample.

        Parameters
        ----------
        X, w, y : numpy.ndarray
            Full training arrays.
        sample_weight : numpy.ndarray or None
            Full row weights.
        idx : numpy.ndarray
            Row indices of the subsample.
        arm : int
            The non-control arm.
        streams : tuple of _SeedStream
            ``(fold, outcome, propensity)`` seed streams.

        Returns
        -------
        tuple
            ``(m_pair, e_pair, warnings)``, each array aligned to ``idx``.
        """
        fold_rng, mu_seeds, prop_seeds = streams
        w_bin = (w[idx] == arm).astype(np.int64)
        sw_sub = None if sample_weight is None else sample_weight[idx]
        folds, used, warns = _make_folds(
            w_bin, self.n_splits, fold_rng.rng, context=f"RLearner arm {arm} pairwise refit"
        )
        m_pair = _oof_regression(self._outcome_factory(), X[idx], y[idx], sw_sub, folds, seeds=mu_seeds)
        e_pair2, _, _, prop_warn = _oof_propensity(
            X[idx],
            w_bin,
            folds,
            n_arms=2,
            n_splits=used,
            clip=self.propensity_clip,
            base_estimator=self.base_propensity,
            n_estimators=int(self.n_estimators),
            learning_rate=float(max(self.learning_rate, 0.05)),
            seeds=prop_seeds,
            supplied=None,
            need_model=False,
        )
        return m_pair, e_pair2[:, 1], warns + prop_warn

    def _fit(
        self,
        X: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        propensity: np.ndarray | None,
    ) -> None:
        fold_rng, mu_seeds, prop_seeds, eff_seeds = self._seed_streams()
        folds, used, warnings_ = _make_folds(w, self.n_splits, fold_rng.rng, context="RLearner")
        self.n_splits_ = used

        mu, fallbacks, mu_warn = _oof_per_arm_regression(
            self._outcome_factory(),
            X,
            w,
            y,
            sample_weight,
            folds,
            n_arms=self.n_arms,
            min_arm_rows=self.min_arm_rows,
            seeds=mu_seeds,
            context="RLearner",
        )
        warnings_ += mu_warn

        e_oof, _, source, prop_warn = _oof_propensity(
            X,
            w,
            folds,
            n_arms=self.n_arms,
            n_splits=used,
            clip=self.propensity_clip,
            base_estimator=self.base_propensity,
            n_estimators=int(self.n_estimators),
            learning_rate=float(max(self.learning_rate, 0.05)),
            seeds=prop_seeds,
            supplied=propensity,
            need_model=False,
        )
        warnings_ += prop_warn
        e = _clip_simplex(e_oof, *self.propensity_clip)

        lo, hi = self.propensity_clip
        effect_factory = self._effect_factory()
        self.models_: dict[int, Any] = {}
        rows: list[dict[str, Any]] = []

        for a in range(1, self.n_arms):
            idx = np.flatnonzero((w == 0) | (w == a))
            if idx.size < max(2 * self.min_arm_rows, 20) or int(np.sum(w[idx] == a)) < 2:
                self.models_[a] = None
                msg = (
                    f"RLearner: the {{0, {a}}} subsample has {int(idx.size)} rows "
                    f"({int(np.sum(w[idx] == a))} treated); too few to residualise, so this CATE column "
                    f"is identically zero"
                )
                LOGGER.warning(msg)
                warnings_.append(msg)
                rows.append({"arm": a, "n": int(idx.size), "mean_abs_w_res": np.nan,
                             "clip_fraction": np.nan, "m_r2_oof": np.nan})
                continue

            if self.pairwise_nuisance == "refit":
                m_pair, e_pair, refit_warn = self._pairwise_refit(
                    X, w, y, sample_weight, idx, a, (fold_rng, mu_seeds, prop_seeds)
                )
                warnings_ += refit_warn
                e_pair = np.clip(e_pair, lo, hi)
            else:
                denom = e[idx, 0] + e[idx, a]
                denom = np.maximum(denom, _TINY)
                e_pair = np.clip(e[idx, a] / denom, lo, hi)
                m_pair = (e[idx, 0] * mu[idx, 0] + e[idx, a] * mu[idx, a]) / denom

            y_res = y[idx] - m_pair
            w_res = (w[idx] == a).astype(np.float64) - e_pair

            magnitude = np.abs(w_res)
            clipped = magnitude < self.residual_clip
            sign = np.where(w_res < 0.0, -1.0, 1.0)  # sign(0) := +1, sign PRESERVED
            w_safe = sign * np.maximum(magnitude, self.residual_clip)

            with np.errstate(divide="ignore", invalid="ignore"):
                pseudo = y_res / w_safe
            pseudo = np.nan_to_num(pseudo, nan=0.0, posinf=0.0, neginf=0.0)

            weight = w_res**2
            if sample_weight is not None:
                weight = weight * sample_weight[idx]
            total = float(weight.sum())
            if total <= _TINY:  # pragma: no cover - defensive
                self.models_[a] = None
                warnings_.append(f"RLearner: arm {a} has zero total R-weight; CATE column set to zero")
                continue
            weight = weight * (weight.size / total)  # mean-1, so regularisation scales are stable

            self.models_[a] = _fit_model(effect_factory(eff_seeds()), X[idx], pseudo, weight)
            rows.append(
                {
                    "arm": a,
                    "n": int(idx.size),
                    "mean_abs_w_res": float(magnitude.mean()),
                    "clip_fraction": float(clipped.mean()),
                    "m_r2_oof": _weighted_r2(
                        y[idx], m_pair, None if sample_weight is None else sample_weight[idx]
                    ),
                }
            )

        self.residual_diagnostics_ = pd.DataFrame(rows)
        clipped_shares = [r["clip_fraction"] for r in rows if np.isfinite(r["clip_fraction"])]
        clip_fraction = float(np.mean(clipped_shares)) if clipped_shares else float("nan")

        self.nuisance_ = _nuisance_summary(
            learner=type(self).__name__,
            y=y,
            w=w,
            sample_weight=sample_weight,
            n_arms=self.n_arms,
            n_splits=used,
            crossfit=True,
            mu_oof=mu,
            e_oof=e,
            propensity_source=source,
            extra={
                "pairwise_nuisance": self.pairwise_nuisance,
                "residual_clip": float(self.residual_clip),
                "residual_clip_fraction": clip_fraction,
                "outcome_fallback_arms": sorted(fallbacks),
                "pairwise_m_r2_oof": {
                    int(r["arm"]): float(r["m_r2_oof"]) for r in rows if np.isfinite(r["m_r2_oof"])
                },
            },
            warnings_=warnings_,
        )

    def _predict_cate(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], self.n_arms - 1), dtype=np.float64)
        for a in range(1, self.n_arms):
            model = self.models_.get(a)
            if model is not None:
                out[:, a - 1] = np.asarray(model.predict(X), dtype=np.float64).ravel()
        return out


# ======================================================================================
# factory
# ======================================================================================
_REGISTRY: dict[str, type[BaseCATELearner]] = {
    "s": SLearner,
    "t": TLearner,
    "x": XLearner,
    "dr": DRLearner,
    "r": RLearner,
}

_LAZY: dict[str, tuple[str, str]] = {
    "forest": ("prism.causal.forest", "CausalForest"),
    "survival": ("prism.causal.survival_uplift", "CausalSurvivalUplift"),
}

_ALIASES: dict[str, str] = {
    "single": "s",
    "two": "t",
    "cross": "x",
    "doublyrobust": "dr",
    "aipw": "dr",
    "robinson": "r",
    "causalforest": "forest",
    "grf": "forest",
    "survivaluplift": "survival",
    "causalsurvivaluplift": "survival",
    "survivalup": "survival",
}


def _canonical_name(name: str) -> str:
    """Normalise a learner name: case, separators and a trailing ``learner`` are all ignored."""
    key = str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if key.endswith("learner"):
        key = key[: -len("learner")]
    return _ALIASES.get(key, key)


def make_learner(name: str, **kw: Any) -> BaseCATELearner:
    """Build a CATE learner by name.

    Parameters
    ----------
    name : str
        One of ``"s"``, ``"t"``, ``"x"``, ``"dr"``, ``"r"``, ``"forest"``, ``"survival"``.
        Case, hyphens, underscores and a trailing ``"learner"`` are ignored, so ``"DR-Learner"``,
        ``"dr_learner"`` and ``"dr"`` are the same request. A few descriptive aliases are also
        accepted (``"aipw"``, ``"robinson"``, ``"grf"``, ...); see :data:`LEARNER_NAMES` for the
        canonical set.
    **kw
        Forwarded verbatim to the learner's constructor. Because every learner accepts
        ``**kw`` and ignores what it does not recognise, one shared dictionary can be passed to
        all of them -- which is exactly what the leaderboard loop in
        ``prism.pipelines.run_all`` does.

    Returns
    -------
    BaseCATELearner
        An unfitted learner.

    Raises
    ------
    ValueError
        If ``name`` is not recognised. The message lists the valid names.
    ImportError
        If ``"forest"`` or ``"survival"`` is requested and the sibling module cannot be imported.
        Those two are imported **inside this function**, never at module import time, so a
        half-written or broken sibling module can never stop ``prism.causal.learners`` from being
        imported -- it only fails when that specific learner is actually requested, and then with
        a message that names the module and the underlying error.

    Examples
    --------
    >>> make_learner("DR-Learner", n_arms=3, n_splits=3).__class__.__name__
    'DRLearner'
    >>> make_learner("r", n_arms=2).n_arms
    2
    """
    key = _canonical_name(name)

    if key in _REGISTRY:
        return _REGISTRY[key](**kw)

    if key in _LAZY:
        module_path, class_name = _LAZY[key]
        try:
            import importlib

            module = importlib.import_module(module_path)
        except Exception as exc:
            raise ImportError(
                f"make_learner({name!r}) needs {module_path}.{class_name}, but importing "
                f"{module_path} failed: {type(exc).__name__}: {exc}. That module is part of the "
                f"causal layer and may not be available yet; the meta-learners "
                f"({', '.join(sorted(_REGISTRY))}) do not depend on it."
            ) from exc
        cls = getattr(module, class_name, None)
        if cls is None:
            raise ImportError(
                f"make_learner({name!r}) imported {module_path} successfully but it does not define "
                f"{class_name}."
            )
        return cls(**kw)

    raise ValueError(
        f"unknown learner {name!r}. Valid names are {list(LEARNER_NAMES)} "
        f"(case, '-', '_' and a trailing 'learner' are ignored)."
    )


# ======================================================================================
# smoke test: a confounded 3-arm problem with known heterogeneous effects
# ======================================================================================
def _simulate_confounded_multiarm(
    n: int = 6_000,
    p: int = 8,
    n_arms: int = 3,
    *,
    confounding: float = 1.0,
    noise: float = 1.0,
    random_state: int | np.random.Generator | None = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a confounded multi-arm problem with analytically known heterogeneous effects.

    A miniature of ``prism.data.dgp`` with the three properties that make CATE estimation
    non-trivial, and nothing else:

    1. **Confounding.** ``P(A = a | X)`` depends on the same covariates that drive the outcome
       (``x1`` for arm 1, ``x3`` for arm 2), so the naive difference in means is badly biased --
       both because treated units have a different baseline *and* because they have a different
       true effect (an ATT-versus-ATE gap on top of the selection bias).
    2. **Overlap.** Propensities are projected into ``[0.05, 0.90]``, so every arm is possible
       for every customer and the effects are identified.
    3. **Heterogeneous, interaction-driven effects.** ``tau_1`` depends on an interaction
       ``x1 * x2`` and ``tau_2`` on a threshold ``1{x4 > 0}``, so a constant-effect model cannot
       fit them and a linear model cannot either.

    Parameters
    ----------
    n : int, default 6000
        Rows.
    p : int, default 8
        Covariates. Must be at least 6.
    n_arms : int, default 3
        Arms including control. Only 3 is fully specified; larger values repeat the arm-2 effect.
    confounding : float, default 1.0
        Multiplier on the assignment logits. ``0`` gives a randomised design.
    noise : float, default 1.0
        Standard deviation of the outcome noise.
    random_state : int or numpy.random.Generator or None, default 7
        Seed material.

    Returns
    -------
    tuple of numpy.ndarray
        ``(X, w, y, tau, e_true)`` with shapes ``(n, p)``, ``(n,)``, ``(n,)``,
        ``(n, n_arms - 1)`` and ``(n, n_arms)``. ``tau[:, j]`` is the exact individual effect of
        arm ``j + 1`` against control.
    """
    if p < 6:
        raise ValueError("p must be at least 6")
    rng = as_rng(random_state)
    X = rng.normal(size=(n, p))
    X[:, 0] = np.abs(X[:, 0])  # one skewed covariate, as in the real panel

    baseline = (
        1.0 * X[:, 0]
        + 1.5 * X[:, 1]
        - 1.2 * X[:, 3]
        + 0.8 * np.sin(2.0 * X[:, 2])
        + 0.6 * X[:, 4] * X[:, 5]
    )

    tau = np.empty((n, n_arms - 1), dtype=np.float64)
    tau[:, 0] = 1.0 + 1.5 * X[:, 1] + 1.0 * X[:, 1] * X[:, 2]
    for j in range(1, n_arms - 1):
        tau[:, j] = 2.0 - 1.8 * X[:, 3] + 1.2 * (X[:, 4] > 0.0)

    logits = np.zeros((n, n_arms), dtype=np.float64)
    logits[:, 1] = confounding * (1.5 * X[:, 1] + 0.6 * X[:, 0]) - 0.4
    for a in range(2, n_arms):
        logits[:, a] = confounding * (-1.5 * X[:, 3] + 0.7 * X[:, 2]) - 0.4
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    e /= e.sum(axis=1, keepdims=True)
    e = np.clip(e, 0.05, 0.90)
    e /= e.sum(axis=1, keepdims=True)

    cdf = np.cumsum(e, axis=1)
    u = rng.random(size=(n, 1))
    w = np.clip((u > cdf).sum(axis=1), 0, n_arms - 1).astype(np.int64)

    tau_full = np.hstack([np.zeros((n, 1)), tau])
    y = baseline + tau_full[np.arange(n), w] + rng.normal(scale=noise, size=n)
    return X, w, y.astype(np.float64), tau, e


def _pehe(true_cate: np.ndarray, est_cate: np.ndarray) -> np.ndarray:
    """Per-arm precision in estimating heterogeneous effects, ``sqrt(mean((tau - tau_hat)^2))``.

    Computed inline rather than imported from ``prism.causal.evaluate`` so that this smoke test
    has no dependency on a module being written in parallel.

    Parameters
    ----------
    true_cate, est_cate : numpy.ndarray
        ``(n, n_arms - 1)`` arrays.

    Returns
    -------
    numpy.ndarray
        ``(n_arms - 1,)`` root mean squared error per arm.
    """
    diff = np.asarray(est_cate, dtype=np.float64) - np.asarray(true_cate, dtype=np.float64)
    return np.sqrt(np.mean(diff**2, axis=0))


def _naive_difference_in_means(y: np.ndarray, w: np.ndarray, n_arms: int) -> np.ndarray:
    """The estimator a dashboard would produce: ``mean(Y | A = a) - mean(Y | A = 0)``."""
    control = float(np.mean(y[w == 0]))
    return np.array(
        [float(np.mean(y[w == a])) - control if int(np.sum(w == a)) else np.nan for a in range(1, n_arms)],
        dtype=np.float64,
    )


def _check(rows: list[dict[str, Any]], name: str, ok: bool, detail: str) -> bool:
    """Append one pass/fail row to the smoke-test table and return ``ok``."""
    rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    if not ok:
        print(f"  !! FAILED: {name} -- {detail}")
    return bool(ok)


if __name__ == "__main__":  # pragma: no cover - smoke test
    t_start = time.perf_counter()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    N, P, K = 6_000, 8, 3
    SEED = 7
    X, w, y, TAU, E_TRUE = _simulate_confounded_multiarm(N, P, K, confounding=1.0, noise=1.0, random_state=SEED)

    true_ate = TAU.mean(axis=0)
    naive_ate = _naive_difference_in_means(y, w, K)
    naive_err = np.abs(naive_ate - true_ate)
    # The best possible CONSTANT-effect predictor is the true ATE, whose PEHE is sd(tau).
    pehe_const = _pehe(TAU, np.tile(true_ate, (N, 1)))

    print("=" * 108)
    print("PRISM / prism.causal.learners -- confounded 3-arm simulation with known heterogeneous effects")
    print("=" * 108)
    print(f"n = {N:,} | p = {P} | K = {K} | arm shares = {np.round(np.bincount(w, minlength=K) / N, 4).tolist()}")
    print(f"true propensity range  = [{E_TRUE.min():.4f}, {E_TRUE.max():.4f}]  (overlap holds)")
    print(f"true ATE               = {np.round(true_ate, 4).tolist()}")
    print(f"sd(tau) per arm        = {np.round(TAU.std(axis=0), 4).tolist()}   <- PEHE of the best constant model")
    print(f"naive diff-in-means    = {np.round(naive_ate, 4).tolist()}  (|error| = {np.round(naive_err, 4).tolist()})")
    print()

    COMMON = {
        "n_arms": K,
        "n_splits": 3,
        "diagnostic_splits": 2,
        "random_state": SEED,
        "n_estimators": 80,
        "learning_rate": 0.12,
        "min_arm_rows": 50,
    }

    # One shared, cross-fitted propensity for the whole leaderboard -- this is how
    # prism.pipelines.run_all uses them (SPEC_PERF section 6: fit each nuisance once, reuse it),
    # and it keeps the comparison like-for-like: every learner sees the same e(x), so a
    # difference in the table is a difference between estimators, not between their propensities.
    # The internal-fit path (propensity=None) is exercised separately by the clip check below.
    from prism.models.propensity import PropensityModel

    t_prop = time.perf_counter()
    E_HAT = PropensityModel(n_arms=K, n_splits=3, random_state=SEED, n_estimators=80).fit(X, w).oof_proba()
    print(f"shared cross-fitted propensity: {time.perf_counter() - t_prop:.2f}s | "
          f"range [{E_HAT.min():.4f}, {E_HAT.max():.4f}] vs true [{E_TRUE.min():.4f}, {E_TRUE.max():.4f}]")
    print()

    checks: list[dict[str, Any]] = []
    all_ok = True
    board: list[dict[str, Any]] = []
    fitted: dict[str, BaseCATELearner] = {}

    for short in ("s", "t", "x", "dr", "r"):
        learner = make_learner(short, **COMMON)
        t0 = time.perf_counter()
        learner.fit(X, w, y, propensity=E_HAT)
        seconds = time.perf_counter() - t0
        cate = learner.predict_cate(X)
        fitted[short] = learner

        shape_ok = cate.shape == (N, K - 1)
        finite_ok = bool(np.isfinite(cate).all())
        pehe = _pehe(TAU, cate)
        est_ate = cate.mean(axis=0)
        ate_err = np.abs(est_ate - true_ate)

        board.append(
            {
                "learner": type(learner).__name__,
                "pehe_1": pehe[0],
                "pehe_2": pehe[1],
                "pehe_mean": float(pehe.mean()),
                "ate_err_1": ate_err[0],
                "ate_err_2": ate_err[1],
                "ate_err_mean": float(ate_err.mean()),
                "outcome_r2_oof": float(learner.nuisance_.get("outcome_r2_oof", np.nan)),
                "prop_ll_skill": float(learner.nuisance_.get("propensity_logloss_skill", np.nan)),
                "seconds": seconds,
            }
        )

        tag = type(learner).__name__
        all_ok &= _check(checks, f"{tag}: predict_cate shape == (n, n_arms-1)", shape_ok and finite_ok,
                         f"{cate.shape}, all finite = {finite_ok}")
        all_ok &= _check(checks, f"{tag}: PEHE beats the best constant model", bool(np.all(pehe < pehe_const)),
                         f"PEHE {np.round(pehe, 4).tolist()} vs constant {np.round(pehe_const, 4).tolist()}")
        all_ok &= _check(checks, f"{tag}: ATE error beats the naive difference in means",
                         bool(np.all(ate_err < naive_err)),
                         f"|err| {np.round(ate_err, 4).tolist()} vs naive {np.round(naive_err, 4).tolist()}")
        all_ok &= _check(checks, f"{tag}: predict() is an alias of predict_cate()",
                         bool(np.array_equal(learner.predict(X[:200]), learner.predict_cate(X[:200]))),
                         "identical arrays")

    leaderboard = pd.DataFrame(board).sort_values("pehe_mean").reset_index(drop=True)

    print("--- LEADERBOARD (sorted by mean PEHE; lower is better) ---")
    print(
        leaderboard.assign(
            pehe_1=lambda d: d["pehe_1"].round(4),
            pehe_2=lambda d: d["pehe_2"].round(4),
            pehe_mean=lambda d: d["pehe_mean"].round(4),
            ate_err_1=lambda d: d["ate_err_1"].round(4),
            ate_err_2=lambda d: d["ate_err_2"].round(4),
            ate_err_mean=lambda d: d["ate_err_mean"].round(4),
            outcome_r2_oof=lambda d: d["outcome_r2_oof"].round(4),
            prop_ll_skill=lambda d: d["prop_ll_skill"].round(4),
            seconds=lambda d: d["seconds"].round(2),
        ).to_string(index=False)
    )
    print(
        f"\nbaselines: constant-effect PEHE = {np.round(pehe_const, 4).tolist()} "
        f"(mean {pehe_const.mean():.4f}) | naive diff-in-means |ATE error| = "
        f"{np.round(naive_err, 4).tolist()} (mean {naive_err.mean():.4f})"
    )

    # ---- the orthogonal learners should be at the sharp end -------------------------------
    order = leaderboard["learner"].tolist()
    orthogonal_rank = min(order.index("DRLearner"), order.index("RLearner"))
    median_pehe = float(leaderboard["pehe_mean"].median())
    best_orthogonal = float(leaderboard.loc[leaderboard["learner"].isin(["DRLearner", "RLearner"]), "pehe_mean"].min())
    all_ok &= _check(
        checks,
        "an orthogonal learner (DR or R) is in the top two by PEHE",
        orthogonal_rank <= 1,
        f"ranking = {order}",
    )
    all_ok &= _check(
        checks,
        "best orthogonal PEHE beats the median learner",
        best_orthogonal <= median_pehe,
        f"best(DR, R) = {best_orthogonal:.4f} vs median {median_pehe:.4f}",
    )

    # ---- THE test of the doubly-robust property itself ------------------------------------
    # Theory, worked through exactly. Break the outcome model as hard as it can be broken:
    # mu_a(x) := c_a, a constant per arm. Then the plug-in contrast c_a - c_0 IS the naive
    # difference in means, so the T-learner reproduces the confounded answer. The AIPW
    # pseudo-outcome, given the CORRECT e, collapses to plain IPW and stays consistent:
    #     E[psi] = (c_a - c_0) + E[1{A=a}(Y - c_a)/e_a] - E[1{A=0}(Y - c_0)/e_0]
    #            = (c_a - c_0) + (E[Y(a)] - c_a) - (E[Y(0)] - c_0)  =  ATE_a
    # If this check fails, the AIPW algebra in DRLearner._fit is wrong.
    from sklearn.dummy import DummyRegressor

    WRONG = {"n_arms": K, "n_splits": 3, "random_state": SEED,
             "base_outcome": DummyRegressor(strategy="mean"), "nuisance_diagnostics": False}
    t_wrong = TLearner(**WRONG).fit(X, w, y, propensity=E_TRUE)
    dr_wrong = DRLearner(**WRONG).fit(X, w, y, propensity=E_TRUE)
    t_wrong_err = float(np.abs(t_wrong.predict_cate(X).mean(axis=0) - true_ate).mean())
    dr_wrong_err = float(np.abs(dr_wrong.ate_aipw_ - true_ate).mean())
    all_ok &= _check(
        checks,
        "double robustness: with a WORTHLESS outcome model but the right e(x), DR survives and T does not",
        dr_wrong_err < 0.25 * t_wrong_err,
        f"mean |ATE error| -- plug-in T = {t_wrong_err:.4f} (= the naive difference in means, "
        f"{naive_err.mean():.4f}), doubly-robust AIPW = {dr_wrong_err:.4f} "
        f"({t_wrong_err / max(dr_wrong_err, 1e-9):.0f}x less bias)",
    )

    # ---- nuisance diagnostics are real, not decoration -------------------------------------
    dr = fitted["dr"]
    nz = dr.nuisance_
    all_ok &= _check(checks, "DR: out-of-fold outcome model has real signal", nz["outcome_r2_oof"] > 0.5,
                     f"R2 = {nz['outcome_r2_oof']:.4f} ({nz['outcome_flag']})")
    all_ok &= _check(checks, "DR: the shared propensity beats the marginal log-loss",
                     nz["propensity_logloss"] < nz["propensity_logloss_marginal"],
                     f"{nz['propensity_logloss']:.4f} < {nz['propensity_logloss_marginal']:.4f} "
                     f"(skill {nz['propensity_logloss_skill']:.4f}, flag {nz['propensity_flag']})")
    # Two things in one fit: the internal propensity path (propensity=None -> PropensityModel,
    # imported lazily), and the clip knob. The clip is the only thing standing between AIPW and
    # an unbounded weight, and tightening it must do what the docstring claims -- fewer extreme
    # weights and a bigger effective sample, bought with bias.
    tight = DRLearner(n_arms=K, n_splits=2, random_state=SEED, n_estimators=60,
                      propensity_clip=(0.05, 0.95)).fit(X, w, y)
    tz = tight.nuisance_
    all_ok &= _check(
        checks,
        "propensity fits internally when none is given, and propensity_clip bounds the AIPW weight",
        tz["propensity_source"] == "PropensityModel"
        and nz["max_inverse_propensity"] <= 1.0 / 0.01 + 1e-9
        and tz["max_inverse_propensity"] <= 1.0 / 0.05 + 1e-9
        and tz["ess_fraction"] > nz["ess_fraction"],
        f"source = {tz['propensity_source']} | clip 0.01 -> max 1/e = {nz['max_inverse_propensity']:.1f}, "
        f"ESS = {nz['ess_fraction']:.1%} of n | clip 0.05 -> max 1/e = {tz['max_inverse_propensity']:.1f}, "
        f"ESS = {tz['ess_fraction']:.1%} of n",
    )
    aipw_err = np.abs(dr.ate_aipw_ - true_ate)
    all_ok &= _check(checks, "DR: the AIPW ATE is within 3 influence-function SEs of the truth",
                     bool(np.all(aipw_err < 3.0 * dr.ate_aipw_se_)),
                     f"AIPW ATE {np.round(dr.ate_aipw_, 4).tolist()} +/- "
                     f"{np.round(1.96 * dr.ate_aipw_se_, 4).tolist()} vs truth {np.round(true_ate, 4).tolist()}")
    r_learner = fitted["r"]
    all_ok &= _check(checks, "R: the sign-preserving residual clip binds on ~no rows",
                     float(r_learner.nuisance_["residual_clip_fraction"]) < 1e-6,
                     f"clip fraction = {float(r_learner.nuisance_['residual_clip_fraction']):.2e} "
                     f"(|W_res| >= min(lo, 1-hi) by construction)")

    print("\n--- DRLearner.nuisance_ (the 'can I trust this?' panel) ---")
    print(dr.nuisance_frame().head(18).to_string(index=False))
    print("\n--- RLearner.residual_diagnostics_ ---")
    print(r_learner.residual_diagnostics_.round(5).to_string(index=False))

    # ---- determinism (SPEC_PERF section 10) ------------------------------------------------
    DET = {"n_arms": K, "n_splits": 2, "random_state": SEED, "n_estimators": 50,
           "nuisance_diagnostics": False}
    det_a = RLearner(**DET).fit(X, w, y, propensity=E_TRUE).predict_cate(X)
    det_b = RLearner(**DET).fit(X, w, y, propensity=E_TRUE).predict_cate(X)
    all_ok &= _check(checks, "two fits with the same seed give bit-identical CATE",
                     bool(np.array_equal(det_a, det_b)),
                     f"max abs diff = {np.max(np.abs(det_a - det_b)):.2e}")
    det_c = RLearner(**{**DET, "random_state": SEED + 1}).fit(X, w, y, propensity=E_TRUE).predict_cate(X)
    all_ok &= _check(checks, "a different seed gives a different fit (the seed is really wired in)",
                     not np.array_equal(det_a, det_c),
                     f"mean |delta| = {np.mean(np.abs(det_a - det_c)):.5f}")

    # ---- sample_weight is honoured, not silently dropped ---------------------------------
    CHEAP = {"n_arms": K, "n_splits": 2, "random_state": SEED, "n_estimators": 40,
             "learning_rate": 0.15, "nuisance_diagnostics": False}
    base_t = TLearner(**CHEAP).fit(X, w, y)
    tilt = 1.0 + 9.0 * (X[:, 1] > 0.5)
    tilted_t = TLearner(**CHEAP).fit(X, w, y, sample_weight=tilt)
    all_ok &= _check(checks, "sample_weight changes the fit (it is not dropped)",
                     not np.allclose(base_t.predict_cate(X), tilted_t.predict_cate(X)),
                     f"mean |delta CATE| = {np.mean(np.abs(base_t.predict_cate(X) - tilted_t.predict_cate(X))):.4f}")

    half = np.zeros(N, dtype=np.float64)
    half[as_rng(11).permutation(N)[: N // 2]] = 1.0
    mask = half > 0
    zero_weighted = TLearner(**CHEAP).fit(X, w, y, sample_weight=half)
    subset_only = TLearner(**CHEAP).fit(X[mask], w[mask], y[mask])
    corr = float(
        np.corrcoef(zero_weighted.predict_cate(X).ravel(), subset_only.predict_cate(X).ravel())[0, 1]
    )
    all_ok &= _check(checks, "zero-weight rows are really excluded", corr > 0.60,
                     f"corr(weight=0/1 fit, subset fit) = {corr:.4f}")

    # ---- supplied propensity, and the factory ---------------------------------------------
    dr_true_e = DRLearner(**COMMON).fit(X, w, y, propensity=E_TRUE)
    pehe_true_e = _pehe(TAU, dr_true_e.predict_cate(X))
    all_ok &= _check(checks, "a supplied propensity is used (source recorded, estimate still good)",
                     dr_true_e.nuisance_["propensity_source"] == "supplied"
                     and bool(np.all(pehe_true_e < pehe_const)),
                     f"source = {dr_true_e.nuisance_['propensity_source']}, "
                     f"PEHE with the TRUE e(x) = {np.round(pehe_true_e, 4).tolist()}")

    factory_ok = all(
        isinstance(make_learner(nm, n_arms=K), cls)
        for nm, cls in [("S", SLearner), ("t-learner", TLearner), ("X_Learner", XLearner),
                        ("DR", DRLearner), ("robinson", RLearner)]
    )
    all_ok &= _check(checks, "make_learner resolves names, aliases and casing", factory_ok,
                     "s / t-learner / X_Learner / DR / robinson all resolve")

    # sklearn clone() must survive the learners that absorb shared params through **kw,
    # because prism.causal.refute clones an estimator once per simulation.
    from sklearn.base import clone as _sk_clone

    clone_ok = True
    clone_detail = []
    for proto in (XLearner(n_arms=K, n_splits=3, blend="constant"),
                  RLearner(n_arms=K, n_splits=3, residual_clip=5e-3),
                  DRLearner(n_arms=K, n_splits=3, propensity_clip=(0.02, 0.98))):
        twin = _sk_clone(proto)
        same = twin.get_params() == proto.get_params() and twin.n_arms == K
        clone_ok &= bool(same)
        clone_detail.append(f"{type(proto).__name__}:{'ok' if same else 'LOST PARAMS'}")
    all_ok &= _check(checks, "sklearn clone() preserves every constructor parameter", clone_ok,
                     " | ".join(clone_detail))

    try:
        make_learner("nope")
        bad_name_ok = False
        bad_detail = "no exception raised"
    except ValueError as exc:
        bad_name_ok = "unknown learner" in str(exc)
        bad_detail = f"ValueError: {str(exc)[:70]}..."
    all_ok &= _check(checks, "make_learner rejects an unknown name with ValueError", bad_name_ok, bad_detail)

    lazy_detail = []
    lazy_ok = True
    for nm in ("forest", "survival"):
        try:
            obj = make_learner(nm, n_arms=K)
            lazy_detail.append(f"{nm}={type(obj).__name__}")
        except ImportError as exc:
            lazy_ok &= (nm in str(exc)) or ("prism.causal" in str(exc))
            lazy_detail.append(f"{nm}=ImportError(clear)")
        except Exception as exc:  # any other exception type is a contract violation
            lazy_ok = False
            lazy_detail.append(f"{nm}={type(exc).__name__}!")
    all_ok &= _check(checks, "lazy learners import cleanly or raise a clear ImportError", lazy_ok,
                     " | ".join(lazy_detail))

    # ---- rare-arm guard: two arms with 5 rows each must share one pooled fallback -----------
    # Exact test of the mechanism: both starved arms get the SAME pooled model, so their CATE
    # columns must be bit-identical -- and a 5-row model, had the guard not fired, would have
    # produced two different (and wild) columns instead.
    w_rare = w.copy()
    w_rare[w_rare == 2] = 1
    w_rare[:5] = 2
    w_rare[5:10] = 3
    rare = TLearner(n_arms=4, random_state=SEED, n_estimators=40, nuisance_diagnostics=False).fit(X, w_rare, y)
    rare_cate = rare.predict_cate(X)
    unguarded = TLearner(n_arms=4, random_state=SEED, n_estimators=40, min_arm_rows=2,
                         nuisance_diagnostics=False).fit(X, w_rare, y)
    ug_cate = unguarded.predict_cate(X)
    all_ok &= _check(
        checks,
        "two 5-row arms fall back to one shared pooled model (identical columns), and are logged",
        rare.fallback_arms_ == [2, 3]
        and rare_cate.shape == (N, 3)
        and bool(np.isfinite(rare_cate).all())
        and bool(np.array_equal(rare_cate[:, 1], rare_cate[:, 2]))
        and not np.array_equal(ug_cate[:, 1], ug_cate[:, 2]),
        f"fallback_arms_ = {rare.fallback_arms_}; guarded columns identical = "
        f"{bool(np.array_equal(rare_cate[:, 1], rare_cate[:, 2]))}; unguarded 5-row models disagree by "
        f"{float(np.abs(ug_cate[:, 1] - ug_cate[:, 2]).mean()):.3f} on average",
    )

    # ---- DataFrame input with categoricals --------------------------------------------------
    frame = pd.DataFrame(X[:1500, :4], columns=["a", "b", "c", "d"])
    frame["plan_tier"] = np.where(X[:1500, 4] > 0, "pro", "basic")
    t_frame = TLearner(n_arms=K, random_state=SEED, n_estimators=40, nuisance_diagnostics=False)
    t_frame.fit(frame, w[:1500], y[:1500])
    shifted = frame.iloc[:100].copy()
    shifted["plan_tier"] = "basic"  # a level vanishes at predict time
    all_ok &= _check(checks, "DataFrame with categoricals round-trips through predict",
                     t_frame.predict_cate(frame).shape == (1500, K - 1)
                     and t_frame.predict_cate(shifted).shape == (100, K - 1),
                     f"{t_frame.n_features_in_} expanded columns, missing level filled with zeros")

    elapsed = time.perf_counter() - t_start
    all_ok &= _check(checks, "smoke test runs in under 60 s", elapsed < 60.0, f"{elapsed:.1f}s")

    print("\n--- CHECKS ---")
    print(pd.DataFrame(checks).to_string(index=False))
    n_fail = int(sum(c["status"] == "FAIL" for c in checks))
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed in {elapsed:.1f}s")
    if not all_ok:
        raise SystemExit(f"learners.py SMOKE TEST FAILED ({n_fail} failing check(s))")
    print("learners.py OK")
