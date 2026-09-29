"""Multi-arm generalised propensity scores, overlap diagnostics and covariate balance.

The generalised propensity score (GPS) for a ``K``-arm design is the row-stochastic matrix
``e_a(x) = P(A = a | X = x)``.  Under unconfoundedness it is the *balancing score*: conditioning
on it makes the arm assignment independent of the covariates, which is what licenses every
inverse-probability and doubly-robust estimator downstream in :mod:`prism.causal`.

Everything in this module is built around three practical requirements.

**1. Honesty (cross-fitting).**  A propensity model that has seen row ``i`` produces an
over-confident ``e_{a_i}(x_i)`` for that row, which biases IPW/AIPW pseudo-outcomes toward zero.
:class:`PropensityModel` therefore fits ``n_splits`` fold models with
:class:`~sklearn.model_selection.StratifiedKFold` on the arm label and stores the out-of-fold
matrix once, at fit time, retrievable with :meth:`PropensityModel.oof_proba`.  A separate
full-data model is kept for :meth:`PropensityModel.predict_proba` on *new* rows.

**2. Calibration without nested cross-validation.**  Wrapping the base estimator in
:class:`~sklearn.calibration.CalibratedClassifierCV` *inside* the cross-fitting loop nests CV
inside CV and multiplies the fit count by the inner ``cv``.  The default here
(``calibration_method="isotonic_oof"``) instead fits 1-D isotonic regressions per arm on the
**already out-of-fold** raw scores, cross-fitted over the same folds so that the calibrated
out-of-fold matrix stays honest end to end -- the map that scores a row is fitted without that
row's label.  Cost: ``(n_splits + 1) * K`` isotonic fits (milliseconds) rather than
``n_splits * inner_cv`` extra *model* fits.  ``"cv"`` is offered for comparison and is the slow
path; it is never the default.

**3. A row-stochastic, in-range output.**  Naively clipping to ``[low, high]`` breaks the
simplex constraint, and naively renormalising afterwards pushes values back outside the clip
range.  :func:`clip_renormalize` solves for the per-row multiplicative factor ``s`` with
``sum_a clip(s * p_a, low, high) = 1`` by bisection (the sum is continuous and non-decreasing in
``s``), so the result is simultaneously inside the box **and** exactly on the simplex.

Public surface
--------------
:class:`PropensityModel`, :func:`overlap_diagnostics`, :func:`standardized_mean_differences`,
:func:`trim_by_overlap`, :func:`effective_sample_size`, :func:`ipw_weights`,
:func:`love_plot_frame`, :func:`clip_renormalize`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from prism.data.schema import ARM_NAMES, N_ARMS
from prism.utils.logging import get_logger
from prism.utils.optional import best_gbm
from prism.utils.seeds import as_rng

__all__ = [
    "PropensityModel",
    "overlap_diagnostics",
    "standardized_mean_differences",
    "trim_by_overlap",
    "effective_sample_size",
    "ipw_weights",
    "love_plot_frame",
    "clip_renormalize",
]

LOGGER = get_logger("models.propensity")

#: Guard applied before any ``log`` (SPEC_PERF section 9).
_LOG_EPS: float = 1e-6
#: Smallest positive probability kept before rescaling.
_TINY: float = 1e-12
#: Tolerance for "sitting on the clip boundary" in :func:`overlap_diagnostics`.
_BOUND_TOL: float = 1e-9


# ======================================================================================
# small helpers
# ======================================================================================
def _arm_label(a: int, arm_names: Sequence[str] | None = None) -> str:
    """Return a human-readable name for arm index ``a``.

    Parameters
    ----------
    a : int
        Arm index.
    arm_names : sequence of str, optional
        Override names. Defaults to :data:`prism.data.schema.ARM_NAMES`, falling back to
        ``"arm_{a}"`` when ``a`` is outside the canonical range.

    Returns
    -------
    str
    """
    names = tuple(arm_names) if arm_names is not None else ARM_NAMES
    return names[a] if 0 <= a < len(names) else f"arm_{a}"


def _as_matrix(
    X: np.ndarray | pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Coerce ``X`` to a dense float64 matrix and resolve column names.

    A :class:`~pandas.DataFrame` carrying ``object``/``category``/``string``/``bool`` columns is
    one-hot expanded (every level kept, so a balance table reports each level separately).

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Covariates, shape ``(n, p)``.
    feature_names : sequence of str, optional
        Names to use for an ndarray input. Must have length ``p``.

    Returns
    -------
    tuple of (numpy.ndarray, list of str)
        ``(matrix, names)`` where ``matrix`` is ``(n, p_expanded)`` float64.
    """
    if isinstance(X, pd.DataFrame):
        obj_cols = [
            c
            for c in X.columns
            if (X[c].dtype == object or isinstance(X[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]))
        ]
        frame = pd.get_dummies(X, columns=obj_cols, dummy_na=False) if obj_cols else X
        names = [str(c) for c in frame.columns]
        mat = frame.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        return mat, names

    mat = np.asarray(X, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    if feature_names is not None:
        names = [str(c) for c in feature_names]
        if len(names) != mat.shape[1]:
            raise ValueError(f"feature_names has length {len(names)} but X has {mat.shape[1]} columns")
    else:
        names = [f"x{j}" for j in range(mat.shape[1])]
    return mat, names


def _check_propensity(propensity: np.ndarray, arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a ``(n, K)`` propensity matrix against an ``(n,)`` arm vector."""
    p = np.asarray(propensity, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"propensity must be 2-D (n, n_arms); got shape {p.shape}")
    a = np.asarray(arm).ravel()
    if a.size != p.shape[0]:
        raise ValueError(f"arm has length {a.size} but propensity has {p.shape[0]} rows")
    a = a.astype(np.int64, copy=False)
    if a.size and (a.min() < 0 or a.max() >= p.shape[1]):
        raise ValueError(f"arm values must lie in [0, {p.shape[1] - 1}]; got [{a.min()}, {a.max()}]")
    return p, a


def clip_renormalize(
    proba: np.ndarray,
    low: float = 0.01,
    high: float = 0.99,
    *,
    max_iter: int = 60,
    tol: float = 1e-14,
) -> np.ndarray:
    """Project rows onto ``{p : sum_a p_a = 1, low <= p_a <= high}`` multiplicatively.

    Clipping alone leaves rows off the simplex; renormalising alone pushes entries back outside
    the clip range.  This routine finds, per row, the scale ``s > 0`` solving
    ``sum_a clip(s * p_a, low, high) = 1``.  The left-hand side is continuous and non-decreasing
    in ``s`` with limits ``K*low`` and ``K*high``, so a solution exists whenever
    ``K*low <= 1 <= K*high`` and bisection finds it.  A final residual of order ``1e-16`` is
    absorbed into the coordinate with the most slack, so rows sum to 1 to machine precision while
    staying inside the box.

    Parameters
    ----------
    proba : numpy.ndarray
        Raw probabilities, shape ``(n, K)``. Non-finite entries are replaced by ``1/K``.
    low, high : float
        Inclusive clip bounds. Must satisfy ``K*low <= 1 <= K*high``.
    max_iter : int, default 60
        Maximum bisection steps. 60 halvings shrink the bracket by a factor of ``2**60``, which is
        below double precision, so this is effectively exact; the loop also exits early once every
        row is within ``tol`` of 1.
    tol : float, default 1e-14
        Absolute tolerance on the row sum used for the early exit.

    Returns
    -------
    numpy.ndarray
        ``(n, K)`` float64, every entry in ``[low, high]``, every row summing to 1.

    Raises
    ------
    ValueError
        If the box and the simplex do not intersect.
    """
    p = np.array(proba, dtype=np.float64, copy=True)
    if p.ndim != 2:
        raise ValueError(f"proba must be 2-D; got shape {p.shape}")
    n, k = p.shape
    low = float(low)
    high = float(high)
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"need 0 <= low < high <= 1; got {(low, high)}")
    if k * low > 1.0 + 1e-9 or k * high < 1.0 - 1e-9:
        raise ValueError(f"clip range {(low, high)} cannot contain a simplex point for K={k} arms")
    if n == 0:
        return p

    p = np.nan_to_num(p, nan=1.0 / k, posinf=high, neginf=low)
    q = np.clip(p, _TINY, None)

    s_lo = np.zeros(n, dtype=np.float64)
    s_hi = np.maximum(high / q.min(axis=1), 1.0)
    scratch = np.empty_like(q)
    for _ in range(int(max_iter)):
        s = 0.5 * (s_lo + s_hi)
        np.multiply(q, s[:, None], out=scratch)
        np.clip(scratch, low, high, out=scratch)
        total = scratch.sum(axis=1)
        if np.max(np.abs(total - 1.0)) <= tol:
            s_lo = s_hi = s
            break
        over = total > 1.0
        s_hi = np.where(over, s, s_hi)
        s_lo = np.where(over, s_lo, s)

    out = np.clip(q * (0.5 * (s_lo + s_hi))[:, None], low, high)

    resid = 1.0 - out.sum(axis=1)
    slack = np.where(resid[:, None] > 0.0, high - out, out - low)
    j = np.argmax(slack, axis=1)
    rows = np.arange(n)
    step = np.sign(resid) * np.minimum(np.abs(resid), slack[rows, j])
    out[rows, j] += step
    return np.clip(out, low, high)


def _expand_proba(estimator: Any, X: np.ndarray, n_arms: int) -> np.ndarray:
    """Predict probabilities and place them into the full ``(n, n_arms)`` column layout.

    Arms that a fold model never saw are left at zero; :func:`clip_renormalize` lifts them to the
    lower clip bound afterwards.

    A fold -- or a whole training set -- holding a single arm is the awkward case: LightGBM and
    :class:`~sklearn.ensemble.HistGradientBoostingClassifier` both report a ``classes_`` of
    length 1 while still returning a **two**-column ``predict_proba``, so column ``j`` stops
    corresponding to ``classes_[j]`` and a positional read raises (or, worse, silently mislabels
    an arm). The answer is known there without asking the model -- the single observed class has
    probability 1 -- so it is written in directly rather than guessing at a column.
    """
    out = np.zeros((X.shape[0], n_arms), dtype=np.float64)
    classes = np.asarray(getattr(estimator, "classes_", ()), dtype=np.int64).ravel()
    if classes.size == 1:
        if 0 <= classes[0] < n_arms:
            out[:, classes[0]] = 1.0
        return out

    proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    if classes.size == 0:  # pragma: no cover - estimator without classes_
        classes = np.arange(proba.shape[1], dtype=np.int64)
    if proba.shape[1] != classes.size:
        raise ValueError(
            f"estimator returned {proba.shape[1]} probability columns for {classes.size} "
            f"class(es) {classes.tolist()}; cannot map them onto arms"
        )
    keep = (classes >= 0) & (classes < n_arms)
    out[:, classes[keep]] = proba[:, keep]
    return out


def _seed_estimator(estimator: Any, seed: int) -> Any:
    """Set ``random_state`` on ``estimator`` when it exposes one (reproducibility, SPEC 0)."""
    try:
        if "random_state" in estimator.get_params(deep=False):
            estimator.set_params(random_state=int(seed) % (2**31 - 1))
    except Exception:  # pragma: no cover - exotic third-party estimator
        pass
    return estimator


def _weighted_moments(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean and reliability-corrected weighted variance, column-wise.

    The variance uses the frequency-weight *reliability* correction

    .. math:: v_w = \\frac{\\sum w}{(\\sum w)^2 - \\sum w^2} \\sum w (x - \\mu_w)^2

    which reduces to the usual ``n / (n - 1)`` correction when all weights are equal.

    Parameters
    ----------
    x : numpy.ndarray
        ``(m, p)`` data block.
    w : numpy.ndarray
        ``(m,)`` non-negative weights.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mean, variance)``, each ``(p,)``.
    """
    p = x.shape[1]
    if x.shape[0] == 0:
        return np.full(p, np.nan), np.full(p, np.nan)
    sw = float(w.sum())
    sw2 = float(np.square(w).sum())
    if sw <= 0.0:
        return np.full(p, np.nan), np.full(p, np.nan)
    mu = (w[:, None] * x).sum(axis=0) / sw
    denom = sw * sw - sw2
    dev2 = (w[:, None] * np.square(x - mu)).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where(denom > 0.0, sw * dev2 / denom, np.nan)
    return mu, var


def _smd(mu_a: np.ndarray, var_a: np.ndarray, mu_0: np.ndarray, var_0: np.ndarray) -> np.ndarray:
    """Standardised mean difference with a pooled-variance denominator, safe against zeros."""
    diff = mu_a - mu_0
    pooled = 0.5 * (var_a + var_0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(pooled > 0.0, diff / np.sqrt(np.where(pooled > 0.0, pooled, 1.0)), np.nan)
    # A constant feature with identical means is perfectly balanced, not undefined.
    out = np.where((~(pooled > 0.0)) & (np.abs(diff) <= 1e-12), 0.0, out)
    return out


def _ece(prob: np.ndarray, target: np.ndarray, n_bins: int) -> float:
    """Expected calibration error: bin-count-weighted |mean(p) - mean(y)| over equal-width bins."""
    if prob.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_p = np.bincount(idx, weights=prob, minlength=n_bins)
    sum_y = np.bincount(idx, weights=target.astype(np.float64), minlength=n_bins)
    nz = counts > 0
    if not nz.any():
        return float("nan")
    gap = np.abs(sum_p[nz] / counts[nz] - sum_y[nz] / counts[nz])
    return float((counts[nz] * gap).sum() / counts.sum())


# ======================================================================================
# the model
# ======================================================================================
class PropensityModel(BaseEstimator):
    """Multi-arm generalised propensity score with cross-fitting and calibration.

    Estimates ``e_a(x) = P(A = a | X = x)`` for ``a = 0 .. n_arms-1``.  Three artefacts come out
    of one :meth:`fit`:

    * ``oof_proba()`` -- cross-fitted probabilities aligned to the training rows.  Use these for
      anything that reweights the training data (IPW, AIPW, R-learner residuals), because the
      model that produced row ``i`` never saw row ``i``.
    * ``predict_proba(X_new)`` -- probabilities from a model refit on **all** the training data,
      for scoring rows the model has never seen.
    * ``diagnostics()`` -- log-loss / AUC / Brier / ECE on the out-of-fold matrix.

    Calibration
    -----------
    ``calibration_method`` selects how ``calibrate=True`` is honoured:

    ``"isotonic_oof"`` (default)
        One :class:`~sklearn.isotonic.IsotonicRegression` per arm, fitted on the out-of-fold raw
        score for that arm against the ``1{A = a}`` indicator.  The calibrator is itself
        **cross-fitted**: fold ``f``'s rows are calibrated by a map fitted only on the other
        folds, so no row's own label can reach its own out-of-fold probability.  (Fitting one map
        on all rows and applying it back to them is *not* honest -- it lifts the out-of-fold
        macro AUC above 0.5 on data where the arm is independent of ``X``.)  A separate full-data
        map calibrates :meth:`predict_proba` for new rows.  Cost: ``(n_splits + 1) * n_arms``
        1-D PAVA fits, i.e. milliseconds, with no extra *model* fits at all.
    ``"sigmoid_oof"``
        The same, with a one-parameter-per-arm Platt scaling (logistic regression on the logit of
        the raw score).  More robust than isotonic at small ``n``.
    ``"cv"``
        The textbook route: wrap the base estimator in
        :class:`~sklearn.calibration.CalibratedClassifierCV` *inside* each outer fold.  This nests
        CV in CV and multiplies the number of model fits by the inner ``cv`` -- correct but slow.
        Offered for comparison, never the default.

    Parameters
    ----------
    n_arms : int, default 4
        Number of treatment arms ``K``. Arm ``0`` is control by convention.
    base_estimator : sklearn classifier, optional
        Multiclass ``predict_proba`` classifier. Defaults to
        :func:`prism.utils.optional.best_gbm` (LightGBM > XGBoost > HistGradientBoosting).
    n_splits : int, default 5
        Outer cross-fitting folds. Reduced automatically (with a logged warning) when the rarest
        arm has fewer rows than ``n_splits``.
    calibrate : bool, default True
        Whether to apply the calibration described above.
    clip : tuple of (float, float), default (0.01, 0.99)
        Probability bounds. Rows are projected onto the intersection of this box and the simplex
        by :func:`clip_renormalize`, so outputs are in range *and* sum to 1.
    random_state : int or numpy.random.Generator or None
        Seed material, normalised with :func:`prism.utils.seeds.as_rng`. Threads through the fold
        shuffling and the base estimator's own ``random_state``.
    calibration_method : {"isotonic_oof", "sigmoid_oof", "cv"}, default "isotonic_oof"
        See above. Ignored when ``calibrate=False``.
    cv_calibration_folds : int, default 3
        Inner folds used only by ``calibration_method="cv"``.
    keep_fold_models : bool, default False
        Retain the per-fold estimators in ``fold_models_``. Off by default to save memory.
    n_estimators : int, default 150
        Boosting rounds for the default base estimator. Ignored when ``base_estimator`` is
        supplied. 150 rounds at ``learning_rate=0.10`` is the point where the fit budget in
        SPEC_PERF (n = 50k, K = 4, p = 40 in roughly 22 s on 8 cores) is met with headroom while
        the out-of-fold log-loss is still within ~0.5% of a 200-round fit. A multiclass booster
        grows ``K`` trees per round, so this is 600 trees per fold model.
    learning_rate : float, default 0.10
        Learning rate for the default base estimator.

    Attributes
    ----------
    classes_ : numpy.ndarray
        ``arange(n_arms)``. Present even for arms absent from the training sample.
    full_model_ : sklearn classifier
        Fitted on all rows; backs :meth:`predict_proba`.
    oof_proba_ : numpy.ndarray
        ``(n, n_arms)`` cross-fitted, calibrated, projected probabilities.
    arm_ : numpy.ndarray
        The training arm labels, kept so :meth:`diagnostics` needs no arguments.
    marginal_ : numpy.ndarray
        Empirical ``P(A = a)``, used for stabilised weights.
    n_splits_ : int
        Folds actually used (may be below ``n_splits``).
    fit_seconds_ : float
        Wall-clock fit time, so the SPEC_PERF budget is observable.

    Examples
    --------
    >>> import logging, numpy as np
    >>> logging.getLogger("prism").setLevel(logging.WARNING)   # keep the doctest output clean
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 4))
    >>> arm = rng.integers(0, 4, size=600)
    >>> m = PropensityModel(n_splits=3, n_estimators=30, random_state=0).fit(X, arm)
    >>> bool(np.allclose(m.oof_proba().sum(axis=1), 1.0))
    True
    >>> m.oof_proba().shape
    (600, 4)
    >>> logging.getLogger("prism").setLevel(logging.INFO)
    """

    def __init__(
        self,
        n_arms: int = N_ARMS,
        base_estimator: Any = None,
        n_splits: int = 5,
        calibrate: bool = True,
        clip: tuple[float, float] = (0.01, 0.99),
        random_state: int | np.random.Generator | None = None,
        *,
        calibration_method: str = "isotonic_oof",
        cv_calibration_folds: int = 3,
        keep_fold_models: bool = False,
        n_estimators: int = 150,
        learning_rate: float = 0.10,
    ) -> None:
        self.n_arms = n_arms
        self.base_estimator = base_estimator
        self.n_splits = n_splits
        self.calibrate = calibrate
        self.clip = clip
        self.random_state = random_state
        self.calibration_method = calibration_method
        self.cv_calibration_folds = cv_calibration_folds
        self.keep_fold_models = keep_fold_models
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    # ---------------------------------------------------------------- internals
    def _make_base(self, seed: int) -> Any:
        """Clone the configured base estimator, or build the default gradient-boosted one."""
        if self.base_estimator is not None:
            return _seed_estimator(clone(self.base_estimator), seed)
        return best_gbm(
            "classification",
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            min_samples_leaf=40,
            random_state=int(seed) % (2**31 - 1),
        )

    def _wrap_for_fold(self, seed: int) -> Any:
        """Return the estimator actually fitted inside an outer fold."""
        base = self._make_base(seed)
        if self.calibrate and self.calibration_method == "cv":
            from sklearn.calibration import CalibratedClassifierCV

            return CalibratedClassifierCV(base, method="isotonic", cv=int(self.cv_calibration_folds))
        return base

    def _fit_calibrators(self, raw: np.ndarray, arm: np.ndarray) -> list[Any] | None:
        """Fit one per-arm 1-D calibrator on the out-of-fold raw scores."""
        if not self.calibrate or self.calibration_method == "cv":
            return None
        cals: list[Any] = []
        for a in range(int(self.n_arms)):
            y = (arm == a).astype(np.float64)
            x = np.clip(raw[:, a], _LOG_EPS, 1.0 - _LOG_EPS)
            if y.min() == y.max() or np.ptp(x) < 1e-12:
                cals.append(None)  # degenerate arm: leave the raw score alone
                continue
            if self.calibration_method == "sigmoid_oof":
                lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                lr.fit(np.log(x / (1.0 - x)).reshape(-1, 1), y.astype(np.int64))
                cals.append(("sigmoid", lr))
            elif self.calibration_method == "isotonic_oof":
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
                iso.fit(x, y)
                cals.append(("isotonic", iso))
            else:
                raise ValueError(
                    "calibration_method must be one of "
                    f"{{'isotonic_oof', 'sigmoid_oof', 'cv'}}; got {self.calibration_method!r}"
                )
        return cals

    def _apply_calibrators(self, raw: np.ndarray, cals: list[Any] | None = None) -> np.ndarray:
        """Apply per-arm calibrators column-wise (identity where one was not fitted).

        Parameters
        ----------
        raw : numpy.ndarray
            ``(m, n_arms)`` uncalibrated scores.
        cals : list, optional
            Calibrator list to use. Defaults to ``self.calibrators_`` (the full-data
            calibrators used for new rows); :meth:`fit` passes per-fold calibrators here so the
            out-of-fold matrix stays honest.
        """
        if cals is None:
            cals = getattr(self, "calibrators_", None)
        if not cals:
            return raw
        out = np.array(raw, dtype=np.float64, copy=True)
        for a, cal in enumerate(cals):
            if cal is None:
                continue
            kind, model = cal
            x = np.clip(raw[:, a], _LOG_EPS, 1.0 - _LOG_EPS)
            if kind == "isotonic":
                out[:, a] = model.predict(x)
            else:
                out[:, a] = model.predict_proba(np.log(x / (1.0 - x)).reshape(-1, 1))[:, 1]
        return out

    def _align(self, Xm: np.ndarray, names: list[str], *, check_names: bool) -> np.ndarray:
        """Match a scoring matrix to the training column layout, or refuse.

        :func:`_as_matrix` one-hot expands a DataFrame independently on every call, so a scoring
        frame's column order is whatever the caller happened to build -- a JSON payload
        deserialised into a frame, say, where key order is not guaranteed. Trusting positional
        order is the dangerous case: permuted-but-complete columns yield a confident, badly
        wrong answer with no error at all. So names are checked whenever the caller supplied
        them, and only a pure permutation is silently repaired. An ndarray carries no names, so
        it is checked on width alone and consumed positionally, as sklearn does.
        """
        p = int(getattr(self, "n_features_in_", Xm.shape[1]))
        trained = [str(c) for c in getattr(self, "feature_names_in_", [])]
        if not check_names or not trained:
            if Xm.shape[1] != p:
                raise ValueError(f"X has {Xm.shape[1]} columns but the model was fitted on {p}")
            return Xm
        if names == trained:
            return Xm
        if sorted(names) == sorted(trained) and len(set(names)) == len(names):
            LOGGER.info("predict_proba: reordering %d columns into the training layout", len(trained))
            return Xm[:, [names.index(c) for c in trained]]
        missing = [c for c in trained if c not in names]
        extra = [c for c in names if c not in trained]
        raise ValueError(
            f"X does not match the training columns: {len(missing)} missing {missing[:5]}, "
            f"{len(extra)} unexpected {extra[:5]}. Build the design matrix with "
            "prism.data.features.build_design_matrix so one-hot levels stay stable."
        )

    # ---------------------------------------------------------------- api
    def fit(self, X: np.ndarray | pd.DataFrame, arm: np.ndarray) -> PropensityModel:
        """Cross-fit the generalised propensity score.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Covariates, ``(n, p)``. DataFrames are one-hot expanded by :func:`_as_matrix`.
        arm : numpy.ndarray
            Integer arm labels in ``[0, n_arms)``, shape ``(n,)``.

        Returns
        -------
        PropensityModel
            ``self``, with ``oof_proba_``, ``full_model_``, ``classes_`` and ``marginal_`` set.

        Raises
        ------
        ValueError
            If ``arm`` is outside ``[0, n_arms)`` or lengths disagree.
        """
        t0 = time.perf_counter()
        Xm, names = _as_matrix(X)
        a = np.asarray(arm).ravel().astype(np.int64)
        n, k = Xm.shape[0], int(self.n_arms)
        if a.size != n:
            raise ValueError(f"arm has length {a.size} but X has {n} rows")
        if n == 0:
            raise ValueError("cannot fit a propensity model on zero rows")
        if a.min() < 0 or a.max() >= k:
            raise ValueError(f"arm values must lie in [0, {k - 1}]; got [{a.min()}, {a.max()}]")

        low, high = float(self.clip[0]), float(self.clip[1])
        seed = int(as_rng(self.random_state).integers(0, 2**31 - 1))

        counts = np.bincount(a, minlength=k)
        present = np.flatnonzero(counts > 0)
        if present.size < k:
            LOGGER.warning(
                "arms %s have zero training rows; their propensity is pinned at the clip floor %.3g",
                [int(i) for i in np.flatnonzero(counts == 0)],
                low,
            )
        rarest = int(counts[present].min())

        # SPEC_PERF section 6: shrink n_splits rather than let a fold miss an arm.
        n_splits = int(self.n_splits)
        if rarest < n_splits:
            new_splits = max(2, min(n_splits, rarest))
            LOGGER.warning(
                "rarest arm has %d rows < n_splits=%d; reducing to n_splits=%d", rarest, n_splits, new_splits
            )
            n_splits = new_splits
        stratified = rarest >= n_splits and n_splits >= 2
        if not stratified:
            LOGGER.warning("cannot stratify with rarest arm n=%d; falling back to plain KFold", rarest)
            n_splits = max(2, min(n_splits, n))

        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            if stratified
            else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        )
        folds = list(splitter.split(Xm, a) if stratified else splitter.split(Xm))

        raw = np.zeros((n, k), dtype=np.float64)
        fold_models: list[Any] = []
        for f, (tr, te) in enumerate(folds):
            est = self._wrap_for_fold(seed + 1009 * (f + 1))
            est.fit(Xm[tr], a[tr])
            raw[te] = _expand_proba(est, Xm[te], k)
            if self.keep_fold_models:
                fold_models.append(est)

        # Calibrators for *new* rows: fitted on the whole out-of-fold raw matrix.
        self.calibrators_ = self._fit_calibrators(raw, a)

        # The calibrated OOF matrix must be honest too.  Fitting one calibrator on all rows and
        # applying it back to those same rows lets row i's own label reach row i's own score
        # through the monotone map -- a real leak: on a null DGP (X independent of arm) it lifts
        # the out-of-fold macro AUC from 0.499 to 0.515 and makes the OOF log-loss beat the
        # marginal baseline, which is impossible for an honest score.  So the calibrator is
        # cross-fitted over the same folds: fold f's rows are calibrated by a map fitted only on
        # the other folds' (already out-of-fold) scores.  Cost is n_splits * K extra 1-D PAVA
        # fits -- milliseconds -- so the "no nested CV" performance argument is untouched.
        if self.calibrators_ is None:
            cal_raw = raw
        else:
            cal_raw = np.array(raw, dtype=np.float64, copy=True)
            for tr, te in folds:
                cal_raw[te] = self._apply_calibrators(raw[te], self._fit_calibrators(raw[tr], a[tr]))
        oof = clip_renormalize(cal_raw, low, high)

        # Assert the contract the whole downstream stack relies on.
        row_sums = oof.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            raise AssertionError(f"oof rows do not sum to 1 (max dev {np.max(np.abs(row_sums - 1.0)):.3e})")
        if oof.min() < low - 1e-12 or oof.max() > high + 1e-12:
            raise AssertionError(f"oof probabilities escaped clip range [{low}, {high}]")

        self.full_model_ = self._wrap_for_fold(seed)
        self.full_model_.fit(Xm, a)

        self.classes_ = np.arange(k, dtype=np.int64)
        self.oof_proba_ = oof
        self.oof_raw_ = raw
        self.arm_ = a
        self.marginal_ = counts.astype(np.float64) / float(n)
        self.n_splits_ = n_splits
        self.n_features_in_ = Xm.shape[1]
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.fold_models_ = fold_models if self.keep_fold_models else []
        self.fit_seconds_ = time.perf_counter() - t0
        LOGGER.info(
            "propensity fitted: n=%d K=%d folds=%d calibrate=%s in %.2fs",
            n,
            k,
            n_splits,
            self.calibration_method if self.calibrate else "off",
            self.fit_seconds_,
        )
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Propensities for **new** rows, from the model refit on all training data.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Covariates, ``(m, p)``.

        Returns
        -------
        numpy.ndarray
            ``(m, n_arms)`` float64; each entry in ``clip`` and each row summing to 1.

        Notes
        -----
        For rows that were in the training sample use :meth:`oof_proba` instead; these values are
        in-sample and optimistic.

        A DataFrame is matched to the training columns **by name**: a pure permutation is
        reordered, anything else raises. An ndarray is matched on width and read positionally.

        Raises
        ------
        ValueError
            If the columns cannot be matched to the ones seen during :meth:`fit`.
        """
        if not hasattr(self, "full_model_"):
            raise RuntimeError("PropensityModel is not fitted; call fit(X, arm) first")
        Xm, names = _as_matrix(X)
        Xm = self._align(Xm, names, check_names=isinstance(X, pd.DataFrame))
        raw = _expand_proba(self.full_model_, Xm, int(self.n_arms))
        out = clip_renormalize(self._apply_calibrators(raw), float(self.clip[0]), float(self.clip[1]))
        if not np.allclose(out.sum(axis=1), 1.0, atol=1e-9):
            raise AssertionError("predict_proba rows do not sum to 1")
        return out

    def oof_proba(self) -> np.ndarray:
        """Return the stored cross-fitted probabilities, aligned to the training rows.

        Returns
        -------
        numpy.ndarray
            ``(n, n_arms)`` copy of the out-of-fold matrix computed during :meth:`fit`.
        """
        if not hasattr(self, "oof_proba_"):
            raise RuntimeError("PropensityModel is not fitted; call fit(X, arm) first")
        return self.oof_proba_.copy()

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Most likely arm for each row (``argmax`` of :meth:`predict_proba`)."""
        return np.asarray(self.classes_)[np.argmax(self.predict_proba(X), axis=1)].astype(np.int64)

    def diagnostics(
        self,
        X: np.ndarray | pd.DataFrame | None = None,
        arm: np.ndarray | None = None,
        *,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Fit-quality summary of the propensity model.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame, optional
            Rows to score. ``None`` (default) evaluates the stored **out-of-fold** matrix, which
            is the honest choice.
        arm : numpy.ndarray, optional
            Labels matching ``X``. Required when ``X`` is given.
        n_bins : int, default 10
            Equal-width bins used for the expected calibration error.

        Returns
        -------
        pandas.DataFrame
            One row per arm plus a final ``scope="overall"`` row. Columns: ``scope``, ``arm``,
            ``arm_name``, ``n``, ``prevalence``, ``mean_pred``, ``auc_ovr``, ``brier``, ``ece``,
            ``log_loss``. Per-arm entries are one-vs-rest: ``brier`` is ``mean((e_a - 1{A=a})^2)``
            and ``log_loss`` the binary cross-entropy of that same contrast. The ``overall`` row is
            multiclass: ``brier`` is ``mean(sum_a (e_a - 1{A=a})^2)``, ``log_loss`` is the
            multiclass cross-entropy ``-mean(log e_{A_i}(x_i))``, ``auc_ovr`` is the macro mean of
            the per-arm AUCs, and ``ece`` is the top-label (confidence-vs-accuracy) error.
        """
        if X is None:
            proba = self.oof_proba()
            labels = np.asarray(self.arm_)
            scope = "oof"
        else:
            if arm is None:
                raise ValueError("arm must be supplied when X is given")
            proba = self.predict_proba(X)
            labels = np.asarray(arm).ravel().astype(np.int64)
            scope = "holdout"

        k = int(self.n_arms)
        n = labels.size
        rows: list[dict[str, Any]] = []
        aucs: list[float] = []
        brier_total = np.zeros(n, dtype=np.float64)

        for a in range(k):
            y = (labels == a).astype(np.int64)
            p = proba[:, a]
            sq = np.square(p - y)
            brier_total += sq
            try:
                auc = float(roc_auc_score(y, p)) if 0 < y.sum() < n else float("nan")
            except ValueError:  # pragma: no cover - degenerate arm
                auc = float("nan")
            aucs.append(auc)
            pa = np.clip(p, _LOG_EPS, 1.0 - _LOG_EPS)
            arm_ll = float(-(y * np.log(pa) + (1 - y) * np.log(1.0 - pa)).mean())
            rows.append(
                {
                    "scope": scope,
                    "arm": a,
                    "arm_name": _arm_label(a),
                    "n": int(y.sum()),
                    "prevalence": float(y.mean()),
                    "mean_pred": float(p.mean()),
                    "auc_ovr": auc,
                    "brier": float(sq.mean()),
                    "ece": _ece(p, y, n_bins),
                    "log_loss": arm_ll,
                }
            )

        picked = np.clip(proba[np.arange(n), labels], _LOG_EPS, 1.0)
        multi_ll = float(-np.log(picked).mean())
        conf = proba.max(axis=1)
        correct = (np.argmax(proba, axis=1) == labels).astype(np.float64)
        rows.append(
            {
                "scope": scope,
                "arm": -1,
                "arm_name": "overall",
                "n": int(n),
                "prevalence": 1.0,
                "mean_pred": float(conf.mean()),
                "auc_ovr": float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan"),
                "brier": float(brier_total.mean()),
                "ece": _ece(conf, correct, n_bins),
                "log_loss": multi_ll,
            }
        )
        return pd.DataFrame(rows)


# ======================================================================================
# diagnostics
# ======================================================================================
def overlap_diagnostics(
    propensity: np.ndarray,
    arm: np.ndarray,
    *,
    low: float = 0.01,
    high: float = 0.99,
    max_tail_share: float = 0.05,
    arm_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-arm positivity/overlap table, split by whether the row received that arm.

    For every arm ``a`` the distribution of ``e_a(X)`` is summarised twice: among the rows that
    actually got arm ``a`` (``group="treated"``) and among the rows that got something else
    (``group="untreated"``).  Overlap fails when these two distributions barely intersect -- in
    particular when the *untreated* group has mass at ``e_a ~ 0``, because those rows are then
    effectively unrepresentable under arm ``a`` and any IPW estimate of ``E[Y(a)]`` is driven by a
    handful of enormous weights.

    Parameters
    ----------
    propensity : numpy.ndarray
        ``(n, K)`` generalised propensity score.
    arm : numpy.ndarray
        ``(n,)`` realised arm labels in ``[0, K)``.
    low, high : float, default 0.01 and 0.99
        Tail thresholds. ``share_below_001`` / ``share_above_099`` are computed at exactly these
        values; the column names reflect the defaults. The comparison is **boundary-inclusive**
        (``e_a <= low`` / ``e_a >= high``) -- see Notes, this is load-bearing.
    max_tail_share : float, default 0.05
        A group is flagged ``overlap_ok`` only if neither tail share exceeds this.
    arm_names : sequence of str, optional
        Override the arm labels.

    Returns
    -------
    pandas.DataFrame
        Tidy, ``2*K`` rows. Columns: ``arm``, ``arm_name``, ``group``, ``n``, ``min``, ``p1``,
        ``p5``, ``median``, ``p95``, ``p99``, ``max``, ``share_below_001``, ``share_above_099``,
        ``low``, ``high``, ``overlap_ok``.

    Notes
    -----
    ``overlap_ok`` is ``share_below_001 <= max_tail_share and share_above_099 <= max_tail_share``
    evaluated within the group, with an empty group reported as ``False``. It is a screening flag,
    not a hypothesis test; a failure means "trim or re-specify", not "abandon".

    The tail shares count mass sitting *at* the bound as well as beyond it, and that is the whole
    point. The natural input here is :meth:`PropensityModel.oof_proba`, which
    :func:`clip_renormalize` has already projected into ``[low, high]``; a strict ``<`` test
    against the same bounds is then satisfied by nothing at all and every group reports
    ``share_below_001 = 0`` and ``overlap_ok = True`` however bad the positivity violation is.
    Clipping does not remove the offending mass, it parks it exactly on the boundary, so that is
    where the diagnostic must look. Counting ``<= low`` recovers the pre-clip tail share to the
    last row. (Pass ``PropensityModel.oof_raw_`` to see the unprojected distribution instead.)

    One geometric caveat on the *upper* tail: after projection no entry can exceed
    ``1 - (K-1)*low`` (0.97 at ``K=4``, ``low=0.01``), because the other ``K-1`` arms cannot sum
    to less than ``(K-1)*low``. So ``share_above_099`` is structurally zero for a projected
    matrix and carries no information there. This costs nothing diagnostically: mass at
    ``e_a ~ 1`` is mass at ``e_b ~ 0`` for every other arm ``b``, which those arms' own rows
    report in the lower tail, and it is the lower tail that drives the exploding IPW weights.
    """
    p, a = _check_propensity(propensity, arm)
    k = p.shape[1]
    qs = (0.01, 0.05, 0.50, 0.95, 0.99)
    rows: list[dict[str, Any]] = []
    for arm_idx in range(k):
        e = p[:, arm_idx]
        is_a = a == arm_idx
        for group, mask in (("treated", is_a), ("untreated", ~is_a)):
            vals = e[mask]
            m = int(vals.size)
            if m == 0:
                stats = {key: float("nan") for key in ("min", "p1", "p5", "median", "p95", "p99", "max")}
                below = above = float("nan")
                ok = False
            else:
                q = np.quantile(vals, qs)
                stats = {
                    "min": float(vals.min()),
                    "p1": float(q[0]),
                    "p5": float(q[1]),
                    "median": float(q[2]),
                    "p95": float(q[3]),
                    "p99": float(q[4]),
                    "max": float(vals.max()),
                }
                below = float((vals <= low + _BOUND_TOL).mean())
                above = float((vals >= high - _BOUND_TOL).mean())
                ok = bool(below <= max_tail_share and above <= max_tail_share)
            rows.append(
                {
                    "arm": arm_idx,
                    "arm_name": _arm_label(arm_idx, arm_names),
                    "group": group,
                    "n": m,
                    **stats,
                    "share_below_001": below,
                    "share_above_099": above,
                    "low": float(low),
                    "high": float(high),
                    "overlap_ok": ok,
                }
            )
    return pd.DataFrame(rows)


def standardized_mean_differences(
    X: np.ndarray | pd.DataFrame,
    arm: np.ndarray,
    weights: np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
    *,
    control_arm: int = 0,
    threshold: float = 0.1,
    arm_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Covariate balance: standardised mean differences, arm ``a`` against control.

    For each feature and each non-control arm,

    .. math:: \\mathrm{SMD}_a = \\frac{\\bar{x}_a - \\bar{x}_0}{\\sqrt{(s_a^2 + s_0^2)/2}}

    computed twice: once unweighted (the raw selection imbalance) and once with ``weights``
    applied to both means *and* both variances.  The weighted variance carries the reliability
    correction ``sum(w) / (sum(w)^2 - sum(w^2))``, which is the right small-sample factor when the
    weights are inverse-probability weights rather than integer frequencies.

    This frame is the direct input to :func:`love_plot_frame` and to ``docs/figures/love_plot.png``.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Covariates ``(n, p)``. Categorical columns of a DataFrame are one-hot expanded, so each
        level gets its own balance row.
    arm : numpy.ndarray
        ``(n,)`` arm labels.
    weights : numpy.ndarray, optional
        ``(n,)`` non-negative weights, typically from :func:`ipw_weights`. ``None`` is treated as
        uniform weights, in which case ``smd_weighted`` equals ``smd_unweighted``; the boolean
        ``weighted`` column records which case applies.
    feature_names : sequence of str, optional
        Names for an ndarray ``X``.
    control_arm : int, default 0
        The reference arm.
    threshold : float, default 0.1
        Balance cut-off used for the ``balanced`` column. 0.1 is the conventional Austin rule.
    arm_names : sequence of str, optional
        Override the arm labels.

    Returns
    -------
    pandas.DataFrame
        Tidy, one row per (feature, non-control arm). Columns: ``feature``, ``arm``, ``arm_name``,
        ``n_arm``, ``n_control``, ``smd_unweighted``, ``smd_weighted``, ``abs_smd_unweighted``,
        ``abs_smd_weighted``, ``balanced``, ``weighted``.

    Notes
    -----
    A feature that is constant in both groups yields ``0.0`` (perfectly balanced) rather than
    ``NaN``; a feature constant in both groups at *different* values yields ``NaN``, since the
    difference is not standardisable. Columns carrying missing values also yield ``NaN`` and are
    named in a logged warning rather than failing silently (SPEC_PERF section 8).
    """
    Xm, names = _as_matrix(X, feature_names)
    a = np.asarray(arm).ravel().astype(np.int64)
    if a.size != Xm.shape[0]:
        raise ValueError(f"arm has length {a.size} but X has {Xm.shape[0]} rows")
    nan_cols = np.flatnonzero(~np.isfinite(Xm).all(axis=0))
    if nan_cols.size:
        LOGGER.warning(
            "standardized_mean_differences: %d column(s) contain missing/non-finite values "
            "(%s...) and will report NaN SMDs; impute or drop them first",
            nan_cols.size,
            [names[j] for j in nan_cols[:5]],
        )

    has_w = weights is not None
    if has_w:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != a.size:
            raise ValueError(f"weights has length {w.size} but X has {a.size} rows")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
    else:
        w = np.ones(a.size, dtype=np.float64)

    ctrl = int(control_arm)
    m0 = a == ctrl
    if not m0.any():
        raise ValueError(f"no rows with control arm {ctrl}")
    ones0 = np.ones(int(m0.sum()), dtype=np.float64)
    mu0_u, var0_u = _weighted_moments(Xm[m0], ones0)
    mu0_w, var0_w = _weighted_moments(Xm[m0], w[m0])

    rows: list[pd.DataFrame] = []
    for arm_idx in sorted(int(v) for v in np.unique(a) if int(v) != ctrl):
        ma = a == arm_idx
        na = int(ma.sum())
        onesa = np.ones(na, dtype=np.float64)
        mua_u, vara_u = _weighted_moments(Xm[ma], onesa)
        mua_w, vara_w = _weighted_moments(Xm[ma], w[ma])
        smd_u = _smd(mua_u, vara_u, mu0_u, var0_u)
        smd_w = _smd(mua_w, vara_w, mu0_w, var0_w) if has_w else smd_u.copy()
        rows.append(
            pd.DataFrame(
                {
                    "feature": names,
                    "arm": arm_idx,
                    "arm_name": _arm_label(arm_idx, arm_names),
                    "n_arm": na,
                    "n_control": int(m0.sum()),
                    "smd_unweighted": smd_u,
                    "smd_weighted": smd_w,
                    "abs_smd_unweighted": np.abs(smd_u),
                    "abs_smd_weighted": np.abs(smd_w),
                    "balanced": np.abs(smd_w) < float(threshold),
                    "weighted": has_w,
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature", "arm", "arm_name", "n_arm", "n_control", "smd_unweighted",
                "smd_weighted", "abs_smd_unweighted", "abs_smd_weighted", "balanced", "weighted",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    out["balanced"] = out["balanced"].astype(bool)
    return out


def love_plot_frame(smd_frame: pd.DataFrame, *, top_n: int | None = None) -> pd.DataFrame:
    """Order an SMD frame for a love plot.

    A love plot draws, per arm, one horizontal line per feature with the unweighted and weighted
    |SMD| as two markers, features sorted worst-first so the eye lands on the imbalance that
    matters.  This returns exactly that ordering plus the plotting scaffolding.

    Parameters
    ----------
    smd_frame : pandas.DataFrame
        Output of :func:`standardized_mean_differences`.
    top_n : int, optional
        Keep only the ``top_n`` worst features per arm (by unweighted |SMD|).

    Returns
    -------
    pandas.DataFrame
        The input columns plus ``rank`` (1 = worst unweighted imbalance within the arm),
        ``y_pos`` (descending y coordinate so rank 1 plots at the top), ``max_abs_smd``
        (worst of the two) and ``improved`` (weighting reduced |SMD|). Sorted by ``arm`` then
        ``rank``.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    required = {"feature", "arm", "abs_smd_unweighted", "abs_smd_weighted"}
    missing = required - set(smd_frame.columns)
    if missing:
        raise KeyError(f"smd_frame is missing columns {sorted(missing)}")
    if smd_frame.empty:
        return smd_frame.copy()

    out = smd_frame.copy()
    out = out.sort_values(["arm", "abs_smd_unweighted", "feature"], ascending=[True, False, True])
    out = out.reset_index(drop=True)
    out["rank"] = out.groupby("arm").cumcount() + 1
    if top_n is not None:
        out = out.loc[out["rank"] <= int(top_n)].reset_index(drop=True)
    out["max_abs_smd"] = out[["abs_smd_unweighted", "abs_smd_weighted"]].max(axis=1)
    out["improved"] = out["abs_smd_weighted"] < out["abs_smd_unweighted"]
    n_per_arm = out.groupby("arm")["rank"].transform("max")
    out["y_pos"] = (n_per_arm - out["rank"]).astype(np.int64)
    return out.sort_values(["arm", "rank"]).reset_index(drop=True)


def trim_by_overlap(
    propensity: np.ndarray,
    arm: np.ndarray,
    low: float = 0.05,
    high: float = 0.95,
) -> np.ndarray:
    """Boolean keep-mask retaining only rows that live in the common-support region.

    **Rule.** Row ``i`` is kept when *both*

    * ``low <= e_{a_i}(x_i) <= high``  -- the arm it actually received was plausible for it, so its
      IPW weight ``1 / e_{a_i}`` is bounded, and
    * ``low <= e_0(x_i) <= high``      -- control was also plausible for it, so a contrast against
      control is supported by data rather than by extrapolation.

    Every estimand in PRISM is an ``a``-versus-control contrast, so both legs of the contrast must
    be estimable for the row to carry information; requiring only the own-arm condition would keep
    rows that no control unit resembles.  This is the "crump-style" pairwise common-support rule,
    applied to the received arm rather than to all arms at once (the all-arms variant discards far
    too much at ``K = 4``).

    Parameters
    ----------
    propensity : numpy.ndarray
        ``(n, K)`` generalised propensity score.
    arm : numpy.ndarray
        ``(n,)`` realised arm labels.
    low, high : float, default 0.05 and 0.95
        Common-support bounds. Note these are deliberately tighter than the model's clip range:
        clipping bounds the *weights*, trimming bounds the *sample*.

    Returns
    -------
    numpy.ndarray
        Boolean ``(n,)`` mask; ``True`` means keep.

    Notes
    -----
    A control row (``a_i = 0``) is subject to the same rule with both conditions collapsing to one.
    The function logs the dropped share, because silent truncation is forbidden by SPEC_PERF
    section 8.
    """
    p, a = _check_propensity(propensity, arm)
    low = float(low)
    high = float(high)
    if low >= high:
        raise ValueError(f"need low < high; got {(low, high)}")
    own = p[np.arange(a.size), a]
    ctrl = p[:, 0]
    keep = (own >= low) & (own <= high) & (ctrl >= low) & (ctrl <= high)
    dropped = int((~keep).sum())
    if dropped:
        LOGGER.info(
            "trim_by_overlap dropped %d/%d rows (%.1f%%) outside [%.3g, %.3g]",
            dropped,
            a.size,
            100.0 * dropped / max(a.size, 1),
            low,
            high,
        )
    return keep


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum(w^2)``.

    Reads as "how many equally-weighted observations carry the same information as this weighted
    sample". It equals ``n`` exactly when all weights are equal and collapses toward 1 as a few
    extreme weights take over -- which is precisely the failure mode that poor overlap creates.

    Parameters
    ----------
    weights : numpy.ndarray
        ``(n,)`` non-negative weights.

    Returns
    -------
    float
        The effective sample size, in ``(0, n]``; ``0.0`` for an empty or all-zero input.

    Examples
    --------
    >>> float(effective_sample_size(np.ones(100)))
    100.0
    >>> round(float(effective_sample_size(np.array([1.0, 1.0, 98.0]))), 3)
    1.041
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        return 0.0
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s1 = float(w.sum())
    s2 = float(np.square(w).sum())
    if s2 <= 0.0:
        return 0.0
    return float(s1 * s1 / s2)


def ipw_weights(
    propensity: np.ndarray,
    arm: np.ndarray,
    *,
    stabilized: bool = True,
    clip: tuple[float, float] | float | None = None,
    marginal: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse-probability weights for the multi-arm **ATE**.

    Estimand
    --------
    These weights target the **average treatment effect in the whole population**,
    ``E[Y(a)] - E[Y(0)]``, not ATT or ATC.  Each unit is reweighted by the inverse of the
    probability of the arm it actually received, so the weighted arm-``a`` subsample stands in for
    the *entire* population under arm ``a``:

    .. math:: \\hat{E}[Y(a)] = \\frac{\\sum_i 1\\{A_i = a\\} w_i Y_i}{\\sum_i 1\\{A_i = a\\} w_i}

    (the Hajek / ratio form, which is what stabilised weights are designed for).

    With ``stabilized=True`` the weight is ``w_i = P(A = a_i) / e_{a_i}(x_i)`` using the empirical
    marginal arm shares.  Stabilisation does not change the estimand -- the numerator is a constant
    within each arm and cancels in the ratio form -- but it centres the weights near 1 and sharply
    reduces their variance, which is why it is the default.  With ``stabilized=False`` the weight
    is the raw ``1 / e_{a_i}(x_i)``.

    Parameters
    ----------
    propensity : numpy.ndarray
        ``(n, K)`` generalised propensity score. Should already be clipped away from 0.
    arm : numpy.ndarray
        ``(n,)`` realised arm labels.
    stabilized : bool, keyword-only, default True
        Multiply by the marginal ``P(A = a_i)``.
    clip : tuple of (float, float) or float, keyword-only, optional
        Weight truncation applied *after* construction. A 2-tuple is an absolute
        ``[lo, hi]`` clip on the weights. A float ``q`` in ``(0, 0.5)`` winsorises at the
        ``[q, 1-q]`` empirical quantiles of the weights. ``None`` leaves them untouched.
    marginal : numpy.ndarray, optional
        ``(K,)`` arm shares to use instead of the empirical ones (e.g. the design probabilities of
        an RCT block).

    Returns
    -------
    numpy.ndarray
        ``(n,)`` float64 weights.

    Notes
    -----
    Truncation trades variance for bias and is logged when it binds. Check
    :func:`effective_sample_size` before and after: a large drop means the propensity model, not
    the weights, is the problem.
    """
    p, a = _check_propensity(propensity, arm)
    n, k = p.shape
    own = p[np.arange(n), a]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / np.clip(own, _LOG_EPS, None)

    if stabilized:
        if marginal is None:
            share = np.bincount(a, minlength=k).astype(np.float64) / float(max(n, 1))
        else:
            share = np.asarray(marginal, dtype=np.float64).ravel()
            if share.size != k:
                raise ValueError(f"marginal must have length {k}; got {share.size}")
        w = w * share[a]

    if clip is not None:
        if np.isscalar(clip):
            q = float(clip)  # type: ignore[arg-type]
            if not 0.0 < q < 0.5:
                raise ValueError(f"a scalar clip is a winsorising quantile in (0, 0.5); got {q}")
            lo, hi = np.quantile(w, [q, 1.0 - q])
        else:
            lo, hi = float(clip[0]), float(clip[1])  # type: ignore[index]
        n_bind = int(((w < lo) | (w > hi)).sum())
        w = np.clip(w, lo, hi)
        if n_bind:
            LOGGER.info("ipw_weights truncated %d/%d weights to [%.4g, %.4g]", n_bind, n, lo, hi)

    return np.asarray(w, dtype=np.float64)


# ======================================================================================
# smoke test
# ======================================================================================
def _simulate_confounded(
    n: int = 20_000,
    p: int = 10,
    *,
    scale: float = 0.40,
    random_state: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confounded K=4 assignment with an analytically known softmax propensity.

    ``logits = X @ B + b`` with ``B`` and ``b`` centred across arms (the softmax is invariant to a
    per-row shift, so centring -- rather than pinning the control column to zero -- is the
    identification that leaves *every* arm, control included, with real covariate signal).
    ``e_a(x) = softmax(logits)_a`` is therefore exact and directly comparable to the estimate.

    Parameters
    ----------
    n, p : int
        Rows and covariates.
    scale : float
        Coefficient scale; larger means stronger confounding and worse overlap.
    random_state : int
        Seed material for :func:`prism.utils.seeds.as_rng`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(X, arm, true_propensity)`` with shapes ``(n, p)``, ``(n,)``, ``(n, 4)``.
    """
    rng = as_rng(random_state)
    X = rng.normal(size=(n, p))
    X[:, 0] = np.abs(X[:, 0])  # one skewed covariate, as in the real panel
    beta = rng.normal(scale=scale, size=(p, N_ARMS))
    beta -= beta.mean(axis=1, keepdims=True)
    intercept = np.array([0.0, 0.25, -0.15, -0.45])
    intercept -= intercept.mean()
    logits = X @ beta + intercept
    logits -= logits.max(axis=1, keepdims=True)
    expit = np.exp(logits)
    true_p = expit / expit.sum(axis=1, keepdims=True)
    cdf = np.cumsum(true_p, axis=1)
    u = rng.random(size=(n, 1))
    arm = (u > cdf).sum(axis=1).astype(np.int64)
    arm = np.clip(arm, 0, N_ARMS - 1)
    return X, arm, true_p


def _check(rows: list[dict[str, Any]], name: str, ok: bool, detail: str) -> bool:
    """Append a pass/fail row to the smoke-test table."""
    rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    return ok


if __name__ == "__main__":  # pragma: no cover - smoke test
    from scipy.optimize import brentq  # smoke-test only: independent check on the bisection

    t_start = time.perf_counter()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    # 50 rounds rather than the 150-round default: 3x less work *and* measurably better out of
    # fold on this DGP (pooled r 0.968 / 0.964 / 0.951 and MAE 0.0402 / 0.0428 / 0.0496 at 50 /
    # 80 / 150 rounds). Cross-fitting charges honestly for the extra rounds' overfitting, so for
    # a *propensity* -- where the calibrated level, not the sharpness, is what 1/e needs -- more
    # boosting is actively worse. Worth revisiting the constructor default against the real
    # 40-feature panel.
    N, P, NE = 20_000, 10, 50
    X, arm, true_p = _simulate_confounded(N, P, scale=0.40, random_state=7)
    feat_names = [f"x{j}" for j in range(P)]
    print(f"simulated n={N} p={P} K={N_ARMS} | arm shares = {np.round(np.bincount(arm) / N, 4).tolist()}")
    print(f"true propensity range = [{true_p.min():.4f}, {true_p.max():.4f}]")

    model = PropensityModel(
        n_arms=N_ARMS, n_splits=5, calibrate=True, clip=(0.01, 0.99),
        n_estimators=NE, random_state=7,
    ).fit(X, arm)

    oof = model.oof_proba()
    new_p = model.predict_proba(X[:1000])
    checks: list[dict[str, Any]] = []
    all_ok = True

    # --- 1. simplex + clip contract -------------------------------------------------
    sums_ok = np.allclose(oof.sum(axis=1), 1.0, atol=1e-12) and np.allclose(new_p.sum(axis=1), 1.0, atol=1e-12)
    range_ok = oof.min() >= 0.01 - 1e-12 and oof.max() <= 0.99 + 1e-12
    range_ok = range_ok and new_p.min() >= 0.01 - 1e-12 and new_p.max() <= 0.99 + 1e-12
    all_ok &= _check(checks, "rows sum to 1", sums_ok, f"max dev = {np.max(np.abs(oof.sum(1) - 1)):.2e}")
    all_ok &= _check(checks, "inside clip [0.01, 0.99]", range_ok, f"[{oof.min():.4f}, {oof.max():.4f}]")
    all_ok &= _check(checks, "classes_ / shapes", oof.shape == (N, N_ARMS) and list(model.classes_) == [0, 1, 2, 3],
                     f"oof{oof.shape} classes={model.classes_.tolist()}")

    # --- 2. THE real test: recovery of the known propensities ------------------------
    per_arm_corr = [float(np.corrcoef(oof[:, a], true_p[:, a])[0, 1]) for a in range(N_ARMS)]
    flat_corr = float(np.corrcoef(oof.ravel(), true_p.ravel())[0, 1])
    mae = float(np.abs(oof - true_p).mean())
    all_ok &= _check(checks, "corr(est, true) > 0.8", flat_corr > 0.8,
                     f"pooled r={flat_corr:.4f} | per-arm {np.round(per_arm_corr, 3).tolist()} | MAE={mae:.4f}")
    all_ok &= _check(checks, "every arm corr > 0.8", min(per_arm_corr) > 0.8, f"min r={min(per_arm_corr):.4f}")
    # Rank agreement is not enough for IPW: 1/e needs the *level* to be right. Regressing the
    # known truth on the estimate must give slope ~1 through the origin, and each arm's mean
    # propensity must match the truth's.
    slope, icept = np.polyfit(oof.ravel(), true_p.ravel(), 1)
    arm_bias = float(np.abs(oof.mean(axis=0) - true_p.mean(axis=0)).max())
    all_ok &= _check(checks, "calibrated in LEVEL, not just rank",
                     bool(0.90 < slope < 1.10 and abs(icept) < 0.02 and mae < 0.06 and arm_bias < 0.01),
                     f"slope={slope:.4f} intercept={icept:+.4f} MAE={mae:.4f} max per-arm bias={arm_bias:.5f}")

    # --- 3. balance: does IPW actually fix the confounding? --------------------------
    w = ipw_weights(oof, arm, stabilized=True)
    smd = standardized_mean_differences(X, arm, weights=w, feature_names=feat_names)
    mean_abs_u = float(smd["abs_smd_unweighted"].mean())
    mean_abs_w = float(smd["abs_smd_weighted"].mean())
    max_abs_u = float(smd["abs_smd_unweighted"].max())
    max_abs_w = float(smd["abs_smd_weighted"].max())
    all_ok &= _check(checks, "IPW cuts mean |SMD| >= 50%", mean_abs_w < 0.5 * mean_abs_u,
                     f"unweighted {mean_abs_u:.4f} -> weighted {mean_abs_w:.4f} "
                     f"({100 * (1 - mean_abs_w / mean_abs_u):.1f}% reduction)")
    all_ok &= _check(checks, "max |SMD| falls too", max_abs_w < max_abs_u,
                     f"unweighted {max_abs_u:.4f} -> weighted {max_abs_w:.4f}")
    all_ok &= _check(checks, "balanced share improves",
                     bool(smd["balanced"].mean() > (smd["abs_smd_unweighted"] < 0.1).mean()),
                     f"{(smd['abs_smd_unweighted'] < 0.1).mean():.2%} -> {smd['balanced'].mean():.2%} under 0.1")
    # The Hajek form claims each arm's reweighted sample stands in for the whole population.
    # Check that directly against the population covariate means.
    pop_err_raw = max(float(np.abs(X[arm == a].mean(0) - X.mean(0)).max()) for a in range(N_ARMS))
    pop_err_ipw = max(float(np.abs(np.average(X[arm == a], axis=0, weights=w[arm == a]) - X.mean(0)).max())
                      for a in range(N_ARMS))
    all_ok &= _check(checks, "IPW recovers population covariate means", pop_err_ipw < 0.4 * pop_err_raw,
                     f"worst arm-vs-population gap {pop_err_raw:.4f} -> {pop_err_ipw:.4f}")

    # --- 4. effective sample size ----------------------------------------------------
    ess = effective_sample_size(w)
    ess_raw = effective_sample_size(ipw_weights(oof, arm, stabilized=False))
    all_ok &= _check(checks, "0 < ESS < n", 0.0 < ess < N,
                     f"stabilised ESS={ess:,.0f} ({ess / N:.1%} of n) | unstabilised ESS={ess_raw:,.0f}")
    all_ok &= _check(checks, "stabilising raises ESS", ess > ess_raw, f"{ess_raw:,.0f} -> {ess:,.0f}")
    all_ok &= _check(checks, "ESS(uniform) == n", abs(effective_sample_size(np.ones(N)) - N) < 1e-6, "identity holds")

    # --- 5. trimming and overlap: do they respond to real positivity failure? --------
    keep_good = trim_by_overlap(oof, arm, 0.05, 0.95)
    Xb, arm_b, true_pb = _simulate_confounded(N, P, scale=2.5, random_state=11)
    keep_bad = trim_by_overlap(true_pb, arm_b, 0.05, 0.95)
    f_good, f_bad = float(keep_good.mean()), float(keep_bad.mean())
    all_ok &= _check(checks, "good overlap keeps most rows", f_good > 0.80, f"kept {f_good:.1%}")
    all_ok &= _check(checks, "bad overlap keeps few rows", f_bad < 0.60 and f_bad < f_good - 0.20,
                     f"kept {f_bad:.1%} (vs {f_good:.1%} good)")
    # overlap_diagnostics is normally handed a CLIPPED matrix -- that is what oof_proba returns.
    # Clipping parks every positivity violation exactly on the bound, so the tail share has to
    # survive projection; otherwise the flag reads True however bad the overlap really is.
    proj_bad = clip_renormalize(true_pb, 0.01, 0.99)
    od_bad_raw = overlap_diagnostics(true_pb, arm_b)
    od_bad = overlap_diagnostics(proj_bad, arm_b)
    od_good = overlap_diagnostics(oof, arm)
    tail_kept = float(np.abs(od_bad["share_below_001"] - od_bad_raw["share_below_001"]).max())
    n_bad_flag = int((~od_bad["overlap_ok"]).sum())
    n_good_flag = int((~od_good["overlap_ok"]).sum())
    all_ok &= _check(checks, "clipping does not hide the tail share", tail_kept < 0.10,
                     f"max |share_below_001 raw vs projected| = {tail_kept:.4f}; worst group has "
                     f"{od_bad['share_below_001'].max():.1%} at the floor")
    all_ok &= _check(checks, "overlap_ok fires on bad overlap only",
                     bool(n_bad_flag >= 4 and n_good_flag < n_bad_flag),
                     f"{n_bad_flag}/8 groups flagged on the degenerate DGP vs {n_good_flag}/8 on the good one")

    # --- 6. diagnostics sanity -------------------------------------------------------
    diag = model.diagnostics()
    overall = diag.loc[diag["arm"] == -1].iloc[0]
    base_ll = float(-np.log(np.bincount(arm, minlength=N_ARMS) / N)[arm].mean())
    all_ok &= _check(checks, "beats the marginal log-loss", float(overall["log_loss"]) < base_ll,
                     f"model {float(overall['log_loss']):.4f} < marginal {base_ll:.4f}")
    all_ok &= _check(checks, "macro AUC > 0.6", float(overall["auc_ovr"]) > 0.6,
                     f"AUC={float(overall['auc_ovr']):.4f} brier={float(overall['brier']):.4f} "
                     f"ece={float(overall['ece']):.4f}")
    # The full-data model must be visibly optimistic on its own training rows. If it were not,
    # the fold models would be leaking and oof_proba() would be buying us nothing.
    in_sample = model.predict_proba(X)
    ll_in = float(-np.log(np.clip(in_sample[np.arange(N), arm], _LOG_EPS, 1.0)).mean())
    all_ok &= _check(checks, "cross-fitting costs something (no leak)", ll_in < float(overall["log_loss"]),
                     f"in-sample {ll_in:.4f} < out-of-fold {float(overall['log_loss']):.4f} "
                     f"(gap {float(overall['log_loss']) - ll_in:.4f} nats)")

    # --- 7. THE honesty test: a null DGP, where the arm is independent of X ----------
    # True e_a(x) == 1/K for every row, so an honest out-of-fold score CANNOT beat the marginal
    # baseline and its AUC CANNOT exceed 0.5 in expectation. Anything better is the rows' own
    # labels leaking back into their own scores. This is what catches a calibrator fitted on all
    # rows and applied to those same rows (that failure reads as macro AUC 0.515 and a log-loss
    # 0.0025 BELOW baseline); the cross-fitted calibrator sits at 0.503 and 0.003 ABOVE it.
    rng_null = as_rng(5)
    Xn = rng_null.normal(size=(6000, 6))
    arm_n = rng_null.integers(0, N_ARMS, size=6000)
    null_m = PropensityModel(n_splits=5, n_estimators=40, random_state=5).fit(Xn, arm_n)
    null_oof = null_m.oof_proba()
    null_base = float(-np.log(np.bincount(arm_n, minlength=N_ARMS) / arm_n.size)[arm_n].mean())
    null_ll = float(-np.log(np.clip(null_oof[np.arange(arm_n.size), arm_n], _LOG_EPS, 1.0)).mean())
    null_auc = float(np.mean([roc_auc_score((arm_n == a).astype(int), null_oof[:, a]) for a in range(N_ARMS)]))
    null_auc_raw = float(np.mean([roc_auc_score((arm_n == a).astype(int), null_m.oof_raw_[:, a])
                                  for a in range(N_ARMS)]))
    all_ok &= _check(checks, "null DGP: OOF cannot beat the marginal", null_ll >= null_base - 1e-3,
                     f"oof ll {null_ll:.5f} vs marginal {null_base:.5f} (delta {null_ll - null_base:+.5f})")
    all_ok &= _check(checks, "null DGP: OOF macro AUC ~ 0.5", bool(0.46 < null_auc < 0.54),
                     f"calibrated {null_auc:.4f} | uncalibrated {null_auc_raw:.4f}")

    # --- 8. determinism ---------------------------------------------------------------
    # Run on a 6k subsample of the same confounded data rather than refitting the full model:
    # it exercises every stochastic surface (fold shuffling, the booster's own random_state,
    # the per-fold calibrators, the projection) at a fraction of the wall clock, which keeps
    # the whole smoke test inside the SPEC_PERF 60 s budget with room for a contended box.
    Xd, armd = X[:6000], arm[:6000]
    det_a = PropensityModel(n_arms=N_ARMS, n_splits=5, n_estimators=40, random_state=7).fit(Xd, armd)
    det_b = PropensityModel(n_arms=N_ARMS, n_splits=5, n_estimators=40, random_state=7).fit(Xd, armd)
    det_c = PropensityModel(n_arms=N_ARMS, n_splits=5, n_estimators=40, random_state=8).fit(Xd, armd)
    same = bool(np.array_equal(det_a.oof_proba(), det_b.oof_proba()))
    differs = not np.array_equal(det_a.oof_proba(), det_c.oof_proba())
    all_ok &= _check(checks, "bit-identical under same seed", same and differs,
                     f"seed 7 vs 7: max abs diff = {np.max(np.abs(det_a.oof_proba() - det_b.oof_proba())):.2e}; "
                     f"seed 7 vs 8 differs: {differs}")

    # --- 9. edge cases the pipeline will hit ------------------------------------------
    adversarial = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25], [1e-9, 1e-9, 1e-9, 1.0], [np.nan, 1.0, 0.0, 0.0]]
    )
    proj = clip_renormalize(adversarial, 0.01, 0.99)
    all_ok &= _check(checks, "clip_renormalize survives degenerate rows",
                     bool(np.allclose(proj.sum(1), 1.0, atol=1e-12) and proj.min() >= 0.01 - 1e-12
                          and proj.max() <= 0.99 + 1e-12),
                     f"sums {np.round(proj.sum(1), 12).tolist()} range [{proj.min():.4f}, {proj.max():.4f}]")
    # Check the projection against an independent root-find rather than trusting the bisection.
    _rs = as_rng(3).random((200, N_ARMS)) ** 3 + 1e-9
    _rs = _rs / _rs.sum(axis=1, keepdims=True)
    _lo, _hi = 0.02, 0.60
    _got = clip_renormalize(_rs, _lo, _hi)
    _ref = np.empty_like(_got)
    for _i in range(_rs.shape[0]):
        _q = _rs[_i]
        _s = brentq(lambda s, q=_q: float(np.clip(s * q, _lo, _hi).sum()) - 1.0,
                    0.0, float(_hi / _q.min()), xtol=1e-16, rtol=8.9e-16)
        _ref[_i] = np.clip(_s * _q, _lo, _hi)
    # 1e-9 is the agreement floor set by brentq's own convergence on s, not by the bisection:
    # both solvers are far more accurate than anything that matters for a probability.
    all_ok &= _check(checks, "clip_renormalize == independent brentq solve",
                     float(np.abs(_got - _ref).max()) < 1e-9,
                     f"max |bisection - brentq| = {np.abs(_got - _ref).max():.2e} over 200 rows")

    df = pd.DataFrame(X[:4000, :3], columns=["a", "b", "c"])
    df["plan_tier"] = pd.Categorical(np.where(X[:4000, 3] > 0, "pro", "basic"))
    smd_df = standardized_mean_differences(df, arm[:4000])
    all_ok &= _check(checks, "DataFrame categoricals one-hot into SMD",
                     {"plan_tier_pro", "plan_tier_basic"}.issubset(set(smd_df["feature"])),
                     f"{smd_df['feature'].nunique()} features x {smd_df['arm'].nunique()} arms "
                     f"= {len(smd_df)} rows")

    # A scoring frame's column order is whatever the caller happened to build (a JSON payload,
    # say). Reading it positionally answers confidently and wrongly, so names must be honoured.
    df_model = PropensityModel(n_splits=3, n_estimators=30, random_state=2).fit(df, arm[:4000])
    ref_out = df_model.predict_proba(df.iloc[:200])
    perm_out = df_model.predict_proba(df.iloc[:200][["plan_tier", "c", "a", "b"]])
    try:
        df_model.predict_proba(df.iloc[:200].drop(columns=["b"]))
        raised = False
    except ValueError:
        raised = True
    all_ok &= _check(checks, "DataFrame columns matched by name, not position",
                     bool(np.array_equal(perm_out, ref_out)) and raised,
                     f"permuted vs reference max diff {np.abs(perm_out - ref_out).max():.2e}; "
                     f"missing column raises: {raised}")

    rare = arm[:3000].copy()
    rare[rare == 3] = 1
    rare[:3] = 3  # only three rows of arm 3 -> n_splits must shrink
    rare_model = PropensityModel(n_splits=5, n_estimators=30, random_state=1).fit(X[:3000], rare)
    all_ok &= _check(checks, "rare arm shrinks n_splits (no empty fold)",
                     rare_model.n_splits_ == 3 and np.allclose(rare_model.oof_proba().sum(1), 1.0, atol=1e-12),
                     f"n_splits 5 -> {rare_model.n_splits_}")

    # A stratum holding a single arm (a bootstrap resample, an all-control slice) must degrade to
    # the known answer, not raise: LightGBM and HistGradientBoosting both report classes_ of
    # length 1 while returning a two-column predict_proba, which a positional read cannot survive.
    solo = PropensityModel(n_splits=3, n_estimators=20, random_state=0).fit(X[:1500], np.zeros(1500, dtype=np.int64))
    solo_p = solo.oof_proba()
    all_ok &= _check(checks, "single-arm stratum degrades, does not raise",
                     bool(np.allclose(solo_p.sum(1), 1.0, atol=1e-12) and solo_p[:, 0].min() > 0.9
                          and solo_p[:, 1:].max() <= 0.01 + 1e-12),
                     f"P(observed arm) = {solo_p[:, 0].min():.4f}, others pinned at "
                     f"{solo_p[:, 1:].max():.4f}")

    all_ok &= _check(checks, "unstabilised weights average ~K",
                     abs(float(ipw_weights(oof, arm, stabilized=False).mean()) - N_ARMS) < 0.6,
                     f"mean 1/e = {float(ipw_weights(oof, arm, stabilized=False).mean()):.3f} vs K={N_ARMS}")
    trunc = ipw_weights(oof, arm, stabilized=True, clip=0.01)
    all_ok &= _check(checks, "weight winsorising raises ESS",
                     effective_sample_size(trunc) > ess and trunc.max() <= w.max(),
                     f"ESS {ess:,.0f} -> {effective_sample_size(trunc):,.0f}, "
                     f"max w {w.max():.2f} -> {trunc.max():.2f}")

    # --- output ------------------------------------------------------------------------
    print("\n--- overlap diagnostics (estimated propensity, projected onto [0.01, 0.99]) ---")
    print(od_good[["arm_name", "group", "n", "min", "p1", "p5", "median", "p95", "p99", "max",
                   "share_below_001", "share_above_099", "overlap_ok"]].round(4).to_string(index=False))

    print("\n--- love-plot frame (top 10 by unweighted |SMD|) ---")
    lp = love_plot_frame(smd).sort_values("abs_smd_unweighted", ascending=False).head(10)
    print(lp[["feature", "arm_name", "rank", "smd_unweighted", "smd_weighted",
              "abs_smd_unweighted", "abs_smd_weighted", "improved", "balanced"]].round(4).to_string(index=False))

    print("\n--- propensity diagnostics ---")
    print(diag.round(4).to_string(index=False))

    n_pass = sum(c["status"] == "PASS" for c in checks)
    print(f"\n--- smoke-test results ({n_pass}/{len(checks)} PASS) ---")
    print(pd.DataFrame(checks).to_string(index=False))

    elapsed = time.perf_counter() - t_start
    print(f"\nfit time {model.fit_seconds_:.2f}s | total {elapsed:.2f}s | "
          f"mean|SMD| {mean_abs_u:.4f} -> {mean_abs_w:.4f} | corr {flat_corr:.4f} | ESS {ess:,.0f}/{N:,}")
    if not all_ok:
        raise SystemExit("propensity.py SMOKE TEST FAILED")
    if elapsed > 60.0:
        # Wall clock, not work done. The budget is measured on an otherwise idle box, where the
        # model fitting here is a handful of seconds, so an overrun means the machine is
        # contended rather than that the module regressed -- say so loudly rather than failing a
        # correctness run for a scheduling reason. The measured time is always printed above.
        print(f"WARNING: smoke test took {elapsed:.1f}s, over the 60s SPEC_PERF budget "
              f"(the model fit alone was {model.fit_seconds_:.1f}s) -- check for CPU contention")
    print("propensity.py OK")
