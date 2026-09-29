"""Discrete-time hazard, Cox proportional-hazards and random-survival-forest models,
plus the censoring-aware metrics used to score them.

This module is the time-to-event backbone of PRISM. Retention is not a binary event that
happens at a convenient moment: a customer churns *at some time*, and most customers in any
training window have not churned yet. Everything here is therefore built around right
censoring, and every metric re-weights for it explicitly rather than quietly dropping the
censored rows.

Time convention (used identically by all three estimators)
----------------------------------------------------------
Time is measured in **whole periods** (months in PRISM). Period ``k`` covers the interval
``(k - 1, k]``. A model's survival curve is

``S(k) = P(alive at the end of period k) = prod_{j=1..k} (1 - h(j))``

so ``predict_survival`` returns the ``horizon``-length vector ``[S(1), ..., S(H)]`` and

``RMST(H) = sum_{k=1..H} S(k)``

i.e. the **right Riemann sum** of the survival curve on the integer grid. This is the single
convention used by :class:`DiscreteTimeHazardModel`, :class:`CoxPHModel` and
:class:`RandomSurvivalForestLite`, so their RMST predictions are directly comparable (which
matters, because ``prism.causal.survival_uplift`` differences them across arms). A fractional
horizon ``H = floor(H) + f`` adds the partial term ``f * S(H)``.

Note that the right Riemann sum is a slight *under*-estimate of the continuous integral
``int_0^H S(t) dt`` (by roughly ``(1 - S(H)) / 2``). That bias is identical across arms and so
cancels in every treatment contrast, which is the quantity PRISM actually spends money on.

Implementation notes (see ``SPEC_PERF.md`` sections 3-5)
--------------------------------------------------------
* The person-period expansion is built with ``numpy.repeat`` / ``cumsum`` arithmetic
  (:func:`person_period_expansion`); there is no Python loop over customers anywhere.
* :class:`CoxPHModel` maximises the **Efron** partial likelihood with an analytic gradient
  under ``scipy.optimize.minimize(method="L-BFGS-B")``. Risk-set sums are reversed cumulative
  sums; tie blocks are reduced with ``np.add.reduceat``; the Efron inner sum over tied ranks
  is collapsed algebraically so no ``(n_times, max_ties, n_features)`` tensor is ever formed.
* :class:`RandomSurvivalForestLite` uses **strategy (A)**, the pre-binned vectorised log-rank
  search, because it evaluates all ~31 thresholds of a feature for the price of one cumulative
  sum -- strictly more information per unit of work than one random threshold (strategy B),
  while costing the same two ``np.bincount`` calls per node.
"""

from __future__ import annotations

import math
import time as _time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator

from prism.utils.logging import get_logger
from prism.utils.optional import HAS_TORCH
from prism.utils.seeds import as_rng, spawn_rngs

__all__ = [
    "PersonPeriodExpansion",
    "person_period_expansion",
    "kaplan_meier",
    "censoring_survival",
    "step_eval",
    "DiscreteTimeHazardModel",
    "CoxPHModel",
    "RandomSurvivalForestLite",
    "concordance_index",
    "time_dependent_auc",
    "brier_score",
    "integrated_brier_score",
    "calibration_curve_survival",
]

LOGGER = get_logger("models.survival")

#: Probabilities are clipped into this open interval before any ``log`` or ``cumprod``.
_P_LO = 1e-6
_P_HI = 1.0 - 1e-6
#: Survival curves are clipped here so they are *strictly* inside ``(0, 1)`` as the contract
#: requires. Clipping is elementwise with constant bounds, so monotonicity is preserved.
_S_LO = 1e-12
_S_HI = 1.0 - 1e-12

_TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz


# ======================================================================================
# Small shared helpers
# ======================================================================================
def _as_2d(X: Any, name: str = "X") -> np.ndarray:
    """Coerce ``X`` to a C-contiguous float64 2-D array."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def _as_1d(v: Any, name: str, n: int | None = None) -> np.ndarray:
    """Coerce ``v`` to a 1-D float64 array, optionally checking its length."""
    arr = np.asarray(v, dtype=np.float64).ravel()
    if n is not None and arr.size != n:
        raise ValueError(f"{name} has length {arr.size}, expected {n}")
    return arr


def _check_survival_inputs(
    event_time: Any, event_observed: Any, n: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a ``(event_time, event_observed)`` pair.

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Observed follow-up time, strictly non-negative and finite.
    event_observed : array-like of shape (n,)
        ``1`` if the event was observed, ``0`` if right-censored.
    n : int, optional
        Expected length.

    Returns
    -------
    event_time : ndarray of float64, shape (n,)
    event_observed : ndarray of int64, shape (n,)

    Raises
    ------
    ValueError
        If lengths disagree, times are non-finite/negative, or the indicator is not binary.
    """
    t = _as_1d(event_time, "event_time", n)
    d = np.asarray(event_observed).ravel()
    if d.size != t.size:
        raise ValueError(f"event_observed has length {d.size}, expected {t.size}")
    if not np.all(np.isfinite(t)):
        raise ValueError("event_time contains non-finite values")
    if np.any(t < 0):
        raise ValueError("event_time contains negative values")
    d = np.asarray(np.rint(np.asarray(d, dtype=np.float64)), dtype=np.int64)
    if not np.all((d == 0) | (d == 1)):
        raise ValueError("event_observed must be binary (0/1)")
    return t, d


def _resolve_times(times: Any, name: str = "times") -> np.ndarray:
    """Coerce a time grid to a finite, non-decreasing float64 1-D array."""
    ts = _as_1d(times, name)
    if ts.size == 0:
        raise ValueError(f"{name} must contain at least one time point")
    if not np.all(np.isfinite(ts)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(np.diff(ts) < 0):
        raise ValueError(f"{name} must be non-decreasing (otherwise survival is not monotone)")
    return ts


def _clip_survival(S: np.ndarray) -> np.ndarray:
    """Clip a survival array into the open interval ``(0, 1)`` without breaking monotonicity."""
    return np.clip(S, _S_LO, _S_HI)


def step_eval(
    grid: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    *,
    left_limit: bool = False,
    before: float = 1.0,
) -> np.ndarray:
    """Evaluate a right-continuous step function on arbitrary query points.

    The function equals ``values[j]`` on ``[grid[j], grid[j + 1])`` and ``values[-1]``
    beyond ``grid[-1]``.

    Parameters
    ----------
    grid : ndarray of shape (m,)
        Non-decreasing jump times.
    values : ndarray of shape (m,)
        Value of the function *at and after* each jump time.
    query : ndarray
        Points at which to evaluate. Any shape; the result has the same shape.
    left_limit : bool, default False
        If True evaluate the **left limit** ``f(t-)`` instead of ``f(t)``. This is what
        IPCW needs for a subject whose event happens exactly at a censoring jump.
    before : float, default 1.0
        Value returned for query points strictly before ``grid[0]``. The survival-function
        default is ``1.0``; pass ``0.0`` for a cumulative hazard.

    Returns
    -------
    ndarray
        Same shape as ``query``.

    Notes
    -----
    A single ``np.searchsorted``, so this is ``O((m + q) log m)`` with no Python loop.
    """
    grid = np.asarray(grid, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    side = "left" if left_limit else "right"
    idx = np.searchsorted(grid, q, side=side) - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, values.size - 1)], float(before))


# ======================================================================================
# Person-period expansion (SPEC_PERF section 4)
# ======================================================================================
@dataclass(frozen=True)
class PersonPeriodExpansion:
    """Result of :func:`person_period_expansion`.

    Attributes
    ----------
    row_index : ndarray of int64, shape (m,)
        Index into the original ``n`` subjects for each person-period row.
    period : ndarray of int64, shape (m,)
        Zero-based period index within the subject; ``period = k - 1`` for calendar period
        ``k``, so ``period`` ranges over ``0 .. horizon - 1``.
    event : ndarray of int64, shape (m,)
        ``1`` on the single row where an observed event occurred, ``0`` everywhere else.
    n_periods : ndarray of int64, shape (n,)
        Number of rows each subject contributes (``1 .. horizon``).
    """

    row_index: np.ndarray
    period: np.ndarray
    event: np.ndarray
    n_periods: np.ndarray

    @property
    def n_rows(self) -> int:
        """Total number of person-period rows."""
        return int(self.row_index.size)


def person_period_expansion(
    event_time: Any, event_observed: Any, horizon: int
) -> PersonPeriodExpansion:
    """Expand ``n`` subjects into person-period rows, fully vectorised.

    A subject observed for ``t`` months contributes ``min(ceil(t), horizon)`` rows, one per
    period they were alive *at the start of*. The binary target is ``1`` only on the final
    row of a subject whose event was observed **inside** the horizon; a subject who survives
    past ``horizon`` is administratively censored at ``horizon`` and contributes all-zero rows.

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Observed follow-up time in periods (may be fractional).
    event_observed : array-like of shape (n,)
        ``1`` if the event was observed, ``0`` if right-censored.
    horizon : int
        Maximum number of periods any subject may contribute.

    Returns
    -------
    PersonPeriodExpansion
        Index / period / target arrays plus the per-subject row counts.

    Notes
    -----
    Built exactly as ``SPEC_PERF.md`` section 4 prescribes::

        n_rows_i = minimum(ceil(event_time), horizon)
        idx      = np.repeat(np.arange(n), n_rows_i)
        period   = np.arange(total) - np.repeat(cumsum(n_rows_i) - n_rows_i, n_rows_i)

    There is no Python-level loop over subjects, so a 100k x 12 expansion costs a handful of
    milliseconds. ``ceil`` is computed as ``ceil(t - 1e-9)`` so that an exactly-integral
    ``t = 3.0`` yields three periods rather than four.

    Examples
    --------
    >>> exp = person_period_expansion([2.0, 5.0], [1, 0], horizon=3)
    >>> exp.n_periods.tolist()
    [2, 3]
    >>> exp.period.tolist()
    [0, 1, 0, 1, 2]
    >>> exp.event.tolist()
    [0, 1, 0, 0, 0]
    """
    t, d = _check_survival_inputs(event_time, event_observed)
    H = int(horizon)
    if H < 1:
        raise ValueError(f"horizon must be >= 1, got {H}")
    n = t.size
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return PersonPeriodExpansion(z, z.copy(), z.copy(), z.copy())

    ceil_t = np.ceil(t - 1e-9)
    n_rows_i = np.clip(ceil_t, 1.0, float(H)).astype(np.int64)

    row_index = np.repeat(np.arange(n, dtype=np.int64), n_rows_i)
    ends = np.cumsum(n_rows_i)
    starts = ends - n_rows_i
    total = int(ends[-1])
    period = np.arange(total, dtype=np.int64) - np.repeat(starts, n_rows_i)

    event = np.zeros(total, dtype=np.int64)
    churned_inside = (d == 1) & (ceil_t <= H)
    if np.any(churned_inside):
        event[ends[churned_inside] - 1] = 1
    return PersonPeriodExpansion(row_index, period, event, n_rows_i)


# ======================================================================================
# Kaplan-Meier / censoring distribution
# ======================================================================================
def kaplan_meier(
    event_time: Any, event_observed: Any, *, times: Any | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier product-limit estimator of the survival function.

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Observed follow-up times.
    event_observed : array-like of shape (n,)
        ``1`` if the event was observed, ``0`` if right-censored.
    times : array-like, optional
        If given, the estimator is evaluated on this grid (right-continuous step lookup)
        instead of at its own jump times.

    Returns
    -------
    grid : ndarray of shape (m,)
        Distinct event times (or ``times`` if supplied).
    surv : ndarray of shape (m,)
        ``S(t)`` at each grid point. ``S`` is non-increasing and starts at
        ``1 - d_1 / n_1 <= 1``.

    Notes
    -----
    Fully vectorised: one ``np.unique`` plus a reversed ``cumsum`` for the risk sets.
    """
    t, d = _check_survival_inputs(event_time, event_observed)
    if t.size == 0:
        empty = np.zeros(0)
        return empty, empty.copy()

    order = np.argsort(t, kind="stable")
    ts, ds = t[order], d[order]
    uniq, first, counts = np.unique(ts, return_index=True, return_counts=True)
    # events per distinct time, and the risk set at that time
    n_events = np.add.reduceat(ds, first).astype(np.float64)
    at_risk = (ts.size - first).astype(np.float64)

    keep = n_events > 0
    grid = uniq[keep]
    n_ev = n_events[keep]
    n_at = at_risk[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(n_at > 0, 1.0 - n_ev / np.maximum(n_at, 1.0), 1.0)
    surv = np.cumprod(np.clip(factors, 0.0, 1.0))

    if grid.size == 0:  # no events at all -> S == 1 everywhere
        grid = uniq[:1] if uniq.size else np.zeros(1)
        surv = np.ones(grid.size)

    if times is not None:
        q = _resolve_times(times)
        return q, step_eval(grid, surv, q)
    return grid, surv


def censoring_survival(
    event_time: Any, event_observed: Any, *, min_prob: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimate ``G(t)`` of the **censoring** distribution (reverse KM).

    ``G(t) = P(C > t)`` is the denominator of every inverse-probability-of-censoring weight
    in this module (Graf's Brier score, Uno's time-dependent AUC).

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Observed follow-up times.
    event_observed : array-like of shape (n,)
        ``1`` if the event was observed; censoring is the complement.
    min_prob : float, default 1e-3
        Floor applied to ``G`` so IPCW weights cannot explode past ``1 / min_prob``. A
        warning is logged when the floor binds on more than 0.5% of the grid.

    Returns
    -------
    grid : ndarray of shape (m,)
        Distinct censoring times.
    G : ndarray of shape (m,)
        ``G(t)`` at each grid point, floored at ``min_prob``.
    """
    t, d = _check_survival_inputs(event_time, event_observed)
    grid, g = kaplan_meier(t, 1 - d)
    # A single floored point is the ordinary right tail of G and is not worth a warning;
    # a floored *block* means the IPCW weights are being capped where it matters.
    n_floored = int(np.sum(g < min_prob))
    if n_floored > max(2, 0.05 * max(g.size, 1)):
        LOGGER.warning(
            "censoring_survival: floored G(t) at %.1e on %d/%d grid points "
            "(IPCW weights capped at %.0f)",
            min_prob,
            n_floored,
            g.size,
            1.0 / min_prob,
        )
    return grid, np.maximum(g, min_prob)


# ======================================================================================
# Torch hazard network (lazily defined so importing this module never needs torch)
# ======================================================================================
_HAZARD_NET_CLS: Any = None


def _hazard_net_class() -> Any:
    """Build (once) and return the torch ``nn.Module`` subclass used by the torch backend.

    Returns
    -------
    type
        A ``torch.nn.Module`` subclass exposing ``.body`` (the MLP producing the scalar
        subject term ``f(x)``) and ``.alpha`` (the free per-period baseline log-odds).
    """
    global _HAZARD_NET_CLS
    if _HAZARD_NET_CLS is not None:
        return _HAZARD_NET_CLS
    import torch
    from torch import nn

    class _HazardNet(nn.Module):
        """``logit h(k | x) = f(x) + alpha_k`` with an MLP ``f`` and free ``alpha``."""

        def __init__(self, n_in: int, hidden: Sequence[int], horizon: int, alpha0: np.ndarray):
            super().__init__()
            layers: list[Any] = []
            prev = int(n_in)
            for h in hidden:
                layers += [nn.Linear(prev, int(h)), nn.ReLU()]
                prev = int(h)
            last = nn.Linear(prev, 1)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            layers.append(last)
            self.body = nn.Sequential(*layers)
            self.alpha = nn.Parameter(torch.as_tensor(np.asarray(alpha0, dtype=np.float32)))

        def forward(self, x):  # type: ignore[no-untyped-def]
            """Return the ``(batch, horizon)`` matrix of hazard logits."""
            return self.body(x).squeeze(-1)[:, None] + self.alpha[None, :]

    _HAZARD_NET_CLS = _HazardNet
    return _HAZARD_NET_CLS


def _build_hazard_net(n_in: int, hidden: Sequence[int], horizon: int, alpha0: np.ndarray) -> Any:
    """Instantiate the torch hazard network. Torch is imported only on this path."""
    return _hazard_net_class()(n_in, hidden, horizon, alpha0)


# ======================================================================================
# 1. Discrete-time hazard model
# ======================================================================================
class DiscreteTimeHazardModel(BaseEstimator):
    """Pooled logistic discrete-time hazard ``h(k | x) = sigmoid(f(x) + alpha_k)``.

    The model factorises the hazard into a *subject* term ``f(x)`` and a free *baseline*
    term ``alpha_k`` per period. With a torch backend ``f`` is an MLP; with the sklearn
    backend ``f`` is linear and the whole thing is one ``LogisticRegression`` fitted on the
    person-period expansion (period entering as a near-unpenalised one-hot block).

    Because the two terms are additive, the fitted model needs ``f(x)`` **once per subject**
    at prediction time -- the ``(n, horizon)`` hazard matrix is then one outer sum. Nothing
    is re-expanded to predict.

    Parameters
    ----------
    horizon : int, default 12
        Number of discrete periods modelled. ``predict_hazard``/``predict_survival`` return
        ``(n, horizon)`` matrices covering periods ``1 .. horizon``.
    hidden : tuple of int, default (64, 32)
        Hidden layer widths of the torch MLP. Ignored by the sklearn backend.
    epochs : int, default 40
        Passes over the data (torch backend only).
    lr : float, default 1e-3
        Adam learning rate (torch backend only).
    batch_size : int, default 1024
        Subjects per optimisation step (torch backend only).
    l2 : float, default 1e-4
        Ridge strength. For the torch backend this is AdamW's ``weight_decay`` on the mean
        loss. For the sklearn backend it is the ridge coefficient on the **summed** log-loss,
        i.e. ``C = 1 / l2``; the period one-hot columns are rescaled so their penalty is
        2500x weaker and the baseline hazard is never shrunk toward ``0.5``.
    backend : {"auto", "torch", "sklearn"}, default "auto"
        ``"auto"`` picks torch when it is importable, else sklearn.
    random_state : int or numpy.random.Generator or None, default None
        Seed for batch shuffling and torch parameter initialisation.
    max_expansion_rows : int, default 3_000_000
        Cap on the person-period matrix materialised for the sklearn backend. If the
        expansion would be larger, subjects are subsampled and the drop is **logged**.
    min_steps, max_steps : int, default 300 / 12_000
        Guard rails on the torch optimiser budget. If ``epochs`` would give fewer than
        ``min_steps`` gradient steps the epoch count is raised; if it would give more than
        ``max_steps`` the batch size is raised (and ``lr`` scaled by ``sqrt``). Both
        adjustments are logged.
    torch_threads : int, optional
        Temporarily cap ``torch.set_num_threads`` for the duration of the fit and restore
        the previous value afterwards. Useful when this model is one of several jobs on a
        shared box: on a tiny MLP, oversubscribed BLAS threads cost more than they buy.
    verbose : bool, default False
        Log the training loss every ten epochs (torch backend).

    Attributes
    ----------
    backend_ : str
        The backend actually used, ``"torch"`` or ``"sklearn"``.
    alpha_ : ndarray of shape (horizon,)
        Fitted baseline log-odds per period.
    baseline_hazard_ : ndarray of shape (horizon,)
        ``sigmoid(alpha_)``, the hazard of a subject with ``f(x) = 0``.
    n_features_in_ : int
        Number of columns seen during ``fit``.
    fit_seconds_ : float
        Wall-clock seconds spent in the last ``fit``.

    Notes
    -----
    ``predict_survival`` is guaranteed monotone non-increasing along axis 1 and strictly
    inside ``(0, 1)``: hazards are clipped to ``[1e-6, 1 - 1e-6]`` before the cumulative
    product, so every factor lies strictly in ``(0, 1)``.

    ``predict_rmst`` is exactly ``predict_survival(X)[:, :horizon].sum(axis=1)`` -- the
    module-level right-Riemann convention ``RMST(H) = sum_{k=1..H} S(k)``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 3))
    >>> t = rng.exponential(6.0, size=300)
    >>> e = (t < 8).astype(int)
    >>> m = DiscreteTimeHazardModel(horizon=6, backend="sklearn").fit(X, t, e)
    >>> S = m.predict_survival(X)
    >>> bool(np.all(np.diff(S, axis=1) <= 1e-12)), S.shape
    (True, (300, 6))
    """

    def __init__(
        self,
        horizon: int = 12,
        hidden: tuple[int, ...] = (64, 32),
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int = 1024,
        l2: float = 1e-4,
        backend: str = "auto",
        random_state: int | np.random.Generator | None = None,
        *,
        max_expansion_rows: int = 3_000_000,
        min_steps: int = 300,
        max_steps: int = 12_000,
        torch_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.horizon = horizon
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.l2 = l2
        self.backend = backend
        self.random_state = random_state
        self.max_expansion_rows = max_expansion_rows
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.torch_threads = torch_threads
        self.verbose = verbose

    # -- internals ---------------------------------------------------------------------
    def _resolve_backend(self) -> str:
        """Resolve ``backend="auto"`` to a concrete backend name."""
        if self.backend not in {"auto", "torch", "sklearn"}:
            raise ValueError(f"backend must be one of auto/torch/sklearn, got {self.backend!r}")
        if self.backend == "torch":
            if not HAS_TORCH:
                raise ImportError("backend='torch' requested but torch is not installed")
            return "torch"
        if self.backend == "sklearn":
            return "sklearn"
        return "torch" if HAS_TORCH else "sklearn"

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Median-impute non-finite entries, then apply the fitted mean/scale."""
        Xf = np.where(np.isfinite(X), X, np.nan)
        Xf = np.where(np.isnan(Xf), self._x_median_[None, :], Xf)
        return (Xf - self._x_mean_[None, :]) / self._x_scale_[None, :]

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """Learn the median/mean/scale used by :meth:`_standardize` and return ``X`` scaled."""
        Xf = np.where(np.isfinite(X), X, np.nan)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(Xf, axis=0)
        self._x_median_ = np.where(np.isfinite(med), med, 0.0)
        Xf = np.where(np.isnan(Xf), self._x_median_[None, :], Xf)
        self._x_mean_ = Xf.mean(axis=0)
        scale = Xf.std(axis=0)
        self._x_scale_ = np.where(scale > 1e-12, scale, 1.0)
        return (Xf - self._x_mean_[None, :]) / self._x_scale_[None, :]

    def _linear_predictor(self, X: Any, chunk_size: int = 20_000) -> np.ndarray:
        """Return the subject term ``f(x)`` for every row of ``X``."""
        Xa = _as_2d(X)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {Xa.shape[1]} features, model was fitted with {self.n_features_in_}"
            )
        Xs = self._standardize(Xa)
        if self.backend_ == "sklearn":
            return Xs @ self._beta_
        import torch

        self._net_.eval()
        out = np.empty(Xs.shape[0], dtype=np.float64)
        with torch.no_grad():
            for s in range(0, Xs.shape[0], chunk_size):
                block = torch.as_tensor(Xs[s : s + chunk_size], dtype=torch.float32)
                out[s : s + chunk_size] = self._net_.body(block).squeeze(-1).numpy().astype(np.float64)
        return out

    # -- fit ---------------------------------------------------------------------------
    # -- pickling ----------------------------------------------------------------------
    # The torch backend's nn.Module subclass is created inside _hazard_net_class() so that
    # torch is never imported at module import time. A locally-defined class cannot be
    # pickled, so a fitted torch-backed model would otherwise be unserialisable -- which
    # breaks the serving bundle, and breaks it *silently* halfway through writing the file.
    # Round-trip the weights as plain numpy instead and rebuild the module on load.

    def __getstate__(self) -> dict:
        """Return picklable state, replacing the torch module with its weights."""
        state = self.__dict__.copy()
        net = state.pop("_net_", None)
        if net is not None:
            state["_net_weights_"] = {
                k: v.detach().cpu().numpy() for k, v in net.state_dict().items()
            }
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state, rebuilding the torch module from its weights when present."""
        weights = state.pop("_net_weights_", None)
        self.__dict__.update(state)
        if not weights:
            return
        try:
            import torch

            # The first Linear layer's weight is (out, n_in), which recovers the input width
            # without needing it to have been stored separately.
            first = next(v for k, v in weights.items() if k.endswith("body.0.weight"))
            n_in = int(np.asarray(first).shape[1])
            net = _build_hazard_net(n_in, tuple(self.hidden), int(self.horizon), self._alpha_init_)
            net.load_state_dict({k: torch.as_tensor(np.asarray(v)) for k, v in weights.items()})
            net.eval()
            self._net_ = net
        except Exception as exc:  # pragma: no cover - torch missing on the loading machine
            LOGGER.warning(
                "could not rebuild the torch hazard network on unpickle (%s); "
                "refit or install torch to use this model",
                exc,
            )

    def fit(
        self,
        X: Any,
        event_time: Any,
        event_observed: Any,
        sample_weight: Any | None = None,
    ) -> DiscreteTimeHazardModel:
        """Fit the pooled logistic hazard on the person-period expansion.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates, point-in-time safe. Non-finite entries are median-imputed.
        event_time : array-like of shape (n,)
            Observed follow-up time in periods.
        event_observed : array-like of shape (n,)
            ``1`` if the event was observed, ``0`` if right-censored.
        sample_weight : array-like of shape (n,), optional
            Per-subject weights, broadcast to every person-period row of that subject.

        Returns
        -------
        DiscreteTimeHazardModel
            ``self``, fitted.
        """
        t0 = _time.perf_counter()
        Xa = _as_2d(X)
        n = Xa.shape[0]
        t, d = _check_survival_inputs(event_time, event_observed, n)
        H = int(self.horizon)
        if H < 1:
            raise ValueError(f"horizon must be >= 1, got {H}")
        w = np.ones(n) if sample_weight is None else _as_1d(sample_weight, "sample_weight", n)
        if np.any(w < 0):
            raise ValueError("sample_weight must be non-negative")

        self.backend_ = self._resolve_backend()
        self.n_features_in_ = Xa.shape[1]
        rng = as_rng(self.random_state)
        Xs = self._fit_scaler(Xa)

        exp = person_period_expansion(t, d, H)
        # Baseline initialisation: empirical hazard per period.
        per_period_n = np.bincount(exp.period, weights=w[exp.row_index], minlength=H)
        per_period_e = np.bincount(exp.period, weights=w[exp.row_index] * exp.event, minlength=H)
        with np.errstate(divide="ignore", invalid="ignore"):
            h0 = np.where(per_period_n > 0, per_period_e / np.maximum(per_period_n, 1e-12), 0.05)
        h0 = np.clip(h0, 1e-3, 1 - 1e-3)
        self._alpha_init_ = np.log(h0 / (1.0 - h0))

        if self.backend_ == "sklearn":
            self._fit_sklearn(Xs, exp, w, H, rng)
        else:
            self._fit_torch(Xs, exp, w, H, rng)

        self.alpha_ = np.asarray(self.alpha_, dtype=np.float64)
        self.baseline_hazard_ = expit(self.alpha_)
        self.n_subjects_ = n
        self.fit_seconds_ = _time.perf_counter() - t0
        return self

    def _fit_sklearn(
        self,
        Xs: np.ndarray,
        exp: PersonPeriodExpansion,
        w: np.ndarray,
        H: int,
        rng: np.random.Generator,
    ) -> None:
        """Fit a linear ``f(x)`` with sklearn LogisticRegression on the expansion."""
        from sklearn.linear_model import LogisticRegression

        row_index, period, event = exp.row_index, exp.period, exp.event
        m = row_index.size
        if m > int(self.max_expansion_rows):
            n = Xs.shape[0]
            keep_n = max(1000, int(n * self.max_expansion_rows / max(m, 1)))
            keep = rng.choice(n, size=min(keep_n, n), replace=False)
            keep_mask = np.zeros(n, dtype=bool)
            keep_mask[keep] = True
            sel = keep_mask[row_index]
            LOGGER.warning(
                "DiscreteTimeHazardModel: expansion would be %d rows (> %d); "
                "subsampled to %d subjects / %d rows",
                m,
                self.max_expansion_rows,
                int(keep_mask.sum()),
                int(sel.sum()),
            )
            row_index, period, event = row_index[sel], period[sel], event[sel]
            m = row_index.size

        # Design = [standardised X, scaled period one-hot]. The one-hot block is multiplied
        # by PERIOD_SCALE so the ridge penalty on its coefficients is PERIOD_SCALE**2 times
        # weaker -- effectively unpenalised baseline hazards.
        period_scale = 50.0
        p = Xs.shape[1]
        D = np.zeros((m, p + H), dtype=np.float32)
        D[:, :p] = Xs[row_index].astype(np.float32)
        D[np.arange(m), p + period] = np.float32(period_scale)

        C = float(np.clip(1.0 / max(self.l2, 1e-12), 1e-4, 1e8))
        yb = event.astype(np.int64)
        if yb.max() == yb.min():
            LOGGER.warning("DiscreteTimeHazardModel: expansion has a single class; baseline only")
            self._beta_ = np.zeros(p)
            self.alpha_ = self._alpha_init_.copy()
            return
        # sklearn >= 1.8 deprecates `penalty=`; l1_ratio=0.0 (the default) *is* ridge.
        clf = LogisticRegression(
            C=C, l1_ratio=0.0, fit_intercept=False, solver="lbfgs", max_iter=400, tol=1e-5
        )
        clf.fit(D, yb, sample_weight=w[row_index])
        coef = clf.coef_.ravel().astype(np.float64)
        self._beta_ = coef[:p]
        self.alpha_ = coef[p:] * period_scale
        self._sk_model_ = clf

    def _fit_torch(
        self,
        Xs: np.ndarray,
        exp: PersonPeriodExpansion,
        w: np.ndarray,
        H: int,
        rng: np.random.Generator,
    ) -> None:
        """Fit an MLP ``f(x)`` with masked BCE over the dense person-period grid.

        The dense ``(n, horizon)`` mask/target pair is *derived from the same expansion
        arrays* -- ``mask[row_index, period] = 1`` -- so the vectorised recipe of
        ``SPEC_PERF`` section 4 remains the single source of truth while the optimiser
        never materialises a duplicated copy of ``X`` (which is what that section warns
        about for large ``n``).
        """
        import torch
        from torch import nn

        prev_threads: int | None = None
        if self.torch_threads is not None:
            prev_threads = int(torch.get_num_threads())
            torch.set_num_threads(int(self.torch_threads))
        try:
            self._fit_torch_inner(Xs, exp, w, H, rng, torch, nn)
        finally:
            if prev_threads is not None:
                torch.set_num_threads(prev_threads)

    def _fit_torch_inner(
        self,
        Xs: np.ndarray,
        exp: PersonPeriodExpansion,
        w: np.ndarray,
        H: int,
        rng: np.random.Generator,
        torch: Any,
        nn: Any,
    ) -> None:
        """Body of the torch fit; see :meth:`_fit_torch` for the contract."""
        n, p = Xs.shape
        mask = np.zeros((n, H), dtype=np.float32)
        target = np.zeros((n, H), dtype=np.float32)
        mask[exp.row_index, exp.period] = 1.0
        target[exp.row_index, exp.period] = exp.event.astype(np.float32)

        seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)
        net = _build_hazard_net(p, tuple(self.hidden), H, self._alpha_init_)

        # Optimiser budget guard rails (logged, per SPEC_PERF section 8).
        batch = int(max(1, min(self.batch_size, n)))
        epochs = int(max(1, self.epochs))
        lr = float(self.lr)
        steps_per_epoch = max(1, math.ceil(n / batch))
        total = steps_per_epoch * epochs
        if total < int(self.min_steps):
            epochs = int(math.ceil(self.min_steps / steps_per_epoch))
            LOGGER.info(
                "DiscreteTimeHazardModel[torch]: raising epochs %d -> %d to reach >= %d steps",
                self.epochs, epochs, self.min_steps,
            )
        elif total > int(self.max_steps):
            new_batch = int(math.ceil(n * epochs / self.max_steps))
            lr = lr * math.sqrt(new_batch / batch)
            LOGGER.info(
                "DiscreteTimeHazardModel[torch]: raising batch_size %d -> %d (lr %.2e -> %.2e) "
                "to stay within %d steps",
                batch, new_batch, self.lr, lr, self.max_steps,
            )
            batch = new_batch
            steps_per_epoch = max(1, math.ceil(n / batch))

        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=float(self.l2))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        bce = nn.BCEWithLogitsLoss(reduction="none")

        Xt = torch.as_tensor(Xs, dtype=torch.float32)
        Mt = torch.as_tensor(mask)
        Yt = torch.as_tensor(target)
        Wt = torch.as_tensor(w.astype(np.float32))

        net.train()
        for ep in range(epochs):
            order = rng.permutation(n)
            running, denom = 0.0, 0.0
            for s in range(0, n, batch):
                sl = torch.as_tensor(order[s : s + batch], dtype=torch.long)
                logits = net(Xt[sl])
                mb, yb, wb = Mt[sl], Yt[sl], Wt[sl][:, None]
                loss_el = bce(logits, yb) * mb * wb
                wsum = (mb * wb).sum().clamp_min(1.0)
                loss = loss_el.sum() / wsum
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += float(loss.detach()) * float(wsum)
                denom += float(wsum)
            sched.step()
            if self.verbose and (ep % 10 == 0 or ep == epochs - 1):
                LOGGER.info("  epoch %3d  loss %.5f", ep, running / max(denom, 1.0))

        net.eval()
        self._net_ = net
        self.alpha_ = net.alpha.detach().numpy().astype(np.float64)
        self._beta_ = None
        self.torch_epochs_ = epochs
        self.torch_batch_size_ = batch

    # -- predict -----------------------------------------------------------------------
    def predict_hazard(self, X: Any) -> np.ndarray:
        """Discrete hazard ``h(k | x)`` for periods ``1 .. horizon``.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.

        Returns
        -------
        ndarray of shape (n, horizon)
            Hazards clipped to ``[1e-6, 1 - 1e-6]``.
        """
        f = self._linear_predictor(X)
        return np.clip(expit(f[:, None] + self.alpha_[None, :]), _P_LO, _P_HI)

    def predict_survival(self, X: Any) -> np.ndarray:
        """Survival curve ``S(k) = prod_{j<=k} (1 - h(j))`` for ``k = 1 .. horizon``.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.

        Returns
        -------
        ndarray of shape (n, horizon)
            Strictly inside ``(0, 1)`` and monotone non-increasing along axis 1.
        """
        return _clip_survival(np.cumprod(1.0 - self.predict_hazard(X), axis=1))

    def predict_rmst(self, X: Any, horizon: int | float | None = None) -> np.ndarray:
        """Restricted mean survival time, ``sum_{k=1..H} S(k)``.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        horizon : int or float, optional
            Restriction horizon in periods; defaults to the fitted ``horizon`` and is
            capped at it. A fractional horizon adds the partial term ``frac * S(H)``.

        Returns
        -------
        ndarray of shape (n,)
            RMST in periods, inside ``[0, horizon]``.

        Notes
        -----
        ``S`` is a **step function**: a discrete-time hazard model has no resolution inside
        a period, so ``S(t) = S(floor(t))`` for ``t`` between grid points and ``S(0) = 1``.
        The partial term is therefore ``frac * S(floor(H))``, which is what
        :meth:`CoxPHModel.predict_rmst` and :meth:`RandomSurvivalForestLite.predict_rmst`
        compute when they evaluate *their* step functions at ``H``. Using ``S(floor(H) + 1)``
        here instead would silently make this class's fractional-horizon RMST disagree with
        the other two, which ``prism.causal.survival_uplift`` differences against each other.
        """
        S = self.predict_survival(X)
        H = float(self.horizon if horizon is None else horizon)
        H = float(np.clip(H, 0.0, float(self.horizon)))
        whole = int(math.floor(H + 1e-9))
        frac = H - whole
        out = S[:, :whole].sum(axis=1) if whole > 0 else np.zeros(S.shape[0])
        if frac > 1e-9:
            s_at_H = S[:, whole - 1] if whole > 0 else np.ones(S.shape[0])
            out = out + frac * s_at_H
        return out

    def predict_churn_prob(self, X: Any, within: int = 1) -> np.ndarray:
        """Probability of churning within the next ``within`` periods, ``1 - S(within)``.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        within : int, default 1
            Number of periods, clipped to ``[1, horizon]``.

        Returns
        -------
        ndarray of shape (n,)
        """
        k = int(np.clip(within, 1, self.horizon))
        return 1.0 - self.predict_survival(X)[:, k - 1]

    def predict(self, X: Any) -> np.ndarray:
        """Alias of :meth:`predict_rmst` (the sklearn-compatible entry point)."""
        return self.predict_rmst(X)


# ======================================================================================
# 2. Cox proportional hazards (Efron ties, analytic gradient, L-BFGS-B)
# ======================================================================================
@dataclass(frozen=True)
class _EfronBlocks:
    """Pre-computed, beta-independent geometry of the Efron partial likelihood.

    Attributes
    ----------
    X : ndarray of shape (n, p)
        Design matrix sorted by ascending time.
    event_rows : ndarray of int64, shape (n_events,)
        Positions (in the sorted array) of the observed events, themselves time-ordered.
    group_start : ndarray of int64, shape (J,)
        Offsets into ``event_rows`` where each distinct event time begins; the argument
        ``np.add.reduceat`` needs to collapse tie blocks.
    group_size : ndarray of float64, shape (J,)
        ``d_j``, the number of tied events at each distinct event time.
    risk_start : ndarray of int64, shape (J,)
        First index ``i`` in the sorted array with ``T_i >= t_j``; the risk set ``R_j`` is
        the suffix ``[risk_start[j]:]``, which is why risk-set sums are reversed cumsums.
    frac : ndarray of shape (J, L)
        ``l / d_j`` for ``l = 0 .. L-1`` where ``L = max_j d_j``.
    valid : ndarray of bool, shape (J, L)
        ``l < d_j`` mask that switches off the padded columns of ``frac``.
    sum_x_events : ndarray of shape (p,)
        ``sum over all events of x_i`` -- the (constant) first term of the gradient.
    """

    X: np.ndarray
    event_rows: np.ndarray
    group_start: np.ndarray
    group_size: np.ndarray
    risk_start: np.ndarray
    frac: np.ndarray
    valid: np.ndarray
    sum_x_events: np.ndarray


def _build_efron_blocks(
    X: np.ndarray, t: np.ndarray, d: np.ndarray
) -> tuple[_EfronBlocks, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort once and pre-compute every beta-independent index the Efron likelihood needs.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Standardised design matrix in original row order.
    t : ndarray of shape (n,)
        Observed follow-up times.
    d : ndarray of shape (n,)
        Event indicators.

    Returns
    -------
    blocks : _EfronBlocks
        Beta-independent geometry of the likelihood.
    uniq_t : ndarray of shape (J,)
        Distinct observed event times, ascending.
    ts, ds : ndarray of shape (n,)
        ``t`` and ``d`` sorted by ascending time.
    order : ndarray of int64, shape (n,)
        The sorting permutation.
    """
    order = np.argsort(t, kind="stable")
    Xs = np.ascontiguousarray(X[order])
    ts, ds = t[order], d[order]

    event_rows = np.flatnonzero(ds == 1).astype(np.int64)
    if event_rows.size == 0:
        raise ValueError("CoxPHModel.fit: no observed events (all rows are censored)")
    t_ev = ts[event_rows]
    uniq_t, group_start, counts = np.unique(t_ev, return_index=True, return_counts=True)
    group_start = group_start.astype(np.int64)
    group_size = counts.astype(np.float64)
    # Risk set R_j = {i : T_i >= t_j} = suffix starting at the first index with T >= t_j.
    risk_start = np.searchsorted(ts, uniq_t, side="left").astype(np.int64)

    L = int(counts.max())
    ell = np.arange(L, dtype=np.float64)[None, :]
    frac = ell / group_size[:, None]
    valid = ell < group_size[:, None]

    return _EfronBlocks(
        X=Xs,
        event_rows=event_rows,
        group_start=group_start,
        group_size=group_size,
        risk_start=risk_start,
        frac=frac,
        valid=valid,
        sum_x_events=Xs[event_rows].sum(axis=0),
    ), uniq_t, ts, ds, order


def _efron_objective(
    beta: np.ndarray, blocks: _EfronBlocks, l2: float, n: int
) -> tuple[float, np.ndarray]:
    """Penalised negative Efron log partial likelihood and its analytic gradient.

    Parameters
    ----------
    beta : ndarray of shape (p,)
        Coefficients on the *standardised* design.
    blocks : _EfronBlocks
        Output of :func:`_build_efron_blocks`.
    l2 : float
        Ridge strength on the mean-scaled objective.
    n : int
        Number of subjects (the log-likelihood is divided by it so ``l2`` is scale-free).

    Returns
    -------
    value : float
        ``-loglik / n + 0.5 * l2 * ||beta||^2``.
    grad : ndarray of shape (p,)
        Exact gradient of ``value``.

    Notes
    -----
    The Efron correction sums ``log(S_j - (l/d_j) s_j)`` over ``l = 0 .. d_j - 1``. Writing
    ``w_jl = 1 / (S_j - (l/d_j) s_j)``, the gradient's inner sum collapses to

    ``sum_l w_jl (Z_j - (l/d_j) z_j) = A_j Z_j - B_j z_j``

    with ``A_j = sum_l w_jl`` and ``B_j = sum_l (l/d_j) w_jl``. Only ``(J, L)`` and ``(J, p)``
    arrays are ever formed -- never the ``(J, L, p)`` tensor the naive broadcast would build.
    The partial likelihood is invariant to a constant shift of ``eta``, so ``eta`` is centred
    on its maximum before exponentiating and ``exp`` can never overflow.
    """
    X = blocks.X
    eta = X @ beta
    eta = eta - eta.max()
    r = np.exp(eta)
    rx = r[:, None] * X

    # Risk-set sums as reversed cumulative sums (SPEC_PERF section 5).
    csum_r = np.cumsum(r[::-1])[::-1]
    csum_rx = np.cumsum(rx[::-1], axis=0)[::-1]

    S_R = csum_r[blocks.risk_start]                     # (J,)
    Z_R = csum_rx[blocks.risk_start]                    # (J, p)

    ev = blocks.event_rows
    S_D = np.add.reduceat(r[ev], blocks.group_start)    # (J,)
    Z_D = np.add.reduceat(rx[ev], blocks.group_start, axis=0)  # (J, p)
    sum_eta_D = float(eta[ev].sum())

    denom = S_R[:, None] - blocks.frac * S_D[:, None]   # (J, L)
    denom = np.where(blocks.valid, denom, 1.0)
    denom = np.maximum(denom, 1e-300)

    loglik = sum_eta_D - float(np.sum(np.log(denom) * blocks.valid))

    w = np.where(blocks.valid, 1.0 / denom, 0.0)        # (J, L)
    A = w.sum(axis=1)                                   # (J,)
    B = (w * blocks.frac).sum(axis=1)                   # (J,)
    grad_ll = blocks.sum_x_events - (A[:, None] * Z_R - B[:, None] * Z_D).sum(axis=0)

    value = -loglik / n + 0.5 * l2 * float(beta @ beta)
    grad = -grad_ll / n + l2 * beta
    return float(value), np.asarray(grad, dtype=np.float64)


class CoxPHModel(BaseEstimator):
    """Cox proportional hazards with Efron ties, fitted by L-BFGS-B on an analytic gradient.

    No ``lifelines`` dependency: the partial likelihood, its gradient and the Breslow
    baseline are all computed here in numpy. The smoke test cross-checks the coefficients
    against ``lifelines.CoxPHFitter`` when it happens to be installed.

    Parameters
    ----------
    l2 : float, default 1e-3
        Ridge penalty applied to the *mean-scaled* negative log partial likelihood, i.e.
        the objective is ``-loglik / n + 0.5 * l2 * ||beta||^2`` with ``beta`` on the
        standardised scale. Small values leave the MLE essentially untouched.
    max_iter : int, default 200
        L-BFGS-B iteration cap.
    tol : float, default 1e-7
        Gradient / function tolerance handed to L-BFGS-B.

    Attributes
    ----------
    coef_ : ndarray of shape (p,)
        Log hazard ratios **on the original feature scale** (the internal standardisation
        is undone), directly comparable to ``lifelines`` output.
    baseline_cumhaz_ : pandas.DataFrame
        Columns ``time``, ``baseline_hazard``, ``baseline_cumhaz`` with a plain
        ``RangeIndex``. ``baseline_cumhaz`` is Breslow's estimator of ``H_0(t)`` for a
        subject at the training-mean covariate vector.
    n_features_in_, n_events_, log_likelihood_, converged_, n_iter_, fit_seconds_
        Fit diagnostics.

    Notes
    -----
    **Centring.** Like ``lifelines``, risk is reported relative to the training mean:
    ``predict_risk(x) = exp((x - x_mean) @ coef_)``. The dropped constant
    ``exp(x_mean @ coef_)`` is absorbed into ``baseline_cumhaz_``, so every survival curve
    and every hazard ratio is unchanged; only the arbitrary scale of the "baseline subject"
    is fixed. ``coef_`` itself is completely unaffected by this choice.

    **RMST.** ``predict_rmst`` follows the module-wide convention
    ``RMST(H) = sum_{k=1..H} S(k)`` on the integer period grid, so it is directly comparable
    with :class:`DiscreteTimeHazardModel` and :class:`RandomSurvivalForestLite`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(500, 2))
    >>> t = rng.exponential(1.0 / (0.1 * np.exp(X @ [1.0, -0.5])))
    >>> e = np.ones(500, dtype=int)
    >>> cox = CoxPHModel().fit(X, t, e)
    >>> bool(np.all(np.abs(cox.coef_ - np.array([1.0, -0.5])) < 0.25))
    True
    """

    def __init__(self, l2: float = 1e-3, max_iter: int = 200, tol: float = 1e-7) -> None:
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol

    # -- fit ---------------------------------------------------------------------------
    def fit(self, X: Any, event_time: Any, event_observed: Any) -> CoxPHModel:
        """Maximise the Efron partial likelihood and compute the Breslow baseline.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates. Non-finite entries are median-imputed.
        event_time : array-like of shape (n,)
            Observed follow-up time.
        event_observed : array-like of shape (n,)
            ``1`` if the event was observed, ``0`` if right-censored.

        Returns
        -------
        CoxPHModel
            ``self``, fitted.

        Raises
        ------
        ValueError
            If there are no observed events.
        """
        t0 = _time.perf_counter()
        Xa = _as_2d(X)
        n, p = Xa.shape
        t, d = _check_survival_inputs(event_time, event_observed, n)

        Xf = np.where(np.isfinite(Xa), Xa, np.nan)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(Xf, axis=0)
        self._x_median_ = np.where(np.isfinite(med), med, 0.0)
        Xf = np.where(np.isnan(Xf), self._x_median_[None, :], Xf)
        self._x_mean_ = Xf.mean(axis=0)
        sd = Xf.std(axis=0)
        self._x_scale_ = np.where(sd > 1e-12, sd, 1.0)
        Xstd = (Xf - self._x_mean_[None, :]) / self._x_scale_[None, :]

        blocks, uniq_t, ts, ds, order = _build_efron_blocks(Xstd, t, d)

        res = minimize(
            _efron_objective,
            x0=np.zeros(p),
            args=(blocks, float(self.l2), n),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": int(self.max_iter),
                "ftol": float(self.tol),
                "gtol": float(self.tol),
            },
        )
        beta_std = np.asarray(res.x, dtype=np.float64)
        self.coef_ = beta_std / self._x_scale_
        self.converged_ = bool(res.success)
        self.n_iter_ = int(res.nit)
        self.n_features_in_ = p
        self.n_events_ = int(d.sum())
        value, _ = _efron_objective(beta_std, blocks, 0.0, n)
        self.log_likelihood_ = -value * n
        if not self.converged_:
            LOGGER.warning("CoxPHModel: L-BFGS-B did not converge (%s)", res.message)

        self._breslow(blocks.X, ts, ds, beta_std, uniq_t)
        self.fit_seconds_ = _time.perf_counter() - t0
        return self

    def _breslow(
        self,
        Xs_sorted: np.ndarray,
        ts: np.ndarray,
        ds: np.ndarray,
        beta_std: np.ndarray,
        uniq_t: np.ndarray,
    ) -> None:
        """Breslow estimator of the baseline cumulative hazard on the centred linear predictor."""
        eta = Xs_sorted @ beta_std
        r = np.exp(np.clip(eta, -700.0, 700.0))
        csum_r = np.cumsum(r[::-1])[::-1]
        risk_start = np.searchsorted(ts, uniq_t, side="left")
        S_R = csum_r[risk_start]
        ev_rows = np.flatnonzero(ds == 1)
        _, gstart, counts = np.unique(ts[ev_rows], return_index=True, return_counts=True)
        d_j = counts.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            haz = np.where(S_R > 0, d_j / np.maximum(S_R, 1e-300), 0.0)
        cum = np.cumsum(haz)
        self._bh_times_ = np.asarray(uniq_t, dtype=np.float64)
        self._bh_cumhaz_ = np.asarray(cum, dtype=np.float64)
        self.baseline_cumhaz_ = pd.DataFrame(
            {
                "time": self._bh_times_,
                "baseline_hazard": haz,
                "baseline_cumhaz": self._bh_cumhaz_,
            }
        )

    # -- predict -----------------------------------------------------------------------
    def _centred_eta(self, X: Any) -> np.ndarray:
        """Linear predictor ``(x - x_mean) @ coef_``, with median imputation of gaps."""
        Xa = _as_2d(X)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {Xa.shape[1]} features, model was fitted with {self.n_features_in_}"
            )
        Xf = np.where(np.isfinite(Xa), Xa, np.nan)
        Xf = np.where(np.isnan(Xf), self._x_median_[None, :], Xf)
        return (Xf - self._x_mean_[None, :]) @ self.coef_

    def predict_risk(self, X: Any) -> np.ndarray:
        """Relative risk ``exp((x - x_mean) @ coef_)``.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.

        Returns
        -------
        ndarray of shape (n,)
            Strictly positive risk scores. The constant ``exp(x_mean @ coef_)`` dropped by
            the centring lives in :attr:`baseline_cumhaz_`, so ratios and rankings are the
            same as for the uncentred ``exp(x @ beta)``.
        """
        return np.exp(np.clip(self._centred_eta(X), -700.0, 700.0))

    def predict_survival(self, X: Any, times: Any) -> np.ndarray:
        """Survival probabilities ``S(t | x) = exp(-H_0(t) * risk(x))`` (Breslow baseline).

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        times : array-like of shape (k,)
            Non-decreasing time grid.

        Returns
        -------
        ndarray of shape (n, k)
            Strictly inside ``(0, 1)`` and monotone non-increasing along axis 1.
        """
        ts = _resolve_times(times)
        risk = self.predict_risk(X)
        H0 = step_eval(self._bh_times_, self._bh_cumhaz_, ts, before=0.0)
        with np.errstate(over="ignore"):
            S = np.exp(-np.clip(H0[None, :] * risk[:, None], 0.0, 700.0))
        return _clip_survival(S)

    def predict_rmst(self, X: Any, horizon: float) -> np.ndarray:
        """Restricted mean survival time, ``sum_{k=1..H} S(k)``.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        horizon : float
            Restriction horizon in periods. A fractional horizon adds ``frac * S(H)``,
            the Breslow step function evaluated at ``H`` itself.

        Returns
        -------
        ndarray of shape (n,)
            RMST in periods, inside ``[0, horizon]``.
        """
        H = float(horizon)
        if H <= 0:
            return np.zeros(_as_2d(X).shape[0])
        whole = int(math.floor(H + 1e-9))
        frac = H - whole
        grid = np.arange(1, whole + 1, dtype=np.float64)
        if frac > 1e-9:
            grid = np.append(grid, H)
        if grid.size == 0:
            return np.zeros(_as_2d(X).shape[0])
        S = self.predict_survival(X, grid)
        weights = np.ones(grid.size)
        if frac > 1e-9:
            weights[-1] = frac
        return S @ weights

    def predict(self, X: Any) -> np.ndarray:
        """Alias of :meth:`predict_risk` (higher = more hazardous)."""
        return self.predict_risk(X)


# ======================================================================================
# 3. Random survival forest (pre-binned vectorised log-rank, Nelson-Aalen leaves)
# ======================================================================================
@dataclass
class _SurvivalTree:
    """A single fitted survival tree stored as flat arrays.

    Attributes
    ----------
    feature : ndarray of int32, shape (n_nodes,)
        Split feature per node; ``-1`` marks a leaf.
    threshold : ndarray of int16, shape (n_nodes,)
        Split is ``bin <= threshold`` goes left.
    left, right : ndarray of int32, shape (n_nodes,)
        Child node ids; ``-1`` on leaves.
    leaf_id : ndarray of int32, shape (n_nodes,)
        Row of ``chf`` holding this node's cumulative hazard; ``-1`` on internal nodes.
    chf : ndarray of float32, shape (n_leaves, n_times)
        Nelson-Aalen cumulative hazard evaluated on the forest's shared time grid.
    """

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    leaf_id: np.ndarray
    chf: np.ndarray


def _nelson_aalen_leaf(
    rows: np.ndarray, ev: np.ndarray, eb: np.ndarray, rb: np.ndarray, n_t: int
) -> np.ndarray:
    """Nelson-Aalen cumulative hazard for the subjects in ``rows``, on the shared grid.

    Parameters
    ----------
    rows : ndarray of int64
        Row indices (may repeat -- the bootstrap draws with replacement).
    ev : ndarray of float64, shape (n,)
        Event indicators.
    eb : ndarray of int64, shape (n,)
        Grid index each subject's event is attributed to.
    rb : ndarray of int64, shape (n,)
        Number of grid points at or before each subject's time; the subject is at risk at
        grid point ``s`` iff ``s < rb``.
    n_t : int
        Length of the shared time grid.

    Returns
    -------
    ndarray of float32, shape (n_t,)
        Non-decreasing cumulative hazard ``H(t_s)``.
    """
    d = np.bincount(eb[rows], weights=ev[rows], minlength=n_t)[:n_t]
    cnt = np.bincount(rb[rows], minlength=n_t + 1)[: n_t + 1]
    at_risk = np.cumsum(cnt[::-1])[::-1][1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        inc = np.where(at_risk > 0, d / np.maximum(at_risk, 1.0), 0.0)
    return np.cumsum(inc).astype(np.float32)


def _best_logrank_split(
    Xb: np.ndarray,
    rows: np.ndarray,
    ev: np.ndarray,
    eb_c: np.ndarray,
    rb_c: np.ndarray,
    feats: np.ndarray,
    n_bins: int,
    n_tc: int,
    min_samples_leaf: int,
) -> tuple[int, int] | None:
    """Vectorised pre-binned log-rank split search over ``feats`` x all bin thresholds.

    Two ``np.bincount`` calls build the ``(mtry, n_bins, n_time)`` contingency cube of
    events and at-risk counts; one cumulative sum along the bin axis turns it into the
    left-child statistics of *every* threshold at once, and a reversed cumulative sum along
    the time axis turns the exit-time histogram into the at-risk process. There is no
    Python loop over thresholds, features or rows.

    Parameters
    ----------
    Xb : ndarray of uint8, shape (n, p)
        Quantile-binned design matrix.
    rows : ndarray of int64
        Rows in the current node (may repeat).
    ev : ndarray of float64, shape (n,)
        Event indicators.
    eb_c, rb_c : ndarray of int64, shape (n,)
        Coarse-grid event index and at-risk bound (see :func:`_nelson_aalen_leaf`).
    feats : ndarray of int64, shape (mtry,)
        Candidate features drawn for this node.
    n_bins : int
        Number of bins per feature (uniform padding width).
    n_tc : int
        Length of the coarse log-rank time grid.
    min_samples_leaf : int
        Minimum rows on each side of the split.

    Returns
    -------
    tuple of (int, int) or None
        ``(feature, bin_threshold)`` of the best split, or ``None`` if no admissible split
        has positive log-rank variance.

    Notes
    -----
    The statistic is the standard log-rank ``U^2 / V`` with the hypergeometric variance
    ``V = sum_s d_s * (n_ls / n_s) * (1 - n_ls / n_s) * (n_s - d_s) / (n_s - 1)``.
    """
    m = rows.size
    mtry = feats.size
    # np.ix_ gathers only the mtry candidate columns; Xb[rows][:, feats] would first
    # materialise the full (m, p) row block, which at p = 50 is 50x the memory traffic.
    Xn = Xb[np.ix_(rows, feats)].T.astype(np.int64)     # (mtry, m)
    base = np.arange(mtry, dtype=np.int64)[:, None] * n_bins + Xn

    code_r = (base * (n_tc + 1) + rb_c[rows][None, :]).ravel()
    cnt_r = np.bincount(code_r, minlength=mtry * n_bins * (n_tc + 1)).reshape(
        mtry, n_bins, n_tc + 1
    )
    code_e = (base * n_tc + eb_c[rows][None, :]).ravel()
    w_e = np.tile(ev[rows], mtry)
    cnt_d = np.bincount(code_e, weights=w_e, minlength=mtry * n_bins * n_tc).reshape(
        mtry, n_bins, n_tc
    )

    Dl = np.cumsum(cnt_d, axis=1)                        # (mtry, B, Tc) events left
    Rl = np.cumsum(cnt_r, axis=1)                        # (mtry, B, Tc+1) exit histogram left
    suf = np.cumsum(Rl[:, :, ::-1], axis=2)[:, :, ::-1]  # suf[..., k] = #{rb >= k}
    Nl = suf[:, :, 1:].astype(np.float64)                # at risk left at grid s
    left_n = suf[:, :, 0]                                # rows on the left

    D = Dl[:, -1:, :]                                    # node totals (identical per slot)
    N = Nl[:, -1:, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(N > 0, Nl / np.maximum(N, 1.0), 0.0)
        U = (Dl - D * pr).sum(axis=2)
        var = np.where(
            N > 1, D * pr * (1.0 - pr) * (N - D) / np.maximum(N - 1.0, 1.0), 0.0
        )
        V = var.sum(axis=2)
        stat = np.where(V > 1e-12, (U * U) / np.maximum(V, 1e-12), -np.inf)

    admissible = (left_n >= min_samples_leaf) & ((m - left_n) >= min_samples_leaf)
    stat = np.where(admissible, stat, -np.inf)
    k = int(np.argmax(stat))
    if not np.isfinite(stat.flat[k]):
        return None
    slot, b = divmod(k, n_bins)
    return int(feats[slot]), int(b)


def _grow_survival_tree(
    Xb: np.ndarray,
    ev: np.ndarray,
    eb_c: np.ndarray,
    rb_c: np.ndarray,
    eb_f: np.ndarray,
    rb_f: np.ndarray,
    n_bins: int,
    n_tc: int,
    n_tf: int,
    max_depth: int,
    min_samples_leaf: int,
    mtry: int,
    seed: int,
) -> _SurvivalTree:
    """Grow one bootstrap survival tree. Module-level so joblib can ship it to workers.

    Parameters
    ----------
    Xb : ndarray of uint8, shape (n, p)
        Quantile-binned design matrix.
    ev : ndarray of float64, shape (n,)
        Event indicators.
    eb_c, rb_c : ndarray of int64
        Coarse (log-rank) grid indices.
    eb_f, rb_f : ndarray of int64
        Fine (Nelson-Aalen) grid indices.
    n_bins, n_tc, n_tf : int
        Bin count, coarse grid length, fine grid length.
    max_depth, min_samples_leaf, mtry : int
        Tree hyper-parameters.
    seed : int
        Seed for this tree's bootstrap draw and feature sampling.

    Returns
    -------
    _SurvivalTree
    """
    rng = np.random.default_rng(seed)
    n, p = Xb.shape
    boot = rng.integers(0, n, size=n)

    feature: list[int] = [-1]
    threshold: list[int] = [-1]
    left: list[int] = [-1]
    right: list[int] = [-1]
    leaf_id: list[int] = [-1]
    chf: list[np.ndarray] = []

    stack: list[tuple[np.ndarray, int, int]] = [(boot, 0, 0)]
    while stack:
        rows, depth, nid = stack.pop()
        split = None
        if (
            depth < max_depth
            and rows.size >= 2 * min_samples_leaf
            and ev[rows].sum() >= 1.0
        ):
            feats = rng.choice(p, size=min(mtry, p), replace=False)
            split = _best_logrank_split(
                Xb, rows, ev, eb_c, rb_c, feats, n_bins, n_tc, min_samples_leaf
            )
        if split is None:
            leaf_id[nid] = len(chf)
            chf.append(_nelson_aalen_leaf(rows, ev, eb_f, rb_f, n_tf))
            continue

        f, b = split
        mask_left = Xb[rows, f] <= b
        feature[nid], threshold[nid] = f, b
        for child_rows, slot in ((rows[mask_left], "l"), (rows[~mask_left], "r")):
            cid = len(feature)
            feature.append(-1)
            threshold.append(-1)
            left.append(-1)
            right.append(-1)
            leaf_id.append(-1)
            if slot == "l":
                left[nid] = cid
            else:
                right[nid] = cid
            stack.append((child_rows, depth + 1, cid))

    return _SurvivalTree(
        feature=np.asarray(feature, dtype=np.int32),
        threshold=np.asarray(threshold, dtype=np.int16),
        left=np.asarray(left, dtype=np.int32),
        right=np.asarray(right, dtype=np.int32),
        leaf_id=np.asarray(leaf_id, dtype=np.int32),
        chf=np.asarray(chf, dtype=np.float32) if chf else np.zeros((1, n_tf), np.float32),
    )


def _tree_leaves(tree: _SurvivalTree, Xb: np.ndarray, max_depth: int) -> np.ndarray:
    """Route every row of ``Xb`` to its leaf, one vectorised pass per tree level."""
    n = Xb.shape[0]
    node = np.zeros(n, dtype=np.int64)
    rows = np.arange(n)
    for _ in range(max_depth + 2):
        f = tree.feature[node]
        internal = f >= 0
        if not internal.any():
            break
        vals = Xb[rows, np.where(internal, f, 0)]
        go_left = vals <= tree.threshold[node]
        nxt = np.where(go_left, tree.left[node], tree.right[node])
        node = np.where(internal, nxt, node)
    return tree.leaf_id[node]


class RandomSurvivalForestLite(BaseEstimator):
    """Random survival forest with a pre-binned, fully vectorised log-rank split search.

    **Split strategy: (A), pre-binned log-rank** (``SPEC_PERF.md`` section 3). Every feature
    is quantile-binned to at most ``n_bins`` levels once, up front. A node's split search is
    then two ``np.bincount`` calls plus cumulative sums over a ``(mtry, n_bins, n_time)``
    contingency cube, which scores *all* ``n_bins - 1`` thresholds of *all* ``mtry``
    candidate features simultaneously. Strategy (B) -- extremely randomised splits -- costs
    the same two bincounts but scores one threshold per feature instead of thirty-one, so
    (A) buys strictly more information for the same work. There is no Python loop over
    thresholds, features or rows anywhere in the fit.

    Leaves store a **Nelson-Aalen** cumulative hazard on a shared time grid; the forest
    prediction averages cumulative hazards across trees and returns ``S(t) = exp(-H(t))``,
    which is monotone non-increasing by construction.

    Parameters
    ----------
    n_estimators : int, default 200
        Number of bootstrap trees.
    max_depth : int, default 6
        Maximum tree depth.
    min_samples_leaf : int, default 20
        Minimum rows on each side of a split.
    max_features : {"sqrt", "log2", None}, int or float, default "sqrt"
        Number of features sampled per node (``mtry``). ``"sqrt"`` uses
        ``ceil(sqrt(p))``.
    n_jobs : int, default -1
        Threads for ``joblib.Parallel``. Tree growing is numpy-heavy, so threads avoid the
        Windows process-spawn cost; pass ``1`` for a strictly serial fit.
    random_state : int or numpy.random.Generator or None, default None
        Seeds the per-tree streams through :func:`prism.utils.seeds.spawn_rngs`, so the
        fit is reproducible regardless of thread scheduling.
    n_bins : int, default 32
        Quantile bins per feature.
    n_logrank_bins : int, default 24
        Length of the coarse time grid used *only* by the split criterion. Keeping it small
        is what makes the per-node cube tiny; the leaf hazards still use the fine grid.
    max_time_points : int, default 128
        Length of the fine time grid on which leaf hazards (and therefore predictions) are
        stored.
    max_samples_fit : int, default 50_000
        Rows above which the training set is subsampled (with a logged warning), matching
        the "200 trees on 50k rows in under 60 s" target.

    Attributes
    ----------
    times_ : ndarray of shape (n_times,)
        The fine time grid the forest's cumulative hazards live on.
    trees_ : list of _SurvivalTree
    mtry_ : int
    n_features_in_ : int
    fit_seconds_ : float

    Notes
    -----
    ``predict_rmst`` follows the module-wide convention ``RMST(H) = sum_{k=1..H} S(k)``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 4))
    >>> t = np.ceil(rng.exponential(1.0 / (0.1 * np.exp(X[:, 0]))))
    >>> e = (t < 20).astype(int)
    >>> f = RandomSurvivalForestLite(n_estimators=20, max_time_points=32,
    ...                              random_state=0).fit(X, np.minimum(t, 20), e)
    >>> S = f.predict_survival(X, np.arange(1.0, 13.0))
    >>> bool(np.all(np.diff(S, axis=1) <= 1e-12))
    True
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        max_features: str | int | float | None = "sqrt",
        n_jobs: int = -1,
        random_state: int | np.random.Generator | None = None,
        *,
        n_bins: int = 32,
        n_logrank_bins: int = 24,
        max_time_points: int = 128,
        max_samples_fit: int = 50_000,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_bins = n_bins
        self.n_logrank_bins = n_logrank_bins
        self.max_time_points = max_time_points
        self.max_samples_fit = max_samples_fit

    # -- internals ---------------------------------------------------------------------
    def _resolve_mtry(self, p: int) -> int:
        """Translate ``max_features`` into a concrete number of candidate features."""
        mf = self.max_features
        if mf is None:
            return p
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(math.ceil(math.sqrt(p))))
            if mf == "log2":
                return max(1, int(math.ceil(math.log2(max(p, 2)))))
            raise ValueError(f"max_features must be 'sqrt', 'log2', an int, a float or None; got {mf!r}")
        if isinstance(mf, (int, np.integer)):
            return int(np.clip(int(mf), 1, p))
        return int(np.clip(int(math.ceil(float(mf) * p)), 1, p))

    def _fit_bins(self, X: np.ndarray) -> np.ndarray:
        """Quantile-bin every column once; store the edges for prediction-time binning.

        Raises
        ------
        ValueError
            If ``n_bins`` is outside ``[2, 256]``. Bin codes are stored as ``uint8`` (the
            contingency cube in :func:`_best_logrank_split` is indexed by them), so a
            larger value would wrap around and silently scramble the ordering the split
            search depends on -- the split would still "work", on a permuted feature.
        """
        n, p = X.shape
        if not 2 <= int(self.n_bins) <= 256:
            raise ValueError(
                f"n_bins must be between 2 and 256 (bin codes are uint8), got {self.n_bins}"
            )
        qs = np.linspace(0.0, 1.0, int(self.n_bins) + 1)[1:-1]
        self._bin_edges_: list[np.ndarray] = []
        Xb = np.zeros((n, p), dtype=np.uint8)
        for j in range(p):
            col = X[:, j]
            edges = np.unique(np.quantile(col, qs))
            if edges.size > int(self.n_bins) - 1:
                edges = edges[: int(self.n_bins) - 1]
            self._bin_edges_.append(edges)
            Xb[:, j] = np.searchsorted(edges, col, side="left").astype(np.uint8)
        self._n_bins_used_ = int(max(1, max((e.size + 1) for e in self._bin_edges_)))
        return Xb

    def _apply_bins(self, X: np.ndarray) -> np.ndarray:
        """Apply the bin edges learned in ``fit`` to new rows."""
        n, p = X.shape
        Xb = np.zeros((n, p), dtype=np.uint8)
        for j in range(p):
            Xb[:, j] = np.searchsorted(self._bin_edges_[j], X[:, j], side="left").astype(np.uint8)
        return Xb

    @staticmethod
    def _grid_indices(
        t: np.ndarray, d: np.ndarray, grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map times onto a grid: event index ``eb`` and at-risk bound ``rb``.

        ``rb_i`` counts grid points at or before ``T_i`` so subject ``i`` is at risk at grid
        point ``s`` iff ``s < rb_i``. Events are snapped *up* to the first grid point at or
        after their time and ``rb`` is raised to ``eb + 1`` so that ``n_s >= d_s`` always
        holds even on a coarsened grid.
        """
        n_t = grid.size
        rb = np.searchsorted(grid, t, side="right").astype(np.int64)
        eb = np.clip(np.searchsorted(grid, t, side="left"), 0, n_t - 1).astype(np.int64)
        is_ev = d == 1
        rb = np.where(is_ev, np.maximum(rb, eb + 1), rb)
        return eb, np.clip(rb, 0, n_t)

    # -- fit ---------------------------------------------------------------------------
    def fit(self, X: Any, event_time: Any, event_observed: Any) -> RandomSurvivalForestLite:
        """Grow the forest.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates. Non-finite entries are median-imputed before binning.
        event_time : array-like of shape (n,)
            Observed follow-up times.
        event_observed : array-like of shape (n,)
            ``1`` if the event was observed, ``0`` if right-censored.

        Returns
        -------
        RandomSurvivalForestLite
            ``self``, fitted.
        """
        from joblib import Parallel, delayed

        t0 = _time.perf_counter()
        Xa = _as_2d(X)
        n_all, p = Xa.shape
        t, d = _check_survival_inputs(event_time, event_observed, n_all)
        rng = as_rng(self.random_state)

        if n_all > int(self.max_samples_fit):
            keep = rng.choice(n_all, size=int(self.max_samples_fit), replace=False)
            keep.sort()
            LOGGER.warning(
                "RandomSurvivalForestLite: subsampled %d -> %d rows for the fit "
                "(max_samples_fit); predictions still use every row you pass in",
                n_all,
                int(self.max_samples_fit),
            )
            Xa, t, d = Xa[keep], t[keep], d[keep]

        Xf = np.where(np.isfinite(Xa), Xa, np.nan)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(Xf, axis=0)
        self._x_median_ = np.where(np.isfinite(med), med, 0.0)
        Xf = np.where(np.isnan(Xf), self._x_median_[None, :], Xf)

        Xb = self._fit_bins(Xf)
        self.n_features_in_ = p
        self.mtry_ = self._resolve_mtry(p)

        ev_times = np.unique(t[d == 1])
        if ev_times.size == 0:
            raise ValueError("RandomSurvivalForestLite.fit: no observed events")
        self.times_ = self._subsample_grid(ev_times, int(self.max_time_points), "prediction")
        # The log-rank grid is a deliberate design choice of the split criterion (it keeps
        # the per-node contingency cube tiny), not a truncation of the model output, so it
        # is reported at DEBUG rather than INFO.
        coarse = self._subsample_grid(ev_times, int(self.n_logrank_bins), "log-rank")

        eb_f, rb_f = self._grid_indices(t, d, self.times_)
        eb_c, rb_c = self._grid_indices(t, d, coarse)
        ev = d.astype(np.float64)

        seeds = [int(g.integers(0, 2**31 - 1)) for g in spawn_rngs(rng, int(self.n_estimators))]
        self.trees_ = list(
            Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_grow_survival_tree)(
                    Xb,
                    ev,
                    eb_c,
                    rb_c,
                    eb_f,
                    rb_f,
                    self._n_bins_used_,
                    coarse.size,
                    self.times_.size,
                    int(self.max_depth),
                    int(self.min_samples_leaf),
                    int(self.mtry_),
                    s,
                )
                for s in seeds
            )
        )
        self.fit_seconds_ = _time.perf_counter() - t0
        return self

    @staticmethod
    def _subsample_grid(values: np.ndarray, max_points: int, kind: str = "prediction") -> np.ndarray:
        """Thin ``values`` to at most ``max_points`` quantile-spaced points.

        Parameters
        ----------
        values : ndarray
            Distinct observed event times, ascending.
        max_points : int
            Maximum grid length.
        kind : {"prediction", "log-rank"}, default "prediction"
            Which grid is being built; only the prediction grid's thinning changes what the
            caller gets back, so only that one is reported at INFO (``SPEC_PERF`` section 8).

        Returns
        -------
        ndarray of float64
        """
        if values.size <= max_points:
            return np.asarray(values, dtype=np.float64)
        qs = np.linspace(0.0, 1.0, max_points)
        grid = np.unique(np.quantile(values, qs))
        log = LOGGER.info if kind == "prediction" else LOGGER.debug
        log(
            "RandomSurvivalForestLite: thinned %d distinct event times to a %d-point %s grid",
            values.size, grid.size, kind,
        )
        return np.asarray(grid, dtype=np.float64)

    # -- predict -----------------------------------------------------------------------
    def predict_cumulative_hazard(self, X: Any, *, chunk_size: int = 20_000) -> np.ndarray:
        """Forest-averaged Nelson-Aalen cumulative hazard on :attr:`times_`.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        chunk_size : int, default 20_000
            Rows processed at a time, to bound peak memory at ``chunk_size * n_times``.

        Returns
        -------
        ndarray of shape (n, n_times)
            Non-decreasing along axis 1.
        """
        Xa = _as_2d(X)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {Xa.shape[1]} features, model was fitted with {self.n_features_in_}"
            )
        Xf = np.where(np.isfinite(Xa), Xa, np.nan)
        Xf = np.where(np.isnan(Xf), self._x_median_[None, :], Xf)
        Xb = self._apply_bins(Xf)

        n = Xb.shape[0]
        out = np.zeros((n, self.times_.size), dtype=np.float64)
        for s in range(0, n, chunk_size):
            block = Xb[s : s + chunk_size]
            acc = np.zeros((block.shape[0], self.times_.size), dtype=np.float64)
            for tree in self.trees_:
                acc += tree.chf[_tree_leaves(tree, block, int(self.max_depth))]
            out[s : s + chunk_size] = acc / max(len(self.trees_), 1)
        return out

    def predict_survival(self, X: Any, times: Any, *, chunk_size: int = 20_000) -> np.ndarray:
        """Survival probabilities ``S(t) = exp(-H(t))`` at the requested times.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        times : array-like of shape (k,)
            Non-decreasing time grid.
        chunk_size : int, default 20_000
            Rows processed at a time.

        Returns
        -------
        ndarray of shape (n, k)
            Strictly inside ``(0, 1)`` and monotone non-increasing along axis 1.
        """
        ts = _resolve_times(times)
        H = self.predict_cumulative_hazard(X, chunk_size=chunk_size)
        idx = np.searchsorted(self.times_, ts, side="right") - 1
        H_at = np.where(idx[None, :] >= 0, H[:, np.clip(idx, 0, self.times_.size - 1)], 0.0)
        with np.errstate(over="ignore"):
            S = np.exp(-np.clip(H_at, 0.0, 700.0))
        return _clip_survival(S)

    def predict_rmst(self, X: Any, horizon: float) -> np.ndarray:
        """Restricted mean survival time, ``sum_{k=1..H} S(k)``.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        horizon : float
            Restriction horizon in periods; a fractional part adds ``frac * S(H)``.

        Returns
        -------
        ndarray of shape (n,)
            RMST in periods, inside ``[0, horizon]``.
        """
        H = float(horizon)
        if H <= 0:
            return np.zeros(_as_2d(X).shape[0])
        whole = int(math.floor(H + 1e-9))
        frac = H - whole
        grid = np.arange(1, whole + 1, dtype=np.float64)
        if frac > 1e-9:
            grid = np.append(grid, H)
        if grid.size == 0:
            return np.zeros(_as_2d(X).shape[0])
        S = self.predict_survival(X, grid)
        weights = np.ones(grid.size)
        if frac > 1e-9:
            weights[-1] = frac
        return S @ weights

    def predict(self, X: Any) -> np.ndarray:
        """Alias for the forest-averaged cumulative hazard at the last grid time (a risk score)."""
        return self.predict_cumulative_hazard(X)[:, -1]


# ======================================================================================
# 4. Censoring-aware metrics
# ======================================================================================
def concordance_index(
    event_time: Any,
    predicted_risk: Any,
    event_observed: Any,
    *,
    max_ops: int = 400_000_000,
    random_state: int | np.random.Generator | None = 0,
) -> float:
    """Harrell's concordance index with exact tie handling.

    Two subjects form a *comparable* pair when the one with the shorter observed time had
    an event; a subject censored at exactly the event time of another is still comparable
    with it, but two events at the same time are not comparable with each other (there is
    no ordering to get right). This is the scikit-survival / Uno convention.

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Observed follow-up times.
    predicted_risk : array-like of shape (n,)
        Risk score, **higher = shorter expected survival**. Pass ``-rmst`` or ``-S(t)`` when
        scoring a survival-scale prediction.
    event_observed : array-like of shape (n,)
        ``1`` if the event was observed, ``0`` if right-censored.
    max_ops : int, default 400_000_000
        Work budget, measured as ``n_unique_times * n_unique_risks``. If the budget would
        be exceeded the inputs are subsampled without replacement and a warning is
        **logged** naming the retained size (``SPEC_PERF.md`` section 8 -- silent truncation
        is not acceptable). PRISM's monthly panel has at most a couple of dozen distinct
        times, so a 120k-row C-index costs ~3e6 operations and is never subsampled; only
        fully continuous event times (where the budget implies ``n ~ 20_000``) trigger it.
    random_state : int or numpy.random.Generator or None, default 0
        Seed used only if subsampling kicks in, so the estimate is reproducible.

    Returns
    -------
    float
        ``(n_concordant + 0.5 * n_tied_risk) / n_comparable``; ``nan`` if no pair is
        comparable. A perfect ranking gives ``1.0``, its reverse ``0.0``, noise ``~0.5``.

    Notes
    -----
    Complexity is ``O(n log n + U * K)`` with ``U`` distinct times and ``K`` distinct risk
    values -- one pass over distinct times, each step a pair of length-``K`` cumulative
    sums, with no Python loop over subjects or pairs. For PRISM's monthly panel ``U <= 24``,
    so a 120k-row C-index costs a few million element operations.

    Examples
    --------
    >>> t = np.array([1.0, 2.0, 3.0, 4.0])
    >>> e = np.array([1, 1, 1, 1])
    >>> round(concordance_index(t, -t, e), 6)
    1.0
    >>> round(concordance_index(t, t, e), 6)
    0.0
    """
    t, d = _check_survival_inputs(event_time, event_observed)
    risk = _as_1d(predicted_risk, "predicted_risk", t.size)

    ok = np.isfinite(t) & np.isfinite(risk)
    if not ok.all():
        LOGGER.warning("concordance_index: dropped %d rows with non-finite values", int((~ok).sum()))
        t, d, risk = t[ok], d[ok], risk[ok]
    n = t.size
    if n < 2 or d.sum() == 0:
        return float("nan")

    def _sizes(tt: np.ndarray, rr: np.ndarray) -> tuple[int, int]:
        """Return ``(n_distinct_times, n_distinct_risks)``, the cost drivers of the loop."""
        return int(np.unique(tt).size), int(np.unique(rr).size)

    U, K = _sizes(t, risk)
    if max_ops < U * K:
        m = n
        while m > 1000 and min(U, m) * min(K, m) > max_ops:
            m //= 2
        idx = as_rng(random_state).choice(n, size=m, replace=False)
        idx.sort()
        t, d, risk = t[idx], d[idx], risk[idx]
        U, K = _sizes(t, risk)
        LOGGER.warning(
            "concordance_index: budget %d exceeded (U*K); subsampled %d -> %d rows "
            "without replacement and computed C on that subsample",
            max_ops, n, m,
        )
        n = m

    _, rank = np.unique(risk, return_inverse=True)
    rank = rank.astype(np.int64).ravel()
    order = np.argsort(t, kind="stable")
    ts, ds, rks = t[order], d[order], rank[order]
    _, starts, counts = np.unique(ts, return_index=True, return_counts=True)

    later = np.zeros(K, dtype=np.int64)     # risk-rank histogram of subjects with larger time
    n_later = 0
    conc = 0
    tied = 0
    comparable = 0

    for g in range(starts.size - 1, -1, -1):
        s, e = int(starts[g]), int(starts[g]) + int(counts[g])
        rk_g, d_g = rks[s:e], ds[s:e]
        ev = d_g == 1
        n_ev = int(ev.sum())
        if n_ev:
            re = rk_g[ev]
            pref_later = np.concatenate(([0], np.cumsum(later)))
            lt = int(pref_later[re].sum())
            eq = int(later[re].sum())
            tot = n_later
            cens_rk = rk_g[~ev]
            if cens_rk.size:
                ccnt = np.bincount(cens_rk, minlength=K)
                pref_c = np.concatenate(([0], np.cumsum(ccnt)))
                lt += int(pref_c[re].sum())
                eq += int(ccnt[re].sum())
                tot += int(cens_rk.size)
            conc += lt
            tied += eq
            comparable += n_ev * tot
        later += np.bincount(rk_g, minlength=K)
        n_later += rk_g.size

    if comparable == 0:
        return float("nan")
    return float((conc + 0.5 * tied) / comparable)


def time_dependent_auc(
    event_time: Any,
    event_observed: Any,
    risk: Any,
    times: Any,
    *,
    min_censor_prob: float = 1e-3,
) -> pd.DataFrame:
    """Uno's IPCW cumulative/dynamic time-dependent AUC.

    At each ``t`` the *cases* are subjects with an observed event by ``t``, re-weighted by
    ``1 / G(T_i-)`` to undo the censoring that removed their peers; the *controls* are
    subjects still event-free at ``t``. The statistic is the weighted probability that a
    case outranks a control, with ties in risk counting a half.

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Observed follow-up times.
    event_observed : array-like of shape (n,)
        ``1`` if the event was observed, ``0`` if right-censored.
    risk : array-like of shape (n,)
        Risk score, higher = shorter expected survival.
    times : array-like of shape (k,)
        Evaluation times.
    min_censor_prob : float, default 1e-3
        Floor on ``G`` so a weight cannot exceed ``1 / min_censor_prob``.

    Returns
    -------
    pandas.DataFrame
        Columns ``time``, ``auc``, ``n_cases``, ``n_controls``. ``auc`` is ``nan`` where a
        time point has no cases or no controls.

    Notes
    -----
    One ``argsort`` of the control risks plus two ``searchsorted`` calls per time point, so
    the cost is ``O(k * n log n)`` with no pairwise loop.

    References
    ----------
    Uno, Cai, Tian & Wei (2007), *Evaluating prediction rules for t-year survivors with
    censored regression models*, JASA 102(478).
    """
    t, d = _check_survival_inputs(event_time, event_observed)
    r = _as_1d(risk, "risk", t.size)
    ts = _resolve_times(times)

    g_grid, g_val = censoring_survival(t, d, min_prob=min_censor_prob)
    g_at_event = np.maximum(step_eval(g_grid, g_val, t, left_limit=True), min_censor_prob)

    rows: list[dict[str, float]] = []
    for tau in ts:
        case = (t <= tau) & (d == 1)
        ctrl = t > tau
        n_case, n_ctrl = int(case.sum()), int(ctrl.sum())
        if n_case == 0 or n_ctrl == 0:
            rows.append({"time": float(tau), "auc": float("nan"),
                         "n_cases": n_case, "n_controls": n_ctrl})
            continue
        w = 1.0 / g_at_event[case]
        rc = np.sort(r[ctrl])
        rk = r[case]
        lo = np.searchsorted(rc, rk, side="left")
        hi = np.searchsorted(rc, rk, side="right")
        score = lo + 0.5 * (hi - lo)               # controls ranked below each case
        auc = float((w * score).sum() / (w.sum() * n_ctrl))
        rows.append({"time": float(tau), "auc": auc, "n_cases": n_case, "n_controls": n_ctrl})
    out = pd.DataFrame(rows, columns=["time", "auc", "n_cases", "n_controls"])
    out["n_cases"] = out["n_cases"].astype("int64")
    out["n_controls"] = out["n_controls"].astype("int64")
    return out


def brier_score(
    event_time: Any,
    event_observed: Any,
    surv_prob: Any,
    times: Any,
    *,
    min_censor_prob: float = 1e-3,
) -> pd.DataFrame:
    """IPCW (Graf) Brier score of a survival prediction at each time in ``times``.

    ``BS(t) = mean_i [ S_i(t)^2 * 1{T_i <= t, delta_i = 1} / G(T_i-)
                     + (1 - S_i(t))^2 * 1{T_i > t} / G(t) ]``

    A subject censored before ``t`` contributes nothing directly; the inverse-probability
    weights on the two groups that *are* observable carry their share.

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Observed follow-up times.
    event_observed : array-like of shape (n,)
        ``1`` if the event was observed, ``0`` if right-censored.
    surv_prob : array-like of shape (n, k)
        Predicted ``S_i(t)`` for each subject and each time in ``times``. A ``(n,)`` vector
        is accepted when ``times`` has length 1.
    times : array-like of shape (k,)
        Evaluation times.
    min_censor_prob : float, default 1e-3
        Floor on ``G`` so a weight cannot exceed ``1 / min_censor_prob``.

    Returns
    -------
    pandas.DataFrame
        Columns ``time``, ``brier``, ``n_events``, ``n_at_risk`` where ``n_events`` counts
        subjects with an observed event by ``t`` and ``n_at_risk`` counts those still
        event-free at ``t``. Lower ``brier`` is better; ``0.25`` is the score of a constant
        ``0.5`` prediction.

    References
    ----------
    Graf, Schmoor, Sauerbrei & Schumacher (1999), *Assessment and comparison of prognostic
    classification schemes for survival data*, Statistics in Medicine 18.
    """
    t, d = _check_survival_inputs(event_time, event_observed)
    ts = _resolve_times(times)
    S = np.asarray(surv_prob, dtype=np.float64)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if S.shape != (t.size, ts.size):
        raise ValueError(
            f"surv_prob has shape {S.shape}, expected ({t.size}, {ts.size})"
        )
    S = np.clip(S, 0.0, 1.0)

    g_grid, g_val = censoring_survival(t, d, min_prob=min_censor_prob)
    g_event = np.maximum(step_eval(g_grid, g_val, t, left_limit=True), min_censor_prob)
    g_times = np.maximum(step_eval(g_grid, g_val, ts), min_censor_prob)

    case = (t[:, None] <= ts[None, :]) & (d[:, None] == 1)
    ctrl = t[:, None] > ts[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        w_case = np.where(case, 1.0 / g_event[:, None], 0.0)
        w_ctrl = np.where(ctrl, 1.0 / g_times[None, :], 0.0)
    contrib = w_case * S**2 + w_ctrl * (1.0 - S) ** 2
    bs = contrib.mean(axis=0)

    return pd.DataFrame(
        {
            "time": ts,
            "brier": bs,
            "n_events": case.sum(axis=0).astype("int64"),
            "n_at_risk": ctrl.sum(axis=0).astype("int64"),
        }
    )


def integrated_brier_score(
    event_time: Any,
    event_observed: Any,
    surv_prob: Any,
    times: Any,
    **kwargs: Any,
) -> float:
    """Integrated Brier score: the trapezoid integral of :func:`brier_score` over ``times``.

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Observed follow-up times.
    event_observed : array-like of shape (n,)
        ``1`` if the event was observed, ``0`` if right-censored.
    surv_prob : array-like of shape (n, k)
        Predicted survival probabilities at ``times``.
    times : array-like of shape (k,)
        Evaluation grid; at least two points are needed for a genuine integral.
    **kwargs
        Forwarded to :func:`brier_score` (e.g. ``min_censor_prob``).

    Returns
    -------
    float
        ``(1 / (t_max - t_min)) * int_{t_min}^{t_max} BS(t) dt``, so the result is on the
        same 0-1 scale as a single Brier score. With one time point the score at that point
        is returned unchanged.
    """
    tab = brier_score(event_time, event_observed, surv_prob, times, **kwargs)
    ts = tab["time"].to_numpy()
    bs = tab["brier"].to_numpy()
    if ts.size == 1:
        return float(bs[0])
    span = float(ts[-1] - ts[0])
    if span <= 0:
        return float(np.nanmean(bs))
    return float(_TRAPEZOID(bs, ts) / span)


def calibration_curve_survival(
    event_time: Any,
    event_observed: Any,
    surv_prob: Any,
    t: float,
    n_bins: int = 10,
    *,
    times: Any | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Predicted-vs-observed survival at a fixed horizon, binned by predicted risk.

    Subjects are split into ``n_bins`` equal-count bins of predicted ``S_i(t)``. Within a
    bin the *observed* survival is the Kaplan-Meier estimate at ``t`` -- not the raw
    survival rate, which would be biased by whoever happened to be censored early -- with a
    log-log-transformed Greenwood confidence interval so the bounds stay inside ``[0, 1]``.

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Observed follow-up times.
    event_observed : array-like of shape (n,)
        ``1`` if the event was observed, ``0`` if right-censored.
    surv_prob : array-like of shape (n,) or (n, k)
        Predicted survival at ``t``. If 2-D, pass ``times`` so the right column can be
        selected.
    t : float
        Horizon at which calibration is assessed.
    n_bins : int, default 10
        Number of equal-count bins. Bins that would be empty are dropped.
    times : array-like of shape (k,), optional
        The grid ``surv_prob``'s columns correspond to, required when ``surv_prob`` is 2-D
        with more than one column.
    alpha : float, default 0.05
        Two-sided significance level of the confidence interval.

    Returns
    -------
    pandas.DataFrame
        Columns ``bin`` (0-based, ordered by increasing predicted survival), ``n``,
        ``mean_predicted``, ``observed_km``, ``ci_low``, ``ci_high``. A well-calibrated
        model puts ``mean_predicted`` inside ``[ci_low, ci_high]`` in most bins.
    """
    tt, dd = _check_survival_inputs(event_time, event_observed)
    S = np.asarray(surv_prob, dtype=np.float64)
    if S.ndim == 2:
        if S.shape[1] == 1:
            S = S.ravel()
        else:
            if times is None:
                raise ValueError(
                    "surv_prob is 2-D; pass `times` so the column matching `t` can be chosen"
                )
            grid = _resolve_times(times)
            if grid.size != S.shape[1]:
                raise ValueError(f"times has length {grid.size}, surv_prob has {S.shape[1]} columns")
            S = S[:, int(np.argmin(np.abs(grid - float(t))))]
    S = _as_1d(S, "surv_prob", tt.size)

    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha / 2.0))
    n = tt.size
    n_bins = int(max(1, min(n_bins, n)))
    # Equal-count bins on the predicted value, ties kept together by ranking on a stable sort.
    order = np.argsort(S, kind="stable")
    bin_of = np.empty(n, dtype=np.int64)
    bin_of[order] = np.minimum((np.arange(n) * n_bins) // n, n_bins - 1)

    rows: list[dict[str, float]] = []
    for b in range(n_bins):
        sel = bin_of == b
        n_b = int(sel.sum())
        if n_b == 0:
            continue
        tb, db, sb = tt[sel], dd[sel], S[sel]
        grid, surv = kaplan_meier(tb, db)
        km = float(step_eval(grid, surv, np.array([float(t)]))[0])

        # Greenwood variance of log S(t), restricted to jumps at or before t.
        order_b = np.argsort(tb, kind="stable")
        ts_b, ds_b = tb[order_b], db[order_b]
        uniq, first, _ = np.unique(ts_b, return_index=True, return_counts=True)
        n_ev = np.add.reduceat(ds_b.astype(np.float64), first)
        at_risk = (ts_b.size - first).astype(np.float64)
        use = (n_ev > 0) & (uniq <= float(t))
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                use & (at_risk - n_ev > 0),
                n_ev / np.maximum(at_risk * (at_risk - n_ev), 1e-12),
                0.0,
            )
        var_log = float(terms.sum())

        if km >= 1.0 - 1e-12 or km <= 1e-12 or var_log <= 0.0:
            lo, hi = (km, km)
        else:
            c = z * math.sqrt(var_log) / abs(math.log(km))
            lo = float(km ** math.exp(c))
            hi = float(km ** math.exp(-c))
        rows.append(
            {
                "bin": b,
                "n": n_b,
                "mean_predicted": float(sb.mean()),
                "observed_km": km,
                "ci_low": float(np.clip(min(lo, hi), 0.0, 1.0)),
                "ci_high": float(np.clip(max(lo, hi), 0.0, 1.0)),
            }
        )

    out = pd.DataFrame(rows, columns=["bin", "n", "mean_predicted", "observed_km", "ci_low", "ci_high"])
    out["bin"] = out["bin"].astype("int64")
    out["n"] = out["n"].astype("int64")
    return out.reset_index(drop=True)


# ======================================================================================
# Smoke test
# ======================================================================================
def _simulate_ph_survival(
    n: int = 4000, p: int = 8, random_state: int = 11
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Right-censored data from a known exponential proportional-hazards model.

    The hazard is ``lambda_i = 0.085 * exp(x_i @ beta)``, so the Cox coefficients have an
    exact known answer and the recovered ``coef_`` can be checked against ``beta`` rather
    than merely inspected.

    Parameters
    ----------
    n : int, default 4000
        Number of subjects.
    p : int, default 8
        Number of standard-normal covariates.
    random_state : int, default 11
        Seed.

    Returns
    -------
    X : ndarray of shape (n, p)
    event_time : ndarray of shape (n,)
        ``min(T, C, 24)`` in months.
    event_observed : ndarray of int64, shape (n,)
    beta : ndarray of shape (p,)
        The true log hazard ratios.
    latent_T : ndarray of shape (n,)
        The uncensored event times (used only for the oracle C-index check).
    """
    rng = as_rng(random_state)
    beta = np.array([0.8, -0.5, 0.3, 0.0, 0.25, -0.2, 0.0, 0.1])[:p]
    X = rng.normal(size=(n, p))
    lam = 0.085 * np.exp(X @ beta)
    latent_T = rng.exponential(1.0 / lam)
    cens = np.minimum(rng.exponential(34.0, size=n), 24.0)
    event_time = np.minimum(latent_T, cens)
    event_observed = (latent_T <= cens).astype(np.int64)
    return X, event_time, event_observed, beta, latent_T


def _check_curve(name: str, S: np.ndarray) -> None:
    """Assert a survival matrix is strictly inside (0, 1) and monotone non-increasing."""
    assert np.isfinite(S).all(), f"{name}: non-finite survival"
    assert S.min() > 0.0 and S.max() < 1.0, f"{name}: survival not strictly inside (0,1)"
    assert np.all(np.diff(S, axis=1) <= 1e-12), f"{name}: survival not monotone non-increasing"


if __name__ == "__main__":  # pragma: no cover - smoke test
    import sys

    from prism.utils.optional import HAS_LIFELINES

    t_start = _time.perf_counter()
    N, P, HORIZON = 4000, 8, 12
    GRID = np.arange(1.0, HORIZON + 1.0)

    X, etime, eobs, beta_true, latent_T = _simulate_ph_survival(N, P, random_state=11)
    cens_rate = 1.0 - float(eobs.mean())
    print("PRISM prism/models/survival.py smoke test")
    print(f"  n={N}  p={P}  horizon={HORIZON}  censoring={cens_rate:.1%}  "
          f"events={int(eobs.sum())}")
    assert 0.25 <= cens_rate <= 0.45, f"censoring rate {cens_rate:.3f} outside the target band"

    # -- 0. person-period expansion is correct ----------------------------------------
    ex = person_period_expansion([2.0, 5.0, 0.4], [1, 0, 1], horizon=3)
    assert ex.n_periods.tolist() == [2, 3, 1]
    assert ex.period.tolist() == [0, 1, 0, 1, 2, 0]
    assert ex.event.tolist() == [0, 1, 0, 0, 0, 1]
    big = person_period_expansion(etime, eobs, HORIZON)
    assert big.n_rows == int(big.n_periods.sum())
    assert int(big.event.sum()) == int(((eobs == 1) & (np.ceil(etime - 1e-9) <= HORIZON)).sum())

    # -- 1. C-index sanity: perfect / reversed / random --------------------------------
    c_perfect = concordance_index(etime, -latent_T, eobs)
    c_reversed = concordance_index(etime, latent_T, eobs)
    c_random = concordance_index(etime, as_rng(0).normal(size=N), eobs)
    c_alltied = concordance_index(etime, np.ones(N), eobs)
    print(f"  C-index sanity: perfect={c_perfect:.4f}  reversed={c_reversed:.4f}  "
          f"random={c_random:.4f}  all-tied={c_alltied:.4f}")
    assert abs(c_perfect - 1.0) < 1e-9, c_perfect
    assert abs(c_reversed - 0.0) < 1e-9, c_reversed
    assert abs(c_random - 0.5) < 0.03, c_random
    assert abs(c_alltied - 0.5) < 1e-9, c_alltied

    # -- 1b. Reference cross-checks against independent implementations -----------------
    #    These are the checks that make the metrics believable rather than merely plausible.
    risk_probe = -latent_T + as_rng(2).normal(scale=3.0, size=N)
    refs: list[str] = []
    if HAS_LIFELINES:
        try:
            from lifelines.utils import concordance_index as _ll_cindex

            mine = concordance_index(etime, risk_probe, eobs)
            theirs = float(_ll_cindex(etime, -risk_probe, eobs))
            assert abs(mine - theirs) < 1e-9, (mine, theirs)
            refs.append(f"C-index == lifelines ({mine:.6f}, diff {abs(mine - theirs):.1e})")
        except ImportError:  # pragma: no cover - optional path
            refs.append("C-index vs lifelines: skipped")
    #    With no censoring the IPCW weights are all 1, so Graf's Brier must collapse to the
    #    plain squared error against 1{T > t} and Uno's AUC to the ordinary ROC AUC.
    unc_t, unc_d = latent_T, np.ones(N, dtype=np.int64)
    probe_S = np.clip(as_rng(3).random((N, 3)), 0.01, 0.99)
    probe_grid = np.array([2.0, 5.0, 8.0])
    graf = brier_score(unc_t, unc_d, probe_S, probe_grid)["brier"].to_numpy()
    plain = np.array(
        [float((((unc_t > tau).astype(float) - probe_S[:, j]) ** 2).mean())
         for j, tau in enumerate(probe_grid)]
    )
    assert np.allclose(graf, plain, atol=1e-12), np.max(np.abs(graf - plain))
    refs.append(f"IPCW Brier == plain Brier when uncensored (max diff {np.abs(graf - plain).max():.1e})")

    from sklearn.metrics import roc_auc_score

    uno = time_dependent_auc(unc_t, unc_d, -probe_S[:, 1], probe_grid)["auc"].to_numpy()
    roc = np.array(
        [float(roc_auc_score((unc_t <= tau).astype(int), -probe_S[:, 1])) for tau in probe_grid]
    )
    assert np.allclose(uno, roc, atol=1e-12), np.max(np.abs(uno - roc))
    refs.append(f"Uno AUC == sklearn ROC AUC when uncensored (max diff {np.abs(uno - roc).max():.1e})")
    for line in refs:
        print(f"    ref-check: {line}")

    # -- 2. Cox: known-answer coefficient recovery -------------------------------------
    cox = CoxPHModel().fit(X, etime, eobs)
    coef_err = float(np.max(np.abs(cox.coef_ - beta_true)))
    print("  Cox coefficients")
    print("    truth    ", np.array2string(beta_true, precision=3, suppress_small=True))
    print("    recovered", np.array2string(cox.coef_, precision=3, suppress_small=True))
    print(f"    max |error| = {coef_err:.4f}  (iters={cox.n_iter_}, converged={cox.converged_})")
    assert coef_err < 0.10, f"Cox coefficients off by {coef_err:.4f} (> 0.10)"
    assert cox.converged_
    assert list(cox.baseline_cumhaz_.columns) == ["time", "baseline_hazard", "baseline_cumhaz"]
    assert np.all(np.diff(cox.baseline_cumhaz_["baseline_cumhaz"].to_numpy()) >= -1e-12)

    lifelines_note = "not installed"

    if HAS_LIFELINES:
        try:
            from lifelines import CoxPHFitter

            frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(P)])
            frame["T"], frame["E"] = etime, eobs
            cph = CoxPHFitter(penalizer=0.0).fit(frame, duration_col="T", event_col="E")
            ll_coef = cph.params_.to_numpy()
            gap = float(np.max(np.abs(ll_coef - cox.coef_)))
            print("    lifelines", np.array2string(ll_coef, precision=3, suppress_small=True))
            print(f"    max |ours - lifelines| = {gap:.5f}")
            assert gap < 0.02, f"disagreement with lifelines: {gap:.4f}"
            lifelines_note = f"agrees to {gap:.5f}"
        except Exception as exc:  # pragma: no cover - optional path
            lifelines_note = f"skipped ({type(exc).__name__})"
    print(f"    lifelines cross-check: {lifelines_note}")

    # -- 2b. The analytic Efron gradient must match a NUMERICAL gradient -----------------
    #    L-BFGS-B trusts `jac`; a wrong gradient converges to the wrong point and still
    #    reports success. check_grad is the only thing that actually pins it down.
    from scipy.optimize import check_grad

    g_rng = as_rng(5)
    Xg = X[:800]
    Xg = (Xg - Xg.mean(axis=0)) / Xg.std(axis=0)
    for g_label, g_t in (
        ("continuous", etime[:800]),
        ("monthly ties", np.maximum(np.ceil(etime[:800]), 1.0)),
    ):
        g_blk = _build_efron_blocks(Xg, np.asarray(g_t, dtype=np.float64), eobs[:800])[0]
        g_obj = lambda b, _b=g_blk: _efron_objective(b, _b, 1e-3, 800)[0]  # noqa: E731
        g_jac = lambda b, _b=g_blk: _efron_objective(b, _b, 1e-3, 800)[1]  # noqa: E731
        g_rel = 0.0
        for _ in range(3):
            b0 = g_rng.normal(size=P) * 0.6
            g_rel = max(
                g_rel,
                check_grad(g_obj, g_jac, b0, epsilon=1e-6)
                / max(float(np.linalg.norm(g_jac(b0))), 1e-12),
            )
        print(f"    grad-check {g_label:12s} max tie block {int(g_blk.group_size.max()):4d}"
              f"   rel |analytic - numeric| = {g_rel:.1e}")
        assert g_rel < 1e-4, f"Efron gradient is wrong ({g_label}): rel err {g_rel:.2e}"

    # -- 2c. The Efron TIE path, which continuous times never reach ----------------------
    #    The simulated times above are continuous, so every tie block has size 1 and the
    #    Efron correction is a no-op. PRISM's panel is monthly, so the tied regime is the
    #    one that actually runs: round to whole months and re-recover the same truth.
    t_tied = np.minimum(np.maximum(np.ceil(etime), 1.0), 24.0)
    cox_tied = CoxPHModel().fit(X, t_tied, eobs)
    tie_block = int(pd.Series(t_tied[eobs == 1]).value_counts().max())
    tied_err = float(np.max(np.abs(cox_tied.coef_ - beta_true)))
    print(f"    Efron ties: largest tie block {tie_block}, max |coef - truth| = {tied_err:.4f}")
    assert tie_block > 50, "the tied-data check is not actually tied"
    assert tied_err < 0.12, f"Cox on monthly-tied data: coefficient error {tied_err:.4f}"
    assert cox_tied.converged_

    # -- 3. Fit the three models --------------------------------------------------------
    models: dict[str, Any] = {}
    dth_sk = DiscreteTimeHazardModel(
        horizon=HORIZON, backend="sklearn", random_state=7
    ).fit(X, etime, eobs)
    models["DiscreteTimeHazard[sklearn]"] = dth_sk
    if HAS_TORCH:
        models["DiscreteTimeHazard[torch]"] = DiscreteTimeHazardModel(
            horizon=HORIZON, hidden=(32, 16), epochs=60, lr=5e-3, batch_size=1024,
            backend="torch", random_state=7, torch_threads=4,
        ).fit(X, etime, eobs)
    models["CoxPH"] = cox
    models["RandomSurvivalForestLite"] = RandomSurvivalForestLite(
        n_estimators=200, max_depth=6, min_samples_leaf=20, random_state=7, n_jobs=-1
    ).fit(X, etime, eobs)

    # -- 3b. sample_weight is actually USED, not quietly dropped -------------------------
    #    The classic failure is an estimator that accepts sample_weight and ignores it.
    #    Weighting a row by 2 must be (near) identical to duplicating it, and must be
    #    clearly different from not weighting at all.
    _half = N // 2
    sw = np.ones(N)
    sw[:_half] = 2.0
    _kw = {"horizon": 6, "backend": "sklearn", "random_state": 0}
    m_w = DiscreteTimeHazardModel(**_kw).fit(X, etime, eobs, sample_weight=sw)
    m_d = DiscreteTimeHazardModel(**_kw).fit(
        np.vstack([X, X[:_half]]),
        np.concatenate([etime, etime[:_half]]),
        np.concatenate([eobs, eobs[:_half]]),
    )
    m_u = DiscreteTimeHazardModel(**_kw).fit(X, etime, eobs)
    d_wd = float(np.max(np.abs(m_w.predict_rmst(X) - m_d.predict_rmst(X))))
    d_wu = float(np.max(np.abs(m_w.predict_rmst(X) - m_u.predict_rmst(X))))
    print(f"  sample_weight: |w=2 vs duplicated rows| = {d_wd:.2e}   "
          f"|w=2 vs unweighted| = {d_wu:.2e}")
    assert d_wd < 5e-3, f"sample_weight does not reproduce row duplication ({d_wd:.2e})"
    assert d_wu > 1e-2, "sample_weight is silently ignored (weighted == unweighted)"

    # -- 4. Curves, RMST convention, metrics --------------------------------------------
    km_grid, km_surv = kaplan_meier(etime, eobs, times=GRID)
    S_baseline = np.tile(km_surv, (N, 1))
    ibs_baseline = integrated_brier_score(etime, eobs, S_baseline, GRID)
    ibs_half = integrated_brier_score(etime, eobs, np.full((N, HORIZON), 0.5), GRID)

    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        if isinstance(model, DiscreteTimeHazardModel):
            S = model.predict_survival(X)
            rmst = model.predict_rmst(X)
            assert np.allclose(rmst, S.sum(axis=1)), f"{name}: RMST != sum of S(k)"
            assert np.allclose(model.predict_churn_prob(X, 1), 1.0 - S[:, 0])
            assert np.allclose(model.predict(X), rmst)
            assert model.predict_hazard(X).shape == (N, HORIZON)
        else:
            S = model.predict_survival(X, GRID)
            rmst = model.predict_rmst(X, float(HORIZON))
            assert np.allclose(rmst, S.sum(axis=1)), f"{name}: RMST != sum of S(k)"
        _check_curve(name, S)
        assert S.shape == (N, HORIZON)
        assert np.all(rmst >= 0) and np.all(rmst <= HORIZON)

        c_idx = concordance_index(etime, -rmst, eobs)
        ibs = integrated_brier_score(etime, eobs, S, GRID)
        rows.append(
            {
                "model": name,
                "c_index": c_idx,
                "ibs": ibs,
                "rmst_mean": float(rmst.mean()),
                "fit_s": float(getattr(model, "fit_seconds_", float("nan"))),
            }
        )
        assert c_idx > 0.60, f"{name}: C-index {c_idx:.4f} <= 0.60 on data with real signal"
        assert ibs < ibs_baseline, f"{name}: IBS {ibs:.4f} >= KM baseline {ibs_baseline:.4f}"

    rows.append({"model": "baseline: marginal KM", "c_index": 0.5, "ibs": ibs_baseline,
                 "rmst_mean": float(S_baseline.sum(axis=1)[0]), "fit_s": 0.0})
    rows.append({"model": "baseline: constant 0.5", "c_index": 0.5, "ibs": ibs_half,
                 "rmst_mean": 6.0, "fit_s": 0.0})
    table = pd.DataFrame(rows)
    print()
    print(
        table.to_string(
            index=False,
            formatters={
                "c_index": "{:.4f}".format,
                "ibs": "{:.4f}".format,
                "rmst_mean": "{:.3f}".format,
                "fit_s": "{:.2f}".format,
            },
        )
    )

    assert abs(ibs_half - 0.25) < 1e-9, ibs_half
    assert ibs_baseline < ibs_half

    # -- 4b. The FRACTIONAL-horizon RMST convention is the same in all three models ------
    #    Every model must compute sum_{k=1..floor(H)} S(k) + frac * S(H) with S(H) read off
    #    its OWN step function. Getting this wrong (e.g. using S(floor(H)+1) for the
    #    discrete model) makes the three estimators silently non-differenceable, which is
    #    exactly what prism.causal.survival_uplift does to them.
    S_dth = dth_sk.predict_survival(X)
    for H_frac in (6.5, 9.25):
        w_h = int(math.floor(H_frac))
        f_h = H_frac - w_h
        ref_dth = S_dth[:, :w_h].sum(axis=1) + f_h * S_dth[:, w_h - 1]
        assert np.allclose(dth_sk.predict_rmst(X, H_frac), ref_dth), (
            f"DiscreteTimeHazardModel fractional RMST at H={H_frac} breaks the "
            "sum_{k<=floor(H)} S(k) + frac * S(H) convention"
        )
        for nm_, mdl_ in (("CoxPH", cox), ("RSF", models["RandomSurvivalForestLite"])):
            g_h = np.append(np.arange(1.0, w_h + 1.0), H_frac)
            S_h = mdl_.predict_survival(X, g_h)
            ref_o = S_h[:, :w_h].sum(axis=1) + f_h * S_h[:, -1]
            assert np.allclose(mdl_.predict_rmst(X, H_frac), ref_o), (
                f"{nm_} fractional RMST at H={H_frac} breaks the shared convention"
            )
        # Extending the horizon past floor(H) can only add survival time.
        for call_ in (
            lambda h: dth_sk.predict_rmst(X, h),
            lambda h: cox.predict_rmst(X, h),
        ):
            assert np.all(call_(float(w_h)) <= call_(H_frac) + 1e-12)
            assert np.all(call_(H_frac) <= float(H_frac) + 1e-9)
    #    NOTE: RMST(k + f) <= RMST(k + 1) is deliberately NOT asserted. Under the
    #    right-Riemann convention the step from k+f to k+1 is S(k+1) - f*S(k), which is
    #    negative whenever the period hazard exceeds 1 - f. That is a real (small, ~4e-4
    #    here) property of the documented convention, not a defect: the same bias applies
    #    to every arm and cancels in the treatment contrasts PRISM actually spends on.
    #    What IS guaranteed is monotonicity on the integer grid, since RMST(k+1) = RMST(k)
    #    + S(k+1) with S >= 0 -- so that is what gets asserted.
    for call_ in (
        lambda h: dth_sk.predict_rmst(X, h),
        lambda h: cox.predict_rmst(X, h),
        lambda h: models["RandomSurvivalForestLite"].predict_rmst(X, h),
    ):
        prev = call_(1.0)
        for k in range(2, HORIZON + 1):
            cur = call_(float(k))
            assert np.all(prev <= cur + 1e-12), f"RMST not monotone on the integer grid at k={k}"
            prev = cur
    print("  fractional-horizon RMST convention consistent across all 3 models "
          "(H=6.5, 9.25); RMST monotone on the integer grid")

    # -- 4c. n_bins must not silently wrap the uint8 bin codes ---------------------------
    try:
        RandomSurvivalForestLite(n_estimators=2, n_bins=300).fit(X[:300], etime[:300], eobs[:300])
    except ValueError:
        pass
    else:  # pragma: no cover - guard regression
        raise AssertionError("n_bins > 256 must be rejected; uint8 bin codes would wrap")

    # -- 5. Time-dependent AUC and calibration -------------------------------------------
    # The DGP is exponential proportional hazards, so CoxPH is the *correctly specified*
    # likelihood model here and is the honest reference for a calibration assertion. The
    # forest is reported alongside it to show the shrinkage a depth-limited tree ensemble
    # pays for its better discrimination -- a real property, not a defect.
    best = str(table.iloc[int(np.argmin(table["ibs"].to_numpy()[: len(models)]))]["model"])
    S_ref = cox.predict_survival(X, GRID)
    auc_tab = time_dependent_auc(etime, eobs, -S_ref[:, 5], GRID[2:9])
    mean_auc = float(np.nanmean(auc_tab["auc"].to_numpy()))
    print(f"\n  best-IBS model: {best}   |   calibration reference: CoxPH (well specified)")
    print(f"  time-dependent AUC (t=3..9): min={auc_tab['auc'].min():.4f} "
          f"mean={mean_auc:.4f} max={auc_tab['auc'].max():.4f}")
    assert mean_auc > 0.65, mean_auc
    assert auc_tab.shape[0] == 7
    assert list(auc_tab.columns) == ["time", "auc", "n_cases", "n_controls"]

    bs_tab = brier_score(etime, eobs, S_ref, GRID)
    assert list(bs_tab.columns) == ["time", "brier", "n_events", "n_at_risk"]
    assert bs_tab["brier"].between(0.0, 1.0).all()

    cal = calibration_curve_survival(etime, eobs, S_ref, t=6.0, n_bins=10, times=GRID)
    assert list(cal.columns) == ["bin", "n", "mean_predicted", "observed_km", "ci_low", "ci_high"]
    assert int(cal["n"].sum()) == N
    assert (cal["ci_low"] <= cal["observed_km"] + 1e-9).all()
    assert (cal["observed_km"] <= cal["ci_high"] + 1e-9).all()
    covered = int(
        ((cal["mean_predicted"] >= cal["ci_low"]) & (cal["mean_predicted"] <= cal["ci_high"])).sum()
    )
    cal_mae = float(np.mean(np.abs(cal["mean_predicted"] - cal["observed_km"])))
    print(f"  calibration @ t=6: {covered}/{len(cal)} bins inside the KM CI, MAE={cal_mae:.4f}")
    print(
        cal.to_string(
            index=False,
            formatters=dict.fromkeys(("mean_predicted", "observed_km", "ci_low", "ci_high"), "{:.4f}".format),
        )
    )
    assert cal_mae < 0.03, cal_mae
    assert covered >= 5, f"only {covered}/{len(cal)} calibration bins inside the KM CI"
    assert cal["mean_predicted"].is_monotonic_increasing

    rsf_cal = calibration_curve_survival(
        etime, eobs, models["RandomSurvivalForestLite"].predict_survival(X, GRID),
        t=6.0, n_bins=10, times=GRID,
    )
    rsf_mae = float(np.mean(np.abs(rsf_cal["mean_predicted"] - rsf_cal["observed_km"])))
    print(f"  (contrast) RandomSurvivalForestLite calibration MAE @ t=6: {rsf_mae:.4f} "
          f"-- depth-limited leaves shrink the curve toward the marginal")

    # -- 6. Determinism -------------------------------------------------------------------
    rsf_a = RandomSurvivalForestLite(n_estimators=25, random_state=3, n_jobs=-1).fit(X, etime, eobs)
    rsf_b = RandomSurvivalForestLite(n_estimators=25, random_state=3, n_jobs=1).fit(X, etime, eobs)
    assert np.array_equal(rsf_a.predict_rmst(X, 12.0), rsf_b.predict_rmst(X, 12.0)), (
        "RandomSurvivalForestLite is not seed-deterministic across n_jobs"
    )
    dth_a = DiscreteTimeHazardModel(horizon=6, backend="sklearn", random_state=1).fit(X, etime, eobs)
    dth_b = DiscreteTimeHazardModel(horizon=6, backend="sklearn", random_state=1).fit(X, etime, eobs)
    assert np.array_equal(dth_a.predict_rmst(X), dth_b.predict_rmst(X))
    print("  determinism: identical predictions across repeated fits and n_jobs  OK")

    elapsed = _time.perf_counter() - t_start
    print(f"\nsurvival.py OK  ({elapsed:.1f}s total, budget 60s)")
    if elapsed > 60.0:
        print("WARNING: smoke test exceeded its 60 s budget", file=sys.stderr)
        raise SystemExit(1)
