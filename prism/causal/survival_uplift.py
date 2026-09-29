"""Causal uplift on *time-to-event* outcomes -- the estimator no library ships.

Why this module exists, commercially
------------------------------------
A retention offer that **delays** churn by three months but does not **prevent** it has
*exactly zero* effect on a 12-month binary churn flag: the customer is gone at month 12
either way, so a conventional uplift model scores the offer at 0.00 and the budget goes
elsewhere. Yet three extra months of gross margin is real money that lands in the P&L.
Restricted mean survival time (RMST) sees that money; binary uplift cannot, by construction.

This module therefore estimates conditional treatment effects on

* **RMST** -- expected months retained inside a horizon ``H`` (an executive-legible unit), and
* **discounted lifetime value** -- the same curve integrated against margin and a discount rate,

under **right-censoring** and with **multiple competing arms**. It is the bridge between
survival analysis and the budget optimiser in :mod:`prism.decision`.

Estimand
--------
For arm ``a`` against control ``0``, with ``S_a(k | x) = P(T(a) > k | X = x)``:

.. math::

    \\tau^{rmst}_a(x)  &= \\sum_{k=1}^{H} \\bigl[ S_a(k|x) - S_0(k|x) \\bigr]
                          \\qquad \\text{(months)} \\\\
    \\tau^{value}_a(x) &= m(x)\\sum_{k=1}^{H} \\bigl[ S_a(k|x) - S_0(k|x) \\bigr](1+d)^{-k}
                          \\qquad \\text{(currency)} \\\\
    \\tau^{churn}_a(x, c) &= P(\\text{churn} \\le c \\mid a) - P(\\text{churn} \\le c \\mid 0)
                          = S_0(c|x) - S_a(c|x)

``m(x)`` is the customer's monthly margin, ``d`` the monthly discount rate. Note that ``m(x)``
factors straight out of the sum, so the module estimates the **margin-free discounted
retention effect** ``\\tau^{disc}_a(x) = \\sum_k [S_a - S_0](1+d)^{-k}`` and multiplies by the
margin at prediction time. That keeps the currency conversion honest when the margin supplied
at scoring time differs from the one seen in training.

RMST convention (read this before comparing numbers)
----------------------------------------------------
:mod:`prism.models.survival` defines ``predict_rmst`` as the **right-Riemann sum**
``RMST(H) = sum_{k=1..H} S(k)`` with ``S(k) = prod_{j<=k} (1 - h(j))``, and
:mod:`prism.data.dgp` computes ``gt_rmst_a`` the same way. **This module adopts exactly that
convention**, with ``k`` starting at 1, so its output is directly comparable to ``gt_rmst_*``
and to ``DiscreteTimeHazardModel.predict_rmst``.

The consequence matters for the doubly-robust step and is the single easiest thing to get
wrong here. On a discrete period grid,

.. math::

    \\sum_{k=1}^{H} S(k) \\;=\\; \\sum_{k=1}^{H} P(T > k) \\;=\\; E\\bigl[\\min(T - 1,\\, H)\\bigr],

i.e. the expected number of periods **fully survived**, *not* ``E[min(T, H)]``. The two differ
by ``P(T <= H)``, which is arm-dependent, so plugging the textbook ``min(T, H)`` into the
IPCW-AIPW correction while the plug-in uses ``sum_k S(k)`` biases every contrast by exactly
``-tau^churn_a`` (about 0.16 months per 16 pp of churn effect -- 10% of a typical effect here).
This module therefore uses the *matched* observable

.. math::  R_i \\;=\\; \\min\\bigl(\\lceil T_i \\rceil - \\delta_i,\\; H\\bigr),
           \\qquad E[R(a) \\mid X] = \\mathrm{RMST}_a(X),

see :func:`restricted_survival_outcome`. ``lceil . rceil`` matches
:func:`prism.models.survival.person_period_expansion`, which puts a subject observed for
``t`` months on periods ``1 .. ceil(t)``.

Pipeline
--------
1. **Censoring model** ``G(t | x) = P(C > t | x)`` -- :class:`CensoringModel`, either a
   Kaplan-Meier estimate (``"km"``) or a covariate-dependent one (``"cox"`` /
   ``"covariate"``). ``G`` is floored (``min_censor_prob``, default 0.05) and the number of
   floored rows is recorded and logged: a weight of ``1/0.001`` is not evidence.
2. **Per-arm discrete-time hazards** -- :class:`prism.models.survival.DiscreteTimeHazardModel`,
   reused, not reimplemented. The **control arm is fitted once** and reused for every contrast
   (``SPEC_PERF`` section 7); the per-arm fits run under ``joblib.Parallel(n_jobs=min(K, 4))``.
3. **Counterfactual survival** ``S_a(k|x) = prod_{j<=k} (1 - h_a(j|x))``, forced monotone
   non-increasing and strictly inside ``(0, 1)``, served in chunks
   (:meth:`CausalSurvivalUplift.predict_survival_curves`, ``chunk_size=50_000``).
4. **Contrasts** as above (plug-in).
5. **Doubly-robust correction** -- the IPCW-AIPW pseudo-outcome

   .. math::

       \\psi_i^{(a)} = \\bigl[\\mathrm{RMST}_a(X_i) - \\mathrm{RMST}_0(X_i)\\bigr]
         + \\frac{1\\{A_i = a\\}}{e_a(X_i)}\\,\\frac{\\Delta_i^H}{G(T_i^H-)}
           \\bigl(R_i - \\mathrm{RMST}_a(X_i)\\bigr)
         - \\frac{1\\{A_i = 0\\}}{e_0(X_i)}\\,\\frac{\\Delta_i^H}{G(T_i^H-)}
           \\bigl(R_i - \\mathrm{RMST}_0(X_i)\\bigr)

   regressed on ``X`` with K-fold cross-fitting of every nuisance. ``E[psi | X] = tau_a(X)``
   if **either** the hazard models **or** the propensity model is correct -- neither has to be.
   Steps 2-4 alone are a plug-in estimator, consistent only if the hazard model is right; the
   correction buys back a second chance. The plug-in stays available through
   ``predict_cate(X, plugin=True)`` so the two can be compared, which the smoke test does.

Assumptions (all of them, plainly)
----------------------------------
* consistency / SUTVA, and no interference between customers;
* positivity: ``e_a(x)`` bounded away from 0 and 1 -- measured, not assumed, via
  :func:`prism.models.propensity.overlap_diagnostics`;
* unconfoundedness given ``X``;
* **non-informative censoring given ``X`` and ``A``** -- with ``censoring_model="km"`` the
  stronger *marginal* independence is required, which is why the covariate-dependent option
  exists and why the smoke test measures the difference.

What this estimator does **not** identify
-----------------------------------------
A **static, single-decision-point** contrast: the effect of the arm recorded at the decision
row, with follow-up as it actually happened afterwards. It is *not* the effect of
**sustaining** arm ``a`` for all ``H`` periods. On PRISM's own panel those are different
numbers, and measurably so -- a customer is re-randomised into a new, confounded arm every
month, so on a 24-period simulation only about 6% of customers keep their period-0 arm for the
whole horizon. ``prism.data.dgp``'s ``gt_rmst_*`` / ``gt_tau_rmst_*`` are defined as the
*sustained*-arm contrast (its own docstring says so), so scoring this estimator's PEHE against
``gt_tau_rmst_*`` on a raw panel decision point compares two different estimands and will not
flatter either of them. Identifying a sustained regime under time-varying confounding -- and
with treatment feeding back into ``discount_depth_hist`` -- needs a marginal structural model
or fitted-Q iteration, which ``docs/METHODOLOGY.md`` section 8.3 names as a known limitation of
the project rather than something this module quietly papers over.

What the estimator *is* doing can be checked directly, and is: fitted ``S_a(k|x)`` averaged
within an arm matches that arm's empirical Kaplan-Meier to within 0.004 in probability at every
``k`` on the simulator's panel. The module's own ``__main__`` therefore validates against a
design where the estimand is identified (static arms, analytic counterfactual curves), which is
the only setting in which a PEHE number means what it says.

See ``docs/METHODOLOGY.md`` section 4.3 and ``SPEC.md`` section 4.3.
"""

from __future__ import annotations

import inspect
import time as _time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold

from prism.data.schema import ARM_NAMES, N_ARMS
from prism.models.propensity import (
    PropensityModel,
    clip_renormalize,
    effective_sample_size,
    overlap_diagnostics,
)
from prism.models.survival import (
    CoxPHModel,
    DiscreteTimeHazardModel,
    kaplan_meier,
    step_eval,
)
from prism.utils.logging import get_logger
from prism.utils.optional import best_gbm
from prism.utils.seeds import as_rng, spawn_rngs

__all__ = [
    "CausalSurvivalUplift",
    "ipcw_weights",
    "rmst_pseudo_outcome",
    "CensoringModel",
    "RestrictedOutcome",
    "restricted_survival_outcome",
    "discount_factors",
]

LOGGER = get_logger("causal.survival_uplift")

#: Survival probabilities are kept strictly inside this open interval.
_S_LO: float = 1e-12
_S_HI: float = 1.0 - 1e-12
#: Default floor on ``G(t|x)``; ``1 / 0.05 = 20`` is the largest IPCW weight allowed by default.
_DEFAULT_MIN_CENSOR_PROB: float = 0.05
#: Default clip on the generalised propensity score, matching ``PropensityModel``.
_DEFAULT_PROP_CLIP: tuple[float, float] = (0.01, 0.99)
#: ``ceil`` tolerance, identical to ``prism.models.survival.person_period_expansion``.
_CEIL_EPS: float = 1e-9


# ======================================================================================
# Base-class resolution
# ======================================================================================
def _resolve_base() -> type:
    """Return ``prism.causal.learners.BaseCATELearner`` **iff** it is a pure interface.

    ``SPEC.md`` section 4.3 writes ``class CausalSurvivalUplift(BaseCATELearner)``, and the
    shared contract (``fit`` / ``predict_cate`` / ``predict`` / ``n_arms``) is honoured here
    exactly. Whether that contract arrives by inheritance or by declaration is an
    implementation detail, and inheriting is only safe when the base is a *protocol*:

    * ``prism.causal.learners.BaseCATELearner`` is not one -- it is a meta-learner scaffold
      whose constructor takes ``base_outcome`` / ``base_effect`` / ``base_propensity``, whose
      ``fit`` has the regression signature ``fit(X, w, y, ...)``, and whose subclasses supply
      a ``_fit`` hook. A survival estimator takes ``(event_time, event_observed)`` and fits
      hazards, so none of that scaffolding applies.
    * more concretely, that class overrides ``_get_param_names`` to *merge in its own*
      constructor parameters. Inheriting from it would make ``get_params`` ask this estimator
      for a ``base_effect`` it has no reason to own, and ``sklearn.base.clone`` -- which
      :mod:`prism.causal.refute` calls for every refutation simulation -- would raise
      ``AttributeError``. A broken ``clone`` is a much worse bug than a missing base class.

    So the import is attempted, checked, and declined when the base is not a pure interface;
    the local stand-in below then declares the identical contract. The check is a real check
    rather than a permanent ``return _LocalBaseCATELearner`` so that a future thin protocol in
    ``learners.py`` is picked up automatically.

    Returns
    -------
    type
        A base class exposing ``fit`` / ``predict_cate`` / ``predict`` / ``n_arms``.
    """
    try:  # pragma: no cover - depends on build order of sibling modules
        from prism.causal.learners import BaseCATELearner as _Imported  # type: ignore

        if not isinstance(_Imported, type):
            return _LocalBaseCATELearner
        unmet = set(getattr(_Imported, "__abstractmethods__", frozenset()))
        unmet -= {"fit", "predict", "predict_cate"}
        own_init = _Imported.__init__ is not object.__init__ and bool(
            [
                q
                for q in inspect.signature(_Imported.__init__).parameters.values()
                if q.name != "self" and q.kind is not q.VAR_KEYWORD
            ]
        )
        overrides_params = "_get_param_names" in vars(_Imported)
        if not unmet and not own_init and not overrides_params:
            return _Imported
    except Exception:  # pragma: no cover - the normal path when learners.py is absent
        pass
    return _LocalBaseCATELearner


class _LocalBaseCATELearner(BaseEstimator):
    """Local stand-in for the shared CATE interface (``SPEC.md`` section 4).

    Every learner in :mod:`prism.causal` exposes the same three entry points so the pipeline
    can loop over them uniformly to build the leaderboard:

    * ``fit(X, w, y, *, propensity=None, sample_weight=None) -> Self``
    * ``predict_cate(X) -> ndarray of shape (n, n_arms - 1)``
    * ``predict(X)`` -- an alias of ``predict_cate``
    * attribute ``n_arms``
    """

    n_arms: int = N_ARMS

    def predict_cate(self, X: Any) -> np.ndarray:  # pragma: no cover - interface only
        """Conditional average treatment effect per non-control arm."""
        raise NotImplementedError

    def predict(self, X: Any) -> np.ndarray:
        """Alias of :meth:`predict_cate`."""
        return self.predict_cate(X)


BaseCATELearner = _resolve_base()


# ======================================================================================
# Small helpers
# ======================================================================================
def _as_2d(X: Any, name: str = "X") -> np.ndarray:
    """Coerce ``X`` to a 2-D float64 array."""
    arr = np.asarray(X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D; got shape {arr.shape}")
    return arr


def _as_1d(v: Any, name: str, n: int | None = None) -> np.ndarray:
    """Coerce ``v`` to a 1-D float64 array, optionally checking its length."""
    arr = np.asarray(v.values if isinstance(v, (pd.Series, pd.DataFrame)) else v, dtype=np.float64).ravel()
    if n is not None and arr.size != n:
        raise ValueError(f"{name} must have length {n}; got {arr.size}")
    return arr


def _periods(event_time: Any) -> np.ndarray:
    """Map follow-up times onto the integer period grid, exactly as the expansion does.

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Follow-up time in periods (may be fractional).

    Returns
    -------
    ndarray of shape (n,)
        ``max(ceil(t - 1e-9), 1)``: the period index in which the event or the loss to
        follow-up occurred. The ``1e-9`` makes an exactly integral ``t = 3.0`` yield 3, not 4,
        matching :func:`prism.models.survival.person_period_expansion`.
    """
    return np.maximum(np.ceil(_as_1d(event_time, "event_time") - _CEIL_EPS), 1.0)


def discount_factors(horizon: int, discount_rate: float = 0.01) -> np.ndarray:
    """Per-period discount factors ``v_k = (1 + d)^{-k}`` for ``k = 1 .. H``.

    Parameters
    ----------
    horizon : int
        Number of periods ``H``.
    discount_rate : float, default 0.01
        Monthly discount rate ``d``; must exceed ``-1``.

    Returns
    -------
    ndarray of shape (horizon,)
        ``v_1 .. v_H``. ``k`` starts at 1, the same indexing as the survival curve, so
        ``S @ discount_factors(H, d)`` is the discounted restricted mean survival time.

    Examples
    --------
    >>> np.round(discount_factors(3, 0.01), 5).tolist()
    [0.9901, 0.9803, 0.97059]
    """
    H = int(horizon)
    if H < 1:
        raise ValueError(f"horizon must be >= 1, got {H}")
    d = float(discount_rate)
    if d <= -1.0:
        raise ValueError(f"discount_rate must be > -1, got {d}")
    return np.power(1.0 + d, -np.arange(1, H + 1, dtype=np.float64))


@dataclass(frozen=True)
class RestrictedOutcome:
    """Observable counterparts of the restricted survival estimands.

    Attributes
    ----------
    periods : ndarray of shape (n,)
        ``R_i = min(ceil(T_i) - delta_i, H)`` -- periods **fully survived**, restricted to
        the horizon. ``E[R(a) | X] = sum_{k=1..H} S_a(k | X)``, the module's RMST convention.
    discounted : ndarray of shape (n,)
        ``sum_{k=1..R_i} (1 + d)^{-k}`` -- the discounted analogue, whose conditional mean is
        ``sum_{k=1..H} S_a(k | X) (1 + d)^{-k}`` (margin-free "discounted retained time").
    complete : ndarray of shape (n,)
        ``1`` when ``R_i`` is actually observed: either the event was seen
        (``delta_i = 1``), or follow-up reached the horizon (``ceil(T_i) >= H``, so the
        subject is administratively complete rather than censored).
    follow_up : ndarray of shape (n,)
        ``T^H_i = min(ceil(T_i), H)``, the time at which the censoring probability must be
        evaluated.

    Notes
    -----
    The completeness rule is worth spelling out because it is the usual source of silent bias.
    ``R_i`` is observed iff ``C_i >= min(T^*_i, H)``. Split by what we see:

    * ``delta_i = 1``: the event beat the censoring, ``C_i >= T^*_i >= min(T^*_i, H)`` -- complete.
    * ``delta_i = 0`` and ``ceil(T_i) >= H``: censoring happened at or beyond the horizon, so
      survival through all ``H`` periods **is** observed -- complete, with ``R_i = H``.
    * ``delta_i = 0`` and ``ceil(T_i) < H``: genuinely incomplete, weight 0.

    Dropping the middle case (the "``delta`` only" reading) throws away every administratively
    censored customer and biases RMST downward by a wide margin.
    """

    periods: np.ndarray
    discounted: np.ndarray
    complete: np.ndarray
    follow_up: np.ndarray


def restricted_survival_outcome(
    event_time: Any,
    event_observed: Any,
    horizon: int,
    *,
    discount_rate: float = 0.0,
) -> RestrictedOutcome:
    """Build the observable RMST / discounted-RMST outcomes and the completeness indicator.

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Observed follow-up time in periods.
    event_observed : array-like of shape (n,)
        ``1`` if the event (churn) was observed, ``0`` if right-censored.
    horizon : int
        Restriction horizon ``H`` in periods.
    discount_rate : float, keyword-only, default 0.0
        Monthly discount rate used for :attr:`RestrictedOutcome.discounted`.

    Returns
    -------
    RestrictedOutcome
        See that class for the exact definitions and the completeness rule.

    Examples
    --------
    >>> o = restricted_survival_outcome([3.0, 5.0, 12.0], [1, 0, 0], horizon=6)
    >>> o.periods.tolist()          # event in period 3 -> 2 periods fully survived
    [2.0, 5.0, 6.0]
    >>> o.complete.tolist()         # the middle subject is censored inside the horizon
    [1.0, 0.0, 1.0]
    """
    H = int(horizon)
    if H < 1:
        raise ValueError(f"horizon must be >= 1, got {H}")
    k = _periods(event_time)
    d = _as_1d(event_observed, "event_observed", k.size)
    if not np.all(np.isin(d, (0.0, 1.0))):
        raise ValueError("event_observed must contain only 0 and 1")

    periods = np.clip(np.minimum(k - d, float(H)), 0.0, float(H))
    follow_up = np.minimum(k, float(H))
    complete = ((d == 1.0) | (k >= float(H))).astype(np.float64)

    if abs(float(discount_rate)) < 1e-15:
        discounted = periods.copy()
    else:
        cum = np.concatenate(([0.0], np.cumsum(discount_factors(H, discount_rate))))
        discounted = cum[periods.astype(np.int64)]
    return RestrictedOutcome(periods, discounted, complete, follow_up)


def _censoring_process(event_time: Any, event_observed: Any) -> tuple[np.ndarray, np.ndarray]:
    """Re-express the data as a survival problem for the **censoring** time ``C``.

    Returns ``(t_c, e_c)`` where ``e_c = 1 - delta`` and ``t_c = ceil(T) - delta``.

    Notes
    -----
    The ``- delta`` shift is the discrete-time tie correction and it is not cosmetic. With
    events and censorings both landing on integer periods, the convention that makes IPCW
    unbiased is "events first, censoring afterwards": a subject whose churn is *observed* in
    period ``k`` was never at risk of being censored in period ``k``, because the event had
    already removed them. Plain reverse Kaplan-Meier on ``(T, 1 - delta)`` leaves those
    subjects in the censoring risk set at ``k``, under-estimates the censoring hazard by a
    factor of roughly ``1 - h(k)``, and inflates ``G`` -- which biased the RMST estimate by
    about -2% on this project's simulator, entirely silently. Shifting their censoring
    follow-up back to ``k - 1`` removes it.

    This is why the module does not call
    :func:`prism.models.survival.censoring_survival` (a continuous-time convention, correct
    for Graf's Brier score where ties are a measure-zero nuisance) and instead applies
    :func:`prism.models.survival.kaplan_meier` to the shifted process.
    """
    k = _periods(event_time)
    d = _as_1d(event_observed, "event_observed", k.size)
    return np.maximum(k - d, 0.0), 1.0 - d


# ======================================================================================
# Censoring model  G(t | x) = P(C > t | X = x)
# ======================================================================================
class CensoringModel(BaseEstimator):
    """Estimator of the censoring survival function ``G(t | x) = P(C > t | X = x)``.

    IPCW re-weights the customers whose outcome we *did* get to see by the inverse of the
    probability that they would be seen, which undoes the selection censoring induces.
    ``1/G`` is therefore the whole ball game, and a small ``G`` is where the method dies: a
    single row with ``G = 0.001`` contributes a weight of 1000 and silently becomes the
    estimate. This class floors ``G`` at ``min_prob`` and **records how many rows were
    floored**, because a clipped weight that is never reported is a lie of omission.

    Parameters
    ----------
    model : {"km", "cox", "covariate"}, default "km"
        ``"km"`` -- marginal Kaplan-Meier on the (tie-shifted) censoring process; valid when
        censoring is independent of ``X``.
        ``"covariate"`` -- a :class:`~prism.models.survival.DiscreteTimeHazardModel` fitted to
        the censoring indicator, giving ``G(k | x)`` on the same period grid as the outcome
        hazards. This is the right default when follow-up depends on the customer.
        ``"cox"`` -- a :class:`~prism.models.survival.CoxPHModel` on the censoring process
        (Breslow baseline), for when a proportional-hazards summary is wanted.
    horizon : int, default 12
        Period grid ``1 .. H``.
    min_prob : float, default 0.05
        Floor on ``G``; caps every IPCW weight at ``1 / min_prob``.
    random_state : int or numpy.random.Generator or None, default None
        Seed forwarded to the covariate model.
    hazard_kwargs : dict, optional
        Extra keyword arguments for the ``"covariate"`` backend.

    Attributes
    ----------
    n_clipped_ : int
        Rows whose ``G(T-)`` hit the floor during :meth:`survival_before` on the training data.
    clip_rate_ : float
        ``n_clipped_ / n``.
    min_g_ : float
        Smallest unclipped ``G(T-)`` seen on the training data -- the honest diagnostic.

    Notes
    -----
    Every lookup returns the **left limit** ``G(t-) = P(C >= t)``, never ``G(t) = P(C > t)``.
    See :func:`rmst_pseudo_outcome` for why that distinction is the classic bug.
    """

    def __init__(
        self,
        model: str = "km",
        horizon: int = 12,
        min_prob: float = _DEFAULT_MIN_CENSOR_PROB,
        random_state: int | np.random.Generator | None = None,
        *,
        hazard_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.min_prob = min_prob
        self.random_state = random_state
        self.hazard_kwargs = hazard_kwargs

    # -- fit -------------------------------------------------------------------------
    def fit(self, event_time: Any, event_observed: Any, X: Any | None = None) -> CensoringModel:
        """Fit the censoring distribution.

        Parameters
        ----------
        event_time : array-like of shape (n,)
            Observed follow-up times.
        event_observed : array-like of shape (n,)
            ``1`` for an observed event, ``0`` for right-censored.
        X : array-like of shape (n, p), optional
            Covariates; required for ``model in {"cox", "covariate"}``.

        Returns
        -------
        CensoringModel
            ``self``, fitted.
        """
        mode = str(self.model).lower()
        if mode not in {"km", "cox", "covariate"}:
            raise ValueError(f"model must be one of km/cox/covariate, got {self.model!r}")
        H = int(self.horizon)
        t_c, e_c = _censoring_process(event_time, event_observed)
        keep = t_c >= 1.0
        self.mode_ = mode
        self.n_train_ = int(t_c.size)
        self.n_clipped_ = 0
        self.clip_rate_ = 0.0
        self.min_g_ = 1.0
        self.censoring_rate_ = float(np.mean(e_c))

        if keep.sum() < 2 or e_c[keep].sum() < 1:
            # No censoring information at all: G == 1 everywhere. Degrade, never crash.
            LOGGER.warning("CensoringModel: no usable censoring events; falling back to G(t) == 1")
            self.mode_ = "none"
            return self

        if mode == "km":
            grid, surv = kaplan_meier(t_c[keep], e_c[keep])
            self._grid_ = np.asarray(grid, dtype=np.float64)
            self._surv_ = np.asarray(surv, dtype=np.float64)
            return self

        if X is None:
            raise ValueError(f"model={mode!r} needs covariates; pass X to fit()")
        Xa = _as_2d(X)
        if Xa.shape[0] != t_c.size:
            raise ValueError(f"X has {Xa.shape[0]} rows but event_time has {t_c.size}")

        seed = int(as_rng(self.random_state).integers(0, 2**31 - 1))
        if mode == "covariate":
            kw: dict[str, Any] = dict(horizon=H, backend="sklearn", random_state=seed)
            kw.update(self.hazard_kwargs or {})
            self._model_ = DiscreteTimeHazardModel(**kw).fit(Xa[keep], t_c[keep], e_c[keep])
        else:
            self._model_ = CoxPHModel().fit(Xa[keep], t_c[keep], e_c[keep])
        return self

    # -- predict ---------------------------------------------------------------------
    def _raw_before(self, times: np.ndarray, X: Any | None) -> np.ndarray:
        """Unclipped ``G(t-)`` for each row."""
        if self.mode_ == "none":
            return np.ones(times.size, dtype=np.float64)
        if self.mode_ == "km":
            return step_eval(self._grid_, self._surv_, times, left_limit=True)

        Xa = _as_2d(X if X is not None else np.zeros((times.size, 1)))
        if self.mode_ == "covariate":
            curve = np.asarray(self._model_.predict_survival(Xa), dtype=np.float64)
            idx = times.astype(np.int64) - 2  # G(t-) on the integer grid == G(t - 1)
            safe = np.clip(idx, 0, curve.shape[1] - 1)
            g = curve[np.arange(curve.shape[0]), safe]
            return np.where(idx < 0, 1.0, g)

        # Cox: G(t- | x) = exp(-H0(t-) * risk(x)); step_eval returns 1.0 before the first
        # jump, which is the survival-function convention, so the cumulative hazard is
        # explicitly zeroed there.
        bh = self._model_.baseline_cumhaz_
        grid = bh["time"].to_numpy(dtype=np.float64)
        cum = bh["baseline_cumhaz"].to_numpy(dtype=np.float64)
        H0 = step_eval(grid, cum, times, left_limit=True)
        H0 = np.where(times <= grid[0], 0.0, H0)
        risk = np.asarray(self._model_.predict_risk(Xa), dtype=np.float64)
        with np.errstate(over="ignore"):
            return np.exp(-np.clip(H0 * risk, 0.0, 700.0))

    def survival_before(self, times: Any, X: Any | None = None, *, record: bool = False) -> np.ndarray:
        """Left limit ``G(t-) = P(C >= t)`` for each row, floored at ``min_prob``.

        Parameters
        ----------
        times : array-like of shape (n,)
            Query times, one per row (typically ``min(ceil(T_i), H)``).
        X : array-like of shape (n, p), optional
            Covariates; ignored by the ``"km"`` backend.
        record : bool, keyword-only, default False
            Store the clip diagnostics from this call on the estimator and log them.

        Returns
        -------
        ndarray of shape (n,)
            ``G(t-)`` in ``[min_prob, 1]``.
        """
        t = _as_1d(times, "times")
        raw = np.asarray(self._raw_before(t, X), dtype=np.float64)
        raw = np.where(np.isfinite(raw), raw, 1.0)
        floor = float(self.min_prob)
        clipped = raw < floor
        n_clipped = int(clipped.sum())
        if record:
            self.n_clipped_ = n_clipped
            self.clip_rate_ = float(n_clipped) / max(t.size, 1)
            self.min_g_ = float(raw.min()) if raw.size else 1.0
            if n_clipped:
                LOGGER.info(
                    "CensoringModel(%s): floored G(T-) at %.3g on %d/%d rows (%.2f%%); "
                    "smallest unclipped G = %.4g, so weights are capped at %.1f",
                    self.mode_, floor, n_clipped, t.size, 100.0 * self.clip_rate_,
                    self.min_g_, 1.0 / floor,
                )
        return np.maximum(raw, floor)

    def curve(self, X: Any | None = None, n: int | None = None) -> np.ndarray:
        """Full censoring curve ``G(k | x)`` for ``k = 1 .. horizon``.

        Parameters
        ----------
        X : array-like of shape (n, p), optional
            Covariates; ignored by the ``"km"`` backend, which then needs ``n``.
        n : int, optional
            Number of rows when ``X`` is not supplied.

        Returns
        -------
        ndarray of shape (n, horizon)
        """
        H = int(self.horizon)
        rows = int(n if n is not None else _as_2d(X).shape[0])
        grid = np.arange(1, H + 1, dtype=np.float64)
        if self.mode_ == "none":
            return np.ones((rows, H))
        if self.mode_ == "km":
            return np.tile(step_eval(self._grid_, self._surv_, grid), (rows, 1))
        if self.mode_ == "covariate":
            return np.asarray(self._model_.predict_survival(_as_2d(X)), dtype=np.float64)
        return np.asarray(self._model_.predict_survival(_as_2d(X), grid), dtype=np.float64)


# ======================================================================================
# Public IPCW helpers (SPEC.md section 4.3)
# ======================================================================================
def ipcw_weights(
    event_time: Any,
    event_observed: Any,
    horizon: int,
    X: Any | None = None,
    model: str = "km",
    *,
    min_prob: float = _DEFAULT_MIN_CENSOR_PROB,
    normalize: bool = False,
    random_state: int | np.random.Generator | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    """Inverse-probability-of-censoring weights for a restricted-mean analysis.

    Estimator maths
    ---------------
    Let ``T^*`` be the churn time, ``C`` the censoring time, ``T = min(T^*, C)``,
    ``delta = 1{T^* <= C}`` and ``H`` the horizon. The restricted outcome ``R`` (periods fully
    survived, see :func:`restricted_survival_outcome`) is observed exactly when
    ``C >= min(T^*, H)``, an event of probability ``G(min(T^*, H)-)``. The weight

    .. math::  w_i = \\frac{\\Delta_i^H}{G\\bigl(T_i^H-\\bigr)},
       \\qquad T_i^H = \\min(\\lceil T_i\\rceil, H),
       \\qquad \\Delta_i^H = 1\\{\\delta_i = 1\\ \\text{or}\\ \\lceil T_i\\rceil \\ge H\\}

    therefore satisfies ``E[w_i | T^*_i, X_i] = 1``, so weighting the complete cases by ``w``
    reproduces the uncensored population -- that is the whole content of IPCW.

    Rows that are censored *inside* the horizon get weight 0: they carry no ``R``. Rows that
    survive to the horizon get weight ``1 / G(H-)`` even when ``delta = 0``, because
    administrative censoring at the horizon is **not** censoring for this estimand.

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Observed follow-up time in periods.
    event_observed : array-like of shape (n,)
        ``1`` if churn was observed, ``0`` if right-censored.
    horizon : int
        Restriction horizon ``H``.
    X : array-like of shape (n, p), optional
        Covariates; required when ``model`` is covariate-dependent.
    model : {"km", "cox", "covariate"}, default "km"
        Censoring model, see :class:`CensoringModel`.
    min_prob : float, keyword-only, default 0.05
        Floor on ``G(T-)``. Weights are hence capped at ``1 / min_prob = 20``. The number of
        floored rows is logged; ignoring that number is how IPCW analyses quietly become a
        report about three customers.
    normalize : bool, keyword-only, default False
        Rescale so the weights average 1 over all ``n`` rows (the Hajek form). Reduces
        variance; leaves the ranking untouched.
    random_state : int or numpy.random.Generator or None, keyword-only
        Seed for the covariate censoring model.
    return_diagnostics : bool, keyword-only, default False
        Also return a dict with ``n_clipped``, ``clip_rate``, ``min_g``, ``ess``,
        ``complete_rate`` and ``max_weight``.

    Returns
    -------
    numpy.ndarray or tuple
        ``(n,)`` float64 weights, or ``(weights, diagnostics)`` when ``return_diagnostics``.

    Examples
    --------
    >>> t = np.array([2.0, 4.0, 6.0, 6.0])
    >>> d = np.array([1, 0, 1, 0])
    >>> w = ipcw_weights(t, d, horizon=6)
    >>> bool(w[1] == 0.0)          # censored inside the horizon -> no information
    True
    >>> bool(w[3] > 0.0)           # censored *at* the horizon -> administratively complete
    True
    """
    H = int(horizon)
    out = restricted_survival_outcome(event_time, event_observed, H)
    cm = CensoringModel(model=model, horizon=H, min_prob=min_prob, random_state=random_state)
    cm.fit(event_time, event_observed, X)
    g = cm.survival_before(out.follow_up, X, record=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(out.complete > 0, out.complete / g, 0.0)
    w = np.asarray(np.nan_to_num(w, nan=0.0, posinf=1.0 / max(min_prob, 1e-12)), dtype=np.float64)
    if normalize:
        mean_w = float(w.mean())
        if mean_w > 0:
            w = w / mean_w
    if not return_diagnostics:
        return w
    diag = {
        "n_clipped": float(cm.n_clipped_),
        "clip_rate": float(cm.clip_rate_),
        "min_g": float(cm.min_g_),
        "ess": float(effective_sample_size(w)),
        "complete_rate": float(out.complete.mean()),
        "max_weight": float(w.max()) if w.size else 0.0,
    }
    return w, diag


def rmst_pseudo_outcome(
    event_time: Any,
    event_observed: Any,
    horizon: int,
    censor_surv: np.ndarray,
    *,
    discount_rate: float = 0.0,
    min_prob: float = _DEFAULT_MIN_CENSOR_PROB,
) -> np.ndarray:
    """IPCW pseudo-outcome for restricted mean survival time.

    Estimator maths
    ---------------
    .. math::

        \\psi_i^{IPCW} \\;=\\; \\frac{\\Delta_i^H}{G\\bigl(T_i^H-\\bigr)} \\; R_i,
        \\qquad E\\bigl[\\psi_i^{IPCW} \\mid X_i\\bigr]
              = \\sum_{k=1}^{H} S(k \\mid X_i) = \\mathrm{RMST}(X_i).

    This is the ``min(T, H) * delta / G(T-)`` construction, written on the discrete grid this
    project uses: ``R_i = min(ceil(T_i) - delta_i, H)`` replaces ``min(T_i, H)`` so that the
    pseudo-outcome targets ``sum_{k=1..H} S(k)`` -- the convention of
    :mod:`prism.models.survival` and of ``gt_rmst_*`` -- rather than ``E[min(T, H)]``, which
    is larger by ``P(T <= H)``. Mixing the two conventions inside a doubly-robust correction
    biases every contrast by the churn effect itself; see the module docstring.

    ``G(T-)``, not ``G(T)`` -- the classic bug
    ------------------------------------------
    The denominator is the probability of *not having been censored strictly before* the event
    time, ``G(t-) = P(C >= t)``, and **not** ``G(t) = P(C > t)``. On a discrete grid the two
    differ by the censoring hazard at ``t``, which is exactly the mass of subjects censored at
    the same period as the event. Using ``G(t)`` divides by too small a number, inflates every
    complete case and biases RMST **upward**; on this project's simulator the error was
    +2.4% with ``G(T)`` versus -0.6% with ``G(T-)``. The same mistake is invisible in
    continuous time (where ties have probability zero), which is precisely why it survives
    code review. :meth:`CensoringModel.survival_before` only ever returns the left limit.

    Parameters
    ----------
    event_time : array-like of shape (n,)
        Observed follow-up time in periods.
    event_observed : array-like of shape (n,)
        ``1`` if churn was observed, ``0`` if right-censored.
    horizon : int
        Restriction horizon ``H``.
    censor_surv : numpy.ndarray
        The censoring survival function, in either of two accepted forms:

        * ``(n,)`` -- already evaluated per row as the **left limit** ``G(T_i^H-)``, e.g. from
          :meth:`CensoringModel.survival_before`;
        * ``(n, H)`` -- the per-row curve ``G(k | x_i)`` for ``k = 1 .. H``, from which this
          function takes the left limit itself (``G(T^H - 1)``, with ``G(0) = 1``).

        Values are floored at ``min_prob`` either way.
    discount_rate : float, keyword-only, default 0.0
        When non-zero, the discounted variant is returned: ``R_i`` is replaced by
        ``sum_{k=1..R_i} (1+d)^{-k}``, whose conditional mean is
        ``sum_k S(k|x) (1+d)^{-k}``. This is the margin-free lifetime-value outcome.
    min_prob : float, keyword-only, default 0.05
        Floor applied to ``censor_surv``.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` float64 pseudo-outcomes. Censored-inside-the-horizon rows are exactly 0.

    Examples
    --------
    >>> t = np.array([3.0, 6.0]); d = np.array([1, 0])
    >>> psi = rmst_pseudo_outcome(t, d, horizon=6, censor_surv=np.array([0.8, 0.5]))
    >>> np.round(psi, 4).tolist()      # 2 periods / 0.8, and the horizon-complete row 6 / 0.5
    [2.5, 12.0]
    """
    H = int(horizon)
    out = restricted_survival_outcome(event_time, event_observed, H, discount_rate=discount_rate)
    n = out.periods.size

    g_arr = np.asarray(censor_surv, dtype=np.float64)
    if g_arr.ndim == 2:
        if g_arr.shape[0] != n:
            raise ValueError(f"censor_surv has {g_arr.shape[0]} rows but event_time has {n}")
        idx = out.follow_up.astype(np.int64) - 2
        safe = np.clip(idx, 0, g_arr.shape[1] - 1)
        g = np.where(idx < 0, 1.0, g_arr[np.arange(n), safe])
    else:
        g = g_arr.ravel()
        if g.size == 1:
            g = np.full(n, float(g[0]))
        if g.size != n:
            raise ValueError(f"censor_surv must have length {n} or shape ({n}, H); got {g.size}")

    g = np.maximum(np.where(np.isfinite(g), g, 1.0), float(min_prob))
    value = out.discounted if abs(float(discount_rate)) > 1e-15 else out.periods
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(out.complete > 0, value * out.complete / g, 0.0)
    return np.asarray(np.nan_to_num(psi, nan=0.0), dtype=np.float64)


# ======================================================================================
# Per-arm hazard fitting (module level so joblib can pickle it on Windows)
# ======================================================================================
def _fit_arm_hazard(
    X_arm: np.ndarray,
    t_arm: np.ndarray,
    d_arm: np.ndarray,
    w_arm: np.ndarray | None,
    horizon: int,
    backend: str,
    seed: int,
    hazard_kwargs: dict[str, Any],
) -> DiscreteTimeHazardModel:
    """Fit one arm's discrete-time hazard model. Defined at module level for Windows spawn."""
    kw: dict[str, Any] = dict(horizon=int(horizon), backend=backend, random_state=int(seed))
    kw.update(hazard_kwargs)
    model = DiscreteTimeHazardModel(**kw)
    return model.fit(X_arm, t_arm, d_arm, sample_weight=w_arm)


# ======================================================================================
# The estimator
# ======================================================================================
class CausalSurvivalUplift(BaseCATELearner):  # type: ignore[misc,valid-type]
    """CATE on restricted mean survival time and on discounted CLV, under right-censoring.

    This is the estimator the project exists to demonstrate: an offer that buys three months
    of retention without preventing churn scores **zero** on a 12-month binary uplift model
    and ``+3`` months here. See the module docstring for the commercial argument and for the
    exact RMST convention (right-Riemann, ``k`` from 1, matching
    :mod:`prism.models.survival` and ``gt_rmst_*``).

    Estimator maths
    ---------------
    **Plug-in.** Per arm ``a`` a pooled logistic discrete-time hazard
    ``h_a(k|x) = sigmoid(f_a(x) + alpha_{a,k})`` gives
    ``S_a(k|x) = prod_{j<=k} (1 - h_a(j|x))`` and hence

    .. math::

        \\hat\\tau^{rmst}_a(x) = \\sum_{k=1}^{H}\\bigl[\\hat S_a(k|x) - \\hat S_0(k|x)\\bigr],
        \\qquad
        \\hat\\tau^{value}_a(x) = m(x)\\sum_{k=1}^{H}\\bigl[\\hat S_a - \\hat S_0\\bigr](1+d)^{-k}.

    Consistent **only** if the hazard models are right.

    **Doubly robust.** With the generalised propensity ``e_a(x)``, the censoring survival
    ``G``, the completeness indicator ``Delta^H`` and the matched restricted outcome ``R``
    (all defined in :func:`restricted_survival_outcome`), form the IPCW-AIPW pseudo-outcome

    .. math::

        \\psi_i^{(a)} = \\bigl[\\mathrm{RMST}_a(X_i) - \\mathrm{RMST}_0(X_i)\\bigr]
          + \\frac{1\\{A_i=a\\}}{e_a(X_i)}\\frac{\\Delta_i^H}{G(T_i^H-)}
            \\bigl(R_i - \\mathrm{RMST}_a(X_i)\\bigr)
          - \\frac{1\\{A_i=0\\}}{e_0(X_i)}\\frac{\\Delta_i^H}{G(T_i^H-)}
            \\bigl(R_i - \\mathrm{RMST}_0(X_i)\\bigr)

    and regress it on ``X``. Why this is *doubly* robust, in one line each:

    * if the **hazard** models are right, the two correction terms have conditional mean zero
      (``E[R | A = a, X] = RMST_a(X)``), so ``psi`` collapses to the correct plug-in;
    * if the **propensity** is right, the inverse-probability terms re-randomise the sample,
      so the arm-specific means are unbiased however wrong ``RMST_a`` is -- the bad plug-in
      cancels between the leading term and the correction.

    So ``E[psi | X] = tau_a(X)`` if *either* nuisance is correct; both being wrong is the only
    failure mode. ``1/G`` plays the same role for censoring that ``1/e_a`` plays for treatment.
    Every nuisance is **cross-fitted** (``n_splits`` folds, nuisances never see the row they
    score), without which ``psi`` inherits the nuisance models' overfitting and ``tau`` shrinks
    toward zero with anticonservative intervals.

    Parameters
    ----------
    horizon : int, default 12
        Periods ``H`` over which survival is restricted.
    n_arms : int, default 4
        Number of arms including control (arm 0).
    discount_rate : float, default 0.01
        Monthly discount rate ``d`` for the value contrast.
    hazard_model : {"discrete"}, default "discrete"
        Outcome nuisance family. Only the reused
        :class:`~prism.models.survival.DiscreteTimeHazardModel` is supported; the argument
        exists because ``SPEC.md`` fixes the signature.
    censoring_model : {"km", "cox", "covariate"}, default "km"
        Censoring nuisance, see :class:`CensoringModel`. ``"km"`` assumes censoring is
        marginally independent of ``X``; the other two only need conditional independence.
    doubly_robust : bool, default True
        Run step 5. When ``False`` the estimator is the plug-in and no propensity is needed.
    n_splits : int, default 5
        Cross-fitting folds. Reduced automatically (with a warning) if some fold would
        contain no rows of some arm.
    random_state : int or numpy.random.Generator or None, default None
        Seed. Fixed seed gives bit-identical predictions.
    min_censor_prob : float, keyword-only, default 0.05
        Floor on ``G``; caps IPCW weights at 20. Clip counts are logged and surfaced in
        :meth:`diagnostics`.
    propensity_clip : tuple of float, keyword-only, default (0.01, 0.99)
        Clip applied to a supplied or fitted generalised propensity before inversion.
    hazard_backend : {"sklearn", "torch", "auto"}, keyword-only, default "sklearn"
        Backend for the hazard models. ``"sklearn"`` is the default because it is fast and
        exactly reproducible; ``"torch"`` buys a non-linear ``f_a(x)`` at the cost of runtime.
    hazard_kwargs : dict, keyword-only, optional
        Extra arguments forwarded to every :class:`DiscreteTimeHazardModel`.
    hazard_features : sequence of int, keyword-only, optional
        Restrict the **hazard models only** (the outcome nuisance) to these columns of ``X``.
        Propensity, censoring and the final DR regression still see all of ``X``. This is an
        ablation hook: deliberately crippling the outcome nuisance while leaving the
        propensity correct is how double robustness is *demonstrated* rather than asserted,
        and the smoke test uses it for exactly that.
    final_estimator : sklearn regressor, keyword-only, optional
        Prototype cloned for the second-stage regression of ``psi`` on ``X``. When ``None``
        the second stage is **chosen by cross-validated MSE on ``psi``** from a small
        candidate set -- see :meth:`_select_final` for why that criterion is the right one
        and why a plain gradient-booster is the wrong default here.
    final_kwargs : dict, keyword-only, optional
        Arguments for the gradient-boosted candidate.
    max_inverse_weight : float, keyword-only, default 20.0
        Truncation applied to the combined inverse weight ``1{A=a}/e_a(X) * Delta^H/G(T-)``
        in the pseudo-outcome. Trades a little bias for a lot of variance; the share
        truncated is recorded in ``weight_truncation_rate_`` and logged, because a truncation
        that is never reported is how an IPW analysis silently becomes a report about six
        customers.
    propensity_kwargs : dict, keyword-only, optional
        Extra arguments for the internally fitted
        :class:`~prism.models.propensity.PropensityModel` (ignored when ``propensity`` is
        supplied to :meth:`fit`).
    dr_value : bool, keyword-only, default True
        Also DR-correct the discounted contrast. When ``False``, ``predict_value_cate``
        rescales the plug-in discounted effect.
    n_jobs : int, keyword-only, optional
        Workers for the per-arm hazard fits. Defaults to ``min(n_arms, 4)``
        (``SPEC_PERF`` section 7), threads, because the work is numpy/BLAS-bound.
    chunk_size : int, keyword-only, default 50_000
        Default row block for :meth:`predict_survival_curves`.
    **kw
        Ignored, recorded on ``extra_kwargs_``. Present so that
        ``prism.causal.learners.make_learner(name, **shared_kwargs)`` can pass one dictionary
        to every learner in the leaderboard loop.

    Attributes
    ----------
    n_arms : int
        Number of arms, per the shared CATE interface.
    hazard_models_ : list of DiscreteTimeHazardModel
        Full-sample per-arm models. ``hazard_models_[0]`` is the control model, **fitted once**
        and reused for every contrast (``SPEC_PERF`` section 7).
    censoring_model_ : CensoringModel
        Full-sample censoring fit.
    propensity_ : ndarray of shape (n, n_arms) or None
        Cross-fitted generalised propensity actually used.
    pseudo_outcomes_ : ndarray of shape (n, n_arms - 1) or None
        The RMST pseudo-outcomes ``psi``. Exposed because
        :mod:`prism.causal.evaluate` (GATES, BLP calibration) needs them.
    final_choice_ : list of str
        Which second-stage candidate won the cross-validated selection, per arm.
    weight_truncation_rate_ : float
        Share of inverse weights that hit ``max_inverse_weight``.
    plugin_ate_, dr_ate_ : ndarray of shape (n_arms - 1,)
        In-sample average RMST effects from each path -- the fastest sanity check there is.
    censor_clip_rate_, n_censor_clipped_ : float, int
        Share and count of rows whose ``G(T-)`` hit the floor.
    ipcw_ess_ : float
        Kish effective sample size of the IPCW weights. A large ``n`` with a small ESS means
        the estimate rests on a handful of rows.
    fit_seconds_ : float
        Wall-clock seconds spent in :meth:`fit`.

    Notes
    -----
    ``predict_survival_curves`` is guaranteed monotone non-increasing along the horizon axis
    and strictly inside ``(0, 1)``: the product form makes it so, and a ``minimum.accumulate``
    plus a clip enforce it against floating-point drift.

    Examples
    --------
    >>> import numpy as np, logging
    >>> logging.getLogger("prism").setLevel(logging.WARNING)
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 3))
    >>> w = rng.integers(0, 2, size=400)
    >>> t = np.ceil(rng.exponential(5.0, size=400))
    >>> e = (t <= 8).astype(int)
    >>> m = CausalSurvivalUplift(horizon=6, n_arms=2, doubly_robust=False)
    >>> _ = m.fit(X, w, t, e)
    >>> m.predict_cate(X).shape
    (400, 1)
    >>> S = m.predict_survival_curves(X)
    >>> S.shape, bool(np.all(np.diff(S, axis=2) <= 1e-12))
    ((400, 2, 6), True)
    >>> logging.getLogger("prism").setLevel(logging.INFO)
    """

    def __init__(
        self,
        horizon: int = 12,
        n_arms: int = N_ARMS,
        discount_rate: float = 0.01,
        hazard_model: str = "discrete",
        censoring_model: str = "km",
        doubly_robust: bool = True,
        n_splits: int = 5,
        random_state: int | np.random.Generator | None = None,
        *,
        min_censor_prob: float = _DEFAULT_MIN_CENSOR_PROB,
        propensity_clip: tuple[float, float] = _DEFAULT_PROP_CLIP,
        hazard_backend: str = "sklearn",
        hazard_kwargs: dict[str, Any] | None = None,
        hazard_features: Sequence[int] | None = None,
        final_estimator: Any = None,
        final_kwargs: dict[str, Any] | None = None,
        max_inverse_weight: float = 20.0,
        propensity_kwargs: dict[str, Any] | None = None,
        dr_value: bool = True,
        n_jobs: int | None = None,
        chunk_size: int = 50_000,
        **kw: Any,
    ) -> None:
        self.horizon = horizon
        self.n_arms = n_arms
        self.discount_rate = discount_rate
        self.hazard_model = hazard_model
        self.censoring_model = censoring_model
        self.doubly_robust = doubly_robust
        self.n_splits = n_splits
        self.random_state = random_state
        self.min_censor_prob = min_censor_prob
        self.propensity_clip = propensity_clip
        self.hazard_backend = hazard_backend
        self.hazard_kwargs = hazard_kwargs
        self.hazard_features = hazard_features
        self.final_estimator = final_estimator
        self.final_kwargs = final_kwargs
        self.max_inverse_weight = max_inverse_weight
        self.propensity_kwargs = propensity_kwargs
        self.dr_value = dr_value
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size
        # Absorbed and ignored, so that prism.causal.learners.make_learner can hand every
        # learner one shared kwargs dict without knowing which knobs each of them reads.
        self.extra_kwargs_ = dict(kw)
        if kw:
            LOGGER.debug("CausalSurvivalUplift ignoring unrecognised kwargs %s", sorted(kw))

    # ---------------------------------------------------------------- internals
    def _hazard_view(self, X: np.ndarray) -> np.ndarray:
        """Columns of ``X`` visible to the hazard models (see ``hazard_features``)."""
        if self.hazard_features is None:
            return X
        cols = np.asarray(list(self.hazard_features), dtype=np.int64)
        if cols.size == 0:
            raise ValueError("hazard_features must not be empty")
        if cols.min() < 0 or cols.max() >= X.shape[1]:
            raise ValueError(f"hazard_features out of range for X with {X.shape[1]} columns")
        return np.ascontiguousarray(X[:, cols])

    def _final_candidates(self, seed: int) -> dict[str, Any]:
        """Candidate second-stage regressors, from most to least regularised."""
        from sklearn.dummy import DummyRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        kw: dict[str, Any] = dict(
            n_estimators=300, learning_rate=0.03, max_depth=3, min_samples_leaf=50,
            random_state=int(seed),
        )
        kw.update(self.final_kwargs or {})
        return {
            "constant": DummyRegressor(strategy="mean"),
            "ridge": make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-2.0, 4.0, 25)),
            ),
            "gbm": best_gbm("regression", **kw),
        }

    def _select_final(self, Xa: np.ndarray, psi: np.ndarray, seed: int) -> tuple[Any, str]:
        """Pick the second-stage regressor by cross-validated MSE on the pseudo-outcome.

        Why not simply reach for a gradient booster? Because ``psi`` is an *orthogonal* score,
        not an outcome: it equals ``tau_a(X)`` plus mean-zero noise whose variance is inflated
        by ``1/e_a`` and ``1/G``. On this project's simulator ``sd(psi) ~ 12`` months against a
        true effect spread of ``sd(tau) ~ 1`` month -- a signal-to-noise ratio near 1:12. An
        unregularised booster happily fits that noise and returns effects with three times the
        true spread, which is worse than reporting the ATE.

        The selection criterion is exactly right for the job, and cheap. Because
        ``E[psi | X] = tau(X)``,

        .. math::  E\\bigl[(\\psi - f(X))^2\\bigr]
                   = E\\bigl[(\\tau(X) - f(X))^2\\bigr] + E\\bigl[\\mathrm{Var}(\\psi \\mid X)\\bigr],

        and the second term does not depend on ``f``. So ranking candidates by out-of-fold MSE
        on ``psi`` ranks them by **PEHE**, the quantity we actually care about and can never
        observe. That is the DR-score model-selection argument, and it is the only honest way
        to choose a CATE model without ground truth.

        The candidate set spans the bias-variance range deliberately: a constant (report the
        ATE, admit there is no usable heterogeneity signal), a ridge on standardised features,
        and a shallow gradient booster. The winner is logged.

        Parameters
        ----------
        Xa : ndarray of shape (n, p)
            Covariates.
        psi : ndarray of shape (n,)
            Pseudo-outcomes for one arm.
        seed : int
            Seed for the fold split and the booster.

        Returns
        -------
        tuple
            ``(unfitted estimator prototype, name)``.
        """
        from sklearn.model_selection import KFold

        cands = self._final_candidates(seed)
        folds = list(
            KFold(n_splits=min(5, max(2, int(self.n_splits))), shuffle=True, random_state=int(seed))
            .split(Xa)
        )
        best_name, best_mse = "constant", np.inf
        for name, proto in cands.items():
            err = 0.0
            for tr, te in folds:
                est = clone(proto).fit(Xa[tr], psi[tr])
                err += float(np.sum((psi[te] - np.asarray(est.predict(Xa[te]), dtype=np.float64)) ** 2))
            mse = err / max(Xa.shape[0], 1)
            LOGGER.debug("final-stage candidate %-8s cv-mse %.4f", name, mse)
            if mse < best_mse:
                best_name, best_mse = name, mse
        return clone(cands[best_name]), best_name

    def _make_final(self, seed: int) -> Any:
        """Clone the user-supplied second-stage regressor, seeding it when possible."""
        est = clone(self.final_estimator)
        if hasattr(est, "random_state"):
            try:
                est.set_params(random_state=int(seed) % (2**31 - 1))
            except Exception:  # pragma: no cover - estimator without that param
                pass
        return est

    def _fit_arm_models(
        self,
        Xh: np.ndarray,
        w: np.ndarray,
        t: np.ndarray,
        d: np.ndarray,
        sw: np.ndarray | None,
        rows: np.ndarray,
        seeds: Sequence[int],
    ) -> list[DiscreteTimeHazardModel | None]:
        """Fit one hazard model per arm on ``rows``, in parallel.

        The control arm is fitted **once** here and the resulting object is shared by every
        contrast downstream (``SPEC_PERF`` section 7): the naive alternative refits it
        ``n_arms - 1`` times for no statistical gain.
        """
        K = int(self.n_arms)
        H = int(self.horizon)
        jobs: list[tuple[int, np.ndarray]] = []
        for a in range(K):
            idx = rows[w[rows] == a]
            jobs.append((a, idx))
        min_rows = max(10, H)
        runnable = [(a, idx) for a, idx in jobs if idx.size >= min_rows]
        skipped = [a for a, idx in jobs if idx.size < min_rows]
        if skipped:
            LOGGER.warning(
                "arms %s have fewer than %d rows in this fit and fall back to the control "
                "hazard (their contrast will be ~0)", skipped, min_rows,
            )
        n_jobs = int(self.n_jobs) if self.n_jobs is not None else min(K, 4)
        kwargs = dict(self.hazard_kwargs or {})
        fitted = Parallel(n_jobs=max(1, n_jobs), prefer="threads")(
            delayed(_fit_arm_hazard)(
                Xh[idx], t[idx], d[idx], None if sw is None else sw[idx],
                H, self.hazard_backend, seeds[a], kwargs,
            )
            for a, idx in runnable
        )
        models: list[DiscreteTimeHazardModel | None] = [None] * K
        for (a, _), model in zip(runnable, fitted):
            models[a] = model
        for a in skipped:  # fall back to control so the arm still produces a curve
            models[a] = models[0]
        if models[0] is None:
            raise RuntimeError("the control arm has too few rows to fit a hazard model")
        return models

    def _survival_from(
        self, models: Sequence[DiscreteTimeHazardModel | None], Xh: np.ndarray, chunk: int
    ) -> np.ndarray:
        """Counterfactual survival ``(n, K, H)`` from a set of per-arm models, in chunks."""
        n, K, H = Xh.shape[0], int(self.n_arms), int(self.horizon)
        out = np.empty((n, K, H), dtype=np.float64)
        step = max(1, int(chunk))
        for s in range(0, n, step):
            block = Xh[s : s + step]
            for a in range(K):
                model = models[a] if models[a] is not None else models[0]
                out[s : s + step, a, :] = model.predict_survival(block)  # type: ignore[union-attr]
        # Belt and braces: the product form is already monotone, but floating-point drift in
        # a backend must never be able to hand the optimiser an increasing survival curve.
        np.minimum.accumulate(out, axis=2, out=out)
        return np.clip(out, _S_LO, _S_HI, out=out)

    def _rmst_from_curves(self, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(RMST, discounted RMST)`` per arm from ``(n, K, H)`` survival curves."""
        return S.sum(axis=2), S @ self._disc_

    # ---------------------------------------------------------------- fit
    def fit(
        self,
        X: Any,
        w: Any,
        event_time: Any,
        event_observed: Any | None = None,
        *,
        margin: Any | None = None,
        propensity: Any | None = None,
        sample_weight: Any | None = None,
    ) -> CausalSurvivalUplift:
        """Fit censoring, per-arm hazard and (optionally) doubly-robust models.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates, point-in-time safe.
        w : array-like of shape (n,)
            Arm index in ``0 .. n_arms - 1``; ``0`` is control.
        event_time : array-like of shape (n,)
            Observed follow-up in periods (months).
        event_observed : array-like of shape (n,), optional
            ``1`` if churn was observed, ``0`` if right-censored. **Optional only so that the
            pipeline's uniform ``fit(X, w, y)`` call works**; omitting it declares the data
            uncensored, which is rarely true -- a warning is logged.
        margin : array-like of shape (n,) or float, keyword-only, optional
            Monthly gross margin ``m(x)``. Used as the default scale for
            :meth:`predict_value_cate`; the effect itself is estimated margin-free.
        propensity : array-like of shape (n, n_arms), keyword-only, optional
            Generalised propensity ``e_a(x)``. If omitted and ``doubly_robust=True`` a
            cross-fitted :class:`~prism.models.propensity.PropensityModel` is fitted here.
        sample_weight : array-like of shape (n,), keyword-only, optional
            Per-row weights, passed through to the hazard fits.

        Returns
        -------
        CausalSurvivalUplift
            ``self``, fitted.
        """
        t0 = _time.perf_counter()
        Xa = _as_2d(X)
        n, p = Xa.shape
        arm = np.asarray(_as_1d(w, "w", n), dtype=np.int64)
        K, H = int(self.n_arms), int(self.horizon)
        if H < 1:
            raise ValueError(f"horizon must be >= 1, got {H}")
        if K < 2:
            raise ValueError(f"n_arms must be >= 2, got {K}")
        if arm.min() < 0 or arm.max() >= K:
            raise ValueError(f"w must lie in [0, {K - 1}]; got [{arm.min()}, {arm.max()}]")
        if str(self.hazard_model).lower() not in {"discrete", "dth", "discrete_time"}:
            raise ValueError(
                f"hazard_model={self.hazard_model!r} is not supported; this module reuses "
                "prism.models.survival.DiscreteTimeHazardModel ('discrete')"
            )
        t = _as_1d(event_time, "event_time", n)
        if event_observed is None:
            LOGGER.warning(
                "event_observed not supplied; treating all %d rows as uncensored events. "
                "Pass event_observed unless the data really is complete.", n,
            )
            d = np.ones(n, dtype=np.float64)
        else:
            d = _as_1d(event_observed, "event_observed", n)
        sw = None if sample_weight is None else _as_1d(sample_weight, "sample_weight", n)
        if sw is not None and np.any(sw < 0):
            raise ValueError("sample_weight must be non-negative")

        self.n_features_in_ = p
        self._disc_ = discount_factors(H, self.discount_rate)
        rng = as_rng(self.random_state)
        self.margin_mean_ = 1.0
        if margin is not None:
            mg = _as_1d(margin, "margin") if np.ndim(margin) else np.full(n, float(margin))
            if mg.size not in (1, n):
                raise ValueError(f"margin must be scalar or length {n}; got {mg.size}")
            self.margin_mean_ = float(np.mean(mg))

        Xh = self._hazard_view(Xa)
        all_rows = np.arange(n, dtype=np.int64)

        # -- 1. censoring model on the full sample -------------------------------------
        self.outcome_ = restricted_survival_outcome(t, d, H, discount_rate=self.discount_rate)
        self.censoring_model_ = CensoringModel(
            model=self.censoring_model,
            horizon=H,
            min_prob=self.min_censor_prob,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ).fit(t, d, Xa)
        g_full = self.censoring_model_.survival_before(self.outcome_.follow_up, Xa, record=True)
        ipcw = np.where(self.outcome_.complete > 0, self.outcome_.complete / g_full, 0.0)
        self.ipcw_weights_ = np.asarray(ipcw, dtype=np.float64)
        self.n_censor_clipped_ = int(self.censoring_model_.n_clipped_)
        self.censor_clip_rate_ = float(self.censoring_model_.clip_rate_)
        self.min_censor_surv_ = float(self.censoring_model_.min_g_)
        self.ipcw_ess_ = float(effective_sample_size(self.ipcw_weights_))
        self.complete_rate_ = float(self.outcome_.complete.mean())
        LOGGER.info(
            "censoring(%s): %.1f%% of rows complete at H=%d, IPCW ESS %.0f/%d, "
            "%d rows floored at G=%.3g",
            self.censoring_model_.mode_, 100.0 * self.complete_rate_, H,
            self.ipcw_ess_, n, self.n_censor_clipped_, float(self.min_censor_prob),
        )

        # -- 2. per-arm hazard models on the full sample -------------------------------
        seeds = [int(g.integers(0, 2**31 - 1)) for g in spawn_rngs(rng, K)]
        self.hazard_models_ = self._fit_arm_models(Xh, arm, t, d, sw, all_rows, seeds)
        self.arm_counts_ = np.bincount(arm, minlength=K).astype(np.int64)

        S_full = self._survival_from(self.hazard_models_, Xh, int(self.chunk_size))
        rmst_full, disc_full = self._rmst_from_curves(S_full)
        self.plugin_ate_ = (rmst_full[:, 1:] - rmst_full[:, :1]).mean(axis=0)
        self.plugin_ate_disc_ = (disc_full[:, 1:] - disc_full[:, :1]).mean(axis=0)

        self.propensity_ = None
        self.pseudo_outcomes_ = None
        self.dr_models_ = None
        self.dr_models_disc_ = None
        self.dr_ate_ = None
        self.overlap_ = None
        self.n_splits_ = 0
        self.final_choice_ = []
        self.weight_truncation_rate_ = 0.0

        if not self.doubly_robust:
            self.fit_seconds_ = _time.perf_counter() - t0
            LOGGER.info("CausalSurvivalUplift (plug-in) fitted in %.2fs", self.fit_seconds_)
            return self

        # -- 3. cross-fitted nuisances -------------------------------------------------
        counts = np.bincount(arm, minlength=K)
        present = counts[counts > 0]
        n_splits = int(self.n_splits)
        if present.size and n_splits > int(present.min()):
            new = max(2, int(present.min()))
            LOGGER.warning(
                "n_splits=%d exceeds the smallest arm count (%d); reducing to %d so every "
                "fold contains every arm", n_splits, int(present.min()), new,
            )
            n_splits = new
        n_splits = max(2, min(n_splits, n))
        self.n_splits_ = n_splits

        if propensity is None:
            pkw: dict[str, Any] = dict(
                n_arms=K,
                n_splits=n_splits,
                random_state=int(rng.integers(0, 2**31 - 1)),
                clip=tuple(self.propensity_clip),
            )
            pkw.update(self.propensity_kwargs or {})
            pm = PropensityModel(**pkw).fit(Xa, arm)
            e = pm.oof_proba()
            self.propensity_model_ = pm
        else:
            e = np.asarray(propensity, dtype=np.float64)
            if e.shape != (n, K):
                raise ValueError(f"propensity must have shape ({n}, {K}); got {e.shape}")
            e = clip_renormalize(e, float(self.propensity_clip[0]), float(self.propensity_clip[1]))
            self.propensity_model_ = None
        self.propensity_ = e
        try:
            self.overlap_ = overlap_diagnostics(e, arm)
        except Exception:  # pragma: no cover - diagnostics must never break a fit
            self.overlap_ = None

        rmst_oof = np.empty((n, K), dtype=np.float64)
        disc_oof = np.empty((n, K), dtype=np.float64)
        g_oof = np.empty(n, dtype=np.float64)
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1))
        )
        for fold, (tr, te) in enumerate(splitter.split(np.zeros(n), arm)):
            fold_seeds = [int(g.integers(0, 2**31 - 1)) for g in spawn_rngs(seeds[0] + fold + 1, K)]
            models = self._fit_arm_models(Xh, arm, t, d, sw, tr, fold_seeds)
            S_te = self._survival_from(models, Xh[te], int(self.chunk_size))
            rmst_oof[te], disc_oof[te] = self._rmst_from_curves(S_te)
            cm = CensoringModel(
                model=self.censoring_model,
                horizon=H,
                min_prob=self.min_censor_prob,
                random_state=seeds[0] + 1000 + fold,
            ).fit(t[tr], d[tr], Xa[tr])
            g_oof[te] = cm.survival_before(self.outcome_.follow_up[te], Xa[te])

        # -- 4. IPCW-AIPW pseudo-outcomes and the second-stage regression --------------
        wc = np.where(self.outcome_.complete > 0, self.outcome_.complete / g_oof, 0.0)
        R = self.outcome_.periods
        Rd = self.outcome_.discounted
        cap = float(self.max_inverse_weight)
        n_trunc = 0

        def _weight(mask: np.ndarray, e_col: np.ndarray) -> np.ndarray:
            """Combined inverse treatment x censoring weight, truncated at ``cap``."""
            nonlocal n_trunc
            raw = mask / e_col * wc
            n_trunc += int(np.sum(raw > cap))
            return np.minimum(raw, cap)

        ctrl_term = _weight((arm == 0).astype(np.float64), e[:, 0])
        psi = np.empty((n, K - 1), dtype=np.float64)
        psi_d = np.empty((n, K - 1), dtype=np.float64)
        for a in range(1, K):
            treat_term = _weight((arm == a).astype(np.float64), e[:, a])
            psi[:, a - 1] = (
                rmst_oof[:, a] - rmst_oof[:, 0]
                + treat_term * (R - rmst_oof[:, a])
                - ctrl_term * (R - rmst_oof[:, 0])
            )
            psi_d[:, a - 1] = (
                disc_oof[:, a] - disc_oof[:, 0]
                + treat_term * (Rd - disc_oof[:, a])
                - ctrl_term * (Rd - disc_oof[:, 0])
            )
        self.pseudo_outcomes_ = psi
        self.pseudo_outcomes_disc_ = psi_d
        self.weight_truncation_rate_ = float(n_trunc) / max(n * K, 1)
        if n_trunc:
            LOGGER.info(
                "truncated %d inverse weights (%.3f%% of %d arm-row terms) at %.1f; "
                "raw weights that large come from e_a(x) near the clip, not from data",
                n_trunc, 100.0 * self.weight_truncation_rate_, n * K, cap,
            )

        # Second stage: regress psi on X. The estimator is *chosen* rather than assumed --
        # see _select_final for the cross-validated-MSE-on-psi argument.
        final_seeds = [int(g.integers(0, 2**31 - 1)) for g in spawn_rngs(rng, K - 1)]
        fit_kw = {"sample_weight": sw} if sw is not None else {}
        self.dr_models_ = []
        self.final_choice_ = []
        protos: list[Any] = []
        for a in range(1, K):
            if self.final_estimator is not None:
                proto, name = self._make_final(final_seeds[a - 1]), "user"
            else:
                proto, name = self._select_final(Xa, psi[:, a - 1], final_seeds[a - 1])
            protos.append(proto)
            self.final_choice_.append(name)
            self.dr_models_.append(clone(proto).fit(Xa, psi[:, a - 1], **fit_kw))
        if self.dr_value:
            # The discounted pseudo-outcome is the same regression problem rescaled by the
            # discount factors, so it reuses the selected family rather than paying for a
            # second search.
            self.dr_models_disc_ = [
                clone(protos[a - 1]).fit(Xa, psi_d[:, a - 1], **fit_kw) for a in range(1, K)
            ]
        LOGGER.info("second-stage model per arm: %s", self.final_choice_)
        self.dr_ate_ = psi.mean(axis=0)
        self.dr_ate_disc_ = psi_d.mean(axis=0)
        self.fit_seconds_ = _time.perf_counter() - t0
        LOGGER.info(
            "CausalSurvivalUplift (DR, %d folds) fitted in %.2fs | plug-in ATE %s | DR ATE %s",
            n_splits, self.fit_seconds_, np.round(self.plugin_ate_, 3), np.round(self.dr_ate_, 3),
        )
        return self

    # ---------------------------------------------------------------- predict
    def _check_fitted(self) -> None:
        if not hasattr(self, "hazard_models_"):
            raise RuntimeError("CausalSurvivalUplift is not fitted; call fit(...) first")

    def predict_survival_curves(self, X: Any, chunk_size: int | None = None) -> np.ndarray:
        """Counterfactual survival curves ``S_a(k | x)`` for every arm.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        chunk_size : int, optional
            Rows processed per block; defaults to the constructor's ``chunk_size`` (50 000).
            At ``n = 1e6``, ``K = 4``, ``H = 12`` the full array is 384 MB, so the work is
            chunked (``SPEC_PERF`` section 7). The returned array is still materialised in
            full -- chunking bounds the *peak* working set, not the result.

        Returns
        -------
        ndarray of shape (n, n_arms, horizon)
            ``S_a(k | x_i)`` for ``k = 1 .. horizon``, monotone non-increasing along the last
            axis and strictly inside ``(0, 1)``.
        """
        self._check_fitted()
        Xa = _as_2d(X)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, model was fitted with {self.n_features_in_}")
        chunk = int(chunk_size) if chunk_size is not None else int(self.chunk_size)
        return self._survival_from(self.hazard_models_, self._hazard_view(Xa), chunk)

    def predict_rmst(self, X: Any, chunk_size: int | None = None) -> np.ndarray:
        """Per-arm plug-in RMST ``sum_{k=1..H} S_a(k | x)``.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        chunk_size : int, optional
            Passed to :meth:`predict_survival_curves`.

        Returns
        -------
        ndarray of shape (n, n_arms)
            Expected months retained under each arm, in ``[0, horizon]``.
        """
        return self.predict_survival_curves(X, chunk_size).sum(axis=2)

    def predict_cate(self, X: Any, *, plugin: bool = False, chunk_size: int | None = None) -> np.ndarray:
        """CATE on restricted mean survival time, in **months**.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        plugin : bool, keyword-only, default False
            Return the plug-in contrast ``sum_k [S_a(k|x) - S_0(k|x)]`` instead of the
            doubly-robust one. Kept so the two can be compared directly -- the smoke test
            shows the DR path winning when the hazard model is misspecified, which is the
            entire point of the correction.
        chunk_size : int, optional
            Passed to :meth:`predict_survival_curves` for the plug-in path.

        Returns
        -------
        ndarray of shape (n, n_arms - 1)
            ``tau^rmst_a(x)`` for ``a = 1 .. n_arms - 1``. Positive = the offer buys months.
        """
        self._check_fitted()
        if plugin or not self.doubly_robust or self.dr_models_ is None:
            S = self.predict_survival_curves(X, chunk_size)
            rmst = S.sum(axis=2)
            return np.asarray(rmst[:, 1:] - rmst[:, :1], dtype=np.float64)
        Xa = _as_2d(X)
        return np.column_stack([np.asarray(m.predict(Xa), dtype=np.float64) for m in self.dr_models_])

    def predict(self, X: Any) -> np.ndarray:
        """Alias of :meth:`predict_cate` (the shared CATE interface)."""
        return self.predict_cate(X)

    def predict_value_cate(
        self,
        X: Any,
        margin: Any | None = None,
        *,
        plugin: bool = False,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        """CATE on discounted lifetime value, in **currency**.

        Estimator maths
        ---------------
        ``tau^value_a(x) = m(x) * sum_{k=1..H} [S_a(k|x) - S_0(k|x)] (1 + d)^{-k}``. The margin
        factors out of the sum, so the discounted retention effect is estimated margin-free
        and multiplied here -- which means a scoring-time margin different from the training
        one is handled correctly instead of being baked into the model.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        margin : array-like of shape (n,) or float, optional
            Monthly gross margin ``m(x)``. Defaults to the mean margin seen in :meth:`fit`
            (1.0 if none was supplied, in which case the output is "discounted months").
        plugin : bool, keyword-only, default False
            Use the plug-in discounted contrast instead of the DR-corrected one.
        chunk_size : int, optional
            Passed to :meth:`predict_survival_curves`.

        Returns
        -------
        ndarray of shape (n, n_arms - 1)
            Incremental discounted margin per customer per arm, before offer cost. Feed this
            to :func:`prism.decision.economics.expected_net_value`.
        """
        self._check_fitted()
        Xa = _as_2d(X)
        n = Xa.shape[0]
        if margin is None:
            m = np.full(n, float(self.margin_mean_))
        elif np.ndim(margin) == 0:
            m = np.full(n, float(margin))
        else:
            m = _as_1d(margin, "margin", n)

        use_dr = (not plugin) and self.doubly_robust and self.dr_models_disc_ is not None
        if use_dr:
            eff = np.column_stack(
                [np.asarray(md.predict(Xa), dtype=np.float64) for md in self.dr_models_disc_]
            )
        else:
            S = self.predict_survival_curves(Xa, chunk_size)
            disc = S @ self._disc_
            eff = disc[:, 1:] - disc[:, :1]
            if (not plugin) and self.doubly_robust and self.dr_models_ is not None:
                # dr_value=False: carry the DR correction across by the plug-in's own
                # discounted-to-undiscounted ratio, which is the best available rescaling.
                plug_rmst = S.sum(axis=2)
                plug = plug_rmst[:, 1:] - plug_rmst[:, :1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(np.abs(plug) > 1e-9, eff / plug, float(self._disc_.mean()))
                eff = ratio * self.predict_cate(Xa)
        return np.asarray(eff * m[:, None], dtype=np.float64)

    def predict_churn_cate(self, X: Any, within: int = 1, *, chunk_size: int | None = None) -> np.ndarray:
        """CATE on the **probability of churning within ``within`` periods**.

        Estimator maths
        ---------------
        ``tau^churn_a(x, c) = [1 - S_a(c|x)] - [1 - S_0(c|x)] = S_0(c|x) - S_a(c|x)``.

        Sign convention: **negative is good** -- the offer lowers churn probability. This is
        the quantity a conventional binary uplift model targets, kept here so the comparison
        can be made on identical data. It is deliberately a plug-in: the doubly-robust
        machinery in this module targets RMST and discounted retention, which are the
        estimands the budget optimiser consumes.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        within : int, default 1
            Number of periods, clipped to ``[1, horizon]``. Pass ``within=horizon`` to get the
            classic "churned by month H" uplift.
        chunk_size : int, optional
            Passed to :meth:`predict_survival_curves`.

        Returns
        -------
        ndarray of shape (n, n_arms - 1)
            Change in churn probability, in ``[-1, 1]``.
        """
        self._check_fitted()
        k = int(np.clip(int(within), 1, int(self.horizon)))
        S = self.predict_survival_curves(X, chunk_size)[:, :, k - 1]
        return np.asarray(S[:, :1] - S[:, 1:], dtype=np.float64)

    # ---------------------------------------------------------------- reporting
    def diagnostics(self) -> pd.DataFrame:
        """Tidy one-row-per-metric diagnostic frame for the reporting layer.

        Returns
        -------
        pandas.DataFrame
            Columns ``metric``, ``arm``, ``value``. Includes the censoring clip count and
            rate (never hidden), the IPCW effective sample size, per-arm row counts and both
            the plug-in and doubly-robust average effects.
        """
        self._check_fitted()
        rows: list[dict[str, Any]] = [
            {"metric": "horizon", "arm": "", "value": float(self.horizon)},
            {"metric": "complete_rate", "arm": "", "value": self.complete_rate_},
            {"metric": "censor_clip_rate", "arm": "", "value": self.censor_clip_rate_},
            {"metric": "n_censor_clipped", "arm": "", "value": float(self.n_censor_clipped_)},
            {"metric": "min_censor_surv", "arm": "", "value": self.min_censor_surv_},
            {"metric": "ipcw_ess", "arm": "", "value": self.ipcw_ess_},
            {"metric": "weight_truncation_rate", "arm": "", "value": self.weight_truncation_rate_},
            {"metric": "n_splits_used", "arm": "", "value": float(self.n_splits_)},
            {"metric": "fit_seconds", "arm": "", "value": float(self.fit_seconds_)},
        ]
        for a in range(int(self.n_arms)):
            label = ARM_NAMES[a] if a < len(ARM_NAMES) else f"arm_{a}"
            rows.append({"metric": "n_rows", "arm": label, "value": float(self.arm_counts_[a])})
        for a in range(1, int(self.n_arms)):
            label = ARM_NAMES[a] if a < len(ARM_NAMES) else f"arm_{a}"
            rows.append({"metric": "ate_rmst_plugin", "arm": label, "value": float(self.plugin_ate_[a - 1])})
            if self.dr_ate_ is not None:
                rows.append({"metric": "ate_rmst_dr", "arm": label, "value": float(self.dr_ate_[a - 1])})
        return pd.DataFrame(rows)


# ======================================================================================
# Smoke test
# ======================================================================================
def _simulate_survival_uplift(
    n: int = 6000, horizon: int = 12, random_state: int = 11
) -> dict[str, Any]:
    """Multi-arm right-censored survival data with a KNOWN discrete-time hazard.

    Everything is generated from ``h_a(k | x) = sigmoid(eta(x) + alpha_k + shift_a(k, x))``
    with all constants fixed below, so the true counterfactual curves ``S_a(k | x)`` and hence
    the true ``tau^rmst_a``, ``tau^value_a`` and ``tau^churn_a`` are **analytic**, not resampled.

    The design deliberately contains:

    * an interaction ``x0 * x3`` in ``eta`` that is **not** a column of ``X``, so the linear
      hazard model is genuinely misspecified (as it would be in reality);
    * confounded assignment (arm depends on ``x0``, ``x1``, ``z``) with overlap preserved;
    * covariate-dependent censoring (``C`` depends on ``eta``), which breaks marginal
      Kaplan-Meier IPCW and is what the covariate censoring model is for;
    * **arm 1 = a pure delay offer** -- hazard down hard for 3 periods, slightly up afterwards
      -- calibrated so that for the ``z = 0`` subgroup the 12-month churn probability is
      unchanged while RMST rises by ~0.8 months. That subgroup is the argument for this whole
      estimator.
    * arm 2 = a genuine prevention offer (uniform hazard reduction), as a contrast.
    """
    from scipy.special import expit

    rng = as_rng(random_state)
    H = int(horizon)
    x = rng.normal(size=(n, 5))
    z = (rng.random(n) < 0.35).astype(np.float64)
    X = np.column_stack([x, z])

    eta = 0.55 * x[:, 0] - 0.45 * x[:, 1] + 0.30 * x[:, 2] + 0.35 * x[:, 0] * x[:, 3] + 0.60 * z
    margin = np.clip(18.0 + 6.0 * x[:, 2], 6.0, 40.0)

    def alpha(k: np.ndarray) -> np.ndarray:
        return -2.75 + 0.045 * k

    def delay_shift(k: np.ndarray) -> np.ndarray:
        return np.where(k <= 3, -1.60, 0.20)

    # -- confounded assignment, overlap preserved ------------------------------------
    score = 0.75 * np.column_stack([
        np.zeros(n),
        0.80 * x[:, 0] - 0.50 * x[:, 1] + 0.60 * z - 0.30,
        0.50 * x[:, 2] + 0.40 * x[:, 0] - 0.60,
    ])
    e_true = np.exp(score)
    e_true /= e_true.sum(axis=1, keepdims=True)
    w = (rng.random(n)[:, None] > np.cumsum(e_true, axis=1)).sum(axis=1).astype(np.int64)

    # -- realised event and censoring times on a long grid ---------------------------
    Hx = 4 * H
    kk = np.arange(1, Hx + 1, dtype=np.float64)
    shift = np.zeros((n, Hx))
    shift[w == 1] = delay_shift(kk)[None, :] - 0.70 * z[w == 1, None]
    shift[w == 2] = -0.60
    hz = expit(eta[:, None] + alpha(kk)[None, :] + shift)
    fired = rng.random((n, Hx)) < hz
    t_star = np.where(fired.any(axis=1), fired.argmax(axis=1) + 1, Hx + 1).astype(np.float64)

    h_cens = expit(-3.15 + 0.55 * eta)
    fired_c = rng.random((n, Hx)) < h_cens[:, None]
    c_time = np.where(fired_c.any(axis=1), fired_c.argmax(axis=1) + 1, Hx + 1).astype(np.float64)

    delta = (t_star <= c_time).astype(np.int64)
    t_obs = np.minimum(t_star, c_time)

    # -- analytic counterfactual truth over the horizon ------------------------------
    k = np.arange(1, H + 1, dtype=np.float64)
    shifts = np.zeros((n, 3, H))
    shifts[:, 1, :] = delay_shift(k)[None, :] - 0.70 * z[:, None]
    shifts[:, 2, :] = -0.60
    haz = expit(eta[:, None, None] + alpha(k)[None, None, :] + shifts)
    S_true = np.cumprod(1.0 - haz, axis=2)

    disc = discount_factors(H, 0.01)
    rmst_true = S_true.sum(axis=2)
    value_true = (S_true @ disc) * margin[:, None]
    tau_rmst = rmst_true[:, 1:] - rmst_true[:, :1]
    tau_value = value_true[:, 1:] - value_true[:, :1]
    tau_churn = S_true[:, :1, -1] - S_true[:, 1:, -1]

    return dict(
        X=X, w=w, event_time=t_obs, event_observed=delta, margin=margin, propensity=e_true,
        S_true=S_true, tau_rmst=tau_rmst, tau_value=tau_value, tau_churn=tau_churn,
        z=z, horizon=H, true_rmst_realised=rmst_true[np.arange(n), w],
    )


def _pehe(true: np.ndarray, est: np.ndarray) -> float:
    """Root mean squared error of the treatment effect (precision in estimating HTE)."""
    return float(np.sqrt(np.mean((np.asarray(true) - np.asarray(est)) ** 2)))


def _check(rows: list[dict[str, Any]], name: str, ok: bool, detail: str) -> bool:
    rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    return ok


if __name__ == "__main__":  # pragma: no cover - smoke test
    import logging

    logging.getLogger("prism").setLevel(logging.WARNING)
    t_start = _time.perf_counter()
    np.set_printoptions(precision=4, suppress=True)

    HORIZON = 12
    data = _simulate_survival_uplift(n=6000, horizon=HORIZON, random_state=11)
    X, w = data["X"], data["w"]
    t, d = data["event_time"], data["event_observed"]
    margin, e_true = data["margin"], data["propensity"]
    tau_rmst_true, tau_value_true = data["tau_rmst"], data["tau_value"]
    tau_churn_true, z = data["tau_churn"], data["z"]
    n = X.shape[0]

    cens_in = float(np.mean((d == 0) & (t < HORIZON)))
    print("=" * 84)
    print("prism.causal.survival_uplift -- smoke test against KNOWN analytic truth")
    print("=" * 84)
    print(
        f"n={n}  arms={np.bincount(w).tolist()}  horizon={HORIZON}  "
        f"censored before horizon={cens_in:.1%}  overlap e in "
        f"[{e_true.min():.3f}, {e_true.max():.3f}]"
    )

    checks: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- 1. IPCW machinery
    truth_mean_rmst = float(data["true_rmst_realised"].mean())
    out = restricted_survival_outcome(t, d, HORIZON)
    naive = float(out.periods[out.complete > 0].mean())
    psi_km = rmst_pseudo_outcome(
        t, d, HORIZON,
        CensoringModel("km", HORIZON).fit(t, d).survival_before(out.follow_up),
    )
    cm_cov = CensoringModel("covariate", HORIZON, random_state=0).fit(t, d, X)
    psi_cov = rmst_pseudo_outcome(t, d, HORIZON, cm_cov.survival_before(out.follow_up, X))
    err_naive = abs(naive / truth_mean_rmst - 1.0)
    err_km = abs(float(psi_km.mean()) / truth_mean_rmst - 1.0)
    err_cov = abs(float(psi_cov.mean()) / truth_mean_rmst - 1.0)
    print("\n-- 1. IPCW recovers the truth that complete-case analysis destroys ---------")
    print(f"   true  E[RMST] (analytic, realised arm)   = {truth_mean_rmst:7.4f}")
    print(f"   complete-case mean (drops censored rows) = {naive:7.4f}  ({err_naive:+.2%})")
    print(f"   IPCW pseudo-outcome, marginal KM  G(T-)  = {psi_km.mean():7.4f}  ({err_km:+.2%})")
    print(f"   IPCW pseudo-outcome, covariate G(T-|x)   = {psi_cov.mean():7.4f}  ({err_cov:+.2%})")
    _check(checks, "ipcw_beats_complete_case", err_cov < err_naive / 2.0,
           f"cov err {err_cov:.2%} vs complete-case {err_naive:.2%}")
    _check(checks, "ipcw_unbiased_within_2pct", err_cov < 0.02, f"|bias| = {err_cov:.2%}")

    wts, wdiag = ipcw_weights(t, d, HORIZON, X, model="covariate", return_diagnostics=True)
    print(
        f"   weights: complete {wdiag['complete_rate']:.1%}, max {wdiag['max_weight']:.2f}, "
        f"ESS {wdiag['ess']:.0f}/{n}, floored {int(wdiag['n_clipped'])} rows "
        f"({wdiag['clip_rate']:.2%}), min G {wdiag['min_g']:.3f}"
    )
    _check(checks, "censored_rows_get_zero_weight",
           bool(np.all(wts[(d == 0) & (t < HORIZON)] == 0.0)),
           "every row censored inside the horizon has weight 0")
    _check(checks, "horizon_complete_rows_kept",
           bool(np.all(wts[t >= HORIZON] > 0.0)),
           "administrative censoring at the horizon is not censoring for RMST")

    # ---------------------------------------------------------------- 2. main fit
    print("\n-- 2. fitting CausalSurvivalUplift (DR, cross-fitted nuisances) ------------")
    MAIN_KW = dict(
        horizon=HORIZON, n_arms=3, discount_rate=0.01, censoring_model="covariate",
        doubly_robust=True, n_splits=3, random_state=7,
        # A light propensity keeps the smoke test inside its 60 s budget even on a loaded box;
        # production callers should leave the PropensityModel defaults (calibrated, 150 trees)
        # alone, since 1/e_a sits in the numerator of every pseudo-outcome.
        propensity_kwargs={"n_estimators": 60, "learning_rate": 0.12, "calibrate": False},
    )
    model = CausalSurvivalUplift(**MAIN_KW).fit(X, w, t, d, margin=margin)
    print(f"   second-stage model chosen by cross-validated MSE on psi: {model.final_choice_}")

    S = model.predict_survival_curves(X, chunk_size=2000)
    _check(checks, "survival_shape", S.shape == (n, 3, HORIZON), f"shape {S.shape}")
    _check(checks, "survival_monotone", bool(np.all(np.diff(S, axis=2) <= 1e-12)),
           f"max increase {float(np.max(np.diff(S, axis=2))):.2e}")
    _check(checks, "survival_in_open_unit_interval", bool(np.all((S > 0.0) & (S < 1.0))),
           f"range [{S.min():.3e}, {1 - S.max():.3e} from 1]")
    _check(checks, "chunking_is_exact",
           bool(np.allclose(S, model.predict_survival_curves(X, chunk_size=50_000))),
           "chunk_size does not change the answer")

    tau_dr = model.predict_cate(X)
    tau_pi = model.predict_cate(X, plugin=True)
    _check(checks, "cate_shape", tau_dr.shape == (n, 2), f"shape {tau_dr.shape}")

    print("\n-- 3. accuracy against the analytic tau^rmst -------------------------------")
    print("   arm                 corr     PEHE   PEHE(const)   true ATE    est ATE")
    corr_all = []
    arm_labels = ("delay", "prevent")
    for a in range(2):
        c = float(np.corrcoef(tau_rmst_true[:, a], tau_dr[:, a])[0, 1])
        corr_all.append(c)
        p_model = _pehe(tau_rmst_true[:, a], tau_dr[:, a])
        p_const = _pehe(tau_rmst_true[:, a], np.full(n, tau_dr[:, a].mean()))
        print(
            f"   arm {a + 1} ({arm_labels[a]:>7s})  {c:6.3f}  {p_model:7.4f}  {p_const:11.4f}  "
            f"{tau_rmst_true[:, a].mean():9.3f}  {tau_dr[:, a].mean():9.3f}"
        )
        _check(checks, f"pehe_beats_constant_arm{a + 1}", p_model < p_const,
               f"PEHE {p_model:.4f} < constant-effect {p_const:.4f}")
    corr_pooled = float(np.corrcoef(tau_rmst_true.ravel(), tau_dr.ravel())[0, 1])
    _check(checks, "corr_with_true_tau_rmst_gt_0.4", corr_pooled > 0.4,
           f"pooled corr = {corr_pooled:.3f} (per arm {np.round(corr_all, 3).tolist()})")
    _check(checks, "corr_delay_arm_gt_0.4", corr_all[0] > 0.4, f"arm 1 corr = {corr_all[0]:.3f}")

    tau_val = model.predict_value_cate(X, margin=margin)
    corr_val = float(np.corrcoef(tau_value_true.ravel(), tau_val.ravel())[0, 1])
    print(f"   tau^value: corr {corr_val:.3f} | true ATE {tau_value_true.mean(axis=0).round(2)} "
          f"| est ATE {tau_val.mean(axis=0).round(2)} (currency)")
    _check(checks, "value_cate_tracks_truth", corr_val > 0.4, f"corr = {corr_val:.3f}")

    # ---------------------------------------------------------------- 4. double robustness
    print("\n-- 4. double robustness: cripple the hazard model, keep the propensity -----")
    crippled = CausalSurvivalUplift(**{**MAIN_KW, "hazard_features": [0]}).fit(
        X, w, t, d, margin=margin, propensity=e_true
    )
    tau_c_dr = crippled.predict_cate(X)
    tau_c_pi = crippled.predict_cate(X, plugin=True)

    pehe_ok_pi = _pehe(tau_rmst_true, tau_pi)
    pehe_ok_dr = _pehe(tau_rmst_true, tau_dr)
    pehe_bad_pi = _pehe(tau_rmst_true, tau_c_pi)
    pehe_bad_dr = _pehe(tau_rmst_true, tau_c_dr)
    print("   nuisances                         plug-in PEHE   DR PEHE   DR improvement")
    print(f"   both decent                          {pehe_ok_pi:8.4f}  {pehe_ok_dr:8.4f}   "
          f"{100 * (1 - pehe_ok_dr / pehe_ok_pi):+6.1f}%")
    print(f"   hazard crippled (1 feature), e true  {pehe_bad_pi:8.4f}  {pehe_bad_dr:8.4f}   "
          f"{100 * (1 - pehe_bad_dr / pehe_bad_pi):+6.1f}%")
    _check(checks, "dr_no_worse_when_nuisances_decent", pehe_ok_dr <= 1.10 * pehe_ok_pi,
           f"DR {pehe_ok_dr:.4f} vs plug-in {pehe_ok_pi:.4f}")
    _check(checks, "dr_rescues_a_broken_hazard_model", pehe_bad_dr < 0.95 * pehe_bad_pi,
           f"DR {pehe_bad_dr:.4f} < plug-in {pehe_bad_pi:.4f} -- consistency survives a "
           "wrong outcome model because the propensity is right")

    # ---------------------------------------------------------------- 5. the argument
    print("\n-- 5. THE ARGUMENT: delay is invisible to binary uplift, visible to RMST ---")
    churn_h_est = model.predict_churn_cate(X, within=HORIZON)
    delay_grp = z == 0.0          # arm 1 delays these customers but does not save them
    prevent_grp = z == 1.0        # arm 1 genuinely prevents churn for these
    print(f"   arm 1 = the 'delay' offer; subgroup split on one observed covariate "
          f"(n={int(delay_grp.sum())} / {int(prevent_grp.sum())})")
    print("   subgroup            churn uplift @12m          tau^rmst (months)      tau^value")
    print("                        true      est             true     est            est (ccy)")
    for label, mask in (("delay-only  (z=0)", delay_grp), ("prevented   (z=1)", prevent_grp)):
        print(
            f"   {label}  {float(tau_churn_true[mask, 0].mean()):+9.4f} "
            f"{float(churn_h_est[mask, 0].mean()):+9.4f}   "
            f"{float(tau_rmst_true[mask, 0].mean()):+9.3f} {float(tau_dr[mask, 0].mean()):+8.3f}"
            f"      {float(tau_val[mask, 0].mean()):+9.2f}"
        )
    true_churn_delay = float(np.abs(tau_churn_true[delay_grp, 0].mean()))
    est_churn_delay = float(np.abs(churn_h_est[delay_grp, 0].mean()))
    est_rmst_delay = float(tau_dr[delay_grp, 0].mean())
    est_value_delay = float(tau_val[delay_grp, 0].mean())
    ratio = est_rmst_delay / max(est_churn_delay, 1e-9)
    print(
        f"\n   For the z=0 subgroup a 12-month binary churn model sees {est_churn_delay:.4f} "
        f"({est_churn_delay * 100:.2f} pp) and would spend nothing."
    )
    print(
        f"   RMST sees {est_rmst_delay:+.2f} months, worth {est_value_delay:+.2f} in discounted "
        f"margin per customer -- {est_value_delay * int(delay_grp.sum()):,.0f} across the subgroup."
    )
    print(f"   Ratio of signals: RMST/binary = {ratio:,.0f}x. That gap is the whole estimator.")
    _check(checks, "truth_has_zero_binary_effect_for_delay_group", true_churn_delay < 0.01,
           f"true |tau^churn| = {true_churn_delay:.4f} by construction")
    _check(checks, "estimator_reproduces_zero_binary_effect", est_churn_delay < 0.02,
           f"estimated |tau^churn| = {est_churn_delay:.4f}")
    _check(checks, "estimator_sees_the_delay_in_rmst", est_rmst_delay > 0.3,
           f"estimated tau^rmst = {est_rmst_delay:+.3f} months")
    _check(checks, "binary_would_rank_the_subgroups_backwards",
           abs(float(churn_h_est[prevent_grp, 0].mean())) > 5.0 * est_churn_delay,
           "binary uplift puts ~all the value in z=1; RMST finds real value in z=0 too")

    # ---------------------------------------------------------------- 6. determinism
    again = CausalSurvivalUplift(**{**MAIN_KW, "hazard_features": [0]}).fit(
        X, w, t, d, margin=margin, propensity=e_true
    )
    _check(checks, "deterministic_under_fixed_seed",
           bool(np.array_equal(again.predict_cate(X), tau_c_dr))
           and bool(np.array_equal(again.predict_survival_curves(X), crippled.predict_survival_curves(X))),
           "two fits with random_state=7 give bit-identical survival curves and CATE")

    print("\n-- diagnostics -------------------------------------------------------------")
    diag = model.diagnostics()
    print(diag.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n-- checks ------------------------------------------------------------------")
    report = pd.DataFrame(checks)
    print(report.to_string(index=False))
    elapsed = _time.perf_counter() - t_start
    n_fail = int((report["status"] == "FAIL").sum())
    print(f"\nsurvival_uplift.py {'OK' if n_fail == 0 else 'FAILED'} "
          f"-- {len(report) - n_fail}/{len(report)} checks passed in {elapsed:.1f}s")
    if n_fail:
        raise AssertionError(f"{n_fail} smoke-test check(s) failed:\n{report.to_string(index=False)}")
