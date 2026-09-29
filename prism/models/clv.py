"""Customer-lifetime-value models: BG/NBD, Gamma-Gamma, DeepCLV and the survival bridge.

This module supplies the *value* half of PRISM. The causal layer answers "how much longer
will this customer stay if we treat them"; this module answers "how much is a month of
staying worth", and the two meet in :func:`clv_from_survival`.

Three estimators, deliberately different in kind
------------------------------------------------
:class:`BGNBD`
    The Beta-Geometric / NBD "buy-till-you-die" model of Fader, Hardie & Lee (2005), for
    the *non-contractual* case where churn is never observed, only inferred from a
    lengthening silence. The MLE is implemented here from first principles with
    :mod:`scipy.optimize` and an analytic gradient -- no ``lifetimes`` dependency.

:class:`GammaGamma`
    The monetary-value companion (Fader & Hardie, 2013). Shrinks a customer's noisy
    observed average spend towards the population mean by exactly the amount their
    transaction count justifies.

:class:`DeepCLV`
    A discriminative two-head neural network for the *contractual* case, where features
    are rich and the retention label is observed. Falls back to a two-stage
    classifier-times-regressor sklearn model when torch is absent.

The generative models (BG/NBD, Gamma-Gamma) need only an RFM summary and extrapolate a
transaction process; the discriminative model (DeepCLV) needs a feature matrix and a
realised value label. PRISM uses the generative pair to build a prior-quality ``clv``
column for the decision layer, and DeepCLV when the panel's 23 features are available.

Model mathematics
-----------------
**BG/NBD.** While a customer is alive, purchases follow a Poisson process with rate
``lambda``; after each purchase they die with probability ``p``. Heterogeneity is
``lambda ~ Gamma(r, alpha)`` (``alpha`` a *rate*) and ``p ~ Beta(a, b)``. For a customer
observed for ``T`` time units who made ``x`` repeat purchases, the last at time ``t_x``,
the likelihood integrates both latent variables out in closed form::

    L = B(a, b+x)/B(a,b) * Gamma(r+x)/Gamma(r) * alpha^r / (alpha+T)^(r+x)
      + 1{x>0} * B(a+1, b+x-1)/B(a,b) * Gamma(r+x)/Gamma(r) * alpha^r / (alpha+t_x)^(r+x)

The first term is "still alive at T", the second "died at some point after purchase x".
Because ``B(a+1, b+x-1)/B(a,b) = [a/(b+x-1)] * B(a, b+x)/B(a,b)``, both terms share a
factor and the log-likelihood collapses to a two-element log-sum-exp::

    A1 = gammaln(r+x) - gammaln(r) + r*log(alpha)
    A2 = gammaln(a+b) + gammaln(b+x) - gammaln(b) - gammaln(a+b+x)
    A3 = -(r+x) * log(alpha + T)
    A4 = log(a) - log(b+x-1) - (r+x)*log(alpha + t_x)          [only when x > 0]
    LL = A1 + A2 + logsumexp([A3, A4])

**Gamma-Gamma.** Transaction values are ``z ~ Gamma(p, nu_i)`` with ``nu_i ~ Gamma(q, v)``.
The posterior-mean spend is a credibility-weighted blend of the customer's own average and
the population average -- see :meth:`GammaGamma.conditional_expected_average_profit`, which
derives the formula rather than asserting it.

Time-unit convention
--------------------
:class:`BGNBD` and :class:`GammaGamma` are unit-agnostic: ``recency`` and ``T`` merely have
to share a unit, and predictions are expressed in that same unit.
:func:`probabilistic_clv` fixes that unit to **months**, because it discounts monthly.

References
----------
.. [1] Fader, P.S., Hardie, B.G.S. and Lee, K.L. (2005). "Counting Your Customers the Easy
       Way: An Alternative to the Pareto/NBD Model." *Marketing Science* 24(2), 275-284.
.. [2] Fader, P.S. and Hardie, B.G.S. (2013). "The Gamma-Gamma Model of Monetary Value."
.. [3] Fader, P.S. and Hardie, B.G.S. (2007). "Incorporating time-invariant covariates into
       the Pareto/NBD and BG/NBD models."
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammainc, gammaln, hyp2f1, logsumexp
from sklearn.base import BaseEstimator

from prism.utils.logging import get_logger
from prism.utils.optional import HAS_TORCH, best_gbm, require
from prism.utils.seeds import as_rng

__all__ = [
    "BGNBD",
    "GammaGamma",
    "probabilistic_clv",
    "DeepCLV",
    "clv_from_survival",
    "simulate_clv_population",
    "annuity_factor",
]

LOGGER = get_logger("models.clv")

#: Floor for quantities that are about to be logged or divided by.
_EPS: float = 1e-12
#: Probabilities are clipped into this open interval before any ``log`` (SPEC_PERF section 9).
_P_LO: float = 1e-6
_P_HI: float = 1.0 - 1e-6
#: Bound on log-parameters during optimisation: exp(+-14) spans 8e-7 .. 1.2e6.
_LOG_BOUND: float = 14.0
#: Step taken off the *removable* singularity of ``c/(a-1)`` at ``a == 1``. Deliberately
#: not smaller: the bracket ``1 - ratio*2F1`` cancels to ``O(a-1)``, so a tinier step loses
#: far more to catastrophic cancellation than it gains in bias (1e-6 is ~20x more accurate
#: at ``a == 1`` than 1e-9 -- measured, not guessed).
_A_STEP: float = 1e-6
#: Step taken off ``c == 0``, the only genuine pole of the 2F1 form reachable with a, b > 0.
_C_STEP: float = 1e-9
#: Seed used for the jittered optimiser restarts when ``random_state is None``, so that a
#: fit without a seed is still reproducible (the class docstrings promise this).
_DEFAULT_JITTER_SEED: int = 20050205


# ======================================================================================
# small shared helpers
# ======================================================================================
def _as_float_1d(values: Any, name: str) -> np.ndarray:
    """Coerce ``values`` to a contiguous 1-D float64 array.

    Parameters
    ----------
    values : array-like
        Input sequence, :class:`pandas.Series` or scalar-free array.
    name : str
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        1-D float64 array.

    Raises
    ------
    ValueError
        If the input is not 1-D or contains non-finite entries.
    """
    arr = np.asarray(values, dtype=np.float64).ravel() if np.ndim(values) <= 1 else np.asarray(values, np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf ({int((~np.isfinite(arr)).sum())} entries)")
    return np.ascontiguousarray(arr)


def _check_rfm(frequency: Any, recency: Any, T: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a BG/NBD recency-frequency-age triple.

    Parameters
    ----------
    frequency : array-like of shape (n,)
        Number of *repeat* transactions ``x`` (the first, cohort-defining purchase excluded).
    recency : array-like of shape (n,)
        Time ``t_x`` of the last repeat transaction, measured from the first purchase.
        This is *not* "time since last purchase"; the gap is ``T - recency``.
    T : array-like of shape (n,)
        Total observation age, same unit as ``recency``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(frequency, recency, T)`` as float64 arrays.

    Raises
    ------
    ValueError
        If lengths disagree, values are negative, ``recency > T``, or a customer with
        ``frequency == 0`` has a non-zero ``recency``.
    """
    x = _as_float_1d(frequency, "frequency")
    t_x = _as_float_1d(recency, "recency")
    age = _as_float_1d(T, "T")
    if not (x.shape == t_x.shape == age.shape):
        raise ValueError(f"frequency/recency/T must share a length, got {x.shape}, {t_x.shape}, {age.shape}")
    if x.size == 0:
        raise ValueError("cannot fit on an empty sample")
    if np.any(x < 0):
        raise ValueError("frequency must be non-negative")
    if np.any(t_x < 0) or np.any(age <= 0):
        raise ValueError("recency must be >= 0 and T must be > 0")
    if np.any(t_x > age + 1e-9):
        raise ValueError("recency must not exceed T (recency is the time OF the last purchase)")
    bad_zero = (x == 0) & (t_x > 1e-9)
    if np.any(bad_zero):
        raise ValueError(f"{int(bad_zero.sum())} customers have frequency == 0 but recency > 0")
    return x, t_x, age


def annuity_factor(horizon: int, discount_rate: float) -> float:
    """Present value of one unit received at the end of each of ``horizon`` periods.

    This is the ordinary-annuity factor ``sum_{t=1}^{H} (1+d)^-t``, which equals
    ``(1 - (1+d)^-H) / d`` for ``d > 0`` and ``H`` for ``d == 0``. PRISM discounts at the
    *end* of each period throughout, so ``t`` starts at 1 everywhere -- see
    :func:`clv_from_survival`.

    Parameters
    ----------
    horizon : int
        Number of periods, must be >= 0.
    discount_rate : float
        Per-period discount rate ``d``, must be > -1.

    Returns
    -------
    float
        The annuity factor.
    """
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if discount_rate <= -1.0:
        raise ValueError("discount_rate must be greater than -1")
    if abs(discount_rate) < 1e-15:
        return float(horizon)
    return float((1.0 - (1.0 + discount_rate) ** (-horizon)) / discount_rate)


# ======================================================================================
# BG/NBD
# ======================================================================================
class BGNBD(BaseEstimator):
    """Beta-Geometric / Negative-Binomial-Distribution model, fitted by maximum likelihood.

    The "buy-till-you-die" model for non-contractual settings. Churn is a latent event: a
    customer is never observed leaving, so the model infers death probability from how
    long they have been quiet relative to how often they used to buy. See the module
    docstring for the likelihood derivation.

    The optimiser works on ``log(r, alpha, a, b)`` so the parameters stay strictly
    positive without constrained optimisation, uses an **analytic gradient**, and runs
    from several starting points keeping the best optimum (the BG/NBD surface has flat
    ridges, so a single start is not trustworthy).

    Parameters
    ----------
    penalizer : float, default 1e-4
        Coefficient of an L2 penalty on the *log* parameters, added to the mean negative
        log-likelihood: ``obj = -mean(LL) + penalizer * sum(log_params**2)``. Shrinks the
        parameters towards 1 and keeps the ridge-shaped surface identified. Because the
        likelihood term is a mean, the penalty does not have to be rescaled with ``n``.
    n_restarts : int, default 6
        Number of starting points. The first four are fixed (so a fit is reproducible
        without a seed), the remainder are random jitters drawn from ``random_state``.
    max_iter : int, default 2000
        L-BFGS-B iteration cap per restart.
    tol : float, default 1e-10
        ``ftol`` passed to L-BFGS-B.
    random_state : int or numpy.random.Generator or None, optional
        Seed for the jittered restarts. Only consumed when ``n_restarts > 4``. ``None``
        uses a fixed internal seed rather than a fresh entropy draw, so a fit is
        reproducible whether or not the caller supplies a seed.

    Attributes
    ----------
    params_ : dict
        Fitted parameters with keys ``"r"``, ``"alpha"``, ``"a"``, ``"b"``.
    log_likelihood_ : float
        Total (not mean) log-likelihood at the optimum.
    converged_ : bool
        Whether the winning restart reported success.
    n_ : int
        Number of customers used in the fit.
    restart_table_ : pandas.DataFrame
        One row per restart with its objective and resulting parameters -- evidence about
        how flat the surface was.

    Examples
    --------
    >>> import logging; logging.getLogger("prism").setLevel(logging.WARNING)
    >>> summary = simulate_clv_population(500, random_state=0)
    >>> model = BGNBD().fit(summary["frequency"], summary["recency"], summary["T"])
    >>> sorted(model.params_)
    ['a', 'alpha', 'b', 'r']
    >>> bool(np.all((model.probability_alive(summary["frequency"], summary["recency"],
    ...                                      summary["T"]) >= 0)))
    True
    """

    def __init__(
        self,
        penalizer: float = 1e-4,
        *,
        n_restarts: int = 6,
        max_iter: int = 2000,
        tol: float = 1e-10,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.penalizer = penalizer
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # ---------------------------------------------------------------- likelihood
    @staticmethod
    def _objective(
        log_params: np.ndarray,
        x: np.ndarray,
        t_x: np.ndarray,
        T: np.ndarray,
        w: np.ndarray,
        w_sum: float,
        penalizer: float,
    ) -> tuple[float, np.ndarray]:
        """Weighted-mean negative log-likelihood and its gradient w.r.t. the log-parameters.

        Parameters
        ----------
        log_params : numpy.ndarray of shape (4,)
            ``log([r, alpha, a, b])``.
        x, t_x, T : numpy.ndarray of shape (n,)
            Frequency, recency and age.
        w : numpy.ndarray of shape (n,)
            Non-negative case weights.
        w_sum : float
            ``w.sum()``, precomputed.
        penalizer : float
            L2 coefficient on ``log_params``.

        Returns
        -------
        tuple of (float, numpy.ndarray)
            Objective value and its 4-vector gradient.
        """
        r, alpha, a, b = np.exp(log_params)
        pos = x > 0
        rx = r + x
        bx1 = np.maximum(b + x - 1.0, _EPS)  # >= b > 0 whenever x >= 1

        log_aT = np.log(alpha + T)
        log_atx = np.log(alpha + t_x)

        a1 = gammaln(rx) - gammaln(r) + r * np.log(alpha)
        a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        a3 = -rx * log_aT
        a4 = np.where(pos, np.log(a) - np.log(bx1) - rx * log_atx, -np.inf)

        stacked = np.stack((a3, a4))
        lse = logsumexp(stacked, axis=0)
        ll = a1 + a2 + lse
        if not np.all(np.isfinite(ll)):
            return 1e12, np.zeros(4)

        obj = -float(np.dot(w, ll)) / w_sum + penalizer * float(np.dot(log_params, log_params))

        # d/dtheta logsumexp = softmax-weighted average of the two branch gradients
        mix = np.exp(stacked - lse)
        m3, m4 = mix[0], mix[1]

        d_r = (digamma(rx) - digamma(r) + np.log(alpha)) + m3 * (-log_aT) + m4 * np.where(pos, -log_atx, 0.0)
        d_alpha = (r / alpha) + m3 * (-rx / (alpha + T)) + m4 * np.where(pos, -rx / (alpha + t_x), 0.0)
        d_a = (digamma(a + b) - digamma(a + b + x)) + m4 * np.where(pos, 1.0 / a, 0.0)
        d_b = (digamma(a + b) + digamma(b + x) - digamma(b) - digamma(a + b + x)) + m4 * np.where(pos, -1.0 / bx1, 0.0)

        grad_nat = np.array([np.dot(w, d_r), np.dot(w, d_alpha), np.dot(w, d_a), np.dot(w, d_b)]) / w_sum
        # chain rule to log-space: d/d log(theta) = theta * d/d theta
        grad = -grad_nat * np.exp(log_params) + 2.0 * penalizer * log_params
        if not np.all(np.isfinite(grad)):
            return 1e12, np.zeros(4)
        return obj, grad

    # ---------------------------------------------------------------- fit
    def fit(
        self,
        frequency: Any,
        recency: Any,
        T: Any,
        weights: Any | None = None,
    ) -> BGNBD:
        """Fit ``(r, alpha, a, b)`` by penalised maximum likelihood.

        Parameters
        ----------
        frequency : array-like of shape (n,)
            Repeat-transaction counts ``x``.
        recency : array-like of shape (n,)
            Time of the last repeat transaction ``t_x`` (0 when ``x == 0``).
        T : array-like of shape (n,)
            Observation age.
        weights : array-like of shape (n,), optional
            Non-negative case weights, for callers who pre-aggregated duplicate RFM rows.
            Defaults to 1 for every customer.

        Returns
        -------
        BGNBD
            ``self``, fitted.

        Raises
        ------
        RuntimeError
            If every restart failed to produce a finite optimum.
        """
        x, t_x, age = _check_rfm(frequency, recency, T)
        if weights is None:
            w = np.ones_like(x)
        else:
            w = _as_float_1d(weights, "weights")
            if w.shape != x.shape:
                raise ValueError("weights must align with frequency")
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
        w_sum = float(w.sum())
        if w_sum <= 0:
            raise ValueError("weights sum to zero")

        args = (x, t_x, age, w, w_sum, float(self.penalizer))
        bounds = [(-_LOG_BOUND, _LOG_BOUND)] * 4
        rows: list[dict[str, Any]] = []
        best = None
        for i, init in enumerate(self._initial_points(x, t_x, age, w)):
            res = minimize(
                self._objective,
                np.log(np.clip(init, np.exp(-_LOG_BOUND), np.exp(_LOG_BOUND))),
                args=args,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(self.max_iter), "ftol": float(self.tol), "gtol": 1e-8},
            )
            p = np.exp(res.x)
            rows.append(
                {
                    "restart": i,
                    "init_r": init[0], "init_alpha": init[1], "init_a": init[2], "init_b": init[3],
                    "objective": float(res.fun) if np.isfinite(res.fun) else np.inf,
                    "r": p[0], "alpha": p[1], "a": p[2], "b": p[3],
                    "success": bool(res.success),
                }
            )
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res

        if best is None:
            raise RuntimeError("BGNBD: every optimisation restart failed; check the input RFM summary")

        r, alpha, a, b = (float(v) for v in np.exp(best.x))
        self.params_: dict[str, float] = {"r": r, "alpha": alpha, "a": a, "b": b}
        self.converged_ = bool(best.success)
        self.n_ = int(x.size)
        self.optim_result_ = best
        self.restart_table_ = pd.DataFrame(rows).sort_values("objective", ignore_index=True)
        # recover the un-penalised total log-likelihood from the penalised mean objective
        penalty = float(self.penalizer) * float(np.dot(best.x, best.x))
        self.log_likelihood_ = float(-(best.fun - penalty) * w_sum)
        spread = self.restart_table_["objective"].replace(np.inf, np.nan).dropna()
        self.restart_spread_ = float(spread.max() - spread.min()) if len(spread) else 0.0

        LOGGER.info(
            "BGNBD fit on n=%d: r=%.4f alpha=%.4f a=%.4f b=%.4f | LL=%.2f | converged=%s | restart spread=%.2e",
            self.n_, r, alpha, a, b, self.log_likelihood_, self.converged_, self.restart_spread_,
        )
        if a <= 1.0:
            LOGGER.info(
                "BGNBD: fitted a=%.4f <= 1. This is the ordinary empirical regime (the canonical "
                "CDNOW fit has a=0.79): the Beta dropout prior is J-shaped and both the expected-"
                "transaction formulas evaluate through a sign cancellation that is handled exactly. "
                "Predictions are valid; only treat the separate a and b estimates as weakly "
                "identified (r/alpha and a/(a+b) are the well-determined summaries).",
                a,
            )
        if not self.converged_:
            LOGGER.warning("BGNBD: best restart did not report convergence (%s)", best.message)
        return self

    def _initial_points(self, x: np.ndarray, t_x: np.ndarray, T: np.ndarray, w: np.ndarray) -> list[np.ndarray]:
        """Build the list of starting parameter vectors for the multi-restart search.

        The first point is a crude moment match (``alpha`` set so that ``r/alpha`` equals
        the observed purchase rate), followed by three fixed spread-out points and then
        random jitters drawn from ``random_state`` -- or from a fixed internal seed when
        that is ``None``, so every fit is reproducible at any ``n_restarts``.

        Parameters
        ----------
        x, t_x, T : numpy.ndarray
            Frequency, recency, age.
        w : numpy.ndarray
            Case weights.

        Returns
        -------
        list of numpy.ndarray
            Starting points, each a 4-vector ``[r, alpha, a, b]`` on the natural scale.
        """
        w_sum = float(w.sum())
        xbar = float(np.dot(w, x) / w_sum)
        tbar = float(np.dot(w, T) / w_sum)
        rate = max(xbar, 0.05) / max(tbar, _EPS)
        # fraction of customers that look lapsed -> a crude read on the dropout prior mean
        gap_frac = float(np.dot(w, (T - t_x) / np.maximum(T, _EPS)) / w_sum)
        gap_frac = float(np.clip(gap_frac, 0.05, 0.95))

        fixed = [
            np.array([1.0, max(1.0 / rate, 1e-3), 1.0, max((1.0 - gap_frac) / gap_frac, 0.1)]),
            np.array([0.5, 5.0, 0.5, 2.0]),
            np.array([2.0, 10.0, 2.0, 3.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
        ]
        n_extra = max(int(self.n_restarts) - len(fixed), 0)
        if n_extra:
            # ``random_state=None`` falls back to a fixed seed rather than a fresh
            # non-deterministic stream, so that the reproducibility the class docstring
            # promises holds at the DEFAULT n_restarts=6 too, not only at n_restarts<=4.
            # Without this, two unseeded fits can pick different near-tied restarts and
            # disagree in the 7th significant figure.
            rng = as_rng(_DEFAULT_JITTER_SEED if self.random_state is None else self.random_state)
            base = np.log(np.array([1.0, max(1.0 / rate, 1e-3), 1.0, 1.5]))
            jitter = rng.normal(scale=1.0, size=(n_extra, 4))
            fixed.extend(np.exp(base + jitter))
        return fixed[: max(int(self.n_restarts), 1)]

    # ---------------------------------------------------------------- predictions
    def _unpack(self) -> tuple[float, float, float, float]:
        """Return ``(r, alpha, a, b)``, raising if the model is not fitted."""
        if not hasattr(self, "params_"):
            raise RuntimeError("BGNBD is not fitted; call fit() first")
        p = self.params_
        return p["r"], p["alpha"], p["a"], p["b"]

    def _log_death_odds(self, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Log of the posterior dead-to-alive odds ``log[(a/(b+x-1)) * ((alpha+T)/(alpha+t_x))^(r+x)]``.

        Computed in log space because ``((alpha+T)/(alpha+t_x))^(r+x)`` overflows float64
        for a long-lapsed, high-frequency customer.

        Parameters
        ----------
        x, t_x, T : numpy.ndarray
            Frequency, recency, age.

        Returns
        -------
        numpy.ndarray
            Log-odds, ``-inf`` where ``x == 0`` (a customer who never repeated cannot yet
            have died, since death only follows a purchase).
        """
        r, alpha, a, b = self._unpack()
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(
                x > 0,
                np.log(a) - np.log(np.maximum(b + x - 1.0, _EPS)) + (r + x) * (np.log(alpha + T) - np.log(alpha + t_x)),
                -np.inf,
            )
        return out

    def probability_alive(self, frequency: Any, recency: Any, T: Any) -> np.ndarray:
        """Posterior probability that a customer is still alive at time ``T``.

        ``P(alive | x, t_x, T) = 1 / (1 + 1{x>0} * (a/(b+x-1)) * ((alpha+T)/(alpha+t_x))^(r+x))``.

        The expression is monotonically **decreasing** in the recency gap ``T - t_x``:
        holding ``x`` and ``T`` fixed, a smaller ``t_x`` raises ``(alpha+T)/(alpha+t_x)``
        and so raises the dead-to-alive odds. It is also decreasing in ``x`` for a fixed
        gap -- a frequent buyer's silence is more damning than an occasional one's.

        Parameters
        ----------
        frequency : array-like of shape (n,)
            Repeat-transaction counts.
        recency : array-like of shape (n,)
            Time of the last repeat transaction.
        T : array-like of shape (n,)
            Observation age.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Probabilities in ``[0, 1]``. Exactly 1 where ``frequency == 0``.
        """
        x, t_x, age = _check_rfm(frequency, recency, T)
        log_odds = self._log_death_odds(x, t_x, age)
        # 1/(1+exp(z)) computed as expit(-z), branch-free and overflow-safe
        out = np.empty_like(log_odds)
        neg = log_odds < 0
        out[neg] = 1.0 / (1.0 + np.exp(log_odds[neg]))
        ez = np.exp(-log_odds[~neg])
        out[~neg] = ez / (1.0 + ez)
        return np.clip(out, 0.0, 1.0)

    def conditional_expected_transactions(self, t: Any, frequency: Any, recency: Any, T: Any) -> np.ndarray:
        """Expected repeat transactions in the next ``t`` time units, given RFM history.

        Equation (10) of Fader, Hardie & Lee (2005)::

            E[Y(t) | x, t_x, T] =
                (a + b + x - 1)/(a - 1)
                * [1 - ((alpha+T)/(alpha+T+t))^(r+x) * 2F1(r+x, b+x; a+b+x-1; t/(alpha+T+t))]
                / [1 + 1{x>0} * (a/(b+x-1)) * ((alpha+T)/(alpha+t_x))^(r+x)]

        The denominator is exactly ``1/P(alive)``, so this is
        ``P(alive) * E[future purchases | alive]``. The Gaussian hypergeometric ``2F1``
        sums the geometric series of "survive the next dropout opportunity and buy again";
        it is evaluated with :func:`scipy.special.hyp2f1`.

        When ``a < 1`` both ``(a-1)`` and the bracket are negative, so the product remains
        positive; this is the common empirical regime (the canonical CDNOW fit has
        ``a = 0.79``) and is handled correctly, as is the further corner ``a + b < 1``
        where the 2F1's ``c`` parameter itself goes negative. Results are floored at 0 to
        absorb the last ulp of cancellation. Verified against
        :func:`_series_conditional_transactions` across all of these regimes.

        Parameters
        ----------
        t : float or array-like
            Length of the forecast window, in the same time unit as ``T``. Broadcast
            against the customer arrays.
        frequency : array-like of shape (n,)
            Repeat-transaction counts.
        recency : array-like of shape (n,)
            Time of the last repeat transaction.
        T : array-like of shape (n,)
            Observation age.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Expected number of transactions in ``(T, T + t]``, non-negative.
        """
        r, alpha, a, b = self._unpack()
        x, t_x, age = _check_rfm(frequency, recency, T)
        horizon = np.asarray(t, dtype=np.float64)
        if np.any(horizon < 0):
            raise ValueError("t must be non-negative")
        horizon = np.broadcast_to(horizon, x.shape).astype(np.float64, copy=False)

        # ``a == 1`` is a removable singularity of ``c/(a-1)``; step off it.
        if abs(a - 1.0) < _A_STEP:
            a = 1.0 + _A_STEP

        # ``c = a + b + x - 1`` may legitimately be NEGATIVE -- that happens exactly when
        # ``x == 0`` and ``a + b < 1``. 2F1 is perfectly well defined for negative
        # *non-integer* ``c``, and the closed form stays exact there (the smoke test pins
        # it against an independent exact series in this very regime, and gets ~1e-15).
        # With ``a, b > 0``
        # the only non-positive integer ``c`` reachable at all is ``c == 0`` (``x == 0``,
        # ``a + b == 1``), so step off that and nothing else. Clipping every negative ``c``
        # up to a small positive number, as an earlier version did, silently biased those
        # customers by 10-60%.
        c = a + b + x - 1.0
        near_pole = np.abs(c) < _C_STEP
        if np.any(near_pole):
            c = np.where(near_pole, np.where(c >= 0.0, _C_STEP, -_C_STEP), c)

        denom_t = alpha + age + horizon
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ratio = np.power((alpha + age) / denom_t, r + x)
            hyp = hyp2f1(r + x, b + x, c, horizon / denom_t)
            bracket = 1.0 - ratio * hyp
            expected_if_alive = (c / (a - 1.0)) * bracket
            result = expected_if_alive * self.probability_alive(x, t_x, age)

        bad = ~np.isfinite(result)
        if np.any(bad):
            LOGGER.warning(
                "BGNBD: 2F1 returned %d non-finite values (extreme frequency/recency); "
                "those predictions fall back to 0",
                int(bad.sum()),
            )
            result = np.where(bad, 0.0, result)
        return np.maximum(result, 0.0)

    def expected_number_of_transactions(self, t: Any) -> np.ndarray:
        """Unconditional ``E[X(t)]`` for a randomly chosen *new* customer.

        ``E[X(t)] = (a+b-1)/(a-1) * [1 - (alpha/(alpha+t))^r * 2F1(r, b; a+b-1; t/(alpha+t))]``.
        Useful for cohort-level forecasting and for the acquisition side of the business
        case, where no RFM history exists yet.

        Parameters
        ----------
        t : float or array-like
            Forecast window length.

        Returns
        -------
        numpy.ndarray
            Expected transactions, same shape as ``t`` (at least 1-D).

        Notes
        -----
        ``a < 1`` is **not** a degenerate case. The count over a finite window is bounded
        by the Poisson count, so its mean always exists; what happens for ``a < 1`` is
        simply that ``(a-1)`` and the bracket are both negative and their ratio stays
        positive, exactly as in :meth:`conditional_expected_transactions`. The smoke test
        checks this branch against the model's own pmf.
        """
        r, alpha, a, b = self._unpack()
        tt = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if np.any(tt < 0):
            raise ValueError("t must be non-negative")
        # Same two guards as conditional_expected_transactions, and for the same reasons:
        # ``a < 1`` is a perfectly ordinary regime (it is in fact the commonest empirical
        # one), so only the removable singularity at ``a == 1`` and the true pole at
        # ``c == 0`` are stepped over. An earlier version replaced any ``a <= 1`` with
        # ``1 + 1e-9``, which is not a guard but a different model: it under-stated
        # E[X(t)] by up to 56% at the canonical CDNOW-style fit.
        if abs(a - 1.0) < _A_STEP:
            a = 1.0 + _A_STEP
        c = a + b - 1.0
        if abs(c) < _C_STEP:
            c = _C_STEP if c >= 0.0 else -_C_STEP
        with np.errstate(over="ignore", invalid="ignore"):
            out = (c / (a - 1.0)) * (1.0 - np.power(alpha / (alpha + tt), r) * hyp2f1(r, b, c, tt / (alpha + tt)))
        bad = ~np.isfinite(out)
        if np.any(bad):
            LOGGER.warning(
                "BGNBD.expected_number_of_transactions: 2F1 returned %d non-finite values; "
                "those entries fall back to 0",
                int(bad.sum()),
            )
        return np.maximum(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    def probability_of_n_purchases(self, n: Any, t: Any) -> np.ndarray:
        """``P(X(t) = n)``: the BG/NBD marginal count distribution.

        ::

            P(X(t)=n) = B(a,b+n)/B(a,b) * Gamma(r+n)/(Gamma(r) n!) * (alpha/(alpha+t))^r * (t/(alpha+t))^n
                      + 1{n>0} * B(a+1,b+n-1)/B(a,b) * P(N_nbd(t) >= n)

        The first term is "alive throughout ``[0, t]`` having bought ``n`` times"; the
        second is "bought ``n`` times then died", whose probability needs only that the
        ``n``-th purchase happened before ``t`` -- an event whose probability is the upper
        tail of the plain NBD count. Summing over ``n`` gives exactly 1, which the smoke
        test asserts.

        Parameters
        ----------
        n : int or array-like of int
            Transaction counts, broadcast against ``t``.
        t : float or array-like
            Observation window length.

        Returns
        -------
        numpy.ndarray
            Probabilities, broadcast to the common shape of ``n`` and ``t``.
        """
        r, alpha, a, b = self._unpack()
        n_arr, t_arr = np.broadcast_arrays(
            np.asarray(n, dtype=np.float64), np.asarray(t, dtype=np.float64)
        )
        if np.any(n_arr < 0) or np.any(np.abs(n_arr - np.round(n_arr)) > 1e-9):
            raise ValueError("n must contain non-negative integers")
        if np.any(t_arr <= 0):
            raise ValueError("t must be positive")
        n_arr = np.round(n_arr)
        shape = n_arr.shape
        flat_n = n_arr.ravel()
        flat_t = t_arr.ravel()

        log_ratio_alpha = r * (np.log(alpha) - np.log(alpha + flat_t))
        log_z = np.log(flat_t) - np.log(alpha + flat_t)

        # B(a, b+n)/B(a, b) in logs
        log_beta = gammaln(a + b) - gammaln(b) + gammaln(b + flat_n) - gammaln(a + b + flat_n)
        log_nbd = gammaln(r + flat_n) - gammaln(r) - gammaln(flat_n + 1.0) + log_ratio_alpha + flat_n * log_z
        term1 = np.exp(log_beta + log_nbd)

        # P(N_nbd(t) >= n) = 1 - sum_{j=0}^{n-1} P(N = j), built once on a shared grid
        k_max = int(flat_n.max())
        if k_max > 0:
            js = np.arange(k_max, dtype=np.float64)  # j = 0 .. n_max-1
            log_pj = (
                gammaln(r + js)[None, :] - gammaln(r) - gammaln(js + 1.0)[None, :]
                + log_ratio_alpha[:, None] + js[None, :] * log_z[:, None]
            )
            cdf = np.cumsum(np.exp(log_pj), axis=1)  # cdf[:, k] = P(N <= k)
            idx = np.clip(flat_n.astype(int) - 1, 0, k_max - 1)
            tail = 1.0 - np.take_along_axis(cdf, idx[:, None], axis=1).ravel()
            tail = np.clip(tail, 0.0, 1.0)
            log_beta2 = np.log(a) - np.log(np.maximum(b + flat_n - 1.0, _EPS)) + log_beta
            term2 = np.where(flat_n > 0, np.exp(log_beta2) * tail, 0.0)
        else:
            term2 = np.zeros_like(term1)

        return np.clip(term1 + term2, 0.0, 1.0).reshape(shape)

    def goodness_of_fit(self, frequency: Any, T: Any, max_x: int = 7) -> pd.DataFrame:
        """Observed vs model-expected repeat-transaction histogram.

        The single most informative diagnostic for a BG/NBD fit: the model can hit the mean
        and still get the *shape* of the frequency distribution badly wrong. Expected counts
        are obtained by evaluating :meth:`probability_of_n_purchases` at each customer's own
        observation age ``T`` and summing, so heterogeneous observation windows are handled
        exactly rather than by plugging in a mean ``T``.

        Parameters
        ----------
        frequency : array-like of shape (n,)
            Observed repeat-transaction counts.
        T : array-like of shape (n,)
            Observation age per customer.
        max_x : int, default 7
            Largest count shown individually; everything above is pooled into a final
            ``">= max_x"`` bucket so the table's columns sum to ``n``.

        Returns
        -------
        pandas.DataFrame
            Columns ``n_transactions`` (str), ``observed``, ``expected``,
            ``observed_pct``, ``expected_pct``, ``abs_error_pct``.
        """
        x = _as_float_1d(frequency, "frequency")
        age = _as_float_1d(T, "T")
        if x.shape != age.shape:
            raise ValueError("frequency and T must share a length")
        if max_x < 1:
            raise ValueError("max_x must be >= 1")

        grid = np.arange(max_x + 1, dtype=np.float64)
        probs = self.probability_of_n_purchases(grid[None, :], age[:, None])  # (n, max_x+1)
        expected = probs.sum(axis=0)
        observed = np.array([float((x == k).sum()) for k in grid])
        # pool the tail so both columns total n
        n_total = float(x.size)
        labels = [str(int(k)) for k in grid[:-1]] + [f">={int(grid[-1])}"]
        observed_pooled = np.append(observed[:-1], n_total - observed[:-1].sum())
        expected_pooled = np.append(expected[:-1], n_total - expected[:-1].sum())
        expected_pooled = np.maximum(expected_pooled, 0.0)

        out = pd.DataFrame(
            {
                "n_transactions": labels,
                "observed": observed_pooled,
                "expected": expected_pooled,
            }
        )
        out["observed_pct"] = 100.0 * out["observed"] / n_total
        out["expected_pct"] = 100.0 * out["expected"] / n_total
        out["abs_error_pct"] = (out["observed_pct"] - out["expected_pct"]).abs()
        return out

    def log_likelihood(self, frequency: Any, recency: Any, T: Any) -> float:
        """Total log-likelihood of an arbitrary sample under the fitted parameters.

        Parameters
        ----------
        frequency, recency, T : array-like of shape (n,)
            RFM summary to score; typically a holdout cohort.

        Returns
        -------
        float
            Sum of per-customer log-likelihoods (higher is better).
        """
        if not hasattr(self, "params_"):
            raise RuntimeError("BGNBD is not fitted; call fit() first")
        x, t_x, age = _check_rfm(frequency, recency, T)
        log_params = np.log(np.array([self.params_[k] for k in ("r", "alpha", "a", "b")]))
        w = np.ones_like(x)
        obj, _ = self._objective(log_params, x, t_x, age, w, float(w.sum()), 0.0)
        return float(-obj * x.size)

    def score(self, frequency: Any, recency: Any, T: Any) -> float:
        """Mean log-likelihood per customer (an sklearn-style "higher is better" score)."""
        x = _as_float_1d(frequency, "frequency")
        return self.log_likelihood(frequency, recency, T) / max(x.size, 1)

    def summary(self) -> pd.DataFrame:
        """One-row summary of the fit, for reports and model cards.

        Returns
        -------
        pandas.DataFrame
            Columns ``r``, ``alpha``, ``a``, ``b``, ``purchase_rate_mean``,
            ``dropout_prob_mean``, ``log_likelihood``, ``n``, ``converged``.
        """
        r, alpha, a, b = self._unpack()
        return pd.DataFrame(
            [
                {
                    "r": r,
                    "alpha": alpha,
                    "a": a,
                    "b": b,
                    "purchase_rate_mean": r / alpha,          # E[lambda]
                    "dropout_prob_mean": a / (a + b),         # E[p]
                    "log_likelihood": self.log_likelihood_,
                    "n": self.n_,
                    "converged": self.converged_,
                }
            ]
        )


# ======================================================================================
# Gamma-Gamma
# ======================================================================================
class GammaGamma(BaseEstimator):
    """Gamma-Gamma model of monetary value (Fader & Hardie, 2013).

    Each of a customer's transactions has value ``z ~ Gamma(shape=p, rate=nu_i)``, and the
    customer-level rate is itself ``nu_i ~ Gamma(shape=q, rate=v)``. Integrating ``nu_i``
    out gives the marginal density of the *observed average* ``m_x`` over ``x``
    transactions, which is what is maximised here::

        log f(m_x | x) = gammaln(p*x + q) - gammaln(p*x) - gammaln(q)
                       + q*log(v) + (p*x - 1)*log(m_x) + p*x*log(x)
                       - (p*x + q) * log(v + m_x * x)

    Fitted by L-BFGS-B on ``log(p, q, v)`` with an analytic gradient, on the subset of
    customers with ``frequency > 0`` (a customer with no repeat purchases carries no
    information about spend).

    The model's load-bearing assumption is that **transaction frequency and monetary value
    are independent**. :meth:`fit` measures the empirical correlation and logs a warning
    when ``|corr| > independence_tol``, because a violated assumption here biases every
    downstream CLV number in the same direction.

    Parameters
    ----------
    penalizer : float, default 1e-4
        L2 coefficient on the log-parameters, added to the mean negative log-likelihood.
    n_restarts : int, default 5
        Starting points for the optimiser; the first four are fixed.
    max_iter : int, default 2000
        L-BFGS-B iteration cap per restart.
    tol : float, default 1e-10
        ``ftol`` for L-BFGS-B.
    independence_tol : float, default 0.2
        Absolute Pearson correlation between frequency and monetary value above which the
        independence assumption is reported as violated.
    random_state : int or numpy.random.Generator or None, optional
        Seed for jittered restarts beyond the fixed four. ``None`` uses a fixed internal
        seed, so an unseeded fit is still reproducible.

    Attributes
    ----------
    params_ : dict
        Fitted parameters with keys ``"p"``, ``"q"``, ``"v"``.
    freq_monetary_corr_ : float
        Pearson correlation between ``frequency`` and ``monetary_value`` on the fitting
        subset -- the independence diagnostic.
    independence_ok_ : bool
        ``abs(freq_monetary_corr_) <= independence_tol``.
    population_mean_profit_ : float
        ``p * v / (q - 1)``, the prior mean spend per transaction; ``nan`` if ``q <= 1``.
    log_likelihood_ : float
        Total log-likelihood at the optimum.
    n_ : int
        Number of customers with ``frequency > 0`` used in the fit.
    """

    def __init__(
        self,
        penalizer: float = 1e-4,
        *,
        n_restarts: int = 5,
        max_iter: int = 2000,
        tol: float = 1e-10,
        independence_tol: float = 0.2,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.penalizer = penalizer
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.independence_tol = independence_tol
        self.random_state = random_state

    # ---------------------------------------------------------------- likelihood
    @staticmethod
    def _objective(
        log_params: np.ndarray,
        x: np.ndarray,
        m: np.ndarray,
        w: np.ndarray,
        w_sum: float,
        penalizer: float,
    ) -> tuple[float, np.ndarray]:
        """Weighted-mean negative log-likelihood and gradient w.r.t. ``log(p, q, v)``.

        Parameters
        ----------
        log_params : numpy.ndarray of shape (3,)
            ``log([p, q, v])``.
        x : numpy.ndarray of shape (n,)
            Transaction counts (all strictly positive).
        m : numpy.ndarray of shape (n,)
            Observed average transaction value (all strictly positive).
        w : numpy.ndarray of shape (n,)
            Case weights.
        w_sum : float
            ``w.sum()``.
        penalizer : float
            L2 coefficient.

        Returns
        -------
        tuple of (float, numpy.ndarray)
            Objective and its 3-vector gradient.
        """
        p, q, v = np.exp(log_params)
        px = p * x
        vmx = v + m * x
        ll = (
            gammaln(px + q) - gammaln(px) - gammaln(q)
            + q * np.log(v) + (px - 1.0) * np.log(m) + px * np.log(x)
            - (px + q) * np.log(vmx)
        )
        if not np.all(np.isfinite(ll)):
            return 1e12, np.zeros(3)
        obj = -float(np.dot(w, ll)) / w_sum + penalizer * float(np.dot(log_params, log_params))

        d_p = x * (digamma(px + q) - digamma(px) + np.log(m) + np.log(x) - np.log(vmx))
        d_q = digamma(px + q) - digamma(q) + np.log(v) - np.log(vmx)
        d_v = q / v - (px + q) / vmx
        grad_nat = np.array([np.dot(w, d_p), np.dot(w, d_q), np.dot(w, d_v)]) / w_sum
        grad = -grad_nat * np.exp(log_params) + 2.0 * penalizer * log_params
        if not np.all(np.isfinite(grad)):
            return 1e12, np.zeros(3)
        return obj, grad

    # ---------------------------------------------------------------- fit
    def fit(self, frequency: Any, monetary_value: Any, weights: Any | None = None) -> GammaGamma:
        """Fit ``(p, q, v)`` on the customers with at least one repeat transaction.

        Parameters
        ----------
        frequency : array-like of shape (n,)
            Repeat-transaction counts. Rows with ``frequency <= 0`` are dropped (they carry
            no monetary information), as are rows with non-positive monetary value.
        monetary_value : array-like of shape (n,)
            Observed *average* value per repeat transaction, ``m_x``.
        weights : array-like of shape (n,), optional
            Non-negative case weights.

        Returns
        -------
        GammaGamma
            ``self``, fitted.

        Raises
        ------
        ValueError
            If fewer than two usable customers remain.
        """
        x_all = _as_float_1d(frequency, "frequency")
        m_all = _as_float_1d(monetary_value, "monetary_value")
        if x_all.shape != m_all.shape:
            raise ValueError("frequency and monetary_value must share a length")
        w_all = np.ones_like(x_all) if weights is None else _as_float_1d(weights, "weights")
        if w_all.shape != x_all.shape:
            raise ValueError("weights must align with frequency")

        mask = (x_all > 0) & (m_all > 0) & (w_all > 0)
        n_dropped = int((~mask).sum())
        if n_dropped:
            LOGGER.info(
                "GammaGamma: dropping %d/%d rows with frequency <= 0 or monetary_value <= 0 "
                "(they carry no information about spend)",
                n_dropped, x_all.size,
            )
        x, m, w = x_all[mask], m_all[mask], w_all[mask]
        if x.size < 2:
            raise ValueError("GammaGamma needs at least 2 customers with frequency > 0 and monetary_value > 0")

        # --- the assumption check that separates understanding the model from pasting it
        if np.std(x) > 0 and np.std(m) > 0:
            corr = float(np.corrcoef(x, m)[0, 1])
        else:
            corr = 0.0
        self.freq_monetary_corr_ = corr
        self.independence_ok_ = bool(abs(corr) <= float(self.independence_tol))
        if not self.independence_ok_:
            LOGGER.warning(
                "GammaGamma: corr(frequency, monetary_value) = %+.3f exceeds the tolerance of "
                "%.2f. The model ASSUMES these are independent; with %s correlation it will "
                "systematically %s the spend of frequent buyers. Consider modelling monetary "
                "value conditional on frequency, or segmenting before fitting.",
                corr, float(self.independence_tol),
                "positive" if corr > 0 else "negative",
                "under-estimate" if corr > 0 else "over-estimate",
            )

        w_sum = float(w.sum())
        args = (x, m, w, w_sum, float(self.penalizer))
        bounds = [(-_LOG_BOUND, _LOG_BOUND)] * 3
        rows: list[dict[str, Any]] = []
        best = None
        for i, init in enumerate(self._initial_points(m)):
            res = minimize(
                self._objective,
                np.log(np.clip(init, np.exp(-_LOG_BOUND), np.exp(_LOG_BOUND))),
                args=args,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(self.max_iter), "ftol": float(self.tol), "gtol": 1e-8},
            )
            pr = np.exp(res.x)
            rows.append(
                {
                    "restart": i,
                    "objective": float(res.fun) if np.isfinite(res.fun) else np.inf,
                    "p": pr[0], "q": pr[1], "v": pr[2], "success": bool(res.success),
                }
            )
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        if best is None:
            raise RuntimeError("GammaGamma: every optimisation restart failed")

        p, q, v = (float(val) for val in np.exp(best.x))
        self.params_: dict[str, float] = {"p": p, "q": q, "v": v}
        self.converged_ = bool(best.success)
        self.n_ = int(x.size)
        self.n_dropped_ = n_dropped
        self.optim_result_ = best
        self.restart_table_ = pd.DataFrame(rows).sort_values("objective", ignore_index=True)
        penalty = float(self.penalizer) * float(np.dot(best.x, best.x))
        self.log_likelihood_ = float(-(best.fun - penalty) * w_sum)
        self.population_mean_profit_ = float(p * v / (q - 1.0)) if q > 1.0 else float("nan")

        LOGGER.info(
            "GammaGamma fit on n=%d: p=%.4f q=%.4f v=%.4f | E[M]_pop=%.3f | corr(f,m)=%+.3f | LL=%.2f",
            self.n_, p, q, v, self.population_mean_profit_, corr, self.log_likelihood_,
        )
        if q <= 1.0:
            LOGGER.warning(
                "GammaGamma: fitted q=%.4f <= 1. E[1/nu] is infinite, so the expected spend is "
                "undefined; conditional_expected_average_profit will fall back to the observed "
                "customer mean. This usually means the monetary distribution is too heavy-tailed "
                "-- winsorise the input or drop outliers.",
                q,
            )
        return self

    def _initial_points(self, m: np.ndarray) -> list[np.ndarray]:
        """Starting points for the optimiser, scaled to the observed spend level.

        Parameters
        ----------
        m : numpy.ndarray
            Observed average transaction values (used to put ``v`` on the right scale).

        Returns
        -------
        list of numpy.ndarray
            Each a 3-vector ``[p, q, v]`` on the natural scale.
        """
        m_bar = float(np.mean(m))
        fixed = [
            np.array([1.0, 2.0, max(m_bar, 1e-3)]),
            np.array([6.0, 4.0, max(m_bar * 0.5, 1e-3)]),
            np.array([2.0, 2.0, 1.0]),
            np.array([0.5, 3.0, max(m_bar * 2.0, 1e-3)]),
        ]
        n_extra = max(int(self.n_restarts) - len(fixed), 0)
        if n_extra:
            # See BGNBD._initial_points: an unseeded fit is still reproducible.
            rng = as_rng(_DEFAULT_JITTER_SEED if self.random_state is None else self.random_state)
            base = np.log(np.array([2.0, 3.0, max(m_bar, 1e-3)]))
            fixed.extend(np.exp(base + rng.normal(scale=0.8, size=(n_extra, 3))))
        return fixed[: max(int(self.n_restarts), 1)]

    # ---------------------------------------------------------------- predictions
    def conditional_expected_average_profit(self, frequency: Any, monetary_value: Any) -> np.ndarray:
        """Posterior expected value of a future transaction, shrunk towards the population.

        **Derivation** (do not take this on faith; it is short). The total spend over ``x``
        transactions is ``x * m_x ~ Gamma(shape=p*x, rate=nu)``, and the prior is
        ``nu ~ Gamma(shape=q, rate=v)``. Gamma is conjugate for its own rate, so::

            nu | x, m_x  ~  Gamma(shape = p*x + q,  rate = v + x*m_x)

        A future transaction has mean ``p / nu``, so the posterior expected spend is
        ``p * E[1/nu]``. For ``G ~ Gamma(k, theta)`` with ``k > 1``,
        ``E[1/G] = theta/(k-1)``, hence::

            E[M | x, m_x] = p * (v + x*m_x) / (p*x + q - 1)

        Splitting the numerator shows this is a **credibility-weighted average** whose two
        weights sum to exactly 1::

            E[M | x, m_x] = (q - 1)/(p*x + q - 1) * [p*v/(q - 1)]  +  (p*x)/(p*x + q - 1) * m_x
                          = (1 - w) * population_mean         +  w * observed_mean,
              with w = p*x / (p*x + q - 1)

        As ``x`` grows, ``w -> 1`` and the estimate converges to the customer's own average;
        for ``x = 1`` it sits close to the population mean. Note that the frequently quoted
        form ``((p*v + x*m_x)/(p*x + q - 1)) * ((q-1)/(p*x + q - 1))`` applies the shrinkage
        factor **twice** and is wrong -- it does not reduce to ``m_x`` as ``x -> inf``.

        Requires ``q > 1`` for ``E[1/nu]`` to exist. If the fit produced ``q <= 1`` the
        method logs a warning and returns the observed ``monetary_value`` unshrunk, which is
        the only defensible fallback.

        Parameters
        ----------
        frequency : array-like of shape (n,)
            Repeat-transaction counts.
        monetary_value : array-like of shape (n,)
            Observed average transaction value.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Expected value of the customer's next transaction. Customers with
            ``frequency == 0`` receive the population mean, which is all the model knows
            about them.
        """
        if not hasattr(self, "params_"):
            raise RuntimeError("GammaGamma is not fitted; call fit() first")
        p, q, v = self.params_["p"], self.params_["q"], self.params_["v"]
        x = _as_float_1d(frequency, "frequency")
        m = _as_float_1d(monetary_value, "monetary_value")
        if x.shape != m.shape:
            raise ValueError("frequency and monetary_value must share a length")

        if q <= 1.0:
            LOGGER.warning(
                "GammaGamma.conditional_expected_average_profit: q=%.4f <= 1 so the posterior "
                "mean does not exist; returning the observed monetary_value unshrunk",
                q,
            )
            return np.maximum(m, 0.0)

        population_mean = p * v / (q - 1.0)
        denom = p * x + q - 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(denom > _EPS, p * x / denom, 0.0)
        weight = np.clip(weight, 0.0, 1.0)
        out = (1.0 - weight) * population_mean + weight * np.where(x > 0, m, population_mean)
        return np.maximum(np.nan_to_num(out, nan=population_mean), 0.0)

    def log_likelihood(self, frequency: Any, monetary_value: Any) -> float:
        """Total log-likelihood of a sample under the fitted parameters.

        Parameters
        ----------
        frequency, monetary_value : array-like of shape (n,)
            Sample to score; rows with non-positive frequency or value are ignored.

        Returns
        -------
        float
            Sum of per-customer log-likelihoods over the usable rows.
        """
        if not hasattr(self, "params_"):
            raise RuntimeError("GammaGamma is not fitted; call fit() first")
        x = _as_float_1d(frequency, "frequency")
        m = _as_float_1d(monetary_value, "monetary_value")
        mask = (x > 0) & (m > 0)
        if not mask.any():
            return float("nan")
        xs, ms = x[mask], m[mask]
        w = np.ones_like(xs)
        log_params = np.log(np.array([self.params_[k] for k in ("p", "q", "v")]))
        obj, _ = self._objective(log_params, xs, ms, w, float(w.sum()), 0.0)
        return float(-obj * xs.size)

    def score(self, frequency: Any, monetary_value: Any) -> float:
        """Mean log-likelihood per usable customer (higher is better)."""
        x = _as_float_1d(frequency, "frequency")
        m = _as_float_1d(monetary_value, "monetary_value")
        n = int(((x > 0) & (m > 0)).sum())
        return self.log_likelihood(frequency, monetary_value) / max(n, 1)

    def summary(self) -> pd.DataFrame:
        """One-row summary of the fit, including the independence diagnostic.

        Returns
        -------
        pandas.DataFrame
            Columns ``p``, ``q``, ``v``, ``population_mean_profit``,
            ``freq_monetary_corr``, ``independence_ok``, ``log_likelihood``, ``n``.
        """
        if not hasattr(self, "params_"):
            raise RuntimeError("GammaGamma is not fitted; call fit() first")
        return pd.DataFrame(
            [
                {
                    "p": self.params_["p"],
                    "q": self.params_["q"],
                    "v": self.params_["v"],
                    "population_mean_profit": self.population_mean_profit_,
                    "freq_monetary_corr": self.freq_monetary_corr_,
                    "independence_ok": self.independence_ok_,
                    "log_likelihood": self.log_likelihood_,
                    "n": self.n_,
                }
            ]
        )


# ======================================================================================
# probabilistic CLV
# ======================================================================================
def probabilistic_clv(
    summary: pd.DataFrame,
    horizon_months: int,
    discount_rate: float = 0.01,
    margin: float = 0.3,
    *,
    bgnbd: BGNBD | None = None,
    gamma_gamma: GammaGamma | None = None,
    penalizer: float = 1e-4,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Combine BG/NBD and Gamma-Gamma into a discounted expected-value estimate per customer.

    The two models factorise CLV into "how many more purchases" and "how much each is
    worth", which are then multiplied by a gross-margin rate and discounted::

        CLV_discounted = margin * E[M | x, m_x] * sum_{t=1}^{H} dY_t * (1 + d)^-t
        where dY_t = E[Y(t) | RFM] - E[Y(t-1) | RFM]

    Discounting is applied to the expected transactions **month by month**, not to the
    horizon total: ``dY_t`` is the expected number of purchases falling in month ``t``, and
    it is discounted at ``(1+d)^-t``. Cash is taken to arrive at the end of its month, so
    ``t`` runs from 1 to ``horizon_months`` -- the same convention as
    :func:`clv_from_survival` and :func:`annuity_factor`. Because BG/NBD purchases are
    front-loaded (a customer alive today is more likely to buy soon than later), the
    discount applied is slightly *less* severe than a flat end-of-horizon discount.

    The ``summary`` frame's ``recency`` and ``T`` are interpreted in **months**, since the
    discount rate is monthly.

    Parameters
    ----------
    summary : pandas.DataFrame
        RFM summary with columns ``customer_id``, ``frequency``, ``recency``, ``T``,
        ``monetary_value``. ``recency`` is the *time of* the last repeat purchase, not the
        gap since it.
    horizon_months : int
        Forecast horizon in months, must be >= 1.
    discount_rate : float, default 0.01
        Monthly discount rate ``d``.
    margin : float, default 0.3
        Gross-margin fraction applied to revenue, in ``[0, 1]``.
    bgnbd : BGNBD, optional
        A pre-fitted transaction model. Fitted on ``summary`` when omitted.
    gamma_gamma : GammaGamma, optional
        A pre-fitted monetary model. Fitted on ``summary`` when omitted.
    penalizer : float, default 1e-4
        Penalizer used for any model that has to be fitted here.
    random_state : int or numpy.random.Generator or None, optional
        Seed threaded into the models' jittered optimiser restarts.

    Returns
    -------
    pandas.DataFrame
        One row per customer, in the input's order, with columns:

        ``customer_id``
            Identifier copied from ``summary``.
        ``p_alive``
            Posterior probability the customer is still active.
        ``expected_transactions``
            Expected repeat purchases over the whole horizon (undiscounted count).
        ``expected_avg_profit``
            Shrunken expected revenue per transaction (before margin).
        ``clv``
            ``margin * expected_avg_profit * expected_transactions``, undiscounted.
        ``clv_discounted``
            The same with monthly discounting applied to the timing of purchases.

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If ``horizon_months < 1`` or ``margin`` is outside ``[0, 1]``.

    Notes
    -----
    Cost is ``horizon_months`` evaluations of ``2F1`` over ``n`` rows. At ``n = 100k`` and
    ``H = 12`` that is a couple of seconds; callers scoring a full panel should score the
    decision-point sample instead (SPEC_PERF section 1).
    """
    required = ["customer_id", "frequency", "recency", "T", "monetary_value"]
    missing = [c for c in required if c not in summary.columns]
    if missing:
        raise KeyError(f"summary is missing required columns: {missing}")
    horizon_months = int(horizon_months)
    if horizon_months < 1:
        raise ValueError("horizon_months must be >= 1")
    if not (0.0 <= margin <= 1.0):
        raise ValueError("margin must lie in [0, 1]")
    if discount_rate <= -1.0:
        raise ValueError("discount_rate must be greater than -1")

    freq = _as_float_1d(summary["frequency"].to_numpy(), "frequency")
    rec = _as_float_1d(summary["recency"].to_numpy(), "recency")
    age = _as_float_1d(summary["T"].to_numpy(), "T")
    mon = _as_float_1d(summary["monetary_value"].to_numpy(), "monetary_value")

    if bgnbd is None:
        bgnbd = BGNBD(penalizer=penalizer, random_state=random_state).fit(freq, rec, age)
    if gamma_gamma is None:
        gamma_gamma = GammaGamma(penalizer=penalizer, random_state=random_state).fit(freq, mon)

    p_alive = bgnbd.probability_alive(freq, rec, age)
    avg_profit = gamma_gamma.conditional_expected_average_profit(freq, mon)

    # cumulative expected transactions at each month boundary -> per-month increments
    cumulative = np.zeros((freq.size, horizon_months + 1), dtype=np.float64)
    for t in range(1, horizon_months + 1):
        cumulative[:, t] = bgnbd.conditional_expected_transactions(float(t), freq, rec, age)
    increments = np.diff(cumulative, axis=1)
    # the closed form is monotone in t up to float noise; clip the last ulp
    increments = np.maximum(increments, 0.0)

    discount = (1.0 + discount_rate) ** (-np.arange(1, horizon_months + 1, dtype=np.float64))
    expected_transactions = increments.sum(axis=1)
    discounted_transactions = increments @ discount

    unit_margin = float(margin) * avg_profit
    out = pd.DataFrame(
        {
            "customer_id": summary["customer_id"].to_numpy(),
            "p_alive": p_alive,
            "expected_transactions": expected_transactions,
            "expected_avg_profit": avg_profit,
            "clv": unit_margin * expected_transactions,
            "clv_discounted": unit_margin * discounted_transactions,
        }
    )
    LOGGER.info(
        "probabilistic_clv: n=%d horizon=%dm d=%.4f margin=%.2f | mean CLV=%.2f -> discounted %.2f "
        "(%.1f%% of undiscounted) | mean p_alive=%.3f",
        len(out), horizon_months, discount_rate, margin,
        float(out["clv"].mean()), float(out["clv_discounted"].mean()),
        100.0 * float(out["clv_discounted"].sum()) / max(float(out["clv"].sum()), _EPS),
        float(out["p_alive"].mean()),
    )
    return out


# ======================================================================================
# the bridge to the causal / survival layer
# ======================================================================================
def clv_from_survival(
    survival_curve: np.ndarray,
    monthly_margin: np.ndarray | float,
    discount_rate: float = 0.01,
) -> np.ndarray:
    """Discounted expected margin implied by a survival curve -- the survival/value bridge.

    ::

        CLV_i = sum_{t=1}^{H} S_i(t) * m_i * (1 + d)^-t

    **Indexing convention (important).** Column ``j`` of ``survival_curve`` is ``S(t)`` at
    ``t = j + 1``: the probability of surviving *through* month ``j+1``. There is no
    ``S(0) = 1`` column, because a customer known to be alive at the decision point
    contributes nothing to discount. Consequently ``t`` starts at **1**, both in the
    survival index and in the discount exponent, and margin for month ``t`` is earned only
    if the customer is still there at the end of it.

    This matches ``prism.models.survival.DiscreteTimeHazardModel.predict_rmst``, which is
    documented as the plain sum of the same ``(n, horizon)`` survival matrix: setting
    ``discount_rate = 0`` and ``monthly_margin = 1`` here reproduces RMST exactly. That
    equivalence is what makes ``tau_value`` and ``tau_rmst`` in
    :class:`prism.causal.survival_uplift.CausalSurvivalUplift` two views of one quantity.

    Parameters
    ----------
    survival_curve : numpy.ndarray of shape (n, H) or (H,)
        Survival probabilities. A 1-D input is treated as a single customer's curve and the
        result has shape ``(1,)``. Values are clipped into ``[0, 1]``; entries outside that
        range, or a curve that is not monotone non-increasing, are computed anyway but
        logged as a warning, since both indicate a broken upstream survival model.
    monthly_margin : numpy.ndarray of shape (n,) or float
        Gross margin earned per surviving month. A scalar is broadcast to every customer.
    discount_rate : float, default 0.01
        Monthly discount rate ``d``; must be greater than -1.

    Returns
    -------
    numpy.ndarray of shape (n,)
        Discounted expected margin over the horizon, float64.

    Raises
    ------
    ValueError
        If shapes disagree, the curve is not 1-D/2-D, or ``discount_rate <= -1``.

    Examples
    --------
    A customer certain to survive the whole horizon is worth an ordinary annuity:

    >>> s = np.ones((1, 12))
    >>> float(clv_from_survival(s, 100.0, 0.01)[0])            # doctest: +ELLIPSIS
    1125.50...
    >>> float(100.0 * annuity_factor(12, 0.01))                # doctest: +ELLIPSIS
    1125.50...
    """
    curve = np.asarray(survival_curve, dtype=np.float64)
    if curve.ndim == 1:
        curve = curve[None, :]
    if curve.ndim != 2:
        raise ValueError(f"survival_curve must be 1-D or 2-D, got shape {curve.shape}")
    n, horizon = curve.shape
    if horizon == 0:
        return np.zeros(n, dtype=np.float64)
    if discount_rate <= -1.0:
        raise ValueError("discount_rate must be greater than -1")
    if not np.all(np.isfinite(curve)):
        raise ValueError("survival_curve contains NaN or inf")

    # A survival curve must be monotone non-increasing and inside [0, 1] (SPEC_PERF
    # section 4 requires the producers to guarantee both). Clipping and summing a curve
    # that violates either would return a plausible-looking number for a broken upstream
    # model, so say so loudly instead. This is a warning rather than an exception because
    # this function is a consumer, not the contract owner -- but a silent pass here is
    # exactly how an upstream monotonicity bug reaches a business decision unnoticed.
    n_bad_range = int(np.count_nonzero((curve < -1e-9) | (curve > 1.0 + 1e-9)))
    if n_bad_range:
        LOGGER.warning(
            "clv_from_survival: %d of %d survival entries lie outside [0, 1] (min %.4g, "
            "max %.4g); clipping. Check the model that produced this curve.",
            n_bad_range, curve.size, float(curve.min()), float(curve.max()),
        )
    if horizon > 1:
        rises = np.diff(curve, axis=1) > 1e-9
        n_rising = int(np.count_nonzero(rises))
        if n_rising:
            LOGGER.warning(
                "clv_from_survival: the survival curve INCREASES at %d of %d steps "
                "(worst rise %.4g, affecting %d of %d rows). S(t) must be monotone "
                "non-increasing; the result is still computed but the upstream "
                "predict_survival is violating its contract.",
                n_rising, rises.size, float(np.diff(curve, axis=1).max()),
                int(np.count_nonzero(rises.any(axis=1))), n,
            )
    curve = np.clip(curve, 0.0, 1.0)

    margin = np.asarray(monthly_margin, dtype=np.float64)
    if margin.ndim == 0:
        margin = np.full(n, float(margin))
    else:
        margin = margin.ravel()
        if margin.shape[0] != n:
            raise ValueError(f"monthly_margin has length {margin.shape[0]} but survival_curve has {n} rows")
    if not np.all(np.isfinite(margin)):
        raise ValueError("monthly_margin contains NaN or inf")

    # t = 1 .. H, cash at the end of each month
    discount = (1.0 + discount_rate) ** (-np.arange(1, horizon + 1, dtype=np.float64))
    return np.asarray((curve @ discount) * margin, dtype=np.float64)


# ======================================================================================
# DeepCLV
# ======================================================================================
class DeepCLV(BaseEstimator):
    """Two-head neural CLV model: ``E[value] = P(alive) * E[value | alive]``.

    A discriminative alternative to BG/NBD for the contractual case, where the retention
    label is actually observed and the 23 panel features carry signal that an RFM summary
    throws away. One shared trunk feeds two heads:

    * **head 1** -- a logit for ``P(alive)``, i.e. retained over the horizon, trained with
      binary cross-entropy;
    * **head 2** -- expected value *conditional on being alive*, trained only on the alive
      rows (weighted by the alive indicator) so it never learns to predict the structural
      zeros that head 1 already explains.

    Factorising this way matters: value is a zero-inflated, heavy-tailed variable, and a
    single regression head spends its capacity straddling the spike at zero instead of
    ranking the survivors. The heads share a trunk so the representation is learned once.

    Value-head losses (``value_loss``)
    ----------------------------------
    ``"logcosh"`` (default)
        Log-cosh on ``log1p(value)``. Smooth, quadratic near zero and linear in the tail,
        so a single whale cannot dominate the gradient. Fast.
    ``"tweedie"``
        Tweedie deviance with power ``tweedie_power`` in ``(1, 2)`` -- the compound
        Poisson-Gamma family, which has an atom at zero and a continuous positive part.
        This is the textbook likelihood for "sometimes zero, otherwise skewed-positive"
        money, and models the mean directly via a log link. Slower than log-cosh.

    Both are trained jointly with the BCE head, ``loss = bce + value_weight * value_loss``.
    After training, a single positive scalar rescales predictions so their mean matches the
    training mean -- a mean calibration that removes the retransformation bias of the log
    link. The factor is exposed as ``calibration_``.

    Backends
    --------
    ``backend="torch"`` uses the network above. ``backend="sklearn"`` (or ``"auto"`` when
    torch is missing) uses the documented fallback: a gradient-boosted classifier for
    ``P(alive)`` multiplied by a gradient-boosted regressor for ``log1p(value)`` fitted on
    the alive rows, with the same mean calibration. The fallback keeps the same two-stage
    factorisation, so predictions remain interpretable in the same way.

    Parameters
    ----------
    horizon : int, default 12
        Number of months the ``value`` label covers. Used only by
        :meth:`predict_discounted`, which spreads the predicted value across the horizon.
    hidden : tuple of int, default (128, 64)
        Trunk layer widths (torch backend only).
    epochs : int, default 60
        Training epochs (torch backend only).
    lr : float, default 1e-3
        Adam learning rate (torch backend only).
    discount_rate : float, default 0.01
        Monthly discount rate. **Does not** affect :meth:`predict`, which stays on the
        training target's scale so it can be compared against ``value`` directly; it is
        applied by :meth:`predict_discounted`.
    backend : {"auto", "torch", "sklearn"}, default "auto"
        ``"auto"`` prefers torch and silently falls back. ``"torch"`` raises a helpful
        ImportError if torch is absent.
    random_state : int or numpy.random.Generator or None, optional
        Seed. With a fixed seed and a fixed thread count, two fits on the same data give
        bit-identical predictions.
    batch_size : int, default 256
        Minibatch size (torch backend only).
    weight_decay : float, default 1e-4
        Adam L2 regularisation (torch backend only).
    value_loss : {"logcosh", "tweedie"}, default "logcosh"
        Regression loss for head 2, see above.
    tweedie_power : float, default 1.5
        Tweedie variance power, must lie strictly in ``(1, 2)``.
    value_weight : float, default 1.0
        Weight on the value loss relative to the BCE term.
    dropout : float, default 0.0
        Dropout probability between trunk layers (torch backend only).
    n_estimators : int, default 300
        Trees per stage in the sklearn fallback.
    verbose : bool, default False
        Log the training loss every few epochs.

    Attributes
    ----------
    backend_ : str
        The backend actually used, ``"torch"`` or ``"sklearn"``.
    calibration_ : float
        Multiplicative mean-calibration factor learned on the training set.
    alive_rate_ : float
        Fraction of training rows labelled alive.
    train_loss_ : list of float
        Mean epoch loss (torch backend only).
    n_features_in_ : int
        Number of input columns seen during fit.

    Notes
    -----
    Determinism relies on a fixed BLAS thread count; torch's CPU reductions are not
    order-invariant across different ``torch.set_num_threads`` values. Within one process
    the model is bit-reproducible.
    """

    def __init__(
        self,
        horizon: int = 12,
        hidden: Sequence[int] = (128, 64),
        epochs: int = 60,
        lr: float = 1e-3,
        discount_rate: float = 0.01,
        backend: str = "auto",
        random_state: int | np.random.Generator | None = None,
        *,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        value_loss: str = "logcosh",
        tweedie_power: float = 1.5,
        value_weight: float = 1.0,
        dropout: float = 0.0,
        n_estimators: int = 300,
        verbose: bool = False,
    ) -> None:
        self.horizon = horizon
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.discount_rate = discount_rate
        self.backend = backend
        self.random_state = random_state
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.value_loss = value_loss
        self.tweedie_power = tweedie_power
        self.value_weight = value_weight
        self.dropout = dropout
        self.n_estimators = n_estimators
        self.verbose = verbose

    # ---------------------------------------------------------------- plumbing
    def _resolve_backend(self) -> str:
        """Decide which backend to use, honouring an explicit request.

        Returns
        -------
        str
            ``"torch"`` or ``"sklearn"``.
        """
        backend = str(self.backend).lower()
        if backend not in {"auto", "torch", "sklearn"}:
            raise ValueError(f"backend must be 'auto', 'torch' or 'sklearn', got {self.backend!r}")
        if backend == "torch":
            require("torch")  # raises with an install hint if truly missing
            return "torch"
        if backend == "sklearn":
            return "sklearn"
        if HAS_TORCH:
            return "torch"
        LOGGER.info("DeepCLV: torch unavailable, falling back to the two-stage sklearn backend")
        return "sklearn"

    @staticmethod
    def _prepare(X: Any, value: Any, event_observed: Any | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate inputs and derive the alive label.

        The alive label is ``1 - event_observed`` when a survival indicator is supplied,
        following the PRISM convention that ``event_observed == 1`` means *churn was
        observed*. With no indicator, ``value > 0`` is used, which treats a zero-value
        horizon as death -- correct for a subscription panel where a churned customer bills
        nothing.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Design matrix.
        value : array-like of shape (n,)
            Realised value over the horizon, non-negative.
        event_observed : array-like of shape (n,) or None
            1 = churn observed, 0 = censored/retained.

        Returns
        -------
        tuple of numpy.ndarray
            ``(X, value, alive)`` as float64 arrays.
        """
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {Xa.shape}")
        y = _as_float_1d(value, "value")
        if y.shape[0] != Xa.shape[0]:
            raise ValueError(f"value has length {y.shape[0]} but X has {Xa.shape[0]} rows")
        if not np.all(np.isfinite(Xa)):
            raise ValueError("X contains NaN or inf; impute before fitting (see prism.data.features)")
        if np.any(y < 0):
            raise ValueError("value must be non-negative")
        if event_observed is None:
            alive = (y > 0).astype(np.float64)
        else:
            ev = _as_float_1d(event_observed, "event_observed")
            if ev.shape[0] != Xa.shape[0]:
                raise ValueError("event_observed must align with X")
            if not np.all(np.isin(ev, (0.0, 1.0))):
                raise ValueError("event_observed must contain only 0 and 1")
            alive = 1.0 - ev
        return Xa, y, alive

    # ---------------------------------------------------------------- fit
    def fit(self, X: Any, value: Any, event_observed: Any | None = None) -> DeepCLV:
        """Train the two-head model.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Numeric design matrix; encode categoricals first with
            :func:`prism.data.features.build_design_matrix`.
        value : array-like of shape (n,)
            Realised (non-negative) value over the horizon, e.g. the panel's ``value_h``.
        event_observed : array-like of shape (n,), optional
            Survival indicator, 1 = churn observed. When omitted the alive label is
            ``value > 0``.

        Returns
        -------
        DeepCLV
            ``self``, fitted.
        """
        Xa, y, alive = self._prepare(X, value, event_observed)
        self.n_features_in_ = int(Xa.shape[1])
        self.alive_rate_ = float(alive.mean())
        self.value_mean_ = float(y.mean())
        self.backend_ = self._resolve_backend()

        rng = as_rng(self.random_state)
        self._seed_ = int(rng.integers(0, 2**31 - 1))

        # standardise once; harmless for trees, essential for the net
        self._x_mean_ = Xa.mean(axis=0)
        self._x_std_ = Xa.std(axis=0)
        self._x_std_[self._x_std_ < 1e-8] = 1.0

        if self.alive_rate_ <= 0.0 or self.alive_rate_ >= 1.0:
            LOGGER.warning(
                "DeepCLV: the alive label is constant (%.0f%% alive); the classifier head is "
                "degenerate and predictions reduce to the value head alone",
                100.0 * self.alive_rate_,
            )

        if self.backend_ == "torch":
            self._fit_torch(Xa, y, alive, rng)
        else:
            self._fit_sklearn(Xa, y, alive)

        # mean calibration over ALL rows: removes the log-link retransformation bias
        raw_pred = self._raw_predict(Xa)
        total = float(raw_pred.sum())
        self.calibration_ = float(y.sum() / total) if total > _EPS else 1.0
        if not np.isfinite(self.calibration_) or self.calibration_ <= 0:
            self.calibration_ = 1.0
        LOGGER.info(
            "DeepCLV fit: backend=%s n=%d p=%d alive_rate=%.3f value_loss=%s calibration=%.4f",
            self.backend_, Xa.shape[0], Xa.shape[1], self.alive_rate_, self.value_loss, self.calibration_,
        )
        return self

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Standardise ``X`` with the statistics learned during fit."""
        return (X - self._x_mean_) / self._x_std_

    # ---------------------------------------------------------------- torch backend
    def _fit_torch(self, X: np.ndarray, y: np.ndarray, alive: np.ndarray, rng: np.random.Generator) -> None:
        """Train the shared-trunk two-head network.

        Parameters
        ----------
        X : numpy.ndarray of shape (n, p)
            Raw design matrix (standardised inside).
        y : numpy.ndarray of shape (n,)
            Value target.
        alive : numpy.ndarray of shape (n,)
            Binary retention label.
        rng : numpy.random.Generator
            Source of the minibatch permutations, so shuffling is seeded rather than
            dependent on torch's global state.
        """
        torch = require("torch")
        nn = torch.nn

        if str(self.value_loss).lower() not in {"logcosh", "tweedie"}:
            raise ValueError(f"value_loss must be 'logcosh' or 'tweedie', got {self.value_loss!r}")
        rho = float(self.tweedie_power)
        if str(self.value_loss).lower() == "tweedie" and not (1.0 < rho < 2.0):
            raise ValueError("tweedie_power must lie strictly between 1 and 2")

        torch.manual_seed(self._seed_)
        gen = torch.Generator().manual_seed(self._seed_)

        Xs = self._scale(X).astype(np.float32)
        n, p = Xs.shape
        widths = [int(h) for h in self.hidden] or [32]

        layers: list[Any] = []
        d_in = p
        for width in widths:
            layers.append(nn.Linear(d_in, width))
            layers.append(nn.ReLU())
            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(float(self.dropout)))
            d_in = width
        trunk = nn.Sequential(*layers)
        head_alive = nn.Linear(d_in, 1)
        head_value = nn.Linear(d_in, 1)

        # seeded init: torch's default reset_parameters() ignores our generator
        for module in list(trunk) + [head_alive, head_value]:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=float(np.sqrt(5.0)), generator=gen)
                nn.init.zeros_(module.bias)

        pos = alive > 0
        mean_alive_value = float(y[pos].mean()) if pos.any() else max(self.value_mean_, 1e-3)
        rate = float(np.clip(self.alive_rate_, 1e-3, 1 - 1e-3))
        with torch.no_grad():
            head_alive.bias.fill_(float(np.log(rate / (1.0 - rate))))
            # start the value head at the right order of magnitude so exp()/expm1() cannot blow up
            head_value.bias.fill_(
                float(np.log(max(mean_alive_value, 1e-3)))
                if self.value_loss == "tweedie"
                else float(np.log1p(max(mean_alive_value, 0.0)))
            )

        params = list(trunk.parameters()) + list(head_alive.parameters()) + list(head_value.parameters())
        opt = torch.optim.Adam(params, lr=float(self.lr), weight_decay=float(self.weight_decay))
        bce = nn.BCEWithLogitsLoss()

        Xt = torch.from_numpy(Xs)
        yt = torch.from_numpy(y.astype(np.float32))
        at = torch.from_numpy(alive.astype(np.float32))
        batch = max(int(self.batch_size), 1)
        log2 = float(np.log(2.0))
        self.train_loss_ = []

        for epoch in range(int(self.epochs)):
            perm = rng.permutation(n)
            running, seen = 0.0, 0
            for start in range(0, n, batch):
                idx = perm[start : start + batch]
                xb, yb, ab = Xt[idx], yt[idx], at[idx]
                z = trunk(xb)
                logit = head_alive(z).squeeze(-1)
                raw = head_value(z).squeeze(-1).clamp(-15.0, 15.0)

                loss_alive = bce(logit, ab)
                w_sum = ab.sum()
                if float(w_sum) < 1.0:
                    loss_value = raw.sum() * 0.0
                elif self.value_loss == "tweedie":
                    # deviance with a log link, written with exp() rather than pow(): much faster
                    dev = -yb * torch.exp((1.0 - rho) * raw) / (1.0 - rho) + torch.exp((2.0 - rho) * raw) / (2.0 - rho)
                    loss_value = (dev * ab).sum() / w_sum
                else:
                    resid = raw - torch.log1p(yb)
                    abs_r = resid.abs()
                    logcosh = torch.nn.functional.softplus(-2.0 * abs_r) + abs_r - log2
                    loss_value = (logcosh * ab).sum() / w_sum

                loss = loss_alive + float(self.value_weight) * loss_value
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                running += float(loss.detach()) * len(idx)
                seen += len(idx)
            self.train_loss_.append(running / max(seen, 1))
            if self.verbose and (epoch % 10 == 0 or epoch == int(self.epochs) - 1):
                LOGGER.info("DeepCLV epoch %3d/%d loss=%.5f", epoch + 1, int(self.epochs), self.train_loss_[-1])

        trunk.eval()
        self._torch_modules_ = (trunk, head_alive, head_value)

    def _torch_forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the trained network and return ``(p_alive, value_given_alive)`` uncalibrated."""
        torch = require("torch")
        trunk, head_alive, head_value = self._torch_modules_
        Xs = self._scale(X).astype(np.float32)
        with torch.no_grad():
            z = trunk(torch.from_numpy(Xs))
            p_alive = torch.sigmoid(head_alive(z).squeeze(-1)).numpy().astype(np.float64)
            raw = head_value(z).squeeze(-1).clamp(-15.0, 15.0).numpy().astype(np.float64)
        value = np.exp(raw) if self.value_loss == "tweedie" else np.expm1(raw)
        return p_alive, np.maximum(value, 0.0)

    # ---------------------------------------------------------------- sklearn backend
    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray, alive: np.ndarray) -> None:
        """Fit the two-stage gradient-boosted fallback (classifier x regressor).

        Parameters
        ----------
        X : numpy.ndarray of shape (n, p)
            Raw design matrix.
        y : numpy.ndarray of shape (n,)
            Value target.
        alive : numpy.ndarray of shape (n,)
            Binary retention label.
        """
        from prism.utils.optional import HAS_LIGHTGBM

        Xs = self._scale(X)
        kwargs: dict[str, Any] = {
            "n_estimators": int(self.n_estimators),
            "learning_rate": 0.05,
            "random_state": self._seed_,
        }
        if HAS_LIGHTGBM:
            # single-threaded + deterministic histogram construction => bit-reproducible
            kwargs.update(deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1)

        if 0.0 < self.alive_rate_ < 1.0:
            clf = best_gbm("classification", **kwargs)
            clf.fit(Xs, alive.astype(int))
            self._sk_classifier_ = clf
        else:
            self._sk_classifier_ = None

        pos = alive > 0
        if pos.sum() >= 10:
            reg = best_gbm("regression", **kwargs)
            reg.fit(Xs[pos], np.log1p(y[pos]))
            self._sk_regressor_ = reg
        else:
            LOGGER.warning("DeepCLV(sklearn): only %d alive rows, the value head falls back to a constant", int(pos.sum()))
            self._sk_regressor_ = None
        self._sk_constant_ = float(y[pos].mean()) if pos.any() else float(y.mean())

    def _sklearn_forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the two-stage fallback and return ``(p_alive, value_given_alive)`` uncalibrated."""
        Xs = self._scale(X)
        if self._sk_classifier_ is not None:
            proba = self._sk_classifier_.predict_proba(Xs)
            classes = list(getattr(self._sk_classifier_, "classes_", [0, 1]))
            col = classes.index(1) if 1 in classes else proba.shape[1] - 1
            p_alive = np.clip(proba[:, col].astype(np.float64), _P_LO, _P_HI)
        else:
            p_alive = np.full(Xs.shape[0], float(np.clip(self.alive_rate_, 0.0, 1.0)))
        if self._sk_regressor_ is not None:
            value = np.expm1(np.asarray(self._sk_regressor_.predict(Xs), dtype=np.float64))
        else:
            value = np.full(Xs.shape[0], self._sk_constant_)
        return p_alive, np.maximum(value, 0.0)

    # ---------------------------------------------------------------- predictions
    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated ``P(alive) * E[value | alive]``."""
        p_alive, value = self._forward(X)
        return p_alive * value

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Dispatch to the fitted backend."""
        if not hasattr(self, "backend_"):
            raise RuntimeError("DeepCLV is not fitted; call fit() first")
        return self._torch_forward(X) if self.backend_ == "torch" else self._sklearn_forward(X)

    def predict(self, X: Any) -> np.ndarray:
        """Expected value per customer, on the same scale as the training ``value``.

        Returns ``P(alive) * E[value | alive] * calibration_``. The discount rate is
        deliberately **not** applied here so that ``predict`` can be scored directly
        against the label; use :meth:`predict_discounted` for a present value.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Design matrix with the same columns as at fit time.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Non-negative expected value, float64.
        """
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {Xa.shape}")
        if Xa.shape[1] != getattr(self, "n_features_in_", Xa.shape[1]):
            raise ValueError(f"X has {Xa.shape[1]} columns but the model was fitted on {self.n_features_in_}")
        return np.maximum(self._raw_predict(Xa) * self.calibration_, 0.0)

    def predict_components(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return the two heads separately: ``(p_alive, expected_value_if_alive)``.

        Exposing the decomposition is the point of the architecture -- the retention
        probability feeds the churn-risk column of the decision frame while the value head
        feeds the CLV column.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Design matrix.

        Returns
        -------
        tuple of numpy.ndarray
            ``p_alive`` in ``[0, 1]`` and the calibrated conditional value, both ``(n,)``.
        """
        Xa = np.asarray(X, dtype=np.float64)
        p_alive, value = self._forward(Xa)
        return np.clip(p_alive, 0.0, 1.0), np.maximum(value * self.calibration_, 0.0)

    def predict_discounted(self, X: Any) -> np.ndarray:
        """Present value of :meth:`predict`, assuming value accrues evenly over the horizon.

        The predicted horizon total is spread across ``horizon`` equal monthly instalments
        and each is discounted at ``(1 + discount_rate)^-t`` for ``t = 1..horizon``, so the
        result is ``predict(X) * annuity_factor(horizon, d) / horizon``. Same end-of-month
        convention as :func:`clv_from_survival`.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Design matrix.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Discounted expected value.
        """
        horizon = max(int(self.horizon), 1)
        factor = annuity_factor(horizon, float(self.discount_rate)) / horizon
        return self.predict(X) * factor

    def score(self, X: Any, value: Any) -> float:
        """Coefficient of determination ``R^2`` of :meth:`predict` against ``value``.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Design matrix.
        value : array-like of shape (n,)
            Observed values.

        Returns
        -------
        float
            ``1 - SSE/SST``; 0 means "no better than predicting the sample mean".
        """
        y = _as_float_1d(value, "value")
        pred = self.predict(X)
        sse = float(np.sum((y - pred) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - sse / sst if sst > _EPS else float("nan")


# ======================================================================================
# simulation helper (used by the smoke test; also handy for unit tests elsewhere)
# ======================================================================================
def simulate_clv_population(
    n_customers: int = 3000,
    *,
    r: float = 2.0,
    alpha: float = 4.0,
    a: float = 1.3,
    b: float = 2.6,
    p: float = 6.0,
    q: float = 3.5,
    v: float = 15.0,
    t_min: float = 10.0,
    t_max: float = 40.0,
    holdout: float = 0.0,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate an exact BG/NBD + Gamma-Gamma population with known parameters.

    Generates from the *models' own* generative story rather than an approximation, so a
    fit on the output is a genuine test of the likelihood:

    1. ``lambda_i ~ Gamma(r, rate=alpha)`` and ``p_i ~ Beta(a, b)``;
    2. the customer buys at Exponential(``lambda_i``) intervals and, after each purchase,
       dies with probability ``p_i`` -- so the number of purchases in their lifetime is
       Geometric(``p_i``);
    3. purchases after the observation age ``T_i ~ Uniform(t_min, t_max)`` are discarded,
       leaving the observable ``(x, t_x, T)``;
    4. transaction values are ``Gamma(p, rate=nu_i)`` with ``nu_i ~ Gamma(q, rate=v)``,
       drawn independently of ``lambda_i`` so the Gamma-Gamma independence assumption holds.

    The transaction loop is fully vectorised: lifetimes are capped both by the geometric
    draw and by a generous Poisson bound on how many purchases could fit in ``T``, so the
    gap matrix stays small.

    Parameters
    ----------
    n_customers : int, default 3000
        Number of customers to simulate.
    r, alpha : float
        Gamma prior on the purchase rate (``alpha`` is a *rate*, so ``E[lambda] = r/alpha``).
    a, b : float
        Beta prior on the per-purchase dropout probability, ``E[p] = a/(a+b)``.
    p, q, v : float
        Gamma-Gamma parameters; the population mean spend is ``p*v/(q-1)`` (needs ``q > 1``).
    t_min, t_max : float
        Bounds of the uniform *total* simulated span. Heterogeneous ages sharpen
        identification.
    holdout : float, default 0.0
        Length of a forward validation window. When positive, the returned
        ``frequency``/``recency``/``T`` describe only the calibration window
        ``[0, T_total - holdout]`` and an extra column ``frequency_holdout`` counts the
        purchases that really happened in the window after it. That makes a genuine
        out-of-time forecast test possible: fit on the calibration columns, predict
        ``holdout`` periods ahead, and compare against ``frequency_holdout``. Must be
        smaller than ``t_min``.
    random_state : int or numpy.random.Generator or None, optional
        Seed.

    Returns
    -------
    pandas.DataFrame
        One row per customer with the columns :func:`probabilistic_clv` expects --
        ``customer_id``, ``frequency``, ``recency``, ``T``, ``monetary_value`` -- plus the
        latent truths ``lambda_true``, ``p_true``, ``nu_true`` and ``alive_true`` for
        diagnostics, and ``frequency_holdout`` when ``holdout > 0``.
    """
    if n_customers < 1:
        raise ValueError("n_customers must be >= 1")
    if q <= 1.0:
        raise ValueError("q must exceed 1 for the Gamma-Gamma population mean to exist")
    if holdout < 0.0:
        raise ValueError("holdout must be non-negative")
    if holdout >= t_min:
        raise ValueError(f"holdout ({holdout}) must be smaller than t_min ({t_min})")
    rng = as_rng(random_state)
    n = int(n_customers)

    lam = rng.gamma(shape=r, scale=1.0 / alpha, size=n)
    p_drop = np.clip(rng.beta(a, b, size=n), 1e-9, 1 - 1e-9)
    age = rng.uniform(t_min, t_max, size=n)

    lifetime_purchases = rng.geometric(p_drop)  # >= 1: purchases made before dying
    expected = lam * age
    poisson_cap = np.ceil(expected + 6.0 * np.sqrt(expected + 1.0) + 10.0).astype(np.int64)
    n_draw = np.minimum(lifetime_purchases, poisson_cap)
    k_max = int(max(n_draw.max(), 1))

    gaps = rng.exponential(scale=1.0 / lam[:, None], size=(n, k_max))
    gaps = np.where(np.arange(k_max)[None, :] < n_draw[:, None], gaps, np.inf)
    times = np.cumsum(gaps, axis=1)

    # the calibration window ends `holdout` periods before the simulated span does
    cal_age = age - float(holdout)
    inside = times <= cal_age[:, None]
    frequency = inside.sum(axis=1).astype(np.float64)
    recency = np.where(inside, np.where(np.isfinite(times), times, 0.0), 0.0).max(axis=1)
    recency = np.where(frequency > 0, recency, 0.0)
    # "alive" means the geometric lifetime was not exhausted inside the calibration window
    alive_true = (frequency < lifetime_purchases).astype(np.int64)
    # purchases that really occurred in the forward window (cal_age, age]
    freq_holdout = ((times > cal_age[:, None]) & (times <= age[:, None])).sum(axis=1).astype(np.float64)

    nu = rng.gamma(shape=q, scale=1.0 / v, size=n)
    monetary = np.zeros(n, dtype=np.float64)
    pos = frequency > 0
    if pos.any():
        total = rng.gamma(shape=p * frequency[pos], scale=1.0 / nu[pos])
        monetary[pos] = total / frequency[pos]

    out = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "frequency": frequency,
            "recency": recency,
            "T": cal_age,
            "monetary_value": monetary,
            "lambda_true": lam,
            "p_true": p_drop,
            "nu_true": nu,
            "alive_true": alive_true,
        }
    )
    if holdout > 0.0:
        out["frequency_holdout"] = freq_holdout
    return out


def _series_conditional_transactions(
    t: float, x: float, t_x: float, T: float, r: float, alpha: float, a: float, b: float,
    n_terms: int = 4000,
) -> float:
    """Exact (quadrature-free) reference for :meth:`BGNBD.conditional_expected_transactions`.

    This is the *precise* independent check on the ``2F1`` closed form. It reaches the
    same quantity by a completely different route -- term-by-term summation with both
    latent integrals done analytically -- so agreement to machine precision means the
    hypergeometric expression was derived correctly, not merely self-consistently.

    Derivation. Write the unnormalised posterior kernel for a customer observed as
    ``(x, t_x, T)``::

        L(lam, p) = (1-p)^x lam^x e^{-lam T}                        [still alive]
                  + 1{x>0} p (1-p)^{x-1} lam^x e^{-lam t_x}         [died after purchase x]

    and the forward expectation for a customer known alive at ``T``::

        E[Y(t) | alive, lam, p] = sum_{j>=1} (1-p)^{j-1} P(Pois(lam t) >= j)

    Both latent integrals are available in closed form and, crucially, **factor** inside
    each term, so no numerical integration is needed anywhere::

        int (1-p)^s Beta(a,b) dp          = B(a, b+s) / B(a, b)
        int lam^x e^{-lam s} Gamma(r,alpha) dlam = alpha^r Gamma(r+x) / (Gamma(r) (alpha+s)^(r+x))

    Expanding ``P(Pois(lam t) >= j) = sum_{k>=j} e^{-lam t}(lam t)^k / k!`` and swapping
    the order of summation leaves a single geometrically convergent series::

        numerator   = sum_{k>=1} D_k * sum_{j=1..k} B(a, b+x+j-1)/B(a, b)
        D_k         = alpha^r / Gamma(r) * t^k / k! * Gamma(r+x+k) / (alpha+T+t)^(r+x+k)
        denominator = G(T) B(a,b+x)/B(a,b) + 1{x>0} G(t_x) B(a+1,b+x-1)/B(a,b)

    Everything is evaluated in log space, so the reference is stable for the extreme
    parameter values the grid version cannot reach.

    Parameters
    ----------
    t : float
        Forecast window.
    x, t_x, T : float
        One customer's frequency, recency and age.
    r, alpha, a, b : float
        BG/NBD parameters.
    n_terms : int, default 4000
        Series truncation. The terms are Poisson-like in ``k``, so convergence is
        geometric and the default is far beyond what any realistic ``lambda*t`` needs.

    Returns
    -------
    float
        ``E[Y(t) | x, t_x, T]``.
    """
    t, x, t_x, T = float(t), float(x), float(t_x), float(T)
    if t <= 0.0:
        return 0.0

    def log_beta_ratio(s: np.ndarray | float) -> np.ndarray:
        """``log B(a, b+s) / B(a, b)``."""
        return gammaln(a + b) - gammaln(b) + gammaln(b + s) - gammaln(a + b + s)

    k = np.arange(0, n_terms + 1, dtype=np.float64)
    log_d = (
        r * np.log(alpha) - gammaln(r) + k * np.log(t) - gammaln(k + 1.0)
        + gammaln(r + x + k) - (r + x + k) * np.log(alpha + T + t)
    )
    j = np.arange(1, n_terms + 1, dtype=np.float64)
    partial = np.concatenate(([0.0], np.cumsum(np.exp(log_beta_ratio(x + j - 1.0)))))
    with np.errstate(divide="ignore"):
        log_numer = logsumexp(log_d[1:] + np.log(partial[1:]))

    log_g_T = r * np.log(alpha) + gammaln(r + x) - gammaln(r) - (r + x) * np.log(alpha + T)
    log_br_x = log_beta_ratio(x)
    parts = [log_g_T + log_br_x]
    if x > 0:
        log_g_tx = r * np.log(alpha) + gammaln(r + x) - gammaln(r) - (r + x) * np.log(alpha + t_x)
        # B(a+1, b+x-1)/B(a,b) = (a/(b+x-1)) * B(a, b+x)/B(a,b)
        parts.append(log_g_tx + np.log(a) - np.log(b + x - 1.0) + log_br_x)
    return float(np.exp(log_numer - logsumexp(np.array(parts))))


def _quadrature_conditional_transactions(
    t: float, x: float, t_x: float, T: float, r: float, alpha: float, a: float, b: float,
    n_lambda: int = 600, n_p: int = 400, n_terms: int = 300,
) -> tuple[float, float]:
    """Brute-force grid reference for :meth:`BGNBD.conditional_expected_transactions`.

    Independent of the ``2F1`` closed form and of the analytic integrals used by
    :func:`_series_conditional_transactions`: this one integrates the posterior over
    ``(lambda, p)`` on a dense grid, assuming nothing but the definition of the posterior.
    The inner expectation for a customer known to be alive is the elementary series
    ``sum_{j>=1} (1-p)^(j-1) * P(Poisson(lambda*t) >= j)`` -- "survive ``j-1`` dropout
    opportunities and fit ``j`` purchases into the window".

    **Accuracy caveat -- read before tightening any tolerance against this function.**
    It is a midpoint rule on a linear grid, so it is only good to ``~1e-2`` relative, and
    it degrades badly when either prior density is singular at the origin: the Beta
    density behaves like ``p^(a-1)`` and the Gamma like ``lambda^(r-1)``, so for ``a < 1``
    or ``r < 1`` the first few cells are grossly under-integrated (measured: 100% error at
    ``r = 0.24, a = 0.79, x = 0``). Use it as a coarse structural check only, and use
    :func:`_series_conditional_transactions` -- which is exact -- for anything tight.

    It does, however, return ``probability_alive`` as a by-product, and that quantity is
    well conditioned here, so the smoke test keeps using it for that.

    Parameters
    ----------
    t : float
        Forecast window.
    x, t_x, T : float
        One customer's frequency, recency and age.
    r, alpha, a, b : float
        BG/NBD parameters.
    n_lambda, n_p : int
        Grid resolution for the two latent dimensions.
    n_terms : int
        Number of series terms.

    Returns
    -------
    tuple of (float, float)
        ``(expected_transactions, probability_alive)`` from quadrature.
    """
    from scipy.stats import gamma as _gamma_dist

    lo, hi = _gamma_dist.ppf([1e-8, 1.0 - 1e-8], a=r, scale=1.0 / alpha)
    lam = np.linspace(max(float(lo), 1e-9), float(hi), n_lambda)
    f_lam = np.exp(r * np.log(alpha) + (r - 1.0) * np.log(lam) - alpha * lam - gammaln(r))
    pg = (np.arange(n_p) + 0.5) / n_p
    f_p = np.exp((a - 1.0) * np.log(pg) + (b - 1.0) * np.log1p(-pg) + gammaln(a + b) - gammaln(a) - gammaln(b))

    L, P = lam[:, None], pg[None, :]
    alive_branch = np.exp(x * np.log(L) + x * np.log1p(-P) - L * T)
    dead_branch = np.exp(x * np.log(L) + (x - 1.0) * np.log1p(-P) - L * t_x) * P if x > 0 else 0.0
    like = alive_branch + dead_branch

    j = np.arange(1, n_terms + 1, dtype=np.float64)
    tail = gammainc(j[None, :], (lam * t)[:, None])       # P(Poisson(lam*t) >= j)
    geo = np.power(1.0 - pg[:, None], (j - 1.0)[None, :])
    e_future = tail @ geo.T

    w = f_lam[:, None] * f_p[None, :]
    cell = (lam[1] - lam[0]) * (pg[1] - pg[0])
    denom = float(np.sum(w * like)) * cell
    numer = float(np.sum(w * alive_branch * e_future)) * cell
    p_alive = float(np.sum(w * alive_branch)) * cell / denom
    return numer / denom, p_alive


# ======================================================================================
# smoke test
# ======================================================================================
if __name__ == "__main__":  # pragma: no cover - smoke test
    import time

    from scipy.optimize import approx_fprime

    t_start = time.perf_counter()
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        """Record an assertion result for the summary table and fail fast when broken."""
        checks.append((name, bool(ok), detail))
        assert ok, f"FAILED: {name} -- {detail}"

    print("=" * 96)
    print("prism/models/clv.py smoke test")
    print("=" * 96)

    # ---------------------------------------------------------------- 1. simulate
    TRUTH = {"r": 2.0, "alpha": 4.0, "a": 1.3, "b": 2.6}
    GG_TRUTH = {"p": 6.0, "q": 3.5, "v": 15.0}
    N = 3000
    pop = simulate_clv_population(N, **TRUTH, **GG_TRUTH, t_min=10.0, t_max=40.0, random_state=11)
    freq = pop["frequency"].to_numpy()
    rec = pop["recency"].to_numpy()
    age_ = pop["T"].to_numpy()
    mon = pop["monetary_value"].to_numpy()
    print(
        f"\n[1] simulated n={N}  mean x={freq.mean():.2f}  max x={freq.max():.0f}  "
        f"zero-repeat={100 * (freq == 0).mean():.1f}%  mean gap={np.mean(age_ - rec):.1f}"
    )

    # ---------------------------------------------------------------- 2. gradient check
    probe = np.log(np.array([1.5, 3.0, 1.1, 2.0]))
    w_ones = np.ones_like(freq)
    # several probe points, not one: a gradient can be right at the truth and wrong
    # elsewhere, and L-BFGS-B spends almost all its time "elsewhere"
    worst_grad = 0.0
    for probe in (
        np.log(np.array([1.5, 3.0, 1.1, 2.0])),
        np.log(np.array([2.0, 4.0, 1.3, 2.6])),
        np.log(np.array([0.3, 10.0, 0.5, 0.8])),
        np.log(np.array([5.0, 1.0, 3.0, 9.0])),
    ):
        f_val, g_ana = BGNBD._objective(probe, freq, rec, age_, w_ones, float(N), 1e-4)
        g_num = approx_fprime(probe, lambda z: BGNBD._objective(z, freq, rec, age_, w_ones, float(N), 1e-4)[0], 1e-6)
        worst_grad = max(worst_grad, float(np.abs(g_ana - g_num).max()))
    check("bgnbd analytic gradient == finite differences", worst_grad < 1e-5, f"max|diff| = {worst_grad:.2e}")
    print(f"[2] BG/NBD analytic gradient vs finite differences (4 probe points): "
          f"max abs diff = {worst_grad:.3e}")

    # the same check for Gamma-Gamma, whose gradient is equally hand-derived
    _gm = (freq > 0) & (mon > 0)
    w_gg = np.ones_like(freq[_gm])
    worst_gg_grad = 0.0
    for probe in (np.log(np.array([6.0, 3.5, 15.0])), np.log(np.array([1.0, 2.0, 5.0])),
                  np.log(np.array([0.5, 8.0, 40.0]))):
        _, g_ana = GammaGamma._objective(probe, freq[_gm], mon[_gm], w_gg, float(w_gg.sum()), 1e-4)
        g_num = approx_fprime(
            probe, lambda z: GammaGamma._objective(z, freq[_gm], mon[_gm], w_gg, float(w_gg.sum()), 1e-4)[0], 1e-6)
        worst_gg_grad = max(worst_gg_grad, float(np.abs(g_ana - g_num).max()))
    check("gamma-gamma analytic gradient == finite differences", worst_gg_grad < 1e-5,
          f"max|diff| = {worst_gg_grad:.2e}")
    print(f"    Gamma-Gamma analytic gradient vs finite differences (3 probe points): "
          f"max abs diff = {worst_gg_grad:.3e}")

    # ---------------------------------------------------------------- 3. fit + recovery
    t0 = time.perf_counter()
    bg = BGNBD(penalizer=1e-4, random_state=0).fit(freq, rec, age_)
    fit_s = time.perf_counter() - t0
    check("bgnbd converged", bg.converged_, str(bg.optim_result_.message))

    print(f"\n[3] BG/NBD parameter recovery (fit in {fit_s:.2f}s, LL = {bg.log_likelihood_:.1f})")
    # Tolerances are calibrated, not decorative: a sweep over 8 independent simulation
    # seeds at this n put the worst raw-parameter error at 19.3%, the worst E[lambda]
    # error at 3.2% and the worst E[p] error at 5.6%. The bounds below sit just above
    # those, so they still pass on any seed but would FAIL a materially wrong likelihood.
    # (The old "within a factor of 2" bound would have passed an implementation that was
    # wrong by 99%.)
    RAW_TOL = 0.30
    print(f"    {'param':<8}{'truth':>10}{'fitted':>12}{'ratio':>10}   within {RAW_TOL:.0%}")
    for key in ("r", "alpha", "a", "b"):
        truth_v, fit_v = TRUTH[key], bg.params_[key]
        ratio = fit_v / truth_v
        ok = abs(ratio - 1.0) <= RAW_TOL
        print(f"    {key:<8}{truth_v:>10.3f}{fit_v:>12.4f}{ratio:>10.3f}   {'yes' if ok else 'NO'}")
        check(f"bgnbd recovers {key} within {RAW_TOL:.0%}", ok,
              f"truth={truth_v} fitted={fit_v:.4f} ratio={ratio:.3f}")
    # the derived quantities are far better identified than the raw parameters
    rate_err = abs((bg.params_["r"] / bg.params_["alpha"]) / (TRUTH["r"] / TRUTH["alpha"]) - 1.0)
    drop_err = abs(
        (bg.params_["a"] / (bg.params_["a"] + bg.params_["b"])) / (TRUTH["a"] / (TRUTH["a"] + TRUTH["b"])) - 1.0
    )
    check("bgnbd recovers E[lambda] within 7%", rate_err < 0.07, f"relative error {rate_err:.3%}")
    check("bgnbd recovers E[p] within 9%", drop_err < 0.09, f"relative error {drop_err:.3%}")
    print(f"    E[lambda] rel.err = {rate_err:.2%}   E[dropout p] rel.err = {drop_err:.2%}")

    # A recovery check is only worth anything if a WRONG parameter set would fail it.
    # Perturb the truth by 40% and confirm the likelihood actually prefers the fit --
    # this is what rules out "the tolerance would pass for any code".
    ll_fit = bg.log_likelihood(freq, rec, age_)
    worse = 0
    for key in ("r", "alpha", "a", "b"):
        for factor in (0.6, 1.4):
            probe_model = BGNBD()
            probe_model.params_ = dict(bg.params_)
            probe_model.params_[key] = bg.params_[key] * factor
            if probe_model.log_likelihood(freq, rec, age_) < ll_fit:
                worse += 1
    check("the fitted optimum beats every 40% perturbation of it", worse == 8,
          f"{worse}/8 perturbations scored worse, as they must")

    # ---------------------------------------------------------------- 4. pmf integrity
    grid = np.arange(0, 400)
    pmf_sums = [float(bg.probability_of_n_purchases(grid, t).sum()) for t in (5.0, 20.0, 40.0)]
    worst_pmf = max(abs(s - 1.0) for s in pmf_sums)
    check("bgnbd count pmf sums to 1", worst_pmf < 1e-6, f"worst |sum-1| = {worst_pmf:.2e}")
    print(f"\n[4] P(X(t)=n) sums to 1 at t in (5, 20, 40): worst deviation = {worst_pmf:.2e}")

    gof = bg.goodness_of_fit(freq, age_, max_x=6)
    worst_bin = float(gof["abs_error_pct"].max())
    check("bgnbd frequency histogram matches", worst_bin < 3.0, f"worst bin error {worst_bin:.2f} pct-points")
    print("    observed vs expected repeat-purchase histogram:")
    print("      " + gof.to_string(index=False, float_format=lambda z: f"{z:8.2f}").replace("\n", "\n      "))

    # ---------------------------------------------------------------- 5. closed form vs quadrature
    print("\n[5] conditional_expected_transactions: hyp2f1 closed form vs 2-D quadrature")
    print(f"    {'x':>4}{'t_x':>7}{'T':>7}{'analytic':>12}{'quadrature':>13}{'rel.err':>11}")
    profiles = [(5.0, 28.0, 30.0), (0.0, 0.0, 30.0), (1.0, 25.0, 30.0), (20.0, 29.0, 30.0)]
    worst_rel = 0.0
    for xv, tv, Tv in profiles:
        analytic = float(bg.conditional_expected_transactions(12.0, [xv], [tv], [Tv])[0])
        quad, quad_alive = _quadrature_conditional_transactions(12.0, xv, tv, Tv, **bg.params_)
        rel = abs(analytic - quad) / max(quad, 1e-9)
        worst_rel = max(worst_rel, rel)
        print(f"    {xv:>4.0f}{tv:>7.1f}{Tv:>7.1f}{analytic:>12.5f}{quad:>13.5f}{rel:>11.2e}")
        pa = float(bg.probability_alive([xv], [tv], [Tv])[0])
        check(
            f"p_alive closed form == quadrature (x={xv:.0f})",
            abs(pa - quad_alive) < 1e-3,
            f"closed form {pa:.5f} vs quadrature {quad_alive:.5f}",
        )
    check("E[Y(t)] closed form == quadrature", worst_rel < 2e-2, f"worst relative error {worst_rel:.2e}")
    print("    (the grid reference is only good to ~1e-2; the exact series below is the tight check)")

    # ---------------------------------------------------------------- 5b. exact series
    # The grid quadrature above assumes nothing, but it is a midpoint rule and cannot be
    # trusted past ~1e-2 -- and it collapses entirely when a < 1 or r < 1, where the Beta
    # and Gamma densities are singular at the origin. _series_conditional_transactions
    # does both latent integrals analytically, so it is exact, and it is what pins the
    # hypergeometric form down to machine precision. Crucially it is checked across the
    # a < 1 and a + b < 1 regimes, which the fitted parameters here never visit but real
    # data routinely does (the canonical CDNOW fit is r=0.24, alpha=4.41, a=0.79, b=2.43).
    print("\n[5b] conditional_expected_transactions vs an EXACT series reference")
    print(f"     {'r':>6}{'alpha':>7}{'a':>6}{'b':>6}   regime                worst rel.err")
    series_worst = 0.0
    regimes = [
        (bg.params_["r"], bg.params_["alpha"], bg.params_["a"], bg.params_["b"], "fitted (a>1)"),
        (0.24, 4.41, 0.79, 2.43, "CDNOW-like (a<1, r<1)"),
        (1.50, 6.00, 0.40, 1.20, "strong J-shaped dropout"),
        (0.80, 2.00, 0.20, 0.50, "a+b<1 -> c<0 in 2F1"),
        (3.00, 2.00, 1.00, 2.00, "a==1 exactly (removable)"),
    ]
    for r_v, al_v, a_v, b_v, label in regimes:
        probe_bg = BGNBD()
        probe_bg.params_ = {"r": r_v, "alpha": al_v, "a": a_v, "b": b_v}
        worst_here = 0.0
        for xv, tv, Tv in [(0.0, 0.0, 39.0), (1.0, 4.0, 39.0), (2.0, 30.0, 39.0), (7.0, 38.0, 39.0)]:
            got = float(probe_bg.conditional_expected_transactions(12.0, [xv], [tv], [Tv])[0])
            want = _series_conditional_transactions(12.0, xv, tv, Tv, r_v, al_v, a_v, b_v)
            worst_here = max(worst_here, abs(got - want) / max(want, 1e-12))
        series_worst = max(series_worst, worst_here)
        print(f"     {r_v:>6.2f}{al_v:>7.2f}{a_v:>6.2f}{b_v:>6.2f}   {label:<22}{worst_here:>13.2e}")
        check(f"E[Y(t)] exact in regime '{label}'", worst_here < 1e-5, f"worst rel.err {worst_here:.2e}")
    check("E[Y(t)] matches the exact series to 1e-5 in every regime", series_worst < 1e-5,
          f"worst {series_worst:.2e}")

    # the unconditional E[X(t)] must agree with the mean of the model's own count pmf --
    # an internal consistency check that catches an a<=1 mishandling in either one
    print("     unconditional E[X(t)] vs the mean of P(X(t)=n):")
    n_grid = np.arange(0, 800)
    worst_uncond = 0.0
    for r_v, al_v, a_v, b_v, _label in regimes:
        probe_bg = BGNBD()
        probe_bg.params_ = {"r": r_v, "alpha": al_v, "a": a_v, "b": b_v}
        for tv in (6.0, 20.0):
            pmf_mean = float((n_grid * probe_bg.probability_of_n_purchases(n_grid, tv)).sum())
            closed = float(probe_bg.expected_number_of_transactions(tv)[0])
            worst_uncond = max(worst_uncond, abs(pmf_mean - closed) / max(closed, 1e-12))
    check("expected_number_of_transactions == the pmf mean in every regime", worst_uncond < 1e-5,
          f"worst rel.err {worst_uncond:.2e}")
    print(f"       worst relative disagreement across all 5 regimes: {worst_uncond:.2e}")

    # ---------------------------------------------------------------- 6. p_alive properties
    p_all = bg.probability_alive(freq, rec, age_)
    check("p_alive in [0,1]", bool(np.all((p_all >= 0) & (p_all <= 1))), f"range [{p_all.min():.4f}, {p_all.max():.4f}]")
    check("p_alive == 1 when frequency == 0", bool(np.allclose(p_all[freq == 0], 1.0)), "zero-repeat customers")
    gaps_grid = np.linspace(0.0, 29.0, 40)
    p_gap = bg.probability_alive(np.full(40, 5.0), 30.0 - gaps_grid, np.full(40, 30.0))
    check(
        "p_alive strictly decreasing in the recency gap",
        bool(np.all(np.diff(p_gap) < 0.0)),
        f"{p_gap[0]:.4f} at gap 0 -> {p_gap[-1]:.6f} at gap 29",
    )
    # and decreasing in frequency at a fixed gap: a regular's silence is louder
    p_freq = bg.probability_alive(np.arange(1.0, 21.0), np.full(20, 20.0), np.full(20, 30.0))
    check("p_alive decreasing in frequency at a fixed gap", bool(np.all(np.diff(p_freq) < 0.0)),
          f"{p_freq[0]:.4f} (x=1) -> {p_freq[-1]:.4f} (x=20)")
    print(
        f"\n[6] p_alive: range [{p_all.min():.4f}, {p_all.max():.4f}], mean {p_all.mean():.4f}; "
        f"gap 0 -> 29 gives {p_gap[0]:.4f} -> {p_gap[-1]:.2e} (monotone)"
    )
    # simulated ground truth: predicted p_alive should discriminate the truly-alive
    truly_alive = pop["alive_true"].to_numpy().astype(bool)
    sep = float(p_all[truly_alive].mean() - p_all[~truly_alive].mean())
    check("p_alive separates truly-alive from truly-dead", sep > 0.25, f"mean difference {sep:.3f}")
    print(f"    mean p_alive: truly alive {p_all[truly_alive].mean():.3f} vs truly dead "
          f"{p_all[~truly_alive].mean():.3f}  (separation {sep:.3f})")

    # monotone in the forecast window -- PER CUSTOMER, not merely in the aggregate, since
    # a sum can be monotone while individual curves wobble
    cum_all = np.stack([bg.conditional_expected_transactions(float(t), freq, rec, age_) for t in range(0, 13)], axis=1)
    curve = cum_all.sum(axis=0)
    worst_dip = float(np.diff(cum_all, axis=1).min())
    check("E[Y(t)] increasing in t for every customer", worst_dip >= -1e-9,
          f"worst per-customer decrement {worst_dip:.2e}")
    check("E[Y(0)] == 0 for every customer", float(np.abs(cum_all[:, 0]).max()) < 1e-12,
          f"max |E[Y(0)]| = {float(np.abs(cum_all[:, 0]).max()):.2e}")
    check("E[Y(t)] finite for every customer", bool(np.isfinite(cum_all).all()), "no NaN/inf")

    # ---------------------------------------------------------------- 6b. weights
    # An estimator that silently ignores sample_weight is a classic, invisible bug: the
    # only way to catch it is to prove that weighting is EXACTLY equivalent to physically
    # duplicating rows, and that it changes the answer at all.
    sub = slice(0, 400)
    reps = np.tile(np.array([1.0, 2.0, 3.0, 1.0]), 100)
    dup = np.repeat(np.arange(400), reps.astype(int))
    bg_w = BGNBD(n_restarts=4).fit(freq[sub], rec[sub], age_[sub], weights=reps)
    bg_d = BGNBD(n_restarts=4).fit(freq[sub][dup], rec[sub][dup], age_[sub][dup])
    bg_u = BGNBD(n_restarts=4).fit(freq[sub], rec[sub], age_[sub])
    w_err = max(abs(bg_w.params_[k] / bg_d.params_[k] - 1.0) for k in ("r", "alpha", "a", "b"))
    w_shift = max(abs(bg_w.params_[k] / bg_u.params_[k] - 1.0) for k in ("r", "alpha", "a", "b"))
    check("BGNBD weights == physically duplicating rows", w_err < 1e-6, f"max rel param diff {w_err:.2e}")
    check("BGNBD weights are not silently ignored", w_shift > 1e-4, f"weighting moved params by {w_shift:.2%}")
    gg_w = GammaGamma(n_restarts=4).fit(freq[sub], mon[sub], weights=reps)
    gg_d = GammaGamma(n_restarts=4).fit(freq[sub][dup], mon[sub][dup])
    gw_err = max(abs(gg_w.params_[k] / gg_d.params_[k] - 1.0) for k in ("p", "q", "v"))
    check("GammaGamma weights == physically duplicating rows", gw_err < 1e-4, f"max rel param diff {gw_err:.2e}")
    print(f"    weights: BGNBD matches row-duplication to {w_err:.1e} and shifts params "
          f"{w_shift:.1%} vs unweighted; GammaGamma matches to {gw_err:.1e}")

    # an unseeded fit must still be reproducible -- the default n_restarts=6 draws two
    # jittered starts, and near-tied restarts otherwise make the winner a coin flip
    unseeded = [BGNBD().fit(freq[sub], rec[sub], age_[sub]).params_ for _ in range(3)]
    check("an unseeded BGNBD fit is bit-identical across calls",
          all(u == unseeded[0] for u in unseeded[1:]), "random_state=None is still deterministic")

    # ---------------------------------------------------------------- 7. Gamma-Gamma
    gg = GammaGamma(penalizer=1e-4, random_state=0).fit(freq, mon)
    print(f"\n[7] Gamma-Gamma recovery (n={gg.n_} with frequency > 0, {gg.n_dropped_} dropped)")
    # as above, calibrated against an 8-seed sweep: worst observed raw error 22.3%,
    # worst population-mean error 3.2%
    GG_TOL = 0.35
    print(f"    {'param':<8}{'truth':>10}{'fitted':>12}{'ratio':>10}   within {GG_TOL:.0%}")
    for key in ("p", "q", "v"):
        truth_v, fit_v = GG_TRUTH[key], gg.params_[key]
        ratio = fit_v / truth_v
        ok = abs(ratio - 1.0) <= GG_TOL
        print(f"    {key:<8}{truth_v:>10.3f}{fit_v:>12.4f}{ratio:>10.3f}   {'yes' if ok else 'NO'}")
        check(f"gamma-gamma recovers {key} within {GG_TOL:.0%}", ok, f"truth={truth_v} fitted={fit_v:.4f}")
    pop_truth = GG_TRUTH["p"] * GG_TRUTH["v"] / (GG_TRUTH["q"] - 1.0)
    pop_err = abs(gg.population_mean_profit_ / pop_truth - 1.0)
    check("gamma-gamma recovers the population mean spend within 6%", pop_err < 0.06,
          f"truth {pop_truth:.2f} vs fitted {gg.population_mean_profit_:.2f}")
    print(f"    population mean spend: truth {pop_truth:.2f}  fitted {gg.population_mean_profit_:.2f} "
          f"({pop_err:.2%} error)  empirical {mon[freq > 0].mean():.2f}")

    # independence diagnostic must PASS on this (correctly independent) population
    check("independence check passes on independent data", gg.independence_ok_,
          f"corr(f, m) = {gg.freq_monetary_corr_:+.3f}")
    print(f"    corr(frequency, monetary_value) = {gg.freq_monetary_corr_:+.3f} -> assumption OK")

    # ... and must FIRE on data where spend rises with frequency
    dep_mon = np.where(freq > 0, mon * (1.0 + 0.45 * freq), 0.0)
    gg_dep = GammaGamma(random_state=0).fit(freq, dep_mon)
    check("independence check fires on correlated data", (not gg_dep.independence_ok_) and gg_dep.freq_monetary_corr_ > 0.2,
          f"corr(f, m) = {gg_dep.freq_monetary_corr_:+.3f}")
    print(f"    deliberately correlated variant: corr = {gg_dep.freq_monetary_corr_:+.3f} -> warning raised")

    # shrinkage behaves like a credibility weight
    m_obs = 100.0
    shrunk = gg.conditional_expected_average_profit([1.0, 3.0, 10.0, 50.0, 300.0], [m_obs] * 5)
    pm = gg.population_mean_profit_
    check("shrinkage is monotone towards the observed mean", bool(np.all(np.diff(shrunk) > 0)),
          " -> ".join(f"{s:.2f}" for s in shrunk))
    check("shrunk estimate lies between the population and observed means",
          bool(np.all((shrunk > min(pm, m_obs) - 1e-9) & (shrunk < max(pm, m_obs) + 1e-9))),
          f"population mean {pm:.2f}, observed {m_obs:.2f}")
    check("shrinkage converges to the observed mean as x grows", abs(shrunk[-1] - m_obs) < 1.0,
          f"x=300 gives {shrunk[-1]:.3f} vs observed {m_obs:.1f}")
    print(f"    shrinkage of m_x={m_obs:.0f} at x = 1, 3, 10, 50, 300: "
          + ", ".join(f"{s:.2f}" for s in shrunk) + f"  (population mean {pm:.2f})")

    # ---------------------------------------------------------------- 8. probabilistic_clv
    clv_df = probabilistic_clv(pop, horizon_months=12, discount_rate=0.01, margin=0.3, bgnbd=bg, gamma_gamma=gg)
    expected_cols = ["customer_id", "p_alive", "expected_transactions", "expected_avg_profit", "clv", "clv_discounted"]
    check("probabilistic_clv returns the contracted columns", list(clv_df.columns) == expected_cols, str(list(clv_df.columns)))
    check("probabilistic_clv is finite", bool(np.isfinite(clv_df.select_dtypes("number").to_numpy()).all()), "no NaN/inf")
    check("clv is non-negative", bool((clv_df["clv"] >= 0).all()), f"min {clv_df['clv'].min():.4f}")
    check("discounting strictly reduces value where clv > 0",
          bool((clv_df.loc[clv_df["clv"] > 0, "clv_discounted"] < clv_df.loc[clv_df["clv"] > 0, "clv"]).all()),
          "clv_discounted < clv")
    ratio_total = float(clv_df["clv_discounted"].sum() / clv_df["clv"].sum())
    floor = annuity_factor(12, 0.01) / 12.0
    check("aggregate discount factor is between the 12-month floor and 1",
          floor < ratio_total < 1.0, f"{ratio_total:.4f} vs flat-annuity floor {floor:.4f}")
    print("\n[8] probabilistic_clv over 12 months at d=1%, margin=30%:")
    print(f"    mean CLV {clv_df['clv'].mean():8.2f} -> discounted {clv_df['clv_discounted'].mean():8.2f} "
          f"({ratio_total:.1%} retained; flat-annuity floor {floor:.1%})")
    print(f"    mean E[transactions] {clv_df['expected_transactions'].mean():.3f}, "
          f"mean E[avg profit] {clv_df['expected_avg_profit'].mean():.2f}")

    # ---------------------------------------------------------------- 9. clv_from_survival
    H, MARGIN, DRATE = 12, 100.0, 0.01
    certain = np.ones((4, H))
    analytic = MARGIN * annuity_factor(H, DRATE)
    got = clv_from_survival(certain, MARGIN, DRATE)
    check("clv_from_survival on S(t)=1 equals the ordinary annuity",
          bool(np.allclose(got, analytic, rtol=0, atol=1e-9)), f"got {got[0]:.6f}, analytic {analytic:.6f}")
    check("clv_from_survival on S(t)=0 is zero", bool(np.allclose(clv_from_survival(np.zeros((4, H)), MARGIN, DRATE), 0.0)), "zero curve")
    check("clv_from_survival at d=0 reduces to RMST * margin",
          bool(np.allclose(clv_from_survival(certain, 1.0, 0.0), float(H))),
          "discount_rate=0, margin=1 must equal sum(S) = RMST")
    vec_margin = np.full(4, MARGIN)
    check("scalar and vector margin agree", bool(np.allclose(got, clv_from_survival(certain, vec_margin, DRATE))), "broadcast")
    # a decaying curve must be worth strictly less than the certain one, and geometric decay
    # has its own closed form: sum_t s^t m (1+d)^-t is a geometric series
    s = 0.9
    decay = np.cumprod(np.full((1, H), s), axis=1)
    ratio_geo = s / (1.0 + DRATE)
    closed = MARGIN * ratio_geo * (1.0 - ratio_geo ** H) / (1.0 - ratio_geo)
    check("clv_from_survival on a geometric curve matches the closed-form series",
          abs(float(clv_from_survival(decay, MARGIN, DRATE)[0]) - closed) < 1e-9,
          f"got {float(clv_from_survival(decay, MARGIN, DRATE)[0]):.6f}, closed form {closed:.6f}")
    check("1-D curve is accepted as a single customer", clv_from_survival(np.ones(H), MARGIN, DRATE).shape == (1,), "shape")
    # a strictly better survival curve must be worth strictly more (the bridge must be
    # monotone in S, or the decision layer could prefer a worse offer)
    better = np.clip(decay + 0.05, 0.0, 1.0)
    check("clv_from_survival is monotone in the survival curve",
          float(clv_from_survival(better, MARGIN, DRATE)[0]) > float(clv_from_survival(decay, MARGIN, DRATE)[0]),
          "a uniformly higher S(t) is worth more")
    # discounting must be strictly between "no discount" and "everything at the horizon"
    undisc = float(clv_from_survival(decay, MARGIN, 0.0)[0])
    at_horizon = float(clv_from_survival(decay, MARGIN, DRATE)[0])
    check("clv_from_survival: discounting strictly reduces value",
          0.0 < at_horizon < undisc, f"{at_horizon:.4f} < {undisc:.4f}")
    print("\n[9] clv_from_survival (t starts at 1, cash at month end):")
    print(f"    S(t)=1, m={MARGIN:.0f}, d={DRATE:.0%}  -> {got[0]:12.6f}   annuity {analytic:12.6f}")
    print(f"    S(t)=0.9^t                            -> {float(clv_from_survival(decay, MARGIN, DRATE)[0]):12.6f}"
          f"   closed form {closed:12.6f}")
    print(f"    d=0, m=1 (must equal RMST={H})         -> {float(clv_from_survival(certain, 1.0, 0.0)[0]):12.6f}")

    # ---------------------------------------------------------------- 10. DeepCLV
    rng_d = as_rng(4)
    n_deep, p_deep = 4000, 10
    Xd = rng_d.normal(size=(n_deep, p_deep))
    lin = 1.1 * Xd[:, 0] - 0.8 * Xd[:, 1] + 0.6 * Xd[:, 2] * Xd[:, 3]
    p_ret = 1.0 / (1.0 + np.exp(-(lin - 0.15)))
    retained = (rng_d.random(n_deep) < p_ret).astype(np.float64)
    mu_val = np.exp(3.0 + 0.7 * Xd[:, 4] - 0.5 * Xd[:, 5] + 0.35 * Xd[:, 0] * Xd[:, 4])
    value = rng_d.gamma(shape=3.0, scale=mu_val / 3.0) * retained
    value[(retained == 1) & (rng_d.random(n_deep) < 0.12)] = 0.0   # zero inflation among survivors
    event_observed = 1.0 - retained                                # PRISM convention: 1 = churn observed

    n_tr = 3000
    Xtr, ytr, etr = Xd[:n_tr], value[:n_tr], event_observed[:n_tr]
    Xte, yte = Xd[n_tr:], value[n_tr:]
    baseline_mse = float(np.mean((yte - ytr.mean()) ** 2))

    print(f"\n[10] DeepCLV on n_train={n_tr}, n_test={n_deep - n_tr}, p={p_deep} "
          f"(retention {retained.mean():.1%}, zero-value share {(value == 0).mean():.1%})")
    print(f"     {'backend':<10}{'loss':<10}{'fit s':>8}{'test MSE':>12}{'mean MSE':>12}{'R2 vs mean':>12}{'spearman':>10}")
    deep_rows = []
    configs = [("torch", "logcosh"), ("torch", "tweedie"), ("sklearn", "logcosh")]
    for backend, loss_kind in configs:
        if backend == "torch" and not HAS_TORCH:
            print(f"     {'torch':<10}{loss_kind:<10}  skipped (torch not installed)")
            continue
        t0 = time.perf_counter()
        model = DeepCLV(
            horizon=12, hidden=(64, 32), epochs=18, lr=4e-3, discount_rate=0.01,
            backend=backend, random_state=7, batch_size=512, value_loss=loss_kind, n_estimators=120,
        ).fit(Xtr, ytr, etr)
        elapsed = time.perf_counter() - t0
        pred = model.predict(Xte)
        mse = float(np.mean((yte - pred) ** 2))
        r2 = 1.0 - mse / baseline_mse
        order = np.argsort(np.argsort(pred)).astype(float)
        order_y = np.argsort(np.argsort(yte)).astype(float)
        spearman = float(np.corrcoef(order, order_y)[0, 1])
        print(f"     {backend:<10}{loss_kind:<10}{elapsed:>8.1f}{mse:>12.1f}{baseline_mse:>12.1f}{r2:>12.4f}{spearman:>10.3f}")
        deep_rows.append((backend, loss_kind, mse, r2, spearman))

        check(f"DeepCLV[{backend}/{loss_kind}] beats predicting the mean", mse < baseline_mse,
              f"model MSE {mse:.1f} vs mean-baseline {baseline_mse:.1f}")
        check(f"DeepCLV[{backend}/{loss_kind}] ranks customers", spearman > 0.20, f"spearman {spearman:.3f}")
        check(f"DeepCLV[{backend}/{loss_kind}] predictions non-negative and finite",
              bool(np.all(np.isfinite(pred)) and np.all(pred >= 0)), f"min {pred.min():.4f}")
        pa, vga = model.predict_components(Xte)
        check(f"DeepCLV[{backend}/{loss_kind}] p_alive in [0,1]", bool(np.all((pa >= 0) & (pa <= 1))),
              f"range [{pa.min():.3f}, {pa.max():.3f}]")
        check(f"DeepCLV[{backend}/{loss_kind}] heads multiply back to predict",
              bool(np.allclose(pa * vga, pred, rtol=1e-9, atol=1e-9)), "p_alive * value == predict")
        check(f"DeepCLV[{backend}/{loss_kind}] p_alive separates retained from churned",
              float(pa[retained[n_tr:] == 1].mean() - pa[retained[n_tr:] == 0].mean()) > 0.10,
              "retention head has signal")
        disc = model.predict_discounted(Xte)
        check(f"DeepCLV[{backend}/{loss_kind}] discounting reduces value",
              bool(np.all(disc <= pred + 1e-12)) and disc.sum() < pred.sum(), "predict_discounted < predict")

    # determinism: same seed -> bit-identical, different seed -> different
    det_backend = "torch" if HAS_TORCH else "sklearn"
    m1 = DeepCLV(hidden=(32, 16), epochs=5, backend=det_backend, random_state=3, n_estimators=50).fit(Xtr, ytr, etr)
    m2 = DeepCLV(hidden=(32, 16), epochs=5, backend=det_backend, random_state=3, n_estimators=50).fit(Xtr, ytr, etr)
    check(f"DeepCLV[{det_backend}] is bit-identical under a fixed seed",
          bool(np.array_equal(m1.predict(Xte), m2.predict(Xte))), "two fits, same random_state")
    print(f"     determinism ({det_backend}): two fits with random_state=3 are bit-identical")

    # the sklearn fallback must be deterministic too -- when torch is present the loop
    # above never tests it, which is exactly when a non-reproducible fallback hides
    s1 = DeepCLV(backend="sklearn", random_state=3, n_estimators=50).fit(Xtr, ytr, etr)
    s2 = DeepCLV(backend="sklearn", random_state=3, n_estimators=50).fit(Xtr, ytr, etr)
    check("DeepCLV[sklearn] is bit-identical under a fixed seed",
          bool(np.array_equal(s1.predict(Xte), s2.predict(Xte))), "two fits, same random_state")
    # ... and a DIFFERENT seed must actually change something, or "determinism" is vacuous
    s3 = DeepCLV(backend="sklearn", random_state=99, n_estimators=50).fit(Xtr, ytr, etr)
    check("DeepCLV[sklearn] random_state actually does something",
          not np.array_equal(s1.predict(Xte), s3.predict(Xte)), "different seed -> different fit")

    # sklearn fallback is reachable even when torch exists
    forced = DeepCLV(backend="sklearn", random_state=1, n_estimators=60).fit(Xtr, ytr, etr)
    check("backend='sklearn' forces the fallback even when torch is installed", forced.backend_ == "sklearn",
          f"backend_={forced.backend_}")

    # event_observed must be read with the PRISM convention (1 = churn OBSERVED). If the
    # sign were flipped the retention head would be anti-correlated with truth, which the
    # separation check below would catch but only for this one wiring.
    flipped = DeepCLV(hidden=(32, 16), epochs=8, backend=det_backend, random_state=5,
                      n_estimators=50).fit(Xtr, ytr, 1.0 - etr)
    pa_right, _ = m1.predict_components(Xte)
    pa_wrong, _ = flipped.predict_components(Xte)
    ret_te = retained[n_tr:]
    sep_right = float(pa_right[ret_te == 1].mean() - pa_right[ret_te == 0].mean())
    sep_wrong = float(pa_wrong[ret_te == 1].mean() - pa_wrong[ret_te == 0].mean())
    check("event_observed=1 means churn OBSERVED (sign convention)", sep_right > 0 > sep_wrong,
          f"separation {sep_right:+.3f} correct vs {sep_wrong:+.3f} with the label flipped")

    # ---------------------------------------------------------------- 11. misc contracts
    check("BGNBD params_ has exactly the contracted keys", set(bg.params_) == {"r", "alpha", "a", "b"}, str(sorted(bg.params_)))
    check("GammaGamma params_ has exactly the contracted keys", set(gg.params_) == {"p", "q", "v"}, str(sorted(gg.params_)))
    check("estimators expose sklearn get_params", "penalizer" in BGNBD().get_params() and "horizon" in DeepCLV().get_params(),
          "BaseEstimator contract")
    holdout = simulate_clv_population(800, **TRUTH, **GG_TRUTH, random_state=12)
    ll_true = BGNBD().fit(holdout["frequency"], holdout["recency"], holdout["T"]).log_likelihood(
        holdout["frequency"], holdout["recency"], holdout["T"])
    ll_fit = bg.log_likelihood(holdout["frequency"], holdout["recency"], holdout["T"])
    check("fitted parameters generalise to a fresh sample", ll_fit <= ll_true + 1e-6 and ll_fit > ll_true - 25.0,
          f"transfer LL {ll_fit:.2f} vs in-sample-refit LL {ll_true:.2f}")
    print(f"\n[11] transfer to a fresh 800-customer cohort: LL {ll_fit:.2f} "
          f"(a model refitted on that cohort scores {ll_true:.2f})")

    # ---------------------------------------------------------------- 12. does it FORECAST?
    # Every check so far is internal: the likelihood is self-consistent, the closed forms
    # agree with references, the parameters come back. None of that proves the model
    # predicts anything. This does. Fit on a calibration window only, forecast the next
    # HOLD periods, and compare against the purchases that actually happened there --
    # data the fit never saw. This is the check a marketing team would actually run.
    HOLD = 10.0
    fc = simulate_clv_population(12000, **TRUTH, **GG_TRUTH, t_min=25.0, t_max=45.0,
                                 holdout=HOLD, random_state=101)
    f_c, r_c, T_c = (fc[c].to_numpy() for c in ("frequency", "recency", "T"))
    actual = fc["frequency_holdout"].to_numpy()
    bg_fc = BGNBD(random_state=0).fit(f_c, r_c, T_c)
    pred = bg_fc.conditional_expected_transactions(HOLD, f_c, r_c, T_c)
    agg_ratio = float(pred.sum() / max(actual.sum(), 1.0))
    check("forward forecast is unbiased in aggregate", 0.92 < agg_ratio < 1.08,
          f"predicted {pred.sum():.0f} vs actual {actual.sum():.0f} ({agg_ratio:.4f})")
    fc_corr = float(np.corrcoef(pred, actual)[0, 1])
    check("forward forecast is correlated with what actually happened", fc_corr > 0.45,
          f"corr {fc_corr:.3f}")
    print(f"\n[12] out-of-time forecast: fit on [0, T], predict the next {HOLD:.0f} periods,")
    print(f"     score against purchases the fit never saw (n={len(fc)})")
    print(f"     aggregate predicted {pred.sum():8.0f}   actual {actual.sum():8.0f}   "
          f"ratio {agg_ratio:.4f}   corr {fc_corr:.3f}")
    print(f"     {'x in calib.':>12}{'n':>7}{'predicted':>12}{'actual':>10}{'ratio':>8}")
    worst_bin_ratio = 0.0
    for lo, hi, lbl in [(0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 10**6, ">=8")]:
        s = (f_c >= lo) & (f_c <= hi)
        if s.sum() < 100:
            continue
        pm, am = float(pred[s].mean()), float(actual[s].mean())
        print(f"     {lbl:>12}{int(s.sum()):>7}{pm:>12.4f}{am:>10.4f}{pm / max(am, 1e-9):>8.3f}")
        worst_bin_ratio = max(worst_bin_ratio, abs(pm / max(am, 1e-9) - 1.0))
    check("forward forecast is unbiased WITHIN every frequency stratum", worst_bin_ratio < 0.20,
          f"worst per-stratum ratio error {worst_bin_ratio:.1%}")

    # p_alive must be CALIBRATED against the simulator's latent aliveness, not merely
    # separate the two groups on average -- a monotone but badly scaled score would pass
    # the separation check and fail this one
    pa_fc = bg_fc.probability_alive(f_c, r_c, T_c)
    truth_fc = fc["alive_true"].to_numpy().astype(float)
    worst_gap, edges = 0.0, np.linspace(0.0, 1.0, 11)
    for i in range(10):
        sel = (pa_fc >= edges[i]) & (pa_fc < edges[i + 1] + (1e-9 if i == 9 else 0.0))
        if sel.sum() < 100:
            continue
        worst_gap = max(worst_gap, abs(float(pa_fc[sel].mean()) - float(truth_fc[sel].mean())))
    brier = float(np.mean((pa_fc - truth_fc) ** 2))
    brier_base = float(np.mean((truth_fc.mean() - truth_fc) ** 2))
    check("p_alive is calibrated against latent aliveness", worst_gap < 0.08,
          f"worst decile gap {worst_gap:.4f}")
    check("p_alive has positive Brier skill", brier < brier_base,
          f"Brier {brier:.4f} vs base rate {brier_base:.4f} (skill {1 - brier / brier_base:.3f})")
    print(f"     p_alive calibration: worst decile gap {worst_gap:.4f}, "
          f"Brier skill {1 - brier / brier_base:.3f}")

    # the Gamma-Gamma posterior mean must estimate the LATENT per-customer spend better
    # than the raw observed average -- that is what shrinkage is FOR
    gg_fc = GammaGamma(random_state=0).fit(fc["frequency"], fc["monetary_value"])
    latent_spend = GG_TRUTH["p"] / fc["nu_true"].to_numpy()
    m_fc = fc["monetary_value"].to_numpy()
    obs = fc["frequency"].to_numpy() > 0
    shrunk_pred = gg_fc.conditional_expected_average_profit(fc["frequency"], m_fc)
    mse_shrunk = float(np.mean((shrunk_pred[obs] - latent_spend[obs]) ** 2))
    mse_raw = float(np.mean((m_fc[obs] - latent_spend[obs]) ** 2))
    check("Gamma-Gamma shrinkage beats the raw observed mean at recovering latent spend",
          mse_shrunk < mse_raw, f"MSE {mse_shrunk:.1f} vs raw {mse_raw:.1f} "
                                f"({100 * (1 - mse_shrunk / mse_raw):.1f}% reduction)")
    print(f"     Gamma-Gamma shrinkage vs latent truth: MSE {mse_shrunk:.1f} "
          f"vs {mse_raw:.1f} unshrunk ({100 * (1 - mse_shrunk / mse_raw):.1f}% better)")

    for bad_call, label in [
        (lambda: bg.probability_alive([1.0], [5.0], [3.0]), "recency > T rejected"),
        (lambda: bg.probability_alive([0.0], [5.0], [30.0]), "frequency=0 with recency>0 rejected"),
        (lambda: clv_from_survival(np.ones((3, 4)), np.ones(2)), "margin length mismatch rejected"),
        (lambda: probabilistic_clv(pop.drop(columns=["T"]), 12), "missing column rejected"),
    ]:
        try:
            bad_call()
            check(label, False, "no exception raised")
        except (ValueError, KeyError):
            check(label, True, "raised as expected")

    # ---------------------------------------------------------------- summary
    elapsed_total = time.perf_counter() - t_start
    print("\n" + "=" * 96)
    print(f"{'assertion':<64}{'result':>10}   detail")
    print("-" * 96)
    for name, ok, detail in checks:
        detail_short = detail if len(detail) <= 40 else detail[:37] + "..."
        print(f"{name:<64}{'PASS' if ok else 'FAIL':>10}   {detail_short}")
    print("-" * 96)
    print(f"{len(checks)} assertions, all passed, in {elapsed_total:.1f}s")
    print("clv.py OK")
