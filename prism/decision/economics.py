"""Unit economics -- the translation layer from a causal effect into a decision in money.

Everything upstream of this module speaks in *months of retained life* or in *discounted
currency of incremental value*. Everything downstream (budget allocation, off-policy
evaluation, the API, the dashboard) speaks in *money you would actually book*. This module
is the only place where the exchange rate between the two is defined, so there is exactly
one file to audit when a finance partner disputes a number.

The one equation
----------------
For customer ``i`` and offer arm ``a >= 1``::

    net_ia = tau_value_a(X_i) - cost_a * redemption_a - fixed_campaign_cost
    net_i0 = 0                                     (doing nothing is always available)

Two properties of that equation carry all the weight.

**Arm 0 is exactly zero.** The do-nothing arm is free and, by the definition of an
incremental effect, has no incremental value: it is the reference point every contrast is
measured against. Making it exactly ``0.0`` (not "approximately zero", not "the predicted
baseline value") is what lets a downstream allocator read ``net_ia > 0`` as "this offer
pays for itself" and ``argmax_a net_ia == 0`` as "leave this customer alone".

**Cost is multiplied by the redemption rate.** A 25%-off coupon that a customer never
redeems costs the business the price of the send, not the price of the discount. Charging
the full face value of every offer to every targeted customer systematically *understates*
ROI -- typically by 30-50% at the redemption rates seen in retention programmes -- and is
one of the most common real-world errors in this analysis. It makes profitable campaigns
look unprofitable and pushes the optimiser towards under-treating. If your finance team
insists on booking face value, set ``offer_redemption_rate`` to all ones and say so out
loud; do not leave the assumption implicit.

Contents
--------
:class:`EconomicConfig`
    The unit-economics contract, defaulted from :mod:`prism.data.schema`.
:func:`expected_net_value`
    ``(n, n_arms-1)`` value CATE -> ``(n, n_arms)`` net value, column 0 exactly zero.
:func:`expected_cost_matrix`
    The redemption-adjusted cost actually charged to each customer-arm cell.
:func:`breakeven_effect`
    One quotable number per arm: the minimum effect that pays for the offer.
:func:`build_decision_frame`
    One tidy row per customer for the allocator, the API and the dashboard.
:func:`roi_summary`
    The campaign-level P&L of a finished assignment.

This module deliberately imports **nothing** from ``prism.models`` or ``prism.causal``: it
takes plain numpy arrays so it can be unit-tested, and reasoned about, on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from prism.data.schema import ARM_COSTS, ARM_NAMES, ARM_REDEMPTION, N_ARMS, SEGMENTS
from prism.utils.logging import get_logger

__all__ = [
    "EconomicConfig",
    "expected_net_value",
    "expected_cost_matrix",
    "breakeven_effect",
    "build_decision_frame",
    "roi_summary",
]

logger = get_logger(__name__)

#: Heuristic segment read-out: (best-arm effect is positive, churn-risk tercile) -> label.
#: Terciles are 0 = lowest risk, 1 = middle, 2 = highest.
_SEGMENT_TABLE: dict[tuple[bool, int], str] = {
    (True, 0): "sure_thing",     # offer helps, but they were not going to leave anyway
    (True, 1): "persuadable",    # offer helps and there is churn to prevent
    (True, 2): "persuadable",
    (False, 0): "sleeping_dog",  # offer hurts a customer who was content: classic wake-up
    (False, 1): "sleeping_dog",
    (False, 2): "lost_cause",    # already leaving and nothing on the menu moves them
}


# =====================================================================================
# Configuration
# =====================================================================================
@dataclass
class EconomicConfig:
    """Unit economics that convert a causal effect into a booked currency amount.

    Defaults are taken from :mod:`prism.data.schema` rather than retyped, so the cost of
    an offer is defined in exactly one place in the codebase.

    Parameters
    ----------
    arm_costs : tuple of float, default :data:`prism.data.schema.ARM_COSTS`
        Face value (marginal cost) of delivering each arm to one customer, in currency
        units. ``arm_costs[0]`` is the do-nothing arm and must be ``0``.
    margin_rate : float, default 0.30
        Gross margin fraction applied to revenue when a caller converts revenue into
        contribution. Must lie in ``(0, 1]``. This module does not apply it to
        ``value_cate``: by contract the incremental value handed in is *already* a
        margin-and-discount adjusted quantity (see ``SPEC.md`` section 4.3). The rate is
        carried here so that every consumer reads the same number.
    discount_rate : float, default 0.01
        Monthly discount rate used to present-value future margin. Must be ``>= 0``.
    horizon : int, default 12
        Number of months the value outcome is restricted to. Must be ``>= 1``.
    offer_redemption_rate : tuple of float, default :data:`prism.data.schema.ARM_REDEMPTION`
        Probability that a targeted customer actually redeems the offer. Cost is charged
        only on redemption. Every entry must lie in ``[0, 1]``.
    fixed_campaign_cost : float, default 0.0
        Per-customer fixed cost of *contacting* someone with any non-control arm (the
        send, the call, the postage). Charged whether or not the offer is redeemed, and
        never charged to arm 0.

    Attributes
    ----------
    n_arms : int
        Number of arms, inferred from ``arm_costs``.
    expected_arm_cost : numpy.ndarray
        Redemption-adjusted expected charge per arm, shape ``(n_arms,)``.

    Raises
    ------
    ValueError
        If the arm vectors disagree in length with :data:`prism.data.schema.N_ARMS`, if
        ``arm_costs[0] != 0``, if any redemption rate falls outside ``[0, 1]``, if
        ``margin_rate`` is outside ``(0, 1]``, if ``discount_rate < 0`` or if
        ``horizon < 1``.

    Examples
    --------
    >>> econ = EconomicConfig()
    >>> econ.n_arms
    4
    >>> float(np.round(econ.expected_arm_cost[1], 2))   # 12.0 * 0.62
    7.44
    """

    arm_costs: tuple[float, ...] = ARM_COSTS
    margin_rate: float = 0.30
    discount_rate: float = 0.01
    horizon: int = 12
    offer_redemption_rate: tuple[float, ...] = ARM_REDEMPTION
    fixed_campaign_cost: float = 0.0

    # -- validation -------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Coerce the arm vectors to tuples of float and validate every field.

        Raises
        ------
        ValueError
            On any violated invariant; the message names the offending field and value.
        """
        try:
            self.arm_costs = tuple(float(c) for c in self.arm_costs)
            self.offer_redemption_rate = tuple(float(r) for r in self.offer_redemption_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"arm_costs and offer_redemption_rate must be numeric sequences: {exc}"
            ) from exc

        self.margin_rate = float(self.margin_rate)
        self.discount_rate = float(self.discount_rate)
        self.horizon = int(self.horizon)
        self.fixed_campaign_cost = float(self.fixed_campaign_cost)

        if len(self.arm_costs) != N_ARMS or len(self.offer_redemption_rate) != N_ARMS:
            raise ValueError(
                "EconomicConfig expects one entry per arm: "
                f"len(arm_costs)={len(self.arm_costs)}, "
                f"len(offer_redemption_rate)={len(self.offer_redemption_rate)}, N_ARMS={N_ARMS}"
            )
        if self.arm_costs[0] != 0.0:
            raise ValueError(
                f"arm 0 is the do-nothing arm and must be free; got arm_costs[0]={self.arm_costs[0]!r}"
            )
        if any((not np.isfinite(c)) or c < 0.0 for c in self.arm_costs):
            raise ValueError(f"arm_costs must be finite and non-negative; got {self.arm_costs!r}")
        if not all(np.isfinite(r) and 0.0 <= r <= 1.0 for r in self.offer_redemption_rate):
            raise ValueError(
                f"offer_redemption_rate entries must lie in [0, 1]; got {self.offer_redemption_rate!r}"
            )
        if not np.isfinite(self.margin_rate) or not (0.0 < self.margin_rate <= 1.0):
            raise ValueError(f"margin_rate must lie in (0, 1]; got {self.margin_rate!r}")
        if not np.isfinite(self.discount_rate) or self.discount_rate < 0.0:
            raise ValueError(f"discount_rate must be >= 0; got {self.discount_rate!r}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1 month; got {self.horizon!r}")
        if not np.isfinite(self.fixed_campaign_cost) or self.fixed_campaign_cost < 0.0:
            raise ValueError(
                f"fixed_campaign_cost must be finite and non-negative; got {self.fixed_campaign_cost!r}"
            )
        if self.offer_redemption_rate[0] != 0.0:
            # Harmless (arm 0 is never charged) but almost always a copy/paste slip.
            logger.warning(
                "offer_redemption_rate[0]=%s is ignored: the control arm is never charged.",
                self.offer_redemption_rate[0],
            )

    # -- derived quantities -----------------------------------------------------------
    @property
    def n_arms(self) -> int:
        """int : Number of arms, including the do-nothing arm."""
        return len(self.arm_costs)

    @property
    def expected_arm_cost(self) -> np.ndarray:
        """numpy.ndarray : Expected charge per targeted customer, shape ``(n_arms,)``.

        ``arm_costs[a] * offer_redemption_rate[a] + fixed_campaign_cost`` for ``a >= 1``,
        and exactly ``0.0`` for the control arm.
        """
        cost = np.asarray(self.arm_costs, dtype=np.float64)
        red = np.asarray(self.offer_redemption_rate, dtype=np.float64)
        out = cost * red + self.fixed_campaign_cost
        out[0] = 0.0
        return out

    def discount_factors(self, horizon: int | None = None) -> np.ndarray:
        """Return present-value factors ``(1 + discount_rate) ** -t`` for ``t = 0..H-1``.

        Parameters
        ----------
        horizon : int, optional
            Number of monthly periods. Defaults to :attr:`horizon`.

        Returns
        -------
        numpy.ndarray
            Shape ``(horizon,)``, float64; strictly decreasing when ``discount_rate > 0``.

        Raises
        ------
        ValueError
            If the effective horizon is below 1.
        """
        h = int(self.horizon if horizon is None else horizon)
        if h < 1:
            raise ValueError(f"horizon must be >= 1; got {h}")
        return (1.0 + self.discount_rate) ** -np.arange(h, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serialisable dict of the configuration.

        Returns
        -------
        dict
            Keys mirror the dataclass fields; tuples become lists.
        """
        return {
            "arm_costs": list(self.arm_costs),
            "margin_rate": self.margin_rate,
            "discount_rate": self.discount_rate,
            "horizon": self.horizon,
            "offer_redemption_rate": list(self.offer_redemption_rate),
            "fixed_campaign_cost": self.fixed_campaign_cost,
        }


# =====================================================================================
# Internal shape helpers
# =====================================================================================
def _as_value_matrix(value_cate: Any, n_arms: int, *, name: str = "value_cate") -> np.ndarray:
    """Normalise a CATE argument to a ``(n, n_arms - 1)`` float64 matrix.

    Parameters
    ----------
    value_cate : array-like
        ``(n, n_arms - 1)`` matrix, or a 1-D array. A 1-D array is read as ``(n, 1)`` in
        the two-arm case, and as a single customer ``(1, n_arms - 1)`` when its length
        equals ``n_arms - 1``.
    n_arms : int
        Total number of arms including control.
    name : str, default "value_cate"
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, n_arms - 1)``, dtype float64, C-contiguous.

    Raises
    ------
    ValueError
        If the array is empty, has more than two dimensions, or cannot be read as one
        column per non-control arm.
    """
    arr = np.asarray(value_cate, dtype=np.float64)
    k = n_arms - 1
    if arr.size == 0:
        raise ValueError(f"{name} is empty; expected at least one row of shape (n, {k})")
    if arr.ndim == 1:
        if k == 1:
            arr = arr.reshape(-1, 1)
        elif arr.shape[0] == k:
            arr = arr.reshape(1, k)
        else:
            raise ValueError(
                f"{name} is 1-D with length {arr.shape[0]}; a 1-D {name} is only accepted for the "
                f"two-arm case or as a single customer of length n_arms-1={k}. "
                f"Pass a 2-D array with one column per non-control arm, shape (n, {k})."
            )
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D; got shape {arr.shape}")
    if arr.shape[1] != k:
        raise ValueError(
            f"{name} must have one column per non-control arm: expected shape (n, {k}) "
            f"for n_arms={n_arms}, got {arr.shape}"
        )
    return np.ascontiguousarray(arr)


def _broadcast_arm_matrix(
    value: Any,
    n: int,
    n_arms: int,
    *,
    name: str,
    default: np.ndarray,
) -> np.ndarray:
    """Broadcast a per-arm / per-row / per-cell override to a ``(n, n_arms)`` matrix.

    Parameters
    ----------
    value : scalar or array-like or None
        ``None`` tiles ``default`` over rows. Accepted shapes: scalar, ``(n_arms,)``,
        ``(n_arms - 1,)`` (non-control arms only), ``(n,)`` (one value per customer,
        shared by all arms), ``(n, n_arms)`` and ``(n, n_arms - 1)``.
    n : int
        Number of customers.
    n_arms : int
        Number of arms including control.
    name : str
        Argument name used in error messages.
    default : numpy.ndarray
        Shape ``(n_arms,)`` fallback, also used for the control column when only the
        non-control arms are supplied.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, n_arms)``, dtype float64.

    Raises
    ------
    ValueError
        If the shape cannot be broadcast, or if the result is not finite.

    Notes
    -----
    When ``n == n_arms`` a 1-D input is ambiguous; the per-arm reading wins. Pass a 2-D
    array to be explicit.
    """
    base = np.tile(np.asarray(default, dtype=np.float64).reshape(1, n_arms), (n, 1))
    if value is None:
        return base

    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        out = np.full((n, n_arms), float(arr), dtype=np.float64)
    elif arr.ndim == 1:
        if arr.shape[0] == n_arms:
            out = np.tile(arr.reshape(1, n_arms), (n, 1))
        elif arr.shape[0] == n_arms - 1:
            out = base.copy()
            out[:, 1:] = np.tile(arr.reshape(1, n_arms - 1), (n, 1))
        elif arr.shape[0] == n:
            out = np.tile(arr.reshape(n, 1), (1, n_arms))
        else:
            raise ValueError(
                f"{name} has length {arr.shape[0]}; expected a scalar, {n_arms} (per arm), "
                f"{n_arms - 1} (per non-control arm) or {n} (per customer) values"
            )
    elif arr.ndim == 2:
        if arr.shape == (n, n_arms):
            out = arr.astype(np.float64, copy=True)
        elif arr.shape == (n, n_arms - 1):
            out = base.copy()
            out[:, 1:] = arr
        else:
            raise ValueError(
                f"{name} has shape {arr.shape}; expected ({n}, {n_arms}) or ({n}, {n_arms - 1})"
            )
    else:
        raise ValueError(f"{name} must be at most 2-D; got shape {arr.shape}")

    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values; a cost or redemption override must be finite")
    return np.ascontiguousarray(out)


# =====================================================================================
# Core economics
# =====================================================================================
def expected_cost_matrix(
    n: int,
    econ: EconomicConfig,
    *,
    redemption_override: Any = None,
    cost_matrix: Any = None,
) -> np.ndarray:
    """Return the expected charge for every customer-arm cell.

    This is the cost a downstream allocator should budget against: the face value of the
    offer multiplied by the probability it is redeemed, plus the fixed contact cost. The
    control column is exactly ``0.0``.

    Parameters
    ----------
    n : int
        Number of customers.
    econ : EconomicConfig
        Unit economics.
    redemption_override : scalar or array-like, optional
        Customer-specific redemption probabilities replacing
        :attr:`EconomicConfig.offer_redemption_rate`. Accepted shapes: scalar,
        ``(n_arms,)``, ``(n_arms - 1,)``, ``(n,)``, ``(n, n_arms)``, ``(n, n_arms - 1)``.
        Validated to lie in ``[0, 1]``.
    cost_matrix : scalar or array-like, optional
        Customer-specific *face value* costs replacing :attr:`EconomicConfig.arm_costs`.
        Same accepted shapes. This is the pre-redemption cost: it is still multiplied by
        the redemption rate.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, n_arms)``, dtype float64, column 0 exactly zero.

    Raises
    ------
    TypeError
        If ``econ`` is not an :class:`EconomicConfig`.
    ValueError
        If ``n < 1``, if a shape cannot be broadcast, if any cost is negative, or if any
        redemption rate falls outside ``[0, 1]``.
    """
    if not isinstance(econ, EconomicConfig):
        raise TypeError(f"econ must be an EconomicConfig, got {type(econ).__name__}")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    k = econ.n_arms

    costs = _broadcast_arm_matrix(
        cost_matrix, n, k, name="cost_matrix", default=np.asarray(econ.arm_costs, dtype=np.float64)
    )
    red = _broadcast_arm_matrix(
        redemption_override,
        n,
        k,
        name="redemption_override",
        default=np.asarray(econ.offer_redemption_rate, dtype=np.float64),
    )
    if np.any(costs < 0.0):
        raise ValueError("cost_matrix contains negative costs")
    if np.any(red[:, 1:] < 0.0) or np.any(red[:, 1:] > 1.0):
        raise ValueError("redemption_override contains probabilities outside [0, 1]")

    out = costs * red + econ.fixed_campaign_cost
    out[:, 0] = 0.0
    return out


def expected_net_value(
    value_cate: np.ndarray,
    econ: EconomicConfig,
    *,
    redemption_override: Any = None,
    cost_matrix: Any = None,
) -> np.ndarray:
    """Convert an incremental value CATE into expected net value per arm.

    ``net_ia = tau_value_a(X_i) - cost_a * redemption_a - fixed_campaign_cost`` for every
    non-control arm, and ``net_i0 = 0.0`` exactly.

    Parameters
    ----------
    value_cate : numpy.ndarray
        Shape ``(n, n_arms - 1)``: the estimated incremental *discounted value* of each
        non-control arm relative to control, in currency units. A 1-D array is accepted
        in the two-arm case (reshaped to ``(n, 1)``) and as a single customer when its
        length equals ``n_arms - 1``.
    econ : EconomicConfig
        Unit economics.
    redemption_override : scalar or array-like, optional
        Per-customer redemption probabilities. Accepted shapes: scalar, ``(n_arms,)``,
        ``(n_arms - 1,)``, ``(n,)``, ``(n, n_arms)``, ``(n, n_arms - 1)``. Use this when a
        propensity-to-redeem model exists: a customer who always redeems genuinely costs
        more to treat than one who never does, and holding that constant leaves money on
        the table in both directions.
    cost_matrix : scalar or array-like, optional
        Per-customer *face value* offer costs; same accepted shapes. Use this when the
        offer is proportional to spend (a 10% discount costs more on a large basket) or
        when channel costs differ by region. It is multiplied by the redemption rate
        exactly as the flat ``arm_costs`` are.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, n_arms)``, dtype float64. Column 0 is the do-nothing arm and is
        exactly ``0.0``; it is the reference point for every contrast, so an allocator can
        read ``argmax`` directly as "treat / do not treat".

    Raises
    ------
    TypeError
        If ``econ`` is not an :class:`EconomicConfig`.
    ValueError
        If ``value_cate`` cannot be read as one column per non-control arm, or if an
        override cannot be broadcast.

    Notes
    -----
    **Why cost is multiplied by the redemption rate.** An offer that is never redeemed
    costs only the send. Charging every targeted customer the full face value understates
    ROI -- at the default rates a 25%-off offer is charged at 30.00 when its true expected
    cost is 21.30 -- and makes the optimiser too timid, shrinking the treated population
    and the incremental profit with it. The flip side is that redemption and effect are
    not independent: the customers who redeem are usually the ones the offer moves. If you
    have a redemption model, pass it through ``redemption_override`` rather than relying on
    the campaign-average constant.

    ``NaN`` in ``value_cate`` propagates to that customer-arm cell rather than being
    silently zeroed; a downstream ``argmax`` would then be ill-defined, so callers should
    impute or drop before allocating.

    Examples
    --------
    >>> econ = EconomicConfig()
    >>> net = expected_net_value(np.array([[40.0, 40.0, 40.0]]), econ)
    >>> net.shape
    (1, 4)
    >>> float(net[0, 0])
    0.0
    >>> bool(np.isclose(net[0, 1], 40.0 - 12.0 * 0.62))
    True
    """
    if not isinstance(econ, EconomicConfig):
        raise TypeError(f"econ must be an EconomicConfig, got {type(econ).__name__}")

    tau = _as_value_matrix(value_cate, econ.n_arms)
    n = tau.shape[0]
    charge = expected_cost_matrix(
        n, econ, redemption_override=redemption_override, cost_matrix=cost_matrix
    )

    net = np.zeros((n, econ.n_arms), dtype=np.float64)
    net[:, 1:] = tau - charge[:, 1:]
    net[:, 0] = 0.0  # explicit: the do-nothing arm is exactly free and exactly zero
    return net


def breakeven_effect(econ: EconomicConfig, **kwargs: Any) -> np.ndarray:
    """Minimum incremental value each arm must deliver to pay for itself.

    The single most quotable number this project produces: *"the concierge save call has
    to buy 26.40 currency units of discounted margin before it breaks even."* Any
    estimated effect below this line loses money, however statistically significant it is.

    Parameters
    ----------
    econ : EconomicConfig
        Unit economics.
    **kwargs
        Forwarded to :func:`expected_cost_matrix` (``redemption_override``,
        ``cost_matrix``) when a per-arm, campaign-level override is wanted. Per-customer
        overrides make no sense here because the return value is per arm, not per
        customer, and are rejected by the shape check.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_arms,)``, dtype float64. Entry 0 is ``0.0`` -- doing nothing breaks
        even by definition. Entry ``a`` is
        ``arm_costs[a] * offer_redemption_rate[a] + fixed_campaign_cost``.

    Examples
    --------
    >>> np.round(breakeven_effect(EconomicConfig()), 2).tolist()
    [0.0, 7.44, 21.3, 26.4]
    """
    if kwargs:
        return expected_cost_matrix(1, econ, **kwargs)[0]
    if not isinstance(econ, EconomicConfig):
        raise TypeError(f"econ must be an EconomicConfig, got {type(econ).__name__}")
    return econ.expected_arm_cost


# =====================================================================================
# Decision frame
# =====================================================================================
def _risk_terciles(churn_risk: np.ndarray) -> np.ndarray:
    """Return 0/1/2 tercile labels of ``churn_risk`` (2 = highest risk).

    Parameters
    ----------
    churn_risk : numpy.ndarray
        Length-``n`` baseline churn probabilities.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``, dtype int64.

    Notes
    -----
    Average ranks are used so ties are handled deterministically, and a degenerate
    (constant or all-missing) risk vector maps to the middle tercile rather than piling
    every customer into one edge. A *missing* risk for an individual customer also maps
    to the middle tercile: ranking ``NaN`` alongside the observed values would quietly
    assert that an unscored customer is the safest customer in the book.
    """
    x = np.asarray(churn_risk, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=np.int64)
    ok = np.isfinite(x)
    if not ok.any() or float(np.ptp(x[ok])) == 0.0:
        return np.full(x.shape[0], 1, dtype=np.int64)
    # Rank the observed risks only. A missing risk is NOT evidence of low risk, so it
    # must not be ranked alongside the observed values: it lands in the neutral middle
    # tercile, the same fallback the degenerate case above uses.
    out = np.full(x.shape[0], 1, dtype=np.int64)
    pct = pd.Series(x[ok]).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    out[ok] = np.where(pct <= 1.0 / 3.0, 0, np.where(pct <= 2.0 / 3.0, 1, 2))
    return out


def build_decision_frame(
    customer_id: Sequence[Any] | np.ndarray | pd.Series,
    value_cate: np.ndarray,
    rmst_cate: np.ndarray | None,
    churn_risk: np.ndarray | pd.Series,
    clv: np.ndarray | pd.Series,
    econ: EconomicConfig,
    *,
    redemption_override: Any = None,
    cost_matrix: Any = None,
    arm_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Assemble the tidy, one-row-per-customer decision table.

    This frame is the hand-off artefact of the whole project: the budget optimiser reads
    the ``net_value_*`` columns, the API serialises one row per scored customer, and the
    dashboard renders it directly. It is *unconstrained* -- ``recommended_arm`` is the
    profit-maximising arm ignoring any budget. Budget enters later, in
    :mod:`prism.decision.optimize`.

    Parameters
    ----------
    customer_id : sequence
        Length-``n`` identifiers, copied verbatim. Not required to be unique, though the
        caller almost certainly wants it to be.
    value_cate : numpy.ndarray
        ``(n, n_arms - 1)`` incremental discounted value per non-control arm, currency.
    rmst_cate : numpy.ndarray or None
        ``(n, n_arms - 1)`` incremental restricted mean survival time per non-control arm,
        in months. ``None`` fills the ``cate_rmst_*`` columns with ``NaN`` -- reported as
        missing rather than faked as zero -- and logs a warning.
    churn_risk : array-like
        Length-``n`` baseline (untreated) churn probability, used for ``segment_guess``.
        Values outside ``[0, 1]`` are clipped, with a logged warning.
    clv : array-like
        Length-``n`` predicted customer lifetime value. Carried through for display and
        for the ``highest_clv`` baseline policy; it does not affect ``recommended_arm``.
    econ : EconomicConfig
        Unit economics.
    redemption_override : scalar or array-like, optional
        Forwarded to :func:`expected_net_value`; see there for accepted shapes.
    cost_matrix : scalar or array-like, optional
        Forwarded to :func:`expected_net_value`; see there for accepted shapes.
    arm_names : sequence of str, optional
        Display names, defaulting to :data:`prism.data.schema.ARM_NAMES`. Must have
        ``n_arms`` entries.

    Returns
    -------
    pandas.DataFrame
        Exactly ``n`` rows, with columns in this order:

        ``customer_id``, ``churn_risk``, ``clv``; then, for each arm ``a`` in
        ``0..n_arms-1``, ``cate_value_<a>``, ``cate_rmst_<a>``, ``cost_<a>``,
        ``net_value_<a>`` (all four are ``0.0`` for arm 0, the free reference arm); then
        ``recommended_arm`` (int64, unconstrained argmax of net value with arm 0 winning
        ties), ``recommended_arm_name`` (str), ``expected_net_value`` (that maximum,
        floored at 0), ``uplift_rank`` (int64 dense rank by ``expected_net_value``,
        1 = best) and ``segment_guess`` (str).

    Raises
    ------
    TypeError
        If ``econ`` is not an :class:`EconomicConfig`.
    ValueError
        If any input length disagrees with ``len(customer_id)``, or if ``rmst_cate`` does
        not have the same shape as ``value_cate``.

    Notes
    -----
    **``segment_guess`` is a heuristic read-out for the dashboard, NOT the ground-truth
    segment.** It is derived from model output, never from the simulator's latent
    ``gt_segment``, and it must not be used to score the estimator -- doing so would be
    circular. It crosses the sign of the best available arm effect with the churn-risk
    tercile:

    ===================  ============  ============  ============
    best effect / risk   low tercile   mid tercile   high tercile
    ===================  ============  ============  ============
    positive             sure_thing    persuadable   persuadable
    non-positive         sleeping_dog  sleeping_dog  lost_cause
    ===================  ============  ============  ============

    The reading: a positive effect only matters if there is churn to prevent, so a
    low-risk customer who "responds" is really a *sure thing* you would have kept anyway;
    a negative effect on a contented customer is the *sleeping dog* the offer wakes up;
    a negative or zero effect on someone already heading for the door is a *lost cause*.
    Every label lands in :data:`prism.data.schema.SEGMENTS`.

    The sign is taken from the best *available* arm effect rather than from the chosen
    arm's effect: an arm can be helpful yet unaffordable, and the segment story is about
    how the customer responds, not about what the offer costs.
    """
    if not isinstance(econ, EconomicConfig):
        raise TypeError(f"econ must be an EconomicConfig, got {type(econ).__name__}")

    if isinstance(customer_id, (pd.Series, pd.Index)):
        ids = pd.Series(np.asarray(customer_id))
    else:
        ids = pd.Series(np.asarray(list(customer_id) if not isinstance(customer_id, np.ndarray) else customer_id))
    ids = ids.reset_index(drop=True)
    n = int(len(ids))
    if n == 0:
        raise ValueError("customer_id is empty; build_decision_frame needs at least one customer")

    k = econ.n_arms
    names = tuple(ARM_NAMES if arm_names is None else arm_names)
    if len(names) != k:
        raise ValueError(f"arm_names must have {k} entries, got {len(names)}")

    tau_value = _as_value_matrix(value_cate, k)
    if tau_value.shape[0] != n:
        raise ValueError(f"value_cate has {tau_value.shape[0]} rows but customer_id has {n}")

    if rmst_cate is None:
        tau_rmst = np.full((n, k - 1), np.nan, dtype=np.float64)
        logger.warning("build_decision_frame: rmst_cate is None; cate_rmst_* columns filled with NaN")
    else:
        tau_rmst = _as_value_matrix(rmst_cate, k, name="rmst_cate")
        if tau_rmst.shape != tau_value.shape:
            raise ValueError(
                f"rmst_cate shape {tau_rmst.shape} does not match value_cate shape {tau_value.shape}"
            )

    risk = np.asarray(churn_risk, dtype=np.float64).ravel()
    clv_arr = np.asarray(clv, dtype=np.float64).ravel()
    for arr, label in ((risk, "churn_risk"), (clv_arr, "clv")):
        if arr.shape[0] != n:
            raise ValueError(f"{label} has length {arr.shape[0]} but customer_id has {n}")
    if bool(np.any(np.isfinite(risk) & ((risk < 0.0) | (risk > 1.0)))):
        logger.warning("build_decision_frame: churn_risk had values outside [0, 1]; clipping for segment_guess")
        risk = np.clip(risk, 0.0, 1.0)

    charge = expected_cost_matrix(
        n, econ, redemption_override=redemption_override, cost_matrix=cost_matrix
    )
    net = expected_net_value(
        tau_value, econ, redemption_override=redemption_override, cost_matrix=cost_matrix
    )

    zeros = np.zeros(n, dtype=np.float64)
    data: dict[str, Any] = {
        "customer_id": ids.to_numpy(),
        "churn_risk": risk,
        "clv": clv_arr,
    }
    for a in range(k):
        data[f"cate_value_{a}"] = zeros.copy() if a == 0 else tau_value[:, a - 1]
        data[f"cate_rmst_{a}"] = zeros.copy() if a == 0 else tau_rmst[:, a - 1]
        data[f"cost_{a}"] = charge[:, a]
        data[f"net_value_{a}"] = net[:, a]

    # Unconstrained argmax. np.argmax returns the FIRST maximal index, and column 0 is
    # both first and exactly 0.0, so ties resolve to "do nothing" -- the conservative
    # default a retention team should want. NaN cells are sent to -inf so a missing
    # estimate can never win an arm.
    recommended = np.argmax(np.nan_to_num(net, nan=-np.inf), axis=1).astype(np.int64)
    best_net = np.maximum(net[np.arange(n), recommended], 0.0)

    # np.nanmax would emit "All-NaN slice encountered" for a customer with no usable
    # estimate on any arm, and np.errstate does not suppress that (it is a warnings
    # warning, not a floating-point flag). Mask by hand instead: a missing arm is simply
    # not a candidate, and a customer with no candidates is treated as non-positive.
    if tau_value.shape[1]:
        best_effect = np.where(np.isfinite(tau_value), tau_value, -np.inf).max(axis=1)
    else:
        best_effect = zeros.copy()
    n_unscored = int(np.count_nonzero(~np.isfinite(best_effect)))
    if n_unscored:
        logger.warning(
            "build_decision_frame: %d of %d customers have no finite value_cate on any arm; "
            "their segment_guess falls back to the non-positive row of the heuristic table.",
            n_unscored,
            n,
        )
    best_effect = np.where(np.isfinite(best_effect), best_effect, 0.0)
    positive = best_effect > 0.0
    tercile = _risk_terciles(risk)
    segment = np.empty(n, dtype=object)
    for (sign_positive, terc), label in _SEGMENT_TABLE.items():
        segment[(positive == sign_positive) & (tercile == terc)] = label

    data["recommended_arm"] = recommended
    data["recommended_arm_name"] = np.asarray([names[int(a)] for a in recommended], dtype=object)
    data["expected_net_value"] = best_net
    data["uplift_rank"] = pd.Series(best_net).rank(method="dense", ascending=False).to_numpy(dtype=np.int64)
    data["segment_guess"] = segment

    return pd.DataFrame(data)


# =====================================================================================
# Campaign P&L
# =====================================================================================
def roi_summary(
    assignment: np.ndarray,
    net_value: np.ndarray,
    cost_matrix: np.ndarray | None,
    econ: EconomicConfig,
    *,
    rmst_cate: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarise the realised P&L of a finished assignment.

    Parameters
    ----------
    assignment : numpy.ndarray
        Shape ``(n,)`` integer arm chosen per customer, ``0`` meaning "no offer".
    net_value : numpy.ndarray
        Shape ``(n, n_arms)`` expected net value, as returned by
        :func:`expected_net_value`.
    cost_matrix : numpy.ndarray or None
        Shape ``(n, n_arms)`` cost charged per customer-arm cell. Pass the
        **redemption-adjusted** matrix from :func:`expected_cost_matrix` -- which is what
        ``None`` builds -- so that ``gross_incremental_value`` reads as the sum of the
        targeted customers' incremental value. If you pass raw face-value costs instead,
        the identity ``net == gross - cost`` still holds by construction, but ``gross`` is
        then "net plus invoiced spend", not the causal effect.
    econ : EconomicConfig
        Unit economics, used for the default cost matrix and for arm-count validation.
    rmst_cate : numpy.ndarray, optional
        Shape ``(n, n_arms - 1)`` incremental RMST in months. When supplied, the summary
        gains ``cost_per_retained_month``.

    Returns
    -------
    dict
        ``n_targeted`` (int), ``total_cost``, ``gross_incremental_value``,
        ``net_incremental_value``, ``roi``, ``value_per_currency_spent``, and
        ``cost_per_retained_month`` when ``rmst_cate`` is given. ``roi = net / cost`` and
        ``value_per_currency_spent = gross / cost``; both are ``nan`` when nothing was
        spent, because a ratio to zero spend is not a number a slide should carry.

    Raises
    ------
    TypeError
        If ``econ`` is not an :class:`EconomicConfig`.
    ValueError
        On shape disagreement, or on an arm index outside ``0..n_arms-1``.

    Notes
    -----
    ``net_incremental_value == gross_incremental_value - total_cost`` holds exactly, by
    construction: ``gross`` is recovered as ``net + cost`` rather than recomputed from the
    effects, so the three headline numbers on the report can never disagree with each
    other. Only customers with ``assignment > 0`` contribute cost; the control arm is free.

    A targeted customer whose assigned cell holds a non-finite net value or cost cannot be
    booked and is excluded from all three money figures. They remain in ``n_targeted``
    (they were, after all, targeted) and the exclusion is logged with its count and share,
    so a P&L built on incomplete estimates cannot read as full coverage.
    """
    if not isinstance(econ, EconomicConfig):
        raise TypeError(f"econ must be an EconomicConfig, got {type(econ).__name__}")

    raw = np.asarray(assignment).ravel()
    if raw.dtype.kind == "b":
        raw = raw.astype(np.int64)
    elif raw.dtype.kind in "fc":
        # A float assignment is almost always a slip (an argmax routed through a
        # DataFrame, say). Casting it would silently floor 1.9 to arm 1, and would emit
        # "invalid value encountered in cast" on a NaN, so validate before casting.
        real = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(real)):
            raise ValueError(
                f"assignment contains {int(np.count_nonzero(~np.isfinite(real)))} non-finite "
                "arm indices; every customer must be assigned a concrete arm"
            )
        if not np.all(real == np.rint(real)):
            raise ValueError(
                "assignment must contain whole-number arm indices; got fractional values "
                "(an arm index is a label, not a quantity)"
            )
        raw = np.rint(real)
    arm = raw.astype(np.int64)
    net = np.asarray(net_value, dtype=np.float64)
    if net.ndim != 2 or net.shape[1] != econ.n_arms:
        raise ValueError(f"net_value must have shape (n, {econ.n_arms}); got {net.shape}")
    n = int(net.shape[0])
    if arm.shape[0] != n:
        raise ValueError(f"assignment has length {arm.shape[0]} but net_value has {n} rows")
    if arm.size and (int(arm.min()) < 0 or int(arm.max()) >= econ.n_arms):
        raise ValueError(f"assignment contains arm indices outside 0..{econ.n_arms - 1}")

    if cost_matrix is None:
        # n == 0 is a legitimate degenerate campaign (an empty fold, an empty protected
        # group); expected_cost_matrix rejects it, so build the empty basis directly.
        cost = expected_cost_matrix(n, econ) if n else np.zeros((0, econ.n_arms), dtype=np.float64)
    else:
        cost = np.asarray(cost_matrix, dtype=np.float64)
        if cost.shape != (n, econ.n_arms):
            raise ValueError(f"cost_matrix must have shape ({n}, {econ.n_arms}); got {cost.shape}")

    rows = np.arange(n)
    treated = arm > 0
    raw_net = net[rows, arm]
    raw_cost = cost[rows, arm]

    # A targeted customer whose assigned cell is NaN or infinite cannot be booked. The
    # previous np.nansum dropped them without a word, which is exactly the failure mode
    # that makes a P&L look clean when it rests on incomplete estimates. They are still
    # counted in n_targeted, so say out loud how many were excluded.
    unusable = treated & (~np.isfinite(raw_net) | ~np.isfinite(raw_cost))
    n_unusable = int(np.count_nonzero(unusable))
    if n_unusable:
        n_treated = int(np.count_nonzero(treated))
        logger.warning(
            "roi_summary: %d of %d targeted customers (%.2f%%) have a non-finite net value "
            "or cost for their assigned arm; they are excluded from total_cost, "
            "gross_incremental_value and net_incremental_value but remain in n_targeted.",
            n_unusable,
            n_treated,
            100.0 * n_unusable / max(n_treated, 1),
        )
    bookable = treated & ~unusable
    chosen_net = np.where(bookable, raw_net, 0.0)
    chosen_cost = np.where(bookable, raw_cost, 0.0)

    total_cost = float(chosen_cost.sum())
    net_incremental = float(chosen_net.sum())
    gross_incremental = net_incremental + total_cost

    with np.errstate(divide="ignore", invalid="ignore"):
        roi = float(net_incremental / total_cost) if total_cost > 0.0 else float("nan")
        value_per_currency = float(gross_incremental / total_cost) if total_cost > 0.0 else float("nan")

    out: dict[str, float] = {
        "n_targeted": int(np.count_nonzero(treated)),
        "total_cost": total_cost,
        "gross_incremental_value": gross_incremental,
        "net_incremental_value": net_incremental,
        "roi": roi,
        "value_per_currency_spent": value_per_currency,
    }

    if rmst_cate is not None:
        tau_rmst = _as_value_matrix(rmst_cate, econ.n_arms, name="rmst_cate")
        if tau_rmst.shape[0] != n:
            raise ValueError(f"rmst_cate has {tau_rmst.shape[0]} rows but net_value has {n}")
        padded = np.zeros((n, econ.n_arms), dtype=np.float64)
        padded[:, 1:] = tau_rmst
        raw_months = padded[rows, arm]
        months = float(np.where(bookable & np.isfinite(raw_months), raw_months, 0.0).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            out["cost_per_retained_month"] = float(total_cost / months) if months > 0.0 else float("nan")

    return out


# =====================================================================================
# Smoke test
# =====================================================================================
if __name__ == "__main__":  # pragma: no cover - smoke test
    import logging
    import time
    import warnings

    from prism.data.schema import SEGMENT_SHARES
    from prism.utils.seeds import as_rng

    t_start = time.perf_counter()
    rng = as_rng(7)
    n_cust = 5_000
    econ = EconomicConfig()
    K = econ.n_arms
    assert K == N_ARMS == 4

    # ---- synthetic inputs, built inline (this module imports no sibling PRISM models) --
    # Four latent responder segments, so the heuristic segment read-out is exercised
    # against something with real structure rather than pure noise.
    true_seg = rng.choice(len(SEGMENTS), size=n_cust, p=SEGMENT_SHARES)
    # persuadable / sure_thing / lost_cause / sleeping_dog
    risk_a = np.array([3.0, 1.5, 5.0, 1.8])[true_seg]
    risk_b = np.array([5.0, 9.0, 4.0, 8.0])[true_seg]
    churn_risk = np.clip(rng.beta(risk_a, risk_b), 0.0, 1.0)
    clv = np.round(np.exp(rng.normal(5.6, 0.55, size=n_cust)), 2)

    # Effect scale by segment: the offer saves persuadables, barely moves sure things,
    # does nothing for lost causes, and actively pushes sleeping dogs out of the door.
    effect_scale = np.array([1.0, 0.25, 0.03, -0.60])[true_seg][:, None]
    base = (churn_risk * clv * 0.28)[:, None]
    tilt = np.array([[0.55, 0.95, 1.25]])  # richer offers move people more, and cost more
    value_cate = effect_scale * base * tilt + rng.normal(0.0, 3.5, size=(n_cust, K - 1))
    rmst_cate = value_cate / 14.0 + rng.normal(0.0, 0.15, size=(n_cust, K - 1))

    # ---- 1. breakeven: one quotable number per arm -------------------------------------
    be = breakeven_effect(econ)
    expected_be = np.asarray(econ.arm_costs, dtype=float) * np.asarray(econ.offer_redemption_rate, dtype=float)
    assert be.shape == (K,), be.shape
    assert be[0] == 0.0, "doing nothing breaks even by definition"
    assert np.allclose(be, expected_be), (be, expected_be)
    assert np.allclose(
        breakeven_effect(replace(econ, fixed_campaign_cost=1.5))[1:], expected_be[1:] + 1.5
    ), "the fixed contact cost must raise the breakeven bar"
    print("breakeven_effect -- minimum incremental discounted value an arm must deliver:")
    for a in range(K):
        print(
            f"  arm {a}  {ARM_NAMES[a]:<12s} cost={econ.arm_costs[a]:6.2f}"
            f" x redemption={econ.offer_redemption_rate[a]:.2f}  ->  breakeven={be[a]:7.3f}"
        )

    # ---- 2. expected_net_value ----------------------------------------------------------
    net = expected_net_value(value_cate, econ)
    assert net.shape == (n_cust, K), net.shape
    assert net.dtype == np.float64
    assert np.all(net[:, 0] == 0.0), "column 0 (do nothing) must be EXACTLY zero"
    assert np.allclose(net[:, 1:], value_cate - expected_be[1:])

    pricey = replace(econ, arm_costs=(0.0, 24.0, 60.0, 110.0))
    net_pricey = expected_net_value(value_cate, pricey)
    assert np.all(net_pricey[:, 1:] < net[:, 1:]), "raising costs did not reduce net value"
    assert np.all(net_pricey[:, 0] == 0.0), "the do-nothing arm must stay free"

    net_fixed = expected_net_value(value_cate, replace(econ, fixed_campaign_cost=2.5))
    assert np.allclose(net_fixed[:, 1:], net[:, 1:] - 2.5), "fixed cost must hit every treated arm"
    assert np.all(net_fixed[:, 0] == 0.0), "fixed cost must never hit the control arm"

    net_never = expected_net_value(value_cate, econ, redemption_override=0.0)
    assert np.allclose(net_never[:, 1:], value_cate), "zero redemption should mean zero offer cost"
    net_always = expected_net_value(value_cate, econ, redemption_override=1.0)
    assert np.allclose(net_always[:, 1:], value_cate - np.asarray(econ.arm_costs)[1:]), "full face value"
    assert np.all(net_always[:, 1:] <= net[:, 1:]), "ignoring redemption must look strictly worse"

    per_row_cost = np.tile(np.asarray(econ.arm_costs, dtype=float), (n_cust, 1)) * 2.0
    net_costly = expected_net_value(value_cate, econ, cost_matrix=per_row_cost)
    assert np.allclose(net_costly[:, 1:], value_cate - 2.0 * expected_be[1:])

    two_arm = replace(econ, arm_costs=(0.0, 12.0, 0.0, 0.0), offer_redemption_rate=(0.0, 0.62, 0.0, 0.0))
    one_row = expected_net_value(np.array([10.0, 20.0, 30.0]), two_arm)
    assert one_row.shape == (1, K), one_row.shape
    try:
        expected_net_value(np.zeros((n_cust, 2)), econ)
    except ValueError as exc:
        assert "one column per non-control arm" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a wrong-width value_cate must raise ValueError")

    for bad in (
        dict(arm_costs=(1.0, 12.0, 30.0, 55.0)),
        dict(offer_redemption_rate=(0.0, 1.4, 0.7, 0.5)),
        dict(margin_rate=0.0),
        dict(margin_rate=1.5),
        dict(discount_rate=-0.01),
        dict(horizon=0),
        dict(arm_costs=(0.0, 12.0)),
    ):
        try:
            EconomicConfig(**bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"EconomicConfig accepted invalid parameters: {bad}")

    # ---- 3. decision frame ---------------------------------------------------------------
    customer_id = [f"C{i:06d}" for i in range(n_cust)]
    frame = build_decision_frame(customer_id, value_cate, rmst_cate, churn_risk, clv, econ)
    assert len(frame) == n_cust, "decision frame must have exactly one row per customer"
    assert frame["customer_id"].nunique() == n_cust
    assert frame["recommended_arm"].between(0, K - 1).all(), "recommended_arm out of range"
    assert frame["recommended_arm"].dtype == np.int64
    assert (frame["expected_net_value"] >= 0.0).all(), "expected_net_value must be floored at 0"
    assert np.allclose(frame["expected_net_value"].to_numpy(), np.maximum(net.max(axis=1), 0.0))
    assert np.all(frame["net_value_0"].to_numpy() == 0.0)
    assert np.all(frame["cost_0"].to_numpy() == 0.0)
    assert np.all(frame["cate_value_0"].to_numpy() == 0.0)
    # ties go to arm 0: a customer with no profitable arm is left alone
    assert (frame.loc[net[:, 1:].max(axis=1) <= 0.0, "recommended_arm"] == 0).all()
    assert (frame.loc[frame["recommended_arm"] == 0, "expected_net_value"] == 0.0).all()
    assert int(frame["uplift_rank"].min()) == 1
    assert int(frame["uplift_rank"].max()) <= n_cust
    assert np.isclose(
        frame.loc[frame["uplift_rank"] == 1, "expected_net_value"].max(),
        frame["expected_net_value"].max(),
    ), "rank 1 must be the best customer"
    assert set(frame["segment_guess"].unique()) <= set(SEGMENTS), frame["segment_guess"].unique()
    assert frame["segment_guess"].notna().all()
    # The heuristic is not ground truth, but it must carry signal in the right direction.
    guess = frame["segment_guess"].to_numpy()
    is_persuadable, is_sleeping_dog = true_seg == 0, true_seg == 3
    assert (guess[is_persuadable] == "persuadable").mean() > (guess[is_sleeping_dog] == "persuadable").mean()
    assert (guess[is_sleeping_dog] == "sleeping_dog").mean() > (guess[is_persuadable] == "sleeping_dog").mean()
    assert frame["recommended_arm_name"].tolist() == [ARM_NAMES[a] for a in frame["recommended_arm"]]
    expected_cols = (
        ["customer_id", "churn_risk", "clv"]
        + [f"{p}_{a}" for a in range(K) for p in ("cate_value", "cate_rmst", "cost", "net_value")]
        + ["recommended_arm", "recommended_arm_name", "expected_net_value", "uplift_rank", "segment_guess"]
    )
    assert list(frame.columns) == expected_cols, list(frame.columns)
    pd.testing.assert_frame_equal(
        frame, build_decision_frame(customer_id, value_cate, rmst_cate, churn_risk, clv, econ)
    )

    # ---- 4. roi_summary ------------------------------------------------------------------
    assignment = frame["recommended_arm"].to_numpy()
    cost_mat = expected_cost_matrix(n_cust, econ)
    roi = roi_summary(assignment, net, cost_mat, econ, rmst_cate=rmst_cate)
    assert roi["n_targeted"] == int((assignment > 0).sum())
    assert abs(roi["net_incremental_value"] - (roi["gross_incremental_value"] - roi["total_cost"])) < 1e-6, roi
    assert roi["total_cost"] > 0.0 and roi["net_incremental_value"] > 0.0
    assert np.isclose(roi["roi"], roi["net_incremental_value"] / roi["total_cost"])
    assert np.isclose(roi["value_per_currency_spent"], roi["gross_incremental_value"] / roi["total_cost"])
    assert "cost_per_retained_month" in roi and roi["cost_per_retained_month"] > 0.0

    idle = roi_summary(np.zeros(n_cust, dtype=int), net, cost_mat, econ)
    assert idle["n_targeted"] == 0 and idle["total_cost"] == 0.0 and idle["net_incremental_value"] == 0.0
    assert np.isnan(idle["roi"]) and "cost_per_retained_month" not in idle

    for a in range(1, K):
        blanket = roi_summary(np.full(n_cust, a), net, cost_mat, econ)
        assert roi["net_incremental_value"] > blanket["net_incremental_value"], (
            f"selective targeting must beat treating everyone with arm {a}"
        )

    # ---- 5. is the unconstrained argmax actually optimal? (brute force, not faith) ----
    # The whole decision layer rests on "pick the argmax net value per customer". That is
    # only the global optimum because the unconstrained problem separates across
    # customers -- which is an argument, not a proof. So enumerate every one of the
    # 4**9 = 262,144 possible assignments of a small instance and check the module lands
    # exactly on the best of them. Budget constraints break the separability and move the
    # problem to prism.decision.optimize; this test is the unconstrained baseline that
    # allocator must degrade gracefully towards.
    n_bf = 9
    tau_bf = rng.normal(8.0, 22.0, size=(n_bf, K - 1))
    net_bf = expected_net_value(tau_bf, econ)
    digits = (np.arange(K**n_bf)[:, None] // (K ** np.arange(n_bf)[None, :])) % K
    brute_best = float(net_bf[np.arange(n_bf)[None, :], digits].sum(axis=1).max())
    frame_bf = build_decision_frame(
        [f"B{i}" for i in range(n_bf)], tau_bf, tau_bf / 14.0,
        rng.random(n_bf), rng.random(n_bf) * 500.0, econ,
    )
    policy_bf = float(net_bf[np.arange(n_bf), frame_bf["recommended_arm"].to_numpy()].sum())
    assert abs(brute_best - policy_bf) < 1e-12, (brute_best, policy_bf)
    assert abs(float(frame_bf["expected_net_value"].sum()) - brute_best) < 1e-12
    del digits

    # Sections 6 and 7 feed deliberately broken inputs, so they SHOULD log. Capture
    # those records instead of letting them onto the report: the report must stay
    # byte-identical run to run (a log line carries a wall-clock timestamp), and the
    # captured records are then asserted on directly, which is a stronger test than
    # eyeballing stderr. SPEC_PERF section 8: silent truncation is the unforgivable sin,
    # so "it was logged" is part of the contract, not decoration.
    class _CaptureHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    _capture = _CaptureHandler()
    _saved_handlers, _saved_propagate = logger.handlers[:], logger.propagate
    logger.handlers, logger.propagate = [_capture], False
    try:
        # ---- 6. no RuntimeWarning may reach a user (SPEC_PERF section 9) ---------------
        # Missing estimates are the normal case in production, not an exotic one: a customer
        # can be unscoreable on every arm. That must degrade to a logged fallback, never to a
        # bare "All-NaN slice encountered" or "invalid value encountered in cast" on stderr.
        holed_tau = value_cate[:400].copy()
        holed_tau[0, :] = np.nan          # no usable estimate on any arm
        holed_tau[1, 2] = np.nan          # one arm missing
        holed_risk = churn_risk[:400].copy()
        holed_risk[:5] = np.nan           # unscored churn risk
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            holed_frame = build_decision_frame(
                customer_id[:400], holed_tau, rmst_cate[:400], holed_risk, clv[:400], econ
            )
            holed_net = expected_net_value(holed_tau, econ)
            _ = roi_summary(holed_frame["recommended_arm"].to_numpy(), holed_net, None, econ)
        assert int(holed_frame.loc[0, "recommended_arm"]) == 0, "an all-NaN customer must be left alone"
        assert float(holed_frame.loc[0, "expected_net_value"]) == 0.0
        assert int(holed_frame.loc[1, "recommended_arm"]) != 3, "a NaN arm must never win the argmax"
        assert holed_frame["segment_guess"].notna().all()
        assert set(holed_frame["segment_guess"]) <= set(SEGMENTS)
        # An unscored churn risk must land in the middle tercile. Ranking it alongside the
        # observed values would silently assert the customer is the safest in the book, which
        # is how a missing feature turns into a "sure_thing" on a slide.
        assert set(holed_frame["segment_guess"].to_numpy()[:5]) <= {"persuadable", "sleeping_dog"}, (
            "missing churn_risk leaked into an edge tercile"
        )

        # ---- 7. a P&L built on incomplete estimates must not read as full coverage ---------
        n_holes = 400
        net_holes = net[:n_holes].copy()
        net_holes[0, 1] = np.nan
        cost_holes = expected_cost_matrix(n_holes, econ)
        all_arm1 = np.ones(n_holes, dtype=np.int64)
        roi_holes = roi_summary(all_arm1, net_holes, cost_holes, econ)
        roi_clean = roi_summary(all_arm1[1:], net_holes[1:], cost_holes[1:], econ)
        assert roi_holes["n_targeted"] == n_holes, "an excluded customer was still targeted"
        assert np.isfinite(roi_holes["total_cost"]) and np.isfinite(roi_holes["net_incremental_value"])
        assert np.isclose(roi_holes["net_incremental_value"], roi_clean["net_incremental_value"])
        assert np.isclose(roi_holes["total_cost"], roi_clean["total_cost"]), (
            "an unbookable customer must not be charged either"
        )
        assert abs(
            roi_holes["net_incremental_value"]
            - (roi_holes["gross_incremental_value"] - roi_holes["total_cost"])
        ) < 1e-6

        # An empty campaign is a legitimate degenerate case (an empty fold, an empty
        # protected group) and must return NaN ratios, not divide by zero or raise.
        empty = roi_summary(np.zeros(0, dtype=np.int64), np.zeros((0, K)), None, econ)
        assert empty["n_targeted"] == 0 and empty["total_cost"] == 0.0
        assert np.isnan(empty["roi"]) and np.isnan(empty["value_per_currency_spent"])

        # An arm index is a label, not a quantity: fractional or non-finite indices are a
        # caller bug and must be refused rather than floored to a neighbouring arm.
        for bad_assign in (np.array([1.9, 0.0]), np.array([np.nan, 1.0]), np.array([0, K])):
            try:
                roi_summary(bad_assign, net[:2], None, econ)
            except ValueError:
                pass
            else:  # pragma: no cover
                raise AssertionError(f"roi_summary accepted a bad assignment: {bad_assign}")

    finally:
        logger.handlers, logger.propagate = _saved_handlers, _saved_propagate
    assert any("no finite value_cate" in m for m in _capture.messages), (
        "an unscoreable customer must be logged, not silently bucketed"
    )
    assert any("non-finite net value" in m for m in _capture.messages), (
        "customers dropped from the P&L must be logged with their count"
    )

    # ---- 8. config derived quantities ---------------------------------------------------
    dfac = econ.discount_factors()
    assert dfac.shape == (econ.horizon,) and dfac[0] == 1.0
    assert np.all(np.diff(dfac) < 0.0), "discount factors must strictly decrease"
    assert np.isclose(dfac[1], 1.0 / (1.0 + econ.discount_rate))
    assert np.allclose(EconomicConfig(**econ.to_dict()).expected_arm_cost, econ.expected_arm_cost)

    # Policy value must be monotone non-decreasing as offers get cheaper -- the discrete
    # analogue of a well-formed efficient frontier, and the first thing to break if the
    # control column ever stops being pinned to exactly zero.
    policy_curve = []
    for mult in (3.0, 2.0, 1.5, 1.0, 0.5, 0.25):
        scaled = replace(econ, arm_costs=tuple(c * mult for c in econ.arm_costs))
        scaled_net = expected_net_value(value_cate, scaled)
        assert np.all(scaled_net[:, 0] == 0.0)
        policy_curve.append(float(np.maximum(scaled_net.max(axis=1), 0.0).sum()))
    assert all(b >= a for a, b in zip(policy_curve, policy_curve[1:])), policy_curve

    # ---- report ----------------------------------------------------------------------------
    print("\ndecision frame head:")
    head_cols = [
        "customer_id", "churn_risk", "clv", "net_value_1", "net_value_2", "net_value_3",
        "recommended_arm", "recommended_arm_name", "expected_net_value", "uplift_rank", "segment_guess",
    ]
    print(frame[head_cols].head(5).round(3).to_string(index=False))
    print("\ntop 5 customers by expected net value:")
    print(frame.nsmallest(5, "uplift_rank")[head_cols].round(3).to_string(index=False))
    print("\nsegment_guess vs the latent segment (heuristic read-out, NOT ground truth):")
    print(
        pd.crosstab(
            pd.Series([SEGMENTS[s] for s in true_seg], name="true_segment"),
            pd.Series(guess, name="segment_guess"),
        ).to_string()
    )
    print("\nrecommended arm mix (unconstrained, no budget):")
    print(frame["recommended_arm_name"].value_counts().to_string())
    print("\ncampaign P&L of the unconstrained policy:")
    for key, val in roi.items():
        print(f"  {key:26s} {val:,.3f}" if isinstance(val, float) else f"  {key:26s} {val:,}")
    print("\nadversarial checks:")
    print(f"  brute force over {K**n_bf:,} assignments (n={n_bf}, arms={K}):"
          f" optimum={brute_best:.6f}, module argmax={policy_bf:.6f}")
    print("  all-NaN customer left alone, NaN arm never wins, missing risk -> middle tercile")
    print(f"  unbookable targeted rows excluded from the P&L and logged"
          f" ({len(_capture.messages)} warnings captured), empty campaign -> NaN roi")
    print(f"  policy value monotone as offers get cheaper: "
          f"{' <= '.join(f'{v:,.0f}' for v in policy_curve)}")
    print(f"\neconomics.py OK  ({time.perf_counter() - t_start:.2f}s, n={n_cust}, arms={K})")
