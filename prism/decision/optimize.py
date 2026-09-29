"""Budget-constrained multi-arm offer allocation: the multiple-choice knapsack core.

PRISM's causal layer answers *what would happen* if customer ``i`` were given offer ``a``.
This module answers the only question a retention team can actually act on: **given a fixed
budget, who gets which offer?**

Formally this is a **multiple-choice knapsack problem** (MCKP). With ``n`` customers and
``K`` arms (arm ``0`` is always "do nothing"):

.. math::

    \\max_{x} \\sum_i \\sum_a v_{ia} x_{ia}
    \\quad\\text{s.t.}\\quad
    \\sum_a x_{ia} = 1 \\;\\forall i, \\quad
    \\sum_{i,a} c_{ia} x_{ia} \\le B, \\quad x_{ia} \\in \\{0, 1\\}

Each customer occupies exactly one *class* of the knapsack and contributes exactly one item,
which is what makes this harder (and more realistic) than the binary "treat / do not treat"
targeting problem that most churn projects stop at. MCKP is NP-hard, so this module ships
three solvers with different cost/quality trade-offs and **measures** the gap between them
rather than asserting one is good enough:

==========================  ==================  ================  ==================================
solver                      complexity          optimality        what it is for
==========================  ==================  ================  ==================================
:func:`greedy_knapsack`     O(nK log nK)        no guarantee      fast, explainable sanity baseline
:func:`lagrangian_allocate` O(nK log(1/tol))    certified gap     production; yields the shadow price
:func:`lp_allocate`         HiGHS simplex       optimal LP bound  the reference; prices integrality
==========================  ==================  ================  ==================================

The shadow price is the point
-----------------------------
:func:`lagrangian_allocate` relaxes the budget row into the objective and bisects on the
multiplier ``lambda >= 0``. At the optimum, ``lambda*`` is **the marginal return on the next
unit of retention budget** -- the number that settles a budget argument with a CFO. Read it
two ways, both exact:

* **Decision rule.** Customer ``i`` is given arm ``a`` only if ``v_ia > lambda* * c_ia``: an
  offer must clear a hurdle rate, and ``lambda*`` *is* that hurdle rate.
* **Budget rule.** Spending one more currency unit returns about ``lambda*`` units of
  incremental value. ``lambda* > 1`` means the campaign is under-funded (the next unit of
  budget returns more than it costs); ``lambda* == 0`` means the budget is not binding and
  the remaining money has nowhere profitable to go. A greedy sort cannot produce this number.

Weak duality also hands every solver a **certificate**: for any ``lambda >= 0``,

.. math::  \\mathrm{OPT} \\le D(\\lambda) = \\sum_i \\max_a (v_{ia} - \\lambda c_{ia}) + \\lambda B

so ``diagnostics["optimality_gap"]`` on each :class:`AllocationResult` is a *proved* bound on
how much value the returned integer solution could possibly be leaving behind -- not a hope.

Conventions (shared by every function here)
-------------------------------------------
``net_value`` : ``(n, n_arms)`` float, **column 0 is do-nothing and must be 0.0**. Entries are
    incremental value *already net of expected offer cost* (see
    :func:`prism.decision.economics.expected_net_value`). Negative entries are expected and
    meaningful: they are the sleeping dogs, whom a good policy must refuse to treat.
``cost`` : ``(n, n_arms)`` float, **column 0 is 0.0**. This is the cash that leaves the budget,
    which need not equal the cost already netted out of ``net_value`` (redemption rates,
    accrual versus cash timing).
``budget`` : scalar, a **hard** constraint. Every solver runs an exact :func:`math.fsum`
    reconciliation before returning and repairs any float-level overspend, so
    ``result.total_cost <= budget`` always holds. Overspending by one unit is a bug, not a
    rounding detail.

Inputs are plain numpy arrays: this module deliberately imports nothing from
:mod:`prism.causal` or :mod:`prism.models`, so it can be unit-tested and reused standalone.

Public surface
--------------
:class:`AllocationResult`, :func:`greedy_knapsack`, :func:`lagrangian_allocate`,
:func:`lp_allocate`, :func:`efficient_frontier`, :func:`baseline_policies`, :func:`allocate`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from prism.data.schema import ARM_NAMES
from prism.utils.logging import get_logger
from prism.utils.seeds import as_rng

__all__ = [
    "AllocationResult",
    "greedy_knapsack",
    "lagrangian_allocate",
    "lp_allocate",
    "efficient_frontier",
    "baseline_policies",
    "allocate",
]

LOGGER = get_logger("decision.optimize")

#: Sentinel used to bake infeasible (customer, arm) cells into the arrays without NaN/inf
#: arithmetic. An infeasible arm scores ``-_BIG - lambda * _BIG``, so it can never win an
#: argmax against arm 0, whose score is identically 0.
_BIG: float = 1e18
#: Tolerance for "column 0 is zero" and for treating a budget as exhausted.
_COL0_TOL: float = 1e-9
#: Rows above which :func:`lp_allocate` refuses to build an LP and falls back to the dual.
_LP_MAX_ROWS: int = 200_000


# =====================================================================================
# Internal problem container and input hygiene
# =====================================================================================


@dataclass(frozen=True)
class _Problem:
    """Validated, re-centred view of an allocation problem (internal)."""

    net: np.ndarray
    cost: np.ndarray
    budget: float
    n: int
    n_arms: int
    arm_names: tuple[str, ...]
    rows: np.ndarray
    n_infeasible: int


def _budget_tol(budget: float) -> float:
    """Return the float slack allowed when comparing spend against ``budget``.

    Scaled to the magnitude of the budget and to machine epsilon, so it absorbs summation
    noise (order ``1e-11`` on a 1e5 budget) while staying far below any real currency unit.

    Parameters
    ----------
    budget : float
        The budget being compared against.

    Returns
    -------
    float
        Absolute tolerance.
    """
    return 64.0 * float(np.finfo(np.float64).eps) * max(1.0, abs(float(budget)))


def _arm_names(n_arms: int) -> tuple[str, ...]:
    """Return display names for ``n_arms`` arms, reusing the canonical schema names.

    Parameters
    ----------
    n_arms : int
        Number of arms including the do-nothing arm.

    Returns
    -------
    tuple of str
        Names, padded with ``arm_<k>`` if the caller has more arms than the schema defines.
    """
    if n_arms <= len(ARM_NAMES):
        return tuple(ARM_NAMES[:n_arms])
    extra = tuple(f"arm_{a}" for a in range(len(ARM_NAMES), n_arms))
    return tuple(ARM_NAMES) + extra


def _prepare(net_value: np.ndarray, cost: np.ndarray, budget: float) -> _Problem:
    """Validate and normalise solver inputs into a :class:`_Problem`.

    Performs, in order: shape and dtype checks; non-finite cells are marked infeasible; the
    do-nothing column is re-centred to exactly ``(net=0, cost=0)`` if the caller did not
    already supply it that way; negative costs are clipped to zero. Every repair is logged --
    silent input mangling would make every downstream number untraceable.

    Parameters
    ----------
    net_value : numpy.ndarray
        ``(n, n_arms)`` incremental value per (customer, arm); column 0 = do nothing.
    cost : numpy.ndarray
        ``(n, n_arms)`` budget consumption per (customer, arm); column 0 = 0.
    budget : float
        Hard spend limit.

    Returns
    -------
    _Problem
        Owned copies of the arrays, safe for the solvers to mutate.

    Raises
    ------
    ValueError
        If the arrays are not 2-D, disagree in shape, have no columns, or the budget is
        not finite.
    """
    net = np.array(np.asarray(net_value, dtype=np.float64), copy=True, order="C")
    cst = np.array(np.asarray(cost, dtype=np.float64), copy=True, order="C")
    if net.ndim != 2 or cst.ndim != 2:
        raise ValueError(f"net_value and cost must be 2-D (n, n_arms); got {net.shape} and {cst.shape}")
    if net.shape != cst.shape:
        raise ValueError(f"net_value {net.shape} and cost {cst.shape} must have the same shape")
    n, k = net.shape
    if k < 1:
        raise ValueError("net_value must have at least one column (the do-nothing arm)")
    b = float(budget)
    if not np.isfinite(b):
        raise ValueError(f"budget must be finite; got {budget!r}")
    if b < 0.0:
        LOGGER.warning("negative budget %.6g is not meaningful; treating it as 0", b)
        b = 0.0

    bad = ~(np.isfinite(net) & np.isfinite(cst))
    n_bad = int(bad.sum())
    if n_bad:
        net[bad] = 0.0
        cst[bad] = 0.0
        if bad[:, 0].any():
            LOGGER.warning("the do-nothing column contained %d non-finite cells; forced to (0, 0)",
                           int(bad[:, 0].sum()))
            bad[:, 0] = False
            n_bad = int(bad.sum())
        if n_bad:
            LOGGER.warning("%d of %d (customer, arm) cells are non-finite; those arms are marked "
                           "infeasible and can never be chosen", n_bad, net.size)

    if n:
        off = net[:, 0].copy()
        off_max = float(np.abs(off).max(initial=0.0))
        if off_max > _COL0_TOL:
            LOGGER.warning("net_value[:, 0] is not identically zero (max |v0| = %.6g); re-centring "
                           "every column on the do-nothing arm so the reported value stays "
                           "incremental", off_max)
            net -= off[:, None]
        coff = cst[:, 0].copy()
        coff_max = float(np.abs(coff).max(initial=0.0))
        if coff_max > _COL0_TOL:
            LOGGER.warning("cost[:, 0] is not identically zero (max |c0| = %.6g); re-centring costs "
                           "on the do-nothing arm", coff_max)
            cst -= coff[:, None]
        neg = cst < 0.0
        if neg.any():
            LOGGER.warning("%d cost cells were negative after re-centring; clipped to 0", int(neg.sum()))
            cst[neg] = 0.0
        net[:, 0] = 0.0
        cst[:, 0] = 0.0

    if n_bad:
        net[bad] = -_BIG
        cst[bad] = _BIG

    return _Problem(net=net, cost=cst, budget=b, n=int(n), n_arms=int(k),
                    arm_names=_arm_names(int(k)), rows=np.arange(int(n)), n_infeasible=n_bad)


def _sum_selected(mat: np.ndarray, assignment: np.ndarray, rows: np.ndarray) -> float:
    """Exactly-rounded sum of ``mat[i, assignment[i]]`` via :func:`math.fsum`.

    Pairwise numpy summation is accurate to about ``1e-11`` at these magnitudes, which is fine
    for reporting but not for deciding whether a hard budget was breached. ``fsum`` is
    correctly rounded, so the feasibility check is exact to one ulp.

    Parameters
    ----------
    mat : numpy.ndarray
        ``(n, n_arms)`` matrix to gather from.
    assignment : numpy.ndarray
        ``(n,)`` chosen column per row.
    rows : numpy.ndarray
        ``(n,)`` row index vector (``arange(n)``).

    Returns
    -------
    float
        The gathered sum.
    """
    if rows.size == 0:
        return 0.0
    return float(math.fsum(mat[rows, assignment].tolist()))


def _dual_bound(prob: _Problem, lam: float) -> float:
    """Lagrangian (weak-duality) upper bound on the integer optimum at multiplier ``lam``.

    ``D(lam) = sum_i max_a (v_ia - lam * c_ia) + lam * B >= OPT`` for every ``lam >= 0``,
    because relaxing the budget row can only help. The bound is valid for the *integer*
    problem, not merely its LP relaxation, which is what makes it usable as a shipped
    optimality certificate.

    Parameters
    ----------
    prob : _Problem
        The prepared problem.
    lam : float
        Non-negative multiplier.

    Returns
    -------
    float
        An upper bound on the best achievable ``expected_incremental_value``.
    """
    lam = max(0.0, float(lam))
    if prob.n == 0:
        return 0.0
    score = prob.net - lam * prob.cost
    return float(math.fsum(score.max(axis=1).tolist())) + lam * prob.budget


def _greedy_fill(
    rows: np.ndarray,
    costs: np.ndarray,
    budget: float,
    spent: float = 0.0,
    *,
    n_rows: int = 0,
    one_per_row: bool = True,
) -> tuple[np.ndarray, float]:
    """Walk pre-sorted candidates and take each one that still fits.

    **Skip, do not stop.** When a candidate does not fit, the scan continues instead of
    terminating. Stopping at the first miss would strand budget whenever the head of the list
    is expensive, and a cheap, still-high-ratio customer further down deserves the leftover
    money. The price is that the result depends on the whole ordering rather than a prefix of
    it; the benefit is measured in this module's smoke test.

    An early exit is still available and is *exact*: a running suffix-minimum of candidate
    costs proves that once the remaining budget drops below the cheapest remaining candidate,
    nothing later can fit, so the scan may break with no loss.

    Parameters
    ----------
    rows : numpy.ndarray
        Customer index of each candidate, already in the order they should be considered.
    costs : numpy.ndarray
        Budget consumption of each candidate, aligned to ``rows``.
    budget : float
        Hard spend limit.
    spent : float, default 0.0
        Budget already committed before this pass.
    n_rows : int, default 0
        Total number of customers; required when ``one_per_row`` is True.
    one_per_row : bool, default True
        Enforce the multiple-choice constraint: at most one accepted candidate per customer.

    Returns
    -------
    taken : numpy.ndarray
        Boolean mask over candidates.
    spent : float
        Budget committed after the pass.
    """
    m = int(rows.size)
    taken = np.zeros(m, dtype=bool)
    if m == 0:
        return taken, float(spent)
    suffix_min = np.minimum.accumulate(costs[::-1])[::-1]
    rows_l = rows.tolist()
    costs_l = costs.tolist()
    smin_l = suffix_min.tolist()
    used = bytearray(int(n_rows)) if one_per_row else None
    cap = float(budget) + _budget_tol(budget)
    spent = float(spent)
    for j in range(m):
        i = rows_l[j]
        if used is not None and used[i]:
            continue
        c = costs_l[j]
        if spent + c <= cap:
            taken[j] = True
            spent += c
            if used is not None:
                used[i] = 1
        elif cap - spent < smin_l[j]:
            break
    return taken, spent


def _repair_budget(assignment: np.ndarray, prob: _Problem) -> tuple[np.ndarray, float, int]:
    """Guarantee the hard budget, dropping the least efficient assignments if needed.

    A safety net, not an algorithm: with correct solvers it fires only on float-summation
    noise, and then for at most one customer. Any drop is logged at WARNING, because a solver
    that needs repairing is a solver with a bug.

    Parameters
    ----------
    assignment : numpy.ndarray
        ``(n,)`` chosen arm per customer; modified in place.
    prob : _Problem
        The prepared problem.

    Returns
    -------
    assignment : numpy.ndarray
        The (possibly modified) assignment.
    total_cost : float
        Exactly-rounded spend of the returned assignment.
    n_dropped : int
        How many customers were reverted to arm 0.
    """
    tol = _budget_tol(prob.budget)
    total = _sum_selected(prob.cost, assignment, prob.rows)
    if total <= prob.budget + tol:
        return assignment, total, 0

    LOGGER.warning("allocation overspent by %.6g on a budget of %.6g; repairing by dropping the "
                   "least efficient assignments", total - prob.budget, prob.budget)
    n_dropped = 0
    for _ in range(3):
        treated = np.flatnonzero(assignment > 0)
        if treated.size == 0:
            break
        c = prob.cost[treated, assignment[treated]]
        v = prob.net[treated, assignment[treated]]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(c > 0.0, v / np.where(c > 0.0, c, 1.0), np.inf)
        order = treated[np.lexsort((treated, ratio))]  # worst value-per-unit-cost first
        running = total
        for i in order:
            if running <= prob.budget + tol:
                break
            running -= float(prob.cost[i, assignment[i]])
            assignment[i] = 0
            n_dropped += 1
        total = _sum_selected(prob.cost, assignment, prob.rows)
        if total <= prob.budget + tol:
            break
    if total > prob.budget + tol:  # pragma: no cover - unreachable with the solvers here
        LOGGER.error("budget repair failed: spend %.6g still exceeds budget %.6g", total, prob.budget)
    return assignment, total, n_dropped


def _per_arm_frame(assignment: np.ndarray, prob: _Problem) -> pd.DataFrame:
    """Build the per-arm breakdown of an allocation.

    Parameters
    ----------
    assignment : numpy.ndarray
        ``(n,)`` chosen arm per customer.
    prob : _Problem
        The prepared problem.

    Returns
    -------
    pandas.DataFrame
        Columns ``arm, arm_name, n_assigned, total_cost, total_net_value, mean_net_value``,
        one row per arm including arm 0.
    """
    recs: list[dict[str, Any]] = []
    for a in range(prob.n_arms):
        mask = assignment == a
        k = int(mask.sum())
        if k:
            tc = float(math.fsum(prob.cost[mask, a].tolist()))
            tv = float(math.fsum(prob.net[mask, a].tolist()))
        else:
            tc = tv = 0.0
        recs.append({
            "arm": a,
            "arm_name": prob.arm_names[a],
            "n_assigned": k,
            "total_cost": tc,
            "total_net_value": tv,
            "mean_net_value": (tv / k) if k else 0.0,
        })
    out = pd.DataFrame.from_records(recs)
    return out.astype({"arm": "int64", "n_assigned": "int64"})


def _finalise(
    assignment: np.ndarray,
    prob: _Problem,
    method: str,
    dual_lambda: float | None,
    diagnostics: dict[str, Any] | None = None,
) -> AllocationResult:
    """Repair, score and package an assignment vector into an :class:`AllocationResult`.

    Parameters
    ----------
    assignment : numpy.ndarray
        ``(n,)`` chosen arm per customer.
    prob : _Problem
        The prepared problem.
    method : str
        Solver label.
    dual_lambda : float or None
        Shadow price, when the solver produces one.
    diagnostics : dict, optional
        Solver-specific extras; the weak-duality certificate is added here.

    Returns
    -------
    AllocationResult
    """
    assignment = np.asarray(assignment, dtype=np.int64).reshape(-1)
    assignment, total_cost, n_dropped = _repair_budget(assignment, prob)
    value = _sum_selected(prob.net, assignment, prob.rows)
    diag: dict[str, Any] = dict(diagnostics or {})
    diag.setdefault("n_customers", prob.n)
    diag["n_repaired"] = int(n_dropped)
    if prob.n_infeasible:
        diag["n_infeasible_cells"] = int(prob.n_infeasible)
    lam_for_bound = dual_lambda if dual_lambda is not None else diag.get("cutoff_ratio")
    if lam_for_bound is not None and np.isfinite(float(lam_for_bound)):
        ub = _dual_bound(prob, float(lam_for_bound))
        diag["dual_upper_bound"] = ub
        diag["optimality_gap"] = float(max(0.0, (ub - value) / max(abs(ub), 1.0)))
    return AllocationResult(
        assignment=assignment,
        total_cost=float(total_cost),
        expected_incremental_value=float(value),
        n_treated=int((assignment > 0).sum()),
        per_arm=_per_arm_frame(assignment, prob),
        dual_lambda=(None if dual_lambda is None else float(dual_lambda)),
        method=str(method),
        budget=float(prob.budget),
        diagnostics=diag,
    )


# =====================================================================================
# Result container
# =====================================================================================


@dataclass
class AllocationResult:
    """The output of every solver here: who gets which offer, and what it bought.

    Parameters
    ----------
    assignment : numpy.ndarray
        ``(n,)`` int64 chosen arm per customer; ``0`` means no offer. Exactly one arm per
        customer by construction (the multiple-choice constraint).
    total_cost : float
        Exactly-rounded budget consumed. Guaranteed ``<= budget``.
    expected_incremental_value : float
        Sum of ``net_value[i, assignment[i]]``. With ``net_value`` from
        :func:`prism.decision.economics.expected_net_value` this is incremental profit
        *after* offer cost, so a positive number means the campaign pays for itself.
    n_treated : int
        Customers assigned a non-zero arm.
    per_arm : pandas.DataFrame
        One row per arm: ``arm, arm_name, n_assigned, total_cost, total_net_value,
        mean_net_value``.
    dual_lambda : float or None
        Shadow price of the budget constraint, when the solver produces one
        (:func:`lagrangian_allocate`, :func:`lp_allocate`). ``None`` for
        :func:`greedy_knapsack` and the baselines, which genuinely cannot produce one; see
        ``diagnostics["cutoff_ratio"]`` for the descriptive analogue.

        **Interpretation.** ``lambda*`` is the *marginal return on the next unit of budget*.
        An arm is worth giving only when ``net_value > lambda* * cost``, so ``lambda*`` is the
        hurdle rate every offer must clear; and the next currency unit added to the campaign
        returns approximately ``lambda*`` units of incremental value. ``lambda* == 0`` means
        the budget is not binding: more money would buy nothing.
    method : str
        Solver label, used in comparison tables.
    budget : float, optional
        The budget this allocation was solved against, kept for reporting.
    diagnostics : dict, optional
        Solver-specific extras: ``dual_upper_bound`` and ``optimality_gap`` (a *proved*
        weak-duality bound on the value left on the table), bisection counters, LP status,
        integrality gap, and the number of budget repairs.

    Examples
    --------
    >>> import numpy as np
    >>> net = np.array([[0.0, 5.0], [0.0, -2.0], [0.0, 9.0]])
    >>> cost = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    >>> greedy_knapsack(net, cost, budget=1.0).assignment.tolist()
    [0, 0, 1]
    """

    assignment: np.ndarray
    total_cost: float
    expected_incremental_value: float
    n_treated: int
    per_arm: pd.DataFrame
    dual_lambda: float | None
    method: str
    budget: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce the assignment to int64 and the scalar fields to plain Python numbers."""
        self.assignment = np.asarray(self.assignment, dtype=np.int64).reshape(-1)
        self.total_cost = float(self.total_cost)
        self.expected_incremental_value = float(self.expected_incremental_value)
        self.n_treated = int(self.n_treated)
        if self.dual_lambda is not None:
            self.dual_lambda = float(self.dual_lambda)

    @property
    def n_customers(self) -> int:
        """Number of customers covered by this allocation."""
        return int(self.assignment.size)

    @property
    def roi(self) -> float:
        """Incremental value per unit of budget spent; ``nan`` when nothing was spent."""
        if self.total_cost > 0.0:
            return float(self.expected_incremental_value / self.total_cost)
        return float("nan")

    def summary(self) -> pd.DataFrame:
        """Return a one-row summary frame, ready to concatenate into a comparison table.

        Returns
        -------
        pandas.DataFrame
            One row with ``method, n_customers, n_treated, treat_rate, total_cost, budget,
            budget_utilisation, expected_incremental_value, value_per_treated, roi,
            dual_lambda, optimality_gap``.
        """
        n = self.n_customers
        budget = self.budget
        row = {
            "method": self.method,
            "n_customers": n,
            "n_treated": self.n_treated,
            "treat_rate": (self.n_treated / n) if n else 0.0,
            "total_cost": self.total_cost,
            "budget": (float(budget) if budget is not None else float("nan")),
            "budget_utilisation": (self.total_cost / budget) if budget else float("nan"),
            "expected_incremental_value": self.expected_incremental_value,
            "value_per_treated": (self.expected_incremental_value / self.n_treated) if self.n_treated else 0.0,
            "roi": self.roi,
            "dual_lambda": (float("nan") if self.dual_lambda is None else self.dual_lambda),
            "optimality_gap": float(self.diagnostics.get("optimality_gap", float("nan"))),
        }
        return pd.DataFrame([row])


# =====================================================================================
# 1. Greedy
# =====================================================================================


def greedy_knapsack(
    net_value: np.ndarray,
    cost: np.ndarray,
    budget: float,
    *,
    candidates_per_customer: int = 1,
) -> AllocationResult:
    """Rank offers by value per unit of cost and buy down the list until the money runs out.

    For each customer the *single best-ratio* arm is nominated:
    ``ratio = (net_value[i, a] - net_value[i, 0]) / cost[i, a]`` over arms with strictly
    positive net value (the do-nothing column is 0 by convention, so this reduces to
    ``net / cost``). All nominations are sorted by ratio descending and bought in order.

    **Non-fitting candidates are skipped, not stopped at.** Terminating at the first
    unaffordable item would strand budget whenever the head of the list is expensive; a cheap,
    still-high-ratio customer further down deserves the leftover money. The scan does exit
    early, but only when a suffix-minimum over the remaining costs proves nothing left can
    fit, so the early exit costs nothing.

    This is the explainable baseline, not the recommended solver: it carries no optimality
    guarantee and -- by design -- **no shadow price**. A sorted list cannot tell you what the
    next unit of budget is worth; only the dual can (:func:`lagrangian_allocate`). The
    descriptive analogue, the ratio of the last accepted offer, is reported as
    ``diagnostics["cutoff_ratio"]``. That ratio is also fed to the weak-duality bound, so
    greedy's ``optimality_gap`` is still a *valid* certificate but a deliberately loose one:
    the bound is evaluated at the cutoff ratio rather than at the optimal multiplier, which
    only the dual solver finds.

    Parameters
    ----------
    net_value : numpy.ndarray
        ``(n, n_arms)`` incremental value net of offer cost; column 0 (do nothing) must be 0.
    cost : numpy.ndarray
        ``(n, n_arms)`` budget consumption; column 0 must be 0.
    budget : float
        Hard spend limit. The returned ``total_cost`` never exceeds it.
    candidates_per_customer : int, default 1
        How many arms each customer may nominate. ``1`` is the contract behaviour described
        above. Raising it to ``n_arms - 1`` lets a customer's cheaper arm be reconsidered once
        the expensive one has been passed over -- a strictly better greedy at the cost of a
        longer candidate list. The multiple-choice constraint is enforced either way.

    Returns
    -------
    AllocationResult
        With ``method="greedy"`` and ``dual_lambda=None``.

    Notes
    -----
    Complexity is ``O(n K + m log m)`` for ``m <= n * candidates_per_customer`` candidates.
    """
    t0 = time.perf_counter()
    prob = _prepare(net_value, cost, budget)
    n, k = prob.n, prob.n_arms
    assignment = np.zeros(n, dtype=np.int64)
    if n == 0 or k < 2:
        return _finalise(assignment, prob, "greedy", None,
                         {"cutoff_ratio": float("nan"), "n_candidates": 0,
                          "seconds": time.perf_counter() - t0})

    offers = prob.net[:, 1:]
    ocost = prob.cost[:, 1:]
    positive = offers > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_cost = np.where(ocost > 0.0, ocost, 1.0)
        ratio = np.where(ocost > 0.0, offers / safe_cost, np.inf)
    ratio = np.where(positive, ratio, -np.inf)

    n_cand = int(np.clip(candidates_per_customer, 1, max(1, k - 1)))
    if n_cand == 1:
        arm_idx = np.argmax(ratio, axis=1)
        keep = ratio[prob.rows, arm_idx] > 0.0
        cand_rows = prob.rows[keep]
        cand_arms = (arm_idx[keep] + 1).astype(np.int64)
        cand_ratio = ratio[cand_rows, cand_arms - 1]
    else:
        take = np.argsort(-ratio, axis=1, kind="stable")[:, :n_cand]
        rr = np.repeat(prob.rows, n_cand)
        aa = take.reshape(-1)
        vals = ratio[rr, aa]
        keep = vals > 0.0
        cand_rows = rr[keep]
        cand_arms = (aa[keep] + 1).astype(np.int64)
        cand_ratio = vals[keep]

    if cand_rows.size == 0:
        LOGGER.info("greedy: no arm has positive net value; the do-nothing policy is optimal")
        return _finalise(assignment, prob, "greedy", None,
                         {"cutoff_ratio": float("nan"), "n_candidates": 0,
                          "seconds": time.perf_counter() - t0})

    order = np.lexsort((cand_arms, cand_rows, -cand_ratio))  # ratio desc, deterministic ties
    cand_rows = cand_rows[order]
    cand_arms = cand_arms[order]
    cand_ratio = cand_ratio[order]
    cand_cost = prob.cost[cand_rows, cand_arms]

    taken, _ = _greedy_fill(cand_rows, cand_cost, prob.budget, 0.0, n_rows=n, one_per_row=True)
    assignment[cand_rows[taken]] = cand_arms[taken]

    n_taken = int(taken.sum())
    # The cutoff is the ratio of the last *priced* offer bought. A zero-cost offer has an
    # infinite ratio and must not become the multiplier fed to the weak-duality bound.
    taken_ratio = cand_ratio[taken]
    finite_ratio = taken_ratio[np.isfinite(taken_ratio)]
    diag = {
        "cutoff_ratio": float(finite_ratio[-1]) if finite_ratio.size else float("nan"),
        "n_candidates": int(cand_rows.size),
        "n_skipped_unaffordable": int(cand_rows.size - n_taken),
        "candidates_per_customer": n_cand,
        "seconds": time.perf_counter() - t0,
    }
    return _finalise(assignment, prob, "greedy", None, diag)


# =====================================================================================
# 2. Lagrangian dual -- the production solver
# =====================================================================================


def _assign_at_lambda(prob: _Problem, lam: float, buf: np.ndarray) -> np.ndarray:
    """Per-customer argmax of ``net - lam * cost`` (the Lagrangian inner problem).

    With the budget row relaxed into the objective the customers decouple completely, so the
    inner problem is a single vectorised argmax. ``np.argmax`` returns the first maximiser and
    arm 0 always scores exactly 0, so ties resolve to *not treating* -- the conservative and
    economically correct default.

    Parameters
    ----------
    prob : _Problem
        The prepared problem.
    lam : float
        Non-negative multiplier (the price of one unit of budget).
    buf : numpy.ndarray
        ``(n, n_arms)`` scratch buffer, reused across bisection steps to avoid reallocating.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` int64 arm choice.
    """
    np.multiply(prob.cost, lam, out=buf)
    np.subtract(prob.net, buf, out=buf)
    return np.argmax(buf, axis=1).astype(np.int64, copy=False)


def _topup_capacitated(
    assignment: np.ndarray,
    prob: _Problem,
    spent: float,
    gain: np.ndarray,
    extra: np.ndarray,
    cap_left: np.ndarray,
    stats: dict[str, float],
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Residual-budget top-up that also honours per-arm capacity.

    Runs as an explicit loop rather than the vectorised fill of :func:`_greedy_fill`, because
    accepting an upgrade both *consumes* a slot on the target arm and *releases* one on the
    source arm, so the feasible set changes after every acceptance and cannot be decided from
    a precomputed mask. Only reached from :func:`lp_allocate` with ``arm_capacity``, where the
    problem is small by construction.

    Parameters
    ----------
    assignment : numpy.ndarray
        ``(n,)`` current arm per customer; modified in place.
    prob : _Problem
        The prepared problem.
    spent : float
        Budget already committed.
    gain, extra : numpy.ndarray
        ``(n, n_arms)`` value gain and extra cost of moving customer ``i`` to arm ``a``.
    cap_left : numpy.ndarray
        ``(n_arms,)`` remaining headroom per arm; updated in place.
    stats : dict
        Accumulator, mutated and returned.

    Returns
    -------
    assignment, spent, stats
    """
    n = prob.n
    eligible = gain > 0.0
    if not eligible.any():
        return assignment, spent, stats
    rr, aa = np.nonzero(eligible)
    ex = extra[rr, aa]
    gn = gain[rr, aa]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ex > 0.0, gn / np.where(ex > 0.0, ex, 1.0), np.inf)
    order = np.lexsort((aa, rr, -ratio))  # free upgrades first, then best gain per unit cost
    rr_l = rr[order].tolist()
    aa_l = aa[order].tolist()
    ex_l = ex[order].tolist()
    gn_l = gn[order].tolist()
    cap = np.asarray(cap_left, dtype=np.float64)
    used = bytearray(n)
    tol = _budget_tol(prob.budget)
    for j in range(len(rr_l)):
        i = rr_l[j]
        if used[i]:
            continue
        a = aa_l[j]
        if cap[a] <= 0.0:
            continue
        c = ex_l[j]
        if spent + c > prob.budget + tol:
            continue
        src = int(assignment[i])
        assignment[i] = a
        spent += c
        cap[a] -= 1.0
        cap[src] += 1.0
        used[i] = 1
        stats["topup_value_gain"] += gn_l[j]
        stats["topup_cost"] += max(c, 0.0)
        stats["n_upgraded"] += 1.0
        if c <= 0.0:
            stats["n_free_upgrades"] += 1.0
    cap_left[:] = cap
    return assignment, spent, stats


def _topup(
    assignment: np.ndarray,
    prob: _Problem,
    spent: float,
    *,
    cap_left: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Spend the budget a solver left unspent, by upgrading customers one step.

    The dual solution is feasible but usually leaves a sliver of budget unspent, because the
    multiplier moves the whole population at once and cannot buy a fraction of a customer.
    This pass enumerates every *upgrade* ``(i, current -> a)`` with a positive value gain,
    ranks them by gain per extra unit of cost, and applies them greedily -- at most one per
    customer, so the multiple-choice constraint still holds.

    Parameters
    ----------
    assignment : numpy.ndarray
        ``(n,)`` current arm per customer; modified in place.
    prob : _Problem
        The prepared problem.
    spent : float
        Budget already committed by the incoming solution.
    cap_left : numpy.ndarray, optional
        ``(n_arms,)`` remaining per-arm headroom. When given, the capacity-aware loop in
        :func:`_topup_capacitated` is used instead of the vectorised fill and ``cap_left`` is
        updated in place.

    Returns
    -------
    assignment : numpy.ndarray
        The upgraded assignment.
    spent : float
        Budget committed afterwards.
    stats : dict
        ``topup_value_gain``, ``topup_cost``, ``n_upgraded``, ``n_free_upgrades``.

    Notes
    -----
    On return no single customer can be moved to a different arm for a strictly higher value
    without breaking the budget (or a capacity): the result is a **local optimum under
    one-customer moves**. That is what makes this pass worth running after *any* solver, not
    just the dual -- LP rounding strands budget in exactly the same way.
    """
    n, k = prob.n, prob.n_arms
    stats = {"topup_value_gain": 0.0, "topup_cost": 0.0, "n_upgraded": 0.0, "n_free_upgrades": 0.0}
    if n == 0 or k < 2:
        return assignment, spent, stats
    cur_net = prob.net[prob.rows, assignment]
    cur_cost = prob.cost[prob.rows, assignment]
    gain = prob.net - cur_net[:, None]
    extra = prob.cost - cur_cost[:, None]

    if cap_left is not None:
        return _topup_capacitated(assignment, prob, spent, gain, extra, cap_left, stats)

    free = (gain > 0.0) & (extra <= 0.0)
    if free.any():  # impossible at the dual optimum, but never leave free money behind
        best_free = np.argmax(np.where(free, gain, -np.inf), axis=1)
        rows_free = np.flatnonzero(free[prob.rows, best_free])
        if rows_free.size:
            LOGGER.info("top-up: %d strictly dominating upgrades were free", int(rows_free.size))
            stats["n_free_upgrades"] = float(rows_free.size)
            stats["topup_value_gain"] += float(gain[rows_free, best_free[rows_free]].sum())
            spent += float(extra[rows_free, best_free[rows_free]].sum())
            assignment[rows_free] = best_free[rows_free]
            cur_net = prob.net[prob.rows, assignment]
            cur_cost = prob.cost[prob.rows, assignment]
            gain = prob.net - cur_net[:, None]
            extra = prob.cost - cur_cost[:, None]

    eligible = (gain > 0.0) & (extra > 0.0)
    if not eligible.any():
        return assignment, spent, stats
    rr, aa = np.nonzero(eligible)
    ratio = gain[rr, aa] / extra[rr, aa]
    order = np.lexsort((aa, rr, -ratio))
    rr, aa = rr[order], aa[order]
    ex = extra[rr, aa]
    taken, spent = _greedy_fill(rr, ex, prob.budget, spent, n_rows=n, one_per_row=True)
    if taken.any():
        sel_r, sel_a = rr[taken], aa[taken]
        stats["topup_value_gain"] += float(gain[sel_r, sel_a].sum())
        stats["topup_cost"] += float(ex[taken].sum())
        stats["n_upgraded"] = float(taken.sum())
        assignment[sel_r] = sel_a.astype(np.int64)
    return assignment, spent, stats


def lagrangian_allocate(
    net_value: np.ndarray,
    cost: np.ndarray,
    budget: float,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
    topup: bool = True,
) -> AllocationResult:
    """Solve the multiple-choice knapsack by bisecting on the budget's shadow price.

    Relax the budget row into the objective with a multiplier ``lambda >= 0``. The customers
    then decouple: each independently takes ``argmax_a (net[i, a] - lambda * cost[i, a])``.
    Total spend is **non-increasing in lambda** (raising the price of budget can only make
    offers less attractive), so the smallest feasible multiplier is found by bisection on
    ``[0, lambda_max]``, where ``lambda_max`` is the largest single-offer ratio ``net / cost``
    -- above it nothing is worth buying and spend is exactly zero, which gives a guaranteed
    feasible upper bracket.

    Three edge cases are handled explicitly rather than by luck:

    * **budget <= 0** -- nothing affordable is bought, so the assignment is all zeros. (An arm
      with strictly positive value that costs exactly nothing is still taken: it consumes no
      budget, and refusing free money would be wrong.) ``dual_lambda`` is then the best
      available ratio, which is precisely the value of the *first* unit of budget.
    * **budget above the unconstrained optimum** -- the budget is not binding, so
      ``lambda* == 0`` and every arm with positive net value is given. This is the case where
      the honest answer to "should we spend more?" is *no*.
    * **all net values negative** -- no offer clears any hurdle, do-nothing wins, spend is 0
      and ``lambda* == 0``. Sleeping dogs are refused automatically, with no special-casing,
      because their net value is negative.

    After bisection, a greedy top-up spends the residual budget on the best remaining
    single-customer upgrades.

    Parameters
    ----------
    net_value : numpy.ndarray
        ``(n, n_arms)`` incremental value net of offer cost; column 0 must be 0.
    cost : numpy.ndarray
        ``(n, n_arms)`` budget consumption; column 0 must be 0.
    budget : float
        Hard spend limit.
    tol : float, default 1e-6
        Relative width of the lambda bracket at which bisection stops.
    max_iter : int, default 100
        Maximum bisection steps. 100 steps shrink the bracket by ``2 ** -100``, so ``tol``
        almost always binds first.
    topup : bool, default True
        Run the residual-budget greedy top-up. Disable it to inspect the pure dual solution.

    Returns
    -------
    AllocationResult
        ``method="lagrangian"``, ``dual_lambda=lambda*``, and ``diagnostics`` carrying
        ``dual_upper_bound`` / ``optimality_gap`` (a proved weak-duality certificate),
        ``budget_binding``, the final bracket and the top-up contribution.

    Notes
    -----
    Complexity is ``O(n K log(1 / tol))`` with no allocation inside the loop: scores are
    written into one preallocated buffer, so a 100-step bisection on 120k x 4 costs a few
    hundred milliseconds.

    ``lambda*`` is the number to quote when someone asks what another 100k of retention
    budget is worth: approximately ``lambda*`` units of incremental value per unit spent,
    until the next kink in the efficient frontier.
    """
    t0 = time.perf_counter()
    prob = _prepare(net_value, cost, budget)
    n, k = prob.n, prob.n_arms
    tol = max(float(tol), 0.0)
    max_iter = max(int(max_iter), 1)
    assignment = np.zeros(n, dtype=np.int64)
    if n == 0 or k < 2:
        return _finalise(assignment, prob, "lagrangian", 0.0,
                         {"budget_binding": False, "n_iter": 0, "seconds": time.perf_counter() - t0})

    offers = prob.net[:, 1:]
    ocost = prob.cost[:, 1:]
    priced = (ocost > 0.0) & (offers > 0.0)
    ratio_max = float(np.max(offers[priced] / ocost[priced])) if priced.any() else 0.0

    # --- edge case: no budget ---------------------------------------------------------
    if prob.budget <= _COL0_TOL:
        freebie = np.where(prob.cost <= 0.0, prob.net, -np.inf)
        freebie[:, 0] = 0.0
        assignment = np.argmax(freebie, axis=1).astype(np.int64)
        n_free = int((assignment > 0).sum())
        if n_free:
            LOGGER.info("budget is zero, but %d arms cost nothing and add value; taking them", n_free)
        diag = {"budget_binding": True, "n_iter": 0, "lambda_lo": ratio_max, "lambda_hi": ratio_max,
                "spend_before_topup": 0.0, "seconds": time.perf_counter() - t0}
        return _finalise(assignment, prob, "lagrangian", ratio_max, diag)

    buf = np.empty_like(prob.net)

    # --- is the budget binding at all? ------------------------------------------------
    a0 = _assign_at_lambda(prob, 0.0, buf)
    spend0 = float(np.add.reduce(prob.cost[prob.rows, a0]))
    btol = _budget_tol(prob.budget)
    if spend0 <= prob.budget + btol:
        LOGGER.info("budget %.6g is not binding (the unconstrained optimum spends %.6g); lambda* = 0",
                    prob.budget, spend0)
        diag = {"budget_binding": False, "n_iter": 0, "lambda_lo": 0.0, "lambda_hi": 0.0,
                "spend_before_topup": spend0, "unconstrained_spend": spend0,
                "seconds": time.perf_counter() - t0}
        return _finalise(a0, prob, "lagrangian", 0.0, diag)

    # --- bisection: the smallest lambda whose spend fits ------------------------------
    lo, hi = 0.0, ratio_max * (1.0 + 1e-9) + 1e-12
    best = _assign_at_lambda(prob, hi, buf)
    best_spend = float(np.add.reduce(prob.cost[prob.rows, best]))
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007 - read after the loop
        mid = 0.5 * (lo + hi)
        cand = _assign_at_lambda(prob, mid, buf)
        spend = float(np.add.reduce(prob.cost[prob.rows, cand]))
        if spend <= prob.budget + btol:
            hi, best, best_spend = mid, cand, spend
        else:
            lo = mid
        if (hi - lo) <= tol * max(1.0, hi):
            break
    lam_star = float(hi)

    assignment = best
    spend_dual = best_spend
    stats: dict[str, float] = {}
    if topup:
        assignment, _, stats = _topup(assignment, prob, spend_dual)
    diag: dict[str, Any] = {
        "budget_binding": True,
        "n_iter": int(n_iter),
        "lambda_lo": float(lo),
        "lambda_hi": float(hi),
        "lambda_bracket": float(hi - lo),
        "spend_before_topup": float(spend_dual),
        "unconstrained_spend": float(spend0),
        "seconds": time.perf_counter() - t0,
    }
    diag.update(stats)
    return _finalise(assignment, prob, "lagrangian", lam_star, diag)


# =====================================================================================
# 3. LP relaxation + rounding -- the reference solver
# =====================================================================================


def _pareto_keep(prob: _Problem) -> np.ndarray:
    """Boolean ``(n, n_arms)`` mask of the arms that can appear in an optimal solution.

    Within a customer, arm ``a`` is **dominated** if some other arm ``b`` costs no more and is
    worth at least as much; a dominated arm can be swapped out of any solution without losing
    value or breaking the budget, so it can be deleted before the LP is even built. Sorting
    each row by cost and keeping only the arms that strictly beat the running maximum value
    leaves exactly the upper-left staircase of the (cost, value) frontier -- the classical
    MCKP dominance reduction, done once and fully vectorised.

    This is pure presolve: it removes every arm with non-positive net value (dominated by
    do-nothing, which is free) and every infeasible cell. On realistic four-arm data it
    deletes about 40 percent of the LP columns and cuts the HiGHS solve time by a third,
    while provably leaving the optimum unchanged.

    .. warning::
       Dominance is only valid for the *unconstrained-per-arm* problem. Under per-arm
       capacity limits a dominated arm can become necessary once the dominating arm is
       exhausted, so :func:`lp_allocate` skips this reduction whenever ``arm_capacity`` is
       supplied.

    Parameters
    ----------
    prob : _Problem
        The prepared problem.

    Returns
    -------
    numpy.ndarray
        ``(n, n_arms)`` boolean mask; column 0 is always kept.
    """
    net, cost = prob.net, prob.cost
    order = np.argsort(cost, axis=1, kind="stable")
    net_sorted = np.take_along_axis(net, order, axis=1)
    running = np.maximum.accumulate(net_sorted, axis=1)
    prev = np.empty_like(running)
    prev[:, 0] = -np.inf
    if running.shape[1] > 1:
        prev[:, 1:] = running[:, :-1]
    keep_sorted = net_sorted > prev
    keep = np.empty_like(keep_sorted)
    np.put_along_axis(keep, order, keep_sorted, axis=1)
    keep[:, 0] = True
    return keep


def lp_allocate(
    net_value: np.ndarray,
    cost: np.ndarray,
    budget: float,
    *,
    arm_capacity: Sequence[float] | np.ndarray | None = None,
    max_rows: int = _LP_MAX_ROWS,
    prune: bool = True,
) -> AllocationResult:
    """Solve the LP relaxation with HiGHS, round it, and report the integrality gap.

    The relaxation drops the integrality requirement from the multiple-choice knapsack::

        maximise    sum_i sum_a  net[i, a] * x[i, a]
        subject to  sum_a x[i, a] == 1            for every customer i
                    sum_{i,a} cost[i, a] * x[i, a] <= budget
                    x >= 0

    **The constraint matrix is built sparsely and must stay that way.** Dense, the equality
    block alone is ``n x (n * K)``: at ``n = 100_000`` and four arms that is 1.6e11 float64
    cells, roughly 1.3 terabytes, to store what is really just ``n * K`` ones. Both blocks are
    therefore assembled as :class:`scipy.sparse.csr_matrix`. The equality block needs no
    coordinate arrays at all -- each row owns one contiguous run of columns, so it is written
    directly in CSR form as ``indptr = cumsum(arms kept per customer)`` with
    ``indices = arange(nnz)``. Total memory is linear in ``n * K``: about 8 MB at ``n = 120k``.

    **Rounding.** The LP has ``n`` equality rows and one knapsack row, so any basic optimal
    solution has at most ``n + 1`` basic variables and therefore **at most one fractional
    customer** (with ``arm_capacity`` there may be one per extra binding capacity row). Every
    integral customer keeps its arm; each fractional customer is given the most valuable arm
    that the remaining budget -- and any remaining capacity -- can still afford.

    Rounding down strands budget, so the same **residual-budget top-up** that
    :func:`lagrangian_allocate` runs is applied afterwards (capacity-aware here). Without it
    the money reserved for a fractional slice is never re-offered to the customers rounded
    down around it, and the LP can return strictly less value than the far cheaper dual
    solver -- which it did on 9 percent of random small instances before this was added. With
    it, the returned allocation is a local optimum under one-customer moves.

    **The gap is measured, not claimed.** The relaxed objective is a true upper bound on the
    integer optimum, so ``diagnostics["integrality_gap"]`` is the exact relative price of
    rounding. It is typically a few parts per million at these sizes, which is the evidence
    behind the claim that the fast :func:`lagrangian_allocate` is good enough for production.

    **The dual is free.** HiGHS returns the marginal of the budget row, which *is* the shadow
    price ``lambda*``, obtained here without bisection. It should agree with
    :func:`lagrangian_allocate`'s bisected value; comparing the two is a cheap and strong
    cross-check that both solvers are correct.

    Parameters
    ----------
    net_value : numpy.ndarray
        ``(n, n_arms)`` incremental value net of offer cost; column 0 must be 0.
    cost : numpy.ndarray
        ``(n, n_arms)`` budget consumption; column 0 must be 0.
    budget : float
        Hard spend limit.
    arm_capacity : sequence of float, optional
        Maximum number of customers per arm (``inf`` or ``None`` entries mean unlimited), for
        operational limits such as concierge agent hours. Adds one inequality row per finite
        entry and disables the dominance presolve.
    max_rows : int, default 200000
        Above this many customers the LP is not attempted; the call falls back to
        :func:`lagrangian_allocate` and says so in the log and in ``method``.
    prune : bool, default True
        Apply the dominance presolve. Ignored (forced off) when ``arm_capacity`` is given.

    Returns
    -------
    AllocationResult
        ``method="lp"`` with ``dual_lambda`` from the HiGHS budget marginal, or
        ``method="lp->lagrangian_fallback"`` if the LP was skipped or failed.

    Notes
    -----
    Falls back to :func:`lagrangian_allocate` -- always with a WARNING, never silently -- when
    ``n > max_rows``, when SciPy raises, or when HiGHS returns a non-optimal status.

    Measured on this machine with four arms: ``n = 20k`` solves in about 0.8 s (1.3 s without
    the presolve), ``n = 120k`` in about 31 s. The dual solver returns the same allocation at
    ``n = 20k`` in 0.02 s, which is why :func:`allocate` routes anything above a few thousand
    rows to :func:`lagrangian_allocate` and treats the LP as the reference used to *price* that
    choice rather than as the production path.
    """
    t0 = time.perf_counter()
    prob = _prepare(net_value, cost, budget)
    n, k = prob.n, prob.n_arms

    def _fallback(reason: str) -> AllocationResult:
        LOGGER.warning("lp_allocate falling back to the Lagrangian solver: %s", reason)
        res = lagrangian_allocate(net_value, cost, budget)
        diag = dict(res.diagnostics)
        diag["lp_fallback_reason"] = reason
        diag["seconds"] = time.perf_counter() - t0
        return replace(res, method="lp->lagrangian_fallback", diagnostics=diag)

    if n == 0 or k < 2:
        return _finalise(np.zeros(n, dtype=np.int64), prob, "lp", 0.0,
                         {"lp_status": "trivial", "seconds": time.perf_counter() - t0})
    if n > int(max_rows):
        return _fallback(f"n={n} exceeds max_rows={int(max_rows)}")

    try:
        from scipy import sparse
        from scipy.optimize import linprog
    except Exception as exc:  # pragma: no cover - scipy is a hard dependency
        return _fallback(f"scipy unavailable ({exc})")

    cap = None
    if arm_capacity is not None:
        cap = np.asarray(arm_capacity, dtype=np.float64).ravel()
        if cap.size != k:
            raise ValueError(f"arm_capacity must have {k} entries (one per arm); got {cap.size}")
        cap = np.where(np.isnan(cap), np.inf, cap)
        if prune:
            LOGGER.info("arm_capacity supplied; dominance presolve disabled (a dominated arm can "
                        "become necessary once the dominating arm is capped)")
        prune = False

    keep = _pareto_keep(prob) if prune else (prob.cost < _BIG * 0.5)
    keep[:, 0] = True
    rows_idx, arms_idx = np.nonzero(keep)          # row-major, so rows are already grouped
    m = int(rows_idx.size)
    counts = keep.sum(axis=1).astype(np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])

    try:
        a_eq = sparse.csr_matrix((np.ones(m), np.arange(m, dtype=np.int64), indptr), shape=(n, m))
        flat_cost = prob.cost[rows_idx, arms_idx]
        ub_data = [flat_cost]
        ub_rows = [np.zeros(m, dtype=np.int64)]
        ub_cols = [np.arange(m, dtype=np.int64)]
        b_ub = [prob.budget]
        n_ub = 1
        if cap is not None:
            for a in range(1, k):
                if not np.isfinite(cap[a]) or cap[a] >= n:
                    continue
                sel = np.flatnonzero(arms_idx == a)
                ub_data.append(np.ones(sel.size))
                ub_rows.append(np.full(sel.size, n_ub, dtype=np.int64))
                ub_cols.append(sel.astype(np.int64))
                b_ub.append(float(cap[a]))
                n_ub += 1
        a_ub = sparse.csr_matrix(
            (np.concatenate(ub_data), (np.concatenate(ub_rows), np.concatenate(ub_cols))),
            shape=(n_ub, m),
        )
        t_solve = time.perf_counter()
        res = linprog(c=-prob.net[rows_idx, arms_idx], A_ub=a_ub, b_ub=np.asarray(b_ub, dtype=float),
                      A_eq=a_eq, b_eq=np.ones(n), bounds=(0.0, None), method="highs")
        solve_seconds = time.perf_counter() - t_solve
    except MemoryError as exc:
        return _fallback(f"out of memory building or solving the LP ({exc})")
    except Exception as exc:  # pragma: no cover - defensive
        return _fallback(f"scipy.optimize.linprog raised {type(exc).__name__}: {exc}")

    if not res.success or res.x is None:
        return _fallback(f"HiGHS status {res.status}: {res.message}")

    lp_objective = float(-res.fun)
    lam = 0.0
    try:
        lam = float(max(0.0, -np.asarray(res.ineqlin.marginals, dtype=float).ravel()[0]))
    except Exception:  # pragma: no cover - older scipy without marginals
        LOGGER.info("HiGHS did not return duals; falling back to the bisected shadow price")
        lam = float(lagrangian_allocate(net_value, cost, budget).dual_lambda or 0.0)

    # --- rounding ---------------------------------------------------------------------
    x = np.zeros((n, k), dtype=np.float64)
    x[rows_idx, arms_idx] = res.x
    best = np.argmax(x, axis=1)
    integral = x[prob.rows, best] >= 1.0 - 1e-6
    assignment = np.zeros(n, dtype=np.int64)
    assignment[integral] = best[integral].astype(np.int64)
    frac_rows = np.flatnonzero(~integral)

    spent = _sum_selected(prob.cost, assignment, prob.rows)
    tol = _budget_tol(prob.budget)
    cap_left = None
    if cap is not None:
        used = np.bincount(assignment, minlength=k).astype(np.float64)
        cap_left = cap - used
        if np.any(cap_left < -1e-9):
            LOGGER.warning("integral rounding exceeded arm capacity on arms %s; the fractional "
                           "pass will not add to them", np.flatnonzero(cap_left < -1e-9).tolist())

    if frac_rows.size:
        rank = np.argsort(-prob.net[frac_rows].max(axis=1), kind="stable")
        for i in frac_rows[rank].tolist():
            pick, pick_val = 0, 0.0
            for a in range(1, k):
                if not keep[i, a]:
                    continue
                if cap_left is not None and cap_left[a] <= 0.0:
                    continue
                c = float(prob.cost[i, a])
                v = float(prob.net[i, a])
                if v > pick_val and spent + c <= prob.budget + tol:
                    pick, pick_val = a, v
            if pick:
                assignment[i] = pick
                spent += float(prob.cost[i, pick])
                if cap_left is not None:
                    cap_left[pick] -= 1.0

    # --- residual-budget top-up -------------------------------------------------------
    # Rounding strands budget. A fractional customer whose own arms no longer fit is left at
    # arm 0, and the money that was reserved for its fractional slice is never offered to the
    # customers the LP had already rounded down -- so the returned integer solution can be
    # strictly worse than the dual solver's, which does run this pass. Measured on random
    # 8-customer instances before this was added: the LP lost to lagrangian_allocate on 9% of
    # instances and left an affordable, strictly profitable offer unbought on 7% of them.
    # This is the same one-customer-move local search lagrangian_allocate uses, extended to
    # honour arm_capacity, so the LP path is a local optimum too.
    assignment, spent, topup_stats = _topup(assignment, prob, spent, cap_left=cap_left)

    value = _sum_selected(prob.net, assignment, prob.rows)
    diag: dict[str, Any] = {
        "lp_status": str(res.message).split(".")[0],
        "lp_objective": lp_objective,
        "integrality_gap": float(max(0.0, (lp_objective - value) / max(abs(lp_objective), 1.0))),
        "n_fractional": int(frac_rows.size),
        "n_lp_columns": m,
        "n_columns_pruned": int(prob.net.size - m),
        "lp_rows": int(n + n_ub),
        "solve_seconds": float(solve_seconds),
        "seconds": time.perf_counter() - t0,
    }
    diag.update(topup_stats)
    LOGGER.info("lp_allocate: %d columns (%d pruned), HiGHS %.2fs, %d fractional row(s), "
                "integrality gap %.3g, lambda* %.6g", m, diag["n_columns_pruned"], solve_seconds,
                int(frac_rows.size), diag["integrality_gap"], lam)
    return _finalise(assignment, prob, "lp", lam, diag)


# =====================================================================================
# 4. Efficient frontier
# =====================================================================================

_FRONTIER_COLUMNS: tuple[str, ...] = (
    "budget",
    "expected_incremental_value",
    "n_treated",
    "total_cost",
    "dual_lambda",
    "roi",
    "marginal_value_per_unit",
    "budget_utilisation",
    "optimality_gap",
    "monotonicity_repaired",
)


def efficient_frontier(
    net_value: np.ndarray,
    cost: np.ndarray,
    budgets: np.ndarray,
    *,
    solver: str = "lagrangian",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> pd.DataFrame:
    """Re-solve the allocation across a budget grid: the chart every executive asks for.

    Each grid point is a full :func:`lagrangian_allocate` solve, so every row carries not only
    the value bought but the **shadow price that bought it**. Two columns answer the "how much
    should we spend?" question from opposite directions and should broadly agree:

    * ``dual_lambda`` -- the *analytic* marginal return on the next unit of budget at that
      point, straight from the dual.
    * ``marginal_value_per_unit`` -- the *empirical* backward finite difference
      ``d(value) / d(budget)`` between adjacent grid points.

    The curve is concave in theory, so both should fall as the budget grows; the budget is
    worth funding up to the point where ``dual_lambda`` drops through 1.0 (below that, a unit
    of budget returns less than a unit of value), and there is nothing left to buy once it
    reaches 0.

    **Monotonicity is enforced structurally, not cosmetically.** More budget can never buy
    less value: any allocation feasible at budget ``b`` is still feasible at ``b' > b``. If a
    grid point ever comes back worse than its predecessor -- possible because these are
    heuristic solutions to an NP-hard problem, not because more money hurts -- the previous
    *allocation* is carried forward wholesale and the row is flagged with
    ``monotonicity_repaired``. Carrying the allocation, rather than clipping the number with a
    running maximum, keeps ``n_treated`` and ``total_cost`` consistent with the value reported
    beside them. Any repair is logged at WARNING.

    Parameters
    ----------
    net_value : numpy.ndarray
        ``(n, n_arms)`` incremental value net of offer cost; column 0 must be 0.
    cost : numpy.ndarray
        ``(n, n_arms)`` budget consumption; column 0 must be 0.
    budgets : numpy.ndarray
        Grid of budgets. Sorted ascending on output; negatives are clipped to 0.
    solver : {"lagrangian", "greedy", "lp"}, default "lagrangian"
        Which solver to run at each grid point. ``"lagrangian"`` is the only one that yields a
        shadow price at every point and is far cheaper than ``"lp"`` over a grid.
    tol : float, default 1e-6
        Passed to :func:`lagrangian_allocate`.
    max_iter : int, default 100
        Passed to :func:`lagrangian_allocate`.

    Returns
    -------
    pandas.DataFrame
        One row per budget with ``budget, expected_incremental_value, n_treated, total_cost,
        dual_lambda, roi, marginal_value_per_unit, budget_utilisation, optimality_gap,
        monotonicity_repaired``, sorted by ``budget`` ascending.

    Raises
    ------
    RuntimeError
        If the value column is still not non-decreasing after the repair pass, which would
        mean a broken invariant rather than a numerical wobble.
    """
    grid = np.asarray(budgets, dtype=np.float64).ravel()
    if grid.size == 0:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in _FRONTIER_COLUMNS})
    if np.any(~np.isfinite(grid)):
        raise ValueError("budgets must all be finite")
    if np.any(grid < 0):
        LOGGER.warning("%d negative budgets clipped to 0", int((grid < 0).sum()))
        grid = np.clip(grid, 0.0, None)
    grid = np.sort(grid, kind="stable")

    solve = {"lagrangian": lagrangian_allocate, "greedy": greedy_knapsack, "lp": lp_allocate}.get(solver)
    if solve is None:
        raise ValueError(f"unknown solver {solver!r}; expected 'lagrangian', 'greedy' or 'lp'")
    kwargs: dict[str, Any] = {"tol": tol, "max_iter": max_iter} if solver == "lagrangian" else {}

    recs: list[dict[str, Any]] = []
    prev: dict[str, Any] | None = None
    n_repaired = 0
    for b in grid.tolist():
        res = solve(net_value, cost, float(b), **kwargs)
        value = res.expected_incremental_value
        n_treated = res.n_treated
        spent = res.total_cost
        repaired = False
        if prev is not None:
            slack = 1e-9 * max(1.0, abs(prev["expected_incremental_value"]))
            if value < prev["expected_incremental_value"] - slack:
                LOGGER.warning("frontier dipped at budget %.6g (%.6g < %.6g); carrying the "
                               "allocation from budget %.6g forward, which is feasible here",
                               b, value, prev["expected_incremental_value"], prev["budget"])
                value = prev["expected_incremental_value"]
                n_treated = prev["n_treated"]
                spent = prev["total_cost"]
                repaired = True
                n_repaired += 1
        rec = {
            "budget": float(b),
            "expected_incremental_value": float(value),
            "n_treated": int(n_treated),
            "total_cost": float(spent),
            "dual_lambda": float("nan") if res.dual_lambda is None else float(res.dual_lambda),
            "roi": float(value / spent) if spent > 0 else float("nan"),
            "budget_utilisation": float(spent / b) if b > 0 else float("nan"),
            "optimality_gap": float(res.diagnostics.get("optimality_gap", float("nan"))),
            "monotonicity_repaired": bool(repaired),
        }
        recs.append(rec)
        prev = rec

    out = pd.DataFrame.from_records(recs)
    vals = out["expected_incremental_value"].to_numpy(dtype=float)
    bud = out["budget"].to_numpy(dtype=float)
    db = np.diff(bud)
    marginal = np.full(out.shape[0], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        marginal[1:] = np.where(db > 0, np.diff(vals) / np.where(db > 0, db, 1.0), np.nan)
        if bud.size and bud[0] > 0:
            marginal[0] = vals[0] / bud[0]
    out["marginal_value_per_unit"] = marginal

    running = np.maximum.accumulate(vals)
    if np.any(running > vals + 1e-9):
        LOGGER.warning("frontier still dipped after the carry-forward repair; enforcing a running "
                       "maximum on expected_incremental_value at %d grid point(s)",
                       int(np.sum(running > vals + 1e-9)))
        out["expected_incremental_value"] = running
        out["monotonicity_repaired"] = out["monotonicity_repaired"] | (running > vals + 1e-9)
    final = out["expected_incremental_value"].to_numpy(dtype=float)
    if final.size > 1 and np.any(np.diff(final) < -1e-9):
        raise RuntimeError(f"efficient_frontier is not monotone after repair: {final}")
    if n_repaired:
        LOGGER.warning("efficient_frontier repaired %d of %d grid points", n_repaired, out.shape[0])
    return out[list(_FRONTIER_COLUMNS)]


# =====================================================================================
# 5. Baseline policies -- what the causal policy has to beat
# =====================================================================================


def _cheapest_offer(prob: _Problem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per customer, the cheapest non-control arm and its cost.

    Parameters
    ----------
    prob : _Problem
        The prepared problem.

    Returns
    -------
    arm : numpy.ndarray
        ``(n,)`` cheapest offer arm index (>= 1); meaningless where ``available`` is False.
    cost : numpy.ndarray
        ``(n,)`` cost of that arm; ``inf`` where unavailable.
    available : numpy.ndarray
        ``(n,)`` boolean, False when the customer has no feasible offer at all.
    """
    if prob.n_arms < 2 or prob.n == 0:
        return (np.zeros(prob.n, dtype=np.int64), np.full(prob.n, np.inf),
                np.zeros(prob.n, dtype=bool))
    oc = prob.cost[:, 1:].copy()
    oc[oc >= _BIG * 0.5] = np.inf
    idx = np.argmin(oc, axis=1)
    c = oc[prob.rows, idx]
    return (idx + 1).astype(np.int64), c, np.isfinite(c)


def _rank_and_fill(order: np.ndarray, prob: _Problem, arm: np.ndarray, unit_cost: np.ndarray,
                   available: np.ndarray, method: str, note: str) -> AllocationResult:
    """Give the cheapest offer to customers in ``order`` until the budget is exhausted.

    Shares :func:`_greedy_fill`, so the baselines get exactly the same skip-do-not-stop
    treatment as :func:`greedy_knapsack`. Treating them worse than the causal policy would
    make the comparison meaningless.

    Parameters
    ----------
    order : numpy.ndarray
        Customer indices in priority order (best first).
    prob : _Problem
        The prepared problem.
    arm : numpy.ndarray
        ``(n,)`` arm each customer would receive.
    unit_cost : numpy.ndarray
        ``(n,)`` cost of that arm.
    available : numpy.ndarray
        ``(n,)`` whether the customer can be treated at all.
    method : str
        Policy label.
    note : str
        Short description stored in ``diagnostics["policy"]``.

    Returns
    -------
    AllocationResult
    """
    t0 = time.perf_counter()
    assignment = np.zeros(prob.n, dtype=np.int64)
    order = order[available[order]]
    # Note the absence of a `budget <= 0` short circuit: a zero-cost offer consumes no budget,
    # so _greedy_fill still (correctly) takes it, and the baselines stay comparable with the
    # solvers on the same input instead of silently refusing free value.
    if order.size == 0:
        return _finalise(assignment, prob, method, None,
                         {"policy": note, "seconds": time.perf_counter() - t0})
    taken, _ = _greedy_fill(order, unit_cost[order], prob.budget, 0.0,
                            n_rows=prob.n, one_per_row=True)
    chosen = order[taken]
    assignment[chosen] = arm[chosen]
    return _finalise(assignment, prob, method, None,
                     {"policy": note, "n_considered": int(order.size),
                      "seconds": time.perf_counter() - t0})


def baseline_policies(
    churn_risk: np.ndarray,
    clv: np.ndarray,
    net_value: np.ndarray,
    cost: np.ndarray,
    budget: float,
    *,
    random_state: int | np.random.Generator | None = 7,
) -> dict[str, AllocationResult]:
    """Build the six reference policies the causal allocation has to beat.

    These are the incumbents: what a competent team ships without a causal model. They are
    implemented **in good faith** -- same budget, same skip-do-not-stop fill as
    :func:`greedy_knapsack`, same cheapest-offer rule, no artificial handicaps -- because a
    causal policy that only beats a straw man has proved nothing. Each policy ranks customers
    by its own score and gives the *cheapest* available offer down the list until the budget
    is exhausted.

    ================  ==========================================================================
    key               scoring rule
    ================  ==========================================================================
    ``treat_none``    Do nothing. The zero-cost, zero-risk reference; with sleeping dogs in the
                      population it is a surprisingly hard baseline to beat.
    ``treat_all``     The blanket campaign: everybody gets the cheapest offer. When the budget
                      cannot cover everyone, coverage is maximised (cheapest customers first)
                      and remaining ties are broken by a seeded draw rather than by row order,
                      since row order is often silently correlated with tenure or cohort.
    ``highest_risk``  Rank by ``churn_risk`` descending. The classic churn-model deployment,
                      and the one this project exists to argue against: it spends on lost
                      causes and on sleeping dogs, both of whom are high risk.
    ``highest_clv``   Rank by ``clv`` descending. Protects revenue concentration, but ignores
                      whether the offer changes anyone's behaviour.
    ``risk_x_clv``    Rank by ``churn_risk * clv``, i.e. expected revenue at risk. The strongest
                      non-causal heuristic, and the one worth beating: it is right that value
                      and risk both matter, and wrong only in assuming the offer works.
    ``random``        Seeded uniform ranking. The floor that any real policy must clear.
    ================  ==========================================================================

    None of these look at ``net_value`` when ranking -- that is the causal information they do
    not have. They are *scored* on ``net_value``, which is what makes the comparison fair.

    Parameters
    ----------
    churn_risk : numpy.ndarray
        ``(n,)`` predicted probability of churn. Only its ranking is used.
    clv : numpy.ndarray
        ``(n,)`` predicted customer lifetime value. Only its ranking is used.
    net_value : numpy.ndarray
        ``(n, n_arms)`` incremental value net of offer cost; column 0 must be 0.
    cost : numpy.ndarray
        ``(n, n_arms)`` budget consumption; column 0 must be 0.
    budget : float
        Hard spend limit, applied identically to every policy.
    random_state : int or numpy.random.Generator or None, default 7
        Seed for the ``random`` policy and the ``treat_all`` tie-break. Defaults to a fixed
        seed so the contract signature alone is reproducible.

    Returns
    -------
    dict of str to AllocationResult
        Keyed by the six names above, in that order.

    Raises
    ------
    ValueError
        If ``churn_risk`` or ``clv`` does not have exactly ``n`` entries.

    Notes
    -----
    When the budget binds and every customer faces the same offer cost, ``treat_all`` and
    ``random`` become statistically equivalent -- both are then an unbiased random subset of
    the population. That is a property of the problem, not a bug; they are reported separately
    because they diverge as soon as offer costs vary by customer.
    """
    prob = _prepare(net_value, cost, budget)
    n = prob.n
    # np.array(..., copy=True): never mutate the caller's score arrays while cleaning them.
    risk = np.array(churn_risk, dtype=np.float64, copy=True).ravel()
    value = np.array(clv, dtype=np.float64, copy=True).ravel()
    if risk.size != n or value.size != n:
        raise ValueError(f"churn_risk ({risk.size}) and clv ({value.size}) must both have "
                         f"{n} entries to match net_value")
    for name, arr in (("churn_risk", risk), ("clv", value)):
        bad = ~np.isfinite(arr)
        if bad.any():
            LOGGER.warning("%d non-finite %s values ranked last", int(bad.sum()), name)
            floor = float(np.min(arr[~bad])) - 1.0 if (~bad).any() else 0.0
            arr[bad] = floor

    rng = as_rng(random_state)
    tie_key = rng.random(n)      # drawn first, always, so both policies stay reproducible
    random_score = rng.random(n)

    arm, unit_cost, available = _cheapest_offer(prob)
    rows = prob.rows
    out: dict[str, AllocationResult] = {}

    out["treat_none"] = _finalise(np.zeros(n, dtype=np.int64), prob, "treat_none", None,
                                  {"policy": "no customer receives an offer"})
    order_all = np.lexsort((tie_key, np.where(available, unit_cost, np.inf)))
    out["treat_all"] = _rank_and_fill(order_all, prob, arm, unit_cost, available, "treat_all",
                                      "blanket campaign, cheapest offer, budget-truncated")
    out["highest_risk"] = _rank_and_fill(np.lexsort((rows, -risk)), prob, arm, unit_cost, available,
                                         "highest_risk", "rank by predicted churn risk")
    out["highest_clv"] = _rank_and_fill(np.lexsort((rows, -value)), prob, arm, unit_cost, available,
                                        "highest_clv", "rank by predicted lifetime value")
    out["risk_x_clv"] = _rank_and_fill(np.lexsort((rows, -(risk * value))), prob, arm, unit_cost,
                                       available, "risk_x_clv", "rank by revenue at risk")
    out["random"] = _rank_and_fill(np.lexsort((rows, -random_score)), prob, arm, unit_cost, available,
                                   "random", "seeded uniform ranking")
    return out


# =====================================================================================
# 6. Convenience dispatcher
# =====================================================================================


def allocate(
    net_value: np.ndarray,
    cost: np.ndarray,
    budget: float,
    *,
    method: str = "auto",
    **kwargs: Any,
) -> AllocationResult:
    """Dispatch to a solver by name; ``"auto"`` trades exactness for speed by problem size.

    ``"auto"`` uses :func:`lp_allocate` for small problems (``n <= 5000``), where the exact LP
    bound is nearly free, and :func:`lagrangian_allocate` above that, where the certified gap
    is already negligible and the bisection is orders of magnitude faster.

    Parameters
    ----------
    net_value : numpy.ndarray
        ``(n, n_arms)`` incremental value net of offer cost; column 0 must be 0.
    cost : numpy.ndarray
        ``(n, n_arms)`` budget consumption; column 0 must be 0.
    budget : float
        Hard spend limit.
    method : {"auto", "greedy", "lagrangian", "lp"}, default "auto"
        Solver to use.
    **kwargs
        Forwarded to the chosen solver.

    Returns
    -------
    AllocationResult

    Raises
    ------
    ValueError
        If ``method`` is not recognised.
    """
    n = int(np.asarray(net_value).shape[0]) if np.asarray(net_value).ndim == 2 else 0
    if method == "auto":
        method = "lp" if n <= 5_000 else "lagrangian"
    solvers = {"greedy": greedy_knapsack, "lagrangian": lagrangian_allocate, "lp": lp_allocate}
    if method not in solvers:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(solvers) + ['auto']}")
    return solvers[method](net_value, cost, budget, **kwargs)


# =====================================================================================
# Smoke test
# =====================================================================================


def _synthetic_problem(n: int = 20_000, random_state: int = 7) -> dict[str, np.ndarray]:
    """Build a heterogeneous multi-arm allocation problem for the smoke test.

    Mirrors the economics of the real DGP without importing it: four latent responder
    segments (persuadable / sure thing / lost cause / **sleeping dog**), lifetime value and
    churn risk that vary by segment, offers of increasing potency and price, and per-customer
    cost heterogeneity so the knapsack is genuinely non-degenerate. Sleeping dogs give
    strictly negative net value at every arm, so a correct solver must refuse to treat them
    even when budget is left over.

    Parameters
    ----------
    n : int, default 20000
        Number of customers.
    random_state : int, default 7
        Seed.

    Returns
    -------
    dict of str to numpy.ndarray
        ``net`` ``(n, 4)``, ``cost`` ``(n, 4)``, ``churn_risk`` ``(n,)``, ``clv`` ``(n,)``,
        ``segment`` ``(n,)``.
    """
    from prism.data.schema import ARM_COSTS, ARM_REDEMPTION, SEGMENT_SHARES, SEGMENTS

    rng = as_rng(random_state)
    k = len(ARM_COSTS)
    seg = rng.choice(len(SEGMENTS), size=n, p=np.asarray(SEGMENT_SHARES, dtype=float))
    clv = rng.gamma(2.2, 260.0, size=n)
    churn_risk = np.clip(rng.beta(2.0, 5.0, size=n) + 0.18 * (seg == 2) - 0.10 * (seg == 1), 0.01, 0.97)
    responsiveness = np.array([1.00, 0.10, 0.03, -0.85])[seg]
    potency = np.array([0.0, 0.30, 0.55, 0.85])
    noise = rng.normal(0.0, 0.12, size=(n, k))
    gross = clv[:, None] * churn_risk[:, None] * responsiveness[:, None] * (potency[None, :] + noise)
    gross[:, 0] = 0.0

    unit = np.asarray(ARM_COSTS, dtype=float) * np.asarray(ARM_REDEMPTION, dtype=float)
    cost = unit[None, :] * (0.65 + 0.70 * rng.random((n, k)))
    cost[:, 0] = 0.0

    net = gross - cost
    net[:, 0] = 0.0
    return {"net": net, "cost": cost, "churn_risk": churn_risk, "clv": clv, "segment": seg}


def _check(checks: list[dict[str, str]], name: str, ok: bool, detail: str = "") -> bool:
    """Record one smoke-test assertion and return its outcome."""
    checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    return bool(ok)


if __name__ == "__main__":  # pragma: no cover - smoke test
    from prism.data.schema import SEGMENTS

    t_start = time.perf_counter()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    N = 20_000
    data = _synthetic_problem(N, random_state=7)
    net, cost = data["net"], data["cost"]
    risk, clv, seg = data["churn_risk"], data["clv"], data["segment"]
    rows = np.arange(N)

    cheap_total = float(cost[:, 1].sum())
    BUDGET = 0.15 * cheap_total
    unconstrained = float(cost[rows, np.argmax(net, axis=1)].sum())
    print(f"synthetic problem: n={N:,} arms={net.shape[1]} | any-positive-arm "
          f"{float((net[:, 1:].max(axis=1) > 0).mean()):.1%} | all-negative "
          f"{float((net[:, 1:].max(axis=1) <= 0).mean()):.1%}")
    print(f"budget {BUDGET:,.0f} | cost of the blanket campaign {cheap_total:,.0f} | "
          f"unconstrained optimal spend {unconstrained:,.0f}")

    checks: list[dict[str, str]] = []
    all_ok = True

    # --- 1. the hard budget, across a sweep -------------------------------------------
    sweep = [0.0, 1_000.0, BUDGET, 4.0 * BUDGET, 10.0 * unconstrained]
    worst_over, worst_name = -np.inf, ""
    sweep_ok = True
    for b in sweep:
        for fn in (greedy_knapsack, lagrangian_allocate):
            r = fn(net, cost, b)
            over = r.total_cost - b
            if over > worst_over:
                worst_over, worst_name = over, f"{fn.__name__}@{b:,.0f}"
            sweep_ok &= bool(r.total_cost <= b + 1e-6)
            sweep_ok &= bool(r.assignment.shape == (N,) and r.assignment.dtype == np.int64)
            sweep_ok &= bool(r.assignment.min() >= 0 and r.assignment.max() < net.shape[1])
    all_ok &= _check(checks, "greedy + lagrangian never overspend (5 budgets)", sweep_ok,
                     f"worst overspend {worst_over:+.3e} at {worst_name}")

    # --- 2. the three solvers at the working budget -----------------------------------
    g = greedy_knapsack(net, cost, BUDGET)
    lag = lagrangian_allocate(net, cost, BUDGET)
    lp = lp_allocate(net, cost, BUDGET)
    solvers = {"greedy": g, "lagrangian": lag, "lp": lp}

    all_ok &= _check(checks, "lp_allocate also respects the budget",
                     bool(lp.total_cost <= BUDGET + 1e-6),
                     f"spent {lp.total_cost:,.2f} of {BUDGET:,.2f}")
    all_ok &= _check(checks, "lagrangian value >= greedy value",
                     bool(lag.expected_incremental_value >= g.expected_incremental_value - 1e-6),
                     f"lagrangian {lag.expected_incremental_value:,.2f} vs greedy "
                     f"{g.expected_incremental_value:,.2f} "
                     f"(+{lag.expected_incremental_value - g.expected_incremental_value:,.2f})")
    eps = 1e-3 * abs(lag.expected_incremental_value)
    lp_gap = lp.expected_incremental_value - lag.expected_incremental_value
    all_ok &= _check(checks, "lp value >= lagrangian value - eps",
                     bool(lp_gap >= -eps),
                     f"lp {lp.expected_incremental_value:,.2f} vs lagrangian "
                     f"{lag.expected_incremental_value:,.2f} (gap {lp_gap:+,.2f}, "
                     f"{lp_gap / max(abs(lag.expected_incremental_value), 1.0):+.4%})")

    lp_bound = float(lp.diagnostics["lp_objective"])
    best_achieved = max(r.expected_incremental_value for r in solvers.values())
    all_ok &= _check(checks, "LP relaxation upper-bounds every integer solution",
                     bool(lp_bound >= best_achieved - 1e-6),
                     f"LP bound {lp_bound:,.2f} >= best integer {best_achieved:,.2f}; "
                     f"integrality gap {lp.diagnostics['integrality_gap']:.3e} on "
                     f"{lp.diagnostics['n_fractional']} fractional row(s)")

    # --- 3. weak duality: every reported gap is a proved certificate ------------------
    duality_ok = True
    detail = []
    for name, r in solvers.items():
        ub = r.diagnostics.get("dual_upper_bound")
        if ub is None:
            continue
        duality_ok &= bool(r.expected_incremental_value <= float(ub) + 1e-6)
        detail.append(f"{name} gap {r.diagnostics['optimality_gap']:.3e}")
    all_ok &= _check(checks, "achieved value <= weak-duality upper bound", duality_ok,
                     "; ".join(detail))

    # --- 4. the two independent shadow prices agree -----------------------------------
    lam_lag = float(lag.dual_lambda or 0.0)
    lam_lp = float(lp.dual_lambda or 0.0)
    rel = abs(lam_lag - lam_lp) / max(abs(lam_lp), 1e-9)
    all_ok &= _check(checks, "bisected lambda* matches the HiGHS budget marginal", bool(rel < 0.05),
                     f"lagrangian {lam_lag:.6f} vs LP dual {lam_lp:.6f} (rel. diff {rel:.3%})")

    # --- 5. budget 0 gives an all-zero assignment -------------------------------------
    zero_ok = True
    zero_detail = []
    for name, fn in (("greedy", greedy_knapsack), ("lagrangian", lagrangian_allocate), ("lp", lp_allocate)):
        r = fn(net, cost, 0.0)
        zero_ok &= bool(np.all(r.assignment == 0) and r.total_cost == 0.0 and r.n_treated == 0)
        zero_detail.append(f"{name} n_treated={r.n_treated}")
    all_ok &= _check(checks, "budget of 0 assigns nobody", zero_ok, ", ".join(zero_detail))

    # --- 6. an unbinding budget reproduces the unconstrained argmax -------------------
    big = lagrangian_allocate(net, cost, 10.0 * unconstrained)
    want = np.argmax(net, axis=1)
    all_ok &= _check(checks, "unbinding budget => lambda* == 0 and the unconstrained optimum",
                     bool(big.dual_lambda == 0.0 and np.array_equal(big.assignment, want)),
                     f"lambda*={big.dual_lambda}, treated {big.n_treated:,}, "
                     f"spent {big.total_cost:,.0f} of {10 * unconstrained:,.0f}")

    # --- 7. the efficient frontier ----------------------------------------------------
    budgets = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.60, 1.00]) * cheap_total
    frontier = efficient_frontier(net, cost, budgets)
    fvals = frontier["expected_incremental_value"].to_numpy(dtype=float)
    flam = frontier["dual_lambda"].to_numpy(dtype=float)
    fcost = frontier["total_cost"].to_numpy(dtype=float)
    fbud = frontier["budget"].to_numpy(dtype=float)
    all_ok &= _check(checks, "frontier is monotone non-decreasing in budget",
                     bool(np.all(np.diff(fvals) >= -1e-6)),
                     f"min step {float(np.min(np.diff(fvals))):+,.2f}; "
                     f"{int(frontier['monotonicity_repaired'].sum())} row(s) repaired")
    all_ok &= _check(checks, "every frontier row respects its own budget",
                     bool(np.all(fcost <= fbud + 1e-6)),
                     f"max overspend {float(np.max(fcost - fbud)):+.3e}")
    lam_tol = 1e-6 * max(1.0, float(np.nanmax(flam)))
    all_ok &= _check(checks, "dual_lambda decreases as the budget increases",
                     bool(np.all(np.diff(flam) <= lam_tol) and flam[0] > flam[-1]),
                     f"lambda* {flam[0]:.4f} -> {flam[-1]:.4f}; max increase "
                     f"{float(np.max(np.diff(flam))):+.3e}")

    # --- 8. baseline policies ---------------------------------------------------------
    base = baseline_policies(risk, clv, net, cost, BUDGET)
    base_ok = all(r.total_cost <= BUDGET + 1e-6 for r in base.values())
    all_ok &= _check(checks, "all six baseline policies respect the budget", base_ok,
                     "max spend " + f"{max(r.total_cost for r in base.values()):,.2f} of {BUDGET:,.2f}")
    all_ok &= _check(checks, "baselines expose the full six-policy contract",
                     set(base) == {"treat_none", "treat_all", "highest_risk", "highest_clv",
                                   "risk_x_clv", "random"},
                     ", ".join(base))
    best_base_name = max(base, key=lambda kk: base[kk].expected_incremental_value)
    best_base = base[best_base_name].expected_incremental_value
    all_ok &= _check(checks, "causal allocation beats every non-causal baseline",
                     bool(lag.expected_incremental_value > best_base),
                     f"lagrangian {lag.expected_incremental_value:,.0f} vs best baseline "
                     f"'{best_base_name}' {best_base:,.0f} "
                     f"(+{lag.expected_incremental_value - best_base:,.0f})")

    # --- 9. sleeping dogs are refused treatment ---------------------------------------
    sd = seg == 3
    pers = seg == 0
    sd_rate = float((lag.assignment[sd] > 0).mean())
    pers_rate = float((lag.assignment[pers] > 0).mean())
    all_ok &= _check(checks, "sleeping dogs are not treated", bool(sd_rate == 0.0 and pers_rate > 0.2),
                     f"sleeping-dog treat rate {sd_rate:.4%}, persuadable {pers_rate:.2%}")
    risk_sd_rate = float((base["highest_risk"].assignment[sd] > 0).mean())
    all_ok &= _check(checks, "risk targeting does treat sleeping dogs (the failure mode)",
                     bool(risk_sd_rate > 0.0),
                     f"highest_risk treats {risk_sd_rate:.2%} of sleeping dogs and loses "
                     f"{base['highest_risk'].expected_incremental_value:,.0f}")

    # --- 10. determinism --------------------------------------------------------------
    again = lagrangian_allocate(net, cost, BUDGET)
    base_again = baseline_policies(risk, clv, net, cost, BUDGET)
    det = bool(np.array_equal(again.assignment, lag.assignment)
               and again.dual_lambda == lag.dual_lambda
               and all(np.array_equal(base_again[kk].assignment, base[kk].assignment) for kk in base))
    all_ok &= _check(checks, "bit-identical under the same seed", det,
                     "lagrangian + all six baselines re-ran identically")

    # --- 11. bookkeeping reconciles ---------------------------------------------------
    pa = lag.per_arm
    recon = bool(
        abs(float(pa["total_cost"].sum()) - lag.total_cost) < 1e-6
        and abs(float(pa["total_net_value"].sum()) - lag.expected_incremental_value) < 1e-6
        and int(pa["n_assigned"].sum()) == N
        and int(pa.loc[pa["arm"] > 0, "n_assigned"].sum()) == lag.n_treated
        and list(pa.columns) == ["arm", "arm_name", "n_assigned", "total_cost", "total_net_value",
                                 "mean_net_value"]
    )
    all_ok &= _check(checks, "per_arm reconciles with the headline totals", recon,
                     f"{int(pa['n_assigned'].sum()):,} rows across {len(pa)} arms")
    s = lag.summary()
    all_ok &= _check(checks, "summary() is a one-row frame", bool(s.shape[0] == 1 and s.shape[1] >= 10),
                     f"shape {s.shape}, columns {list(s.columns)[:4]}...")

    # --- 12. degenerate inputs do not crash -------------------------------------------
    allneg = np.zeros((500, 3))
    allneg[:, 1:] = -np.abs(as_rng(1).normal(5, 2, (500, 2)))
    ccost = np.zeros((500, 3))
    ccost[:, 1:] = 10.0
    r_neg = lagrangian_allocate(allneg, ccost, 5_000.0)
    r_empty = lagrangian_allocate(np.zeros((0, 4)), np.zeros((0, 4)), 100.0)
    all_ok &= _check(checks, "all-negative net values => do nothing, lambda* == 0",
                     bool(r_neg.n_treated == 0 and r_neg.dual_lambda == 0.0 and r_empty.n_treated == 0),
                     f"treated {r_neg.n_treated}, lambda* {r_neg.dual_lambda}, "
                     f"empty problem value {r_empty.expected_incremental_value:.1f}")

    # --- 13. the dominance presolve provably does not change the LP optimum -----------
    sl = slice(0, 4_000)
    sub_net, sub_cost = net[sl], cost[sl]
    sub_budget = 0.15 * float(sub_cost[:, 1].sum())
    lp_on = lp_allocate(sub_net, sub_cost, sub_budget, prune=True)
    lp_off = lp_allocate(sub_net, sub_cost, sub_budget, prune=False)
    all_ok &= _check(
        checks, "dominance presolve leaves the LP optimum unchanged",
        bool(abs(lp_on.diagnostics["lp_objective"] - lp_off.diagnostics["lp_objective"]) < 1e-6
             and np.array_equal(lp_on.assignment, lp_off.assignment)),
        f"objective {lp_on.diagnostics['lp_objective']:,.4f} either way; "
        f"{lp_on.diagnostics['n_columns_pruned']:,} of {sub_net.size:,} columns pruned, "
        f"{lp_off.diagnostics['solve_seconds']:.2f}s -> {lp_on.diagnostics['solve_seconds']:.2f}s")

    # --- 14. per-arm capacity limits are honoured -------------------------------------
    caps = np.array([np.inf, np.inf, np.inf, 50.0])
    lp_cap = lp_allocate(sub_net, sub_cost, sub_budget, arm_capacity=caps)
    n_concierge = int((lp_cap.assignment == 3).sum())
    all_ok &= _check(checks, "arm_capacity caps the expensive arm without breaking the budget",
                     bool(n_concierge <= 50 and lp_cap.total_cost <= sub_budget + 1e-6
                          and lp_cap.expected_incremental_value <= lp_on.expected_incremental_value + 1e-6),
                     f"concierge assigned {n_concierge} (cap 50), value "
                     f"{lp_cap.expected_incremental_value:,.0f} vs uncapped "
                     f"{lp_on.expected_incremental_value:,.0f}")

    # --- 15. exact optimum by enumeration on a small instance -------------------------
    # The only check here that does not trust any of this module's own machinery: enumerate
    # all 4 ** 10 assignments of a 10-customer, 4-arm problem and compare. Run over several
    # seeds and several budget fractions, because a single instance is easy to pass by luck.
    def _brute_force_optimum(bnet: np.ndarray, bcost: np.ndarray, b: float) -> float:
        """Exact MCKP optimum by enumerating every assignment. O(K ** n) -- tiny n only."""
        bn, bk = bnet.shape
        total = bk ** bn
        val = np.zeros(total)
        csum = np.zeros(total)
        code = np.arange(total, dtype=np.int64)
        for j in range(bn):
            digit = (code // (bk ** j)) % bk
            val += bnet[j, digit]
            csum += bcost[j, digit]
        feasible = csum <= b + 1e-12
        return float(np.max(np.where(feasible, val, -np.inf))) if feasible.any() else 0.0

    brute_ok = True
    worst_gap = {"greedy": 0.0, "lagrangian": 0.0, "lp": 0.0}
    understated = 0
    exact_hits = 0
    n_brute = 0
    for bseed in (11, 23, 41, 57):
        brng = as_rng(bseed)
        bn, bk = 10, 4
        bnet = np.zeros((bn, bk))
        bcost = np.zeros((bn, bk))
        bnet[:, 1:] = brng.normal(2.0, 6.0, (bn, bk - 1))
        bcost[:, 1:] = brng.gamma(2.0, 3.0, (bn, bk - 1))
        bcost[brng.integers(0, bn), brng.integers(1, bk)] = 0.0   # a free arm, deliberately
        for frac in (0.05, 0.2, 0.5):
            bb = float(frac * bcost[:, 1:].sum())
            opt = _brute_force_optimum(bnet, bcost, bb)
            n_brute += 1
            for nm, fn in (("greedy", greedy_knapsack), ("lagrangian", lagrangian_allocate),
                           ("lp", lp_allocate)):
                r = fn(bnet, bcost, bb)
                got = float(bnet[np.arange(bn), r.assignment].sum())
                spend = float(bcost[np.arange(bn), r.assignment].sum())
                true_gap = (opt - got) / max(abs(opt), 1.0)
                worst_gap[nm] = max(worst_gap[nm], true_gap)
                if nm == "lagrangian" and true_gap <= 1e-9:
                    exact_hits += 1
                # Hard invariants. No heuristic may beat the exact optimum or overspend; the
                # reported value must be the value of the reported assignment; the shipped
                # certificate must bound the TRUE optimum, not merely its own answer; and --
                # the point of shipping a certificate at all -- the gap a solver *reports*
                # must never be smaller than the gap it actually has.
                ub = r.diagnostics.get("dual_upper_bound")
                claimed = float(r.diagnostics.get("optimality_gap", np.inf))
                if np.isfinite(claimed) and claimed + 1e-9 < true_gap:
                    understated += 1
                brute_ok &= bool(got <= opt + 1e-9 and spend <= bb + 1e-9
                                 and abs(got - r.expected_incremental_value) < 1e-9
                                 and (ub is None or not np.isfinite(float(ub))
                                      or float(ub) >= opt - 1e-7))
    # Quality bars, deliberately loose: at n = 10 a single rounded-down customer is a tenth
    # of the problem, and closing the last LP gap needs a two-customer swap that no
    # one-customer top-up can reach. What must hold exactly is that no reported gap lies.
    brute_ok &= bool(understated == 0)
    brute_ok &= bool(exact_hits >= n_brute - 1 and worst_gap["lagrangian"] < 0.05
                     and worst_gap["lp"] < 0.05)
    all_ok &= _check(checks, f"matches the brute-force optimum on {n_brute} small instances", brute_ok,
                     f"4**10 enumeration: lagrangian exact on {exact_hits}/{n_brute}; worst true gap "
                     f"lagrangian {worst_gap['lagrangian']:.3e}, lp {worst_gap['lp']:.3e}, greedy "
                     f"{worst_gap['greedy']:.3e}; reported gaps that understate reality: {understated}")

    # --- 16. local optimality: no single customer move is left on the table -----------
    # A certified solver must return an allocation that no one-customer move can improve.
    # This is the check that catches a solver stranding budget after rounding: before the
    # residual top-up was added to lp_allocate, the LP failed here and lost to the dual.
    def _missed_upgrade(res: AllocationResult) -> tuple[int, float]:
        """Count (and size) the feasible single-customer moves that would raise the value."""
        a = res.assignment
        gain = net - net[rows, a][:, None]
        extra = cost - cost[rows, a][:, None]
        ok = (gain > 1e-9) & (extra <= (BUDGET - res.total_cost) + 1e-12)
        return int(ok.sum()), (float(gain[ok].max()) if ok.any() else 0.0)

    local_ok = True
    local_detail = []
    for nm, r in (("lagrangian", lag), ("lp", lp)):
        cnt, best_gain = _missed_upgrade(r)
        local_ok &= bool(cnt == 0)
        local_detail.append(f"{nm} {cnt} missed (best +{best_gain:,.2f})")
    g_cnt, _g_gain = _missed_upgrade(g)
    all_ok &= _check(checks, "certified solvers are local optima under one-customer moves",
                     local_ok, "; ".join(local_detail) + f"; greedy {g_cnt} (no guarantee, by design)")

    # --- 17. a zero-cost offer is taken even at a zero budget, by every solver ---------
    # Refusing value that consumes no budget would be wrong, and the three solvers must agree
    # on the budget == 0 edge case rather than each holding its own opinion of it.
    free_net = np.array([[0.0, 10.0, 5.0], [0.0, 7.0, -1.0], [0.0, -3.0, 2.0]])
    free_cost = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    free_res = {nm: fn(free_net, free_cost, 0.0) for nm, fn in
                (("greedy", greedy_knapsack), ("lagrangian", lagrangian_allocate),
                 ("lp", lp_allocate))}
    free_ok = all(r.expected_incremental_value == 17.0 and r.total_cost == 0.0
                  for r in free_res.values())
    all_ok &= _check(checks, "a free offer is taken at budget 0, identically by all 3 solvers",
                     free_ok, ", ".join(f"{nm} {r.expected_incremental_value:.0f}"
                                        for nm, r in free_res.items()) + " (exact optimum 17)")

    # --- 18. no entry point mutates the caller's arrays -------------------------------
    keep_net, keep_cost, keep_risk, keep_clv = net.copy(), cost.copy(), risk.copy(), clv.copy()
    _ = allocate(net, cost, BUDGET)
    _ = efficient_frontier(net, cost, np.array([0.0, BUDGET]))
    _ = baseline_policies(risk, clv, net, cost, BUDGET)
    _ = greedy_knapsack(net, cost, BUDGET, candidates_per_customer=3)
    untouched = bool(np.array_equal(net, keep_net) and np.array_equal(cost, keep_cost)
                     and np.array_equal(risk, keep_risk) and np.array_equal(clv, keep_clv))
    all_ok &= _check(checks, "caller arrays are never mutated", untouched,
                     "net, cost, churn_risk and clv bit-identical after 4 entry points")

    # --- output ------------------------------------------------------------------------
    print("\n--- efficient frontier (lagrangian over the budget grid) ---")
    show = frontier.copy()
    for c in ("budget", "expected_incremental_value", "total_cost"):
        show[c] = show[c].map(lambda v: f"{v:,.0f}")
    for c in ("dual_lambda", "roi", "marginal_value_per_unit", "budget_utilisation"):
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    show["optimality_gap"] = show["optimality_gap"].map(lambda v: f"{v:.2e}")
    print(show.to_string(index=False))

    print(f"\n--- solver comparison at budget {BUDGET:,.0f} ---")
    comp = pd.concat([r.summary() for r in solvers.values()], ignore_index=True)
    comp["seconds"] = [float(r.diagnostics.get("seconds", np.nan)) for r in solvers.values()]
    print(comp.round(4).to_string(index=False))

    print("\n--- baseline policies (what the causal policy has to beat) ---")
    bl = pd.concat([r.summary() for r in base.values()] + [lag.summary()], ignore_index=True)
    bl["vs_best_baseline"] = bl["expected_incremental_value"] - best_base
    print(bl[["method", "n_treated", "total_cost", "expected_incremental_value", "roi",
              "value_per_treated", "vs_best_baseline"]].round(3).to_string(index=False))

    print("\n--- per-arm breakdown of the lagrangian allocation ---")
    print(lag.per_arm.round(3).to_string(index=False))

    print("\n--- treat rate by latent responder segment (lagrangian vs highest_risk) ---")
    seg_tbl = pd.DataFrame({
        "segment": [SEGMENTS[s] for s in seg],
        "causal": lag.assignment > 0,
        "highest_risk": base["highest_risk"].assignment > 0,
        "net_if_causal": net[rows, lag.assignment],
        "net_if_risk": net[rows, base["highest_risk"].assignment],
    }).groupby("segment", as_index=False).agg(
        n=("causal", "size"), causal_treat_rate=("causal", "mean"),
        risk_treat_rate=("highest_risk", "mean"),
        causal_value=("net_if_causal", "sum"), risk_value=("net_if_risk", "sum"))
    print(seg_tbl.round(4).to_string(index=False))

    print("\n--- lagrangian diagnostics ---")
    print(pd.DataFrame([{k: (round(v, 6) if isinstance(v, float) else v)
                         for k, v in lag.diagnostics.items()}]).to_string(index=False))

    print(f"\n--- smoke-test results ---  ({len(checks)} checks)")
    print(pd.DataFrame(checks).to_string(index=False))

    elapsed = time.perf_counter() - t_start
    print(f"\nlambda* = {lam_lag:.4f} -> the next unit of budget returns ~{lam_lag:.2f} units of "
          f"incremental value; an offer is worth giving only if net > {lam_lag:.2f} x cost")
    print(f"total {elapsed:.2f}s | greedy {g.diagnostics['seconds']:.2f}s | "
          f"lagrangian {lag.diagnostics['seconds']:.2f}s ({lag.diagnostics['n_iter']} bisections) | "
          f"lp {lp.diagnostics['seconds']:.2f}s")
    if not all_ok:
        raise SystemExit("optimize.py SMOKE TEST FAILED")
    if elapsed > 60.0:
        raise SystemExit(f"optimize.py smoke test exceeded the 60s budget ({elapsed:.1f}s)")
    print("optimize.py OK")
