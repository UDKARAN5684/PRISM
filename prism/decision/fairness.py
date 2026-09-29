"""Fairness auditing for a retention budget -- who gets *offered* help, not who gets denied.

Most algorithmic-fairness machinery was built for *punitive* or *gatekeeping* decisions: a loan
refused, a resume screened out, bail denied. A retention budget is the mirror image. Nobody is
harmed by the model declining to send them a voucher in the way somebody is harmed by a refused
mortgage, and there is no "qualified applicant" who is owed the offer. What a retention budget *is*
is a **scarce benefit**: a fixed pot of money, more customers worth saving than money to save them
with, and an optimiser that decides whose loyalty gets bought. The fairness question is therefore
**allocative** -- *who receives the offers* -- and not the usual error-rate question about who is
wrongly denied credit.

That reframing changes what the numbers mean, so this module is explicit about it:

* **Demographic parity** here compares *treat rates* (share of a group receiving any offer). A gap
  means one group's customers are systematically being bought back and another's are not.
* **Equal opportunity** is redefined for a benefit. In the classifier setting it conditions on the
  true positive label. Here the analogue of "deserves the positive outcome" is **"the offer would
  actually pay for itself for this customer"** -- ``net_value > 0``. So the equal-opportunity gap
  asks: among customers an offer *would* help, is one group getting served less than the
  best-served group? That is the number that catches a policy which is nominally value-maximising
  but happens to concentrate a public-facing benefit in one region or plan tier.
* **The four-fifths rule** (:func:`disparate_impact_ratio`) is a *borrowed* yardstick, not a legal
  standard for marketing. See that function's docstring; being precise about the borrowing is the
  entire point of quoting it.

And because a fairness constraint is a *business decision*, not a checkbox, the module makes its
cost legible: :func:`reweight_for_parity` produces a parity-constrained allocation and
:func:`price_of_fairness` reports, in currency, exactly how much incremental value that constraint
gave up. An argument about whether parity is worth it only becomes a decision once somebody can see
the price tag.

Notes
-----
Nothing in this module is a legal opinion, and none of these metrics discharges a legal or ethical
obligation on their own. They are instrumented, auditable numbers for a human to argue over.

Examples
--------
>>> import logging, numpy as np, pandas as pd
>>> logging.getLogger("prism").setLevel(logging.ERROR)   # quiet the progress logs
>>> nv = np.column_stack([np.zeros(6), np.array([10.0, -5.0, 40.0, 3.0, -2.0, 25.0])])
>>> cost = np.column_stack([np.zeros(6), np.full(6, 8.0)])
>>> grp = pd.Series(list("AAABBB"))
>>> res = reweight_for_parity(nv, cost, budget=24.0, protected=grp, tolerance=0.34)
>>> int(res.n_treated)
3
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from prism.data.schema import ARM_NAMES, N_ARMS
from prism.utils.logging import get_logger

__all__ = [
    "FOUR_FIFTHS_RULE",
    "UNKNOWN_GROUP",
    "OVERALL_LABEL",
    "FairnessAllocationResult",
    "policy_fairness_audit",
    "disparate_impact_ratio",
    "unconstrained_allocation",
    "reweight_for_parity",
    "price_of_fairness",
]

LOG = get_logger("decision.fairness")

#: Conventional (borrowed) disparate-impact threshold. See :func:`disparate_impact_ratio`.
FOUR_FIFTHS_RULE: float = 0.8
#: Label used for rows whose protected attribute is missing.
UNKNOWN_GROUP: str = "unknown"
#: Label of the pooled row appended to every audit table.
OVERALL_LABEL: str = "OVERALL"

_EPS = 1e-12


# ======================================================================================
# Fallback container (see reweight_for_parity's docstring for why this exists)
# ======================================================================================
@dataclass
class FairnessAllocationResult:
    """Local stand-in for :class:`prism.decision.optimize.AllocationResult`.

    ``prism.decision.optimize`` is imported lazily inside :func:`reweight_for_parity` so that this
    module stays importable while that module is still being written (and so that a circular import
    can never form). When the lazy import fails, an instance of this dataclass -- field-for-field
    identical to the contract in ``SPEC.md`` section 5.2 -- is returned instead. Downstream code
    that only reads attributes cannot tell the two apart.

    Attributes
    ----------
    assignment : numpy.ndarray
        ``(n,)`` int64 chosen arm per customer; ``0`` means "no offer".
    total_cost : float
        Sum of the cost of the chosen arms.
    expected_incremental_value : float
        Sum of the *net* value of the chosen arms (net value is already cost-adjusted, per
        ``SPEC.md`` section 5.1, so this is profit, not revenue).
    n_treated : int
        Number of customers with ``assignment > 0``.
    per_arm : pandas.DataFrame
        One row per arm: ``arm, arm_name, n_assigned, n, share, total_cost, total_net_value,
        mean_net_value``. ``n_assigned`` is the name the sibling module's contract uses; ``n`` is
        kept as an alias so both spellings read.
    dual_lambda : float or None
        Shadow price of the budget constraint (currency of value per currency of spend). ``0.0``
        when the budget is not binding -- more money would buy nothing.
    method : str
        Human-readable description of the solver that produced this result.
    budget : float or None
        The budget this allocation was solved against, kept for reporting.
    """

    assignment: np.ndarray
    total_cost: float
    expected_incremental_value: float
    n_treated: int
    per_arm: pd.DataFrame
    dual_lambda: float | None
    method: str
    budget: float | None = None
    _protected: pd.Series | None = field(default=None, repr=False, compare=False)

    def summary(self) -> pd.DataFrame:
        """Return a one-row summary frame of the headline allocation numbers.

        Returns
        -------
        pandas.DataFrame
            Columns ``method, n_treated, total_cost, expected_incremental_value, roi,
            dual_lambda``.
        """
        roi = float(self.expected_incremental_value / self.total_cost) if self.total_cost > 0 else np.nan
        return pd.DataFrame(
            [
                {
                    "method": self.method,
                    "n_treated": int(self.n_treated),
                    "total_cost": float(self.total_cost),
                    "expected_incremental_value": float(self.expected_incremental_value),
                    "roi": roi,
                    "dual_lambda": self.dual_lambda,
                }
            ]
        )


# ======================================================================================
# Input coercion
# ======================================================================================
def _as_int_vector(assignment: Any, name: str = "assignment") -> np.ndarray:
    """Coerce an assignment-like object to a 1-D int64 array."""
    arr = np.asarray(assignment)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size and not np.all(np.isfinite(np.asarray(arr, dtype=np.float64))):
        raise ValueError(f"{name} must not contain NaN or inf")
    out = np.asarray(arr, dtype=np.int64)
    if out.size and out.min() < 0:
        raise ValueError(f"{name} must be non-negative arm indices (0 = no offer)")
    return out


def _as_float_vector(values: Any, n: int, name: str) -> np.ndarray:
    """Coerce a per-customer quantity to a 1-D float64 array of length ``n``."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} has length {arr.size}, expected {n}")
    return arr


def _warn_non_finite(values: np.ndarray, name: str) -> None:
    """Warn (once per call) when an audit input carries NaN or inf.

    The audit deliberately does **not** use ``nanmean``: dropping the bad rows from one group's
    mean and not another's would make the groups incomparable, which is worse than a visible NaN.
    But a NaN that arrives unannounced in a fairness table is indistinguishable from "not
    applicable", so the arithmetic stays honest and the warning says why the cell is empty.
    """
    bad = int(np.count_nonzero(~np.isfinite(values)))
    if bad:
        LOG.warning(
            "%s contains %d non-finite value(s); the affected group means will be NaN "
            "(they are not dropped -- that would make the groups incomparable)",
            name,
            bad,
        )


def _protected_series(protected: Any, n: int) -> tuple[pd.Series, list[str], int]:
    """Normalise a protected attribute into string labels, never raising on missing values.

    Missing values (``NaN``, ``None``, ``pd.NA``, ``NaT``) are *not* dropped and do not raise: they
    become their own group, labelled :data:`UNKNOWN_GROUP`. Dropping them would silently shrink the
    audit population and hide exactly the customers whose demographics the business failed to
    record, which is usually a non-random slice.

    Parameters
    ----------
    protected : pandas.Series or numpy.ndarray or sequence
        Group membership, one entry per customer. Any dtype; values are stringified.
    n : int
        Expected length.

    Returns
    -------
    labels : pandas.Series
        Length-``n`` object Series of string labels with a fresh ``RangeIndex``.
    categories : list of str
        Sorted label universe. For a :class:`pandas.CategoricalDtype` input this includes
        categories with **zero** members, so empty groups still appear in the audit.
    n_missing : int
        Number of rows folded into the ``unknown`` group.
    """
    s = protected if isinstance(protected, pd.Series) else pd.Series(np.asarray(protected).reshape(-1))
    if len(s) != n:
        raise ValueError(f"protected has length {len(s)}, expected {n}")

    declared: list[str] = []
    if isinstance(s.dtype, pd.CategoricalDtype):
        declared = [str(c) for c in s.dtype.categories]
        s = s.astype(object)

    s = s.reset_index(drop=True)
    missing = s.isna()
    n_missing = int(missing.sum())
    labels = s.astype(object).where(~missing, UNKNOWN_GROUP)
    labels = labels.map(lambda v: v if isinstance(v, str) else str(v))

    if n_missing:
        collides = bool((labels[~missing] == UNKNOWN_GROUP).any())
        LOG.info(
            "protected attribute has %d missing value(s); reported as group '%s'%s",
            n_missing,
            UNKNOWN_GROUP,
            " (merged with pre-existing 'unknown' values)" if collides else "",
        )

    categories = sorted(set(labels.tolist()) | set(declared))
    return labels, categories, n_missing


def _coerce_value_cost(net_value: Any, cost: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``(net_value, cost)`` into aligned ``(n, n_arms)`` float64 matrices.

    Accepts either the multi-arm form from :func:`prism.decision.economics.expected_net_value`
    (``net_value`` of shape ``(n, n_arms)``) or the single-offer form (``net_value`` of shape
    ``(n,)``, silently promoted to two arms: do-nothing and the one offer).

    ``cost`` may be a scalar, a per-arm vector of length ``n_arms``, or a full ``(n, n_arms)``
    matrix. Arm 0 is the do-nothing arm and is normalised to zero value and zero cost.

    Parameters
    ----------
    net_value : numpy.ndarray
        Per-customer, per-arm value already net of offer cost.
    cost : float or numpy.ndarray
        Budget-consuming spend per customer per arm.

    Returns
    -------
    value : numpy.ndarray
        ``(n, n_arms)`` float64.
    cost : numpy.ndarray
        ``(n, n_arms)`` float64, non-negative.
    """
    V = np.asarray(net_value, dtype=np.float64)
    if V.ndim == 1:
        V = np.column_stack([np.zeros(V.shape[0]), V])
    elif V.ndim != 2:
        raise ValueError(f"net_value must be 1-D or 2-D, got shape {V.shape}")
    n, n_arms = V.shape
    if n_arms < 1:
        raise ValueError("net_value must have at least one arm column")

    C = np.asarray(cost, dtype=np.float64)
    if C.ndim == 0:
        C = np.full((n, n_arms), float(C))
    elif C.ndim == 1 and C.shape[0] == n_arms:
        C = np.tile(C.reshape(1, -1), (n, 1))
    elif C.ndim == 1 and C.shape[0] == n and n_arms == 2:
        C = np.column_stack([np.zeros(n), C])
    elif C.ndim == 2 and C.shape == (n, n_arms):
        C = C.copy()
    else:
        raise ValueError(f"cost shape {C.shape} is not compatible with net_value shape {V.shape}")

    V = V.copy()
    if np.any(np.abs(V[:, 0]) > _EPS) or np.any(np.abs(C[:, 0]) > _EPS):
        LOG.info("arm 0 is the do-nothing arm; its value and cost were normalised to zero")
    V[:, 0] = 0.0
    C[:, 0] = 0.0
    if not np.all(np.isfinite(V)) or not np.all(np.isfinite(C)):
        raise ValueError("net_value and cost must be finite")
    if np.any(C < 0.0):
        raise ValueError("cost must be non-negative")
    return V, C


# ======================================================================================
# Per-block Lagrangian allocator (the engine behind reweight_for_parity)
# ======================================================================================
class _Block:
    """One population block (the whole sample, or a single protected group).

    Solves the multi-choice knapsack ``max sum_i v_{i,a_i}  s.t.  sum_i c_{i,a_i} <= B`` by its
    Lagrangian relaxation. At shadow price ``lam`` each customer independently picks
    ``argmax_a (v_a - lam * c_a)``, so a customer is treated exactly while
    ``lam < r_i = max_{a>0} v_{i,a} / c_{i,a}``. Sorting by ``r_i`` therefore gives the order in
    which customers enter treatment as the price falls, which turns "solve for a price" into
    "choose a head-count" -- and a head-count is precisely what a parity constraint pins down.

    Two consequences used throughout:

    * ``cost(k)`` is non-decreasing in ``k``: adding a customer adds spend, and the lower shadow
      price it implies weakly pushes every already-treated customer onto a more expensive arm. So
      the largest affordable ``k`` can be found by bisection.
    * ``lam`` is allowed to go **negative**. A negative shadow price is a fairness *subsidy*: it
      buys customers whose offer does not pay for itself, which is the only way to lift an
      under-served group whose positive-value candidates have run out.
    """

    __slots__ = ("V", "C", "n", "n_arms", "order", "ratio", "n_positive", "lam_hi", "_cache")

    def __init__(self, V: np.ndarray, C: np.ndarray) -> None:
        self.V = np.ascontiguousarray(V, dtype=np.float64)
        self.C = np.ascontiguousarray(C, dtype=np.float64)
        self.n, self.n_arms = self.V.shape
        if self.n_arms > 1 and self.n > 0:
            Vt, Ct = self.V[:, 1:], self.C[:, 1:]
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = np.where(Ct > 0.0, Ct, 1.0)
                per_arm = np.where(Ct > 0.0, Vt / denom, np.where(Vt > 0.0, np.inf, -np.inf))
            self.ratio = per_arm.max(axis=1)
        else:
            self.ratio = np.full(self.n, -np.inf)
        self.order = np.argsort(-self.ratio, kind="stable")
        self.n_positive = int(np.sum(self.ratio > 0.0))
        finite = self.ratio[np.isfinite(self.ratio)]
        scale = float(np.abs(finite).max()) if finite.size else 1.0
        self.lam_hi = 10.0 * max(scale, 1.0) + 1.0
        self._cache: dict[int, tuple[np.ndarray, float, float, float]] = {}

    def lam_for(self, k: int) -> float:
        """Return the shadow price that selects exactly the top ``k`` customers."""
        if k <= 0:
            return self.lam_hi
        lam = float(self.ratio[self.order[int(k) - 1]])
        if np.isposinf(lam):
            return self.lam_hi
        if np.isneginf(lam):
            return -self.lam_hi
        return lam - 1e-9 * max(1.0, abs(lam))

    def solve(self, k: int) -> tuple[np.ndarray, float, float, float]:
        """Allocate to exactly ``k`` customers.

        Parameters
        ----------
        k : int
            Head-count to treat, clipped into ``[0, n]``.

        Returns
        -------
        arms : numpy.ndarray
            ``(n,)`` int64 arm per customer within this block.
        cost : float
        value : float
        lam : float
            The shadow price implied by ``k``.

        Notes
        -----
        Exact ties in ``r_i`` are broken by original row order (a stable argsort), so the head-count
        is honoured exactly rather than overshooting to include every tied customer.
        """
        k = int(min(max(k, 0), self.n))
        hit = self._cache.get(k)
        if hit is not None:
            return hit
        arms = np.zeros(self.n, dtype=np.int64)
        lam = self.lam_for(k)
        if k == 0 or self.n_arms < 2:
            out = (arms, 0.0, 0.0, lam)
        else:
            sel = self.order[:k]
            scores = self.V[sel, 1:] - lam * self.C[sel, 1:]
            best = np.argmax(scores, axis=1) + 1
            arms[sel] = best
            out = (arms, float(self.C[sel, best].sum()), float(self.V[sel, best].sum()), lam)
        self._cache[k] = out
        return out

    def cost(self, k: int) -> float:
        """Total spend when treating the top ``k`` customers."""
        return self.solve(k)[1]

    def value(self, k: int) -> float:
        """Total net value when treating the top ``k`` customers."""
        return self.solve(k)[2]

    def max_k_within(self, budget: float, k_lo: int = 0, k_hi: int | None = None) -> int:
        """Largest ``k`` in ``[k_lo, k_hi]`` whose cost fits ``budget`` (bisection)."""
        hi = self.n if k_hi is None else int(min(k_hi, self.n))
        lo = int(max(0, min(k_lo, hi)))
        if self.cost(hi) <= budget + _EPS:
            return hi
        if self.cost(lo) > budget + _EPS:
            return lo  # caller decides what to do about the overrun
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.cost(mid) <= budget + _EPS:
                lo = mid
            else:
                hi = mid
        return lo


def _spend_slack(
    block: _Block,
    arms: np.ndarray,
    budget: float,
    *,
    allow_new_treated: bool,
    guard: Callable[[int, int, int], bool] | None = None,
    max_rounds: int = 8,
) -> np.ndarray:
    """Greedily spend leftover budget on strictly value-improving arm changes.

    A head-count solve fixes one shadow price for the whole block, so it stops as soon as the
    *next customer* is unaffordable -- and then leaves the remaining slack unspent even when that
    slack would comfortably pay to upgrade somebody already being served onto a stronger arm. That
    unspent slack is pure lost value, and it is what made the unconstrained baseline weaker than
    the parity-constrained allocation it is supposed to bound.

    Each round applies, in order: every *free* improvement (more value at no extra cost, so slack
    can only grow), then *paid* improvements in descending value-per-extra-currency order while
    the slack lasts. At most one move per customer per round; rounds stop as soon as nothing
    changes. Fully deterministic -- ties break on row index.

    Parameters
    ----------
    block : _Block
        The block whose ``V`` / ``C`` matrices ``arms`` indexes into.
    arms : numpy.ndarray
        ``(n,)`` current arm per customer. Not modified in place.
    budget : float
        Hard spend cap that the returned assignment respects.
    allow_new_treated : bool
        When ``False``, a customer's *treated status* is frozen: the untreated stay untreated and
        the treated may only switch between non-control arms.
    guard : callable, optional
        ``guard(i, new_arm, old_arm) -> bool``, consulted only for a move that is already
        improving and affordable, and only for a move that changes treated status. Returning
        ``False`` rejects it. :func:`reweight_for_parity` passes a guard that rejects any addition
        which would push a group outside its tolerance band, so the slack can still be spent
        without re-opening the gap the constraint just closed. **A guard that returns ``True``
        must record the move**, because the caller applies it immediately.
    max_rounds : int, default 8
        Cap on improvement rounds.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` int64 arms, weakly better in value and never over ``budget``.
    """
    V, C, n, n_arms = block.V, block.C, block.n, block.n_arms
    if n == 0 or n_arms < 2:
        return arms
    arms = np.array(arms, dtype=np.int64, copy=True)
    rows = np.arange(n)
    tol = 1e-12

    # Moves that are structurally forbidden, independent of the current assignment.
    banned = np.zeros((n, n_arms), dtype=bool)
    if not allow_new_treated:
        untreated = arms == 0
        banned[untreated, 1:] = True   # an untreated customer must stay untreated
        banned[~untreated, 0] = True   # a treated customer must stay treated

    for _ in range(int(max_rounds)):
        spend = float(C[rows, arms].sum())
        slack = budget - spend
        dV = V - V[rows, arms][:, None]
        dC = C - C[rows, arms][:, None]
        improving = (dV > tol) & ~banned

        # --- free moves: strictly more value at no extra cost -------------------------
        free = improving & (dC <= tol)
        changed = False
        if free.any():
            cand = np.where(free, dV, -np.inf)
            best = np.argmax(cand, axis=1)
            take = free[rows, best]
            if take.any():
                if guard is None:
                    arms[take] = best[take]
                    changed = True
                else:
                    for i in rows[take]:
                        a, old = int(best[i]), int(arms[i])
                        if (a > 0) == (old > 0) or guard(int(i), a, old):
                            arms[i] = a
                            changed = True
            if changed:
                dV = V - V[rows, arms][:, None]
                dC = C - C[rows, arms][:, None]
                improving = (dV > tol) & ~banned
                slack = budget - float(C[rows, arms].sum())

        # --- paid moves: best value per extra currency, greedily, while slack lasts ----
        paid = improving & (dC > tol) & (dC <= slack + 1e-9)
        if paid.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                density = np.where(paid, dV / np.where(dC > tol, dC, 1.0), -np.inf)
            best = np.argmax(density, axis=1)
            has = paid[rows, best]
            cand_rows = rows[has]
            if cand_rows.size:
                # Descending density; ties broken by row index so the result is deterministic.
                order = np.lexsort((cand_rows, -density[cand_rows, best[cand_rows]]))
                for i in cand_rows[order]:
                    a, old = int(best[i]), int(arms[i])
                    step = float(C[i, a] - C[i, old])
                    if step > slack + 1e-9 or V[i, a] - V[i, old] <= tol:
                        continue
                    if guard is not None and (a > 0) != (old > 0) and not guard(int(i), a, old):
                        continue
                    slack -= step
                    arms[i] = a
                    changed = True
        if not changed:
            break

    final = float(C[rows, arms].sum())
    if final > budget + 1e-6:  # pragma: no cover - the greedy never exceeds the slack it tracked
        raise AssertionError(f"_spend_slack overspent: {final} > {budget}")
    return arms


def _per_arm_frame(arms: np.ndarray, V: np.ndarray, C: np.ndarray) -> pd.DataFrame:
    """Build the per-arm breakdown frame of an allocation."""
    n, n_arms = V.shape
    rows: list[dict[str, Any]] = []
    idx = np.arange(n)
    for a in range(n_arms):
        mask = arms == a
        k = int(mask.sum())
        rows.append(
            {
                "arm": int(a),
                "arm_name": ARM_NAMES[a] if a < len(ARM_NAMES) else f"arm_{a}",
                # ``n_assigned`` is the column name ``prism.decision.optimize.AllocationResult``
                # documents and emits. Matching it keeps artifacts/reports/policy_comparison.csv
                # consistent no matter which module produced the row. ``n`` is kept as an alias
                # so existing readers of this module do not break.
                "n_assigned": k,
                "n": k,
                "share": float(k / n) if n else np.nan,
                "total_cost": float(C[idx[mask], a].sum()) if k else 0.0,
                "total_net_value": float(V[idx[mask], a].sum()) if k else 0.0,
                "mean_net_value": float(V[idx[mask], a].mean()) if k else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _make_result(
    arms: np.ndarray,
    V: np.ndarray,
    C: np.ndarray,
    dual_lambda: float | None,
    method: str,
    protected: pd.Series | None = None,
    budget: float | None = None,
) -> Any:
    """Build an ``AllocationResult``, preferring the real one from ``prism.decision.optimize``.

    The import is deliberately *inside* this function. ``prism.decision.optimize`` is authored in
    parallel with this module; a top-level import would make ``prism.decision.fairness`` fail to
    import (or dead-lock in a circular import) whenever that module is missing, half-written or
    broken. On any failure we fall back to :class:`FairnessAllocationResult`, which carries the
    identical field set.
    """
    idx = np.arange(V.shape[0])
    total_cost = float(C[idx, arms].sum())
    total_value = float(V[idx, arms].sum())
    n_treated = int(np.sum(arms > 0))
    per_arm = _per_arm_frame(arms, V, C)

    cls: type | None = None
    try:  # pragma: no cover - depends on a module written by another agent
        from prism.decision.optimize import AllocationResult as _AR

        cls = _AR
    except Exception as exc:  # ImportError, or anything raised while that module loads
        LOG.debug("prism.decision.optimize unavailable (%s); using the local AllocationResult", exc)

    result: Any = None
    if cls is not None:
        kwargs: dict[str, Any] = {
            "assignment": arms,
            "total_cost": total_cost,
            "expected_incremental_value": total_value,
            "n_treated": n_treated,
            "per_arm": per_arm,
            "dual_lambda": dual_lambda,
            "method": method,
        }
        # ``budget`` is an optional field on the sibling's AllocationResult. Populate it when it
        # exists, otherwise its ``summary()`` reports budget and budget_utilisation as NaN for
        # every row this module produces.
        if budget is not None and "budget" in getattr(cls, "__dataclass_fields__", {}):
            kwargs["budget"] = float(budget)
        try:
            result = cls(**kwargs)
        except Exception as exc:  # pragma: no cover - signature drift in the sibling module
            LOG.warning("could not build prism.decision.optimize.AllocationResult (%s); using local", exc)
            result = None
    if result is None:
        result = FairnessAllocationResult(
            assignment=arms,
            total_cost=total_cost,
            expected_incremental_value=total_value,
            n_treated=n_treated,
            per_arm=per_arm,
            dual_lambda=dual_lambda,
            method=method,
            budget=None if budget is None else float(budget),
        )
    _stash_protected(result, protected)
    return result


def _stash_protected(result: Any, protected: pd.Series | None) -> None:
    """Attach the protected attribute to a result so :func:`price_of_fairness` can find it."""
    if protected is None:
        return
    try:
        object.__setattr__(result, "_protected", protected)
    except Exception:  # pragma: no cover - frozen/slotted dataclass in the sibling module
        LOG.debug("could not stash the protected attribute on %s", type(result).__name__)


# ======================================================================================
# 1. Audit
# ======================================================================================
def policy_fairness_audit(
    assignment: np.ndarray,
    protected: pd.Series,
    outcome: np.ndarray | None = None,
    net_value: np.ndarray | None = None,
    *,
    n_arms: int | None = None,
) -> pd.DataFrame:
    """Audit *who receives offers* under a policy, one row per protected group.

    A retention budget is a scarce benefit, so every metric here is about **allocation** (who is
    served) rather than about error rates on a denial. Read the table as: "this group's customers
    were bought back at rate X, that is Y points off the book-wide rate, and among the customers an
    offer would have paid for, they were served Z points behind the best-served group".

    Parameters
    ----------
    assignment : numpy.ndarray
        ``(n,)`` chosen arm per customer; ``0`` means no offer, ``>0`` means treated.
    protected : pandas.Series
        ``(n,)`` protected-group membership. Any dtype. **Missing values never raise**: they are
        folded into their own ``"unknown"`` group (see Notes). If this is a categorical with unused
        categories, those empty groups still get a row (``n = 0``, rates ``NaN``).
    outcome : numpy.ndarray, optional
        ``(n,)`` realised outcome per customer (retention, revenue, RMST -- whatever the caller is
        auditing). Adds a ``mean_outcome`` column.
    net_value : numpy.ndarray, optional
        Either ``(n,)`` per-customer net value of the offer, or the ``(n, n_arms)`` matrix from
        :func:`prism.decision.economics.expected_net_value`. Adds ``mean_net_value`` and is
        **required** for ``equal_opportunity_diff``.
    n_arms : int, optional
        Number of arms to report shares for. Defaults to ``max(N_ARMS, assignment.max() + 1)``.

    Returns
    -------
    pandas.DataFrame
        One row per group plus a final ``"OVERALL"`` row, with columns:

        ``group``
            Group label; ``"unknown"`` for missing, ``"OVERALL"`` for the pooled row.
        ``n``
            Group size (``0`` is allowed and handled).
        ``treat_rate``
            Share of the group receiving any offer. ``NaN`` for an empty group.
        ``mean_net_value``
            Present only when ``net_value`` is given. Mean net value **of the arm actually
            assigned** when a matrix was supplied, else the mean of the supplied vector.
        ``mean_outcome``
            Present only when ``outcome`` is given.
        ``demographic_parity_diff``
            ``treat_rate - overall treat_rate``. Zero by construction on the ``OVERALL`` row.
        ``demographic_parity_ratio``
            ``treat_rate / max(group treat_rate)``. ``1.0`` for the best-served group; the minimum
            over groups is the :func:`disparate_impact_ratio`.
        ``equal_opportunity_diff``
            Restricted to customers who **would benefit** (``net_value > 0``): this group's treat
            rate minus the best-served group's treat rate, so it is ``<= 0`` and ``0.0`` for the
            best-served group. **``NaN`` when ``net_value`` is not supplied** -- there is no way to
            know who would have benefited without it -- and ``NaN`` for a group with no
            beneficiaries.
        ``share_arm_0 ... share_arm_<n_arms-1>``
            Within-group distribution over arms, including ``share_arm_0`` (no offer). Sums to 1.

    Notes
    -----
    *Missing protected values.* Rows with a missing group are reported as the group ``"unknown"``
    rather than dropped. Dropping them would quietly shrink the audited population, and
    "demographics we failed to record" is rarely a random sample of the book -- it is often exactly
    the segment an audit should be looking at.

    *Equal opportunity for a benefit.* The classifier definition conditions on the true label. The
    label analogue for a scarce benefit is "an offer to this customer would have paid for itself",
    i.e. ``net_value > 0``. When a ``(n, n_arms)`` matrix is supplied, a customer counts as a
    beneficiary if **any non-control arm** has positive net value.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> a = np.array([1, 0, 1, 0, 0, 0])
    >>> g = pd.Series(["a", "a", "a", "b", "b", "b"])
    >>> audit = policy_fairness_audit(a, g)
    >>> audit.loc[audit.group == "a", "treat_rate"].item()
    0.6666666666666666
    """
    arms = _as_int_vector(assignment)
    n = int(arms.shape[0])
    labels, categories, _ = _protected_series(protected, n)

    arms_max = int(arms.max()) + 1 if n else 1
    n_arms_eff = int(n_arms) if n_arms is not None else max(N_ARMS, arms_max)
    if arms_max > n_arms_eff:
        raise ValueError(f"assignment contains arm {arms_max - 1} but n_arms={n_arms_eff}")

    treated = (arms > 0).astype(np.float64)

    out_vec: np.ndarray | None = None
    if outcome is not None:
        out_vec = _as_float_vector(outcome, n, "outcome")
        # A NaN anywhere turns that group's mean into NaN. That is the honest arithmetic, but it
        # must be said out loud: a silently NaN fairness metric reads as "nothing to see here".
        _warn_non_finite(out_vec, "outcome")

    realised_nv: np.ndarray | None = None
    benefit: np.ndarray | None = None
    if net_value is not None:
        NV = np.asarray(net_value, dtype=np.float64)
        _warn_non_finite(NV, "net_value")
        if NV.ndim == 1:
            realised_nv = _as_float_vector(NV, n, "net_value")
            benefit = realised_nv > 0.0
        elif NV.ndim == 2:
            if NV.shape[0] != n:
                raise ValueError(f"net_value has {NV.shape[0]} rows, expected {n}")
            realised_nv = NV[np.arange(n), np.clip(arms, 0, NV.shape[1] - 1)]
            benefit = (NV[:, 1:] > 0.0).any(axis=1) if NV.shape[1] > 1 else np.zeros(n, dtype=bool)
        else:
            raise ValueError(f"net_value must be 1-D or 2-D, got shape {NV.shape}")

    masks = {g: (labels.to_numpy() == g) for g in categories}
    sizes = {g: int(m.sum()) for g, m in masks.items()}
    rates = {g: (float(treated[masks[g]].mean()) if sizes[g] else np.nan) for g in categories}

    overall_rate = float(treated.mean()) if n else np.nan
    finite_rates = [r for r in rates.values() if np.isfinite(r)]
    max_rate = max(finite_rates) if finite_rates else np.nan

    # Equal opportunity: restricted to customers an offer would actually pay for.
    eo_rates: dict[str, float] = dict.fromkeys(categories, np.nan)
    eo_overall = np.nan
    eo_best = np.nan
    if benefit is not None:
        for g in categories:
            m = masks[g] & benefit
            eo_rates[g] = float(treated[m].mean()) if int(m.sum()) else np.nan
        eo_overall = float(treated[benefit].mean()) if int(benefit.sum()) else np.nan
        finite_eo = [r for r in eo_rates.values() if np.isfinite(r)]
        eo_best = max(finite_eo) if finite_eo else np.nan

    rows: list[dict[str, Any]] = []
    for g in categories + [OVERALL_LABEL]:
        is_overall = g == OVERALL_LABEL
        m = np.ones(n, dtype=bool) if is_overall else masks[g]
        size = n if is_overall else sizes[g]
        rate = overall_rate if is_overall else rates[g]
        eo_rate = eo_overall if is_overall else eo_rates[g]

        row: dict[str, Any] = {"group": g, "n": int(size), "treat_rate": rate}
        if realised_nv is not None:
            row["mean_net_value"] = float(realised_nv[m].mean()) if size else np.nan
        if out_vec is not None:
            row["mean_outcome"] = float(out_vec[m].mean()) if size else np.nan
        row["demographic_parity_diff"] = (
            0.0 if is_overall else (rate - overall_rate if np.isfinite(rate) else np.nan)
        )
        row["demographic_parity_ratio"] = (
            rate / max_rate if (np.isfinite(rate) and np.isfinite(max_rate) and max_rate > 0) else np.nan
        )
        row["equal_opportunity_diff"] = (
            eo_rate - eo_best if (np.isfinite(eo_rate) and np.isfinite(eo_best)) else np.nan
        )
        for a in range(n_arms_eff):
            row[f"share_arm_{a}"] = float((arms[m] == a).mean()) if size else np.nan
        rows.append(row)

    audit = pd.DataFrame(rows)
    audit["n"] = audit["n"].astype("int64")
    return audit.reset_index(drop=True)


def disparate_impact_ratio(assignment: np.ndarray, protected: pd.Series) -> float:
    """Ratio of the least-served group's treat rate to the best-served group's.

    .. math:: DI = \\frac{\\min_g P(\\text{offer} \\mid G = g)}{\\max_g P(\\text{offer} \\mid G = g)}

    **On the 0.8 threshold.** The "four-fifths rule" -- flag a selection process whose ratio falls
    below 0.8 -- comes from the US EEOC *Uniform Guidelines on Employee Selection Procedures* (29
    CFR 1607.4(D)), where it is a rule of thumb for when an employer's hiring or promotion process
    warrants a closer look. It is being **borrowed** here, and the borrowing is imperfect in three
    ways worth stating plainly:

    1. It governs **employment** decisions, not marketing offers. Nothing in that guidance makes
       0.8 a legal standard for who receives a retention voucher.
    2. It was written for a **gatekeeping harm** (a job denied). A retention budget withholds a
       *benefit* from a customer who is no worse off than before, which is a real allocative
       concern but not the same kind of harm, and the same number should not carry the same weight.
    3. Even in its home domain 0.8 is a screening heuristic, not a verdict: it is famously
       sensitive to sample size and to base rates, and passing it is not a defence.

    Treat the number as a **tripwire that starts a conversation**, not as a compliance gate. What
    it is genuinely good at is being a single scalar that trends over time and across policies --
    which is why :func:`price_of_fairness` reports it before and after.

    Parameters
    ----------
    assignment : numpy.ndarray
        ``(n,)`` chosen arm per customer; ``0`` means no offer.
    protected : pandas.Series
        ``(n,)`` group membership. Missing values form their own ``"unknown"`` group and are
        **included** in the ratio, since an unrecorded-demographic group being starved of offers is
        a finding, not noise.

    Returns
    -------
    float
        The ratio in ``[0, 1]``. Empty groups are excluded. Returns ``1.0`` when there is a single
        group, or when nobody is treated at all (every group has rate 0, so allocation is
        trivially equal -- the policy spends nothing on anyone). Returns ``nan`` for empty input.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> a = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    >>> g = pd.Series(["x"] * 4 + ["y"] * 4)
    >>> round(disparate_impact_ratio(a, g), 3)
    0.5
    """
    arms = _as_int_vector(assignment)
    n = int(arms.shape[0])
    if n == 0:
        return float("nan")
    labels, categories, _ = _protected_series(protected, n)
    treated = (arms > 0).astype(np.float64)
    lab = labels.to_numpy()

    rates: list[float] = []
    for g in categories:
        m = lab == g
        if int(m.sum()) == 0:
            continue  # a group with no members has no treat rate to compare
        rates.append(float(treated[m].mean()))
    if not rates:
        return float("nan")
    hi = max(rates)
    if hi <= 0.0:
        LOG.info("no customer is treated; disparate impact is trivially 1.0")
        return 1.0
    return float(min(rates) / hi)


# ======================================================================================
# 2. Allocation
# ======================================================================================
def unconstrained_allocation(
    net_value: np.ndarray,
    cost: np.ndarray,
    budget: float,
    *,
    protected: pd.Series | None = None,
) -> Any:
    """Solve the budgeted allocation with **no** fairness constraint -- the baseline to price.

    This is the same Lagrangian relaxation :func:`prism.decision.optimize.lagrangian_allocate`
    implements (each customer takes ``argmax_a (v_a - lambda c_a)``, ``lambda`` set so the bill
    fits), solved here by an equivalent and exactly deterministic head-count bisection so that
    :func:`reweight_for_parity` and its baseline are guaranteed to come from the same engine. That
    matters: a price of fairness computed against a *different* solver would be contaminated by the
    solvers' disagreement.

    Parameters
    ----------
    net_value : numpy.ndarray
        ``(n, n_arms)`` (or ``(n,)``) net value per customer per arm, already net of offer cost.
    cost : float or numpy.ndarray
        Budget-consuming spend: scalar, ``(n_arms,)`` or ``(n, n_arms)``.
    budget : float
        Total spend available.
    protected : pandas.Series, optional
        Stashed on the result so :func:`price_of_fairness` can compute disparate impact without
        being handed the groups again.

    Returns
    -------
    AllocationResult
        From ``prism.decision.optimize`` when importable, else
        :class:`FairnessAllocationResult`.
    """
    V, C = _coerce_value_cost(net_value, cost)
    budget = float(budget)
    block = _Block(V, C)
    # Cap the head-count at the number of customers who have *any* positive-value arm. Without
    # this cap a non-binding budget drags every remaining customer into treatment purely because
    # the money is there, and each of them subtracts value: arm 0 is always available at value 0,
    # so no optimal unconstrained policy ever treats a customer whose every arm loses money.
    k = block.max_k_within(budget, k_hi=block.n_positive) if budget > 0 else 0
    arms, _, _, lam_k = block.solve(k)
    arms = _spend_slack(block, arms, budget, allow_new_treated=True)
    # lambda* is the hurdle rate. When every profitable customer is already served the budget is
    # not binding at the margin and the shadow price is zero, matching
    # :func:`prism.decision.optimize.lagrangian_allocate`.
    lam = 0.0 if k >= block.n_positive else float(lam_k)
    prot = None
    if protected is not None:
        prot, _, _ = _protected_series(protected, V.shape[0])
    return _make_result(arms, V, C, float(lam), "lagrangian_unconstrained", prot, budget=budget)


def reweight_for_parity(
    net_value: np.ndarray,
    cost: np.ndarray,
    budget: float,
    protected: pd.Series,
    tolerance: float = 0.1,
    *,
    max_redistribution_rounds: int = 12,
) -> Any:
    """Allocate a budget under a per-group treat-rate parity constraint.

    Every protected group's treat rate must land within ``tolerance`` (absolute, in rate points) of
    the **realised overall** treat rate. The solver is a **per-group Lagrangian**:

    1. **Solve unconstrained.** Run the plain budgeted Lagrangian over everybody to learn the
       overall treat rate ``p*`` the money supports. This is the target the groups are held to; it
       is not imposed from outside, it is what the budget already implies.
    2. **Set per-group head-counts.** Each group ``g`` gets a parity floor of
       ``round(p_target * n_g)`` customers, its own shadow price, and therefore its own budget
       share -- the share is an *output* of hitting the head-count, not an input guessed in
       advance. Within a group, customers still enter in value-per-cost order and still choose
       their own arm, so the allocation is efficient *given* the constraint.
    3. **Back off if the floors do not fit.** Group-level spend is monotone in the head-count, so a
       bisection on ``p_target`` in ``[0, p*]`` finds the highest parity level the budget can
       actually pay for. Parity never overdraws the budget; it lowers the whole book instead.
    4. **Redistribute the slack.** Whatever the floors leave unspent is handed, in rounds, to the
       groups that still have **positive-value candidates**, in descending order of marginal value
       density -- but only up to ``p_target + tolerance/2``. So leftover money goes where it earns
       most without re-opening the gap it was just closed to prevent.

    Why the band is enforced at ``tolerance / 2``: the realised overall rate is a weighted average
    of the group rates. Keeping every group within ``tolerance/2`` of ``p_target`` puts the average
    inside that same interval, which bounds every group-to-overall gap by ``tolerance``. Enforcing
    ``tolerance`` directly against ``p_target`` would only bound it by ``2 * tolerance``.

    A group whose positive-value candidates run out before its floor is met is served anyway, at a
    **negative shadow price** -- offers that do not pay for themselves. That is not a bug; it is the
    parity constraint doing the thing it was asked to do, and it is exactly the spend that
    :func:`price_of_fairness` puts a number on.

    Parameters
    ----------
    net_value : numpy.ndarray
        ``(n, n_arms)`` net value per customer per arm (arm 0 = do nothing, normalised to 0), or
        ``(n,)`` for a single offer.
    cost : float or numpy.ndarray
        Budget-consuming spend: scalar, ``(n_arms,)`` or ``(n, n_arms)``.
    budget : float
        Total spend available.
    protected : pandas.Series
        ``(n,)`` group membership. Missing values form an ``"unknown"`` group which is constrained
        like any other.
    tolerance : float, default 0.1
        Maximum absolute gap, in rate points, between any group's treat rate and the overall treat
        rate. ``0.0`` demands exact parity up to integer rounding.
    max_redistribution_rounds : int, default 12
        Cap on the leftover-budget redistribution rounds.

    Returns
    -------
    AllocationResult
        ``prism.decision.optimize.AllocationResult`` when that module is importable, otherwise a
        :class:`FairnessAllocationResult` with identical fields. **The import is lazy, inside this
        function**, because ``prism/decision/optimize.py`` is authored in parallel with this file:
        a module-level import would break ``import prism.decision.fairness`` outright whenever the
        sibling is absent or mid-edit. ``dual_lambda`` is the treated-count-weighted mean of the
        per-group shadow prices -- under parity each group has its own price, so one scalar cannot
        be exact. A negative mean proves some group is subsidised, but a positive mean does not
        prove none is: a large, profitable group can outvote a subsidised one. **The per-group
        shadow prices, head-counts and spend are logged individually at INFO**, and that log is the
        place to look for which group is being served below break-even.

    Notes
    -----
    Integer head-counts mean a group of size ``n_g`` can only hit rates in steps of ``1 / n_g``, so
    the band is only meaningful when ``n_g >= 1 / tolerance`` (with the default, 10 members). Small
    groups are logged as a warning rather than silently reported as compliant.

    What is actually enforced is therefore ``tolerance + 0.5 / n_g + 0.5 * G / n`` for ``G``
    groups: the group's own rounding, plus the rounding those ``G`` head-counts induce in the
    overall rate the group is being compared against. Both terms are real and unavoidable in whole
    customers, and with ``tolerance = 0`` they are the entire band -- "exact parity" means the
    nearest attainable integer allocation, not a gap of literally zero. At realistic group sizes
    the slack is negligible (three groups in a 20 000-customer book: about 2e-4).

    The result is a deterministic function of its inputs -- no sampling anywhere -- so two calls
    with identical arguments return bit-identical assignments.
    """
    V, C = _coerce_value_cost(net_value, cost)
    n, n_arms = V.shape
    budget = float(budget)
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if budget < 0.0:
        raise ValueError("budget must be non-negative")
    labels, categories, _ = _protected_series(protected, n)
    lab = labels.to_numpy()

    # ---- step 1: unconstrained solve -> the treat rate the budget implies -------------
    whole = _Block(V, C)
    # Capped at the customers who have a positive-value arm: the rate the budget "implies" is the
    # rate at which the money still *buys* something. Without the cap a non-binding budget makes
    # p_star = 1.0 and parity then forces an offer on everybody, including customers every arm
    # loses money on -- a target that is neither unconstrained-optimal nor what parity asks for.
    k_unc = whole.max_k_within(budget, k_hi=whole.n_positive) if budget > 0 else 0
    p_star = float(k_unc / n) if n else 0.0

    present = [g for g in categories if np.any(lab == g)]
    idx_by_group = {g: np.flatnonzero(lab == g) for g in present}
    blocks = {g: _Block(V[idx_by_group[g]], C[idx_by_group[g]]) for g in present}
    for g in present:
        n_g = blocks[g].n
        if tolerance > 0 and n_g * tolerance < 1.0:
            LOG.warning(
                "group '%s' has only %d members; with tolerance=%.3f its treat rate moves in steps "
                "of %.3f, coarser than the band itself",
                g,
                n_g,
                tolerance,
                1.0 / n_g,
            )

    def _floor_counts(p: float) -> dict[str, int]:
        return {g: int(min(blocks[g].n, np.floor(p * blocks[g].n + 0.5))) for g in present}

    def _floor_cost(p: float) -> float:
        return float(sum(blocks[g].cost(k) for g, k in _floor_counts(p).items()))

    # ---- step 3 (applied early): the highest parity level the budget can pay for ------
    p_target = p_star
    if p_star > 0.0 and _floor_cost(p_star) > budget + 1e-6:
        lo, hi = 0.0, p_star
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _floor_cost(mid) <= budget + 1e-6:
                lo = mid
            else:
                hi = mid
        p_target = lo
        if p_star - p_target > 1e-9:
            LOG.info(
                "parity floors at the unconstrained rate %.4f cost more than the budget; "
                "backing the whole book down to %.4f",
                p_star,
                p_target,
            )

    # ---- step 2: per-group head-counts ------------------------------------------------
    counts = _floor_counts(p_target)
    spent = float(sum(blocks[g].cost(k) for g, k in counts.items()))
    leftover = budget - spent

    # ---- step 4: redistribute the slack to groups with positive-value candidates ------
    half = tolerance / 2.0
    caps = {
        g: int(
            max(
                counts[g],
                min(blocks[g].n, int(np.floor((p_target + half) * blocks[g].n + 1e-9)), blocks[g].n_positive),
            )
        )
        for g in present
    }
    for _ in range(int(max_redistribution_rounds)):
        eligible = [g for g in present if counts[g] < caps[g]]
        if not eligible or leftover <= 1e-9:
            break

        def _density(g: str) -> float:
            d_cost = blocks[g].cost(caps[g]) - blocks[g].cost(counts[g])
            d_value = blocks[g].value(caps[g]) - blocks[g].value(counts[g])
            return float(d_value / d_cost) if d_cost > 1e-12 else float("inf")

        eligible.sort(key=lambda g: (-_density(g), g))
        changed = False
        for g in eligible:
            if leftover <= 1e-9:
                break
            blk = blocks[g]
            base = blk.cost(counts[g])
            new_k = blk.max_k_within(base + leftover, k_lo=counts[g], k_hi=caps[g])
            if new_k > counts[g]:
                delta = blk.cost(new_k) - base
                if delta <= leftover + 1e-9:
                    counts[g] = new_k
                    leftover -= delta
                    spent += delta
                    changed = True
        if not changed:
            break

    # ---- assemble -----------------------------------------------------------------------
    arms = np.zeros(n, dtype=np.int64)
    lam_num = 0.0
    lam_den = 0
    for g in present:
        local_arms, cost_g, _, lam_g = blocks[g].solve(counts[g])
        arms[idx_by_group[g]] = local_arms
        if counts[g] > 0:
            lam_num += float(lam_g) * counts[g]
            lam_den += counts[g]
        # The per-group price is the audit trail a blended dual_lambda cannot carry: a negative
        # entry names exactly which group is being served below break-even, and how deep.
        LOG.info(
            "  group '%s': n=%d treated=%d rate=%.4f spend=%.2f shadow_price=%+.4f%s",
            g,
            blocks[g].n,
            counts[g],
            counts[g] / blocks[g].n if blocks[g].n else np.nan,
            cost_g,
            float(lam_g),
            "  <- subsidised below break-even" if lam_g < 0 else "",
        )
    dual_lambda = float(lam_num / lam_den) if lam_den else None

    # ---- safety net: verify the band against the *realised* overall rate ----------------
    total_cost = float(C[np.arange(n), arms].sum())
    if total_cost > budget + 1e-6:  # pragma: no cover - monotonicity makes this unreachable
        LOG.error("parity allocation overshot the budget by %.4f; trimming", total_cost - budget)
    p_real = float(np.mean(arms > 0)) if n else 0.0

    def _band(g: str) -> float:
        """Enforceable band for group ``g``: the constraint plus its integer granularity.

        Two separate roundings sit between a head-count and a rate, and the band has to allow
        for both or it asks for a point that does not exist in whole customers:

        * A group of ``n_g`` customers moves its rate in steps of ``1 / n_g``, so it can sit up
          to ``0.5 / n_g`` from the target. Trimming a customer to chase a smaller gap would only
          move the overall rate and re-open it.
        * The **overall** rate is itself ``sum_g counts_g / n``, so each group's rounding drags it
          too: up to ``0.5 * G / n`` in total for ``G`` groups.

        The two add. Ignoring the second term makes ``tolerance = 0`` unsatisfiable whenever the
        group sizes are not commensurate -- the solver would be reported as violating a band that
        no integer allocation can meet.
        """
        return tolerance + 0.5 / blocks[g].n + 0.5 * len(present) / n + 1e-9

    for _ in range(10):
        offenders = [
            g
            for g in present
            if counts[g] / blocks[g].n - p_real > _band(g)
        ]
        if not offenders:
            break
        for g in offenders:  # pragma: no cover - construction already bounds the gap
            target = int(np.floor((p_real + tolerance) * blocks[g].n + 1e-9))
            LOG.warning("trimming group '%s' from %d to %d to restore the tolerance band", g, counts[g], target)
            counts[g] = max(0, min(counts[g], target))
            local_arms, _, _, _ = blocks[g].solve(counts[g])
            arms[idx_by_group[g]] = local_arms
        p_real = float(np.mean(arms > 0))
    # ---- spend whatever slack the integer head-counts left on the table ----------------
    # Two kinds of leftover value. (a) Arm upgrades for the already-treated cost nothing in
    # fairness terms, because who is treated does not move. (b) Whole additional customers do
    # move the rates, so each one is vetted against the band before it is accepted -- which
    # matters most when the budget is so tight that the single densest customer is unaffordable
    # and the head-count ordering therefore stops at zero while cheaper, profitable customers
    # are still sitting there. Ordering by value density keeps the additions worth making.
    group_of = {i: g for g in present for i in idx_by_group[g].tolist()}
    live = {g: counts[g] for g in present}
    n_live = int(np.sum(arms > 0))

    def _band_guard(i: int, new_arm: int, old_arm: int) -> bool:
        """Accept an add/drop only if every group stays inside its tolerance band."""
        nonlocal n_live
        g = group_of.get(int(i))
        if g is None:  # pragma: no cover - every row belongs to a present group
            return False
        step = 1 if new_arm > 0 else -1
        trial = dict(live)
        trial[g] += step
        if not 0 <= trial[g] <= blocks[g].n:
            return False
        p_trial = (n_live + step) / n if n else 0.0
        if any(abs(trial[h] / blocks[h].n - p_trial) > _band(h) for h in present):
            return False
        live[g] = trial[g]
        n_live += step
        return True

    arms = _spend_slack(_Block(V, C), arms, budget, allow_new_treated=True, guard=_band_guard)
    for g in present:
        counts[g] = int(np.sum(arms[idx_by_group[g]] > 0))
    p_real = float(np.mean(arms > 0)) if n else 0.0

    # ---- final, post-slack band check on the assignment actually returned ---------------
    residual = [g for g in present if blocks[g].n and abs(counts[g] / blocks[g].n - p_real) > _band(g)]
    if residual:  # pragma: no cover - only reachable for groups smaller than 1/tolerance
        LOG.warning(
            "groups %s remain outside the +/-%.3f band (too few members for the band to be "
            "reachable in whole customers)",
            residual,
            tolerance,
        )

    LOG.info(
        "parity allocation: overall treat rate %.4f (unconstrained %.4f), %d groups, "
        "spend %.2f of %.2f budget",
        p_real,
        p_star,
        len(present),
        float(C[np.arange(n), arms].sum()),
        budget,
    )
    return _make_result(
        arms,
        V,
        C,
        dual_lambda,
        f"reweight_for_parity(tolerance={tolerance:g})",
        labels,
        budget=budget,
    )


# ======================================================================================
# 3. The price tag
# ======================================================================================
def price_of_fairness(
    unconstrained: Any,
    constrained: Any,
    protected: pd.Series | None = None,
) -> dict[str, float]:
    """Report, in currency, what the parity constraint cost -- and what it bought.

    Fairness arguments stall when one side has a principle and the other has a P&L. This turns both
    into the same units: *we gave up this much incremental value, and the least-served group's
    access went from this ratio to that one.* A reader can then decide whether the trade is worth
    it, which is a decision; without the number, it is only an argument.

    Parameters
    ----------
    unconstrained : AllocationResult
        The value-maximising allocation under the same budget, e.g. from
        :func:`unconstrained_allocation` or :func:`prism.decision.optimize.lagrangian_allocate`.
    constrained : AllocationResult
        The parity-constrained allocation from :func:`reweight_for_parity`.
    protected : pandas.Series, optional
        Group membership for the disparate-impact terms. If omitted, it is recovered from the
        ``_protected`` attribute that :func:`reweight_for_parity` stashes on its result (the two
        allocations cover the same population, so one series serves both). Without either, the
        disparate-impact keys are ``nan`` and the value keys are still computed.

    Returns
    -------
    dict of str to float
        ``value_unconstrained``
            Expected incremental value of the unconstrained policy.
        ``value_constrained``
            Expected incremental value of the parity-constrained policy.
        ``value_forgone``
            ``value_unconstrained - value_constrained``: the price of fairness, in currency. Under
            a common budget this is ``>= 0`` up to Lagrangian integrality slack, because the
            constrained policy is feasible for the unconstrained problem.
        ``pct_value_forgone``
            ``value_forgone`` as a **percentage** (0-100) of ``value_unconstrained``; ``nan`` when
            the unconstrained value is non-positive.
        ``disparate_impact_before``
            :func:`disparate_impact_ratio` of the unconstrained assignment.
        ``disparate_impact_after``
            :func:`disparate_impact_ratio` of the constrained assignment.

    Examples
    --------
    >>> import logging, numpy as np, pandas as pd
    >>> logging.getLogger("prism").setLevel(logging.ERROR)   # quiet the progress logs
    >>> nv = np.column_stack([np.zeros(8), np.array([50., 45., 40., 35., 6., 5., 4., 3.])])
    >>> c = np.column_stack([np.zeros(8), np.full(8, 10.0)])
    >>> g = pd.Series(["hi"] * 4 + ["lo"] * 4)
    >>> u = unconstrained_allocation(nv, c, 40.0, protected=g)
    >>> f = reweight_for_parity(nv, c, 40.0, g, tolerance=0.05)
    >>> pof = price_of_fairness(u, f)
    >>> pof["value_forgone"] > 0 and pof["disparate_impact_after"] > pof["disparate_impact_before"]
    True
    """
    v_unc = float(getattr(unconstrained, "expected_incremental_value", np.nan))
    v_con = float(getattr(constrained, "expected_incremental_value", np.nan))
    forgone = v_unc - v_con
    pct = float(100.0 * forgone / v_unc) if np.isfinite(v_unc) and v_unc > 0 else float("nan")

    prot = protected
    if prot is None:
        prot = getattr(constrained, "_protected", None)
    if prot is None:
        prot = getattr(unconstrained, "_protected", None)

    di_before = float("nan")
    di_after = float("nan")
    if prot is not None:
        a_unc = getattr(unconstrained, "assignment", None)
        a_con = getattr(constrained, "assignment", None)
        if a_unc is not None:
            di_before = disparate_impact_ratio(a_unc, prot)
        if a_con is not None:
            di_after = disparate_impact_ratio(a_con, prot)
    else:
        LOG.info("no protected attribute available; disparate-impact terms reported as nan")

    if np.isfinite(forgone) and forgone < -1e-6:
        LOG.warning(
            "constrained value exceeds unconstrained by %.4f; check that both used the same budget",
            -forgone,
        )

    return {
        "value_unconstrained": v_unc,
        "value_constrained": v_con,
        "value_forgone": forgone,
        "pct_value_forgone": pct,
        "disparate_impact_before": di_before,
        "disparate_impact_after": di_after,
    }


# ======================================================================================
# Smoke test
# ======================================================================================
if __name__ == "__main__":  # pragma: no cover - smoke test
    import time

    from prism.data.schema import ARM_COSTS, ARM_REDEMPTION
    from prism.utils.seeds import as_rng

    t0 = time.perf_counter()
    rng = as_rng(20240923)
    n = 20_000

    # Three protected groups whose net-value distributions genuinely differ, so the
    # value-maximising optimum really is unequal and parity really has to cost something.
    group_names = np.array(["north", "south", "east"])
    groups = pd.Series(rng.choice(group_names, size=n, p=[0.50, 0.30, 0.20]), name="region")
    g = groups.to_numpy()
    loc = np.select([g == "north", g == "south"], [95.0, 60.0], default=30.0)
    scale = np.select([g == "north", g == "south"], [55.0, 45.0], default=30.0)

    tau = rng.normal(loc, scale)                       # value uplift of the strongest offer
    # Concave strength-vs-cost ladder, so every arm is optimal for some band of tau and the
    # audit's share_arm_* columns actually exercise the whole menu.
    arm_strength = np.array([0.0, 0.35, 0.75, 0.85])   # how much of tau each arm delivers
    exp_cost = np.asarray(ARM_COSTS) * np.asarray(ARM_REDEMPTION)  # cost only if redeemed
    value = tau[:, None] * arm_strength[None, :]
    nv = value - exp_cost[None, :]
    nv[:, 0] = 0.0
    cost_mat = np.tile(exp_cost.reshape(1, -1), (n, 1))

    outcome = (rng.random(n) < np.clip(0.55 + tau / 600.0, 0.05, 0.95)).astype(np.float64)
    budget = 120_000.0

    unc = unconstrained_allocation(nv, cost_mat, budget, protected=groups)
    con = reweight_for_parity(nv, cost_mat, budget, groups, tolerance=0.1)

    audit_before = policy_fairness_audit(unc.assignment, groups, outcome=outcome, net_value=nv)
    audit_after = policy_fairness_audit(con.assignment, groups, outcome=outcome, net_value=nv)
    pof = price_of_fairness(unc, con)

    show = [
        "group", "n", "treat_rate", "mean_net_value", "mean_outcome",
        "demographic_parity_diff", "demographic_parity_ratio", "equal_opportunity_diff",
    ]
    pd.set_option("display.width", 160)
    print("\n=== BEFORE: value-maximising allocation (no fairness constraint) ===")
    print(audit_before[show].round(4).to_string(index=False))
    print(audit_before[[c for c in audit_before.columns if c.startswith("share_arm_")]].round(4).to_string(index=False))
    print("\n=== AFTER: parity-constrained allocation (tolerance = 0.10) ===")
    print(audit_after[show].round(4).to_string(index=False))
    print(audit_after[[c for c in audit_after.columns if c.startswith("share_arm_")]].round(4).to_string(index=False))

    print("\n=== PRICE OF FAIRNESS ===")
    for key, val in pof.items():
        print(f"  {key:26s} {val:14.4f}")
    print(f"  {'budget':26s} {budget:14.4f}")
    print(f"  {'spend_unconstrained':26s} {unc.total_cost:14.4f}")
    print(f"  {'spend_constrained':26s} {con.total_cost:14.4f}")
    print(f"  {'dual_lambda_constrained':26s} {float(con.dual_lambda):14.4f}")
    print(f"  {'result_class':26s} {type(con).__name__:>14s}")

    # ---------------------------------- assertions -------------------------------------
    di_before = pof["disparate_impact_before"]
    di_after = pof["disparate_impact_after"]
    assert di_after > di_before, f"parity did not improve disparate impact: {di_before} -> {di_after}"
    assert 0.0 <= di_before <= 1.0 and 0.0 <= di_after <= 1.0

    assert con.total_cost <= budget + 1e-6, f"constrained policy overspent: {con.total_cost} > {budget}"
    assert unc.total_cost <= budget + 1e-6, f"unconstrained policy overspent: {unc.total_cost}"

    overall_rate = float(audit_after.loc[audit_after.group == OVERALL_LABEL, "treat_rate"].item())
    per_group = audit_after[audit_after.group != OVERALL_LABEL]
    worst = float((per_group["treat_rate"] - overall_rate).abs().max())
    assert worst <= 0.1 + 1e-9, f"a group fell outside the tolerance band: max gap {worst:.4f}"
    before_overall = float(audit_before.loc[audit_before.group == OVERALL_LABEL, "treat_rate"].item())
    before_worst = float(
        (audit_before[audit_before.group != OVERALL_LABEL]["treat_rate"] - before_overall).abs().max()
    )
    assert before_worst > worst, "the unconstrained policy should be the less equal one"

    assert pof["value_forgone"] >= 0.0, f"value_forgone must be non-negative, got {pof['value_forgone']}"
    assert 0.0 <= pof["pct_value_forgone"] <= 100.0
    assert pof["value_constrained"] <= pof["value_unconstrained"] + 1e-6

    # audit-table contract
    assert list(audit_after["group"])[-1] == OVERALL_LABEL
    assert int(audit_after.loc[audit_after.group == OVERALL_LABEL, "n"].item()) == n
    share_cols = [c for c in audit_after.columns if c.startswith("share_arm_")]
    assert len(share_cols) == N_ARMS
    assert np.allclose(audit_after[share_cols].sum(axis=1).to_numpy(), 1.0)
    assert float(audit_after.loc[audit_after.group == OVERALL_LABEL, "demographic_parity_diff"].item()) == 0.0
    assert float(per_group["demographic_parity_ratio"].max()) == 1.0
    assert float(per_group["equal_opportunity_diff"].max()) == 0.0
    assert bool((per_group["equal_opportunity_diff"] <= 1e-12).all())
    assert per_group["equal_opportunity_diff"].notna().all()

    # equal opportunity needs net_value; without it the column is NaN, not an exception
    no_nv = policy_fairness_audit(con.assignment, groups)
    assert no_nv["equal_opportunity_diff"].isna().all()
    assert "mean_net_value" not in no_nv.columns and "mean_outcome" not in no_nv.columns

    # determinism
    con2 = reweight_for_parity(nv, cost_mat, budget, groups, tolerance=0.1)
    assert np.array_equal(con.assignment, con2.assignment), "reweight_for_parity is not deterministic"

    # NaN protected values -> an 'unknown' group, never an exception
    dirty = groups.astype(object).copy()
    dirty.iloc[:500] = np.nan
    dirty_audit = policy_fairness_audit(con.assignment, dirty, net_value=nv)
    assert UNKNOWN_GROUP in set(dirty_audit["group"]), "missing groups must surface as 'unknown'"
    assert int(dirty_audit.loc[dirty_audit.group == UNKNOWN_GROUP, "n"].item()) == 500
    assert np.isfinite(disparate_impact_ratio(con.assignment, dirty))

    # a declared-but-empty group gets a row rather than blowing up
    cat = pd.Series(pd.Categorical(groups, categories=list(group_names) + ["west"]))
    cat_audit = policy_fairness_audit(con.assignment, cat, net_value=nv)
    west = cat_audit[cat_audit.group == "west"]
    assert len(west) == 1 and int(west["n"].item()) == 0 and bool(west["treat_rate"].isna().item())

    # degenerate edges
    assert disparate_impact_ratio(np.zeros(10, dtype=int), pd.Series(["a"] * 5 + ["b"] * 5)) == 1.0
    zero = reweight_for_parity(nv[:500], cost_mat[:500], 0.0, groups.iloc[:500])
    assert int(zero.n_treated) == 0 and float(zero.total_cost) == 0.0

    # ---- a NON-BINDING budget must not buy value-destroying offers --------------------
    # The trap: a head-count solver that just takes "the largest k the budget affords" will drag
    # every remaining customer into treatment once the money stops binding, and each customer
    # whose every arm loses money subtracts value. Arm 0 is always available at exactly zero, so
    # the optimum never treats them.
    fat = unconstrained_allocation(nv, cost_mat, 10_000_000.0)
    treated_nv = nv[np.arange(n), fat.assignment][fat.assignment > 0]
    assert (treated_nv > 0).all(), (
        f"{int((treated_nv <= 0).sum())} treated customers have non-positive net value under a "
        "non-binding budget"
    )
    n_profitable = int((nv[:, 1:] > 0).any(axis=1).sum())
    assert int(fat.n_treated) == n_profitable, f"{fat.n_treated} treated vs {n_profitable} profitable"
    assert float(fat.dual_lambda) == 0.0, "a non-binding budget must have shadow price zero"
    assert fat.expected_incremental_value >= unc.expected_incremental_value

    all_bad = np.column_stack([np.zeros(50), -np.ones((50, 3))])
    all_bad_cost = np.column_stack([np.zeros(50), np.ones((50, 3))])
    none = unconstrained_allocation(all_bad, all_bad_cost, 10_000.0)
    assert int(none.n_treated) == 0 and float(none.expected_incremental_value) == 0.0, (
        "when no arm pays for itself the do-nothing policy is optimal"
    )

    # ---- the efficient frontier must be monotone non-decreasing in budget -------------
    ladder = np.array([0.0, 20_000.0, 60_000.0, 120_000.0, 300_000.0, 1_000_000.0])
    unc_curve, par_curve = [], []
    for b in ladder:
        ub = unconstrained_allocation(nv, cost_mat, float(b))
        pb = reweight_for_parity(nv, cost_mat, float(b), groups, tolerance=0.1)
        assert ub.total_cost <= b + 1e-6 and pb.total_cost <= b + 1e-6, f"overspend at budget {b}"
        unc_curve.append(ub.expected_incremental_value)
        par_curve.append(pb.expected_incremental_value)
        assert ub.expected_incremental_value + 1e-6 >= pb.expected_incremental_value, (
            f"the parity policy beat the unconstrained optimum at budget {b}: "
            f"{pb.expected_incremental_value} > {ub.expected_incremental_value}"
        )
    for name, curve in (("unconstrained", unc_curve), ("parity", par_curve)):
        d = np.diff(np.asarray(curve))
        assert (d >= -1e-6).all(), f"{name} frontier is not monotone in budget: {np.round(d, 4)}"

    # ---- agreement with the sibling solver in prism.decision.optimize -----------------
    # price_of_fairness subtracts two allocations, so the baseline has to be a real optimum and
    # not merely "this module's own answer". Checked against a solver written independently.
    try:
        from prism.decision.optimize import lagrangian_allocate
    except Exception as exc:  # pragma: no cover - sibling module absent
        print(f"  (skipped optimize cross-check: {exc})")
    else:
        for b in (40_000.0, 120_000.0, 1_000_000.0):
            mine = unconstrained_allocation(nv, cost_mat, b)
            theirs = lagrangian_allocate(nv, cost_mat, b)
            rel = abs(mine.expected_incremental_value - theirs.expected_incremental_value) / max(
                abs(theirs.expected_incremental_value), 1.0
            )
            assert rel < 1e-3, (
                f"budget {b:.0f}: this module's baseline is {rel:.2%} off "
                f"prism.decision.optimize.lagrangian_allocate "
                f"({mine.expected_incremental_value} vs {theirs.expected_incremental_value})"
            )
        print("  cross-checked against optimize.lagrangian_allocate at 3 budgets: agreement < 1e-3 rel")

    # ---- the AllocationResult contract the sibling module documents --------------------
    assert set(con.per_arm.columns) >= {
        "arm", "arm_name", "n_assigned", "total_cost", "total_net_value", "mean_net_value",
    }, f"per_arm does not match the SPEC 5.2 contract: {list(con.per_arm.columns)}"
    assert int(con.per_arm["n_assigned"].sum()) == n
    assert abs(float(con.per_arm["total_cost"].sum()) - con.total_cost) < 1e-6
    assert abs(float(con.per_arm["total_net_value"].sum()) - con.expected_incremental_value) < 1e-6
    assert float(getattr(con, "budget", budget)) == budget, "budget must be recorded on the result"
    summary = con.summary()
    assert len(summary) == 1 and not summary["total_cost"].isna().any()

    # tight tolerance squeezes the gap further and costs at least as much value
    tight = reweight_for_parity(nv, cost_mat, budget, groups, tolerance=0.0)
    tight_audit = policy_fairness_audit(tight.assignment, groups)
    tight_overall = float(tight_audit.loc[tight_audit.group == OVERALL_LABEL, "treat_rate"].item())
    tight_gap = float(
        (tight_audit[tight_audit.group != OVERALL_LABEL]["treat_rate"] - tight_overall).abs().max()
    )
    assert tight_gap <= 1e-3, f"exact parity should be nearly exact, got {tight_gap:.5f}"
    assert disparate_impact_ratio(tight.assignment, groups) >= di_after - 1e-9

    print(f"\nprice of fairness: {pof['value_forgone']:.2f} currency units "
          f"({pof['pct_value_forgone']:.2f}% of the unconstrained optimum) to move disparate impact "
          f"{di_before:.3f} -> {di_after:.3f}")
    print(f"fairness.py OK  ({time.perf_counter() - t0:.2f}s)")
