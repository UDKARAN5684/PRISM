"""Point-in-time feature store, temporal splits and the serving-stable design matrix.

This module is the *contract boundary between training and serving*. Three things are
easy to get wrong in a retention/uplift system and all three are handled here explicitly,
with machinery that can be tested rather than trusted:

1. **Label leakage through time.** A feature computed at decision time ``as_of_date`` may
   only see events with ``event_ts <= as_of_date``. :class:`PointInTimeFeatureStore`
   enforces that structurally (the window is an index range that *cannot* reach past the
   as-of boundary), verifies it per row after the fact, and :meth:`~PointInTimeFeatureStore.leakage_report`
   deliberately recomputes every feature with a *forward* window so the difference between
   "honest" and "cheating" is a number on a report rather than an opinion.

2. **Horizon overlap between splits.** Outcomes in PRISM are measured over a 12-month
   horizon, so a training row at period ``t`` and a validation row at period ``t + 1``
   share 11 months of label window. :func:`purged_temporal_split` removes an embargo band
   of periods at each split boundary; :func:`temporal_split` is the plain (unpurged)
   version kept for SPEC compatibility.

3. **Train/serve skew in the feature matrix.** :func:`build_design_matrix` emits a fixed,
   ordered column list driven by :data:`prism.data.schema.CATEGORICAL_LEVELS`, not by
   whatever levels happened to appear in a batch. The fitted encoder is a plain picklable
   dataclass (:class:`DesignMatrixEncoder`) holding only builtin types, so the serving
   bundle does not have to unpickle a scikit-learn ``ColumnTransformer`` (and therefore
   does not have to pin a scikit-learn version).

Notes
-----
All timestamps are handled as ``int64`` nanoseconds internally; ``window_days`` is always
expressed in **days** and windows are half-open ``(as_of - window_days, as_of]``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from prism.data.schema import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_LEVELS,
    FEATURES,
    NUMERIC_FEATURES,
)
from prism.utils.logging import get_logger
from prism.utils.validation import DataContract, Expectation

__all__ = [
    "AGGREGATIONS",
    "EVENT_COLUMNS",
    "EVENTS_CONTRACT",
    "PURCHASE_EVENT_TYPES",
    "FeatureSpec",
    "LeakageError",
    "PointInTimeFeatureStore",
    "DesignMatrixEncoder",
    "assert_no_leakage",
    "build_design_matrix",
    "customer_summary",
    "default_feature_specs",
    "design_matrix_frame",
    "purged_temporal_split",
    "temporal_split",
]

_LOG = get_logger(__name__)

#: Nanoseconds in one day; the single conversion constant used throughout.
_NS_PER_DAY: int = 86_400_000_000_000

#: Average length of a Gregorian month in days (365.2425 / 12), used by RFM summaries.
_DAYS_PER_MONTH: float = 30.436875

#: Aggregations supported by :class:`FeatureSpec`.
AGGREGATIONS: tuple[str, ...] = ("sum", "mean", "count", "max", "nunique", "last", "days_since")

#: Canonical bronze event-log columns produced by ``prism.data.dgp``.
EVENT_COLUMNS: tuple[str, ...] = ("customer_id", "event_ts", "event_type", "amount", "category", "n_items")

#: ``event_type`` values treated as a revenue transaction. Both spellings are accepted
#: because the simulator emits ``"order"`` while most real-world logs (and the telco /
#: online-retail adapters in ``prism.data.real``) say ``"purchase"``. A given log only
#: ever contains one of them, so matching both is free and removes a silent-zero failure
#: mode where every monetary feature comes back as 0.0 for a vocabulary mismatch.
PURCHASE_EVENT_TYPES: tuple[str, ...] = ("purchase", "order", "transaction")

#: Contract for the raw event log. Kept deliberately small: only the columns that every
#: event log must have are ``error`` severity, the payload columns are ``warn``.
EVENTS_CONTRACT: DataContract = DataContract(
    "bronze_events",
    [
        Expectation("*", "row_count", {"min": 1}),
        Expectation("customer_id", "not_null", {}),
        Expectation("event_ts", "column_exists", {}),
        Expectation("event_ts", "dtype", {"dtype": "datetime"}),
        Expectation("event_ts", "not_null", {}),
        Expectation("event_type", "not_null", {}, severity="warn"),
        Expectation("amount", "dtype", {"dtype": "numeric"}, severity="warn"),
        Expectation("n_items", "dtype", {"dtype": "numeric"}, severity="warn"),
    ],
)


def _to_naive_utc(series: pd.Series) -> tuple[pd.Series, Any]:
    """Return ``(tz-naive UTC series, original tz)``.

    Point-in-time comparisons are done on raw ``int64`` nanoseconds, so a tz-aware column
    and a tz-naive one silently compare *different clocks*: an aware series converts to
    UTC wall time while a naive one does not, shifting the as-of boundary by the offset
    and quietly admitting or dropping events. Everything is therefore reduced to one
    clock (UTC, tz stripped) and the store refuses a spine that disagrees with its log.
    """
    tz = getattr(series.dt, "tz", None)
    if tz is None:
        return series, None
    return series.dt.tz_convert("UTC").dt.tz_localize(None), tz


class LeakageError(Exception):
    """Raised when a feature computation demonstrably used information from the future.

    "From the future" means: an event with ``event_ts`` strictly greater than the
    ``as_of_date`` of the spine row whose feature value it contributed to.
    """


# ======================================================================================
# Feature specification
# ======================================================================================
@dataclass(frozen=True)
class FeatureSpec:
    """Declarative definition of one point-in-time aggregate feature.

    A spec says: *for entity E, at decision time T, aggregate column ``source`` over the
    events of type ``event_types`` whose timestamp falls in ``(T - window_days, T]``.*

    Parameters
    ----------
    name : str
        Output column name.
    source : str
        Event-log column to aggregate, or the literal ``"panel"`` to pass a column of the
        same ``name`` straight through from the spine (used for features that are already
        point-in-time by construction, e.g. contract attributes).
    agg : {"sum", "mean", "count", "max", "nunique", "last", "days_since"}
        Aggregation applied to the window. Semantics (deliberately pinned down, because
        "count" and "last" mean different things in different systems):

        ``sum``
            Sum of the non-null ``source`` values in the window; ``0.0`` for an empty
            window (pandas convention).
        ``mean``
            Mean of the non-null values; ``NaN`` for an empty window.
        ``count``
            Number of events in the window whose ``source`` value is **non-null**
            (``pandas.Series.count`` semantics). Point ``source`` at a never-null column
            such as ``event_type`` to get a plain event count.
        ``max``
            Maximum non-null value; ``NaN`` for an empty window.
        ``nunique``
            Number of distinct non-null values in the window.
        ``last``
            Most recent non-null value in the window, ties on ``event_ts`` broken by the
            original row order of ``events``; ``NaN``/``None`` for an empty window.
        ``days_since``
            ``(as_of - event_ts)`` in fractional days for the most recent **event** in the
            window. Unlike ``last`` this ignores nullity of ``source`` -- it is a recency
            of *events*, not of values. ``NaN`` for an empty window.
    window_days : int or None
        Length of the look-back window in days. ``None`` means "all history up to and
        including ``as_of``".
    entity : str, default "customer_id"
        Join key present in both the event log and the spine.
    event_types : tuple of str or None, default None
        Restrict the window to these ``event_type`` values. ``None`` uses every event.
    fill_value : float or None, default None
        Value substituted when the aggregate is undefined -- an empty window for
        ``sum``/``count``/``nunique``/``days_since``, or a window with no non-null
        ``source`` value for ``mean``/``max``/``last`` -- overriding the per-agg default
        above. Useful for tree models, where an explicit sentinel such as ``-1`` beats a
        silent ``NaN``.
    description : str, default ""
        Free-text documentation carried into the feature catalogue.

    Notes
    -----
    The dataclass is frozen (and therefore hashable), which lets the store key its
    internal caches by spec.
    """

    name: str
    source: str
    agg: str
    window_days: int | None
    entity: str = "customer_id"
    event_types: tuple[str, ...] | None = None
    fill_value: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.agg not in AGGREGATIONS:
            raise ValueError(f"unknown agg {self.agg!r} for feature {self.name!r}; valid: {list(AGGREGATIONS)}")
        if self.window_days is not None:
            w = int(self.window_days)
            if w <= 0:
                raise ValueError(f"window_days must be positive or None, got {self.window_days!r}")
            object.__setattr__(self, "window_days", w)
        if self.event_types is not None:
            object.__setattr__(self, "event_types", tuple(self.event_types))
        if not self.name:
            raise ValueError("FeatureSpec.name must be a non-empty string")

    @property
    def is_passthrough(self) -> bool:
        """True when the spec copies a column from the spine instead of aggregating events."""
        return self.source == "panel"

    @property
    def view_key(self) -> tuple[str, tuple[str, ...] | None]:
        """Cache key of the pre-sorted event view this spec needs."""
        return (self.entity, self.event_types)

    def window_label(self) -> str:
        """Human-readable window, e.g. ``"365d"`` or ``"all"``."""
        return "all" if self.window_days is None else f"{int(self.window_days)}d"


# ======================================================================================
# Internal: vectorised range-maximum over a static array (segment tree)
# ======================================================================================
class _RangeMax:
    """Iterative bottom-up segment tree answering many range-max queries at once.

    Built in ``O(n)`` time and ``O(n)`` memory (a flat array of ``2n`` floats); a batch of
    ``m`` queries is answered in ``O(m log n)`` with fully vectorised numpy -- no Python
    loop over queries. A sparse table would give ``O(1)`` queries but costs
    ``O(n log n)`` memory, which is the wrong trade for an event log with millions of rows.

    Parameters
    ----------
    values : numpy.ndarray
        Float array; ``NaN`` entries are treated as "absent" (replaced by ``-inf``).
    """

    __slots__ = ("n", "tree", "levels")

    def __init__(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float64)
        v = np.where(np.isnan(v), -np.inf, v)
        n = int(v.size)
        self.n = n
        if n == 0:
            self.tree = np.zeros(0, dtype=np.float64)
            self.levels = 1
            return
        tree = np.full(2 * n, -np.inf, dtype=np.float64)
        tree[n:] = v
        # Build depth-block by depth-block, deepest first. Every node in [2**k, 2**(k+1))
        # has both children in [2**(k+1), 2**(k+2)), which is either the block finished on
        # the previous iteration or the leaf region [n, 2n) -- so no block ever reads a
        # value it is writing in the same vectorised assignment. (Halving the range
        # instead, [n//2, n) then [n//4, n//2) ..., violates that for odd n: the node
        # n//2 reads its child n-1, which sits in the very block being written.)
        level = 1
        while level * 2 < max(n, 2):
            level *= 2
        while level >= 1:
            lo, hi = level, min(level * 2, n)
            if hi > lo:
                idx = np.arange(lo, hi, dtype=np.int64)
                tree[idx] = np.maximum(tree[2 * idx], tree[2 * idx + 1])
            level //= 2
        self.tree = tree
        self.levels = int(np.ceil(np.log2(max(2 * n, 2)))) + 2

    def query(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Return the maximum over each half-open range ``[lo_i, hi_i)``.

        Parameters
        ----------
        lo, hi : numpy.ndarray of int
            Range bounds, same shape. Empty ranges (``lo == hi``) yield ``-inf``.

        Returns
        -------
        numpy.ndarray
            Float array of maxima, ``-inf`` where the range is empty or all-``NaN``.
        """
        out = np.full(np.shape(lo), -np.inf, dtype=np.float64)
        if self.n == 0 or out.size == 0:
            return out
        size = 2 * self.n
        left = np.asarray(lo, dtype=np.int64) + self.n
        right = np.asarray(hi, dtype=np.int64) + self.n
        for _ in range(self.levels):
            active = left < right
            if not active.any():
                break
            take_l = active & ((left & 1) == 1)
            gathered = self.tree[np.clip(left, 0, size - 1)]
            out = np.where(take_l, np.maximum(out, gathered), out)
            left = np.where(take_l, left + 1, left)

            take_r = active & ((right & 1) == 1)
            right = np.where(take_r, right - 1, right)
            gathered = self.tree[np.clip(right, 0, size - 1)]
            out = np.where(take_r, np.maximum(out, gathered), out)

            left = np.where(active, left >> 1, left)
            right = np.where(active, right >> 1, right)
        return out


# ======================================================================================
# Internal: one pre-sorted, pre-indexed view of the event log
# ======================================================================================
class _EventView:
    """Events for one ``(entity, event_types)`` combination, sorted once and indexed.

    The view materialises the *order statistic* structure that makes the as-of join fast:

    * rows sorted by ``(entity_code, event_ts)`` -- each entity therefore owns one
      contiguous index range;
    * ``uniq_ts``: the sorted distinct timestamps, so a timestamp maps to a small integer
      rank via one ``searchsorted``;
    * ``key = entity_code * (K + 1) + ts_rank``: a single monotonically increasing int64
      array. Because it is monotone, ``np.searchsorted`` on it resolves *both* the entity
      and the time boundary in one vectorised call -- no per-group Python loop.

    Cumulative statistics (prefix sums / counts / last-valid index / segment tree) are
    built lazily per source column and cached, so a spec only pays for what it uses.
    """

    __slots__ = (
        "entity",
        "event_types",
        "n",
        "codes",
        "entity_index",
        "ts_ns",
        "uniq_ts",
        "key",
        "_order",
        "_frame",
        "_cache",
    )

    def __init__(self, events: pd.DataFrame, entity: str, event_types: tuple[str, ...] | None) -> None:
        self.entity = entity
        self.event_types = event_types

        frame = events
        if event_types is not None:
            if "event_type" not in events.columns:
                raise KeyError("events must contain an 'event_type' column to use FeatureSpec.event_types")
            frame = events.loc[events["event_type"].isin(list(event_types))]

        ts_raw = pd.to_datetime(frame["event_ts"]).to_numpy("datetime64[ns]").astype("int64")
        ent_codes, uniques = pd.factorize(frame[entity], sort=True)
        order = np.lexsort((ts_raw, ent_codes.astype(np.int64)))

        self._frame = frame
        self._order = order
        self.n = int(order.size)
        self.codes = ent_codes.astype(np.int64)[order]
        self.entity_index = pd.Index(uniques)
        self.ts_ns = ts_raw[order]

        self.uniq_ts = np.unique(self.ts_ns) if self.n else np.zeros(0, dtype="int64")
        k = int(self.uniq_ts.size)
        ts_rank = np.searchsorted(self.uniq_ts, self.ts_ns) if self.n else np.zeros(0, dtype=np.int64)
        self.key = self.codes * np.int64(k + 1) + ts_rank.astype(np.int64)
        self._cache: dict[tuple[str, str], Any] = {}

    # -- boundary resolution ----------------------------------------------------------
    @property
    def _k(self) -> int:
        return int(self.uniq_ts.size)

    def entity_codes(self, values: pd.Series | np.ndarray) -> np.ndarray:
        """Map spine entity values to this view's entity codes (``-1`` when unseen)."""
        arr = np.asarray(values)
        if len(self.entity_index) == 0:
            return np.full(arr.shape[0], -1, dtype=np.int64)
        return self.entity_index.get_indexer(pd.Index(arr)).astype(np.int64)

    def boundary(self, codes: np.ndarray, ts_ns: np.ndarray) -> np.ndarray:
        """Index of the first event of each entity with ``event_ts > ts_ns``.

        Entities with code ``-1`` (never seen in the event log) produce ``0``: the query
        key becomes negative, which sorts before every real key, so ``lo == hi == 0`` and
        the window is empty. That is the correct answer and needs no special case.
        """
        k = self._k
        rank = np.searchsorted(self.uniq_ts, ts_ns, side="right").astype(np.int64)
        probe = codes * np.int64(k + 1) + rank
        return np.searchsorted(self.key, probe, side="left").astype(np.int64)

    def group_start(self, codes: np.ndarray) -> np.ndarray:
        """Index of each entity's first event (``0`` for entities absent from the log)."""
        probe = codes * np.int64(self._k + 1)
        return np.searchsorted(self.key, probe, side="left").astype(np.int64)

    def group_end(self, codes: np.ndarray) -> np.ndarray:
        """Index one past the last event of each entity."""
        k = self._k
        probe = codes * np.int64(k + 1) + np.int64(k)
        return np.searchsorted(self.key, probe, side="left").astype(np.int64)

    def last_event_ts(self, hi: np.ndarray) -> np.ndarray:
        """``event_ts`` (ns) of the newest event in each window; garbage where empty."""
        if self.n == 0:
            return np.full(np.shape(hi), np.iinfo(np.int64).min, dtype=np.int64)
        return self.ts_ns[np.clip(np.asarray(hi, dtype=np.int64) - 1, 0, self.n - 1)]

    # -- lazily built column statistics ------------------------------------------------
    def raw_values(self, source: str) -> np.ndarray:
        """Object-dtype view of ``source`` in sorted order (preserves strings for ``last``)."""
        cached = self._cache.get((source, "values"))
        if cached is None:
            if source not in self._frame.columns:
                raise KeyError(f"events has no column {source!r} required by a FeatureSpec")
            cached = self._frame[source].to_numpy(dtype=object)[self._order]
            self._cache[(source, "values")] = cached
        return cached

    def _numeric(self, source: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get((source, "numeric"))
        if cached is None:
            if source not in self._frame.columns:
                raise KeyError(f"events has no column {source!r} required by a FeatureSpec")
            vals = pd.to_numeric(self._frame[source], errors="coerce").to_numpy(dtype=np.float64)[self._order]
            notna = ~np.isnan(vals)
            cached = (vals, notna)
            self._cache[(source, "numeric")] = cached
        return cached

    def prefix_sum(self, source: str) -> np.ndarray:
        """Prefix sums of the non-null numeric values (length ``n + 1``)."""
        cached = self._cache.get((source, "csum"))
        if cached is None:
            vals, notna = self._numeric(source)
            cached = np.concatenate(([0.0], np.cumsum(np.where(notna, vals, 0.0))))
            self._cache[(source, "csum")] = cached
        return cached

    def prefix_count(self, source: str) -> np.ndarray:
        """Prefix counts of non-null values (length ``n + 1``)."""
        cached = self._cache.get((source, "ccnt"))
        if cached is None:
            notna = ~pd.isna(self.raw_values(source))
            cached = np.concatenate(([0], np.cumsum(notna.astype(np.int64))))
            self._cache[(source, "ccnt")] = cached
        return cached

    def prefix_count_numeric(self, source: str) -> np.ndarray:
        """Prefix counts of values that survive numeric coercion (length ``n + 1``).

        This is the denominator ``mean`` must use: :meth:`prefix_sum` skips any cell that
        ``pd.to_numeric(errors="coerce")`` turns into ``NaN``, so dividing by the
        *object*-level non-null count (:meth:`prefix_count`) would silently bias the mean
        downwards whenever a numeric column carries a dirty non-numeric cell.
        """
        cached = self._cache.get((source, "ccntnum"))
        if cached is None:
            _, notna = self._numeric(source)
            cached = np.concatenate(([0], np.cumsum(notna.astype(np.int64))))
            self._cache[(source, "ccntnum")] = cached
        return cached

    def range_max(self, source: str) -> _RangeMax:
        """Segment tree over the numeric values of ``source``."""
        cached = self._cache.get((source, "rmax"))
        if cached is None:
            vals, notna = self._numeric(source)
            cached = _RangeMax(np.where(notna, vals, np.nan))
            self._cache[(source, "rmax")] = cached
        return cached

    def last_valid_prefix(self, source: str) -> np.ndarray:
        """``out[i]`` = index of the last non-null value strictly before ``i`` (else ``-1``)."""
        cached = self._cache.get((source, "lastidx"))
        if cached is None:
            notna = ~pd.isna(self.raw_values(source))
            running = np.maximum.accumulate(np.where(notna, np.arange(self.n, dtype=np.int64), np.int64(-1)))
            cached = np.concatenate(([np.int64(-1)], running)) if self.n else np.array([-1], dtype=np.int64)
            self._cache[(source, "lastidx")] = cached
        return cached

    def value_codes(self, source: str) -> tuple[np.ndarray, int]:
        """Factorised codes of ``source`` (``-1`` for null) plus the cardinality."""
        cached = self._cache.get((source, "vcodes"))
        if cached is None:
            if source not in self._frame.columns:
                raise KeyError(f"events has no column {source!r} required by a FeatureSpec")
            codes, uniques = pd.factorize(self._frame[source], use_na_sentinel=True)
            cached = (codes.astype(np.int64)[self._order], int(len(uniques)))
            self._cache[(source, "vcodes")] = cached
        return cached

    def distinct_prefix(self, source: str) -> np.ndarray:
        """Prefix count of *first occurrences within an entity* (length ``n + 1``).

        Enables an O(1) expanding-window ``nunique``: for a window that starts at the
        entity's first event, the number of distinct values equals the number of first
        occurrences inside the range.
        """
        cached = self._cache.get((source, "distpref"))
        if cached is None:
            codes, n_vals = self.value_codes(source)
            first = np.zeros(self.n, dtype=np.int64)
            if self.n:
                combined = self.codes * np.int64(n_vals + 1) + (codes + 1)
                # np.unique(..., return_index=True) uses a stable sort, so these are the
                # indices of the *first* occurrence of each (entity, value) pair.
                _, first_idx = np.unique(combined, return_index=True)
                keep = first_idx[codes[first_idx] >= 0]
                first[keep] = 1
            cached = np.concatenate(([0], np.cumsum(first)))
            self._cache[(source, "distpref")] = cached
        return cached


# ======================================================================================
# The feature store
# ======================================================================================
class PointInTimeFeatureStore:
    """Strictly backward-looking as-of feature computation over a long event log.

    Algorithm
    ---------
    The naive implementation is a loop over spine rows, each filtering the event log:
    ``O(n_events * n_spine)``. This class is ``O((n_events + n_spine) log n_events)``:

    1. **Sort once.** For each distinct ``(entity, event_types)`` pair the events are
       filtered and sorted by ``(entity, event_ts)`` into an :class:`_EventView`. Every
       entity therefore owns a contiguous index range.
    2. **One monotone key.** Timestamps are replaced by their rank among the distinct
       timestamps (``K`` of them) and combined with the entity code into
       ``key = entity_code * (K + 1) + ts_rank``. Sorted by construction.
    3. **Two searchsorted calls per spec.** For a batch of spine rows, the window bounds
       ``(as_of - window_days, as_of]`` become index bounds ``[lo, hi)`` with two
       vectorised ``np.searchsorted`` calls on ``key`` -- both the entity lookup and the
       time cut happen in the same binary search. ``window_days=None`` sets ``lo`` to the
       entity's first event. Entities absent from the event log yield a negative probe
       key, hence ``lo = hi = 0`` and an empty window, with no special-casing.
    4. **O(1) aggregation from precomputed prefixes.** ``sum``/``count``/``mean`` come from
       prefix sums, ``max`` from a segment tree (``O(log n)`` per query, vectorised),
       ``last`` from a running last-non-null-index array, ``days_since`` from
       ``event_ts[hi - 1]``.
    5. **nunique.** Range-distinct-count has no prefix-sum form. For ``window_days=None``
       it is still ``O(1)`` (prefix count of first occurrences within an entity). For a
       finite window it falls back to a **single ordered sliding-window pass**: because
       ``lo`` and ``hi`` are both non-decreasing in ``as_of`` within an entity, sorting the
       queries by ``(lo, hi)`` lets two pointers sweep each event exactly once --
       ``O(n_events + n_spine log n_spine)``. It is a Python loop over *queries*, not a
       nested loop over ``queries x events``.

    Leakage guarantee
    -----------------
    ``hi`` is derived from ``searchsorted(..., as_of, side="right")``, so index ``hi - 1``
    is by construction the newest event with ``event_ts <= as_of``; nothing later is
    addressable. On top of that structural guarantee every call re-checks the actual
    timestamp used per row and raises :class:`LeakageError` if any exceeds its
    ``as_of``. :meth:`leakage_report` additionally recomputes each feature with a
    forward-looking window to quantify how much signal an unsafe implementation would
    have stolen.

    Parameters
    ----------
    events : pandas.DataFrame
        Long event log. Must contain ``event_ts`` plus every ``FeatureSpec.entity`` and
        ``FeatureSpec.source`` column referenced by ``specs``, and ``event_type`` if any
        spec filters on it. Rows with a null entity or null ``event_ts`` are dropped.
    specs : list of FeatureSpec
        Features to compute. Names must be unique.
    validate : bool, default True
        Run :data:`EVENTS_CONTRACT` and log any failures (warnings do not raise).
    leak_lookahead_days : float, default 0.0
        **Testing knob, leave at 0.** Shifts the whole window forward by this many days,
        deliberately breaking the point-in-time guarantee so that
        :meth:`leakage_report` / :func:`assert_no_leakage` can be shown to catch it. When
        non-zero, :meth:`as_of_join` reports instead of raising.

    Attributes
    ----------
    specs : list of FeatureSpec
    feature_names : list of str
    n_events : int
        Rows retained after dropping null keys/timestamps.

    Examples
    --------
    >>> import pandas as pd
    >>> ev = pd.DataFrame({"customer_id": ["a", "a", "b"],
    ...                    "event_ts": pd.to_datetime(["2022-01-01", "2022-03-01", "2022-02-01"]),
    ...                    "event_type": ["purchase"] * 3, "amount": [10.0, 5.0, 7.0]})
    >>> spine = pd.DataFrame({"customer_id": ["a", "b"],
    ...                       "as_of_date": pd.to_datetime(["2022-02-01", "2022-02-01"])})
    >>> store = PointInTimeFeatureStore(ev, [FeatureSpec("spend", "amount", "sum", None)])
    >>> store.as_of_join(spine)["spend"].tolist()
    [10.0, 7.0]
    """

    def __init__(
        self,
        events: pd.DataFrame,
        specs: list[FeatureSpec],
        *,
        validate: bool = True,
        leak_lookahead_days: float = 0.0,
    ) -> None:
        if not isinstance(events, pd.DataFrame):
            raise TypeError(f"events must be a DataFrame, got {type(events).__name__}")
        specs = list(specs)
        if not specs:
            raise ValueError("at least one FeatureSpec is required")
        names = [s.name for s in specs]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate FeatureSpec names: {dupes}")

        self.specs: list[FeatureSpec] = specs
        self.leak_lookahead_days = float(leak_lookahead_days)
        #: Timezone of the event log (``None`` when naive). The spine must agree.
        self._events_tz: Any = None
        #: False when every spec is a pass-through, i.e. no timestamp comparison happens.
        self._uses_events: bool = False
        #: When the store was built deliberately broken, ``as_of_join`` reports the
        #: violation instead of raising, so the breakage stays inspectable in tests.
        self._leak_warn_only: bool = bool(leak_lookahead_days)
        self._views: dict[tuple[str, tuple[str, ...] | None], _EventView] = {}

        needed_entities = {s.entity for s in specs if not s.is_passthrough}
        needed_sources = {s.source for s in specs if not s.is_passthrough}
        if needed_entities and "event_ts" not in events.columns:
            raise KeyError("events must contain an 'event_ts' column")
        missing = sorted((needed_entities | needed_sources) - set(events.columns))
        if missing:
            raise KeyError(f"events is missing columns required by the specs: {missing}")

        clean = events
        self._uses_events = bool(needed_entities)
        if needed_entities:
            keep = pd.to_datetime(events["event_ts"], errors="coerce").notna()
            for ent in needed_entities:
                keep &= events[ent].notna()
            n_drop = int((~keep).sum())
            if n_drop:
                _LOG.warning("dropping %d event rows with a null entity or event_ts", n_drop)
                clean = events.loc[keep]
            clean = clean.copy()
            clean["event_ts"] = pd.to_datetime(clean["event_ts"])
            clean["event_ts"], self._events_tz = _to_naive_utc(clean["event_ts"])

        self._events = clean.reset_index(drop=True)
        self.n_events = int(len(self._events)) if needed_entities else 0

        if validate and needed_entities:
            result = EVENTS_CONTRACT.validate(self._events)
            for failure in result.failures:
                _LOG.warning("events contract [%s] %s: %s", failure["severity"], failure["column"], failure["detail"])
            result.raise_for_status()

        self._last_diagnostics: pd.DataFrame | None = None

    # -- public API --------------------------------------------------------------------
    @property
    def feature_names(self) -> list[str]:
        """Names of the features this store produces, in spec order."""
        return [s.name for s in self.specs]

    def catalogue(self) -> pd.DataFrame:
        """Return a tidy catalogue of the configured specs (documentation artefact).

        Returns
        -------
        pandas.DataFrame
            Columns ``feature, source, agg, window, entity, event_types, description``.
        """
        return pd.DataFrame(
            [
                {
                    "feature": s.name,
                    "source": s.source,
                    "agg": s.agg,
                    "window": s.window_label(),
                    "entity": s.entity,
                    "event_types": "|".join(s.event_types) if s.event_types else "*",
                    "description": s.description,
                }
                for s in self.specs
            ]
        )

    def as_of_join(self, spine: pd.DataFrame, ts_col: str = "as_of_date") -> pd.DataFrame:
        """Attach every feature to the spine, using only events at or before ``as_of``.

        Parameters
        ----------
        spine : pandas.DataFrame
            Decision rows. Must contain ``ts_col`` and every ``FeatureSpec.entity``.
        ts_col : str, default "as_of_date"
            Column holding the decision timestamp.

        Returns
        -------
        pandas.DataFrame
            A copy of ``spine`` (same index, same row order) with one column per spec
            appended. Spine columns colliding with a feature name are replaced.

        Raises
        ------
        LeakageError
            If any computed value used an event with ``event_ts > as_of``. This cannot
            happen unless the store was built with ``leak_lookahead_days > 0`` (in which
            case the check only warns, so the deliberate breakage stays inspectable).
        KeyError
            If the spine lacks ``ts_col`` or an entity column.
        """
        values, diagnostics = self._compute(spine, ts_col, forward=False)
        self._last_diagnostics = diagnostics

        leaked = diagnostics.loc[diagnostics["future_rows_used"] > 0]
        if len(leaked):
            detail = ", ".join(
                f"{r.feature}: {int(r.future_rows_used)} rows, max gap {r.max_gap_days:+.3f}d"
                for r in leaked.itertuples()
            )
            if self._leak_warn_only:
                _LOG.warning("deliberate leakage mode active (lookahead=%.1fd) -> %s", self.leak_lookahead_days, detail)
            else:
                raise LeakageError(f"point-in-time violation in as_of_join -> {detail}")

        out = spine.copy()
        for name, series in values.items():
            out[name] = series
        return out

    def leakage_report(self, spine: pd.DataFrame, ts_col: str = "as_of_date") -> pd.DataFrame:
        """Recompute every feature honestly and dishonestly, and compare.

        For each spec this computes (a) the correct backward window
        ``(as_of - window, as_of]`` and (b) a deliberately forward-looking window
        ``(as_of, as_of + window]`` (all remaining history when ``window_days is None``).
        The correlation and mean absolute difference between the two quantify how much a
        broken implementation would have gained; ``max_event_ts_used`` and
        ``max_gap_days`` prove the honest pass never crossed the boundary.

        Parameters
        ----------
        spine : pandas.DataFrame
            Decision rows.
        ts_col : str, default "as_of_date"
            Column holding the decision timestamp.

        Returns
        -------
        pandas.DataFrame
            One row per feature with columns
            ``feature, n_rows, max_event_ts_used, as_of_min, as_of_max, future_rows_used,
            leak_detected`` (the SPEC set) plus the diagnostics
            ``agg, window, as_of_at_max_event_ts, max_gap_days, n_nonnull, n_nonnull_future,
            corr_with_future, mean_abs_diff_future, frac_rows_changed_by_future``.
        """
        back_vals, diagnostics = self._compute(spine, ts_col, forward=False)
        fwd_vals, _ = self._compute(spine, ts_col, forward=True)

        rows: list[dict[str, Any]] = []
        for spec in self.specs:
            diag = diagnostics.loc[diagnostics["feature"] == spec.name].iloc[0]
            back = pd.to_numeric(back_vals[spec.name], errors="coerce").to_numpy(dtype=np.float64)
            fwd = pd.to_numeric(fwd_vals[spec.name], errors="coerce").to_numpy(dtype=np.float64)
            both = np.isfinite(back) & np.isfinite(fwd)
            if both.sum() >= 2 and np.std(back[both]) > 1e-12 and np.std(fwd[both]) > 1e-12:
                corr = float(np.corrcoef(back[both], fwd[both])[0, 1])
            else:
                corr = float("nan")
            mad = float(np.mean(np.abs(back[both] - fwd[both]))) if both.any() else float("nan")
            changed = float(np.mean(~np.isclose(back[both], fwd[both], equal_nan=True))) if both.any() else float("nan")
            rows.append(
                {
                    "feature": spec.name,
                    "n_rows": int(diag["n_rows"]),
                    "max_event_ts_used": diag["max_event_ts_used"],
                    "as_of_min": diag["as_of_min"],
                    "as_of_max": diag["as_of_max"],
                    "future_rows_used": int(diag["future_rows_used"]),
                    "leak_detected": bool(diag["future_rows_used"] > 0),
                    "agg": spec.agg,
                    "window": spec.window_label(),
                    "as_of_at_max_event_ts": diag["as_of_at_max_event_ts"],
                    "max_gap_days": float(diag["max_gap_days"]),
                    "n_nonnull": int(np.isfinite(back).sum()),
                    "n_nonnull_future": int(np.isfinite(fwd).sum()),
                    "corr_with_future": corr,
                    "mean_abs_diff_future": mad,
                    "frac_rows_changed_by_future": changed,
                }
            )
        return pd.DataFrame(rows)

    # -- internals ----------------------------------------------------------------------
    def _view(self, spec: FeatureSpec) -> _EventView:
        key = spec.view_key
        view = self._views.get(key)
        if view is None:
            view = _EventView(self._events, spec.entity, spec.event_types)
            self._views[key] = view
        return view

    def _compute(
        self, spine: pd.DataFrame, ts_col: str, *, forward: bool
    ) -> tuple[dict[str, pd.Series], pd.DataFrame]:
        """Compute every spec over ``spine``; return values plus per-feature diagnostics."""
        if ts_col not in spine.columns:
            raise KeyError(f"spine is missing the timestamp column {ts_col!r}")
        as_of = pd.to_datetime(spine[ts_col])
        if as_of.isna().any():
            raise ValueError(f"spine[{ts_col!r}] contains null timestamps; point-in-time joins need a decision time")
        as_of, spine_tz = _to_naive_utc(as_of)
        if self._uses_events and (spine_tz is None) != (self._events_tz is None):
            raise ValueError(
                f"timezone mismatch: events are {'tz-aware (' + str(self._events_tz) + ')' if self._events_tz else 'tz-naive'} "
                f"but spine[{ts_col!r}] is {'tz-aware (' + str(spine_tz) + ')' if spine_tz else 'tz-naive'}. "
                "Comparing them would shift the as-of boundary by the UTC offset and silently leak or drop "
                "events; localise both sides to the same clock first."
            )
        t_ns = as_of.to_numpy("datetime64[ns]").astype("int64")
        n_rows = int(t_ns.size)
        as_of_min = as_of.min() if n_rows else pd.NaT
        as_of_max = as_of.max() if n_rows else pd.NaT

        values: dict[str, pd.Series] = {}
        diagnostics: list[dict[str, Any]] = []

        for spec in self.specs:
            if spec.is_passthrough:
                if spec.name not in spine.columns:
                    raise KeyError(f"spec {spec.name!r} has source='panel' but the spine has no such column")
                values[spec.name] = spine[spec.name].copy()
                diagnostics.append(
                    {
                        "feature": spec.name,
                        "n_rows": n_rows,
                        "max_event_ts_used": pd.NaT,
                        "as_of_at_max_event_ts": pd.NaT,
                        "as_of_min": as_of_min,
                        "as_of_max": as_of_max,
                        "future_rows_used": 0,
                        "max_gap_days": float("nan"),
                    }
                )
                continue

            if spec.entity not in spine.columns:
                raise KeyError(f"spine is missing the entity column {spec.entity!r} required by {spec.name!r}")

            view = self._view(spec)
            codes = view.entity_codes(spine[spec.entity].to_numpy())
            lo, hi = self._bounds(view, spec, codes, t_ns, forward=forward)
            expanding = spec.window_days is None and not forward
            series = self._aggregate(view, spec, lo, hi, t_ns, expanding=expanding)
            values[spec.name] = pd.Series(series, index=spine.index, name=spec.name)

            used = view.last_event_ts(hi)
            has_used = hi > lo
            if has_used.any():
                gaps = (used[has_used] - t_ns[has_used]) / _NS_PER_DAY
                max_gap = float(gaps.max())
                future_rows = int((used[has_used] > t_ns[has_used]).sum())
                arg = int(np.argmax(used[has_used]))
                max_ts = pd.Timestamp(int(used[has_used][arg]))
                at_as_of = pd.Timestamp(int(t_ns[has_used][arg]))
            else:
                max_gap = float("nan")
                future_rows = 0
                max_ts = pd.NaT
                at_as_of = pd.NaT
            diagnostics.append(
                {
                    "feature": spec.name,
                    "n_rows": n_rows,
                    "max_event_ts_used": max_ts,
                    "as_of_at_max_event_ts": at_as_of,
                    "as_of_min": as_of_min,
                    "as_of_max": as_of_max,
                    "future_rows_used": future_rows,
                    "max_gap_days": max_gap,
                }
            )

        return values, pd.DataFrame(diagnostics)

    def _bounds(
        self,
        view: _EventView,
        spec: FeatureSpec,
        codes: np.ndarray,
        t_ns: np.ndarray,
        *,
        forward: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve the half-open event-index window ``[lo, hi)`` for every spine row."""
        if forward:
            # Deliberately unsafe comparator: (as_of, as_of + window].
            lo = view.boundary(codes, t_ns)
            if spec.window_days is None:
                hi = view.group_end(codes)
            else:
                hi = view.boundary(codes, t_ns + np.int64(spec.window_days) * _NS_PER_DAY)
            return lo, np.maximum(hi, lo)

        # Honest comparator: (as_of - window, as_of]. `lookahead` is zero unless the
        # store was deliberately built broken for leak-detector testing.
        lookahead = np.int64(round(self.leak_lookahead_days * _NS_PER_DAY))
        right_ns = t_ns + lookahead
        hi = view.boundary(codes, right_ns)
        if spec.window_days is None:
            lo = view.group_start(codes)
        else:
            lo = view.boundary(codes, right_ns - np.int64(spec.window_days) * _NS_PER_DAY)
        return np.minimum(lo, hi), hi

    def _aggregate(
        self,
        view: _EventView,
        spec: FeatureSpec,
        lo: np.ndarray,
        hi: np.ndarray,
        t_ns: np.ndarray,
        *,
        expanding: bool = False,
    ) -> np.ndarray:
        """Evaluate one aggregation over precomputed index windows."""
        empty = hi <= lo
        fill = spec.fill_value

        if spec.agg == "count":
            pref = view.prefix_count(spec.source)
            out = (pref[hi] - pref[lo]).astype(np.float64)
            return out if fill is None else np.where(empty, float(fill), out)

        if spec.agg == "sum":
            pref = view.prefix_sum(spec.source)
            out = pref[hi] - pref[lo]
            return out if fill is None else np.where(empty, float(fill), out)

        if spec.agg == "mean":
            csum = view.prefix_sum(spec.source)
            # numeric-valid count, so the denominator matches exactly the set of values
            # that prefix_sum added up (see _EventView.prefix_count_numeric)
            ccnt = view.prefix_count_numeric(spec.source)
            cnt = (ccnt[hi] - ccnt[lo]).astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(cnt > 0, (csum[hi] - csum[lo]) / np.maximum(cnt, 1.0), np.nan)
            return out if fill is None else np.where(cnt > 0, out, float(fill))

        if spec.agg == "max":
            raw = view.range_max(spec.source).query(lo, hi)
            out = np.where(np.isfinite(raw), raw, np.nan)
            return out if fill is None else np.where(np.isfinite(raw), out, float(fill))

        if spec.agg == "days_since":
            # substitute t_ns where the window is empty so the int64 subtraction below
            # can never overflow on the "no event" sentinel
            last = np.where(empty, t_ns, view.last_event_ts(hi))
            out = np.where(empty, np.nan, (t_ns - last) / _NS_PER_DAY)
            return out if fill is None else np.where(empty, float(fill), out)

        if spec.agg == "last":
            raw = view.raw_values(spec.source)
            pref = view.last_valid_prefix(spec.source)
            j = pref[hi]
            ok = j >= lo
            if view.n == 0:
                picked = np.full(np.shape(lo), np.nan, dtype=object)
            else:
                picked = raw[np.clip(j, 0, view.n - 1)]
            default: Any = np.nan if fill is None else float(fill)
            out_obj = np.where(ok, picked, default)
            as_series = pd.Series(out_obj)
            numeric = pd.to_numeric(as_series, errors="coerce")
            if int(numeric.notna().sum()) == int(as_series.notna().sum()):
                return numeric.to_numpy(dtype=np.float64)
            return out_obj

        if spec.agg == "nunique":
            out = self._nunique(view, spec, lo, hi, expanding=expanding)
            return out if fill is None else np.where(empty, float(fill), out)

        raise ValueError(f"unhandled agg {spec.agg!r}")  # pragma: no cover - guarded in FeatureSpec

    def _nunique(
        self,
        view: _EventView,
        spec: FeatureSpec,
        lo: np.ndarray,
        hi: np.ndarray,
        *,
        expanding: bool = False,
    ) -> np.ndarray:
        """Distinct non-null values per window.

        Expanding windows (``window_days is None`` on the honest pass, where ``lo`` is
        exactly the entity's first event) use the O(1) first-occurrence prefix. Finite
        windows use one ordered two-pointer sweep, which is valid because ``lo`` and
        ``hi`` are both non-decreasing when the queries are sorted by ``(lo, hi)`` --
        entities occupy disjoint contiguous index ranges, and within an entity both
        bounds are monotone in ``as_of``. That sweep touches each event at most twice,
        so it is ``O(n_events + n_spine log n_spine)``, not a nested loop.
        """
        if view.n == 0 or lo.size == 0:
            return np.zeros(np.shape(lo), dtype=np.float64)

        if expanding:
            pref = view.distinct_prefix(spec.source)
            return (pref[hi] - pref[lo]).astype(np.float64)

        codes, n_vals = view.value_codes(spec.source)
        order = np.lexsort((hi.astype(np.int64), lo.astype(np.int64)))
        counts = np.zeros(n_vals + 1, dtype=np.int64)
        out = np.zeros(lo.shape[0], dtype=np.float64)
        p_lo = 0
        p_hi = 0
        distinct = 0
        lo_i = lo.astype(np.int64)
        hi_i = hi.astype(np.int64)
        code_list = codes
        for q in order:
            target_hi = int(hi_i[q])
            target_lo = int(lo_i[q])
            while p_hi < target_hi:
                c = int(code_list[p_hi])
                if c >= 0:
                    counts[c] += 1
                    if counts[c] == 1:
                        distinct += 1
                p_hi += 1
            while p_lo < target_lo:
                c = int(code_list[p_lo])
                if c >= 0:
                    counts[c] -= 1
                    if counts[c] == 0:
                        distinct -= 1
                p_lo += 1
            out[q] = float(distinct)
        return out


def assert_no_leakage(
    store: PointInTimeFeatureStore, spine: pd.DataFrame, ts_col: str = "as_of_date"
) -> pd.DataFrame:
    """Raise :class:`LeakageError` if any feature in ``store`` used a future event.

    This is the assertion you put in CI. It recomputes every feature over ``spine``, both
    honestly and with a forward-looking window, and fails if the honest pass touched any
    event with ``event_ts > as_of_date``.

    Parameters
    ----------
    store : PointInTimeFeatureStore
        Configured store.
    spine : pandas.DataFrame
        Decision rows to audit.
    ts_col : str, default "as_of_date"
        Decision-time column.

    Returns
    -------
    pandas.DataFrame
        The full leakage report, so a caller can log it on success.

    Raises
    ------
    LeakageError
        If ``leak_detected`` is True for any feature, with the offending features, the
        number of affected rows and the worst ``event_ts - as_of`` gap in the message.
    """
    report = store.leakage_report(spine, ts_col=ts_col)
    bad = report.loc[report["leak_detected"]]
    if len(bad):
        detail = "; ".join(
            f"{r.feature} ({r.agg}/{r.window}): {int(r.future_rows_used)} of {int(r.n_rows)} rows, "
            f"max_event_ts_used={r.max_event_ts_used} > as_of={r.as_of_at_max_event_ts} "
            f"(gap {r.max_gap_days:+.3f} days)"
            for r in bad.itertuples()
        )
        raise LeakageError(f"{len(bad)} feature(s) used future information -> {detail}")
    return report


# ======================================================================================
# Default catalogue and RFM summary
# ======================================================================================
def default_feature_specs(
    purchase_types: Sequence[str] = PURCHASE_EVENT_TYPES,
    *,
    session_type: str = "session",
    ticket_type: str = "support_ticket",
    failure_type: str = "payment_failure",
) -> list[FeatureSpec]:
    """Return the standard PRISM catalogue that rebuilds panel features from the event log.

    The event log is assumed to carry the columns in :data:`EVENT_COLUMNS`. Every
    aggregation in :data:`AGGREGATIONS` appears at least once, so this catalogue doubles
    as an integration test of the store.

    Event-type vocabularies differ between sources -- ``prism.data.dgp`` emits ``"order"``
    where the public datasets say ``"purchase"``, and not every log has a
    ``payment_failure`` type at all. The defaults therefore match every known spelling of
    a transaction (:data:`PURCHASE_EVENT_TYPES`), and a spec whose event type is simply
    absent from the log yields ``0`` rather than failing.

    Parameters
    ----------
    purchase_types : sequence of str, default :data:`PURCHASE_EVENT_TYPES`
        ``event_type`` values counted as revenue transactions.
    session_type, ticket_type, failure_type : str
        ``event_type`` values for engagement, support and billing events.

    Returns
    -------
    list of FeatureSpec
        Thirteen specs whose names line up with the numeric block of the gold panel
        (see :data:`prism.data.schema.NUMERIC_FEATURES`) plus three extras.
    """
    purchase = tuple(purchase_types)
    return [
        FeatureSpec("frequency_12m", "event_type", "count", 365, event_types=purchase,
                    description="purchases in the trailing 12 months"),
        FeatureSpec("monetary_12m", "amount", "sum", 365, event_types=purchase,
                    description="revenue in the trailing 12 months"),
        FeatureSpec("avg_order_value", "amount", "mean", 365, event_types=purchase,
                    description="mean basket value over 12 months"),
        FeatureSpec("max_order_value_12m", "amount", "max", 365, event_types=purchase,
                    description="largest single basket over 12 months"),
        FeatureSpec("n_categories_12m", "category", "nunique", 365, event_types=purchase,
                    description="distinct product categories bought over 12 months"),
        FeatureSpec("basket_diversity", "n_items", "mean", 365, event_types=purchase,
                    description="mean items per basket over 12 months"),
        FeatureSpec("last_order_amount", "amount", "last", None, event_types=purchase,
                    description="value of the most recent basket, all history"),
        FeatureSpec("recency_days", "amount", "days_since", None, event_types=purchase,
                    description="days since the most recent purchase"),
        FeatureSpec("monetary_lifetime", "amount", "sum", None, event_types=purchase,
                    description="lifetime revenue to date"),
        FeatureSpec("sessions_30d", "event_type", "count", 30, event_types=(session_type,),
                    description="sessions in the trailing 30 days"),
        FeatureSpec("days_since_last_session", "event_type", "days_since", None, event_types=(session_type,),
                    description="days since the most recent session"),
        FeatureSpec("support_tickets_90d", "event_type", "count", 90, event_types=(ticket_type,),
                    description="support tickets in the trailing 90 days"),
        FeatureSpec("payment_failures_12m", "event_type", "count", 365, event_types=(failure_type,),
                    description="failed payments in the trailing 12 months"),
    ]


def customer_summary(
    events: pd.DataFrame,
    as_of: str | pd.Timestamp,
    horizon_months: int,
    *,
    entity: str = "customer_id",
    purchase_types: Sequence[str] | None = PURCHASE_EVENT_TYPES,
    amount_col: str = "amount",
    freq: str = "D",
    time_unit: Literal["months", "days"] = "months",
) -> pd.DataFrame:
    """Build the BG/NBD + Gamma-Gamma RFM summary as of a fixed observation time.

    Conventions (these are the *standard* ones and they matter -- getting ``frequency``
    off by one silently biases every CLV number):

    ``frequency``
        Number of **repeat** purchases, i.e. ``(number of distinct purchase periods) - 1``,
        floored at 0. Transactions are first collapsed to calendar periods of length
        ``freq`` (default one day), because BG/NBD models a purchase *opportunity* process
        and two baskets on the same day are one opportunity. A customer with a single
        purchase has ``frequency == 0``.
    ``recency``
        Age of the customer **at the time of their last purchase**, i.e.
        ``last_purchase - first_purchase``. It is *not* "time since last purchase".
        Zero for a one-purchase customer.
    ``T``
        Total age at the observation end: ``as_of - first_purchase``.
    ``monetary_value``
        Mean value of the **repeat** transactions only (the first purchase is excluded),
        which is what Gamma-Gamma conditions on. ``0.0`` when ``frequency == 0``.

    ``recency <= T`` holds by construction. Time is expressed in months of
    ``30.436875`` days (the mean Gregorian month) unless ``time_unit="days"``.

    Only events with ``event_ts <= as_of`` are used, so the summary is point-in-time safe;
    customers with no qualifying purchase are omitted entirely (BG/NBD is undefined for
    them).

    Parameters
    ----------
    events : pandas.DataFrame
        Event log with ``entity``, ``event_ts``, ``amount_col`` and (if filtering)
        ``event_type``.
    as_of : str or pandas.Timestamp
        Observation end. Everything after it is discarded.
    horizon_months : int
        Forecast horizon carried onto the frame as a column so downstream CLV code
        (``prism.models.clv.probabilistic_clv``) cannot be run with a mismatched horizon.
    entity : str, default "customer_id"
        Customer key.
    purchase_types : sequence of str or None, default :data:`PURCHASE_EVENT_TYPES`
        ``event_type`` values counted as transactions; ``None`` counts every event.
    amount_col : str, default "amount"
        Monetary column.
    freq : str, default "D"
        Pandas period alias used to collapse same-period transactions.
    time_unit : {"months", "days"}, default "months"
        Unit of ``recency`` and ``T``.

    Returns
    -------
    pandas.DataFrame
        Columns ``customer_id, frequency, recency, T, monetary_value`` (the contract
        required by :mod:`prism.models.clv`) followed by ``n_transactions,
        monetary_total, first_purchase, last_purchase, as_of, horizon_months``. The entity
        column keeps its own name; it is also aliased to ``customer_id`` when ``entity``
        differs. Sorted by the entity key.

    Raises
    ------
    ValueError
        If ``horizon_months`` is not positive.
    """
    if int(horizon_months) <= 0:
        raise ValueError(f"horizon_months must be positive, got {horizon_months!r}")
    stamp = pd.Timestamp(as_of)

    frame = events
    if purchase_types is not None:
        if "event_type" not in frame.columns:
            raise KeyError("events must contain 'event_type' to filter by purchase_types")
        frame = frame.loc[frame["event_type"].isin(list(purchase_types))]
    ts = pd.to_datetime(frame["event_ts"])
    frame = frame.loc[(ts <= stamp) & ts.notna()].copy()
    frame["event_ts"] = pd.to_datetime(frame["event_ts"])

    unit_days = _DAYS_PER_MONTH if time_unit == "months" else 1.0
    cols = [entity, "frequency", "recency", "T", "monetary_value", "n_transactions",
            "monetary_total", "first_purchase", "last_purchase", "as_of", "horizon_months"]
    if frame.empty:
        out = pd.DataFrame(columns=cols)
        if entity != "customer_id":
            out.insert(1, "customer_id", pd.Series(dtype=object))
        return out

    # Collapse to one row per (customer, calendar period): BG/NBD counts opportunities.
    frame["_period"] = frame["event_ts"].dt.to_period(freq)
    amount = pd.to_numeric(frame[amount_col], errors="coerce").fillna(0.0)
    frame["_amount"] = amount
    daily = (
        frame.groupby([entity, "_period"], sort=True, observed=True)
        .agg(_ts=("event_ts", "min"), _amt=("_amount", "sum"), _n=("event_ts", "size"))
        .reset_index()
        .sort_values([entity, "_ts"], kind="mergesort")
    )

    grouped = daily.groupby(entity, sort=True, observed=True)
    first_ts = grouped["_ts"].min()
    last_ts = grouped["_ts"].max()
    n_periods = grouped["_ts"].size()
    n_txn = grouped["_n"].sum()
    total_amt = grouped["_amt"].sum()
    first_amt = grouped["_amt"].first()

    frequency = (n_periods - 1).clip(lower=0).astype("int64")
    recency = ((last_ts - first_ts).dt.total_seconds() / 86400.0) / unit_days
    age = ((stamp - first_ts).dt.total_seconds() / 86400.0) / unit_days
    repeat_total = total_amt - first_amt
    monetary = np.where(frequency.to_numpy() > 0, repeat_total.to_numpy() / np.maximum(frequency.to_numpy(), 1), 0.0)

    out = pd.DataFrame(
        {
            entity: first_ts.index.to_numpy(),
            "frequency": frequency.to_numpy().astype(np.float64),
            "recency": recency.to_numpy().astype(np.float64),
            "T": age.to_numpy().astype(np.float64),
            "monetary_value": np.asarray(monetary, dtype=np.float64),
            "n_transactions": n_txn.to_numpy().astype("int64"),
            "monetary_total": total_amt.to_numpy().astype(np.float64),
            "first_purchase": first_ts.to_numpy(),
            "last_purchase": last_ts.to_numpy(),
        }
    )
    out["as_of"] = stamp
    out["horizon_months"] = int(horizon_months)
    if entity != "customer_id":
        out.insert(1, "customer_id", out[entity].to_numpy())
    # invariants that catch a mis-specified event log early
    assert bool((out["recency"] <= out["T"] + 1e-9).all()), "recency must not exceed T"
    return out.reset_index(drop=True)


# ======================================================================================
# Temporal splits
# ======================================================================================
def temporal_split(panel: pd.DataFrame, train_end: int, valid_end: int) -> pd.DataFrame:
    """Assign ``train``/``valid``/``test`` by period boundary, without shuffling.

    ``period <= train_end`` is train, ``train_end < period <= valid_end`` is valid, and
    everything later is test. Row order and index are preserved exactly; the frame is
    copied so the caller's panel is untouched.

    Parameters
    ----------
    panel : pandas.DataFrame
        Gold panel; must contain an integer ``period`` column.
    train_end : int
        Last period (inclusive) in the training set.
    valid_end : int
        Last period (inclusive) in the validation set; must be ``>= train_end``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``panel`` with a ``split`` column of dtype ``object``.

    Raises
    ------
    KeyError
        If ``period`` is absent.
    ValueError
        If ``valid_end < train_end``.

    See Also
    --------
    purged_temporal_split : same boundaries plus an embargo band; prefer it for the
        12-month-horizon outcomes used by PRISM.
    """
    if "period" not in panel.columns:
        raise KeyError("panel must have a 'period' column to split temporally")
    if int(valid_end) < int(train_end):
        raise ValueError(f"valid_end ({valid_end}) must be >= train_end ({train_end})")

    out = panel.copy()
    period = pd.to_numeric(out["period"], errors="raise").to_numpy()
    split = np.full(period.shape, "test", dtype=object)
    split[period <= int(valid_end)] = "valid"
    split[period <= int(train_end)] = "train"
    out["split"] = split
    return out


def purged_temporal_split(
    panel: pd.DataFrame,
    train_end: int,
    valid_end: int,
    embargo: int = 1,
    *,
    purge_valid: bool = True,
    drop: bool = False,
) -> pd.DataFrame:
    """Temporal split with an embargo band, the correct split for overlapping horizons.

    Why an embargo is required
    --------------------------
    PRISM outcomes (``rmst_h``, ``value_h``, ``event_time``) are measured over a
    **12-month forward horizon**. The label of a row at period ``t`` is therefore a
    function of what happens in periods ``t+1 .. t+12``. A plain boundary split puts the
    row at ``train_end`` in train and the row at ``train_end + 1`` in valid even though
    their label windows overlap by eleven months and are driven by the same realised
    churn events -- often for the *same customer*. Validation then measures how well the
    model memorised shared label noise, and every metric is optimistic.

    Purging removes the last ``embargo`` periods before each boundary so that no training
    label window reaches into the evaluation block. With a horizon of ``h`` periods the
    fully conservative choice is ``embargo = h``; ``embargo = 1`` is the minimum that
    removes the immediately adjacent (and most correlated) rows. The cost is the usual
    purging trade-off: fewer training rows in exchange for an honest estimate.

    Band layout with ``embargo = e``::

        period <= train_end - e                      -> "train"
        train_end - e <  period <= train_end         -> "embargo"
        train_end     <  period <= valid_end - e     -> "valid"     (if purge_valid)
        valid_end - e <  period <= valid_end         -> "embargo"   (if purge_valid)
        period > valid_end                           -> "test"

    Parameters
    ----------
    panel : pandas.DataFrame
        Gold panel with an integer ``period`` column.
    train_end, valid_end : int
        Inclusive boundaries, as in :func:`temporal_split`.
    embargo : int, default 1
        Number of periods removed before each boundary. ``0`` reproduces
        :func:`temporal_split`.
    purge_valid : bool, default True
        Also embargo the band immediately before the test block. Set ``False`` to purge
        only the train/valid boundary.
    drop : bool, default False
        If True the embargoed rows are removed from the returned frame. If False (default)
        they are kept and labelled ``"embargo"``, which is equivalent for any consumer
        that filters on ``split == "train"`` but leaves the discarded rows auditable.

    Returns
    -------
    pandas.DataFrame
        A copy of ``panel`` with a ``split`` column in
        ``{"train", "embargo", "valid", "test"}`` (``"embargo"`` absent when ``drop``),
        plus an ``is_embargoed`` boolean column. Row order is preserved.

    Raises
    ------
    ValueError
        If ``embargo`` is negative or ``valid_end < train_end``.
    """
    if int(embargo) < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo!r}")
    out = temporal_split(panel, train_end, valid_end)
    e = int(embargo)
    if e == 0:
        out["is_embargoed"] = False
        return out

    period = pd.to_numeric(out["period"], errors="raise").to_numpy()
    split = out["split"].to_numpy(dtype=object).copy()

    band_train = (period > int(train_end) - e) & (period <= int(train_end))
    split[band_train] = "embargo"
    if purge_valid:
        band_valid = (period > int(valid_end) - e) & (period <= int(valid_end)) & (period > int(train_end))
        split[band_valid] = "embargo"

    out["split"] = split
    out["is_embargoed"] = split == "embargo"
    if drop:
        out = out.loc[~out["is_embargoed"]].copy()
    return out


# ======================================================================================
# Design matrix
# ======================================================================================
@dataclass
class DesignMatrixEncoder:
    """Frozen, picklable description of how a frame becomes a model matrix.

    Deliberately *not* a scikit-learn ``ColumnTransformer``: it stores nothing but python
    builtins (str/float/tuple/dict), so the serving bundle unpickles without scikit-learn
    and without version-skew risk, and a human can read the whole encoder in a JSON diff.

    Column order is fixed and total:

    1. every numeric feature, in the order given at fit time;
    2. every ``"<col>__isna"`` missing indicator, same order (always emitted, even for a
       column with no missing values at fit time, so the width cannot depend on the batch);
    3. for each categorical feature, ``"<col>__<level>"`` for each level of
       :data:`prism.data.schema.CATEGORICAL_LEVELS` (or the levels learned at fit time for
       a column outside the schema), followed by ``"<col>__other"``.

    Every categorical block sums to exactly 1.0 per row: an unseen *or missing* level
    lands in ``__other``. Keeping the block a proper simplex means the matrix width and
    row sums are invariant to the data, which is what makes serving reproducible.

    Attributes
    ----------
    numeric_features : tuple of str
    categorical_features : tuple of str
    medians : dict of str to float
        Fit-time median of each numeric column, used for imputation.
    levels : dict of str to tuple of str
        Accepted levels per categorical column, excluding the implicit ``other``.
    feature_names : tuple of str
        The output column names, in order.
    other_token : str, default "other"
        Suffix used for the catch-all one-hot column.
    version : str, default "1"
        Encoder format version, recorded in the model card.
    """

    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    medians: dict[str, float] = field(default_factory=dict)
    levels: dict[str, tuple[str, ...]] = field(default_factory=dict)
    feature_names: tuple[str, ...] = ()
    other_token: str = "other"
    version: str = "1"

    @property
    def n_features(self) -> int:
        """Number of output columns."""
        return len(self.feature_names)

    def required_columns(self) -> tuple[str, ...]:
        """Input columns this encoder needs to transform a frame."""
        return tuple(self.numeric_features) + tuple(self.categorical_features)

    def transform(self, frame: pd.DataFrame, *, strict_columns: bool = True) -> np.ndarray:
        """Apply the encoder to ``frame``.

        Parameters
        ----------
        frame : pandas.DataFrame
            Rows to encode.
        strict_columns : bool, default True
            Raise when an expected input column is absent. When False a missing numeric
            column is treated as entirely missing (median-imputed, indicator 1) and a
            missing categorical column as entirely unknown (``__other`` = 1).

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_features)`` float64 matrix, columns ordered as
            :attr:`feature_names`.

        Raises
        ------
        KeyError
            If ``strict_columns`` and a required column is missing.
        """
        missing = [c for c in self.required_columns() if c not in frame.columns]
        if missing and strict_columns:
            raise KeyError(f"frame is missing columns required by the encoder: {missing}")

        n = len(frame)
        blocks: list[np.ndarray] = []

        num_block = np.empty((n, len(self.numeric_features)), dtype=np.float64)
        isna_block = np.empty((n, len(self.numeric_features)), dtype=np.float64)
        for j, col in enumerate(self.numeric_features):
            if col in frame.columns:
                vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
            else:
                vals = np.full(n, np.nan, dtype=np.float64)
            bad = ~np.isfinite(vals)
            isna_block[:, j] = bad.astype(np.float64)
            num_block[:, j] = np.where(bad, float(self.medians.get(col, 0.0)), vals)
        blocks.append(num_block)
        blocks.append(isna_block)

        for col in self.categorical_features:
            lv = tuple(self.levels.get(col, ()))
            if len(set(lv)) != len(lv):
                raise ValueError(f"encoder levels for {col!r} contain duplicates: {lv}")
            block = np.zeros((n, len(lv) + 1), dtype=np.float64)
            if col in frame.columns:
                norm = _normalise_categorical(frame[col])
                pos = pd.Index(lv).get_indexer(pd.Index(norm.to_numpy()))
            else:
                pos = np.full(n, -1, dtype=np.int64)
            known = pos >= 0
            rows = np.arange(n, dtype=np.int64)
            block[rows[known], pos[known]] = 1.0
            block[rows[~known], len(lv)] = 1.0
            blocks.append(block)

        X = np.hstack(blocks) if blocks else np.zeros((n, 0), dtype=np.float64)
        if X.shape[1] != len(self.feature_names):  # pragma: no cover - defensive
            raise RuntimeError(f"encoder produced {X.shape[1]} columns, expected {len(self.feature_names)}")
        return np.ascontiguousarray(X, dtype=np.float64)


def _normalise_categorical(series: pd.Series) -> pd.Series:
    """Canonicalise a categorical column to stripped, lower-cased **strings**.

    Extends ``schema.enforce_schema`` (which only touches ``str`` cells) by also
    stringifying non-string levels. That matters: level *learning* stringifies
    (``str(v)``), so if ``transform`` left booleans or integers untyped, every row of a
    non-string categorical column would miss its one-hot level and fall into
    ``__other`` -- a silently constant, useless block rather than an error.

    Nulls (``None``, ``NaN``, ``NaT``) are preserved as ``None`` so they land in
    ``__other`` by the documented rule.
    """

    def _one(value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, (str, bytes)):
            try:
                if bool(pd.isna(value)):
                    return None
            except (TypeError, ValueError):  # pragma: no cover - exotic array-likes
                pass
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        elif not isinstance(value, str):
            value = str(value)
        return value.strip().lower()

    return series.astype(object).map(_one)


def _partition_features(
    panel: pd.DataFrame, features: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split requested features into (numeric, categorical) using the schema then dtypes."""
    numeric: list[str] = []
    categorical: list[str] = []
    for col in features:
        if col in NUMERIC_FEATURES:
            numeric.append(col)
        elif col in CATEGORICAL_FEATURES:
            categorical.append(col)
        elif col in panel.columns and pd.api.types.is_numeric_dtype(panel[col]) and not pd.api.types.is_bool_dtype(panel[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return tuple(numeric), tuple(categorical)


def _build_names(
    numeric: Sequence[str], categorical: Sequence[str], levels: Mapping[str, Sequence[str]], other_token: str
) -> tuple[str, ...]:
    names = list(numeric)
    names += [f"{c}__isna" for c in numeric]
    for col in categorical:
        names += [f"{col}__{lv}" for lv in levels.get(col, ())]
        names.append(f"{col}__{other_token}")
    return tuple(names)


def build_design_matrix(
    panel: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    fit: bool = True,
    encoder: DesignMatrixEncoder | None = None,
    strict_columns: bool = True,
    other_token: str = "other",
) -> tuple[np.ndarray, list[str], DesignMatrixEncoder]:
    """Turn a panel into a dense, serving-stable model matrix.

    Numerics are median-imputed with an explicit ``"<col>__isna"`` indicator; categoricals
    are one-hot encoded against the **fixed** level set in
    :data:`prism.data.schema.CATEGORICAL_LEVELS`, with unknown or missing levels folded
    into ``"<col>__other"``. The width and column order therefore depend only on the
    encoder, never on the batch -- a request containing a brand-new ``plan_tier`` produces
    the same 23-feature-derived matrix as training did. See :class:`DesignMatrixEncoder`
    for the exact ordering rule.

    Parameters
    ----------
    panel : pandas.DataFrame
        Rows to encode.
    features : sequence of str or None, default None
        Feature columns to use. ``None`` selects :data:`prism.data.schema.FEATURES`
        intersected with ``panel.columns`` (a warning names anything missing), so a
        partially-built panel still works during pipeline bring-up. Ignored when
        ``fit=False``: the encoder's own column list wins.
    fit : bool, default True
        Fit a new encoder (medians and levels) before transforming.
    encoder : DesignMatrixEncoder or None
        Required when ``fit=False``.
    strict_columns : bool, default True
        With ``fit=False``, raise if the frame lacks a column the encoder expects. Set
        False to impute the whole column instead (useful for degraded serving).
    other_token : str, default "other"
        Suffix of the catch-all one-hot column.

    Returns
    -------
    X : numpy.ndarray
        ``(n_rows, n_features)`` float64, C-contiguous.
    feature_names : list of str
        Output column names in matrix order.
    encoder : DesignMatrixEncoder
        Fitted encoder (the one passed in, when ``fit=False``).

    Raises
    ------
    ValueError
        If ``fit=False`` and no encoder is supplied, or if no usable feature is found.

    Examples
    --------
    >>> X, names, enc = build_design_matrix(panel)                      # doctest: +SKIP
    >>> X2, names2, _ = build_design_matrix(batch, fit=False, encoder=enc)  # doctest: +SKIP
    >>> names == names2 and X.shape[1] == X2.shape[1]                   # doctest: +SKIP
    True
    """
    if not fit:
        if encoder is None:
            raise ValueError("build_design_matrix(fit=False) requires a fitted encoder")
        X = encoder.transform(panel, strict_columns=strict_columns)
        return X, list(encoder.feature_names), encoder

    if features is None:
        selected = [c for c in FEATURES if c in panel.columns]
        absent = [c for c in FEATURES if c not in panel.columns]
        if absent:
            _LOG.warning("panel is missing %d schema feature(s): %s", len(absent), absent)
    else:
        selected = list(features)
        unknown = [c for c in selected if c not in panel.columns]
        if unknown:
            raise KeyError(f"panel is missing requested feature columns: {unknown}")
    if not selected:
        raise ValueError("no usable feature columns found in the panel")

    numeric, categorical = _partition_features(panel, selected)

    medians: dict[str, float] = {}
    for col in numeric:
        vals = pd.to_numeric(panel[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        medians[col] = float(np.median(finite)) if finite.size else 0.0

    levels: dict[str, tuple[str, ...]] = {}
    for col in categorical:
        if col in CATEGORICAL_LEVELS:
            source_levels: Sequence[Any] = CATEGORICAL_LEVELS[col]
        else:
            source_levels = sorted(str(v) for v in _normalise_categorical(panel[col]).dropna().unique())
        # dict.fromkeys de-duplicates while preserving order; duplicate levels would
        # otherwise produce duplicate column names and break the encoder's get_indexer.
        levels[col] = tuple(dict.fromkeys(str(v) for v in source_levels))

    enc = DesignMatrixEncoder(
        numeric_features=tuple(numeric),
        categorical_features=tuple(categorical),
        medians=medians,
        levels=levels,
        feature_names=_build_names(numeric, categorical, levels, other_token),
        other_token=other_token,
    )
    X = enc.transform(panel, strict_columns=strict_columns)
    return X, list(enc.feature_names), enc


def design_matrix_frame(
    panel: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    fit: bool = True,
    encoder: DesignMatrixEncoder | None = None,
    strict_columns: bool = True,
    other_token: str = "other",
    index: pd.Index | None = None,
    return_encoder: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, DesignMatrixEncoder]:
    """:func:`build_design_matrix` returning a named DataFrame instead of a bare array.

    Convenient for SHAP, love plots and any diagnostic where the column names matter.

    Parameters
    ----------
    panel : pandas.DataFrame
        Rows to encode.
    features, fit, encoder, strict_columns, other_token
        Passed through to :func:`build_design_matrix`.
    index : pandas.Index or None, default None
        Index for the result; defaults to ``panel.index`` so the frame stays joinable.
    return_encoder : bool, default False
        Also return the fitted encoder.

    Returns
    -------
    pandas.DataFrame or tuple
        The named design matrix, or ``(frame, encoder)`` when ``return_encoder``.
    """
    X, names, enc = build_design_matrix(
        panel,
        features,
        fit=fit,
        encoder=encoder,
        strict_columns=strict_columns,
        other_token=other_token,
    )
    frame = pd.DataFrame(X, columns=names, index=panel.index if index is None else index)
    return (frame, enc) if return_encoder else frame


# ======================================================================================
# Smoke test
# ======================================================================================
def _synthetic_events(rng: np.random.Generator, n_customers: int, n_events: int) -> pd.DataFrame:
    """Build a small inline event log (no dependency on prism.data.dgp)."""
    origin = pd.Timestamp("2022-01-01")
    cust = np.array([f"C{i:04d}" for i in rng.integers(0, n_customers, size=n_events)])
    offsets = rng.integers(0, 400 * 86_400, size=n_events).astype("int64")
    ts = origin.value + offsets * 1_000_000_000 + rng.integers(0, 1_000_000, size=n_events).astype("int64")
    kinds = rng.choice(
        ["purchase", "session", "support_ticket", "payment_failure"],
        size=n_events,
        p=[0.42, 0.44, 0.08, 0.06],
    )
    amount = np.round(rng.gamma(2.4, 21.0, size=n_events), 2)
    amount[kinds != "purchase"] = np.nan
    amount[rng.random(n_events) < 0.03] = np.nan  # genuine nulls, to exercise skipna paths
    category = rng.choice(["grocery", "beauty", "tech", "home", "apparel"], size=n_events)
    n_items = rng.integers(1, 9, size=n_events).astype(np.float64)
    return pd.DataFrame(
        {
            "customer_id": cust,
            "event_ts": pd.to_datetime(ts),
            "event_type": kinds,
            "amount": amount,
            "category": category,
            "n_items": n_items,
        }
    )


def _brute_force(events: pd.DataFrame, spine: pd.DataFrame, spec: FeatureSpec, ts_col: str) -> np.ndarray:
    """Deliberately naive O(n_spine * n_events) reference implementation."""
    ent = events[spec.entity].to_numpy()
    ts = events["event_ts"].to_numpy("datetime64[ns]").astype("int64")
    etype = events["event_type"].to_numpy()
    src = events[spec.source].to_numpy(dtype=object)
    src_num = pd.to_numeric(events[spec.source], errors="coerce").to_numpy(dtype=np.float64)
    src_null = pd.isna(events[spec.source]).to_numpy()
    row_idx = np.arange(len(events))

    spine_ent = spine[spec.entity].to_numpy()
    spine_ts = pd.to_datetime(spine[ts_col]).to_numpy("datetime64[ns]").astype("int64")

    out = np.full(len(spine), np.nan, dtype=np.float64)
    for i in range(len(spine)):
        t = spine_ts[i]
        mask = (ent == spine_ent[i]) & (ts <= t)
        if spec.event_types is not None:
            mask &= np.isin(etype, list(spec.event_types))
        if spec.window_days is not None:
            mask &= ts > t - spec.window_days * _NS_PER_DAY
        if not mask.any():
            if spec.fill_value is not None:
                out[i] = float(spec.fill_value)
            else:
                out[i] = 0.0 if spec.agg in ("sum", "count", "nunique") else np.nan
            continue
        vals_null = src_null[mask]
        valid = ~vals_null
        if spec.agg == "count":
            out[i] = float(valid.sum())
        elif spec.agg == "sum":
            out[i] = float(np.nansum(src_num[mask][valid])) if valid.any() else 0.0
        elif spec.agg == "mean":
            out[i] = float(np.mean(src_num[mask][valid])) if valid.any() else np.nan
        elif spec.agg == "max":
            out[i] = float(np.max(src_num[mask][valid])) if valid.any() else np.nan
        elif spec.agg == "nunique":
            out[i] = float(len(set(src[mask][valid].tolist())))
        elif spec.agg == "last":
            if not valid.any():
                out[i] = np.nan
            else:
                cand_ts = ts[mask][valid]
                cand_idx = row_idx[mask][valid]
                best = np.lexsort((cand_idx, cand_ts))[-1]
                out[i] = float(src_num[mask][valid][best])
        elif spec.agg == "days_since":
            out[i] = float((t - ts[mask].max()) / _NS_PER_DAY)
        else:  # pragma: no cover
            raise AssertionError(spec.agg)
    return out


def _synthetic_panel(rng: np.random.Generator, n_rows: int) -> pd.DataFrame:
    """Build a tiny gold-panel-shaped frame carrying all 23 schema features."""
    frame = pd.DataFrame({"customer_id": [f"C{i % 60:04d}" for i in range(n_rows)]})
    frame["period"] = rng.integers(0, 24, size=n_rows).astype("int64")
    for col in NUMERIC_FEATURES:
        vals = rng.normal(10.0, 3.0, size=n_rows)
        vals[rng.random(n_rows) < 0.12] = np.nan
        frame[col] = vals
    for col in CATEGORICAL_FEATURES:
        frame[col] = rng.choice(list(CATEGORICAL_LEVELS[col]), size=n_rows)
    return frame


if __name__ == "__main__":  # pragma: no cover - smoke test
    import time

    from prism.utils.seeds import as_rng

    t_start = time.perf_counter()
    rng = as_rng(11)
    pd.set_option("display.width", 200)

    # ---------------------------------------------------------------- 1. build inputs
    events = _synthetic_events(rng, n_customers=60, n_events=2500)
    spine_ids = [f"C{i:04d}" for i in rng.integers(0, 66, size=300)]  # 6 ids never seen
    spine_ts = pd.to_datetime(
        pd.Timestamp("2022-03-01").value + rng.integers(0, 330 * 86_400, size=300).astype("int64") * 1_000_000_000
    )
    spine = pd.DataFrame({"customer_id": spine_ids, "as_of_date": spine_ts})

    specs = default_feature_specs() + [
        # exercises the O(1) expanding-nunique fast path and the empty-window override
        FeatureSpec("categories_lifetime", "category", "nunique", None, event_types=("purchase",)),
        FeatureSpec("tickets_30d_filled", "event_type", "count", 30, event_types=("support_ticket",),
                    fill_value=-1.0),
    ]
    store = PointInTimeFeatureStore(events, specs)
    joined = store.as_of_join(spine)
    print(f"events={len(events)}  spine={len(spine)}  specs={len(specs)}  joined_cols={joined.shape[1]}")

    # ------------------------------------------- 2. KEY TEST: match the brute force
    rows: list[dict[str, object]] = []
    for spec in specs:
        ref = _brute_force(events, spine, spec, "as_of_date")
        got = pd.to_numeric(joined[spec.name], errors="coerce").to_numpy(dtype=np.float64)
        both_nan = np.isnan(ref) & np.isnan(got)
        close = both_nan | np.isclose(ref, got, rtol=1e-9, atol=1e-9, equal_nan=True)
        n_bad = int((~close).sum())
        max_err = float(np.nanmax(np.abs(np.where(both_nan, 0.0, ref - got)))) if len(ref) else 0.0
        rows.append({"feature": spec.name, "agg": spec.agg, "window": spec.window_label(),
                     "mismatches": n_bad, "max_abs_err": round(max_err, 12)})
        assert n_bad == 0, f"{spec.name}: {n_bad} rows differ from the brute-force reference"
    print(f"\n[1] as_of_join vs brute force ({len(spine)} spine rows x {len(specs)} specs)")
    print(pd.DataFrame(rows).to_string(index=False))
    assert (joined.loc[joined["frequency_12m"] == 0, "tickets_30d_filled"] >= -1.0).all()
    assert (joined["tickets_30d_filled"] == -1.0).any(), "fill_value path was never taken"

    # source="panel" pass-through, and idempotence of a repeated join
    spine_pt = spine.assign(plan_tier="pro")
    pt_store = PointInTimeFeatureStore(events, [FeatureSpec("plan_tier", "panel", "last", None)])
    assert (pt_store.as_of_join(spine_pt)["plan_tier"] == "pro").all()
    assert joined["monetary_12m"].equals(store.as_of_join(spine)["monetary_12m"]), "join is not idempotent"

    # ------------------------------------------------------ 3. leakage: pass / raise
    report = assert_no_leakage(store, spine)
    assert not report["leak_detected"].any()
    shown = report[["feature", "n_rows", "max_event_ts_used", "as_of_max", "future_rows_used",
                    "leak_detected", "max_gap_days", "corr_with_future"]]
    print("\n[2] leakage_report (honest store) -- max_gap_days <= 0 everywhere")
    print(shown.to_string(index=False))

    leaky = PointInTimeFeatureStore(events, specs, leak_lookahead_days=45.0)
    leaky_report = leaky.leakage_report(spine)
    assert bool(leaky_report["leak_detected"].any()), "leak detector failed to fire"
    try:
        assert_no_leakage(leaky, spine)
        raise AssertionError("assert_no_leakage should have raised on the look-ahead store")
    except LeakageError as exc:
        print(f"\n[3] LeakageError raised as expected on a 45-day look-ahead store: "
              f"{len(leaky_report.loc[leaky_report['leak_detected']])} feature(s) flagged")
        print("    ", str(exc)[:150].replace("\n", " "), "...")
    leaky_join = leaky.as_of_join(spine)  # warn-only, so the breakage stays inspectable
    # an all-history window with look-ahead can only ever see MORE revenue
    assert leaky_join["monetary_lifetime"].sum() > joined["monetary_lifetime"].sum()
    leaky._leak_warn_only = False  # now behave like a production store: fail closed
    try:
        leaky.as_of_join(spine)
        raise AssertionError("as_of_join should fail closed on a point-in-time violation")
    except LeakageError:
        print("    as_of_join also fails closed (LeakageError) when not in warn-only mode")

    # --------------------------------------------------------------- 4. splits
    panel = _synthetic_panel(rng, 900)
    plain = temporal_split(panel, train_end=13, valid_end=17)
    purged = purged_temporal_split(panel, train_end=13, valid_end=17, embargo=2)
    assert list(plain["period"]) == list(panel["period"]), "temporal_split must not shuffle"
    assert "split" not in panel.columns, "temporal_split must not mutate its input"
    assert set(plain.loc[plain["split"] == "train", "period"]) <= set(range(0, 14))
    assert set(purged.loc[purged["split"] == "train", "period"]) <= set(range(0, 12))
    counts = pd.DataFrame({"plain": plain["split"].value_counts(), "purged": purged["split"].value_counts()}).fillna(0)
    print("\n[4] split sizes (embargo=2)")
    print(counts.astype(int).to_string())

    # --------------------------------------------------------------- 5. design matrix
    X_tr, names_tr, enc = build_design_matrix(panel)
    serve = panel.iloc[:25].copy()
    serve.loc[serve.index[:5], "plan_tier"] = "platinum"       # unseen level
    serve.loc[serve.index[5:8], "device"] = None               # missing level
    serve.loc[serve.index[8:12], "nps"] = np.nan               # missing numeric
    X_sv, names_sv, enc2 = build_design_matrix(serve, fit=False, encoder=enc)
    assert names_tr == names_sv and X_tr.shape[1] == X_sv.shape[1] and enc2 is enc
    other_col = names_tr.index("plan_tier__other")
    assert X_sv[:5, other_col].sum() == 5.0, "unseen category did not land in __other"
    dev_block = [i for i, n in enumerate(names_tr) if n.startswith("device__")]
    assert np.allclose(X_sv[:, dev_block].sum(axis=1), 1.0), "one-hot block must sum to 1"
    nps_isna = names_tr.index("nps__isna")
    assert X_sv[8:12, nps_isna].sum() == 4.0
    assert np.isfinite(X_tr).all() and X_tr.dtype == np.float64

    import pickle

    enc_round = pickle.loads(pickle.dumps(enc))
    assert np.allclose(enc_round.transform(serve), X_sv), "encoder must be picklable and stable"
    dm = design_matrix_frame(serve, fit=False, encoder=enc)
    assert list(dm.columns) == names_tr and len(dm) == len(serve)
    print(f"\n[5] design matrix: train {X_tr.shape} -> serve {X_sv.shape}, {len(names_tr)} stable columns")
    print("    first 12 names:", names_tr[:12])
    print("    one-hot tail  :", names_tr[-9:])

    # --------------------------------------------------------------- 6. RFM summary
    summary = customer_summary(events, as_of="2022-12-31", horizon_months=12)
    assert {"customer_id", "frequency", "recency", "T", "monetary_value"} <= set(summary.columns)
    assert (summary["recency"] <= summary["T"] + 1e-9).all()
    assert (summary["frequency"] >= 0).all()
    assert (summary.loc[summary["frequency"] == 0, "monetary_value"] == 0).all()
    print(f"\n[6] customer_summary: {len(summary)} customers | "
          f"freq mean {summary['frequency'].mean():.2f} | "
          f"recency/T mean {(summary['recency'] / summary['T'].clip(lower=1e-9)).mean():.2f} | "
          f"monetary mean {summary['monetary_value'].mean():.2f}")

    # ------------------------------------------- 7. hard invariants (regression guards)
    checks: list[tuple[str, bool]] = []

    def _check(label: str, ok: bool) -> None:
        # raise rather than `assert`, so these survive `python -O` (where the bare
        # asserts above are stripped and a smoke test can "pass" while checking nothing)
        if not ok:
            raise AssertionError(f"invariant failed: {label}")
        checks.append((label, ok))

    # (a) purity: the store must not touch the caller's event log or spine.
    ev_before, sp_before = events.copy(deep=True), spine.copy(deep=True)
    PointInTimeFeatureStore(events, specs).as_of_join(spine)
    _check("caller frames untouched", events.equals(ev_before) and spine.equals(sp_before))

    # (b) determinism: an independently constructed store is bit-identical.
    _check("bit-identical across stores", PointInTimeFeatureStore(events, specs).as_of_join(spine).equals(joined))

    # (c) point-in-time monotonicity: an all-history aggregate can only ever grow as the
    #     decision time moves forward. A backward step here IS leakage, by definition.
    mono_spine = pd.DataFrame(
        {"customer_id": ["C0003"] * 24,
         "as_of_date": pd.Timestamp("2022-01-15") + pd.to_timedelta(np.arange(24) * 15, "D")}
    )
    mono = store.as_of_join(mono_spine)
    _check("all-history sum monotone in as_of", bool((mono["monetary_lifetime"].diff().dropna() >= -1e-9).all()))
    _check("all-history nunique monotone in as_of",
           bool((mono["categories_lifetime"].diff().dropna() >= 0).all()))

    # (d) boundary semantics: (as_of - w, as_of]. as_of == event_ts is IN, one ns later is OUT.
    b0 = pd.Timestamp("2022-06-01").value
    bnd_ev = pd.DataFrame({"customer_id": ["z"] * 3,
                           "event_ts": pd.to_datetime([b0 - 7 * _NS_PER_DAY, b0, b0 + 1]),
                           "event_type": ["purchase"] * 3, "amount": [100.0, 1.0, 999.0]})
    bnd_sp = pd.DataFrame({"customer_id": ["z"], "as_of_date": pd.to_datetime([b0])})
    bnd = PointInTimeFeatureStore(bnd_ev, [FeatureSpec("w7", "amount", "sum", 7),
                                           FeatureSpec("wall", "amount", "sum", None)],
                                  validate=False).as_of_join(bnd_sp)
    _check("as_of itself included, +1ns excluded", float(bnd["wall"].iloc[0]) == 101.0)
    _check("window left edge is exclusive", float(bnd["w7"].iloc[0]) == 1.0)

    # (e) mean must divide by exactly the values sum added up (dirty non-numeric cell).
    dirty = pd.DataFrame({"customer_id": ["d"] * 3,
                          "event_ts": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]),
                          "event_type": ["purchase"] * 3, "amount": [10.0, "n/a", 30.0]})
    dj = PointInTimeFeatureStore(
        dirty, [FeatureSpec("m", "amount", "mean", None), FeatureSpec("s", "amount", "sum", None)],
        validate=False,
    ).as_of_join(pd.DataFrame({"customer_id": ["d"], "as_of_date": [pd.Timestamp("2022-01-04")]}))
    _check("mean denominator == numeric-valid count",
           float(dj["m"].iloc[0]) == pd.to_numeric(dirty["amount"], errors="coerce").mean()
           and float(dj["s"].iloc[0]) == 40.0)

    # (f) range-max segment tree, including odd lengths where the build order matters.
    rmax_ok = True
    for n_v in (1, 2, 3, 5, 9, 25, 101):
        vals = rng.normal(size=n_v) * 10.0
        tree = _RangeMax(vals)
        q_lo = np.array([a for a in range(n_v) for _ in range(a, n_v)], dtype=np.int64)
        q_hi = np.array([b + 1 for a in range(n_v) for b in range(a, n_v)], dtype=np.int64)
        rmax_ok &= bool(np.allclose(tree.query(q_lo, q_hi),
                                    [vals[a:b].max() for a, b in zip(q_lo, q_hi)]))
    _check("segment-tree range max exact for odd and even n", rmax_ok)

    # (g) timezone discipline: aware/naive mixing is refused, aware/aware is offset-invariant.
    tz_ev = bnd_ev.assign(event_ts=bnd_ev["event_ts"].dt.tz_localize("UTC"))
    tz_store = PointInTimeFeatureStore(tz_ev, [FeatureSpec("wall", "amount", "sum", None)], validate=False)
    try:
        tz_store.as_of_join(bnd_sp)
        raise AssertionError("tz-aware events joined to a naive spine must raise")
    except ValueError:
        pass
    tz_sp = bnd_sp.assign(as_of_date=bnd_sp["as_of_date"].dt.tz_localize("UTC"))
    a_utc = float(tz_store.as_of_join(tz_sp)["wall"].iloc[0])
    a_tok = float(tz_store.as_of_join(tz_sp.assign(
        as_of_date=tz_sp["as_of_date"].dt.tz_convert("Asia/Tokyo")))["wall"].iloc[0])
    _check("tz-aware join is offset-invariant", a_utc == a_tok == 101.0)

    # (h) empty spine and an entity absent from the log are answers, not crashes.
    empty_sp = spine.iloc[:0]
    _check("empty spine round-trips", len(store.as_of_join(empty_sp)) == 0
           and len(store.leakage_report(empty_sp)) == len(specs))
    unseen = store.as_of_join(pd.DataFrame({"customer_id": ["NOT_A_CUSTOMER"],
                                            "as_of_date": [pd.Timestamp("2022-06-01")]}))
    _check("unknown entity -> empty window, not a crash",
           float(unseen["monetary_lifetime"].iloc[0]) == 0.0 and bool(np.isnan(unseen["recency_days"].iloc[0])))

    # (i) purged split: exact band layout, drop=True, and embargo=0 == temporal_split.
    ladder = pd.DataFrame({"period": np.arange(24)})
    dropped = purged_temporal_split(ladder, 13, 17, embargo=2, drop=True)
    _check("embargo bands removed exactly",
           set(dropped.loc[dropped["split"] == "train", "period"]) == set(range(0, 12))
           and set(dropped.loc[dropped["split"] == "valid", "period"]) == set(range(14, 16))
           and set(dropped.loc[dropped["split"] == "test", "period"]) == set(range(18, 24)))
    _check("embargo=0 reproduces temporal_split",
           bool((purged_temporal_split(ladder, 13, 17, embargo=0)["split"]
                 == temporal_split(ladder, 13, 17)["split"]).all()))
    _check("no train period may sit inside a valid label window",
           int(purged["period"].loc[purged["split"] == "train"].max())
           + 2 <= int(purged["period"].loc[purged["split"] == "valid"].min()))

    # (j) design matrix: non-string categoricals must hit real levels, never all-__other;
    #     and a whole missing column must degrade rather than crash when asked to.
    odd_cats = pd.DataFrame({"x": np.arange(12, dtype=float),
                             "flag": np.array([True, False] * 6),
                             "grp": np.array([1, 2, 3, 4] * 3, dtype=object)})
    X_odd, names_odd, _ = build_design_matrix(odd_cats, ["x", "flag", "grp"])
    _check("non-string categorical levels are matched, not dumped in __other",
           X_odd[:, names_odd.index("flag__other")].sum() == 0.0
           and X_odd[:, names_odd.index("grp__other")].sum() == 0.0)
    for cat in CATEGORICAL_FEATURES:
        blk = [i for i, nm in enumerate(names_tr) if nm.startswith(cat + "__")]
        _check(f"{cat} one-hot block is a simplex", bool(np.allclose(X_tr[:, blk].sum(axis=1), 1.0)))
    degraded = serve.drop(columns=["nps", "device"])
    X_deg, names_deg, _ = build_design_matrix(degraded, fit=False, encoder=enc, strict_columns=False)
    _check("degraded serving keeps the matrix width",
           X_deg.shape == X_sv.shape and names_deg == names_tr
           and X_deg[:, names_tr.index("nps__isna")].min() == 1.0)
    try:
        build_design_matrix(degraded, fit=False, encoder=enc)
        raise AssertionError("strict_columns=True must reject a missing column")
    except KeyError:
        pass

    # (k) RFM conventions, hand-computed: 3 purchases 10 days apart, observed at day 30.
    rfm_ev = pd.DataFrame({"customer_id": ["a", "a", "a", "b"],
                           "event_ts": pd.to_datetime(["2022-01-01", "2022-01-11", "2022-01-21", "2022-02-01"]),
                           "event_type": ["purchase"] * 4, "amount": [10.0, 20.0, 30.0, 5.0]})
    rfm = customer_summary(rfm_ev, "2022-01-31", 12, time_unit="days").set_index("customer_id")
    _check("frequency = repeat purchases", float(rfm.loc["a", "frequency"]) == 2.0)
    _check("recency = age at last purchase", abs(float(rfm.loc["a", "recency"]) - 20.0) < 1e-9)
    _check("T = age at observation", abs(float(rfm.loc["a", "T"]) - 30.0) < 1e-9)
    _check("monetary = mean of REPEAT transactions", abs(float(rfm.loc["a", "monetary_value"]) - 25.0) < 1e-9)
    _check("purchases after as_of are invisible", "b" not in rfm.index)

    print(f"\n[7] invariants: {len(checks)} hard checks passed "
          f"(purity, determinism, PIT monotonicity, boundary, tz, splits, encoder, RFM)")

    print(f"\nfeatures.py OK in {time.perf_counter() - t_start:.1f}s")
