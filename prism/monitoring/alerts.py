"""Threshold-based alerting over PRISM's drift, decay and causal-stability reports.

This is the *decision* half of :mod:`prism.monitoring`. :mod:`prism.monitoring.drift`
measures things; this module turns those measurements into a small, ordered list of
:class:`Alert` objects, each of which states **what tripped**, **by how much**, and
**what to do about it**, and then renders them for a log line, a CI comment or the
generated monitoring report.

Design notes
------------
* **Every input is optional.** Monitoring runs are partial in practice -- a nightly job
  may have a drift report but no labelled outcomes yet, and a causal re-evaluation may
  have CATE stability but no fresh feature snapshot. ``evaluate_alerts(None, decay)`` is
  a supported call, not a degenerate one.
* **Duck-typed reports.** The reports arrive as :class:`pandas.DataFrame` objects from
  :mod:`prism.monitoring.drift`, but lists of dicts and ``{feature: metrics}`` mappings
  are accepted too, and column names are resolved through alias tables. This module
  therefore does not import the drift module and cannot be broken by a column rename.
* **Ordering is by consequence, not by discovery order.** Alerts sort by severity, then
  by :attr:`Alert.exceedance` -- how far past its threshold the value sits, scaled so the
  number is comparable across metrics with wildly different units (PSI, a p-value, a
  currency amount).
* **Actions are concrete.** ``recommended_action`` names the artefact to rebuild or the
  decision to hold. "Investigate" is not an action.
* **Unmeasured is not clean.** A report that arrives with an all-NaN column, a renamed
  column or an unordered grouping axis raises its own alert. Silence is reserved for
  metrics that were actually computed and were actually inside their thresholds.

The module is dependency-light on purpose: numpy and pandas only, so the alerting layer
still runs when the modelling stack is unavailable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from prism.utils.logging import get_logger

__all__ = [
    "Alert",
    "SEVERITIES",
    "SEVERITY_RANK",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_POLICY_THRESHOLDS",
    "THRESHOLD_ALIASES",
    "resolve_thresholds",
    "evaluate_alerts",
    "render_alerts",
    "alert_summary",
    "should_block_deployment",
    "to_markdown",
]

_LOG = get_logger("monitoring.alerts")

#: Severity vocabulary, least severe first.
SEVERITIES: tuple[str, ...] = ("info", "warn", "critical")

#: Sort key per severity -- lower sorts first, so ``critical`` leads the table.
SEVERITY_RANK: dict[str, int] = {"critical": 0, "warn": 1, "info": 2}

#: Canonical thresholds from SPEC.md section 6. Override any subset via
#: ``evaluate_alerts(..., thresholds={"psi_alert": 0.4})``.
DEFAULT_THRESHOLDS: dict[str, Any] = {
    "psi_warn": 0.10,
    "psi_alert": 0.25,
    "ks_p_alert": 0.01,
    "missing_rate_jump": 0.10,
    "cate_rank_corr_alert": 0.70,
    "sign_flip_alert": 0.15,
    "auc_drop_alert": 0.05,
    "calibration_slope_band": (0.8, 1.25),
}

#: Additional thresholds used only when a ``policy_metrics`` dict is supplied. Kept in a
#: separate mapping so :data:`DEFAULT_THRESHOLDS` stays exactly the spec'd set; both are
#: merged into the effective thresholds, so callers override them the same way.
DEFAULT_POLICY_THRESHOLDS: dict[str, Any] = {
    "policy_value_ci_low_min": 0.0,
    "policy_uplift_vs_baseline_min": 0.0,
    "roi_warn": 0.0,
    "budget_utilisation_info": 0.60,
}

#: Tolerated spellings for threshold keys (``prism.config.MonitoringConfig`` uses
#: ``ks_alert_p``; SPEC.md section 6 uses ``ks_p_alert``).
THRESHOLD_ALIASES: dict[str, str] = {
    "ks_alert_p": "ks_p_alert",
    "ks_p_value_alert": "ks_p_alert",
    "psi_critical": "psi_alert",
    "missing_rate_jump_alert": "missing_rate_jump",
    "auc_drop": "auc_drop_alert",
    "sign_flip_rate_alert": "sign_flip_alert",
    "cate_rank_correlation_alert": "cate_rank_corr_alert",
    "budget_utilization_info": "budget_utilisation_info",
}

# Column alias tables. First match wins.
_FEATURE_KEYS = ("feature", "column", "name", "variable")
_PSI_KEYS = ("psi", "psi_value", "population_stability_index")
_KS_P_KEYS = ("ks_p", "ks_pvalue", "ks_p_value", "p_value", "pvalue", "p")
_MISS_JUMP_KEYS = ("missing_rate_jump", "missing_rate_delta", "missing_delta", "d_missing_rate")
_MISS_REF_KEYS = ("missing_rate_ref", "missing_rate_reference", "missing_ref", "ref_missing_rate")
_MISS_CUR_KEYS = ("missing_rate_cur", "missing_rate_current", "missing_cur", "cur_missing_rate")
_AUC_KEYS = ("auc", "roc_auc", "auc_roc", "c_index", "concordance")
_SLOPE_KEYS = ("calibration_slope", "cal_slope", "slope")
# Only genuinely ordered axes belong here. ``group``/``segment``/``bin`` are deliberately
# absent: sorting a segment label alphabetically and then calling the first-to-last
# difference "decay" invents a trend that the report never claimed.
_PERIOD_KEYS = ("period", "period_index", "as_of_date", "date", "month")
_RANK_CORR_KEYS = ("rank_corr", "rank_correlation", "spearman", "spearman_corr", "kendall_tau")
_SIGN_FLIP_KEYS = ("sign_flip_rate", "sign_flip", "sign_flips", "frac_sign_flip")
_POLICY_CI_KEYS = ("policy_value_ci_low", "ci_low", "value_ci_low")
_POLICY_VALUE_KEYS = ("policy_value", "value", "net_value")
_POLICY_BASE_KEYS = ("baseline_value", "incumbent_value", "control_value")
_ROI_KEYS = ("roi", "return_on_investment")
_UTIL_KEYS = ("budget_utilisation", "budget_utilization", "budget_used_frac")
_EPS = 1e-12


# ======================================================================================
# Alert
# ======================================================================================
@dataclass
class Alert:
    """One tripped monitoring threshold.

    Parameters
    ----------
    name : str
        Stable machine-readable identifier, ``"<rule>:<subject>"`` where a subject exists
        (for example ``"psi:monetary_12m"``). Stable across runs so a CI job can diff.
    severity : str
        One of ``"info"``, ``"warn"``, ``"critical"``. Only ``"critical"`` blocks a
        deployment -- see :func:`should_block_deployment`.
    value : float
        The observed metric value that tripped the rule.
    threshold : float
        The threshold it was compared against. For a two-sided band this is the breached
        edge, not the whole band.
    message : str
        Human-readable sentence naming what tripped and by how much.
    recommended_action : str, optional
        The concrete next step: an artefact to rebuild, a job to inspect, a decision to
        hold. Never a bare "investigate".
    direction : {'above', 'below'}, optional
        Whether the rule fires when the value exceeds the threshold (``"above"``, the
        default) or falls below it (``"below"``). Drives :attr:`exceedance`.

    Attributes
    ----------
    exceedance : float
        Scale-free breach size in ``[0, 1]``; the secondary sort key.

    Examples
    --------
    >>> a = Alert("psi:nps", "warn", 0.18, 0.10, "PSI 0.18 exceeds 0.10.")
    >>> a.severity, round(a.exceedance, 3)
    ('warn', 0.444)
    """

    name: str
    severity: str
    value: float
    threshold: float
    message: str
    recommended_action: str = ""
    direction: str = "above"

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_RANK:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")
        if self.direction not in ("above", "below"):
            raise ValueError(f"direction must be 'above' or 'below', got {self.direction!r}")
        self.value = float(self.value)
        self.threshold = float(self.threshold)

    @property
    def exceedance(self) -> float:
        """How far the value breached the threshold, scaled to ``[0, 1]``.

        The raw gap is divided by ``max(|threshold|, |value|)`` rather than by the
        threshold alone, so a rule with a zero threshold (a currency amount that must stay
        positive, say) cannot dominate the ordering with an unbounded ratio, and metrics
        on different scales stay comparable.

        Returns
        -------
        float
            ``0.0`` when the value is on the compliant side of the threshold.
        """
        gap = (self.value - self.threshold) if self.direction == "above" else (self.threshold - self.value)
        if not np.isfinite(gap) or gap <= 0.0:
            return 0.0
        denom = max(abs(self.threshold), abs(self.value), _EPS)
        return float(min(gap / denom, 1.0))

    @property
    def sort_key(self) -> tuple[int, float, str]:
        """Ordering key: severity first, then breach size descending, then name."""
        return (SEVERITY_RANK[self.severity], -self.exceedance, self.name)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of this alert.

        Returns
        -------
        dict
            The dataclass fields plus the derived ``exceedance``. Values are plain Python
            floats/strings, so the result can go straight into ``json.dump`` or
            ``pd.DataFrame(...)``.
        """
        out = asdict(self)
        out["exceedance"] = round(self.exceedance, 6)
        return out

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.name}: {self.message}"


# ======================================================================================
# Normalisation helpers
# ======================================================================================
def resolve_thresholds(thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge user overrides onto the defaults, normalising known key aliases.

    Parameters
    ----------
    thresholds : mapping, optional
        Subset of threshold keys to override. Unknown keys are kept (so downstream custom
        rules can read them) but logged once at debug level.

    Returns
    -------
    dict
        The effective thresholds: :data:`DEFAULT_THRESHOLDS` and
        :data:`DEFAULT_POLICY_THRESHOLDS` with overrides applied.
    """
    eff: dict[str, Any] = {**DEFAULT_THRESHOLDS, **DEFAULT_POLICY_THRESHOLDS}
    if not thresholds:
        return eff
    for key, val in dict(thresholds).items():
        canon = THRESHOLD_ALIASES.get(key, key)
        if canon not in eff:
            _LOG.debug("unknown threshold key %r kept as-is", key)
        eff[canon] = val
    return eff


def _promote_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose a meaningful DataFrame index as an ordinary column.

    ``DataFrame.to_dict(orient="records")`` drops the index, so a drift report built the
    idiomatic pandas way -- ``report.set_index("feature")`` or
    ``report.groupby("feature").agg(...)`` -- would arrive here with its feature names
    thrown away and every alert would be named ``psi:feature_0``. Alert names are meant to
    be stable identifiers a CI job can diff, so a positional name is a silent regression,
    not a cosmetic one. A plain ``RangeIndex`` carries no information and is left alone.
    """
    idx = frame.index
    if isinstance(idx, (pd.MultiIndex, pd.RangeIndex)):
        return frame
    if _pick(frame.columns, _FEATURE_KEYS) is not None or _pick(frame.columns, _PERIOD_KEYS) is not None:
        return frame
    if idx.name:
        name = str(idx.name)
    elif isinstance(idx, pd.DatetimeIndex):
        name = "date"
    elif idx.dtype == object or isinstance(idx.dtype, pd.CategoricalDtype):
        name = "feature"
    else:
        # A bare integer index is row position after a filter, not a label. Nothing to say.
        return frame
    if name in frame.columns:
        return frame
    out = frame.copy()
    out.insert(0, name, list(idx))
    return out.reset_index(drop=True)


def _as_frame(report: Any, *, what: str) -> pd.DataFrame | None:
    """Coerce a report-like object into a DataFrame, or return ``None``."""
    if report is None:
        return None
    if isinstance(report, pd.DataFrame):
        return _promote_index(report) if not report.empty else None
    if isinstance(report, pd.Series):
        return report.to_frame().T
    if isinstance(report, Mapping):
        values = list(report.values())
        if not values:
            return None  # an empty mapping is "no report", not "a report of nothing"
        if all(isinstance(v, Mapping) for v in values):
            frame = pd.DataFrame.from_dict(dict(report), orient="index")
            if _pick(frame.columns, _FEATURE_KEYS) is None:
                frame.insert(0, "feature", list(report.keys()))
            return frame.reset_index(drop=True)
        return pd.DataFrame([dict(report)])
    if isinstance(report, Sequence) and not isinstance(report, (str, bytes)):
        rows = [dict(r) for r in report if isinstance(r, Mapping)]
        frame = pd.DataFrame(rows) if rows else None
        return frame if (frame is not None and not frame.empty) else None
    _LOG.warning("ignoring %s of unsupported type %s", what, type(report).__name__)
    return None


def _pick(container: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    """Return the first key in ``keys`` present in ``container``."""
    for k in keys:
        if k in container:
            return k
    return None


def _num(row: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    """Read the first present key from ``row`` as a finite float, else ``None``."""
    key = _pick(row, keys)
    if key is None:
        return None
    raw = row[key]
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _band(value: Any) -> tuple[float, float]:
    """Coerce a calibration-slope band into an ordered ``(low, high)`` pair."""
    lo, hi = (float(value[0]), float(value[1])) if isinstance(value, (tuple, list, np.ndarray)) else (float(value), float(value))
    return (lo, hi) if lo <= hi else (hi, lo)


def _missing_jump(row: Mapping[str, Any]) -> float | None:
    """Missing-rate jump, taken precomputed or as ``current - reference``."""
    jump = _num(row, _MISS_JUMP_KEYS)
    if jump is not None:
        return jump
    ref_m, cur_m = _num(row, _MISS_REF_KEYS), _num(row, _MISS_CUR_KEYS)
    return (cur_m - ref_m) if (ref_m is not None and cur_m is not None) else None


def _n_evaluable(rows: Iterable[Mapping[str, Any]], keysets: Sequence[Iterable[str]]) -> int:
    """Count the finite metric values a report actually offers to the rules."""
    total = 0
    for row in rows:
        for keys in keysets:
            total += int((_missing_jump(row) if keys is _MISS_JUMP_KEYS else _num(row, keys)) is not None)
    return total


def _coverage_alert(subject: str, detail: str) -> Alert:
    """A report was supplied but yielded no evaluable metric -- say so, loudly.

    Without this, a drift report whose ``psi`` column is entirely NaN (an empty reference
    bin, a failed upstream join, a renamed column) produces zero alerts, and
    :func:`render_alerts` then prints "every monitored metric is inside its threshold;
    deployment gate OPEN". A monitoring layer that reports green because it measured
    nothing is worse than no monitoring layer, so unmeasured is its own alert rather than
    an absence of one.
    """
    return Alert(
        name=f"monitoring_coverage:{subject}",
        severity="warn",
        value=0.0,
        threshold=1.0,
        direction="below",
        message=(
            f"The {subject} was supplied but yielded 0 evaluable metric values, so none of its "
            f"rules could run: this run is unmeasured, not clean. {detail}"
        ),
        recommended_action=(
            f"treat the {subject} as failed rather than passing: check that the producing job wrote "
            f"finite numbers (an all-NaN column usually means an empty reference window or a failed "
            f"join) and that its column names still match the aliases this module resolves; re-run "
            f"monitoring before reading the deployment gate as OPEN."
        ),
    )


def _fmt_num(value: float | None) -> str:
    """Format a metric for a fixed-width ASCII column."""
    if value is None or not np.isfinite(value):
        return "n/a"
    mag = abs(value)
    if mag != 0.0 and (mag < 1e-3 or mag >= 1e6):
        return f"{value:.2e}"
    return f"{value:.4f}"


# ======================================================================================
# Rule blocks
# ======================================================================================
def _drift_alerts(frame: pd.DataFrame, thr: Mapping[str, Any]) -> list[Alert]:
    """Per-feature PSI, KS and missing-rate rules.

    A feature emits at most one PSI alert. Its KS rule is suppressed when PSI already
    fired at ``critical``, because the two statistics are measuring the same shift and a
    duplicate line adds noise without adding a decision. The missing-rate rule is
    independent: a column going null is a pipeline fault, not a distribution shift.
    """
    alerts: list[Alert] = []
    psi_warn = float(thr["psi_warn"])
    psi_alert = float(thr["psi_alert"])
    ks_p_alert = float(thr["ks_p_alert"])
    miss_jump = float(thr["missing_rate_jump"])

    records = frame.to_dict(orient="records")
    feat_key = _pick(records[0], _FEATURE_KEYS) if records else None
    # Alert names are promised to be stable identifiers a CI job can diff, so two rows for
    # the same feature must not both be called "psi:a".
    seen: dict[str, int] = {}
    if feat_key is not None:
        for row in records:
            seen[str(row.get(feat_key))] = seen.get(str(row.get(feat_key)), 0) + 1

    for pos, row in enumerate(records):
        # A NaN label must not become the literal alert name "psi:nan"; fall back to the
        # row position, which is at least honest about being a position.
        label = row.get(feat_key) if feat_key else None
        blank = label is None or (isinstance(label, float) and not np.isfinite(label)) or str(label).strip() == ""
        feature = f"feature_{pos}" if blank else str(label)
        if not blank and seen.get(feature, 0) > 1:
            feature = f"{feature}#{pos}"
        psi_critical_fired = False

        psi = _num(row, _PSI_KEYS)
        # ``psi > 0`` guards the same zero-threshold trap as the missing-rate rule: a
        # perfectly stable feature has PSI 0.0 and must stay silent even at psi_warn=0.
        if psi is not None and psi > 0.0 and psi >= psi_warn:
            critical = psi >= psi_alert
            psi_critical_fired = critical
            edge = psi_alert if critical else psi_warn
            over = psi - edge
            alerts.append(
                Alert(
                    name=f"psi:{feature}",
                    severity="critical" if critical else "warn",
                    value=psi,
                    threshold=edge,
                    message=(
                        f"PSI for '{feature}' is {psi:.3f}, {over:.3f} above the "
                        f"{'alert' if critical else 'warn'} threshold of {edge:.2f} "
                        f"({psi / max(edge, _EPS):.1f}x)."
                    ),
                    recommended_action=(
                        (
                            f"retrain the propensity model and refresh the outcome models on data that "
                            f"includes the current period; a PSI of {psi:.2f} on '{feature}' means the "
                            f"assignment policy or the population has changed, so the cross-fitted "
                            f"nuisances no longer describe this traffic. Until the refit lands, restrict "
                            f"scoring to rows inside the reference support for '{feature}'."
                        )
                        if critical
                        else (
                            f"add '{feature}' to the weekly drift watchlist and diff the current histogram "
                            f"against the reference bins; if PSI crosses {psi_alert:.2f} at the next run, "
                            f"trigger the scheduled retrain rather than waiting for the quarterly refresh."
                        )
                    ),
                )
            )

        ks_p = _num(row, _KS_P_KEYS)
        if ks_p is not None and ks_p < ks_p_alert and not psi_critical_fired:
            alerts.append(
                Alert(
                    name=f"ks:{feature}",
                    severity="warn",
                    value=ks_p,
                    threshold=ks_p_alert,
                    direction="below",
                    message=(
                        f"Kolmogorov-Smirnov p-value for '{feature}' is {ks_p:.2e}, "
                        f"{ks_p_alert / max(ks_p, _EPS):.0f}x below the alert level of {ks_p_alert:g}: "
                        f"the current distribution is not the reference distribution."
                    ),
                    recommended_action=(
                        f"re-fit the numeric imputer and scaler for '{feature}' on the current window and "
                        f"check the upstream ETL job for a unit, currency or aggregation-window change; "
                        f"a KS break with a low PSI is usually a shape change inside the same range."
                    ),
                )
            )

        jump = _missing_jump(row)
        # ``jump > 0`` as well as ``>= miss_jump``: with ``missing_rate_jump=0`` ("tell me
        # about any increase") an unchanged column has jump == 0.0 and would otherwise be
        # reported as a critical missing-rate collapse.
        if jump is not None and jump > 0.0 and jump >= miss_jump:
            critical = jump >= 2.0 * miss_jump
            alerts.append(
                Alert(
                    name=f"missing_rate:{feature}",
                    severity="critical" if critical else "warn",
                    value=jump,
                    threshold=miss_jump,
                    message=(
                        f"Missing rate for '{feature}' rose by {jump * 100:.1f} percentage points, "
                        f"{(jump - miss_jump) * 100:.1f} pp past the {miss_jump * 100:.0f} pp limit."
                    ),
                    recommended_action=(
                        f"inspect the upstream job that populates '{feature}' and backfill the affected "
                        f"periods, then re-run the gold build before the next scoring window; "
                        + (
                            "do not score on the imputed column in the meantime -- at this missing rate "
                            "the median imputation dominates the signal."
                            if critical
                            else "confirm the missing-indicator column is still being emitted so the "
                            "design matrix keeps the MNAR signal."
                        )
                    ),
                )
            )
    return alerts


def _decay_alerts(frame: pd.DataFrame, thr: Mapping[str, Any]) -> list[Alert]:
    """AUC-drop and calibration-slope rules over a per-period decay report.

    The reference is the first row (earliest period, or a row whose period label is
    ``"reference"``/``"baseline"``); the operative value is the last row, because the
    question a deployment gate answers is "is the model good enough *now*".
    """
    alerts: list[Alert] = []
    auc_drop_alert = float(thr["auc_drop_alert"])
    band_lo, band_hi = _band(thr["calibration_slope_band"])

    work = frame.copy()
    period_key = _pick(work.columns, _PERIOD_KEYS)
    if period_key is not None:
        try:
            work = work.sort_values(period_key, kind="mergesort")
        except TypeError:
            _LOG.debug("decay report column %r is not sortable; using row order", period_key)
    records = work.to_dict(orient="records")
    if not records:
        return alerts

    def _label(row: Mapping[str, Any], role: str = "window") -> str:
        # Without a period column the only ordering available is the row order the caller
        # supplied, so the label says "first/last row", never an invented period name.
        return f"period {row[period_key]}" if period_key is not None else f"the {role} row"

    aucs = [(row, _num(row, _AUC_KEYS)) for row in records]
    aucs = [(row, val) for row, val in aucs if val is not None]
    if period_key is None and len(aucs) >= 2:
        # Refuse to turn row order into a trend. A frame grouped by segment rather than by
        # time sorts "high"/"low"/"mid" alphabetically, and first-minus-last would then
        # report a 0.10 "AUC decay" that is an artefact of the alphabet. Skipping the rule
        # would be silent, so the skip is itself an alert.
        alerts.append(
            Alert(
                name="decay_report_unordered",
                severity="warn",
                value=float(len(aucs)),
                threshold=0.0,
                message=(
                    f"The decay report has {len(aucs)} rows with an AUC but no recognised ordered "
                    f"period column, so the AUC-drop rule was skipped rather than computed from row "
                    f"order (that would report a trend the data does not contain)."
                ),
                recommended_action=(
                    "rename the grouping column to one of "
                    + ", ".join(f"'{k}'" for k in _PERIOD_KEYS)
                    + " if the report really is per period, and re-run monitoring; if it is grouped by "
                    "segment rather than time, evaluate it per segment instead -- a first-to-last "
                    "difference across segments is not decay."
                ),
            )
        )
    elif len(aucs) >= 2:
        ref_row, ref_auc = aucs[0]
        cur_row, cur_auc = aucs[-1]
        worst_row, worst_auc = min(aucs[1:], key=lambda rv: rv[1])
        drop = ref_auc - cur_auc
        if drop >= auc_drop_alert:
            critical = drop >= 2.0 * auc_drop_alert
            alerts.append(
                Alert(
                    name="auc_drop",
                    severity="critical" if critical else "warn",
                    value=drop,
                    threshold=auc_drop_alert,
                    message=(
                        f"Ranking AUC fell from {ref_auc:.3f} ({_label(ref_row, 'first')}) to {cur_auc:.3f} "
                        f"({_label(cur_row, 'last')}), a drop of {drop:.3f} against a limit of {auc_drop_alert:.2f} "
                        f"(worst observed {worst_auc:.3f} at {_label(worst_row, 'worst')})."
                    ),
                    recommended_action=(
                        "retrain the churn/hazard model on the most recent periods and re-validate it on "
                        "the RCT holdout before the next allocation run; "
                        + (
                            "freeze the current budget allocation until the retrained model clears the "
                            "holdout -- at this level of decay the ranking no longer separates churners."
                            if critical
                            else "keep the current allocation but shorten the retrain cadence to weekly "
                            "until AUC recovers."
                        )
                    ),
                )
            )

    slopes = [(row, _num(row, _SLOPE_KEYS)) for row in records]
    slopes = [(row, val) for row, val in slopes if val is not None]
    if slopes:
        cur_row, slope = slopes[-1]
        if slope < band_lo or slope > band_hi:
            width = max(band_hi - band_lo, _EPS)
            below = slope < band_lo
            edge = band_lo if below else band_hi
            critical = slope < band_lo - width or slope > band_hi + width
            gap = (edge - slope) if below else (slope - edge)
            alerts.append(
                Alert(
                    name="calibration_slope",
                    severity="critical" if critical else "warn",
                    value=slope,
                    threshold=edge,
                    direction="below" if below else "above",
                    message=(
                        f"Calibration slope at {_label(cur_row, 'last')} is {slope:.3f}, {gap:.3f} outside the "
                        f"[{band_lo:g}, {band_hi:g}] band -- predicted risks are "
                        f"{'over-dispersed (too extreme)' if below else 'under-dispersed (too flat)'}."
                    ),
                    recommended_action=(
                        "re-fit the calibration layer (isotonic regression on the most recent validation "
                        "fold) and re-check the slope before the scores feed the budget optimiser; "
                        "uncalibrated risk multiplied by a CATE produces a mis-priced net value, so "
                        + (
                            "hold the allocation run until the slope is back inside the band."
                            if critical
                            else "re-run the efficient-frontier chart after recalibration to confirm the "
                            "budget split did not move."
                        )
                    ),
                )
            )
    return alerts


def _cate_alerts(stability: Mapping[str, Any], thr: Mapping[str, Any]) -> list[Alert]:
    """Causal-specific rules over the dict returned by ``drift.cate_stability``."""
    alerts: list[Alert] = []
    corr_thr = float(thr["cate_rank_corr_alert"])
    flip_thr = float(thr["sign_flip_alert"])

    corr = _num(stability, _RANK_CORR_KEYS)
    if corr is not None and corr < corr_thr:
        critical = corr < 0.5 * corr_thr
        alerts.append(
            Alert(
                name="cate_rank_corr",
                severity="critical" if critical else "warn",
                value=corr,
                threshold=corr_thr,
                direction="below",
                message=(
                    f"Rank correlation between the reference and current CATE scores is {corr:.3f}, "
                    f"{corr_thr - corr:.3f} below the floor of {corr_thr:.2f}: the model is ordering "
                    f"customers differently, so last period's targeting list is stale."
                ),
                recommended_action=(
                    "hold the current policy and re-run the RCT holdout evaluation (Qini and GATES) on "
                    "the new scores before reallocating budget; refit the CATE learner with cross-fitting "
                    "on the latest decision-point sample and compare policy value against the incumbent "
                    + ("policy -- do not ship the new ranking on its rank correlation alone." if critical else "policy.")
                ),
            )
        )

    flip = _num(stability, _SIGN_FLIP_KEYS)
    if flip is not None and flip >= flip_thr:
        critical = flip >= 2.0 * flip_thr
        alerts.append(
            Alert(
                name="cate_sign_flip",
                severity="critical" if critical else "warn",
                value=flip,
                threshold=flip_thr,
                message=(
                    f"{flip * 100:.1f}% of customers changed the sign of their estimated treatment "
                    f"effect, {(flip - flip_thr) * 100:.1f} pp past the {flip_thr * 100:.0f}% limit."
                ),
                recommended_action=(
                    "hold the current policy and re-run the RCT holdout before any new send; a sign flip "
                    "means customers previously scored as persuadable now score as sleeping dogs, and "
                    "treating a sleeping dog destroys value, so "
                    + (
                        "suppress every offer to the flipped cohort and re-estimate the CATEs on the "
                        "randomised block only."
                        if critical
                        else "exclude the flipped cohort from the next send while the refit is validated."
                    )
                ),
            )
        )
    return alerts


def _policy_alerts(metrics: Mapping[str, Any], thr: Mapping[str, Any]) -> list[Alert]:
    """Rules over the realised policy-evaluation metrics (see ``decision.policy_eval``)."""
    alerts: list[Alert] = []
    ci_min = float(thr["policy_value_ci_low_min"])
    uplift_min = float(thr["policy_uplift_vs_baseline_min"])
    roi_warn = float(thr["roi_warn"])
    util_info = float(thr["budget_utilisation_info"])

    ci_low = _num(metrics, _POLICY_CI_KEYS)
    if ci_low is not None and ci_low <= ci_min:
        alerts.append(
            Alert(
                name="policy_value_ci",
                severity="critical",
                value=ci_low,
                threshold=ci_min,
                direction="below",
                message=(
                    f"The lower bound of the policy-value confidence interval is {ci_low:,.1f}, at or "
                    f"below {ci_min:,.1f}: the estimated incremental value is not distinguishable from "
                    f"doing nothing."
                ),
                recommended_action=(
                    "do not roll out; keep the incumbent policy live and extend the RCT holdout until the "
                    "doubly-robust interval excludes zero, then re-run the policy comparison."
                ),
            )
        )

    pol = _num(metrics, _POLICY_VALUE_KEYS)
    base = _num(metrics, _POLICY_BASE_KEYS)
    if pol is not None and base is not None and (pol - base) <= uplift_min:
        alerts.append(
            Alert(
                name="policy_vs_baseline",
                severity="critical",
                value=pol - base,
                threshold=uplift_min,
                direction="below",
                message=(
                    f"The candidate policy is worth {pol:,.1f} against the baseline's {base:,.1f}, an "
                    f"uplift of {pol - base:,.1f} -- it does not beat the incumbent."
                ),
                recommended_action=(
                    "revert to the incumbent baseline policy for the next cycle and re-estimate the CATEs "
                    "on the current decision-point sample; re-run the efficient frontier to check whether "
                    "the budget, not the model, is the binding constraint."
                ),
            )
        )

    roi = _num(metrics, _ROI_KEYS)
    if roi is not None and roi < roi_warn:
        alerts.append(
            Alert(
                name="policy_roi",
                severity="warn",
                value=roi,
                threshold=roi_warn,
                direction="below",
                message=f"Realised ROI is {roi:.3f}, below the floor of {roi_warn:.2f}: the campaign spend is not returning.",
                recommended_action=(
                    "cut the two most expensive arms out of the eligible set and re-solve the allocation "
                    "at the same budget; if ROI stays negative, drop the budget to the frontier knee."
                ),
            )
        )

    util = _num(metrics, _UTIL_KEYS)
    if util is not None and util < util_info:
        alerts.append(
            Alert(
                name="budget_utilisation",
                severity="info",
                value=util,
                threshold=util_info,
                direction="below",
                message=(
                    f"Only {util * 100:.1f}% of the budget was allocated, {(util_info - util) * 100:.1f} pp "
                    f"under the {util_info * 100:.0f}% guide: too few customers clear the net-value bar."
                ),
                recommended_action=(
                    "widen eligibility or lower the net-value floor in the optimiser, or move the unspent "
                    "budget to the next cycle; re-run efficient_frontier to see where the marginal value "
                    "of the next currency unit actually goes to zero."
                ),
            )
        )
    return alerts


# ======================================================================================
# Public API
# ======================================================================================
def evaluate_alerts(
    drift_report: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    decay_report: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    thresholds: Mapping[str, Any] | None = None,
    *,
    cate_stability: Mapping[str, Any] | None = None,
    policy_metrics: Mapping[str, Any] | None = None,
) -> list[Alert]:
    """Turn monitoring reports into an ordered list of actionable alerts.

    Parameters
    ----------
    drift_report : pandas.DataFrame or mapping or sequence of mappings, optional
        Per-feature drift measurements, typically the output of
        ``prism.monitoring.drift.feature_drift_report``. Recognised columns (aliases in
        brackets): ``feature`` [``column``, ``name``], ``psi``, ``ks_p`` [``p_value``,
        ``p``], and either ``missing_rate_jump`` or the pair
        ``missing_rate_ref``/``missing_rate_cur``. Unknown columns are ignored; missing
        ones simply disable their rule. ``None`` is allowed.
    decay_report : pandas.DataFrame or mapping or sequence of mappings, optional
        Per-period performance, typically ``prism.monitoring.drift.performance_decay``.
        Recognised columns: ``period`` [``as_of_date``, ``month``], ``auc`` [``roc_auc``,
        ``c_index``], ``calibration_slope`` [``cal_slope``]. ``None`` is allowed.
    thresholds : mapping, optional
        Overrides for :data:`DEFAULT_THRESHOLDS` and :data:`DEFAULT_POLICY_THRESHOLDS`.
        Key aliases in :data:`THRESHOLD_ALIASES` are accepted.
    cate_stability : mapping, optional
        The dict returned by ``prism.monitoring.drift.cate_stability``; read for
        ``rank_corr`` and ``sign_flip_rate``.
    policy_metrics : mapping, optional
        Realised policy economics; read for ``policy_value``, ``policy_value_ci_low``,
        ``baseline_value``, ``roi`` and ``budget_utilisation``.

    Returns
    -------
    list of Alert
        Sorted by severity (critical first) then by :attr:`Alert.exceedance` descending,
        then by name; the order is deterministic for a given input.

    Notes
    -----
    Rule summary (``t`` = the effective threshold):

    ==========================  ======================  ==========================
    metric                      warn                    critical
    ==========================  ======================  ==========================
    ``psi``                     ``>= psi_warn``         ``>= psi_alert``
    ``ks_p``                    ``< ks_p_alert``        --
    missing-rate jump           ``>= t``                ``>= 2t``
    AUC drop (first vs last)    ``>= t``                ``>= 2t``
    calibration slope           outside band            outside band +/- its width
    CATE rank correlation       ``< t``                 ``< t / 2``
    CATE sign-flip rate         ``>= t``                ``>= 2t``
    policy value CI lower bound --                      ``<= 0``
    policy vs baseline uplift   --                      ``<= 0``
    ROI                         ``< 0``                 --
    budget utilisation          -- (``info`` under t)   --
    ==========================  ======================  ==========================

    A feature whose PSI already fired at ``critical`` has its KS rule suppressed: both
    statistics describe the same shift, and one line per cause keeps the CI comment
    readable.

    Two rules fire on the *absence* of a measurement rather than on its value, because an
    alerting layer that reports green when it measured nothing is worse than none:

    * ``monitoring_coverage:<report>`` (``warn``) -- a report was supplied but yielded no
      finite metric at all (an all-NaN column, a renamed column, a failed upstream join).
    * ``decay_report_unordered`` (``warn``) -- a decay report has two or more AUC rows but
      no recognised ordered period column, so the AUC-drop rule was skipped instead of
      computed from row order, which would report a trend the data does not contain.

    Both are ``warn``, not ``critical``: a renamed column should make the gate noisy, not
    slam it shut.

    Examples
    --------
    >>> drift = pd.DataFrame({"feature": ["nps"], "psi": [0.40], "ks_p": [1e-9]})
    >>> [a.severity for a in evaluate_alerts(drift)]
    ['critical']
    >>> evaluate_alerts(pd.DataFrame({"feature": ["nps"], "psi": [0.01]}))
    []
    """
    thr = resolve_thresholds(thresholds)
    alerts: list[Alert] = []

    drift_frame = _as_frame(drift_report, what="drift_report")
    if drift_frame is not None:
        alerts.extend(_drift_alerts(drift_frame, thr))
        rows = drift_frame.to_dict(orient="records")
        if not _n_evaluable(rows, (_PSI_KEYS, _KS_P_KEYS, _MISS_JUMP_KEYS)):
            alerts.append(
                _coverage_alert(
                    "drift_report",
                    f"{len(rows)} row(s) carrying columns {sorted(map(str, drift_frame.columns))[:8]} but no "
                    f"finite psi, ks p-value or missing-rate jump among them.",
                )
            )

    decay_frame = _as_frame(decay_report, what="decay_report")
    if decay_frame is not None:
        alerts.extend(_decay_alerts(decay_frame, thr))
        rows = decay_frame.to_dict(orient="records")
        if not _n_evaluable(rows, (_AUC_KEYS, _SLOPE_KEYS)):
            alerts.append(
                _coverage_alert(
                    "decay_report",
                    f"{len(rows)} row(s) carrying columns {sorted(map(str, decay_frame.columns))[:8]} but no "
                    f"finite auc or calibration slope among them.",
                )
            )

    if cate_stability:
        alerts.extend(_cate_alerts(cate_stability, thr))
        if not _n_evaluable([cate_stability], (_RANK_CORR_KEYS, _SIGN_FLIP_KEYS)):
            alerts.append(
                _coverage_alert(
                    "cate_stability",
                    f"keys {sorted(map(str, cate_stability))[:8]} contain no finite rank correlation or "
                    f"sign-flip rate.",
                )
            )

    if policy_metrics:
        alerts.extend(_policy_alerts(policy_metrics, thr))
        if not _n_evaluable([policy_metrics], (_POLICY_CI_KEYS, _POLICY_VALUE_KEYS, _ROI_KEYS, _UTIL_KEYS)):
            alerts.append(
                _coverage_alert(
                    "policy_metrics",
                    f"keys {sorted(map(str, policy_metrics))[:8]} contain no finite policy value, CI bound, "
                    f"ROI or budget utilisation.",
                )
            )

    if drift_frame is None and decay_frame is None and not cate_stability and not policy_metrics:
        _LOG.warning("evaluate_alerts called with no reports; returning an empty alert list")

    alerts.sort(key=lambda a: a.sort_key)
    _LOG.debug("evaluated %d alert(s)", len(alerts))
    return alerts


def alert_summary(alerts: Sequence[Alert]) -> dict[str, Any]:
    """Summarise an alert list for ``metrics.json`` and for the report header.

    Parameters
    ----------
    alerts : sequence of Alert
        Alerts, in any order.

    Returns
    -------
    dict
        ``{"total": int, "info": int, "warn": int, "critical": int,
        "worst_severity": str | None, "should_block": bool, "names": list[str]}``.
        ``worst_severity`` is ``None`` for an empty list -- "no alerts" is not a severity.

    Examples
    --------
    >>> alert_summary([])["should_block"]
    False
    """
    counts = dict.fromkeys(SEVERITIES, 0)
    for a in alerts:
        counts[a.severity] += 1
    ordered = sorted(alerts, key=lambda a: a.sort_key)
    worst = ordered[0].severity if ordered else None
    return {
        "total": len(alerts),
        "info": counts["info"],
        "warn": counts["warn"],
        "critical": counts["critical"],
        "worst_severity": worst,
        "should_block": should_block_deployment(alerts),
        "names": [a.name for a in ordered],
    }


def should_block_deployment(alerts: Sequence[Alert]) -> bool:
    """Deployment gate: block if and only if at least one ``critical`` alert is present.

    Policy
    ------
    ``critical`` is reserved for conditions where continuing to act on the current model
    is expected to *destroy* value rather than merely earn less of it: a feature whose
    distribution has moved past ``psi_alert`` (the nuisance models are fit on a different
    population), a column that has gone substantially null, a halving of the AUC headroom,
    a CATE sign-flip epidemic (treating sleeping dogs), or a policy whose value interval
    includes zero. Each of those makes the *sign* of the decision unreliable.

    ``warn`` means the estimate is degrading but the ranking is still usable, so the
    release proceeds and the retrain cadence tightens. ``info`` is a note for the memo and
    never gates anything. A warn-only run therefore returns ``False`` by design: blocking
    on warnings trains the team to bypass the gate, which is worse than not having one.

    Parameters
    ----------
    alerts : sequence of Alert
        Alerts from :func:`evaluate_alerts`.

    Returns
    -------
    bool
        ``True`` if any alert has severity ``"critical"``.

    Examples
    --------
    >>> should_block_deployment([Alert("x", "warn", 1.0, 0.5, "m")])
    False
    """
    return any(a.severity == "critical" for a in alerts)


def render_alerts(alerts: Sequence[Alert]) -> str:
    """Render alerts as a plain-ASCII table for a log file or a CI comment.

    Parameters
    ----------
    alerts : sequence of Alert
        Alerts, in any order; they are sorted before rendering.

    Returns
    -------
    str
        A multi-line, newline-free-at-the-end string. Never empty: an empty input renders
        an explicit "no alerts" line so a silent monitoring failure cannot masquerade as a
        clean run.

    Examples
    --------
    >>> "no alerts" in render_alerts([])
    True
    """
    if not alerts:
        return (
            "PRISM ALERTS | no alerts | every monitored metric is inside its threshold; "
            "deployment gate OPEN."
        )

    ordered = sorted(alerts, key=lambda a: a.sort_key)
    summary = alert_summary(ordered)
    header = (
        f"PRISM ALERTS | {summary['total']} total | "
        f"critical {summary['critical']}  warn {summary['warn']}  info {summary['info']} | "
        f"deployment gate {'BLOCKED' if summary['should_block'] else 'OPEN'}"
    )

    sev_w = max(8, max(len(a.severity) for a in ordered))
    # No cap: truncating 'psi:<long_feature_name>' in the one place an operator reads the
    # alert loses the only thing that identifies which feature broke.
    name_w = max(5, max(len(a.name) for a in ordered))
    val_w = max(6, max(len(_fmt_num(a.value)) for a in ordered))
    thr_w = max(9, max(len(_fmt_num(a.threshold)) for a in ordered))

    lines = [header, "-" * max(len(header), 78)]
    lines.append(f"{'SEVERITY':<{sev_w}}  {'ALERT':<{name_w}}  {'VALUE':>{val_w}}  {'THRESHOLD':>{thr_w}}  MESSAGE")
    lines.append(f"{'-' * sev_w}  {'-' * name_w}  {'-' * val_w}  {'-' * thr_w}  {'-' * 40}")
    indent = " " * (sev_w + name_w + val_w + thr_w + 8)
    for a in ordered:
        lines.append(
            f"{a.severity.upper():<{sev_w}}  {a.name:<{name_w}}  "
            f"{_fmt_num(a.value):>{val_w}}  {_fmt_num(a.threshold):>{thr_w}}  {a.message}"
        )
        if a.recommended_action:
            lines.append(f"{indent}-> action: {a.recommended_action}")
    return "\n".join(lines)


def to_markdown(alerts: Sequence[Alert]) -> str:
    """Render alerts as a Markdown section for the generated monitoring report.

    Parameters
    ----------
    alerts : sequence of Alert
        Alerts, in any order; they are sorted before rendering.

    Returns
    -------
    str
        A Markdown fragment: a status line, then a table of alerts and a list of the
        recommended actions. Pipe characters inside messages are escaped so the table
        cannot break.

    Examples
    --------
    >>> to_markdown([]).startswith("## Monitoring alerts")
    True
    """
    head = "## Monitoring alerts"
    if not alerts:
        return (
            f"{head}\n\n**No alerts.** Every monitored metric is inside its threshold; "
            f"the deployment gate is **OPEN**.\n"
        )

    ordered = sorted(alerts, key=lambda a: a.sort_key)
    s = alert_summary(ordered)
    gate = "**BLOCKED**" if s["should_block"] else "**OPEN**"
    out = [
        head,
        "",
        f"{s['total']} alert(s): {s['critical']} critical, {s['warn']} warn, {s['info']} info. "
        f"Deployment gate: {gate}.",
        "",
        "| Severity | Alert | Value | Threshold | What tripped |",
        "| --- | --- | ---: | ---: | --- |",
    ]

    def esc(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")

    for a in ordered:
        out.append(
            f"| `{a.severity}` | `{a.name}` | {_fmt_num(a.value)} | {_fmt_num(a.threshold)} | {esc(a.message)} |"
        )
    out += ["", "### Recommended actions", ""]
    for a in ordered:
        if a.recommended_action:
            out.append(f"- **{a.name}** ({a.severity}) -- {esc(a.recommended_action)}")
    return "\n".join(out) + "\n"


# ======================================================================================
# Smoke test
# ======================================================================================
if __name__ == "__main__":  # pragma: no cover - smoke test
    import time

    _t0 = time.perf_counter()

    # ---- synthetic drift report: three clean features, three broken ones -------------
    drifted_drift = pd.DataFrame(
        {
            "feature": [
                "tenure_months",      # clean
                "engagement_score",   # PSI warn
                "monetary_12m",       # PSI critical (KS suppressed)
                "nps",                # KS only -> warn
                "sessions_30d",       # missing-rate collapse -> critical
                "price_change_pct",   # clean
            ],
            "psi": [0.012, 0.150, 0.420, 0.030, 0.010, 0.004],
            "ks_p": [0.610, 0.180, 1e-9, 5e-4, 0.480, 0.900],
            "missing_rate_ref": [0.01, 0.02, 0.00, 0.03, 0.020, 0.01],
            "missing_rate_cur": [0.01, 0.02, 0.00, 0.03, 0.300, 0.01],
        }
    )
    decay = pd.DataFrame(
        {
            "period": [18, 19, 20, 21],
            "auc": [0.782, 0.774, 0.751, 0.701],   # drop 0.081 -> warn (< 2 * 0.05)
            "calibration_slope": [1.02, 1.08, 1.19, 1.35],  # outside (0.8, 1.25) -> warn
            "n": [9100, 8800, 8500, 8200],
        }
    )
    stability = {"rank_corr": 0.62, "sign_flip_rate": 0.34, "decile_migration": None}
    policy = {
        "policy_value": 124_500.0,
        "baseline_value": 96_200.0,
        "policy_value_ci_low": 4_180.0,
        "roi": 1.84,
        "budget_utilisation": 0.41,   # -> info
    }

    alerts = evaluate_alerts(drifted_drift, decay, cate_stability=stability, policy_metrics=policy)
    summary = alert_summary(alerts)

    # ---- assertions: exact counts per severity ---------------------------------------
    names = {a.name for a in alerts}
    assert summary["critical"] == 3, (summary, names)
    assert summary["warn"] == 5, (summary, names)
    assert summary["info"] == 1, (summary, names)
    assert summary["total"] == 9 == len(alerts)
    assert names == {
        "psi:monetary_12m",
        "missing_rate:sessions_30d",
        "cate_sign_flip",
        "psi:engagement_score",
        "ks:nps",
        "auc_drop",
        "calibration_slope",
        "cate_rank_corr",
        "budget_utilisation",
    }, sorted(names)
    # KS on monetary_12m is suppressed by its critical PSI; clean features stay silent.
    assert "ks:monetary_12m" not in names
    assert not any(a.name.endswith(("tenure_months", "price_change_pct")) for a in alerts)

    # ordering: criticals first, exceedance descending inside a severity
    ranks = [SEVERITY_RANK[a.severity] for a in alerts]
    assert ranks == sorted(ranks), ranks
    for lo, hi in zip(alerts, alerts[1:]):
        if lo.severity == hi.severity:
            assert lo.exceedance >= hi.exceedance - 1e-12, (lo.name, hi.name)

    # every alert says how much, and prescribes something concrete
    for a in alerts:
        assert a.message and any(ch.isdigit() for ch in a.message), a.name
        assert len(a.recommended_action) > 30 and "investigate" not in a.recommended_action.lower()
        d = a.as_dict()
        assert set(d) >= {"name", "severity", "value", "threshold", "message", "recommended_action", "exceedance"}

    # ---- gate behaviour ---------------------------------------------------------------
    assert should_block_deployment(alerts) is True
    assert summary["should_block"] is True and summary["worst_severity"] == "critical"

    warn_only = [a for a in alerts if a.severity != "critical"]
    assert should_block_deployment(warn_only) is False
    assert alert_summary(warn_only)["worst_severity"] == "warn"

    # ---- clean input -> zero alerts ---------------------------------------------------
    clean_drift = pd.DataFrame(
        {
            "feature": ["tenure_months", "engagement_score", "monetary_12m", "nps"],
            "psi": [0.012, 0.031, 0.022, 0.008],
            "ks_p": [0.610, 0.220, 0.410, 0.870],
            "missing_rate_ref": [0.01, 0.02, 0.00, 0.03],
            "missing_rate_cur": [0.01, 0.03, 0.00, 0.03],
        }
    )
    clean_decay = pd.DataFrame(
        {"period": [18, 19, 20, 21], "auc": [0.780, 0.783, 0.776, 0.779], "calibration_slope": [1.0, 0.98, 1.04, 1.01]}
    )
    clean_alerts = evaluate_alerts(
        clean_drift,
        clean_decay,
        cate_stability={"rank_corr": 0.94, "sign_flip_rate": 0.03},
        policy_metrics={"policy_value": 130_000.0, "baseline_value": 96_000.0, "policy_value_ci_low": 9_000.0,
                        "roi": 2.1, "budget_utilisation": 0.93},
    )
    assert clean_alerts == [], [a.name for a in clean_alerts]
    assert should_block_deployment(clean_alerts) is False
    assert alert_summary(clean_alerts)["worst_severity"] is None

    # ---- optional inputs, alternate container types, threshold overrides ---------------
    assert evaluate_alerts(None, None) == []
    assert evaluate_alerts(drift_report=None, decay_report=decay) and all(
        a.name in {"auc_drop", "calibration_slope"} for a in evaluate_alerts(None, decay)
    )
    from_records = evaluate_alerts(drifted_drift.to_dict(orient="records"))
    from_frame = evaluate_alerts(drifted_drift)
    assert [a.as_dict() for a in from_records] == [a.as_dict() for a in from_frame]
    mapping_form = evaluate_alerts({"nps": {"psi": 0.40, "ks_p": 0.5}})
    assert [a.name for a in mapping_form] == ["psi:nps"]
    # overrides (and the MonitoringConfig alias spelling) actually bind
    assert evaluate_alerts(drifted_drift, thresholds={"psi_warn": 0.9, "psi_alert": 0.99,
                                                      "ks_alert_p": 1e-12, "missing_rate_jump": 0.9}) == []
    # determinism
    assert [a.as_dict() for a in evaluate_alerts(drifted_drift, decay, cate_stability=stability,
                                                 policy_metrics=policy)] == [a.as_dict() for a in alerts]

    # ---- unmeasured must not read as clean (regression: silent NaN blindness) ---------
    blind = evaluate_alerts(pd.DataFrame({"feature": ["a", "b"], "psi": [np.nan, np.nan]}))
    assert [a.name for a in blind] == ["monitoring_coverage:drift_report"], [a.name for a in blind]
    assert "unmeasured, not clean" in blind[0].message
    assert evaluate_alerts(pd.DataFrame({"feature": ["a"], "no_metric_here": [1.0]}))
    assert evaluate_alerts(None, pd.DataFrame({"period": [1, 2], "auc": [np.nan, np.nan]}))
    assert [a.name for a in evaluate_alerts(None, None, cate_stability={"decile_migration": None})] == [
        "monitoring_coverage:cate_stability"
    ]
    # a report that IS measured and IS inside its thresholds still renders as silence
    assert evaluate_alerts(pd.DataFrame({"feature": ["a"], "psi": [0.01]})) == []

    # ---- an unordered decay report must not be turned into a trend --------------------
    by_segment = pd.DataFrame({"segment": ["high", "low", "mid"], "auc": [0.80, 0.60, 0.70]})
    seg_alerts = evaluate_alerts(None, by_segment)
    assert not any(a.name == "auc_drop" for a in seg_alerts), [a.message for a in seg_alerts]
    assert [a.name for a in seg_alerts] == ["decay_report_unordered"], [a.name for a in seg_alerts]

    # ---- alert names stay stable, meaningful and unique -------------------------------
    on_index = pd.DataFrame({"psi": [0.42]}, index=pd.Index(["monetary_12m"], name="feature"))
    assert [a.name for a in evaluate_alerts(on_index)] == ["psi:monetary_12m"]
    assert [a.name for a in evaluate_alerts(pd.DataFrame({"feature": [np.nan], "psi": [0.42]}))] == ["psi:feature_0"]
    dup_names = [a.name for a in evaluate_alerts(pd.DataFrame({"feature": ["a", "a"], "psi": [0.42, 0.30]}))]
    assert len(set(dup_names)) == len(dup_names) == 2, dup_names
    assert [a.name for a in evaluate_alerts({"nps": {"feature": "nps", "psi": 0.42}})] == ["psi:nps"]

    # ---- a zero threshold must not fire on a zero change ------------------------------
    no_change = pd.DataFrame({"feature": ["f"], "psi": [0.0], "missing_rate_ref": [0.02], "missing_rate_cur": [0.02]})
    assert evaluate_alerts(no_change, thresholds={"psi_warn": 0.0, "missing_rate_jump": 0.0}) == []

    # ---- ordering contract over randomised alert sets ---------------------------------
    _rng = np.random.default_rng(0)
    for _ in range(200):
        pool = [
            Alert(f"n{i}", ("info", "warn", "critical")[int(_rng.integers(0, 3))],
                  float(_rng.normal()), float(_rng.normal()), "m 1",
                  direction=("above", "below")[int(_rng.integers(0, 2))])
            for i in range(int(_rng.integers(1, 7)))
        ]
        ordered_pool = sorted(pool, key=lambda a: a.sort_key)
        assert all(0.0 <= a.exceedance <= 1.0 for a in ordered_pool)
        assert all(x.sort_key <= y.sort_key for x, y in zip(ordered_pool, ordered_pool[1:]))

    # ---- rendering --------------------------------------------------------------------
    empty_render = render_alerts([])
    assert empty_render.strip() and "no alerts" in empty_render
    table = render_alerts(alerts)
    assert "CRITICAL" in table and "deployment gate BLOCKED" in table
    assert table.isascii() and to_markdown(alerts).isascii()
    assert len(table.splitlines()) >= len(alerts) + 4
    md = to_markdown(alerts)
    assert md.startswith("## Monitoring alerts") and "| Severity |" in md
    assert to_markdown([]).count("No alerts") == 1
    # the name column identifies the feature; truncating it defeats the whole table
    long_name = "psi:a_feature_name_far_wider_than_any_sensible_column"
    assert long_name in render_alerts([Alert(long_name, "critical", 1.0, 0.25, "m 1", "x" * 40)])

    # ---- the docstring examples are part of the contract -------------------------------
    import doctest as _doctest

    _dt = _doctest.testmod(verbose=False)
    assert _dt.failed == 0 and _dt.attempted >= 9, _dt

    print(table)
    print()
    print(render_alerts(clean_alerts))
    print()
    print(pd.DataFrame([a.as_dict() for a in alerts])[["name", "severity", "value", "threshold", "exceedance"]]
          .to_string(index=False))
    print()
    print(f"alerts.py OK  ->  {summary['total']} alerts "
          f"(critical {summary['critical']}, warn {summary['warn']}, info {summary['info']}), "
          f"block={summary['should_block']}, {time.perf_counter() - _t0:.2f}s")
