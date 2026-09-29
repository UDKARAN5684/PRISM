"""Figure generation for the PRISM reports.

One module, one visual language. Every figure is written to ``docs/figures`` at 150 dpi with a
consistent palette and honest axis labels, and every function degrades to a "no data" placeholder
rather than raising, so a partial pipeline still produces a readable report.

Design rules applied throughout:

* One idea per figure, stated in the title, with the *conclusion* in the subtitle where there is one.
* Uncertainty is drawn wherever it exists. A point estimate without an interval invites overclaiming.
* Colour carries meaning (treated vs control, gain vs loss), never decoration.
* Colour-blind-safe palette; nothing relies on red-versus-green alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: this runs in CI and in containers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from prism.utils.logging import get_logger

__all__ = [
    "PALETTE",
    "apply_style",
    "save",
    "plot_qini",
    "plot_uplift_deciles",
    "plot_survival_curves",
    "plot_efficient_frontier",
    "plot_love_plot",
    "plot_calibration",
    "plot_segment_heatmap",
    "plot_sensitivity_contour",
    "plot_drift",
    "plot_policy_comparison",
    "plot_learner_leaderboard",
    "plot_gates",
    "generate_all",
]

log = get_logger("figures")

#: Colour-blind-safe (Okabe-Ito derived). Roles, not decoration.
PALETTE: dict[str, str] = {
    "primary": "#0072B2",     # the model / the causal policy
    "secondary": "#E69F00",   # the comparison / baseline
    "accent": "#009E73",      # gain
    "warn": "#D55E00",        # loss / alert
    "muted": "#999999",       # random / reference line
    "dark": "#333333",
    "light": "#E8E8E8",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
}

ARM_COLOURS = [PALETTE["muted"], PALETTE["sky"], PALETTE["primary"], PALETTE["purple"]]
FIGDIR = Path("docs/figures")


def apply_style() -> None:
    """Apply the shared matplotlib style. Idempotent."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["dark"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["light"],
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.color": PALETTE["dark"],
            "ytick.color": PALETTE["dark"],
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def save(fig: Figure, name: str, figdir: str | Path = FIGDIR) -> Path:
    """Write ``fig`` to ``figdir/name`` and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to write.
    name : str
        File name, e.g. ``"qini.png"``.
    figdir : str or Path
        Destination directory; created if absent.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    out = Path(figdir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    log.debug("wrote %s", out)
    return out


def _placeholder(message: str, name: str, figdir: str | Path = FIGDIR) -> Path:
    """Write a figure that says why there is no figure. Better than a missing file."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color=PALETTE["muted"], wrap=True)
    ax.set_axis_off()
    return save(fig, name, figdir)


def _subtitle(ax: plt.Axes, text: str) -> None:
    """Put a conclusion line under the title, where a reader actually looks."""
    ax.set_title(ax.get_title(), pad=22)
    ax.text(
        0.0, 1.02, text, transform=ax.transAxes, fontsize=9, color=PALETTE["muted"], va="bottom", ha="left"
    )


def _fmt_money(x: float, _pos: Any = None) -> str:
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:,.1f}B"
    if ax >= 1e6:
        return f"{x / 1e6:,.1f}M"
    if ax >= 1e3:
        return f"{x / 1e3:,.0f}k"
    return f"{x:,.0f}"


# =====================================================================================
# Causal model figures
# =====================================================================================


def plot_qini(qini: pd.DataFrame, name: str = "qini.png", figdir: str | Path = FIGDIR,
              title: str = "Qini curve: incremental value from targeting the top x%") -> Path:
    """Qini curve against the random-targeting diagonal.

    Parameters
    ----------
    qini : pandas.DataFrame
        Output of :func:`prism.causal.evaluate.qini_curve`; needs ``frac_targeted`` and
        ``incremental``, optionally ``random_incremental``.
    """
    if qini is None or len(qini) == 0 or "frac_targeted" not in qini:
        return _placeholder("No Qini curve available for this run.", name, figdir)

    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = qini["frac_targeted"].to_numpy(dtype=float)
    y = qini["incremental"].to_numpy(dtype=float)
    ax.plot(x, y, color=PALETTE["primary"], lw=2.2, label="CATE-ranked targeting", zorder=3)

    if "random_incremental" in qini:
        r = qini["random_incremental"].to_numpy(dtype=float)
    else:
        r = np.linspace(0, y[-1] if len(y) else 0.0, len(x))
    ax.plot(x, r, color=PALETTE["muted"], lw=1.5, ls="--", label="random targeting", zorder=2)
    ax.fill_between(x, r, y, where=y >= r, color=PALETTE["accent"], alpha=0.16, zorder=1, label="value captured")
    ax.fill_between(x, r, y, where=y < r, color=PALETTE["warn"], alpha=0.16, zorder=1)

    peak = int(np.argmax(y - r)) if len(y) else 0
    if len(y):
        ax.axvline(x[peak], color=PALETTE["warn"], lw=1.0, ls=":", zorder=2)
        ax.annotate(
            f"best gap at {x[peak]:.0%}",
            xy=(x[peak], y[peak]), xytext=(8, -14), textcoords="offset points",
            fontsize=9, color=PALETTE["warn"],
        )

    ax.set_xlabel("fraction of the base targeted")
    ax.set_ylabel("cumulative incremental outcome")
    ax.set_title(title)
    _subtitle(ax, "Area between the curves is the value a causal ranking adds over targeting at random.")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    return save(fig, name, figdir)


def plot_uplift_deciles(deciles: pd.DataFrame, name: str = "uplift_deciles.png",
                        figdir: str | Path = FIGDIR) -> Path:
    """Observed versus predicted uplift by decile of predicted effect.

    The eyeball test for a CATE model: observed uplift should increase monotonically, and the
    predicted bars should track it. Divergence means the ranking works but the calibration does not.
    """
    if deciles is None or len(deciles) == 0:
        return _placeholder("No decile table available for this run.", name, figdir)

    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    idx = np.arange(len(deciles))
    obs_col = next((c for c in ("observed_uplift", "uplift", "actual_uplift") if c in deciles), None)
    pred_col = next((c for c in ("predicted_uplift", "mean_predicted_cate", "predicted") if c in deciles), None)

    if obs_col is None:
        return _placeholder("Decile table has no observed-uplift column.", name, figdir)

    obs = deciles[obs_col].to_numpy(dtype=float)
    colours = [PALETTE["accent"] if v >= 0 else PALETTE["warn"] for v in obs]
    bars = ax.bar(idx, obs, color=colours, alpha=0.85, label=f"observed ({obs_col})", zorder=3)

    if "se" in deciles:
        ax.errorbar(idx, obs, yerr=1.96 * deciles["se"].to_numpy(dtype=float), fmt="none",
                    ecolor=PALETTE["dark"], elinewidth=1.0, capsize=3, zorder=4)

    if pred_col is not None:
        ax.plot(idx, deciles[pred_col].to_numpy(dtype=float), color=PALETTE["primary"], lw=2.0,
                marker="o", ms=4, label=f"predicted ({pred_col})", zorder=5)

    ax.axhline(0, color=PALETTE["dark"], lw=1.0, zorder=2)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i + 1) for i in idx])
    ax.set_xlabel("decile of predicted treatment effect (1 = highest)")
    ax.set_ylabel("uplift")
    ax.set_title("Uplift by predicted-effect decile")
    _subtitle(ax, "Green bars gain, orange bars lose. Bars below zero are the sleeping dogs.")
    ax.legend(loc="best")
    _ = bars
    return save(fig, name, figdir)


def plot_gates(gates: pd.DataFrame, name: str = "gates.png", figdir: str | Path = FIGDIR) -> Path:
    """Group Average Treatment Effects with confidence intervals.

    The formal version of the decile chart: if the top group's effect is not significantly above
    the bottom group's, the model has not found real heterogeneity.
    """
    if gates is None or len(gates) == 0 or "gate" not in gates:
        return _placeholder("No GATES table available for this run.", name, figdir)

    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    idx = np.arange(len(gates))
    est = gates["gate"].to_numpy(dtype=float)
    lo = gates["ci_low"].to_numpy(dtype=float) if "ci_low" in gates else est
    hi = gates["ci_high"].to_numpy(dtype=float) if "ci_high" in gates else est

    ax.errorbar(idx, est, yerr=[est - lo, hi - est], fmt="o", ms=7, color=PALETTE["primary"],
                ecolor=PALETTE["primary"], elinewidth=1.8, capsize=5, zorder=3)
    ax.axhline(0, color=PALETTE["dark"], lw=1.0, ls="--", zorder=2)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"G{i + 1}" for i in idx])
    ax.set_xlabel("group, ordered by predicted effect")
    ax.set_ylabel("doubly-robust ATE within group")
    ax.set_title("GATES: is the predicted heterogeneity real?")
    _subtitle(ax, "Rising groups with intervals clear of zero is the evidence. A flat line is not.")
    return save(fig, name, figdir)


def plot_calibration(calib: pd.DataFrame, name: str = "calibration.png",
                     figdir: str | Path = FIGDIR) -> Path:
    """Predicted versus observed, with the 45-degree reference line."""
    if calib is None or len(calib) == 0:
        return _placeholder("No calibration table available for this run.", name, figdir)

    pred_col = next((c for c in ("mean_predicted", "predicted", "mean_pred") if c in calib), None)
    obs_col = next((c for c in ("observed_km", "observed", "actual") if c in calib), None)
    if pred_col is None or obs_col is None:
        return _placeholder("Calibration table lacks predicted/observed columns.", name, figdir)

    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    p = calib[pred_col].to_numpy(dtype=float)
    o = calib[obs_col].to_numpy(dtype=float)

    lim = [float(min(p.min(), o.min())), float(max(p.max(), o.max()))]
    pad = 0.04 * (lim[1] - lim[0] + 1e-9)
    lim = [lim[0] - pad, lim[1] + pad]
    ax.plot(lim, lim, color=PALETTE["muted"], ls="--", lw=1.4, label="perfect calibration", zorder=2)

    if {"ci_low", "ci_high"} <= set(calib.columns):
        ax.errorbar(p, o, yerr=[o - calib["ci_low"], calib["ci_high"] - o], fmt="none",
                    ecolor=PALETTE["primary"], elinewidth=1.2, capsize=3, alpha=0.7, zorder=3)
    sizes = 40.0
    if "n" in calib:
        n = calib["n"].to_numpy(dtype=float)
        sizes = 25 + 120 * (n / max(n.max(), 1))
    ax.scatter(p, o, s=sizes, color=PALETTE["primary"], zorder=4, edgecolor="white", linewidth=0.8)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted")
    ax.set_ylabel("observed")
    ax.set_title("Calibration")
    _subtitle(ax, "Points above the line: the model under-predicts. Marker size is bin count.")
    ax.legend(loc="upper left")
    return save(fig, name, figdir)


def plot_survival_curves(curves: np.ndarray | pd.DataFrame, arm_names: Sequence[str] | None = None,
                         name: str = "survival_curves.png", figdir: str | Path = FIGDIR,
                         title: str = "Counterfactual survival under each offer") -> Path:
    """Mean counterfactual survival curve per arm.

    Parameters
    ----------
    curves : ndarray or DataFrame
        ``(n, n_arms, horizon)`` per-customer curves, ``(n_arms, horizon)`` means, or a frame
        whose columns are arms and index is time.
    """
    if curves is None:
        return _placeholder("No survival curves available for this run.", name, figdir)

    if isinstance(curves, pd.DataFrame):
        mean_curves = curves.to_numpy(dtype=float).T
        names = list(curves.columns)
    else:
        arr = np.asarray(curves, dtype=float)
        if arr.ndim == 3:
            mean_curves = arr.mean(axis=0)
        elif arr.ndim == 2:
            mean_curves = arr
        else:
            return _placeholder("Survival curve array has an unexpected shape.", name, figdir)
        names = list(arm_names) if arm_names is not None else [f"arm {i}" for i in range(mean_curves.shape[0])]

    if mean_curves.size == 0:
        return _placeholder("Survival curve array is empty.", name, figdir)

    apply_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    horizon = mean_curves.shape[1]
    t = np.arange(1, horizon + 1)

    for i, row in enumerate(mean_curves):
        ax.plot(t, row, lw=2.2, color=ARM_COLOURS[i % len(ARM_COLOURS)],
                label=names[i] if i < len(names) else f"arm {i}",
                ls="--" if i == 0 else "-")
    ax.set_xlabel("months since the decision")
    ax.set_ylabel("P(still a customer)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    _subtitle(ax, "Dashed = no offer. The vertical gap at each month is the retained share bought.")
    ax.legend(loc="lower left")

    # right panel: the area between each treated curve and control -- the RMST effect
    base = mean_curves[0]
    gains = [(names[i] if i < len(names) else f"arm {i}", float((mean_curves[i] - base).sum()))
             for i in range(1, mean_curves.shape[0])]
    if gains:
        labels, vals = zip(*gains)
        colours = [PALETTE["accent"] if v >= 0 else PALETTE["warn"] for v in vals]
        ax2.barh(np.arange(len(vals)), vals, color=colours, alpha=0.9)
        ax2.set_yticks(np.arange(len(vals)))
        ax2.set_yticklabels(labels)
        ax2.axvline(0, color=PALETTE["dark"], lw=1.0)
        ax2.set_xlabel("extra months retained (RMST effect)")
        ax2.set_title("Effect on expected lifetime")
        _subtitle(ax2, "This is what a binary churn flag at the horizon cannot see.")
        for i, v in enumerate(vals):
            ax2.text(v, i, f" {v:+.2f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=9, color=PALETTE["dark"])
    fig.tight_layout()
    return save(fig, name, figdir)


# =====================================================================================
# Decision figures
# =====================================================================================


def plot_efficient_frontier(frontier: pd.DataFrame, name: str = "efficient_frontier.png",
                            figdir: str | Path = FIGDIR) -> Path:
    """Incremental value as a function of budget, with the shadow price beneath it.

    The chart a budget conversation actually needs: how much value the next unit of spend buys,
    and where that return falls below 1.
    """
    if frontier is None or len(frontier) == 0 or "budget" not in frontier:
        return _placeholder("No efficient frontier available for this run.", name, figdir)

    apply_style()
    f = frontier.sort_values("budget")
    b = f["budget"].to_numpy(dtype=float)
    v = f["expected_incremental_value"].to_numpy(dtype=float)

    has_lambda = "dual_lambda" in f and np.isfinite(f["dual_lambda"].to_numpy(dtype=float)).any()
    if has_lambda:
        fig, (ax, axl) = plt.subplots(2, 1, figsize=(7.6, 6.2), sharex=True,
                                      gridspec_kw={"height_ratios": [2.1, 1]})
    else:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        axl = None

    ax.plot(b, v, color=PALETTE["primary"], lw=2.4, marker="o", ms=5, zorder=3)
    ax.fill_between(b, 0, v, color=PALETTE["primary"], alpha=0.12, zorder=1)

    # mark the point of diminishing returns: where marginal value per unit spent drops below 1
    if len(b) > 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            marginal = np.gradient(v, b)
        # Net value is already net of offer cost, so the budget stops being worth
        # increasing when the marginal return reaches zero, not when it drops below one.
        below = np.where(marginal <= 1e-9)[0]
        if len(below):
            k = int(below[0])
            ax.axvline(b[k], color=PALETTE["warn"], ls=":", lw=1.4, zorder=2)
            ax.annotate(f"budget stops binding\nat {_fmt_money(b[k])}",
                        xy=(b[k], v[k]), xytext=(10, -28), textcoords="offset points",
                        fontsize=9, color=PALETTE["warn"])

    ax.set_ylabel("expected incremental value")
    ax.set_title("Efficient frontier: what each extra unit of retention budget buys")
    _subtitle(ax, "Flattening means the remaining customers are not worth their offer.")
    ax.yaxis.set_major_formatter(_fmt_money)

    if axl is not None:
        lam = f["dual_lambda"].to_numpy(dtype=float)
        axl.plot(b, lam, color=PALETTE["secondary"], lw=2.0, marker="s", ms=4)
        axl.axhline(0.0, color=PALETTE["muted"], ls="--", lw=1.2)
        axl.set_ylabel("shadow price $\\lambda^*$")
        axl.set_xlabel("budget")
        axl.xaxis.set_major_formatter(_fmt_money)
        axl.text(0.01, 0.9, "profit added by the next unit of budget; at zero the budget stops binding",
                 transform=axl.transAxes, fontsize=8.5, color=PALETTE["muted"], va="top")
    else:
        ax.set_xlabel("budget")
        ax.xaxis.set_major_formatter(_fmt_money)

    fig.tight_layout()
    return save(fig, name, figdir)


def plot_policy_comparison(comparison: pd.DataFrame, name: str = "policy_comparison.png",
                           figdir: str | Path = FIGDIR) -> Path:
    """Policy value with confidence intervals, sorted -- the headline business chart."""
    if comparison is None or len(comparison) == 0 or "policy" not in comparison:
        return _placeholder("No policy comparison available for this run.", name, figdir)

    df = comparison.copy()
    if "method" in df.columns and df["method"].nunique() > 1:
        preferred = "dr" if (df["method"] == "dr").any() else df["method"].iloc[0]
        df = df[df["method"] == preferred]
    value_col = next((c for c in ("value", "expected_incremental_value", "net_value") if c in df), None)
    if value_col is None:
        return _placeholder("Policy comparison lacks a value column.", name, figdir)

    df = df.sort_values(value_col)
    apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 0.55 * len(df) + 2.2))
    idx = np.arange(len(df))
    vals = df[value_col].to_numpy(dtype=float)

    is_causal = df["policy"].astype(str).str.contains("causal|prism|optimis|optimiz|uplift", case=False)
    colours = [PALETTE["primary"] if c else PALETTE["muted"] for c in is_causal]
    ax.barh(idx, vals, color=colours, alpha=0.9, zorder=3)

    if {"ci_low", "ci_high"} <= set(df.columns):
        lo = df["ci_low"].to_numpy(dtype=float)
        hi = df["ci_high"].to_numpy(dtype=float)
        ax.errorbar(vals, idx, xerr=[np.maximum(vals - lo, 0), np.maximum(hi - vals, 0)],
                    fmt="none", ecolor=PALETTE["dark"], elinewidth=1.3, capsize=4, zorder=4)

    ax.axvline(0, color=PALETTE["dark"], lw=1.0, zorder=2)
    ax.set_yticks(idx)
    ax.set_yticklabels(df["policy"].astype(str))
    ax.set_xlabel("estimated policy value (doubly robust)")
    ax.set_title("What each targeting policy is worth")
    _subtitle(ax, "Bars are point estimates; whiskers are bootstrap CIs. Overlap means no proven win.")
    ax.xaxis.set_major_formatter(_fmt_money)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {_fmt_money(v)}", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=PALETTE["dark"])
    return save(fig, name, figdir)


def plot_learner_leaderboard(leaderboard: pd.DataFrame, name: str = "learner_leaderboard.png",
                             figdir: str | Path = FIGDIR) -> Path:
    """Estimator comparison across the metrics that matter, as a small-multiple bar chart."""
    if leaderboard is None or len(leaderboard) == 0:
        return _placeholder("No learner leaderboard available for this run.", name, figdir)

    df = leaderboard.copy()
    label_col = next((c for c in ("learner", "model", "name") if c in df), df.columns[0])
    # lower-is-better metrics get an inverted colour reading; note it in the axis label
    metrics = [c for c in ("pehe", "eps_ate", "qini", "auuc", "policy_value") if c in df]
    if not metrics:
        return _placeholder("Leaderboard has none of the expected metric columns.", name, figdir)

    apply_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.9), squeeze=False)
    lower_better = {"pehe", "eps_ate"}
    for ax, metric in zip(axes[0], metrics):
        sub = df[[label_col, metric]].dropna().sort_values(metric, ascending=metric in lower_better)
        vals = sub[metric].to_numpy(dtype=float)
        best = 0 if len(vals) else None
        colours = [PALETTE["accent"] if i == best else PALETTE["muted"] for i in range(len(vals))]
        ax.barh(np.arange(len(sub))[::-1], vals, color=colours, alpha=0.9)
        ax.set_yticks(np.arange(len(sub))[::-1])
        ax.set_yticklabels(sub[label_col].astype(str), fontsize=9)
        arrow = "lower is better" if metric in lower_better else "higher is better"
        ax.set_xlabel(f"{metric}\n({arrow})", fontsize=9)
        ax.set_title(metric.upper(), fontsize=10)
    fig.suptitle("Causal estimator leaderboard", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save(fig, name, figdir)


# =====================================================================================
# Diagnostics
# =====================================================================================


def plot_love_plot(smd: pd.DataFrame, name: str = "love_plot.png", figdir: str | Path = FIGDIR,
                   top_n: int = 25) -> Path:
    """Covariate balance before and after inverse-propensity weighting.

    The standard check that a propensity model has actually removed selection: |SMD| under 0.1
    after weighting.
    """
    if smd is None or len(smd) == 0 or "feature" not in smd:
        return _placeholder("No balance table available for this run.", name, figdir)

    df = smd.copy()
    un_col = next((c for c in ("abs_smd_unweighted", "smd_unweighted", "smd") if c in df), None)
    w_col = next((c for c in ("abs_smd_weighted", "smd_weighted") if c in df), None)
    if un_col is None:
        return _placeholder("Balance table lacks an unweighted SMD column.", name, figdir)

    if "arm" in df.columns and df["arm"].nunique() > 1:
        df = df.groupby("feature", as_index=False).agg({c: "mean" for c in (un_col, w_col) if c})
    df["_u"] = df[un_col].abs()
    if w_col:
        df["_w"] = df[w_col].abs()
    df = df.sort_values("_u", ascending=False).head(top_n).sort_values("_u")

    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 0.28 * len(df) + 2.2))
    idx = np.arange(len(df))
    ax.scatter(df["_u"], idx, s=52, color=PALETTE["warn"], label="unweighted", zorder=4,
               edgecolor="white", linewidth=0.7)
    if w_col:
        ax.scatter(df["_w"], idx, s=52, color=PALETTE["primary"], label="IPW-weighted", zorder=5,
                   edgecolor="white", linewidth=0.7)
        ax.hlines(idx, df["_w"], df["_u"], color=PALETTE["light"], lw=2.0, zorder=3)

    ax.axvline(0.1, color=PALETTE["dark"], ls="--", lw=1.2, zorder=2)
    ax.text(0.1, len(df) - 0.4, " |SMD| = 0.1 balance threshold", fontsize=8.5, color=PALETTE["dark"])
    ax.set_yticks(idx)
    ax.set_yticklabels(df["feature"].astype(str), fontsize=8.5)
    ax.set_xlabel("absolute standardised mean difference")
    ax.set_title("Covariate balance before and after weighting")
    _subtitle(ax, "Points must move left of the dashed line for the propensity model to be doing its job.")
    ax.legend(loc="lower right")
    ax.set_xlim(left=0)
    return save(fig, name, figdir)


def plot_segment_heatmap(segment_table: pd.DataFrame, name: str = "segment_heatmap.png",
                         figdir: str | Path = FIGDIR) -> Path:
    """Mean treatment effect by responder segment and arm.

    Parameters
    ----------
    segment_table : pandas.DataFrame
        Index = segment, columns = arm. Typically a ``groupby(segment)[tau_cols].mean()``.
    """
    if segment_table is None or segment_table.empty:
        return _placeholder("No segment table available for this run.", name, figdir)

    apply_style()
    data = segment_table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.55 * data.shape[1] + 3.2, 0.72 * data.shape[0] + 2.6))

    vmax = float(np.nanmax(np.abs(data))) or 1.0
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels([str(c) for c in segment_table.columns], rotation=20, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels([str(i) for i in segment_table.index])
    ax.grid(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(v) > 0.6 * vmax else PALETTE["dark"],
                    fontweight="bold" if abs(v) > 0.6 * vmax else "normal")

    ax.set_title("Treatment effect by responder segment and offer")
    _subtitle(ax, "Red cells are customers the offer actively costs you: the sleeping dogs.")
    fig.colorbar(im, ax=ax, shrink=0.82, label="mean effect on discounted value")
    fig.tight_layout()
    return save(fig, name, figdir)


def plot_sensitivity_contour(contour: pd.DataFrame, name: str = "sensitivity_contour.png",
                             figdir: str | Path = FIGDIR) -> Path:
    """Austen-style plot: how strong would a hidden confounder have to be to erase the effect?"""
    if contour is None or len(contour) == 0 or not {"r2_y", "r2_w"} <= set(contour.columns):
        return _placeholder("No sensitivity contour available for this run.", name, figdir)

    apply_style()
    grid = contour[~contour.get("benchmark", pd.Series(False, index=contour.index)).astype(bool)]
    bench = contour[contour.get("benchmark", pd.Series(False, index=contour.index)).astype(bool)]
    if grid.empty:
        grid = contour

    value_col = "adjusted_estimate" if "adjusted_estimate" in grid else "bias"
    try:
        pivot = grid.pivot_table(index="r2_y", columns="r2_w", values=value_col, aggfunc="mean")
    except Exception:
        return _placeholder("Sensitivity contour could not be pivoted into a grid.", name, figdir)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    X, Y = np.meshgrid(pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float))
    Z = pivot.to_numpy(dtype=float)

    cf = ax.contourf(X, Y, Z, levels=14, cmap="RdYlBu_r", alpha=0.9)
    cs = ax.contour(X, Y, Z, levels=[0.0], colors=[PALETTE["dark"]], linewidths=2.2)
    ax.clabel(cs, fmt={0.0: "estimate nullified"}, fontsize=9)

    if not bench.empty:
        ax.scatter(bench["r2_w"], bench["r2_y"], s=64, marker="D", color=PALETTE["dark"],
                   zorder=6, edgecolor="white", linewidth=1.0, label="observed covariates")
        label_col = next((c for c in ("feature", "name", "benchmark_name") if c in bench.columns), None)
        if label_col:
            for _, r in bench.iterrows():
                ax.annotate(str(r[label_col]), (r["r2_w"], r["r2_y"]), xytext=(6, 5),
                            textcoords="offset points", fontsize=8, color=PALETTE["dark"])
        ax.legend(loc="upper right")

    ax.set_xlabel("partial $R^2$ of the hidden confounder with treatment")
    ax.set_ylabel("partial $R^2$ with the outcome")
    ax.set_title("Sensitivity to an unmeasured confounder")
    _subtitle(ax, "A confounder must land beyond the black line to explain the result away.")
    fig.colorbar(cf, ax=ax, shrink=0.85, label=value_col.replace("_", " "))
    fig.tight_layout()
    return save(fig, name, figdir)


def plot_drift(drift: pd.DataFrame, name: str = "drift.png", figdir: str | Path = FIGDIR,
               top_n: int = 20) -> Path:
    """Feature drift ranked by PSI, coloured by severity."""
    if drift is None or len(drift) == 0 or "feature" not in drift:
        return _placeholder("No drift report available for this run.", name, figdir)

    df = drift.copy()
    if "psi" not in df:
        return _placeholder("Drift report has no PSI column.", name, figdir)
    df = df.dropna(subset=["psi"]).sort_values("psi", ascending=False).head(top_n).sort_values("psi")

    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 0.3 * len(df) + 2.4))
    idx = np.arange(len(df))
    psi = df["psi"].to_numpy(dtype=float)
    colours = [PALETTE["warn"] if v >= 0.25 else (PALETTE["secondary"] if v >= 0.10 else PALETTE["accent"])
               for v in psi]
    ax.barh(idx, psi, color=colours, alpha=0.9, zorder=3)
    ax.axvline(0.10, color=PALETTE["dark"], ls=":", lw=1.2, zorder=2)
    ax.axvline(0.25, color=PALETTE["dark"], ls="--", lw=1.2, zorder=2)
    ax.text(0.10, len(df) - 0.4, " warn", fontsize=8.5, color=PALETTE["dark"])
    ax.text(0.25, len(df) - 0.4, " alert", fontsize=8.5, color=PALETTE["dark"])
    ax.set_yticks(idx)
    ax.set_yticklabels(df["feature"].astype(str), fontsize=8.5)
    ax.set_xlabel("population stability index")
    ax.set_title("Feature drift, train reference versus current period")
    _subtitle(ax, "PSI above 0.25 means the model is scoring a population it was not trained on.")
    ax.set_xlim(left=0)
    return save(fig, name, figdir)


# =====================================================================================
# Orchestration
# =====================================================================================


def generate_all(artifacts: dict[str, Any], figdir: str | Path = FIGDIR) -> dict[str, Path]:
    """Render every figure the pipeline can produce from whatever is present.

    Parameters
    ----------
    artifacts : dict
        Loose bag of frames keyed by name: ``qini``, ``deciles``, ``gates``, ``calibration``,
        ``survival_curves`` (plus optional ``arm_names``), ``frontier``, ``policy_comparison``,
        ``leaderboard``, ``smd``, ``segment_table``, ``sensitivity``, ``drift``.
        Missing keys are skipped, not fatal.
    figdir : str or Path
        Output directory.

    Returns
    -------
    dict of str to pathlib.Path
        Figure name to the path written.
    """
    apply_style()
    Path(figdir).mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Any, tuple, dict]] = [
        ("qini.png", plot_qini, (artifacts.get("qini"),), {}),
        ("uplift_deciles.png", plot_uplift_deciles, (artifacts.get("deciles"),), {}),
        ("gates.png", plot_gates, (artifacts.get("gates"),), {}),
        ("calibration.png", plot_calibration, (artifacts.get("calibration"),), {}),
        ("survival_curves.png", plot_survival_curves,
         (artifacts.get("survival_curves"), artifacts.get("arm_names")), {}),
        ("efficient_frontier.png", plot_efficient_frontier, (artifacts.get("frontier"),), {}),
        ("policy_comparison.png", plot_policy_comparison, (artifacts.get("policy_comparison"),), {}),
        ("learner_leaderboard.png", plot_learner_leaderboard, (artifacts.get("leaderboard"),), {}),
        ("love_plot.png", plot_love_plot, (artifacts.get("smd"),), {}),
        ("segment_heatmap.png", plot_segment_heatmap, (artifacts.get("segment_table"),), {}),
        ("sensitivity_contour.png", plot_sensitivity_contour, (artifacts.get("sensitivity"),), {}),
        ("drift.png", plot_drift, (artifacts.get("drift"),), {}),
    ]

    written: dict[str, Path] = {}
    for fname, fn, args, kwargs in jobs:
        try:
            written[fname] = fn(*args, name=fname, figdir=figdir, **kwargs)
        except Exception as exc:  # a broken chart must never kill a pipeline run
            log.warning("figure %s failed (%s); writing a placeholder", fname, exc)
            try:
                written[fname] = _placeholder(f"Figure failed to render: {exc}", fname, figdir)
            except Exception:
                log.error("could not even write a placeholder for %s", fname)
    log.info("wrote %d figures to %s", len(written), figdir)
    return written


if __name__ == "__main__":  # pragma: no cover - smoke test
    import shutil
    import tempfile

    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp(prefix="prism_fig_"))

    frac = np.linspace(0, 1, 40)
    art = {
        "qini": pd.DataFrame(
            {"frac_targeted": frac,
             "incremental": 100 * (1 - (1 - frac) ** 2),
             "random_incremental": 100 * frac}
        ),
        "deciles": pd.DataFrame(
            {"decile": np.arange(1, 11),
             "observed_uplift": np.linspace(24, -7, 10) + rng.normal(0, 0.8, 10),
             "predicted_uplift": np.linspace(22, -6, 10),
             "se": np.full(10, 1.6)}
        ),
        "gates": pd.DataFrame(
            {"group": np.arange(1, 6), "gate": [-3.0, 1.0, 4.0, 9.0, 17.0],
             "ci_low": [-6, -1.5, 1.2, 6, 13], "ci_high": [0.2, 3.5, 6.8, 12, 21]}
        ),
        "calibration": pd.DataFrame(
            {"mean_predicted": np.linspace(0.05, 0.9, 8),
             "observed_km": np.linspace(0.08, 0.86, 8) + rng.normal(0, 0.02, 8),
             "ci_low": np.linspace(0.02, 0.8, 8), "ci_high": np.linspace(0.14, 0.93, 8),
             "n": rng.integers(50, 600, 8)}
        ),
        "survival_curves": np.stack([
            np.cumprod(np.full(12, 1 - h)) for h in (0.075, 0.060, 0.050, 0.046)
        ]),
        "arm_names": ["control", "discount_10", "discount_25", "concierge"],
        "frontier": pd.DataFrame(
            {"budget": [0, 25e3, 50e3, 100e3, 200e3, 400e3],
             "expected_incremental_value": [0, 180e3, 320e3, 505e3, 610e3, 640e3],
             "n_treated": [0, 2100, 4200, 8300, 15800, 27000],
             "dual_lambda": [np.inf, 8.4, 5.1, 2.6, 0.9, 0.2]}
        ),
        "policy_comparison": pd.DataFrame(
            {"policy": ["causal (PRISM)", "highest_risk", "risk_x_clv", "highest_clv",
                        "treat_all", "random", "treat_none"],
             "method": ["dr"] * 7,
             "value": [505e3, 280e3, 310e3, 240e3, -90e3, 120e3, 0.0],
             "ci_low": [470e3, 245e3, 272e3, 205e3, -130e3, 88e3, 0.0],
             "ci_high": [540e3, 315e3, 348e3, 275e3, -50e3, 152e3, 0.0]}
        ),
        "leaderboard": pd.DataFrame(
            {"learner": ["dr", "r", "forest", "x", "t", "s"],
             "pehe": [4.1, 4.3, 4.0, 5.2, 6.1, 8.8],
             "eps_ate": [0.3, 0.4, 0.35, 0.9, 1.4, 3.2],
             "qini": [0.41, 0.40, 0.42, 0.34, 0.29, 0.14]}
        ),
        "smd": pd.DataFrame(
            {"feature": [f"f{i}" for i in range(14)],
             "abs_smd_unweighted": np.abs(rng.normal(0.28, 0.14, 14)),
             "abs_smd_weighted": np.abs(rng.normal(0.045, 0.025, 14))}
        ),
        "segment_table": pd.DataFrame(
            [[17.4, 27.7, 32.5], [4.2, 7.1, 8.3], [2.1, 2.8, 0.6], [-9.6, -16.9, -28.7]],
            index=["persuadable", "sure_thing", "lost_cause", "sleeping_dog"],
            columns=["discount_10", "discount_25", "concierge"],
        ),
        "sensitivity": pd.concat([
            pd.DataFrame([{"r2_y": y, "r2_w": w, "adjusted_estimate": 12.0 - 46 * np.sqrt(y * w),
                           "benchmark": False}
                          for y in np.linspace(0.01, 0.3, 12) for w in np.linspace(0.01, 0.3, 12)]),
            pd.DataFrame({"r2_y": [0.05, 0.12], "r2_w": [0.08, 0.04],
                          "adjusted_estimate": [9.1, 7.2], "benchmark": [True, True],
                          "feature": ["tenure_months", "engagement_score"]}),
        ], ignore_index=True),
        "drift": pd.DataFrame(
            {"feature": [f"f{i}" for i in range(12)],
             "psi": np.sort(rng.gamma(1.4, 0.10, 12))[::-1],
             "severity": ["alert"] * 2 + ["warn"] * 3 + ["ok"] * 7}
        ),
    }

    written = generate_all(art, figdir=tmp)
    assert len(written) == 12, f"expected 12 figures, wrote {len(written)}"
    for fname, path in written.items():
        assert path.exists() and path.stat().st_size > 5_000, f"{fname} looks empty ({path})"

    # a completely empty artifact bag must still produce placeholders, never an exception
    empty = generate_all({}, figdir=tmp / "empty")
    assert len(empty) == 12 and all(p.exists() for p in empty.values())

    print(f"rendered {len(written)} figures + {len(empty)} placeholders")
    print("largest:", max(written.items(), key=lambda kv: kv[1].stat().st_size)[0])
    shutil.rmtree(tmp, ignore_errors=True)
    print("figures.py OK")
