"""PRISM executive dashboard.

Run with::

    streamlit run prism/app/dashboard.py

Built for two readers at once. A retention lead should be able to move the budget slider and see
what it buys. A reviewing data scientist should be able to reach the diagnostics that say whether
any of it is trustworthy -- calibration, balance, refutation, drift -- without leaving the page.

Everything is read from ``artifacts/reports`` and ``docs/figures``, so the dashboard never trains
anything. If the pipeline has not been run, each tab says so and tells you the command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PRISM - Causal Retention",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

REPORTS = Path("artifacts/reports")
FIGURES = Path("docs/figures")
RUN_HINT = "`python -m prism.pipelines.run_all --config configs/fast.yaml --steps all`"

ARM_NAMES = ("control", "discount_10", "discount_25", "concierge")


# =====================================================================================
# Loading
# =====================================================================================


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict[str, Any] | None:
    """Read a JSON report, or None if it has not been generated."""
    p = REPORTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    """Read a CSV report, or None if it has not been generated."""
    p = REPORTS / name
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df if len(df) else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_decisions() -> pd.DataFrame | None:
    """Read the per-customer decision frame that powers the live budget simulator."""
    for name in ("decision_frame.parquet", "decision_frame.csv"):
        p = REPORTS / name
        if p.exists():
            try:
                return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            except Exception:
                continue
    return None


def load_text(name: str) -> str | None:
    """Read a generated markdown report."""
    p = REPORTS / name
    return p.read_text(encoding="utf-8") if p.exists() else None


def figure(name: str, caption: str | None = None) -> bool:
    """Render a generated figure if it exists. Returns whether anything was shown."""
    p = FIGURES / name
    if not p.exists():
        return False
    st.image(str(p), caption=caption, use_container_width=True)
    return True


def not_run(what: str) -> None:
    """Consistent empty state."""
    st.info(f"No {what} yet. Generate it with:\n\n{RUN_HINT}")


def fmt_money(x: float | None) -> str:
    """Compact currency formatting."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if a >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if a >= 1e3:
        return f"{x / 1e3:,.1f}k"
    return f"{x:,.0f}"


# =====================================================================================
# Live budget re-optimisation
# =====================================================================================


def net_matrix(decisions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract the (n, n_arms) net-value and cost matrices from the decision frame."""
    net_cols = [c for c in decisions.columns if c.startswith("net_value_")]
    net_cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
    net = decisions[net_cols].to_numpy(dtype=float)

    cost_cols = [c for c in decisions.columns if c.startswith("cost_")]
    cost_cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
    if cost_cols:
        cost = decisions[cost_cols].to_numpy(dtype=float)
    else:
        from prism.data.schema import ARM_COSTS, ARM_REDEMPTION

        unit = np.asarray([c * r for c, r in zip(ARM_COSTS, ARM_REDEMPTION)], dtype=float)
        cost = np.tile(unit[: net.shape[1]], (len(net), 1))
    return net, cost


@st.cache_data(show_spinner=False)
def allocate(net: np.ndarray, cost: np.ndarray, budget: float) -> dict[str, Any]:
    """Solve the budget-constrained assignment. Cached so the slider feels instant."""
    from prism.decision.optimize import lagrangian_allocate

    res = lagrangian_allocate(net, cost, float(budget))
    assign = np.asarray(res.assignment, dtype=int)
    return {
        "assignment": assign,
        "total_cost": float(res.total_cost),
        "value": float(res.expected_incremental_value),
        "n_treated": int((assign > 0).sum()),
        "lambda": (float(res.dual_lambda) if res.dual_lambda is not None and np.isfinite(res.dual_lambda) else None),
    }


# =====================================================================================
# Tabs
# =====================================================================================


def tab_executive(metrics: dict | None, policies: pd.DataFrame | None) -> None:
    """Headline numbers and the charts an executive needs."""
    st.subheader("Is the retention budget being spent on the right customers?")

    if metrics is None:
        not_run("pipeline results")
        return

    c1, c2, c3, c4 = st.columns(4)
    causal = metrics.get("policy_value_optimised")
    risk = metrics.get("policy_value_highest_risk")
    uplift = metrics.get("policy_uplift_vs_risk_targeting")
    oracle = metrics.get("oracle_value_captured_pct")

    c1.metric("Causal policy value", fmt_money(causal))
    c2.metric("Risk-targeting baseline", fmt_money(risk))
    c3.metric(
        "Gain over risk targeting",
        fmt_money(uplift),
        delta=(f"{100 * uplift / abs(risk):.0f}%" if (uplift and risk) else None),
    )
    c4.metric("Share of oracle value captured", f"{oracle:.1f}%" if oracle is not None else "-")

    st.caption(
        "The oracle knows every customer's true treatment effect. Capturing a large share of its "
        "value is the honest measure of how good the targeting is."
    )

    st.divider()
    left, right = st.columns([1.15, 1])
    with left:
        if not figure("policy_comparison.png"):
            if policies is not None:
                st.dataframe(policies, use_container_width=True, hide_index=True)
            else:
                not_run("policy comparison")
    with right:
        if not figure("efficient_frontier.png"):
            not_run("efficient frontier")

    st.divider()
    bias = load_csv("confounding_bias.csv")
    if bias is not None:
        st.markdown("**Why not just compare treated against untreated?**")
        st.caption(
            "Offers were not handed out at random -- riskier customers were contacted more. So a "
            "straight comparison of treated versus untreated measures the selection, not the offer."
        )
        cols = [c for c in ("arm_name", "n_treated", "true_ate", "naive_diff_in_means",
                            "ipw_estimate", "model_estimate", "naive_sign_wrong") if c in bias.columns]
        st.dataframe(
            bias[cols].rename(
                columns={
                    "arm_name": "offer",
                    "true_ate": "true effect",
                    "naive_diff_in_means": "naive comparison",
                    "ipw_estimate": "IPW adjusted",
                    "model_estimate": "PRISM estimate",
                    "naive_sign_wrong": "naive has wrong sign",
                }
            ).round(2),
            use_container_width=True,
            hide_index=True,
        )
        m = metrics or {}
        if m.get("bias_reduction_vs_naive_pct") is not None:
            n_wrong = m.get("n_arms_naive_sign_wrong")
            msg = (
                f"Adjustment removes **{m['bias_reduction_vs_naive_pct']:.0f}%** of the naive bias "
                f"(mean absolute error {m.get('naive_mean_abs_error', float('nan')):.1f} "
                f"to {m.get('model_mean_abs_error', float('nan')):.1f})."
            )
            if n_wrong:
                msg += f" The naive comparison gets the **sign wrong on {n_wrong}** of the offers."
            st.success(msg)

    st.divider()
    memo = load_text("decision_memo.md")
    if memo:
        with st.expander("One-page decision memo", expanded=False):
            st.markdown(memo)


def tab_simulator(decisions: pd.DataFrame | None, metrics: dict | None) -> None:
    """Move the budget, see what it buys. Re-solves the allocation live."""
    st.subheader("Budget simulator")
    st.caption(
        "Re-solves the multiple-choice knapsack at the chosen budget. The shadow price is the "
        "return on the next unit of spend: below 1.0, the next offer destroys value."
    )

    if decisions is None:
        not_run("per-customer decision frame")
        return

    net, cost = net_matrix(decisions)
    if net.size == 0:
        st.warning("The decision frame has no net-value columns.")
        return

    max_spend = float(cost.max(axis=1).sum())
    default = float(metrics.get("budget", max_spend * 0.15)) if metrics else max_spend * 0.15
    default = float(np.clip(default, 0.0, max_spend))

    budget = st.slider(
        "Retention budget",
        min_value=0.0,
        max_value=round(max_spend, 2),
        value=round(default, 2),
        step=max(round(max_spend / 200, 2), 0.01),
        format="%.0f",
    )

    res = allocate(net, cost, budget)
    a, b, c, d, e = st.columns(5)
    a.metric("Customers targeted", f"{res['n_treated']:,}")
    b.metric("Budget used", fmt_money(res["total_cost"]))
    c.metric("Expected incremental value", fmt_money(res["value"]))
    roi = res["value"] / res["total_cost"] if res["total_cost"] > 0 else None
    d.metric("ROI", f"{roi:.2f}x" if roi else "-")
    e.metric("Shadow price", f"{res['lambda']:.2f}" if res["lambda"] is not None else "-")

    # lambda* = d(net value)/d(budget), and net value is already net of offer cost, so the
    # meaningful threshold is zero rather than one.
    if res["lambda"] is not None and res["n_treated"] > 0:
        if res["lambda"] <= 1e-6:
            st.warning(
                "The budget does not bind at this level: every offer worth making is already "
                "funded and money is left over. The same result is achievable for less."
            )
        else:
            st.info(
                f"Each additional unit of budget still returns {res['lambda']:.2f} in incremental "
                "profit, so this programme is under-funded rather than over-funded."
            )

    st.divider()
    assign = res["assignment"]
    n_arms = net.shape[1]
    breakdown = pd.DataFrame(
        {
            "offer": [ARM_NAMES[i] if i < len(ARM_NAMES) else f"arm {i}" for i in range(n_arms)],
            "customers": [int((assign == i).sum()) for i in range(n_arms)],
            "spend": [float(cost[assign == i, i].sum()) for i in range(n_arms)],
            "incremental value": [float(net[assign == i, i].sum()) for i in range(n_arms)],
        }
    )
    breakdown["share of base"] = breakdown["customers"] / max(len(assign), 1)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Who gets what**")
        st.dataframe(
            breakdown.style.format(
                {"spend": "{:,.0f}", "incremental value": "{:,.0f}", "share of base": "{:.1%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("**Spend by offer**")
        st.bar_chart(breakdown.set_index("offer")["incremental value"])

    if "segment_guess" in decisions.columns:
        st.markdown("**Who the budget reaches, by responder segment**")
        seg = (
            pd.DataFrame({"segment": decisions["segment_guess"].astype(str), "treated": assign > 0})
            .groupby("segment")
            .agg(customers=("treated", "size"), targeted=("treated", "sum"))
        )
        seg["targeting rate"] = seg["targeted"] / seg["customers"]
        st.dataframe(
            seg.style.format({"targeting rate": "{:.1%}"}), use_container_width=True
        )
        st.caption(
            "A working causal policy concentrates spend on persuadables and leaves sleeping dogs "
            "alone. A churn model cannot tell them apart."
        )


def tab_segments() -> None:
    """Where the heterogeneity lives."""
    st.subheader("Who responds, and who is harmed")
    st.caption(
        "Offers do not affect everyone the same way. Some customers are actively made more likely "
        "to leave by being contacted -- the sleeping dogs -- and targeting them destroys value."
    )
    left, right = st.columns([1, 1])
    with left:
        if not figure("segment_heatmap.png", "Mean effect on discounted value"):
            not_run("segment heatmap")
    with right:
        if not figure("uplift_deciles.png", "Observed vs predicted uplift by decile"):
            not_run("uplift deciles")
    st.divider()
    if not figure("survival_curves.png", "Counterfactual survival under each offer"):
        not_run("survival curves")


def tab_diagnostics(leaderboard: pd.DataFrame | None) -> None:
    """Is the model any good?"""
    st.subheader("Model diagnostics")

    if leaderboard is not None:
        st.markdown("**Causal estimator leaderboard**")
        st.caption(
            "PEHE and eps_ATE are measured against the simulator's known individual treatment "
            "effects -- the check no observational study can run. Lower is better for both."
        )
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)
    else:
        not_run("estimator leaderboard")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        figure("qini.png", "Qini curve") or not_run("Qini curve")
        figure("calibration.png", "Calibration") or None
    with c2:
        figure("gates.png", "GATES: is the heterogeneity real?") or None
        figure("love_plot.png", "Covariate balance before and after weighting") or None

    st.divider()
    with st.expander("How to read these", expanded=False):
        st.markdown(
            """
- **Qini** measures *ranking*: how much more value you capture than targeting at random.
- **GATES** is the formal test. Group the customers by predicted effect and estimate the true
  effect in each group with confidence intervals. Rising groups with intervals clear of zero is
  real heterogeneity; a flat line means the model found nothing, however good its Qini looked.
- **Calibration** asks whether the *size* of the predicted effects is right, not just their order.
  A model can rank perfectly and still be badly scaled.
- **Love plot** checks the propensity model actually removed selection. Points must move inside
  the 0.1 line after weighting, otherwise the treated and untreated groups are still different.
"""
        )


def tab_validity(refutation: pd.DataFrame | None, metrics: dict | None) -> None:
    """Would this survive a skeptical reviewer?"""
    st.subheader("Causal validity")
    st.caption(
        "Passing a validation metric is not evidence of a causal effect. These are the attempts "
        "to break the finding."
    )

    if refutation is not None:
        passed = int(refutation["passed"].sum()) if "passed" in refutation else None
        if passed is not None:
            st.metric("Refutation tests passed", f"{passed} / {len(refutation)}")
        st.dataframe(refutation, use_container_width=True, hide_index=True)
    else:
        not_run("refutation suite")

    st.divider()
    c1, c2 = st.columns([1.2, 1])
    with c1:
        if not figure("sensitivity_contour.png", "Sensitivity to an unmeasured confounder"):
            not_run("sensitivity analysis")
    with c2:
        ev = (metrics or {}).get("e_value_point")
        evc = (metrics or {}).get("e_value_ci")
        if ev is not None:
            st.metric("E-value (point estimate)", f"{ev:.2f}")
            if evc is not None:
                st.metric("E-value (CI bound nearest the null)", f"{evc:.2f}")
            st.caption(
                "The minimum strength of association an unmeasured confounder would need with "
                "*both* treatment and outcome to explain the result away. Compare it to the "
                "strongest observed covariate: if that is far weaker, the finding is robust."
            )
        st.markdown(
            "**Assumptions this rests on**\n\n"
            "- consistency / SUTVA (no interference between customers)\n"
            "- positivity (measured; see the overlap diagnostics)\n"
            "- unconfoundedness (**known to be false here** -- a hidden confounder is simulated "
            "deliberately, which is what these tests are quantifying)\n"
            "- non-informative censoring given covariates (handled by IPCW)"
        )


def tab_monitoring(drift: pd.DataFrame | None, metrics: dict | None) -> None:
    """Would we notice if this stopped working?"""
    st.subheader("Monitoring")

    c1, c2, c3 = st.columns(3)
    m = metrics or {}
    c1.metric("Features drifting (PSI > 0.25)", m.get("n_features_alert", "-"))
    rc = m.get("cate_rank_correlation")
    c2.metric("CATE rank stability", f"{rc:.2f}" if isinstance(rc, (int, float)) else "-")
    sf = m.get("cate_sign_flip_rate")
    c3.metric("CATE sign-flip rate", f"{sf:.1%}" if isinstance(sf, (int, float)) else "-")

    st.caption(
        "Monitoring a causal model is not the same as monitoring a predictive one. What matters "
        "is whether the *decision* would change. A sign flip turns 'treat' into 'do not treat', "
        "which is far more damaging than a small drop in accuracy."
    )

    st.divider()
    if drift is not None:
        st.dataframe(drift, use_container_width=True, hide_index=True)
    else:
        not_run("drift report")
    figure("drift.png", "Feature drift by PSI")


def tab_fairness(fairness: pd.DataFrame | None) -> None:
    """Who gets the scarce benefit?"""
    st.subheader("Fairness audit")
    st.caption(
        "A retention budget is a scarce benefit, and a profit-maximising allocation concentrates "
        "it. The question is not whether that happens but whether it is acceptable, and what "
        "enforcing parity would cost."
    )
    if fairness is not None:
        st.dataframe(fairness, use_container_width=True, hide_index=True)
    else:
        not_run("fairness audit")


def tab_model_card() -> None:
    """The generated model card and methodology."""
    st.subheader("Model card")
    card = load_text("model_card.md")
    if card:
        st.markdown(card)
    else:
        not_run("model card")
    with st.expander("Methodology (estimands, identification, estimators)", expanded=False):
        meth = Path("docs/METHODOLOGY.md")
        st.markdown(meth.read_text(encoding="utf-8") if meth.exists() else "`docs/METHODOLOGY.md` not found.")


# =====================================================================================
# Main
# =====================================================================================


def main() -> None:
    """Render the dashboard."""
    metrics = load_json("metrics.json")
    leaderboard = load_csv("leaderboard.csv")
    policies = load_csv("policy_comparison.csv")
    refutation = load_csv("refutation.csv")
    drift = load_csv("drift.csv")
    fairness = load_csv("fairness.csv")
    decisions = load_decisions()

    st.title("PRISM")
    st.markdown(
        "**Causal Survival Uplift for retention.** Not *who will churn* -- *who will stay because "
        "you intervened, how much extra lifetime that buys, and how to spend a fixed budget on it.*"
    )

    with st.sidebar:
        st.header("Run")
        if metrics:
            st.success("Pipeline results loaded")
            for key, label in (
                ("model_version", "version"),
                ("estimator", "estimator"),
                ("n_rows_panel", "panel rows"),
                ("n_customers", "customers"),
                ("horizon", "horizon (months)"),
            ):
                if key in metrics:
                    st.caption(f"**{label}**: {metrics[key]}")
        else:
            st.warning("No results found")
            st.code(
                "python -m prism.pipelines.run_all \\\n"
                "  --config configs/fast.yaml --steps all",
                language="bash",
            )
        st.divider()
        st.caption(
            "Artifacts are read from `artifacts/reports` and `docs/figures`. "
            "This page never trains a model."
        )
        if st.button("Reload artifacts"):
            st.cache_data.clear()
            st.rerun()

    tabs = st.tabs(
        [
            "Executive",
            "Budget simulator",
            "Segments",
            "Diagnostics",
            "Causal validity",
            "Monitoring",
            "Fairness",
            "Model card",
        ]
    )
    with tabs[0]:
        tab_executive(metrics, policies)
    with tabs[1]:
        tab_simulator(decisions, metrics)
    with tabs[2]:
        tab_segments()
    with tabs[3]:
        tab_diagnostics(leaderboard)
    with tabs[4]:
        tab_validity(refutation, metrics)
    with tabs[5]:
        tab_monitoring(drift, metrics)
    with tabs[6]:
        tab_fairness(fairness)
    with tabs[7]:
        tab_model_card()


main()
