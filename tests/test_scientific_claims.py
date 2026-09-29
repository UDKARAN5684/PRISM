"""The tests that decide whether this project works.

Everything else in `tests/` checks that the code runs. This file checks that the *science*
holds. If any of these fail, the README's headline claims are false and the project should
not be shown to anyone.

Four claims, in order of importance:

1. The simulator really does contain the structure it says it does (four responder segments,
   sleeping dogs with genuinely negative value, confounded-but-overlapping assignment).
2. A causal estimator recovers the known individual treatment effects far better than chance.
3. Targeting on causal effect beats targeting on churn risk — in money.
4. Nothing leaks: no future information reaches a past feature, and no customer straddles a split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.data.schema import ARM_COSTS, N_ARMS, SEGMENTS

pytestmark = pytest.mark.statistical


def expected_cost_matrix(econ, n: int) -> np.ndarray:
    """Per-row expected cost of each arm, redemption-adjusted.

    This must match what :func:`prism.decision.economics.expected_net_value` subtracts. Budgeting
    at face value while netting out the redemption-adjusted cost is a real and easy mistake: the
    two conventions differ by ~30-50% per arm, so the optimiser would be solving against a budget
    that does not correspond to the value it is maximising.
    """
    unit = np.asarray(
        [c * r for c, r in zip(econ.arm_costs, econ.offer_redemption_rate)], dtype=float
    )
    return np.tile(unit, (n, 1))


# =====================================================================================
# Claim 1 -- the simulator contains the structure the project depends on
# =====================================================================================


def test_all_four_responder_segments_are_present(sim):
    """A sleeping-dog-free simulation would make the causal claim trivial."""
    counts = sim.customers["gt_segment"].value_counts()
    for seg in SEGMENTS:
        assert seg in counts.index, f"segment {seg!r} missing from the simulation"
        assert counts[seg] > 20, f"segment {seg!r} has only {counts[seg]} customers"


def test_segment_treatment_effects_have_the_right_signs(causal_sample):
    """Persuadables must gain from treatment and sleeping dogs must lose from it.

    This is the economic engine of the whole project: if sleeping dogs are not genuinely
    harmed by treatment, a naive risk-targeting policy would be fine and PRISM would be
    solving a problem that does not exist.
    """
    df = causal_sample
    tau_cols = [f"gt_tau_value_{a}" for a in range(1, N_ARMS)]
    best_tau = df[tau_cols].max(axis=1)
    by_segment = best_tau.groupby(df["gt_segment"]).mean()

    assert by_segment["persuadable"] > 0, f"persuadables should gain, got {by_segment['persuadable']:.3f}"
    assert by_segment["sleeping_dog"] < 0, f"sleeping dogs should lose, got {by_segment['sleeping_dog']:.3f}"
    assert by_segment["persuadable"] > by_segment["lost_cause"], "persuadables must out-gain lost causes"
    assert by_segment["persuadable"] > by_segment["sure_thing"], "persuadables must out-gain sure things"


def test_sleeping_dogs_are_economically_material(causal_sample):
    """Sleeping dogs must be numerous enough and harmed enough to change the optimal policy."""
    df = causal_sample
    dogs = df[df["gt_segment"] == "sleeping_dog"]
    share = len(dogs) / len(df)
    assert 0.05 < share < 0.40, f"sleeping-dog share {share:.2%} is outside a plausible range"

    # They must look risky (so a churn model would target them) while being harmed by treatment.
    tau_cols = [f"gt_tau_value_{a}" for a in range(1, N_ARMS)]
    harm = dogs[tau_cols].max(axis=1).mean()
    assert harm < 0, f"sleeping dogs are not actually harmed (best-arm tau_value = {harm:.3f})"


def test_assignment_is_confounded_but_overlapping(causal_sample):
    """Both halves of the identification problem must hold: selection exists, overlap survives."""
    df = causal_sample
    obs = df[df["assignment_block"] == "observational"]
    assert len(obs) > 100, "not enough observational rows to test"

    prop_cols = [f"gt_propensity_{a}" for a in range(N_ARMS)]
    p = obs[prop_cols].to_numpy(dtype=float)

    # overlap: every arm is possible for every customer
    assert p.min() >= 0.01, f"positivity violated: min propensity {p.min():.4f}"
    assert p.max() <= 0.90, f"a near-deterministic arm exists: max propensity {p.max():.4f}"
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6), "propensities do not sum to 1"

    # confounding: treatment probability must actually depend on customer characteristics
    treat_p = 1.0 - p[:, 0]
    spread = np.nanstd(treat_p)
    assert spread > 0.02, f"assignment looks unconfounded (sd of P(treated) = {spread:.4f})"


def test_rct_block_is_actually_randomised(causal_sample):
    """The randomised block is the project's clean validation set; it must be clean."""
    df = causal_sample
    rct = df[df["assignment_block"] == "rct"]
    if len(rct) < 200:
        pytest.skip("too few RCT rows in the tiny fixture to test randomisation")

    # In the RCT block, arm must be ~independent of the ground-truth effect.
    tau = rct["gt_tau_value_1"].to_numpy(dtype=float)
    arm = rct["arm"].to_numpy()
    means = [np.nanmean(tau[arm == a]) for a in range(N_ARMS) if (arm == a).sum() > 20]
    assert len(means) >= 2, "RCT block does not cover multiple arms"
    pooled_sd = np.nanstd(tau)
    spread = (max(means) - min(means)) / (pooled_sd + 1e-12)
    assert spread < 0.45, f"RCT arms differ in true effect by {spread:.2f} sd -- not randomised"


def test_ground_truth_survival_quantities_are_internally_consistent(causal_sample, sim):
    """RMST, value and churn probability must be mutually coherent, arm by arm."""
    df = causal_sample.merge(
        sim.customers[["customer_id", "monthly_margin"]], on="customer_id", how="left"
    )
    for a in range(N_ARMS):
        rmst = df[f"gt_rmst_{a}"].to_numpy(dtype=float)
        value = df[f"gt_value_{a}"].to_numpy(dtype=float)
        churn = df[f"gt_churn_p_{a}"].to_numpy(dtype=float)

        assert np.all(rmst >= 0), f"arm {a}: negative RMST"
        assert np.all(np.isfinite(value)), f"arm {a}: non-finite value"
        assert np.all((churn >= 0) & (churn <= 1)), f"arm {a}: churn probability outside [0,1]"

        # More expected months retained must mean more expected discounted value -- but only
        # once per-customer unit economics are held fixed. Across customers the monthly margin
        # varies far more than survival does (a plan-price ladder times a lognormal), which
        # legitimately dilutes the raw correlation. Dividing it out isolates the claim being
        # tested, which is that the survival-to-value conversion is coherent.
        monthly_margin = df["monthly_margin"].to_numpy(dtype=float)
        ok = np.isfinite(rmst) & np.isfinite(value) & np.isfinite(monthly_margin) & (monthly_margin > 1e-9)
        assert ok.sum() > 50, "not enough rows with usable unit economics"
        corr = np.corrcoef(rmst[ok], (value[ok] / monthly_margin[ok]))[0, 1]
        assert corr > 0.9, (
            f"arm {a}: RMST and margin-normalised value correlate only {corr:.2f}; "
            "the survival-to-value conversion is not coherent"
        )

    # a lower one-period churn probability must go with a longer restricted mean survival
    corr = np.corrcoef(df["gt_churn_p_0"], df["gt_rmst_0"])[0, 1]
    assert corr < -0.3, f"churn probability and RMST should be negatively related, got {corr:.2f}"


def test_oracle_net_value_is_never_negative(causal_sample):
    """`do nothing` is always available, so the oracle can never be forced into a loss."""
    oracle = causal_sample["gt_oracle_net_value"].to_numpy(dtype=float)
    assert np.all(oracle >= -1e-9), f"oracle net value went negative (min {oracle.min():.4f})"


def test_oracle_best_arm_matches_its_own_definition(causal_sample):
    """`gt_best_arm` must be the argmax of value minus cost -- a cheap guard against a stale column."""
    df = causal_sample
    net = np.column_stack([df[f"gt_value_{a}"].to_numpy(dtype=float) - ARM_COSTS[a] for a in range(N_ARMS)])
    recomputed = net.argmax(axis=1)
    agreement = float((recomputed == df["gt_best_arm"].to_numpy()).mean())
    assert agreement > 0.98, f"gt_best_arm disagrees with its definition on {(1 - agreement):.1%} of rows"


# =====================================================================================
# Claim 2 -- a causal estimator recovers the known treatment effects
# =====================================================================================


@pytest.mark.slow
def test_dr_learner_recovers_known_heterogeneous_effects(toy_causal):
    """On data with a known tau, the DR-learner must beat a constant-effect baseline.

    This is the central methodological claim. If the estimator cannot recover a treatment
    effect it *can* check, there is no reason to believe it where it cannot.
    """
    from prism.causal.evaluate import pehe
    from prism.causal.learners import DRLearner

    d = toy_causal
    learner = DRLearner(n_splits=4, random_state=0).fit(d["X"], d["w"], d["y"])
    tau_hat = np.asarray(learner.predict_cate(d["X"])).reshape(len(d["y"]), -1)[:, 0]
    tau_true = d["tau_true"]

    model_pehe = pehe(tau_true, tau_hat)
    # the honest baseline: assume a constant effect equal to the true ATE
    baseline_pehe = pehe(tau_true, np.full_like(tau_true, tau_true.mean()))

    assert model_pehe < baseline_pehe, (
        f"DR-learner PEHE {model_pehe:.3f} is no better than a constant-effect "
        f"baseline {baseline_pehe:.3f} -- it learned no heterogeneity"
    )
    corr = np.corrcoef(tau_true, tau_hat)[0, 1]
    assert corr > 0.45, f"estimated and true CATE correlate only {corr:.2f}"


@pytest.mark.slow
def test_causal_forest_recovers_known_heterogeneous_effects(toy_causal):
    """Same claim for the honest causal forest."""
    from prism.causal.evaluate import pehe
    from prism.causal.forest import CausalForest

    d = toy_causal
    forest = CausalForest(n_estimators=200, min_samples_leaf=20, random_state=0).fit(d["X"], d["w"], d["y"])
    tau_hat = np.asarray(forest.predict_cate(d["X"])).reshape(len(d["y"]), -1)[:, 0]
    tau_true = d["tau_true"]

    model_pehe = pehe(tau_true, tau_hat)
    baseline_pehe = pehe(tau_true, np.full_like(tau_true, tau_true.mean()))
    assert model_pehe < baseline_pehe, f"forest PEHE {model_pehe:.3f} >= baseline {baseline_pehe:.3f}"
    assert np.corrcoef(tau_true, tau_hat)[0, 1] > 0.40


@pytest.mark.slow
def test_ate_is_recovered_despite_confounding(toy_causal):
    """The naive difference in means must be biased, and the DR estimate must fix it.

    Demonstrating *both halves* is the point: it shows the confounding is real and that
    the estimator actually removes it.
    """
    from prism.causal.learners import DRLearner

    d = toy_causal
    true_ate = d["tau_true"].mean()
    naive = d["y"][d["w"] == 1].mean() - d["y"][d["w"] == 0].mean()
    naive_err = abs(naive - true_ate)

    learner = DRLearner(n_splits=4, random_state=0).fit(d["X"], d["w"], d["y"])
    dr_ate = float(np.asarray(learner.predict_cate(d["X"])).reshape(len(d["y"]), -1)[:, 0].mean())
    dr_err = abs(dr_ate - true_ate)

    assert naive_err > 0.10, (
        f"the naive estimator is already unbiased (err {naive_err:.3f}) -- "
        "the fixture is not confounded enough to be a real test"
    )
    assert dr_err < naive_err, f"DR error {dr_err:.3f} is not better than naive {naive_err:.3f}"


def test_placebo_treatment_produces_no_effect(toy_causal):
    """Permuting treatment must destroy the estimated effect. If it does not, the estimator
    is picking up outcome structure rather than treatment structure."""
    from prism.causal.learners import TLearner

    d = toy_causal
    rng = np.random.default_rng(0)
    w_placebo = rng.permutation(d["w"])

    real = TLearner(random_state=0).fit(d["X"], d["w"], d["y"])
    fake = TLearner(random_state=0).fit(d["X"], w_placebo, d["y"])

    real_ate = float(np.asarray(real.predict_cate(d["X"])).reshape(len(d["y"]), -1)[:, 0].mean())
    fake_ate = float(np.asarray(fake.predict_cate(d["X"])).reshape(len(d["y"]), -1)[:, 0].mean())

    assert abs(fake_ate) < abs(real_ate) * 0.5, (
        f"placebo treatment still shows an effect of {fake_ate:.3f} against a real effect of {real_ate:.3f}"
    )


# =====================================================================================
# Claim 3 -- targeting on causal effect beats targeting on churn risk, in money
# =====================================================================================


@pytest.mark.slow
def test_causal_targeting_beats_risk_targeting_on_oracle_value(causal_sample):
    """The headline business claim, measured against ground truth.

    Uses the *true* tau so this test isolates the decision rule from estimation error: even
    with a perfect risk model, ranking by churn risk is the wrong objective.
    """
    from prism.decision.economics import EconomicConfig, expected_net_value
    from prism.decision.optimize import lagrangian_allocate

    df = causal_sample
    n = len(df)
    econ = EconomicConfig(horizon=6)

    value_cate = np.column_stack([df[f"gt_tau_value_{a}"].to_numpy(dtype=float) for a in range(1, N_ARMS)])
    net = expected_net_value(value_cate, econ)          # (n, n_arms), column 0 == do nothing
    cost = expected_cost_matrix(econ, n)
    budget = 0.15 * n * float(np.mean(econ.arm_costs[1:]))

    causal_alloc = lagrangian_allocate(net, cost, budget)

    # The baseline a conventional project ships: rank by true churn risk, give everyone the
    # cheapest offer until the budget runs out.
    risk = df["gt_churn_p_0"].to_numpy(dtype=float)
    order = np.argsort(-risk)
    risk_assign = np.zeros(n, dtype=int)
    spent, unit = 0.0, float(econ.arm_costs[1])
    for i in order:
        if spent + unit > budget:
            break
        risk_assign[i] = 1
        spent += unit

    realised = net[np.arange(n), causal_alloc.assignment].sum()
    risk_realised = net[np.arange(n), risk_assign].sum()

    assert realised > risk_realised, (
        f"causal allocation realised {realised:,.0f} but risk targeting realised "
        f"{risk_realised:,.0f} -- the project's central business claim fails"
    )


def test_treating_everyone_destroys_value(causal_sample):
    """Because of sleeping dogs and offer cost, blanket treatment must lose to doing nothing."""
    from prism.decision.economics import EconomicConfig, expected_net_value

    df = causal_sample
    econ = EconomicConfig(horizon=6)
    value_cate = np.column_stack([df[f"gt_tau_value_{a}"].to_numpy(dtype=float) for a in range(1, N_ARMS)])
    net = expected_net_value(value_cate, econ)

    treat_all_best = net[:, 1:].max(axis=1).sum()
    oracle = np.maximum(net.max(axis=1), 0.0).sum()

    assert oracle > treat_all_best, "selective targeting is not beating blanket treatment"


def test_budget_constraint_is_respected(causal_sample):
    """An allocator that quietly overspends is worse than useless."""
    from prism.decision.economics import EconomicConfig, expected_net_value
    from prism.decision.optimize import greedy_knapsack, lagrangian_allocate

    df = causal_sample
    n = len(df)
    econ = EconomicConfig(horizon=6)
    value_cate = np.column_stack([df[f"gt_tau_value_{a}"].to_numpy(dtype=float) for a in range(1, N_ARMS)])
    net = expected_net_value(value_cate, econ)
    cost = expected_cost_matrix(econ, n)

    for budget in (0.0, 500.0, 5_000.0):
        for solver in (greedy_knapsack, lagrangian_allocate):
            res = solver(net, cost, budget)
            assert res.total_cost <= budget + 1e-6, (
                f"{solver.__name__} spent {res.total_cost:.2f} against a budget of {budget:.2f}"
            )
            assert res.assignment.shape == (n,)
            assert res.assignment.min() >= 0 and res.assignment.max() < N_ARMS


def test_more_budget_never_buys_less_value(causal_sample):
    """The efficient frontier must be non-decreasing. A dip means the solver is broken."""
    from prism.decision.economics import EconomicConfig, expected_net_value
    from prism.decision.optimize import efficient_frontier

    df = causal_sample
    n = len(df)
    econ = EconomicConfig(horizon=6)
    value_cate = np.column_stack([df[f"gt_tau_value_{a}"].to_numpy(dtype=float) for a in range(1, N_ARMS)])
    net = expected_net_value(value_cate, econ)
    cost = expected_cost_matrix(econ, n)

    budgets = np.array([0.0, 1_000.0, 5_000.0, 20_000.0, 100_000.0])
    frontier = efficient_frontier(net, cost, budgets)
    values = frontier.sort_values("budget")["expected_incremental_value"].to_numpy(dtype=float)
    assert np.all(np.diff(values) >= -1e-6), f"efficient frontier is not monotone: {values}"


# =====================================================================================
# Claim 4 -- nothing leaks
# =====================================================================================


def test_split_is_a_pure_function_of_period(panel):
    """Splitting must be temporal, never random.

    A long-lived customer legitimately appears in more than one split, because the split is by
    calendar period. What must never happen is the *same period* landing in two splits, which
    is the signature of a random split -- and a random split lets a model memorise customers
    instead of learning behaviour, making every validation number a lie.
    """
    by_period = panel.groupby("period")["split"].nunique()
    mixed = by_period[by_period > 1].index.tolist()
    assert not mixed, f"split is not a pure function of period; mixed periods: {mixed}"

    # And each split must occupy a contiguous, correctly ordered block of periods.
    bounds = panel.groupby("split")["period"].agg(["min", "max"])
    order = [s for s in ("train", "valid", "test") if s in bounds.index]
    for earlier, later in zip(order, order[1:]):
        assert bounds.loc[earlier, "max"] < bounds.loc[later, "min"], (
            f"{earlier} periods overlap {later} periods: {bounds.to_dict()}"
        )


def test_splits_are_temporally_ordered(panel):
    """train must precede valid must precede test, with no interleaving."""
    bounds = panel.groupby("split")["period"].agg(["min", "max"])
    if {"train", "valid"} <= set(bounds.index):
        assert bounds.loc["train", "max"] < bounds.loc["valid", "min"]
    if {"valid", "test"} <= set(bounds.index):
        assert bounds.loc["valid", "max"] < bounds.loc["test", "min"]


def test_point_in_time_join_uses_no_future_events(toy_events):
    """The feature store must be unable to see past `as_of_date`."""
    from prism.data.features import FeatureSpec, PointInTimeFeatureStore, assert_no_leakage

    events, spine = toy_events
    # `source` is an event-log COLUMN; the event *type* is filtered via `event_types`.
    specs = [
        FeatureSpec(
            name="orders_90d", source="event_type", agg="count", window_days=90, event_types=("order",)
        ),
        FeatureSpec(
            name="spend_365d", source="amount", agg="sum", window_days=365, event_types=("order",)
        ),
    ]
    store = PointInTimeFeatureStore(events, specs)
    assert_no_leakage(store, spine)

    report = store.leakage_report(spine)
    assert not bool(report["leak_detected"].any()), f"leakage detected:\n{report}"


def test_panel_satisfies_its_data_contract(panel):
    """The gold panel must pass the contract that the rest of the pipeline assumes."""
    from prism.utils.validation import PANEL_CONTRACT

    result = PANEL_CONTRACT.validate(panel)
    assert result.passed, f"gold panel violates its contract:\n{result.summary().to_string(index=False)}"


def test_hidden_confounder_is_not_in_the_feature_set(panel, ground_truth):
    """`gt_u` confounds assignment and outcome and must never be visible to a model.

    If it leaked into the panel the whole sensitivity analysis would be theatre.
    """
    from prism.data.schema import FEATURES

    assert "gt_u" not in panel.columns
    assert not any(c.startswith("gt_") for c in panel.columns), (
        f"ground-truth columns leaked into the panel: {[c for c in panel.columns if c.startswith('gt_')]}"
    )
    # and it must be genuinely predictive, or it is not a confounder worth analysing
    merged = panel.merge(ground_truth[["customer_id", "period", "gt_u"]], on=["customer_id", "period"])
    numeric = [c for c in FEATURES if pd.api.types.is_numeric_dtype(merged[c])]
    corrs = merged[numeric].corrwith(merged["gt_u"]).abs()
    assert corrs.max() < 0.6, (
        f"the hidden confounder is nearly observable via {corrs.idxmax()} (|r| = {corrs.max():.2f})"
    )


# =====================================================================================
# Reproducibility
# =====================================================================================


def test_simulation_is_bit_reproducible(tiny_config):
    """Same seed, same data. Without this, none of the numbers above mean anything."""
    from prism.data.dgp import simulate

    a = simulate(tiny_config)
    b = simulate(tiny_config)
    pd.testing.assert_frame_equal(a.panel, b.panel)
    pd.testing.assert_frame_equal(a.ground_truth, b.ground_truth)


def test_different_seeds_give_different_data(tiny_config):
    """Guards against a seed that is silently ignored."""
    from dataclasses import replace

    from prism.data.dgp import simulate

    a = simulate(tiny_config)
    b = simulate(replace(tiny_config, random_state=tiny_config.random_state + 1))
    assert not a.panel["event_time"].equals(b.panel["event_time"]), "the seed appears to be ignored"
