"""Shared fixtures.

Simulation is the expensive part of every test, so the tiny panel is built once per session
and shared. Fixtures return copies where a test could mutate them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):  # noqa: D103
    config.addinivalue_line("markers", "slow: takes more than ~20 seconds")
    config.addinivalue_line("markers", "network: requires internet access")
    config.addinivalue_line("markers", "statistical: asserts a statistical property")


@pytest.fixture(scope="session")
def tiny_config():
    """A DGP config small enough for the whole suite to run in a couple of minutes."""
    from prism.data.dgp import DGPConfig

    return DGPConfig(
        n_customers=2500,
        n_periods=10,
        horizon=6,
        rct_fraction=0.25,
        random_state=7,
    )


@pytest.fixture(scope="session")
def sim(tiny_config):
    """A simulated dataset with known ground truth. Session-scoped: do not mutate."""
    from prism.data.dgp import simulate

    return simulate(tiny_config)


@pytest.fixture
def panel(sim):
    """A fresh copy of the gold panel."""
    return sim.panel.copy()


@pytest.fixture
def ground_truth(sim):
    """A fresh copy of the ground-truth frame."""
    return sim.ground_truth.copy()


@pytest.fixture(scope="session")
def design(sim):
    """(X, feature_names, encoder, aligned panel) for the simulated panel."""
    from prism.data.features import build_design_matrix

    X, names, enc = build_design_matrix(sim.panel, fit=True)
    return X, names, enc, sim.panel


@pytest.fixture(scope="session")
def causal_sample(sim):
    """A merged panel+ground-truth frame at one decision point per customer.

    This is the sample the causal tests estimate on: repeated monthly rows for the same
    customer are not independent, so a single decision point per customer is used.
    """
    rng = np.random.default_rng(0)
    panel = sim.panel
    idx = (
        panel.reset_index()
        .groupby("customer_id", sort=False)["index"]
        .apply(lambda s: s.iloc[rng.integers(0, len(s))])
        .to_numpy()
    )
    rows = panel.loc[idx].reset_index(drop=True)
    gt = sim.ground_truth.merge(rows[["customer_id", "period"]], on=["customer_id", "period"], how="inner")
    merged = rows.merge(gt, on=["customer_id", "period"], how="inner", validate="one_to_one")
    return merged


@pytest.fixture(scope="session")
def rng():
    """A deterministic generator for tests that need their own randomness."""
    return np.random.default_rng(1234)


@pytest.fixture
def toy_survival(rng):
    """Right-censored survival data with a known linear log-hazard.

    Returns
    -------
    dict
        ``X``, ``event_time``, ``event_observed``, ``beta_true``, ``horizon``.
    """
    n, p = 3000, 6
    X = rng.normal(size=(n, p))
    beta = np.array([0.8, -0.5, 0.3, 0.0, 0.0, 0.2])
    rate = np.exp(-2.0 + X @ beta)
    t_event = rng.exponential(1.0 / rate)
    t_censor = rng.uniform(0.5, 18.0, size=n)
    event_time = np.minimum(t_event, t_censor)
    event_observed = (t_event <= t_censor).astype(int)
    return {
        "X": X,
        "event_time": event_time,
        "event_observed": event_observed,
        "beta_true": beta,
        "horizon": 12,
    }


@pytest.fixture
def toy_causal(rng):
    """A confounded two-arm problem with a known, heterogeneous treatment effect.

    Returns
    -------
    dict
        ``X``, ``w``, ``y``, ``tau_true``, ``propensity_true``.
    """
    n, p = 4000, 8
    X = rng.normal(size=(n, p))
    # confounded assignment: treatment depends on X0 and X1
    logit_e = 0.6 * X[:, 0] - 0.4 * X[:, 1]
    e = 1.0 / (1.0 + np.exp(-logit_e))
    e = np.clip(e, 0.05, 0.95)
    w = (rng.uniform(size=n) < e).astype(int)
    # heterogeneous, interaction-driven effect
    tau = 1.5 + 1.0 * X[:, 0] * (X[:, 2] > 0) - 0.8 * np.maximum(X[:, 1], 0)
    baseline = 2.0 + X[:, 0] + 0.5 * X[:, 1] ** 2 - X[:, 3]
    y = baseline + w * tau + rng.normal(scale=1.0, size=n)
    return {"X": X, "w": w, "y": y, "tau_true": tau, "propensity_true": e}


@pytest.fixture
def toy_events(rng):
    """A small event log and matching spine for point-in-time feature tests."""
    n_cust, n_events = 120, 4000
    cust = np.array([f"C{i:04d}" for i in range(n_cust)])
    origin = pd.Timestamp("2022-01-01")
    events = pd.DataFrame(
        {
            "customer_id": rng.choice(cust, size=n_events),
            "event_ts": origin + pd.to_timedelta(rng.integers(0, 720, size=n_events), unit="D"),
            "event_type": rng.choice(["order", "session", "support_ticket"], size=n_events, p=[0.4, 0.5, 0.1]),
            "amount": np.round(rng.gamma(2.0, 30.0, size=n_events), 2),
            "category": rng.choice(["a", "b", "c", "d"], size=n_events),
            "n_items": rng.integers(1, 6, size=n_events),
        }
    )
    events.loc[events["event_type"] != "order", "amount"] = 0.0
    spine = pd.DataFrame(
        {
            "customer_id": np.repeat(cust, 5),
            "as_of_date": np.tile(
                [origin + pd.DateOffset(months=m) for m in (6, 9, 12, 15, 18)],
                n_cust,
            ),
        }
    )
    return events.sort_values("event_ts").reset_index(drop=True), spine
