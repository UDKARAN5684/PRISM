"""Canonical column names, arm definitions and dtypes for the PRISM gold panel.

Single source of truth. Every other module imports names from here rather than
hard-coding strings, so a schema change is one edit.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "N_ARMS",
    "ARM_NAMES",
    "ARM_COSTS",
    "ARM_REDEMPTION",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "FEATURES",
    "CATEGORICAL_LEVELS",
    "ID_COLUMNS",
    "TREATMENT_COLUMNS",
    "OUTCOME_COLUMNS",
    "PANEL_COLUMNS",
    "GT_COLUMNS",
    "PANEL_DTYPES",
    "arm_name",
    "enforce_schema",
]

# --------------------------------------------------------------------------------------
# Treatment arms
# --------------------------------------------------------------------------------------
N_ARMS: int = 4
ARM_NAMES: tuple[str, ...] = ("control", "discount_10", "discount_25", "concierge")
#: Marginal cost of delivering each offer to one customer, in currency units.
ARM_COSTS: tuple[float, ...] = (0.0, 12.0, 30.0, 55.0)
#: Fraction of targeted customers who actually redeem the offer (cost is incurred only then).
ARM_REDEMPTION: tuple[float, ...] = (0.0, 0.62, 0.71, 0.48)

# --------------------------------------------------------------------------------------
# Feature block (exactly 17 numeric + 6 categorical = 23)
# --------------------------------------------------------------------------------------
NUMERIC_FEATURES: tuple[str, ...] = (
    "tenure_months",
    "recency_days",
    "frequency_12m",
    "monetary_12m",
    "avg_order_value",
    "n_categories_12m",
    "sessions_30d",
    "days_since_last_session",
    "engagement_score",
    "support_tickets_90d",
    "nps",
    "payment_failures_12m",
    "discount_depth_hist",
    "price_change_pct",
    "competitor_promo_intensity",
    "seasonality_index",
    "basket_diversity",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "plan_tier",
    "channel",
    "region",
    "device",
    "is_autopay",
    "contract_type",
)

FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORICAL_LEVELS: dict[str, tuple[str, ...]] = {
    "plan_tier": ("basic", "plus", "pro", "enterprise"),
    "channel": ("organic", "paid", "referral", "partner"),
    "region": ("north", "south", "east", "west"),
    "device": ("ios", "android", "web"),
    "is_autopay": ("yes", "no"),
    "contract_type": ("monthly", "annual"),
}

# --------------------------------------------------------------------------------------
# Panel structure
# --------------------------------------------------------------------------------------
ID_COLUMNS: tuple[str, ...] = ("customer_id", "period", "as_of_date", "cohort", "split", "assignment_block")
TREATMENT_COLUMNS: tuple[str, ...] = ("arm", "treated", "offer_cost")
OUTCOME_COLUMNS: tuple[str, ...] = ("churn_next", "event_time", "event_observed", "rmst_h", "value_h", "margin_rate")

PANEL_COLUMNS: tuple[str, ...] = ID_COLUMNS + TREATMENT_COLUMNS + FEATURES + OUTCOME_COLUMNS

# NOTE ON STRING COLUMNS. These are declared "object" rather than pandas "string"
# (StringDtype) deliberately, and the declaration matches what enforce_schema actually
# produces -- a mismatch here would make every downstream dtype assertion a lie.
# Object-of-str is chosen because it round-trips cleanly through parquet and DuckDB, and
# because scikit-learn encoders, numpy string comparisons and `.map` over mixed/missing
# values all behave predictably with it. StringDtype would be tidier in isolation but
# introduces pd.NA into paths that expect np.nan.
PANEL_DTYPES: dict[str, str] = {
    "customer_id": "object",
    "period": "int64",
    "as_of_date": "datetime64[ns]",
    "cohort": "object",
    "split": "object",
    "assignment_block": "object",
    "arm": "int64",
    "treated": "int64",
    "offer_cost": "float64",
    "churn_next": "int64",
    "event_time": "float64",
    "event_observed": "int64",
    "rmst_h": "float64",
    "value_h": "float64",
    "margin_rate": "float64",
    **dict.fromkeys(NUMERIC_FEATURES, "float64"),
    **dict.fromkeys(CATEGORICAL_FEATURES, "object"),
}


def _gt_columns() -> tuple[str, ...]:
    cols = ["customer_id", "period", "gt_segment", "gt_u"]
    for a in range(N_ARMS):
        cols += [f"gt_propensity_{a}", f"gt_churn_p_{a}", f"gt_rmst_{a}", f"gt_value_{a}"]
    for a in range(1, N_ARMS):
        cols += [f"gt_tau_churn_{a}", f"gt_tau_rmst_{a}", f"gt_tau_value_{a}"]
    cols += ["gt_best_arm", "gt_oracle_net_value"]
    return tuple(cols)


GT_COLUMNS: tuple[str, ...] = _gt_columns()

#: The four latent responder segments that make naive risk-targeting lose money.
SEGMENTS: tuple[str, ...] = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")
SEGMENT_SHARES: tuple[float, ...] = (0.22, 0.34, 0.26, 0.18)


def arm_name(a: int) -> str:
    """Return the human-readable name of arm index ``a``."""
    return ARM_NAMES[int(a)]


def enforce_schema(df: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Coerce a panel-like frame to the canonical dtypes in :data:`PANEL_DTYPES`.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to coerce. Columns absent from the frame are skipped.
    strict : bool, default False
        If True, raise when a column in :data:`PANEL_COLUMNS` is missing.

    Returns
    -------
    pandas.DataFrame
        A copy with canonical dtypes; categorical columns are lower-cased and stripped.
    """
    out = df.copy()
    if strict:
        missing = [c for c in PANEL_COLUMNS if c not in out.columns]
        if missing:
            raise KeyError(f"panel is missing required columns: {missing}")
    for col, dtype in PANEL_DTYPES.items():
        if col not in out.columns:
            continue
        if dtype == "datetime64[ns]":
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif dtype == "object":
            out[col] = out[col].astype("object").where(out[col].notna(), None)
            if col in CATEGORICAL_FEATURES:
                out[col] = out[col].map(lambda v: v.strip().lower() if isinstance(v, str) else v)
        elif dtype == "int64":
            num = pd.to_numeric(out[col], errors="coerce").round().astype("Int64")
            # Real feeds legitimately leave arm / treated / churn_next / event_observed
            # missing. Casting straight to int64 raises on pd.NA, so keep the nullable
            # Int64 dtype in that case rather than crashing or silently inventing a value.
            out[col] = num.astype("int64") if not num.isna().any() else num
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


if __name__ == "__main__":  # pragma: no cover - smoke test
    assert len(NUMERIC_FEATURES) == 17, len(NUMERIC_FEATURES)
    assert len(CATEGORICAL_FEATURES) == 6
    assert len(FEATURES) == 23
    assert len(set(FEATURES)) == 23
    assert len(ARM_NAMES) == len(ARM_COSTS) == len(ARM_REDEMPTION) == N_ARMS
    assert abs(sum(SEGMENT_SHARES) - 1.0) < 1e-9
    demo = pd.DataFrame({"customer_id": [1], "period": ["0"], "plan_tier": ["  Basic "], "nps": ["7"]})
    coerced = enforce_schema(demo)
    print(coerced.dtypes.to_dict())
    print("panel columns:", len(PANEL_COLUMNS), "| gt columns:", len(GT_COLUMNS))
    print("schema.py OK")
