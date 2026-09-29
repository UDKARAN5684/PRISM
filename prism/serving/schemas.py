"""Pydantic v2 request/response contracts for the scoring service.

The API surface is deliberately narrow and explicit: a caller sends the same 23 features the
model was trained on, and gets back, per customer, the estimated causal effect of every offer
in both months and currency, the cost, the net value, and a recommendation.

The feature list is generated from :mod:`prism.data.schema`, so the serving contract cannot
drift away from the training contract -- if a feature is added to the panel and not here, the
conformance test in ``tests/test_contracts.py`` fails.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prism.data.schema import (
    ARM_NAMES,
    CATEGORICAL_LEVELS,
    N_ARMS,
    NUMERIC_FEATURES,
)

__all__ = [
    "CustomerFeatures",
    "ScoreRequest",
    "ArmScore",
    "CustomerScore",
    "ScoreResponse",
    "PolicyRequest",
    "PolicyResponse",
    "PolicyArmSummary",
    "ExplainRequest",
    "ExplainResponse",
    "FeatureContribution",
    "HealthResponse",
    "MetadataResponse",
]

PlanTier = Literal["basic", "plus", "pro", "enterprise"]
Channel = Literal["organic", "paid", "referral", "partner"]
Region = Literal["north", "south", "east", "west"]
Device = Literal["ios", "android", "web"]
Autopay = Literal["yes", "no"]
ContractType = Literal["monthly", "annual"]

_EXAMPLE_CUSTOMER: dict[str, Any] = {
    "customer_id": "C0001234",
    "tenure_months": 18.0,
    "recency_days": 12.0,
    "frequency_12m": 9.0,
    "monetary_12m": 540.0,
    "avg_order_value": 60.0,
    "n_categories_12m": 4.0,
    "sessions_30d": 11.0,
    "days_since_last_session": 3.0,
    "engagement_score": 0.61,
    "support_tickets_90d": 1.0,
    "nps": 7.0,
    "payment_failures_12m": 0.0,
    "discount_depth_hist": 0.08,
    "price_change_pct": 0.05,
    "competitor_promo_intensity": 0.42,
    "seasonality_index": 1.03,
    "basket_diversity": 0.55,
    "plan_tier": "plus",
    "channel": "paid",
    "region": "north",
    "device": "ios",
    "is_autopay": "yes",
    "contract_type": "monthly",
}


class CustomerFeatures(BaseModel):
    """One customer at one decision point.

    Every numeric feature is optional: missing values are legitimate at serving time and are
    handled by the same median-impute-plus-indicator encoder used in training. Sending
    ``null`` is therefore correct and preferred over inventing a value.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": _EXAMPLE_CUSTOMER})

    customer_id: str = Field(..., min_length=1, max_length=64, description="Stable customer identifier.")

    # --- numeric block (17) --------------------------------------------------------------
    tenure_months: float | None = Field(None, ge=0, le=1200, description="Months since acquisition.")
    recency_days: float | None = Field(None, ge=0, le=5000, description="Days since the last order.")
    frequency_12m: float | None = Field(None, ge=0, le=5000, description="Orders in the last 12 months.")
    monetary_12m: float | None = Field(None, ge=0, description="Revenue in the last 12 months.")
    avg_order_value: float | None = Field(None, ge=0, description="Mean order value.")
    n_categories_12m: float | None = Field(None, ge=0, le=1000, description="Distinct categories bought.")
    sessions_30d: float | None = Field(None, ge=0, le=10000, description="Sessions in the last 30 days.")
    days_since_last_session: float | None = Field(None, ge=0, le=5000)
    engagement_score: float | None = Field(None, ge=0, le=1, description="Composite engagement index in [0, 1].")
    support_tickets_90d: float | None = Field(None, ge=0, le=1000)
    nps: float | None = Field(None, ge=-100, le=100, description="Net promoter score.")
    payment_failures_12m: float | None = Field(None, ge=0, le=1000)
    discount_depth_hist: float | None = Field(None, ge=0, le=1, description="Historical mean discount fraction.")
    price_change_pct: float | None = Field(None, ge=-1, le=10, description="Recent price change, as a fraction.")
    competitor_promo_intensity: float | None = Field(None, ge=0, le=1)
    seasonality_index: float | None = Field(None, ge=0, le=10)
    basket_diversity: float | None = Field(None, ge=0, le=1)

    # --- categorical block (6) -----------------------------------------------------------
    plan_tier: PlanTier | None = None
    channel: Channel | None = None
    region: Region | None = None
    device: Device | None = None
    is_autopay: Autopay | None = None
    contract_type: ContractType | None = None

    @field_validator(
        "plan_tier", "channel", "region", "device", "is_autopay", "contract_type", mode="before"
    )
    @classmethod
    def _normalise_categorical(cls, v: Any) -> Any:
        """Accept ``"  Basic "`` and ``"IOS"`` -- real callers send dirty strings."""
        if isinstance(v, str):
            cleaned = v.strip().lower()
            return cleaned or None
        if isinstance(v, bool):
            return "yes" if v else "no"
        return v

    def to_row(self) -> dict[str, Any]:
        """Return a plain dict suitable for a one-row :class:`pandas.DataFrame`."""
        return self.model_dump()


class ScoreRequest(BaseModel):
    """A batch of customers to score."""

    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=10_000)
    budget: float | None = Field(
        None,
        ge=0,
        description="Optional. When supplied, the response also carries a budget-constrained recommendation.",
    )
    include_survival_curves: bool = Field(False, description="Return the per-arm survival curves as well.")

    @model_validator(mode="after")
    def _unique_ids(self) -> ScoreRequest:
        ids = [c.customer_id for c in self.customers]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})[:5]
            raise ValueError(f"customer_id must be unique within a request; duplicates: {dupes}")
        return self


class ArmScore(BaseModel):
    """The estimated consequence of giving one specific offer to one customer."""

    arm: Annotated[int, Field(ge=0, lt=N_ARMS)]
    arm_name: str
    cate_rmst: float | None = Field(
        None,
        description=(
            "Causal effect on restricted mean survival time, in months. NULL when the "
            "active estimator does not produce one (only CausalSurvivalUplift does) -- "
            "null means not computed, which is not the same as an effect of zero."
        ),
    )
    cate_value: float = Field(..., description="Causal effect on discounted lifetime value, in currency.")
    cate_churn: float | None = Field(None, description="Causal effect on one-period churn probability.")
    cost: float = Field(..., ge=0, description="Expected cost, already adjusted for the redemption rate.")
    net_value: float = Field(..., description="cate_value minus expected cost. Arm 0 is pinned to 0.0.")
    ci_low: float | None = None
    ci_high: float | None = None

    @field_validator("arm_name")
    @classmethod
    def _known_arm(cls, v: str) -> str:
        if v not in ARM_NAMES:
            raise ValueError(f"unknown arm_name {v!r}; expected one of {ARM_NAMES}")
        return v


class CustomerScore(BaseModel):
    """The full decision record for one customer."""

    customer_id: str
    churn_risk: float = Field(..., ge=0, le=1, description="Probability of churning in the next period.")
    clv: float = Field(..., description="Expected discounted lifetime value under no intervention.")
    rmst: float = Field(..., ge=0, description="Expected retained months under no intervention.")
    arms: list[ArmScore]
    recommended_arm: Annotated[int, Field(ge=0, lt=N_ARMS)]
    recommended_arm_name: str
    expected_net_value: float = Field(..., description="Net value of the recommendation; never negative.")
    segment_guess: str | None = Field(
        None,
        description="Heuristic read-out (persuadable / sure_thing / lost_cause / sleeping_dog), not ground truth.",
    )
    survival_curves: dict[str, list[float]] | None = None

    @model_validator(mode="after")
    def _recommendation_is_consistent(self) -> CustomerScore:
        by_arm = {a.arm: a for a in self.arms}
        if self.recommended_arm not in by_arm:
            raise ValueError(f"recommended_arm {self.recommended_arm} is not present in arms")
        if self.expected_net_value < -1e-9:
            raise ValueError("expected_net_value must be non-negative: doing nothing is always available")
        return self


class ScoreResponse(BaseModel):
    """Response to ``POST /score``."""

    scores: list[CustomerScore]
    model_version: str
    n_scored: int
    latency_ms: float
    warnings: list[str] = Field(default_factory=list)


class PolicyRequest(BaseModel):
    """A batch plus a budget: who should actually receive an offer?"""

    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=50_000)
    budget: float = Field(..., ge=0, description="Total spend available across the whole batch.")
    method: Literal["lagrangian", "greedy", "lp"] = "lagrangian"
    fairness_attribute: str | None = Field(
        None, description="Optional feature name to audit the resulting allocation against."
    )
    fairness_tolerance: float = Field(0.1, ge=0, le=1)


class PolicyArmSummary(BaseModel):
    """Per-arm rollup of an allocation."""

    arm: int
    arm_name: str
    n_assigned: int
    total_cost: float
    total_net_value: float
    mean_net_value: float


class PolicyResponse(BaseModel):
    """Response to ``POST /policy``."""

    assignments: dict[str, int] = Field(..., description="customer_id -> assigned arm index.")
    n_customers: int
    n_treated: int
    total_cost: float
    budget: float
    budget_utilisation: float = Field(..., ge=0, description="total_cost / budget, or 0 when the budget is 0.")
    expected_incremental_value: float
    roi: float | None = Field(None, description="expected_incremental_value / total_cost.")
    shadow_price: float | None = Field(
        None,
        description="Lagrangian lambda*: the marginal return on the next unit of budget. "
        "An arm is worth giving only if its net value exceeds lambda* times its cost.",
    )
    per_arm: list[PolicyArmSummary]
    method: str
    fairness: list[dict[str, Any]] | None = None
    model_version: str
    latency_ms: float


class FeatureContribution(BaseModel):
    """One feature's contribution to one customer's recommendation."""

    feature: str
    value: float | str | None
    contribution: float


class ExplainRequest(BaseModel):
    """A batch to explain. Kept small: explanations are expensive."""

    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=100)
    arm: Annotated[int, Field(ge=1, lt=N_ARMS)] = Field(
        1, description="Which offer's effect to explain. Arm 0 has no effect to explain."
    )
    top_k: Annotated[int, Field(ge=1, le=23)] = 8


class ExplainResponse(BaseModel):
    """Response to ``POST /explain``."""

    explanations: dict[str, list[FeatureContribution]]
    arm: int
    arm_name: str
    baseline: float = Field(..., description="Mean predicted effect, i.e. the SHAP base value.")
    method: str = Field(..., description="'shap' when SHAP is installed, otherwise a documented fallback.")
    model_version: str


class HealthResponse(BaseModel):
    """Response to ``GET /health``."""

    status: Literal["ok", "degraded"]
    model_version: str | None
    loaded_at: str | None
    bundle_path: str | None
    detail: str | None = None


class MetadataResponse(BaseModel):
    """Response to ``GET /metadata`` -- enough for a caller to build a valid request."""

    model_version: str | None
    trained_at: str | None
    training_window: dict[str, Any] | None
    n_arms: int = N_ARMS
    arm_names: list[str] = Field(default_factory=lambda: list(ARM_NAMES))
    arm_costs: list[float]
    numeric_features: list[str] = Field(default_factory=lambda: list(NUMERIC_FEATURES))
    categorical_features: dict[str, list[str]] = Field(
        default_factory=lambda: {k: list(v) for k, v in CATEGORICAL_LEVELS.items()}
    )
    estimator: str | None = None
    headline_metrics: dict[str, Any] | None = None
    dependencies: dict[str, bool] | None = None


if __name__ == "__main__":  # pragma: no cover - smoke test
    from prism.data.schema import FEATURES

    fields = set(CustomerFeatures.model_fields)
    missing = set(FEATURES) - fields
    assert not missing, f"schema is missing trained features: {sorted(missing)}"
    extra = fields - set(FEATURES) - {"customer_id"}
    assert not extra, f"schema has features the model was not trained on: {sorted(extra)}"

    # dirty input must normalise rather than 422
    c = CustomerFeatures(customer_id="C1", plan_tier="  Basic ", device="IOS", is_autopay=True, nps=7)
    assert (c.plan_tier, c.device, c.is_autopay) == ("basic", "ios", "yes")

    # missing numerics are legitimate
    sparse = CustomerFeatures(customer_id="C2")
    assert sparse.nps is None

    req = ScoreRequest(customers=[CustomerFeatures(**_EXAMPLE_CUSTOMER)], budget=1000.0)
    assert req.customers[0].tenure_months == 18.0

    try:
        ScoreRequest(customers=[CustomerFeatures(customer_id="X"), CustomerFeatures(customer_id="X")])
        raise AssertionError("duplicate customer_id should have been rejected")
    except ValueError:
        pass

    arms = [
        ArmScore(arm=0, arm_name="control", cate_rmst=0.0, cate_value=0.0, cost=0.0, net_value=0.0),
        ArmScore(arm=1, arm_name="discount_10", cate_rmst=0.4, cate_value=21.0, cost=7.44, net_value=13.56),
    ]
    score = CustomerScore(
        customer_id="C1", churn_risk=0.07, clv=410.0, rmst=9.2, arms=arms,
        recommended_arm=1, recommended_arm_name="discount_10", expected_net_value=13.56,
        segment_guess="persuadable",
    )
    assert score.recommended_arm == 1

    try:
        CustomerScore(
            customer_id="C2", churn_risk=0.1, clv=1.0, rmst=1.0, arms=arms,
            recommended_arm=1, recommended_arm_name="discount_10", expected_net_value=-5.0,
        )
        raise AssertionError("a negative expected_net_value should have been rejected")
    except ValueError:
        pass

    print(f"features covered : {len(FEATURES)}/{len(FEATURES)}")
    print(f"models defined   : {len(__all__)}")
    print("openapi title    :", ScoreResponse.model_json_schema()["title"])
    print("schemas.py OK")
