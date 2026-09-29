# PRISM — Interface Contract (authoritative)

**PRISM** = *Prescriptive Retention Intelligence & Spend Management.*

Thesis: churn *prediction* is the wrong objective. PRISM estimates the **causal effect of a
retention offer on a customer's expected discounted remaining lifetime value**, then allocates a
fixed budget across competing offers to maximise **incremental** profit — and proves the estimator
is correct against a simulator with known ground-truth individual treatment effects.

This file is the **authoritative contract**. Every module MUST match the signatures here exactly.
Do not rename, do not reorder required args, do not change return types. Add optional kwargs freely.

---

## 0. Global rules

- Python 3.11, Windows-compatible (no POSIX-only calls, no `fcntl`, no `/tmp` hardcoding, use `pathlib`).
- Hard dependencies: `numpy pandas scipy scikit-learn statsmodels matplotlib pyarrow`.
- Soft dependencies (import inside a `try/except ImportError` and degrade gracefully, never crash at
  import time): `lightgbm xgboost torch duckdb shap optuna lifelines fastapi streamlit plotly typer rich yaml mlflow`.
  Use `prism.utils.optional.require(name)` / `prism.utils.optional.HAS_LIGHTGBM` style flags.
- **Determinism**: every stochastic function takes `random_state: int | np.random.Generator | None`.
  Use `prism.utils.seeds.as_rng(random_state)` to normalise. Same seed => bit-identical output.
- **No leakage, ever.** Any feature at period `t` may only use information with
  `as_of_date <= panel.as_of_date[t]`. Splits are **temporal**, never random, unless the function
  name says `random`.
- Type hints on every public function. NumPy-style docstring with `Parameters/Returns` on every
  public function and class.
- Public arrays are `np.ndarray` of dtype float64 (or int64 for labels); tabular data is `pd.DataFrame`.
- Every module ends with a `if __name__ == "__main__":` smoke-test block that runs in < 60 s and
  prints a short result table. This is how the module is verified.
- No wall-clock calls for logic; simulated dates derive from `DGPConfig.origin`.

---

## 1. `prism.utils`

### `prism/utils/seeds.py`
```python
def as_rng(random_state: int | np.random.Generator | None = None) -> np.random.Generator
def seed_everything(seed: int) -> None          # seeds python, numpy, torch (if present)
```

### `prism/utils/optional.py`
```python
HAS_LIGHTGBM: bool; HAS_XGBOOST: bool; HAS_TORCH: bool; HAS_DUCKDB: bool
HAS_SHAP: bool; HAS_LIFELINES: bool; HAS_MLFLOW: bool; HAS_PLOTLY: bool
def require(pkg: str) -> ModuleType             # raises ImportError with an install hint
def best_gbm(task: str = "regression", **kw)    # -> sklearn-compatible estimator:
                                                #    LGBM > XGB > HistGradientBoosting fallback
```

### `prism/utils/logging.py`
```python
def get_logger(name: str) -> logging.Logger     # rich handler if available, else stdlib
```

### `prism/utils/validation.py` — hand-rolled data contracts (no great_expectations dep)
```python
@dataclass
class Expectation:
    column: str
    kind: str              # "not_null"|"dtype"|"in_range"|"in_set"|"unique"|"monotonic"|"regex"|"row_count"
    params: dict
    severity: str = "error"   # "error" | "warn"

@dataclass
class ValidationResult:
    passed: bool
    failures: list[dict]      # {column, kind, severity, detail, n_bad, frac_bad}
    n_rows: int
    def summary(self) -> pd.DataFrame
    def raise_for_status(self) -> None          # raises DataContractError if any severity=="error" failed

class DataContractError(Exception): ...

class DataContract:
    def __init__(self, name: str, expectations: list[Expectation]) -> None
    def validate(self, df: pd.DataFrame) -> ValidationResult
    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataContract"

PANEL_CONTRACT: DataContract      # the gold-panel contract, see section 2.2
```

---

## 2. `prism.data`

### 2.1 `prism/data/dgp.py` — the ground-truth simulator

Multi-arm, confounded, heterogeneous, time-to-event. This is the scientific core: it produces
**known** individual treatment effects so causal estimators can be scored with PEHE.

```python
N_ARMS = 4
ARM_NAMES = ("control", "discount_10", "discount_25", "concierge")
ARM_COSTS = (0.0, 12.0, 30.0, 55.0)          # marginal cost in currency units, per treated customer

@dataclass
class DGPConfig:
    n_customers: int = 60_000
    n_periods: int = 24                       # monthly periods
    horizon: int = 12                         # months used for RMST / value outcomes
    origin: str = "2022-01-01"
    monthly_discount_rate: float = 0.01       # for discounted lifetime value
    rct_fraction: float = 0.15                # customers randomised (ground-truth A/B holdout)
    confounding_strength: float = 1.0         # 0 = randomised everywhere, 1 = strong selection
    hidden_confounder_strength: float = 0.05  # calibrated; see ADR-014. NOT in the feature set
    drift_strength: float = 0.6               # covariate + concept drift over periods
    seasonality_amplitude: float = 0.25
    missing_rate: float = 0.08
    mnar_strength: float = 0.5
    outlier_rate: float = 0.004
    duplicate_rate: float = 0.002
    random_state: int = 7

@dataclass
class SimulatedData:
    panel: pd.DataFrame            # the clean gold panel, see 2.2
    customers: pd.DataFrame        # one row per customer: static attrs + latent segment
    events: pd.DataFrame           # raw transaction/session event log (bronze source)
    ground_truth: pd.DataFrame     # one row per (customer_id, period): gt_* columns, see 2.3
    config: DGPConfig
    def save(self, dirpath: str | Path) -> None       # parquet
    @classmethod
    def load(cls, dirpath: str | Path) -> "SimulatedData"

def simulate(config: DGPConfig | None = None) -> SimulatedData
```

**Required DGP properties** (asserted by tests):
1. **Four latent responder segments** with economically distinct behaviour:
   `persuadable` (offer prevents churn), `sure_thing` (stays anyway), `lost_cause` (leaves anyway),
   `sleeping_dog` (offer *increases* churn — it reminds them of the price). Segment shares ~
   (0.22, 0.34, 0.26, 0.18). Sleeping dogs are what makes naive risk-targeting lose money.
2. **Confounded assignment**: in the observational block, `P(arm | X)` depends on churn risk,
   value, tenure and the **hidden confounder** `U`. Overlap must hold (all propensities in
   [0.02, 0.85]) so estimation is identified but non-trivial.
3. **RCT block**: `rct_fraction` of customers get uniformly random arms — a clean validation set.
4. **Heterogeneous, non-linear, interaction-heavy** treatment effects: `tau` depends on
   interactions (e.g. tenure x engagement, price_change x plan_tier), not a linear score.
5. **Time-to-event**: churn is generated from a discrete-time hazard with customer frailty,
   seasonality and drift; treatment multiplies the hazard by `exp(theta_i)` where `theta_i` is
   heterogeneous and *positive for sleeping dogs*.
6. **Ground truth**: for each arm `a` and customer-period, the counterfactual churn probability,
   RMST over `horizon`, and discounted lifetime value are computed **analytically from the hazard
   path** (not by resampling), so `gt_tau_*` are exact.
7. **Drift**: covariate means and the outcome model shift across periods (so monitoring has signal).

### 2.2 Gold panel schema (`PANEL_CONTRACT`)

One row per `(customer_id, period)` while the customer is active.

| column | dtype | notes |
|---|---|---|
| `customer_id` | str | non-null |
| `period` | int64 | 0..n_periods-1 |
| `as_of_date` | datetime64[ns] | decision timestamp |
| `cohort` | str | acquisition cohort `YYYY-MM` |
| `split` | str | `train` / `valid` / `test` (temporal) |
| `assignment_block` | str | `observational` or `rct` |
| `arm` | int64 | 0..3, the offer actually given |
| `treated` | int64 | `(arm > 0)` |
| `offer_cost` | float64 | `ARM_COSTS[arm]` |
| **features (23)** | | see below, all point-in-time safe |
| `churn_next` | int64 | churned within the next period |
| `event_time` | float64 | months observed until churn or censoring |
| `event_observed` | int64 | 1 = churn observed, 0 = right-censored |
| `rmst_h` | float64 | realised restricted mean survival time over `horizon` |
| `value_h` | float64 | realised discounted revenue over `horizon` |
| `margin_rate` | float64 | gross margin fraction applied to revenue |

**Feature block (exactly these 23 names)** — 17 numeric, 6 categorical:

numeric: `tenure_months`, `recency_days`, `frequency_12m`, `monetary_12m`, `avg_order_value`,
`n_categories_12m`, `sessions_30d`, `days_since_last_session`, `engagement_score`,
`support_tickets_90d`, `nps`, `payment_failures_12m`, `discount_depth_hist`, `price_change_pct`,
`competitor_promo_intensity`, `seasonality_index`, `basket_diversity`

categorical: `plan_tier` (`basic|plus|pro|enterprise`), `channel` (`organic|paid|referral|partner`),
`region` (`north|south|east|west`), `device` (`ios|android|web`), `is_autopay` (`yes|no`),
`contract_type` (`monthly|annual`)

```python
NUMERIC_FEATURES: tuple[str, ...]      # 17 names above, in order
CATEGORICAL_FEATURES: tuple[str, ...]  # 6 names above, in order
FEATURES: tuple[str, ...]              # NUMERIC + CATEGORICAL (23)
```

### 2.3 Ground-truth frame columns
`customer_id`, `period`, `gt_segment` (str), `gt_u` (hidden confounder),
`gt_propensity_0..3`, `gt_churn_p_0..3`, `gt_rmst_0..3`, `gt_value_0..3`,
and the derived contrasts `gt_tau_churn_1..3`, `gt_tau_rmst_1..3`, `gt_tau_value_1..3`
(arm `a` minus control), plus `gt_best_arm` (argmax over arms of `gt_value_a - ARM_COSTS[a]`,
where control has net 0) and `gt_oracle_net_value` (that max, floored at 0 for "do nothing").

### 2.4 `prism/data/messy.py`
```python
def inject_realism(panel, config: DGPConfig, random_state=None) -> pd.DataFrame
    # MAR + MNAR missingness, heavy-tail outliers, duplicated rows, mixed-case categoricals,
    # stray whitespace, a late-arriving-record flag, and a schema wobble (one column renamed
    # in the last 3 periods) that the cleaning layer must repair.
def clean_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]
    # returns (clean_panel, cleaning_report) — report columns: step, column, n_affected, action
```

### 2.5 `prism/data/warehouse.py` — DuckDB medallion (parquet fallback if duckdb absent)
```python
class Warehouse:
    def __init__(self, path: str | Path = "artifacts/warehouse.duckdb") -> None
    def write(self, layer: str, name: str, df: pd.DataFrame) -> None   # layer in {bronze,silver,gold}
    def read(self, layer: str, name: str) -> pd.DataFrame
    def sql(self, query: str) -> pd.DataFrame
    def tables(self) -> pd.DataFrame
    def lineage(self) -> pd.DataFrame     # table, layer, n_rows, n_cols, written_at, sha256
    def close(self) -> None
```

### 2.6 `prism/data/features.py` — point-in-time feature store
```python
@dataclass
class FeatureSpec:
    name: str
    source: str            # event-log column or "panel"
    agg: str               # "sum"|"mean"|"count"|"max"|"nunique"|"last"|"days_since"
    window_days: int | None
    entity: str = "customer_id"

class PointInTimeFeatureStore:
    def __init__(self, events: pd.DataFrame, specs: list[FeatureSpec]) -> None
    def as_of_join(self, spine: pd.DataFrame, ts_col: str = "as_of_date") -> pd.DataFrame
        # strictly backward-looking; asserts no future rows leak in
    def leakage_report(self, spine, ts_col="as_of_date") -> pd.DataFrame

def temporal_split(panel, train_end: int, valid_end: int) -> pd.DataFrame   # writes `split`
def build_design_matrix(panel, features=None, *, fit: bool = True,
                        encoder=None) -> tuple[np.ndarray, list[str], object]
    # one-hot for categoricals, median-impute + missing-indicator for numerics.
    # Returns (X, feature_names, fitted_encoder). `fit=False` requires `encoder`.
```

### 2.7 `prism/data/real.py`
```python
def load_real(name: str, cache_dir="artifacts/data/real") -> pd.DataFrame | None
    # name in {"telco_churn","online_retail_ii","hillstrom"}; downloads with timeout,
    # returns None (never raises) on any network failure so the pipeline still runs offline.
def map_to_panel(df: pd.DataFrame, name: str) -> pd.DataFrame   # best-effort adapter to section 2.2
```

---

## 3. `prism.models`

All estimators are **sklearn-compatible**: `fit(X, y, **kw) -> self`, `predict(X) -> np.ndarray`,
`get_params/set_params`. Accept `X: np.ndarray`, `y: np.ndarray`.

### 3.1 `prism/models/survival.py`
```python
class DiscreteTimeHazardModel:
    """Pooled logistic discrete-time hazard: h(t|x) = sigmoid(f(x) + alpha_t).
    Torch MLP backend when available, else sklearn LogisticRegression on the person-period expansion."""
    def __init__(self, horizon: int = 12, hidden: tuple[int, ...] = (64, 32),
                 epochs: int = 40, lr: float = 1e-3, batch_size: int = 1024,
                 l2: float = 1e-4, backend: str = "auto", random_state: int | None = None)
    def fit(self, X, event_time, event_observed, sample_weight=None) -> "DiscreteTimeHazardModel"
    def predict_hazard(self, X) -> np.ndarray          # (n, horizon)
    def predict_survival(self, X) -> np.ndarray        # (n, horizon), cumulative product
    def predict_rmst(self, X, horizon=None) -> np.ndarray          # (n,) sum of survival
    def predict_churn_prob(self, X, within: int = 1) -> np.ndarray # (n,)
    def predict(self, X) -> np.ndarray                 # == predict_rmst

class CoxPHModel:            # Efron-tie partial likelihood, own gradient solver (no lifelines dep)
    def __init__(self, l2: float = 1e-3, max_iter: int = 200, tol: float = 1e-7)
    def fit(self, X, event_time, event_observed) -> "CoxPHModel"
    def predict_risk(self, X) -> np.ndarray            # exp(x @ beta)
    def predict_survival(self, X, times: np.ndarray) -> np.ndarray   # Breslow baseline
    def predict_rmst(self, X, horizon: float) -> np.ndarray
    coef_: np.ndarray; baseline_cumhaz_: pd.DataFrame

class RandomSurvivalForestLite:      # log-rank split criterion, bootstrap, Nelson-Aalen leaves
    def __init__(self, n_estimators=200, max_depth=6, min_samples_leaf=20,
                 max_features="sqrt", n_jobs=-1, random_state=None)
    def fit(self, X, event_time, event_observed) -> "RandomSurvivalForestLite"
    def predict_survival(self, X, times) -> np.ndarray
    def predict_rmst(self, X, horizon) -> np.ndarray

# metrics
def concordance_index(event_time, predicted_risk, event_observed) -> float
def time_dependent_auc(event_time, event_observed, risk, times) -> pd.DataFrame
def brier_score(event_time, event_observed, surv_prob, times) -> pd.DataFrame   # IPCW-weighted
def integrated_brier_score(event_time, event_observed, surv_prob, times) -> float
def calibration_curve_survival(event_time, event_observed, surv_prob, t: float, n_bins=10) -> pd.DataFrame
```

### 3.2 `prism/models/clv.py`
```python
class BGNBD:      # Beta-Geometric / NBD, scipy-optimised MLE
    def __init__(self, penalizer: float = 1e-4)
    def fit(self, frequency, recency, T) -> "BGNBD"
    def conditional_expected_transactions(self, t, frequency, recency, T) -> np.ndarray
    def probability_alive(self, frequency, recency, T) -> np.ndarray
    params_: dict     # r, alpha, a, b

class GammaGamma:
    def __init__(self, penalizer: float = 1e-4)
    def fit(self, frequency, monetary_value) -> "GammaGamma"
    def conditional_expected_average_profit(self, frequency, monetary_value) -> np.ndarray
    params_: dict     # p, q, v

def probabilistic_clv(summary: pd.DataFrame, horizon_months: int,
                      discount_rate: float = 0.01, margin: float = 0.3) -> pd.DataFrame
    # summary must have: customer_id, frequency, recency, T, monetary_value

class DeepCLV:     # torch two-head net (p_alive, expected value) with a survival-weighted loss;
                   # falls back to a two-stage sklearn model (classifier x regressor)
    def __init__(self, horizon=12, hidden=(128, 64), epochs=60, lr=1e-3,
                 discount_rate=0.01, backend="auto", random_state=None)
    def fit(self, X, value, event_observed=None) -> "DeepCLV"
    def predict(self, X) -> np.ndarray

def clv_from_survival(survival_curve: np.ndarray, monthly_margin: np.ndarray,
                      discount_rate: float = 0.01) -> np.ndarray
    # discounted expected margin: sum_t S(t) * m * (1+d)^-t   <- bridge between 3.1 and 4.3
```

### 3.3 `prism/models/propensity.py`
```python
class PropensityModel:
    """Multi-arm generalised propensity score with cross-fitting and calibration."""
    def __init__(self, n_arms: int = 4, base_estimator=None, n_splits: int = 5,
                 calibrate: bool = True, clip: tuple[float, float] = (0.01, 0.99),
                 random_state=None)
    def fit(self, X, arm) -> "PropensityModel"
    def predict_proba(self, X) -> np.ndarray        # (n, n_arms), rows sum to 1, clipped
    def oof_proba(self) -> np.ndarray               # cross-fitted, aligned to training rows

def overlap_diagnostics(propensity: np.ndarray, arm: np.ndarray) -> pd.DataFrame
    # per-arm: min/max/quantiles of e_a(X) among treated & untreated, share below/above clip
def standardized_mean_differences(X, arm, weights=None, feature_names=None) -> pd.DataFrame
    # SMD per feature per arm-vs-control, before and after weighting (love plot input)
def trim_by_overlap(propensity, arm, low=0.05, high=0.95) -> np.ndarray   # boolean keep-mask
def effective_sample_size(weights: np.ndarray) -> float
```

---

## 4. `prism.causal` — the crown jewel

Convention throughout: `X` covariates, `w` arm index (int, 0=control), `y` outcome (float, higher =
better unless stated). CATE for arm `a` is `E[Y(a) - Y(0) | X]`.

All learners implement:
```python
class BaseCATELearner:
    def fit(self, X, w, y, *, propensity=None, sample_weight=None) -> Self
    def predict_cate(self, X) -> np.ndarray        # (n, n_arms-1) one column per non-control arm
    def predict(self, X) -> np.ndarray             # alias of predict_cate
    n_arms: int
```

### 4.1 `prism/causal/learners.py`
```python
class SLearner(BaseCATELearner):   # single model with arm as a feature
class TLearner(BaseCATELearner):   # one outcome model per arm
class XLearner(BaseCATELearner):   # imputed-effect learner, propensity-weighted blend
class DRLearner(BaseCATELearner):  # cross-fitted AIPW pseudo-outcome, then regress on X
class RLearner(BaseCATELearner):   # Robinson residualisation, weighted by (w - e)^2
    def __init__(self, base_outcome=None, base_effect=None, base_propensity=None,
                 n_splits: int = 5, random_state=None, **kw)
def make_learner(name: str, **kw) -> BaseCATELearner      # "s"|"t"|"x"|"dr"|"r"|"forest"|"survival"
```
`DRLearner` and `RLearner` MUST use honest **cross-fitting** (K-fold; nuisances fit out-of-fold).

### 4.2 `prism/causal/forest.py`
```python
class CausalForest(BaseCATELearner):
    """Generalized Random Forest style: local centering (Y and W residualised out-of-fold),
    honest splitting (separate split/estimate subsamples), subsampling without replacement,
    and a treatment-heterogeneity split criterion."""
    def __init__(self, n_estimators=400, min_samples_leaf=15, max_depth=None,
                 honest_fraction=0.5, subsample_fraction=0.5, mtry=None,
                 min_treated_per_leaf=3, n_jobs=-1, random_state=None)
    def fit(self, X, w, y, *, propensity=None) -> "CausalForest"
    def predict_cate(self, X) -> np.ndarray
    def predict_interval(self, X, alpha=0.05) -> tuple[np.ndarray, np.ndarray]   # bootstrap-of-little-bags
    def feature_importances_(self) -> np.ndarray
```
Binary-arm internals are fine; for K arms fit one forest per non-control arm against control.

### 4.3 `prism/causal/survival_uplift.py` — **THE DIFFERENTIATOR**
Treatment effects on *time-to-event* outcomes with right-censoring, targeting **RMST** and
**discounted lifetime value** rather than a binary conversion.

```python
class CausalSurvivalUplift(BaseCATELearner):
    """CATE on restricted mean survival time and on discounted CLV under right-censoring.

    Pipeline:
      1. Censoring model  G(t|X)  (Kaplan-Meier or covariate-dependent) -> IPCW weights.
      2. Per-arm discrete-time hazard models -> counterfactual survival curves S_a(t|X).
      3. tau_rmst_a(X)  = sum_t [S_a(t|X) - S_0(t|X)]
         tau_value_a(X) = sum_t [S_a(t|X) - S_0(t|X)] * margin(X) * (1+d)^-t
      4. Doubly-robust correction: an IPCW-AIPW pseudo-outcome for RMST is regressed on X so the
         estimate is consistent if either the hazard model or the propensity model is right.
    """
    def __init__(self, horizon: int = 12, n_arms: int = 4, discount_rate: float = 0.01,
                 hazard_model: str = "discrete", censoring_model: str = "km",
                 doubly_robust: bool = True, n_splits: int = 5, random_state=None)
    def fit(self, X, w, event_time, event_observed, *, margin=None, propensity=None) -> Self
    def predict_survival_curves(self, X) -> np.ndarray          # (n, n_arms, horizon)
    def predict_cate(self, X) -> np.ndarray                     # (n, n_arms-1) RMST effect, months
    def predict_value_cate(self, X, margin=None) -> np.ndarray  # (n, n_arms-1) discounted currency
    def predict_churn_cate(self, X, within=1) -> np.ndarray     # (n, n_arms-1) prob. effect

def ipcw_weights(event_time, event_observed, horizon, X=None, model="km") -> np.ndarray
def rmst_pseudo_outcome(event_time, event_observed, horizon,
                        censor_surv: np.ndarray) -> np.ndarray   # IPCW pseudo-outcome for RMST
```

### 4.4 `prism/causal/evaluate.py`
```python
def qini_curve(y, w, cate, treatment_cost=0.0) -> pd.DataFrame     # frac_targeted, incremental, random
def qini_score(y, w, cate) -> float                                 # area over the random line
def auuc(y, w, cate, normalize=True) -> float
def uplift_at_k(y, w, cate, k: float = 0.2) -> float
def uplift_by_decile(y, w, cate, n_bins=10) -> pd.DataFrame
def pehe(true_cate, est_cate) -> float                              # sqrt mean squared error of tau
def eps_ate(true_cate, est_cate) -> float                           # |ATE error|
def policy_value(y, w, policy_arm, propensity, *, method="dr",
                 mu_hat=None, n_boot=500, random_state=None) -> dict
    # returns {value, se, ci_low, ci_high, method, n}; methods: "ipw" | "snipw" | "dr"
def policy_risk(true_outcomes_by_arm: np.ndarray, policy_arm: np.ndarray) -> float
def gates(y, w, cate, propensity, n_groups=5) -> pd.DataFrame       # Chernozhukov GATES with CIs
def blp_calibration(y, w, cate, propensity) -> pd.DataFrame         # best linear predictor: beta1~1, beta2~1
def targeting_operator_characteristic(y, w, cate) -> pd.DataFrame   # TOC curve
def compare_learners(results: dict[str, np.ndarray], true_cate=None,
                     y=None, w=None) -> pd.DataFrame                # tidy leaderboard
```

### 4.5 `prism/causal/refute.py`
```python
@dataclass
class RefutationResult:
    name: str; original: float; refuted: float; p_value: float | None
    passed: bool; detail: str

def placebo_treatment(estimator_factory, X, w, y, n_sim=20, random_state=None) -> RefutationResult
def random_common_cause(estimator_factory, X, w, y, n_sim=10, random_state=None) -> RefutationResult
def subset_refuter(estimator_factory, X, w, y, fraction=0.7, n_sim=10, random_state=None) -> RefutationResult
def add_unobserved_confounder(estimator_factory, X, w, y, confounding_strength: float,
                              random_state=None) -> RefutationResult
def e_value(estimate: float, ci_low: float, ci_high: float) -> dict     # VanderWeele & Ding
def rosenbaum_bounds(y, w, gammas=(1.0,1.2,1.5,2.0,3.0)) -> pd.DataFrame
def sensitivity_contour(estimator_factory, X, w, y,
                        r2_y_grid=None, r2_w_grid=None) -> pd.DataFrame  # Austen / Cinelli-Hazlett style
def run_refutation_suite(estimator_factory, X, w, y, random_state=None) -> pd.DataFrame
```

---

## 5. `prism.decision`

### 5.1 `prism/decision/economics.py`
```python
@dataclass
class EconomicConfig:
    arm_costs: tuple[float, ...] = ARM_COSTS
    margin_rate: float = 0.30
    discount_rate: float = 0.01
    horizon: int = 12
    offer_redemption_rate: tuple[float, ...] = (0.0, 0.62, 0.71, 0.48)  # cost only if redeemed
    fixed_campaign_cost: float = 0.0

def expected_net_value(value_cate: np.ndarray, econ: EconomicConfig) -> np.ndarray
    # (n, n_arms) including arm 0 = do-nothing with value 0; net = tau_value_a - expected_cost_a
def build_decision_frame(customer_id, value_cate, rmst_cate, churn_risk, clv,
                         econ: EconomicConfig) -> pd.DataFrame
```

### 5.2 `prism/decision/optimize.py`
```python
@dataclass
class AllocationResult:
    assignment: np.ndarray          # (n,) chosen arm per customer, 0 = no offer
    total_cost: float
    expected_incremental_value: float
    n_treated: int
    per_arm: pd.DataFrame
    dual_lambda: float | None
    method: str
    def summary(self) -> pd.DataFrame

def greedy_knapsack(net_value: np.ndarray, cost: np.ndarray, budget: float) -> AllocationResult
def lagrangian_allocate(net_value, cost, budget, *, tol=1e-6, max_iter=100) -> AllocationResult
    # bisect on the shadow price lambda; per customer pick argmax_a (v_a - lambda * c_a)
def lp_allocate(net_value, cost, budget, *, arm_capacity=None) -> AllocationResult
    # scipy.optimize.linprog relaxation of the multi-choice knapsack + rounding
def efficient_frontier(net_value, cost, budgets: np.ndarray) -> pd.DataFrame
    # budget -> expected incremental value; the chart every exec wants
def baseline_policies(churn_risk, clv, net_value, cost, budget) -> dict[str, AllocationResult]
    # "treat_none", "treat_all", "highest_risk", "highest_clv", "risk_x_clv", "random"
```

### 5.3 `prism/decision/policy_eval.py`
```python
def evaluate_policy(panel_test: pd.DataFrame, assignment: np.ndarray, propensity: np.ndarray,
                    outcome_col: str, mu_hat: np.ndarray | None = None,
                    n_boot: int = 500, random_state=None) -> pd.DataFrame
def compare_policies(policies: dict[str, np.ndarray], panel_test, propensity, outcome_col,
                     mu_hat=None, **kw) -> pd.DataFrame     # tidy, with bootstrap CIs
def oracle_gap(assignment, ground_truth: pd.DataFrame, econ) -> dict
    # realised net value vs gt_oracle_net_value -> % of achievable value captured
```

### 5.4 `prism/decision/fairness.py`
```python
def policy_fairness_audit(assignment, protected: pd.Series, outcome=None,
                          net_value=None) -> pd.DataFrame
    # per group: treat rate, mean net value, demographic parity diff, equal-opportunity diff
def disparate_impact_ratio(assignment, protected) -> float
def reweight_for_parity(net_value, cost, budget, protected, tolerance=0.1) -> AllocationResult
    # budget allocation with per-group treat-rate constraints
```

---

## 6. `prism.monitoring`
```python
# prism/monitoring/drift.py
def psi(expected: np.ndarray, actual: np.ndarray, n_bins=10) -> float
def ks_statistic(expected, actual) -> tuple[float, float]
def js_divergence(expected, actual, n_bins=20) -> float
def chi2_categorical(expected: pd.Series, actual: pd.Series) -> tuple[float, float]
def feature_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                         features=None) -> pd.DataFrame     # per feature: psi, ks, p, severity
def prediction_drift(ref_pred, cur_pred) -> dict
def cate_stability(cate_ref: np.ndarray, cate_cur: np.ndarray) -> dict
    # rank correlation, decile-migration matrix, sign-flip rate -- causal-specific monitoring
def performance_decay(panel, model_scores, outcome_col, by="period") -> pd.DataFrame

# prism/monitoring/alerts.py
@dataclass
class Alert: name: str; severity: str; value: float; threshold: float; message: str
def evaluate_alerts(drift_report, decay_report, thresholds: dict | None = None) -> list[Alert]
def render_alerts(alerts: list[Alert]) -> str
```

---

## 7. `prism.serving`
```python
# prism/serving/schemas.py  (pydantic v2)
class CustomerFeatures(BaseModel)      # the 23 features, with validators + example
class ScoreRequest(BaseModel)          # customers: list[CustomerFeatures], budget: float | None
class ArmScore(BaseModel)              # arm, arm_name, cate_rmst, cate_value, cost, net_value
class CustomerScore(BaseModel)         # customer_id, churn_risk, clv, rmst, arms: list[ArmScore],
                                       # recommended_arm, expected_net_value
class ScoreResponse(BaseModel)         # scores, model_version, latency_ms
class PolicyResponse(BaseModel)        # assignment summary + budget diagnostics

# prism/serving/api.py
app: FastAPI
# GET  /health            -> {status, model_version, loaded_at}
# GET  /metadata          -> model card summary, feature list, training window
# POST /score             -> ScoreResponse
# POST /policy            -> PolicyResponse (budget-constrained assignment for the batch)
# POST /explain           -> per-customer SHAP-style contributions (graceful if shap missing)
# GET  /metrics           -> prometheus-style text of request counters + latency histogram
class ModelBundle:
    @classmethod
    def load(cls, path="artifacts/models/bundle.joblib") -> "ModelBundle"
    def score(self, df: pd.DataFrame) -> pd.DataFrame
    def policy(self, df: pd.DataFrame, budget: float) -> pd.DataFrame
```
The API must start even if no bundle is on disk (`/health` reports `degraded`).

---

## 8. `prism.pipelines.run_all`
```python
def main(config_path: str = "configs/default.yaml", steps: str = "all") -> int
# steps: simulate,ingest,features,train,causal,evaluate,optimize,monitor,report
# writes artifacts/reports/*.json|*.md|*.csv and docs/figures/*.png; logs to MLflow if available.
```
CLI: `python -m prism.pipelines.run_all --config configs/default.yaml --steps all`

---

## 9. Reporting artifacts the pipeline MUST produce
- `artifacts/reports/metrics.json` — one flat dict of every headline number.
- `artifacts/reports/leaderboard.csv` — learner x {PEHE, eps_ATE, Qini, AUUC, policy value}.
- `artifacts/reports/policy_comparison.csv` — policy x {net value, CI, ROI, n_treated, cost}.
- `artifacts/reports/refutation.csv`, `artifacts/reports/drift.csv`, `artifacts/reports/fairness.csv`
- `docs/figures/`: `qini.png`, `uplift_deciles.png`, `survival_curves.png`, `efficient_frontier.png`,
  `love_plot.png`, `calibration.png`, `segment_heatmap.png`, `sensitivity_contour.png`, `drift.png`
- `artifacts/reports/model_card.md`, `artifacts/reports/decision_memo.md`
