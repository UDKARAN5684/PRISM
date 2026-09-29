# Methodology

This document states precisely what PRISM estimates, what it must assume to do so, how each
estimator works, and how the results are validated. It is written so a reviewer can find the
weak point quickly — because every causal claim has one, and hiding it is worse than naming it.

---

## 1. The decision problem

A subscription business can send each customer one of `K = 4` retention offers each month:

| arm | name | marginal cost | redemption |
|-----|------|---------------|------------|
| 0 | control (no offer) | 0 | — |
| 1 | `discount_10` | 12 | 0.62 |
| 2 | `discount_25` | 30 | 0.71 |
| 3 | `concierge` | 55 | 0.48 |

The budget covers only a fraction of the base. The question is **not** "who is likely to churn"
but:

> Choose an assignment `π: X → {0,…,K−1}` that maximises expected **incremental** discounted
> margin subject to `Σᵢ cost(π(Xᵢ)) ≤ B`.

Churn probability is not the right ranking signal for this problem, and it is easy to show why.
Split customers by how they respond to an offer (Radcliffe & Surry's classical taxonomy):

| segment | churns if left alone? | churns if treated? | should we treat? |
|---|---|---|---|
| **persuadable** | yes | no | **yes — this is the only group that pays** |
| **sure thing** | no | no | no — wasted spend |
| **lost cause** | yes | yes | no — wasted spend |
| **sleeping dog** | no | **yes** | **no — treating them destroys value** |

A churn-risk model ranks *persuadables* and *lost causes* together, because both are likely to
churn. It cannot see sleeping dogs at all. PRISM's simulator gives sleeping dogs an 18% share and a
materially positive treatment effect on the hazard, so a risk-ranked policy provably loses money
against a causal one — and the pipeline measures that gap in currency.

---

## 2. Causal framework

### 2.1 Potential outcomes

For customer-period `i` and arm `a`, let `Yᵢ(a)` be the potential outcome. PRISM works with three
outcome definitions simultaneously, which is unusual and deliberate:

- `Y^churn(a)` — churn within one period (binary)
- `Y^rmst(a)` — **restricted mean survival time** over horizon `H`: expected months retained
- `Y^value(a)` — **discounted margin** over `H`

The target estimand for arm `a` is the conditional average treatment effect

```
τ_a(x) = E[ Y(a) − Y(0) | X = x ]
```

`τ^value_a(x)` is the one that matters economically; `τ^rmst_a(x)` is the one that is most robust to
the revenue model being wrong; `τ^churn_a(x)` is what a conventional uplift model would target and
is kept for comparison.

### 2.2 Identification assumptions

`τ_a(x)` is identified from observational data under:

1. **Consistency / SUTVA** — `Y = Y(a)` for the arm actually received; no interference between
   customers. *Realistic here?* Partly. Word-of-mouth about a discount violates it. The simulator
   generates no interference, so PRISM does not test robustness to it. **Named limitation.**
2. **Positivity / overlap** — `0 < e_a(x) < 1` for all `a`, `x`. *Enforced and measured.* The
   simulator clips generalised propensities to `[0.02, 0.85]`; `prism.models.propensity`
   reports the realised overlap and `trim_by_overlap` removes rows where it fails.
3. **Unconfoundedness** — `{Y(0),…,Y(K−1)} ⊥ A | X`. *This is the assumption that is false in
   every real dataset,* and PRISM makes it false here too: the simulator generates a hidden
   confounder `U` that shifts both assignment and the hazard, and `U` is deliberately excluded from
   the feature set. The refutation suite (§6) quantifies how much bias that causes.
4. **Non-informative censoring** — for the survival outcomes, censoring is independent of the event
   time given `X` and `A`. Handled by IPCW (§4.3).

The **RCT block** (15% of customers randomised) exists so that every one of these can be checked
against a design where 2–4 hold by construction. Estimating on the observational block and
validating on the randomised block is the honest version of this analysis, and it is what the
pipeline does.

### 2.3 Why the RCT block does not make the rest unnecessary

A reasonable reviewer asks: if you have an experiment, why bother with observational estimation?
Because the experiment is small, expensive and historical. The observational log is 5× larger and
covers arms and segments the experiment under-samples. The practical workflow — estimate on the
big biased sample, calibrate and validate on the small clean one — is the one that survives contact
with a real company, and it is the workflow PRISM implements.

---

## 3. Nuisance estimation

### 3.1 Generalised propensity score

`e_a(x) = P(A = a | X = x)` for `K` arms, from a multinomial gradient-boosted classifier with
**5-fold cross-fitting** (each row is scored by a model that never saw it) and isotonic/sigmoid
calibration, then clipped to `[0.01, 0.99]` and renormalised.

Diagnostics produced:

- **Overlap table** — per-arm propensity quantiles among treated and untreated.
- **Standardised mean differences** before and after IPW, per feature, per arm (the *love plot*).
  `|SMD| < 0.1` after weighting is the conventional balance target.
- **Effective sample size** `ESS = (Σw)² / Σw²` — the honest denominator once weights are applied.
  A large nominal `n` with a small ESS is a warning that the estimate rests on a handful of rows.

### 3.2 Outcome models

Gradient-boosted regressors/classifiers (LightGBM → XGBoost → sklearn `HistGradientBoosting`,
whichever is installed) for `μ_a(x) = E[Y | X = x, A = a]`, again cross-fitted.

---

## 4. Estimators

### 4.1 Meta-learners

All are implemented in `prism/causal/learners.py` directly on top of scikit-learn, rather than
imported from a causal-ML package, so the cross-fitting is visible and auditable.

| learner | pseudo-outcome / construction | strength | weakness |
|---|---|---|---|
| **S** | one model on `(X, A)`, contrast predictions | simple, stable in small samples | regularisation shrinks the treatment effect toward zero |
| **T** | one model per arm, subtract | no shrinkage of the contrast | variance is the sum of both arms' errors; bad with imbalanced arms |
| **X** | impute each unit's effect from the other arm's model, then blend by propensity | strong when one arm is rare | two-stage error accumulation |
| **DR** | AIPW pseudo-outcome (below), then regress on `X` | **doubly robust**, Neyman-orthogonal | needs good overlap; unstable if `e_a` is near 0 |
| **R** | Robinson residualisation, minimise `Σ[(Y−m̂(X)) − τ(X)(A−ê(X))]²` | orthogonal, efficient | needs a flexible `m̂` |

The AIPW (doubly-robust) pseudo-outcome for arm `a` vs control:

```
ψᵢ = μ̂_a(Xᵢ) − μ̂_0(Xᵢ)
     + 1{Aᵢ=a}/ê_a(Xᵢ) · (Yᵢ − μ̂_a(Xᵢ))
     − 1{Aᵢ=0}/ê_0(Xᵢ) · (Yᵢ − μ̂_0(Xᵢ))
```

`E[ψ | X] = τ_a(X)` if **either** the outcome models **or** the propensity model is correct. That
is the "doubly robust" property, and it is the reason DR and R are the defaults.

**Cross-fitting** is mandatory for DR and R: nuisances are fit on `K−1` folds and applied to the
held-out fold. Without it the pseudo-outcome inherits the nuisance model's overfitting and the
resulting `τ̂` is biased toward zero with anticonservative intervals.

### 4.2 Causal forest

`prism/causal/forest.py` implements a Generalized-Random-Forest-style estimator from scratch:

- **Local centering** — `Y` and `W` are residualised against out-of-fold `m̂(X)` and `ê(X)` first, so
  splits chase treatment-effect heterogeneity rather than outcome heterogeneity.
- **Honest splitting** — each tree's subsample is halved; one half chooses the splits, the other
  estimates the leaf effects. This is what makes the leaf estimates approximately unbiased and lets
  us quote intervals at all.
- **Subsampling without replacement** (not bootstrap), which is what the GRF asymptotic theory
  requires.
- **Heterogeneity split criterion** — maximise the between-child variance of the treatment effect,
  with a `min_treated_per_leaf` guard so no leaf estimates an effect from one treated unit.
- **Intervals** via bootstrap-of-little-bags across tree groups.

### 4.3 Causal Survival Uplift — the distinctive estimator

Conventional uplift modelling asks "does the offer change the probability of a binary event?".
That throws away *when* churn happens, and it cannot express "this offer buys 1.8 extra months".
`prism/causal/survival_uplift.py` targets the survival outcomes directly.

**Step 1 — censoring model.** Estimate `G(t | x) = P(C > t | X = x)` by Kaplan–Meier (or a
covariate-dependent model). Right-censoring is pervasive here: a customer observed for 5 months
of a 12-month horizon contributes real information and must not be dropped.

**Step 2 — per-arm hazards.** Fit a discrete-time hazard model per arm:

```
h_a(t | x) = σ( f_a(x) + α_t )
```

a pooled logistic model on the person-period expansion (torch MLP when available, otherwise
`LogisticRegression`), with a free intercept per period. Survival follows by the product rule:

```
S_a(k | x) = Π_{j<k} ( 1 − h_a(j | x) )
```

This construction guarantees `S ∈ (0,1)` and monotone non-increasing — properties a directly-fit
regression on RMST does not guarantee, and which the test suite asserts.

**Step 3 — contrasts.**

```
τ^rmst_a(x)  = Σ_{k=1..H} [ S_a(k|x) − S_0(k|x) ]                            (months)
τ^value_a(x) = Σ_{k=1..H} [ S_a(k|x) − S_0(k|x) ] · m(x) · (1+d)^(−k)        (currency)
```

where `m(x)` is the customer's monthly margin and `d` the monthly discount rate. The second line is
the bridge from survival analysis to finance, and it is what the optimiser consumes.

**Step 4 — doubly-robust correction.** Steps 2–3 are a *plug-in* estimator: consistent only if the
hazard model is right. To get the double-robustness property back, PRISM forms an IPCW-AIPW
pseudo-outcome for RMST,

```
ψ^rmst_i = Σ_k [ Ŝ_a(k|Xᵢ) − Ŝ_0(k|Xᵢ) ]
           + 1{Aᵢ=a}/ê_a(Xᵢ) · δᵢ/Ĝ(Tᵢ|Xᵢ) · ( min(Tᵢ,H) − RMST_a(Xᵢ) )
           − 1{Aᵢ=0}/ê_0(Xᵢ) · δᵢ/Ĝ(Tᵢ|Xᵢ) · ( min(Tᵢ,H) − RMST_0(Xᵢ) )
```

and regresses `ψ` on `X` with cross-fitting. The `1/Ĝ` factor up-weights observations that survived
the censoring process, undoing the selection that censoring induces.

**Why this matters commercially.** An offer that delays churn by three months but does not prevent
it has *zero* effect on a 12-month binary churn flag measured at month 12, yet three months of
margin is real money. RMST sees it; binary uplift does not.

---

## 5. Evaluating a causal model without ground truth

The core difficulty: `τᵢ` is never observed, so there is no held-out "accuracy". PRISM uses four
complementary families, all in `prism/causal/evaluate.py`.

1. **Qini / AUUC / uplift-by-decile.** Rank by `τ̂`, walk down the ranking, and compare the
   incremental outcome against random targeting. Qini handles the treated/control imbalance;
   AUUC is its normalised cousin. These measure *ranking*, not calibration.

2. **GATES** (Chernozhukov, Demirer, Duflo & Fernández-Val). Sort into `G` groups by `τ̂` and
   estimate the true ATE inside each group with a doubly-robust estimator, with confidence
   intervals. If the model has real signal, the group effects increase monotonically and the
   top-vs-bottom difference is significantly positive. This is the test that catches a model with a
   good Qini that is only re-discovering the outcome level.

3. **BLP calibration.** Regress the DR pseudo-outcome on `(1, τ̂ − mean τ̂)`. A well-calibrated CATE
   model gives slope `β₂ ≈ 1`. `β₂ ≪ 1` means the spread of `τ̂` is exaggerated — very common, and
   invisible to Qini.

4. **Policy value with confidence intervals.** The decision-relevant metric. For policy `π`:

   ```
   IPW:     V̂ = (1/n) Σ 1{Aᵢ = π(Xᵢ)}/ê_{π(Xᵢ)}(Xᵢ) · Yᵢ
   SNIPW:   self-normalised, lower variance, slightly biased
   DR:      V̂ = (1/n) Σ [ μ̂_{π(Xᵢ)}(Xᵢ) + 1{Aᵢ=π(Xᵢ)}/ê · (Yᵢ − μ̂_{π(Xᵢ)}(Xᵢ)) ]
   ```

   with bootstrap CIs. A policy is only "better" if its interval clears the baseline's.

**And, uniquely here: ground truth.** Because the simulator computes `τᵢ` analytically, PRISM can
also report the metrics no real project can:

```
PEHE   = sqrt( mean( (τᵢ − τ̂ᵢ)² ) )          # precision in estimating heterogeneous effects
ε_ATE  = | mean(τᵢ) − mean(τ̂ᵢ) |
policy regret = mean( Y(a*ᵢ) − Y(π(Xᵢ)) )    # distance from the oracle policy
```

This is the point of building the DGP rather than grabbing a Kaggle CSV: **the estimator can be
proven correct before it is trusted on data where it cannot be checked.**

---

## 6. Refutation and sensitivity

Passing a validation metric is not evidence of a causal effect. `prism/causal/refute.py` runs:

| test | procedure | passes if |
|---|---|---|
| **Placebo treatment** | permute `A`, re-estimate | effect collapses to ≈ 0 |
| **Random common cause** | add an irrelevant covariate | estimate is unchanged |
| **Subset refuter** | re-estimate on random 70% subsets | estimate is stable across subsets |
| **Added confounder** | inject a synthetic confounder of known strength | degradation is proportional and predictable |

Plus two sensitivity analyses that quantify *how strong an unmeasured confounder would have to be*
to overturn the conclusion:

- **E-value** (VanderWeele & Ding) — the minimum strength of association, on the risk-ratio scale,
  that an unmeasured confounder would need with both treatment and outcome to explain away the
  observed effect. Reported for the point estimate and the CI bound nearest the null.
- **Austen / Cinelli–Hazlett contour plot** — a grid over (partial `R²` of the confounder with the
  outcome) × (with the treatment), shading where the estimate would be nullified, with the observed
  covariates plotted as benchmarks: *"you would need a confounder twice as strong as `tenure_months`
  to erase this."*

Because the simulator's hidden confounder `U` has a known strength, PRISM can check the E-value
against reality — another thing no observational study can do.

---

## 7. From effects to decisions

### 7.1 Unit economics

Expected net value of giving arm `a` to customer `i`:

```
netᵢₐ = τ^value_a(Xᵢ) − cost_a · redemption_a − fixed_a
net_i0 = 0                                          (do nothing is always available)
```

Cost is multiplied by the redemption rate because an unredeemed discount costs nothing but the
send. Ignoring this systematically understates ROI, which is a common error in retention analyses.

### 7.2 Budget-constrained multi-choice assignment

Each customer gets **at most one** arm, total spend `≤ B`. This is a *multiple-choice knapsack
problem* — NP-hard exactly, but its LP relaxation has an exploitable structure. Three solvers,
implemented and compared:

- **Greedy** — sort by net value per unit cost, fill. Fast, no guarantee.
- **Lagrangian dual** — bisect on a shadow price `λ` and give each customer
  `argmax_a (netᵢₐ − λ·cost_a)`. At the optimal `λ*` the budget binds, and `λ*` has a direct
  business reading: **the marginal return on the next currency unit of retention budget.** This is
  the number a CFO actually wants, and the greedy solver cannot produce it.
- **LP relaxation** via `scipy.optimize.linprog` plus rounding, used to bound the integrality gap
  and confirm the Lagrangian solution is within a fraction of a percent of optimal.

The **efficient frontier** — value as a function of `B` — falls out of the `λ` sweep for free and is
the single most useful chart for a budget conversation.

### 7.3 Baselines it must beat

`treat_none`, `treat_all`, `highest_risk` (the conventional churn model), `highest_clv`,
`risk × clv`, and `random`. Each is scored with the same DR off-policy estimator and bootstrap CIs,
so the headline claim is a like-for-like comparison, not a straw man.

### 7.4 Fairness

A retention budget is a scarce good, and a policy that optimises profit can concentrate offers in a
way a regulator would object to. `prism/decision/fairness.py` reports per-group treat rates,
demographic-parity and equal-opportunity gaps, and the disparate-impact ratio, and offers a
constrained re-allocation with per-group treat-rate floors so the cost of fairness can be priced
rather than argued about.

---

## 8. Known limitations

Stated plainly, because a reviewer will find them anyway:

1. **No interference.** Customers are independent. Real discounts leak between customers and across
   time; SUTVA is the least defensible assumption here.
2. **Frozen-covariate counterfactuals.** Ground-truth counterfactual survival holds covariates at
   their decision-period values while letting seasonality and drift advance. A fully dynamic
   counterfactual would need a g-formula over the covariate process. This is documented in the DGP
   and is the reason ground truth is exact rather than approximate.
3. **One-shot campaign, not a dynamic regime.** The offer is assigned once and persists, and PRISM
   estimates a static CATE at that decision point. A genuinely sequential policy — re-deciding each
   month, conditioning on what happened since — would need a marginal structural model or fitted-Q
   iteration. An earlier version *did* re-assign monthly, and doing so quietly broke the benchmark:
   the realised data then measured a random treatment *sequence* while the ground truth measured a
   sustained arm, which are different estimands. See ADR-013 in `docs/DECISIONS.md`.
4. **Simulated primary data.** The DGP is realistic but it is a model, and it is a model *I* chose,
   which means the estimators are being graded against assumptions that partly match their own.
   That is exactly why `prism/data/real.py` also runs the pipeline on the Hillstrom randomised
   e-mail experiment and the IBM Telco churn data — real data, real messiness, no ground truth.
5. **Off-policy evaluation needs overlap.** Policy value estimates are only trustworthy where the
   logging policy had some chance of taking the evaluated action. The overlap diagnostics bound
   where the estimates can be believed, and the reports say so.
6. **Heterogeneity is recovered as a ranking, not as calibrated levels.** The best learner's PEHE
   (58.2) beats a constant-effect baseline (61.4) by only about 5%, and the rank correlation with
   the true effect is +0.34. The *aggregate* effects are recovered well (ε_ATE 6.8 against 48.0 for
   the naive estimator) and the ranking is good enough to drive targeting, but PRISM does not
   produce trustworthy individual-level effect estimates at this signal-to-noise ratio. Anyone
   quoting a per-customer number from it would be overclaiming.
7. **Qini and AUUC are near zero here, and that is expected.** The classical uplift curve is
   identified only under randomisation; computing it on confounded observational test data does not
   measure what it measures in an experiment. This is why the headline metrics are ground-truth
   PEHE / rank correlation and doubly-robust policy value rather than Qini. On the randomised block
   (or on the Hillstrom data) Qini is the right tool again.
8. **CATE estimates are unstable across refreshes.** About 30% of customers change the *sign* of
   their estimated effect between the training and test periods — i.e. their treat/do-not-treat
   decision flips. The monitoring layer measures this deliberately (`cate_stability`), and in
   production it would justify a stability constraint or a hysteresis band before acting on a
   refreshed model.

---

## 9. References

- Radcliffe, N. & Surry, P. (2011). *Real-World Uplift Modelling with Significance-Based Uplift Trees.*
- Athey, S. & Imbens, G. (2016). *Recursive partitioning for heterogeneous causal effects.* PNAS.
- Wager, S. & Athey, S. (2018). *Estimation and inference of heterogeneous treatment effects using random forests.* JASA.
- Athey, S., Tibshirani, J. & Wager, S. (2019). *Generalized Random Forests.* Annals of Statistics.
- Künzel, S., Sekhon, J., Bickel, P. & Yu, B. (2019). *Metalearners for estimating heterogeneous treatment effects.* PNAS.
- Nie, X. & Wager, S. (2021). *Quasi-oracle estimation of heterogeneous treatment effects.* Biometrika.
- Chernozhukov, V. et al. (2018). *Double/debiased machine learning.* Econometrics Journal.
- Chernozhukov, V., Demirer, M., Duflo, E. & Fernández-Val, I. (2018). *Generic machine learning inference on heterogeneous treatment effects.* NBER WP 24678.
- Robins, J., Rotnitzky, A. & Zhao, L. (1994). *Estimation of regression coefficients when some regressors are not always observed.* JASA. (IPCW)
- Tsiatis, A. (2006). *Semiparametric Theory and Missing Data.*
- VanderWeele, T. & Ding, P. (2017). *Sensitivity analysis in observational research: introducing the E-value.* Annals of Internal Medicine.
- Cinelli, C. & Hazlett, C. (2020). *Making sense of sensitivity: extending omitted variable bias.* JRSS-B.
- Fader, P., Hardie, B. & Lee, K. (2005). *"Counting your customers" the easy way: an alternative to the Pareto/NBD model.* Marketing Science. (BG/NBD)
- Zhao, Y., Zeng, D., Rush, A. & Kosorok, M. (2012). *Estimating individualized treatment rules using outcome weighted learning.* JASA.
- Uplift evaluation: Gutierrez, P. & Gérardy, J-Y. (2017). *Causal inference and uplift modelling: a review of the literature.* PMLR.
