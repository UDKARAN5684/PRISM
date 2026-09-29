# Architecture Decision Records

Short records of the choices that shaped PRISM, including the ones I would defend in an interview
and the ones where I took a deliberate shortcut. Each records the alternative I rejected, because
a decision without a rejected alternative is not a decision.

---

## ADR-001 — Estimate treatment effects, not churn probability

**Status:** accepted · **Drives:** the entire `prism.causal` package

**Context.** The conventional retention project trains a churn classifier and targets the top
decile. This is the default because churn labels are easy and treatment effects are not.

**Decision.** Target `τ_a(x) = E[Y(a) − Y(0) | X]` and make the churn model a *baseline to beat*,
not the product.

**Why.** Churn probability and treatment effect are different orderings of the same customers, and
the difference is not academic. Customers who will churn regardless (*lost causes*) rank at the top
of a risk model and are worth nothing. Customers whose churn is *caused* by being contacted
(*sleeping dogs*) are invisible to a risk model and have negative value. The simulator gives sleeping
dogs an 18% share specifically so this gap shows up in currency in the results table.

**Rejected.** Churn model + business-rule overlay ("don't contact recent complainers"). Cheaper, and
it is what most teams do, but the rules encode guesses about heterogeneity that the data can
estimate directly.

**Cost.** Causal estimation needs overlap, unconfoundedness and far more care in evaluation. Most of
`docs/METHODOLOGY.md` exists to pay this cost honestly.

---

## ADR-002 — Build a ground-truth simulator instead of using a Kaggle dataset as the primary source

**Status:** accepted · **Drives:** `prism/data/dgp.py`

**Context.** Causal models cannot be validated on observational data, because `τᵢ` is never
observed. Every public churn dataset shares this limit, and most have no treatment variable at all.

**Decision.** Make a rigorous DGP the primary data source, with analytically computed counterfactual
survival and value under every arm, and run the pipeline on real public datasets as a secondary
track (`prism/data/real.py`).

**Why.** It is the only way to report **PEHE** and **policy regret against an oracle** — to *prove*
the estimator works before trusting it where it cannot be checked. This mirrors how the causal-ML
literature itself benchmarks (IHDP, ACIC), and it is the single thing that most distinguishes this
project from a notebook that reports AUC.

**Rejected.** (a) Real data only — no ground truth, so the causal claims would be unfalsifiable.
(b) Simulator only — too easy to build a DGP that flatters my own estimators.

**Cost.** A reviewer can fairly say "your estimator is graded against assumptions it shares." That
objection is recorded in METHODOLOGY §8.4, and the Hillstrom randomised e-mail experiment is wired
in precisely as the answer to it: a real, randomised, three-arm uplift dataset with no simulation.

---

## ADR-003 — Model time-to-event, not a binary churn flag

**Status:** accepted · **Drives:** `prism/models/survival.py`, `prism/causal/survival_uplift.py`

**Context.** Binary churn needs an arbitrary window. Twelve months makes an offer that delays churn
by three months look worthless.

**Decision.** Model the discrete-time hazard and target **RMST** and **discounted lifetime value**.

**Why.** Delay *is* value: three extra months of margin is money whether or not the customer
eventually leaves. RMST is measured in months, which is a unit executives already reason in, and it
converts to currency with one discounting step. It also handles right-censoring properly instead of
discarding partially-observed customers.

**Rejected.** Binary uplift at a fixed horizon (simpler, and the industry default) — kept as a
comparison arm rather than as the target.

**Cost.** Censoring must be handled with IPCW, the estimator is more complex, and there is far less
off-the-shelf tooling. That complexity is the differentiator, not an accident.

---

## ADR-004 — Implement the causal estimators rather than importing EconML/CausalML/grf

**Status:** accepted · **Drives:** `prism/causal/*`

**Decision.** Write the meta-learners, causal forest, uplift metrics and refutation suite directly
on scikit-learn.

**Why.** Three reasons, in order of weight. (1) *Legibility*: cross-fitting, honest splitting and
the AIPW pseudo-outcome are where these methods go wrong, and in a portfolio the point is to show
the mechanism, not to hide it behind one import. (2) *Fit*: no library implements CATE on RMST with
IPCW under multiple arms, which is the estimator this project is about. (3) *Dependencies*: EconML
pins narrow scikit-learn ranges and breaks often; the serving image stays small without it.

**Rejected.** EconML/CausalML. Faster to write and battle-tested — a better choice for production at
a company that already trusts them, a worse choice for demonstrating that I understand the method.

**Cost.** My implementations are less optimised and less tested than `grf`. Mitigated by validating
against known ground truth, checking gradients numerically, and cross-checking against `lifelines`
where an equivalent exists.

---

## ADR-005 — Causal forest via R-learner-weighted honest sklearn trees

**Status:** accepted · **Drives:** `prism/causal/forest.py`, `SPEC_PERF.md` §2

**Context.** A faithful GRF needs a custom split criterion. Written in Python, the tree builder is
the bottleneck and a 400-tree forest takes hours.

**Decision.** Locally center `Y` and `W` out-of-fold, then fit sklearn `DecisionTreeRegressor` with
pseudo-outcome `Ỹ/W̃` and `sample_weight = W̃²`, which is exactly the R-learner least-squares
objective; re-estimate leaf values honestly on a held-out half as a ratio of sums.

**Why.** Same estimating equation as GRF with local centering, but the inner loop runs in C. Honest
splitting and subsampling-without-replacement are preserved, so the statistical properties that
matter survive. 400 trees on 100k×50 in under 90 seconds.

**Rejected.** (a) Pure-Python GRF — correct and unusable. (b) Cython/numba — a build dependency and
a platform-portability problem for marginal gain over this.

**Cost.** Not bit-identical to `grf`; the intervals are bootstrap-of-little-bags rather than the GRF
asymptotic variance. Both limitations are documented in the class docstring.

---

## ADR-006 — Multi-arm, budget-constrained assignment rather than binary targeting

**Status:** accepted · **Drives:** `prism/decision/optimize.py`

**Decision.** Four arms with different costs and effects, and a hard budget, solved as a
multiple-choice knapsack by greedy, Lagrangian dual and LP relaxation.

**Why.** Binary "treat / don't treat" is not the decision a retention team faces; they choose
*which* offer. The Lagrangian dual is the reason this matters: at the optimum, the shadow price `λ*`
**is the marginal return on the next unit of retention budget** — the number that actually settles a
budget argument. A greedy sort cannot produce it.

**Rejected.** Threshold on uplift. Simpler, and silently wrong once offers have different costs.

**Cost.** Exact optimisation is NP-hard. Mitigated by solving the LP relaxation too and reporting
the integrality gap, so the claim "within x% of optimal" is measured rather than asserted.

---

## ADR-007 — Temporal, group-aware splits with an embargo band

**Status:** accepted · **Drives:** `prism/data/features.py`

**Decision.** Split by period index, never randomly; keep each `customer_id` wholly inside one
split; and drop an embargo band of periods between train and validation.

**Why.** Two leaks, both silent and both fatal. A random split puts the same customer on both sides,
so the model memorises customers instead of learning behaviour. And because outcomes are measured
over a 12-month horizon, adjacent periods share label windows — period 13's outcome window overlaps
period 12's, so training on 12 and validating on 13 leaks the future. The embargo removes that
overlap. A model evaluated without these looks excellent and fails in production.

**Cost.** Fewer usable rows and a pessimistic-looking validation score. That is the correct score.

---

## ADR-008 — Point-in-time feature store with an executable leakage test

**Status:** accepted · **Drives:** `prism/data/features.py`

**Decision.** Compute every feature by as-of join over `(as_of_date − window, as_of_date]`, and ship
`leakage_report()` plus `assert_no_leakage()` that fail loudly if any future event was used.

**Why.** "We were careful about leakage" is unverifiable. A test that fails is verifiable. The
report compares each feature against a deliberately forward-looking version, so the guard is
demonstrated rather than claimed. `as_of_join` is also validated against a brute-force reference
implementation in the module's own smoke test.

**Cost.** As-of joins are slower and fiddlier than a `groupby`. Mitigated by a vectorised
searchsorted + cumulative-sum implementation rather than a row loop.

---

## ADR-009 — Off-policy evaluation with confidence intervals as the headline metric

**Status:** accepted · **Drives:** `prism/decision/policy_eval.py`

**Decision.** Report doubly-robust policy value with bootstrap CIs for every policy, including the
baselines, as the top-line result — not AUC, not Qini.

**Why.** Qini measures ranking; the business cares about the value of the decision actually taken.
And a point estimate without an interval invites the classic overclaim: a policy that "wins" by less
than its own standard error has not won. Bootstrapping the influence-function values rather than
refitting the model keeps this cheap enough to run for every policy.

**Rejected.** Reporting Qini alone (standard in uplift write-ups, and it hides whether the gap is
real).

---

## ADR-010 — Soft dependencies degrade, never crash

**Status:** accepted · **Drives:** `prism/utils/optional.py`

**Decision.** Only numpy/pandas/scipy/sklearn/statsmodels/matplotlib/pyarrow are hard requirements.
Everything else — torch, LightGBM, DuckDB, SHAP, MLflow, lifelines — is probed at import and
substituted (LightGBM → XGBoost → `HistGradientBoosting`; DuckDB → parquet; MLflow → a null tracker).

**Why.** The most common failure of a portfolio repo is that it does not run on the reviewer's
machine. Graceful degradation is also what makes the serving image small.

**Validated in practice:** MLflow 3.16 turned out to have a broken file store on Windows. The null
tracker absorbed it and the pipeline kept running, which is exactly the behaviour this ADR is for —
then the root cause was fixed by switching the backend to SQLite.

---

## ADR-011 — A performance contract alongside the interface contract

**Status:** accepted · **Drives:** `SPEC_PERF.md`

**Decision.** Specify runtime budgets and required implementation strategies (vectorised expansions,
binned split search, cached cross-fitting, chunked prediction) as a binding document, not as advice.

**Why.** Statistical correctness and computational tractability trade off constantly here, and the
tradeoff is easier to get right when it is written down before the code. It also makes "this is too
slow" a contract violation with a number attached rather than a matter of taste.

---

## ADR-013 — One persistent offer per customer, not a fresh draw every month

**Status:** accepted · **Drives:** `prism/data/dgp.py`

**Context.** The simulator originally re-drew each customer's offer independently every period —
a "repeated-decision panel". It looked more realistic and more general.

**The problem it caused.** The analytic ground truth is a *sustained-arm* contrast:
`S_a(k) = ∏_{j<k} (1 − h_a(t+j))`, i.e. arm `a` in force for the whole horizon. But if the realised
arm is re-drawn each month, a customer's actual trajectory reflects a random *sequence* of offers.
Those are **two different estimands**, and the sequence version's effect is diluted heavily toward
zero. The symptom was unmistakable once measured against truth: every estimator returned a CATE of
approximately **0.0** against a true effect of **+14 to +22**. Nothing was wrong with any estimator.
They were being graded against a quantity the data could not express.

**Decision.** Assign the offer once, at the customer's first panel period, and let it persist.

**Why this is also more realistic.** A discount stays on the account; a concierge relationship does
not lapse and get re-rolled every month. The one-shot campaign is the normal shape of retention
marketing.

**Effect of the change,** same seed, same config, measured against known truth:

| arm | true τ | model estimate *before* | model estimate *after* |
|---|---:|---:|---:|
| `discount_10` | +13.7 | +2.0 | **+8.9** |
| `discount_25` | +22.0 | +0.0 | **+12.0** |
| `concierge` | +20.0 | −1.1 | **+14.5** |

The causal forest went from *worse* than a constant-effect baseline to beating it (PEHE 58.2 vs
61.4), and rank correlation with the true effect went from −0.32 to +0.34.

**The lesson worth keeping:** this was not a modelling bug, a tuning problem or a sample-size
problem. It was an **estimand mismatch** — the thing being estimated and the thing being measured
were different quantities. It was only findable because the simulator knows the truth. On real data
the models would have returned near-zero effects and the honest conclusion would have been
"retention offers don't work", which would have been wrong.

---

## ADR-014 — The hidden confounder is calibrated, not just asserted

**Status:** accepted · **Drives:** `configs/default.yaml`, `configs/high_confounding.yaml`

**Context.** The simulator generates an unobserved confounder `U` that shifts both treatment
assignment and churn, so that unconfoundedness is false by construction and the sensitivity
analysis has something real to measure. I originally set its strength to `0.35` by eye.

**The problem.** At `0.35` the induced bias is *larger than the treatment effect it perturbs*. The
estimated CATE ranking does not merely degrade, it **inverts** (rank correlation −0.37). A benchmark
on which no correct method can succeed measures nothing.

**Decision.** Calibrate it. Sweep the strength, measure recovery against known truth, and pick a
value where the confounding is real but the problem stays identifiable.

| `U` strength | \|ATE error\| | rank corr. with truth | |
|---:|---:|---:|---|
| 0.00 | 3.04 | +0.37 | no confounding at all |
| **0.05** | **2.29** | **+0.42** | **chosen default** |
| 0.10 | 3.32 | +0.34 | still workable |
| 0.20 | 7.03 | −0.36 | ranking inverts |
| 0.35 | 8.29 | −0.37 | original guess |

**Rejected.** (a) Leaving it at 0.35 — the headline claim would have been false.
(b) Setting it to 0.0 — removes the confounding the refutation suite exists to detect, and makes the
observational problem trivially easy.

**And the failure is shipped, not hidden.** `configs/high_confounding.yaml` reproduces the broken
regime on purpose, so a reader can run it and watch the method fail. Knowing where a method breaks
is worth more than a benchmark tuned until it always wins.

---

## ADR-012 — Fairness audit included, not bolted on

**Status:** accepted · **Drives:** `prism/decision/fairness.py`

**Decision.** Report per-group treat rates, demographic-parity and equal-opportunity gaps and the
disparate-impact ratio for every policy, and provide a constrained re-allocation with per-group
floors.

**Why.** A profit-maximising allocation of a scarce benefit will concentrate it, and "the optimiser
did it" is not a defence. Providing the constrained variant turns the conversation from an argument
into a priced tradeoff: the reports show exactly how much incremental value parity costs.

**Cost.** The fair policy is worth less. Showing that number is the point.
