# PRISM — Performance & Implementation Addendum (binding)

Read this together with `SPEC.md`. Where the two appear to conflict, `SPEC.md` defines *what* the
interface is and this file defines *how* it must be implemented to be fast enough to actually run.

A portfolio project that takes four hours to produce one number is a failed portfolio project. Every
directive below exists because the naive implementation is too slow at the target data size.

---

## 0. Target data sizes

| stage | rows | notes |
|---|---|---|
| gold panel, `configs/default.yaml` | ~600k–1.1M customer-periods | 60k customers x 24 periods, shrinking as customers churn |
| gold panel, `configs/fast.yaml` | ~60k–90k | used by CI, tests and the first local run |
| **CATE training sample** | **<= 120k rows** | see section 1 |
| survival person-period expansion | rows x horizon | expand lazily; never materialise the full 1M x 12 as float64 unless it fits |

**Hard budget:** `python -m prism.pipelines.run_all --config configs/fast.yaml --steps all`
must finish in **under 6 minutes** on 8 CPU cores. The `default.yaml` run must finish in
**under 30 minutes**. Every module's `__main__` smoke test must finish in **under 60 seconds**.

---

## 1. The CATE training sample (important — read before building any causal module)

The panel has one row per customer per month. Those rows are **not independent**: the same customer
appears up to 24 times, and their 12-month outcome windows overlap heavily. Training a CATE model on
all of them would (a) be slow and (b) badly understate variance.

Therefore the pipeline builds a **decision-point sample**:

- Sample **at most `max_decision_points` rows per customer** (default 1, configurable), chosen at a
  random eligible period with a fixed seed.
- Cap the total at `max_train_rows` (default 120_000).
- **Never split a customer across train/valid/test.** Splitting is temporal by period *and* grouped
  by `customer_id`; a customer that appears in train must not appear in test.

Helper to implement in `prism/data/features.py` if it is not already there (additive, does not break
the existing contract):

```python
def decision_point_sample(panel, *, max_per_customer=1, max_rows=120_000,
                          random_state=None) -> pd.DataFrame
```

All causal modules must be written assuming `n <= ~120k` and `p <= ~60` after one-hot encoding.

---

## 2. `CausalForest` — do NOT hand-roll a Python tree builder

A pure-Python recursive tree that scans every feature x every threshold is O(n·p·depth) *in the
Python interpreter* and will take hours. Build the forest on top of scikit-learn's C-implemented
trees using the **R-learner / local-centering** construction, which is both fast and theoretically
grounded (it is the estimating equation `grf` uses once outcomes are locally centered).

### Required construction

1. **Local centering (once, cross-fitted, outside the tree loop).**
   ```
   m_hat(x) = E[Y | X]        # out-of-fold, gradient boosted
   e_hat(x) = P(W = 1 | X)    # out-of-fold, gradient boosted, clipped to [0.01, 0.99]
   Y_res = Y - m_hat(X)
   W_res = W - e_hat(X)
   ```

2. **Per tree** (`n_estimators` of them, embarrassingly parallel via `joblib.Parallel`):
   - draw a **subsample without replacement** of size `subsample_fraction * n` (not a bootstrap —
     the GRF asymptotics require subsampling)
   - split that subsample into a **split half** and an **estimate half** by `honest_fraction`
   - fit a `sklearn.tree.DecisionTreeRegressor` on the *split half* with
     - `y = Y_res / W_res_safe` where `W_res_safe = sign(W_res) * maximum(|W_res|, eps)`
     - `sample_weight = W_res**2`
     - `max_features = mtry`, `min_samples_leaf`, `max_depth` as configured
     This weighted least-squares problem is exactly the R-learner objective
     `min sum (Y_res - tau(x) * W_res)^2`, so the tree partitions on treatment-effect
     heterogeneity, not on outcome level.
   - **honest re-estimation**: push the *estimate half* down the fitted tree with
     `tree.apply(X_est)` and set each leaf's value to the ratio-of-sums
     ```
     tau_leaf = sum(W_res * Y_res) / sum(W_res ** 2)      over the estimate-half rows in that leaf
     ```
     Never divide per-row; always take the ratio of sums (this is the numerically stable and
     statistically correct form).
   - leaves with fewer than `min_treated_per_leaf` treated or control units in the estimate half
     fall back to the parent's estimate; if that fails too, fall back to the global ATE.

3. **Prediction**: `tau_hat(x) = mean over trees of that tree's honest leaf value`.

4. **`predict_interval`**: bootstrap-of-little-bags — partition the trees into `sqrt(n_estimators)`
   groups, compute the between-group variance of the group means, and form a normal interval.
   Document that this is an approximation.

5. **`feature_importances_`**: average the sklearn trees' impurity importances, which on the
   residualised problem read as *heterogeneity* importances. Say so in the docstring.

**Performance target:** 400 trees on 100k x 50 in **under 90 seconds** on 8 cores.

For `K` arms, fit one such forest per non-control arm using only rows with `w in {0, a}`.

---

## 3. `RandomSurvivalForestLite` — bin first, or use random splits

A log-rank split search evaluated in Python over raw continuous features is the same trap. Pick one
of these two, and document which:

- **(A) Pre-binned log-rank search (preferred).** Quantile-bin every feature to <= 32 bins once
  (`np.quantile` + `np.searchsorted`). Then a split search is a set of cumulative sums over bins,
  fully vectorised with numpy — O(p · bins) per node with no Python inner loop.
- **(B) Extremely-randomised splits.** Draw `mtry` features and one random threshold each, score the
  candidates with a vectorised log-rank statistic, keep the best. This is a legitimate and published
  variant and is much faster.

Leaves store a **Nelson–Aalen** cumulative hazard; `predict_survival` is `exp(-H(t))`.

**Performance target:** 200 trees on 50k rows in **under 60 seconds**. If the input is larger,
subsample internally and say so in a logged warning.

---

## 4. `DiscreteTimeHazardModel` — person-period expansion must be vectorised

Build the expansion with `numpy.repeat` / `numpy.tile`, never a Python loop over customers:

```python
n_rows_i = minimum(ceil(event_time), horizon)          # periods each customer contributes
idx      = np.repeat(np.arange(n), n_rows_i)           # row index into X
period   = <ragged arange built with cumsum trickery>  # 0,1,2,... within each customer
event    = <1 only on the last period of an observed churn>
```

Cap the expansion at `horizon` periods per customer. For the torch backend, feed `X[idx]` as a view
and one-hot (or embed) `period`; do not materialise a densely duplicated copy of `X` if it exceeds
~2 GB — chunk instead.

Both backends must produce `predict_survival` that is **monotone non-increasing along axis 1** and
strictly inside `(0, 1)`. Assert this in the smoke test.

**Performance target:** fit on 100k customers x horizon 12 in **under 60 seconds** (sklearn backend)
or **under 120 seconds** (torch backend, CPU).

---

## 5. `CoxPHModel` — vectorised Efron partial likelihood

Sort by time once. Compute risk-set sums with a reversed `cumsum`, and handle ties with the Efron
correction computed on the tie block, all in numpy. Use `scipy.optimize.minimize(method="L-BFGS-B")`
with an analytic gradient. No Python loop over individuals; a loop over *distinct event times* is
acceptable only if the number of distinct times is small (it is — <= horizon here), otherwise
vectorise over tie blocks with `np.add.reduceat`.

---

## 6. Meta-learners — cross-fitting without refitting the world

`DRLearner` and `RLearner` need out-of-fold nuisances. Fit each nuisance **once per fold**, cache the
out-of-fold predictions in an array, and reuse them for every arm rather than refitting per arm.
With `K = 4` arms and `n_splits = 5` a naive implementation fits 40+ models; the cached version fits
`n_splits * (K + 1)`.

Use `sklearn.model_selection.StratifiedKFold` on the arm label so every fold contains every arm.
If a fold would contain zero rows of some arm, reduce `n_splits` automatically and log a warning.

---

## 7. `CausalSurvivalUplift` — share the hazard fits

Per non-control arm the naive approach refits the control hazard model every time. Fit the control
arm's hazard model **once** and reuse it. The per-arm fits are independent, so run them with
`joblib.Parallel(n_jobs=min(n_arms, 4))`.

Counterfactual survival curves are `(n, n_arms, horizon)` float64. At n = 120k, K = 4, H = 12 that
is ~46 MB — fine. At n = 1M it is 384 MB — so `predict_survival_curves` must accept a `chunk_size`
argument (default 50_000) and process in chunks, and the pipeline must call it on the decision-point
sample, not the full panel.

---

## 8. Bootstrap and refutation — cap the work and say so

- `policy_value(..., n_boot=500)`: bootstrap the **influence-function values**, not the model. Never
  refit a model inside a bootstrap loop.
- Refutation simulations DO refit, so they are expensive. Default `n_sim` must be small (<= 20), the
  estimator used inside refutation must be a **cheap** one (a T-learner with 100 trees, not the full
  forest), and the function must `log()` exactly how many simulations it ran.
- Any function that internally subsamples or truncates MUST log what it dropped. Silent truncation
  that reads as full coverage is the one unforgivable sin here.

---

## 9. General numpy discipline

- Prefer `float32` for large intermediate matrices where precision permits; return `float64`.
- Use `np.errstate` guards around divisions; never let a `RuntimeWarning: invalid value` reach a user.
- Clip probabilities to `[1e-6, 1 - 1e-6]` before any `log`.
- `joblib.Parallel(n_jobs=-1, prefer="threads")` for numpy-heavy work that releases the GIL;
  `prefer="processes"` only when the work is genuinely Python-bound (and remember Windows spawns,
  so anything parallel must be importable at module top level, not a closure).
- Set `n_jobs` defaults to `-1` but honour an explicit value; CI runs with 2 cores.

---

## 10. Reproducibility

Every parallel path must be seeded with `prism.utils.seeds.spawn_rngs(random_state, n_workers)` so
results do not depend on thread scheduling. A test asserts that two runs with the same seed give
bit-identical CATE predictions. If an estimator cannot be made deterministic, it must say so in its
docstring and be excluded from that test by name.
