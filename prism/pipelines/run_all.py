"""End-to-end PRISM pipeline.

    python -m prism.pipelines.run_all --config configs/fast.yaml --steps all

Nine steps, each resumable and individually selectable:

===========  ==========================================================================
``simulate`` generate the ground-truth panel, event log and counterfactuals
``ingest``   inject realism, clean it back, validate the contract, load the warehouse
``features`` temporal split, leakage check, decision-point sample, design matrix
``train``    survival, CLV and multi-arm propensity models with overlap diagnostics
``causal``   fit every configured CATE learner, including Causal Survival Uplift
``evaluate`` PEHE / eps_ATE against known truth, Qini, GATES, BLP, refutation suite
``optimize`` unit economics, budget-constrained assignment, off-policy value, fairness
``monitor``  covariate drift, CATE stability, alerts
``report``   metrics.json, CSV tables, 12 figures, model card, decision memo, bundle
===========  ==========================================================================

Design: every step is wrapped so a failure is logged, recorded in the run report and the
pipeline continues to whatever is still computable. A partial run that tells you which step
broke is more useful than a traceback with no artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.config import PrismConfig, load_config
from prism.data.schema import (
    ARM_NAMES,
    CATEGORICAL_FEATURES,
    FEATURES,
    N_ARMS,
    NUMERIC_FEATURES,
)
from prism.utils.logging import get_logger, set_level, timed
from prism.utils.seeds import seed_everything
from prism.utils.tracking import get_tracker

log = get_logger("pipeline")

STEPS = ("simulate", "ingest", "features", "train", "causal", "evaluate", "optimize", "monitor", "report")

__all__ = ["main", "cli", "Pipeline", "STEPS"]


def _jsonable(obj: Any) -> Any:
    """Coerce numpy/pandas scalars into JSON-safe values; non-finite floats become None."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


@dataclass
class StepResult:
    """Outcome of one pipeline step."""

    name: str
    ok: bool
    seconds: float
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Holds the state that flows between steps."""

    def __init__(self, config: PrismConfig) -> None:
        self.cfg = config
        self.metrics: dict[str, Any] = {}
        self.tables: dict[str, pd.DataFrame] = {}
        self.figures_data: dict[str, Any] = {}
        self.results: list[StepResult] = []
        self.state: dict[str, Any] = {}
        self.tracker = get_tracker(
            experiment_name=config.experiment_name,
            tracking_uri=config.paths.path("mlruns"),
            reports_dir=config.paths.path("reports"),
        )

    # -- plumbing ----------------------------------------------------------------------

    def run_step(self, name: str, fn: Callable[[], dict[str, Any] | None]) -> StepResult:
        """Execute one step, catching and recording any failure."""
        t0 = time.perf_counter()
        log.info("=" * 68)
        log.info("STEP %s", name.upper())
        log.info("=" * 68)
        try:
            detail = fn() or {}
            res = StepResult(name, True, time.perf_counter() - t0, detail=detail)
            log.info("step %s OK in %.1fs", name, res.seconds)
        except Exception as exc:
            tb = traceback.format_exc()
            res = StepResult(name, False, time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}")
            log.error("step %s FAILED after %.1fs: %s", name, res.seconds, exc)
            log.debug(tb)
            (self.cfg.paths.path("reports") / f"error_{name}.txt").write_text(tb, encoding="utf-8")
        self.results.append(res)
        self.metrics[f"step_{name}_seconds"] = round(res.seconds, 2)
        self.metrics[f"step_{name}_ok"] = res.ok
        return res

    def need(self, *keys: str) -> tuple:
        """Fetch required state, raising a clear message if an upstream step did not run."""
        missing = [k for k in keys if k not in self.state]
        if missing:
            raise RuntimeError(f"missing pipeline state {missing}; an earlier step did not complete")
        return tuple(self.state[k] for k in keys)

    # =================================================================================
    # 1. simulate
    # =================================================================================

    def step_simulate(self) -> dict[str, Any]:
        """Generate the ground-truth panel, event log and analytic counterfactuals."""
        from prism.data.dgp import simulate

        dgp_cfg = self.cfg.dgp_config()
        with timed(log, f"simulating {dgp_cfg.n_customers:,} customers x {dgp_cfg.n_periods} periods"):
            sim = simulate(dgp_cfg)

        datadir = self.cfg.paths.path("data") / "sim"
        sim.save(datadir)

        self.state["sim"] = sim
        self.state["panel_clean"] = sim.panel
        self.state["ground_truth"] = sim.ground_truth

        self.metrics.update(
            {
                "n_customers": int(sim.customers.shape[0]),
                "n_rows_panel": int(sim.panel.shape[0]),
                "n_events": int(sim.events.shape[0]),
                "n_periods": int(dgp_cfg.n_periods),
                "horizon": int(dgp_cfg.horizon),
                "churn_rate": float(sim.panel["churn_next"].mean()),
                "censoring_rate": float(1.0 - sim.panel["event_observed"].mean()),
                "treated_share": float((sim.panel["arm"] > 0).mean()),
                "rct_share": float((sim.panel["assignment_block"] == "rct").mean()),
            }
        )

        # the economic structure the whole project rests on
        merged = sim.panel.merge(sim.ground_truth, on=["customer_id", "period"], how="inner")
        tau_cols = [f"gt_tau_value_{a}" for a in range(1, N_ARMS)]
        seg = merged.groupby("gt_segment")[tau_cols].mean()
        seg.columns = [ARM_NAMES[int(c.rsplit("_", 1)[1])] for c in seg.columns]
        self.tables["segment_effects"] = seg.reset_index()
        self.figures_data["segment_table"] = seg
        log.info("mean effect on discounted value, by responder segment:\n%s", seg.round(2).to_string())
        return {"rows": int(sim.panel.shape[0]), "segments": seg.index.tolist()}

    # =================================================================================
    # 2. ingest
    # =================================================================================

    def step_ingest(self) -> dict[str, Any]:
        """Dirty the data the way the real world does, clean it back, and prove it is clean."""
        from prism.data.messy import clean_panel, inject_realism
        from prism.data.warehouse import Warehouse
        from prism.utils.validation import PANEL_CONTRACT

        (sim,) = self.need("sim")
        dgp_cfg = self.cfg.dgp_config()

        with timed(log, "injecting realism"):
            dirty = inject_realism(sim.panel, dgp_cfg, random_state=self.cfg.seed)
        with timed(log, "cleaning"):
            clean, report = clean_panel(dirty)

        self.tables["cleaning_report"] = report
        result = PANEL_CONTRACT.validate(clean)
        self.metrics["contract_passed"] = bool(result.passed)
        self.metrics["contract_errors"] = int(result.n_errors)
        self.metrics["contract_warnings"] = int(result.n_warnings)
        if not result.passed:
            log.warning("cleaned panel failed its contract:\n%s", result.summary().to_string(index=False))
            log.warning("falling back to the pre-realism panel so the run can continue")
            clean = sim.panel.copy()
            self.metrics["ingest_fell_back_to_clean_panel"] = True

        rows_lost = len(sim.panel) - len(clean)
        self.metrics["rows_dropped_in_cleaning"] = int(rows_lost)
        log.info("cleaning dropped %d rows (%.2f%%)", rows_lost, 100 * rows_lost / max(len(sim.panel), 1))

        self.state["panel_clean"] = clean

        try:
            wh = Warehouse(self.cfg.paths.path("warehouse"))
            wh.write("bronze", "events", sim.events)
            wh.write("bronze", "customers", sim.customers)
            wh.write("silver", "panel_dirty", dirty.head(50_000))
            wh.write("gold", "panel", clean)
            wh.write("gold", "ground_truth", sim.ground_truth)
            self.tables["lineage"] = wh.lineage()
            wh.close()
            self.metrics["warehouse_tables"] = int(len(self.tables["lineage"]))
        except Exception as exc:  # the warehouse is a convenience, not a dependency
            log.warning("warehouse load skipped: %s", exc)
            self.metrics["warehouse_tables"] = 0

        return {"rows_clean": int(len(clean)), "contract_passed": bool(result.passed)}

    # =================================================================================
    # 3. features
    # =================================================================================

    def step_features(self) -> dict[str, Any]:
        """Split temporally, prove there is no leakage, and build the design matrix."""
        from prism.data.features import build_design_matrix, temporal_split

        (panel,) = self.need("panel_clean")
        sim = self.state.get("sim")

        panel = temporal_split(panel, self.cfg.split.train_end, self.cfg.split.valid_end)
        by_period = panel.groupby("period")["split"].nunique()
        if not (by_period == 1).all():
            raise RuntimeError("temporal_split produced periods with mixed splits -- that is a leak")

        counts = panel["split"].value_counts().to_dict()
        log.info("split sizes: %s", counts)
        for name in ("train", "test"):
            if counts.get(name, 0) == 0:
                raise RuntimeError(f"the {name} split is empty; check split.train_end / split.valid_end")

        # leakage guard on the point-in-time feature store
        leak = False
        try:
            from prism.data.features import PointInTimeFeatureStore, default_feature_specs

            spine = panel[["customer_id", "as_of_date"]].drop_duplicates().head(3000)
            store = PointInTimeFeatureStore(sim.events, default_feature_specs())
            rep = store.leakage_report(spine)
            leak = bool(rep["leak_detected"].any()) if "leak_detected" in rep else False
            self.tables["leakage_report"] = rep
        except Exception as exc:
            log.warning("leakage report skipped: %s", exc)
        self.metrics["leakage_detected"] = bool(leak)
        if leak:
            log.error("POINT-IN-TIME LEAKAGE DETECTED -- results are not trustworthy")

        # one decision point per customer: monthly rows are not independent
        sample = self._decision_point_sample(panel)
        log.info("decision-point sample: %d rows from %d panel rows", len(sample), len(panel))

        X_all, names, encoder = build_design_matrix(sample, fit=True)
        self.state.update(
            {
                "panel": panel,
                "sample": sample,
                "X": np.asarray(X_all, dtype=float),
                "feature_names": list(names),
                "encoder": encoder,
            }
        )
        self.metrics.update(
            {
                "n_train_rows": int(counts.get("train", 0)),
                "n_valid_rows": int(counts.get("valid", 0)),
                "n_test_rows": int(counts.get("test", 0)),
                "n_decision_points": int(len(sample)),
                "n_design_columns": int(X_all.shape[1]),
            }
        )
        return {"design_shape": list(np.shape(X_all))}

    def _decision_point_sample(self, panel: pd.DataFrame) -> pd.DataFrame:
        """One row per customer per split, capped, merged with ground truth.

        Monthly rows for the same customer share overlapping outcome windows, so treating them
        as independent both slows training and understates variance.
        """
        try:
            from prism.data.features import decision_point_sample

            sample = decision_point_sample(
                panel, max_per_customer=1, max_rows=120_000, random_state=self.cfg.seed
            )
        except Exception:
            rng = np.random.default_rng(self.cfg.seed)
            picks = (
                panel.reset_index()
                .groupby(["customer_id", "split"], sort=False)["index"]
                .apply(lambda s: s.iloc[rng.integers(0, len(s))])
                .to_numpy()
            )
            sample = panel.loc[picks]
            if len(sample) > 120_000:
                sample = sample.sample(120_000, random_state=self.cfg.seed)
        sample = sample.sort_values(["period", "customer_id"]).reset_index(drop=True)

        gt = self.state.get("ground_truth")
        if gt is not None:
            sample = sample.merge(gt, on=["customer_id", "period"], how="left")
        return sample

    # =================================================================================
    # 4. train
    # =================================================================================

    def step_train(self) -> dict[str, Any]:
        """Survival, CLV and multi-arm propensity, with balance diagnostics."""
        from prism.models.propensity import (
            PropensityModel,
            effective_sample_size,
            overlap_diagnostics,
            standardized_mean_differences,
        )
        from prism.models.survival import DiscreteTimeHazardModel, concordance_index

        sample, X, names = self.need("sample", "X", "feature_names")
        horizon = int(self.cfg.sim.horizon)
        is_train = (sample["split"] == "train").to_numpy()
        if is_train.sum() < 50:
            is_train = np.ones(len(sample), dtype=bool)
            log.warning("train split too small; fitting nuisances on the full sample")

        et = sample["event_time"].to_numpy(dtype=float)
        eo = sample["event_observed"].to_numpy(dtype=int)
        arm = sample["arm"].to_numpy(dtype=int)

        # -- survival --------------------------------------------------------------
        with timed(log, "fitting the discrete-time hazard model"):
            surv = DiscreteTimeHazardModel(
                horizon=horizon,
                hidden=tuple(self.cfg.model.survival_hidden),
                epochs=int(self.cfg.model.survival_epochs),
                lr=float(self.cfg.model.survival_lr),
                backend=self.cfg.model.survival_backend,
                random_state=self.cfg.seed,
            ).fit(X[is_train], et[is_train], eo[is_train])

        S = np.asarray(surv.predict_survival(X), dtype=float)
        if S.ndim == 2 and S.shape[1] > 1:
            drops = np.diff(S, axis=1)
            if (drops > 1e-8).any():
                log.warning("survival curves are not monotone non-increasing; clipping")
                S = np.minimum.accumulate(S, axis=1)
        rmst = np.asarray(surv.predict_rmst(X), dtype=float)
        churn_risk = np.asarray(surv.predict_churn_prob(X), dtype=float)

        test = ~is_train
        if test.sum() > 30 and eo[test].sum() > 5:
            cidx = float(concordance_index(et[test], churn_risk[test], eo[test]))
        else:
            cidx = float(concordance_index(et, churn_risk, eo))
        self.metrics["survival_c_index"] = cidx
        log.info("survival C-index = %.4f", cidx)

        # -- CLV -------------------------------------------------------------------
        margin_rate = sample.get("margin_rate")
        margin_rate = (
            margin_rate.to_numpy(dtype=float) if margin_rate is not None
            else np.full(len(sample), self.cfg.economics.margin_rate)
        )
        aov = pd.to_numeric(sample.get("avg_order_value"), errors="coerce").to_numpy(dtype=float)
        aov = np.nan_to_num(aov, nan=float(np.nanmedian(aov)) if np.isfinite(aov).any() else 50.0)
        monthly_margin = aov * margin_rate

        try:
            from prism.models.clv import clv_from_survival

            clv = np.asarray(
                clv_from_survival(S, monthly_margin, discount_rate=self.cfg.economics.discount_rate),
                dtype=float,
            )
        except Exception as exc:
            log.warning("clv_from_survival failed (%s); using an undiscounted approximation", exc)
            clv = rmst * monthly_margin
        self.metrics["mean_clv"] = float(np.nanmean(clv))

        # -- propensity ------------------------------------------------------------
        with timed(log, "fitting the multi-arm propensity model"):
            prop_model = PropensityModel(
                n_arms=N_ARMS,
                n_splits=int(self.cfg.model.propensity_splits),
                clip=tuple(self.cfg.model.propensity_clip),
                random_state=self.cfg.seed,
            ).fit(X, arm)
        e_hat = np.asarray(prop_model.predict_proba(X), dtype=float)
        try:
            oof = np.asarray(prop_model.oof_proba(), dtype=float)
            if oof.shape == e_hat.shape:
                e_hat = oof  # cross-fitted scores for the rows the model was trained on
        except Exception:
            pass
        e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)
        e_hat = e_hat / e_hat.sum(axis=1, keepdims=True)

        self.tables["overlap"] = overlap_diagnostics(e_hat, arm)
        weights = 1.0 / e_hat[np.arange(len(arm)), arm]
        smd = standardized_mean_differences(X, arm, weights=weights, feature_names=names)
        self.tables["balance"] = smd
        self.figures_data["smd"] = smd
        self.metrics["effective_sample_size"] = float(effective_sample_size(weights))
        self.metrics["ess_fraction"] = float(self.metrics["effective_sample_size"] / max(len(arm), 1))

        for col in ("abs_smd_weighted", "smd_weighted", "abs_smd_unweighted", "smd_unweighted"):
            if col in smd.columns:
                self.metrics[f"mean_{col}"] = float(smd[col].abs().mean())

        if "gt_propensity_1" in sample.columns:
            true_p = sample[[f"gt_propensity_{a}" for a in range(N_ARMS)]].to_numpy(dtype=float)
            ok = np.isfinite(true_p).all(axis=1)
            if ok.sum() > 30:
                corr = float(np.corrcoef(true_p[ok, 1:].ravel(), e_hat[ok, 1:].ravel())[0, 1])
                self.metrics["propensity_corr_with_truth"] = corr
                log.info("estimated propensity correlates %.3f with the true propensity", corr)

        self.state.update(
            {
                "survival_model": surv,
                "survival_curves": S,
                "rmst": rmst,
                "churn_risk": churn_risk,
                "clv": clv,
                "monthly_margin": monthly_margin,
                "propensity": e_hat,
                "propensity_model": prop_model,
                "is_train": is_train,
            }
        )
        self.figures_data["survival_curves"] = S[: min(len(S), 5000)]
        return {"c_index": cidx}

    # =================================================================================
    # 5. causal
    # =================================================================================

    def step_causal(self) -> dict[str, Any]:
        """Fit every configured CATE learner and keep their test-set predictions."""
        sample, X, e_hat, is_train = self.need("sample", "X", "propensity", "is_train")

        arm = sample["arm"].to_numpy(dtype=int)
        # Train on MARGIN, not revenue. `value_h` is discounted revenue; the ground-truth
        # effects `gt_tau_value_*` and the offer costs are both denominated in margin. Fitting
        # on revenue would inflate every estimated effect by 1/margin_rate (about 3x here),
        # which would both wreck PEHE and make every offer look like it pays for itself.
        margin_rate = (
            sample["margin_rate"].to_numpy(dtype=float)
            if "margin_rate" in sample.columns
            else np.full(len(sample), self.cfg.economics.margin_rate)
        )
        y = sample["value_h"].to_numpy(dtype=float) * margin_rate
        et = sample["event_time"].to_numpy(dtype=float)
        eo = sample["event_observed"].to_numpy(dtype=int)
        margin = self.state.get("monthly_margin")
        self.state["margin_rate"] = margin_rate

        tr = is_train
        cate: dict[str, np.ndarray] = {}
        fitted: dict[str, Any] = {}
        timings: dict[str, float] = {}

        for spec in self.cfg.causal.learners:
            t0 = time.perf_counter()
            try:
                if spec == "survival":
                    from prism.causal.survival_uplift import CausalSurvivalUplift

                    model = CausalSurvivalUplift(
                        horizon=int(self.cfg.sim.horizon),
                        n_arms=N_ARMS,
                        discount_rate=float(self.cfg.economics.discount_rate),
                        doubly_robust=bool(self.cfg.causal.doubly_robust),
                        n_splits=int(self.cfg.causal.n_splits),
                        random_state=self.cfg.seed,
                    )
                    model.fit(
                        X[tr], arm[tr], et[tr], eo[tr],
                        margin=(margin[tr] if margin is not None else None),
                        propensity=e_hat[tr],
                    )
                    pred = np.asarray(model.predict_value_cate(X, margin=margin), dtype=float)
                elif spec == "forest":
                    from prism.causal.forest import CausalForest

                    model = CausalForest(
                        n_estimators=int(self.cfg.causal.forest_trees),
                        min_samples_leaf=int(self.cfg.causal.forest_min_leaf),
                        honest_fraction=float(self.cfg.causal.honest_fraction),
                        random_state=self.cfg.seed,
                    )
                    model.fit(X[tr], arm[tr], y[tr], propensity=e_hat[tr])
                    pred = np.asarray(model.predict_cate(X), dtype=float)
                else:
                    from prism.causal.learners import make_learner

                    model = make_learner(
                        spec, n_splits=int(self.cfg.causal.n_splits), random_state=self.cfg.seed
                    )
                    model.fit(X[tr], arm[tr], y[tr], propensity=e_hat[tr])
                    pred = np.asarray(model.predict_cate(X), dtype=float)

                pred = pred.reshape(len(X), -1)
                if pred.shape[1] != N_ARMS - 1:
                    fixed = np.zeros((len(X), N_ARMS - 1))
                    fixed[:, : min(pred.shape[1], N_ARMS - 1)] = pred[:, : N_ARMS - 1]
                    pred = fixed
                if not np.isfinite(pred).all():
                    log.warning("learner %s produced non-finite CATEs; zero-filling", spec)
                    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

                cate[spec] = pred
                fitted[spec] = model
                timings[spec] = time.perf_counter() - t0
                log.info("learner %-9s fitted in %6.1fs | mean ATE %+.3f",
                         spec, timings[spec], float(pred.mean()))
            except Exception as exc:
                log.error("learner %s failed: %s", spec, exc)
                log.debug(traceback.format_exc())

        if not cate:
            raise RuntimeError("every CATE learner failed; cannot continue")

        self.state.update({"cate": cate, "cate_models": fitted, "cate_timings": timings,
                           "y": y, "arm": arm, "event_time": et, "event_observed": eo})
        self.metrics["learners_fitted"] = sorted(cate)
        self.metrics["n_learners_fitted"] = len(cate)
        return {"learners": sorted(cate)}

    # =================================================================================
    # 6. evaluate
    # =================================================================================

    def step_evaluate(self) -> dict[str, Any]:
        """Score every learner against known truth and against the data, then try to break it."""
        from prism.causal.evaluate import auuc, eps_ate, pehe, qini_score, uplift_by_decile

        sample, cate, y, arm, is_train = self.need("sample", "cate", "y", "arm", "is_train")
        te = ~is_train
        if te.sum() < 50:
            te = np.ones(len(sample), dtype=bool)
            log.warning("test split too small; evaluating in-sample (this flatters the numbers)")

        gt_cols = [f"gt_tau_value_{a}" for a in range(1, N_ARMS)]
        has_gt = all(c in sample.columns for c in gt_cols)
        true_tau = sample[gt_cols].to_numpy(dtype=float) if has_gt else None
        if true_tau is not None and not np.isfinite(true_tau).all():
            true_tau = np.nan_to_num(true_tau, nan=0.0)

        w_bin = (arm > 0).astype(int)
        rows: list[dict[str, Any]] = []
        for name, pred in cate.items():
            row: dict[str, Any] = {"learner": name, "seconds": round(self.state["cate_timings"].get(name, 0.0), 1)}
            if true_tau is not None:
                row["pehe"] = float(pehe(true_tau[te], pred[te]))
                row["eps_ate"] = float(eps_ate(true_tau[te], pred[te]))
                row["rank_corr_vs_truth"] = float(
                    pd.Series(pred[te].max(axis=1)).corr(
                        pd.Series(true_tau[te].max(axis=1)), method="spearman"
                    )
                )
            try:
                score = pred[te].max(axis=1)
                row["qini"] = float(qini_score(y[te], w_bin[te], score))
                row["auuc"] = float(auuc(y[te], w_bin[te], score))
            except Exception as exc:
                log.debug("uplift metrics failed for %s: %s", name, exc)
            rows.append(row)

        leaderboard = pd.DataFrame(rows)
        sort_key = "pehe" if "pehe" in leaderboard else ("qini" if "qini" in leaderboard else "learner")
        leaderboard = leaderboard.sort_values(sort_key, ascending=sort_key == "pehe").reset_index(drop=True)
        self.tables["leaderboard"] = leaderboard
        self.figures_data["leaderboard"] = leaderboard
        log.info("estimator leaderboard:\n%s", leaderboard.round(4).to_string(index=False))

        best_name = str(leaderboard.iloc[0]["learner"])
        self.state["best_learner"] = best_name
        self.state["best_cate"] = cate[best_name]
        self.metrics["best_learner"] = best_name
        for col in ("pehe", "eps_ate", "qini", "auuc", "rank_corr_vs_truth"):
            if col in leaderboard.columns:
                v = leaderboard.iloc[0][col]
                key = "cate_rank_correlation_vs_truth" if col == "rank_corr_vs_truth" else f"best_learner_{col}"
                self.metrics[key] = float(v) if pd.notna(v) else None

        if true_tau is not None:
            const = np.tile(true_tau[te].mean(axis=0), (te.sum(), 1))
            self.metrics["baseline_constant_pehe"] = float(pehe(true_tau[te], const))
            log.info(
                "best PEHE %.3f vs constant-effect baseline %.3f",
                self.metrics.get("best_learner_pehe", float("nan")),
                self.metrics["baseline_constant_pehe"],
            )

        best = self.state["best_cate"]
        try:
            dec = uplift_by_decile(y[te], w_bin[te], best[te].max(axis=1))
            self.tables["uplift_deciles"] = dec
            self.figures_data["deciles"] = dec
        except Exception as exc:
            log.warning("decile table failed: %s", exc)

        try:
            from prism.causal.evaluate import qini_curve

            self.figures_data["qini"] = qini_curve(y[te], w_bin[te], best[te].max(axis=1))
        except Exception as exc:
            log.warning("qini curve failed: %s", exc)

        e_hat = self.state["propensity"]
        for fn_name, key in (("gates", "gates"), ("blp_calibration", "blp")):
            try:
                import prism.causal.evaluate as ev

                fn = getattr(ev, fn_name)
                out = fn(y[te], w_bin[te], best[te].max(axis=1), e_hat[te][:, 1])
                self.tables[key] = out
                self.figures_data[key] = out
            except Exception as exc:
                log.warning("%s failed: %s", fn_name, exc)

        if "blp" in self.tables:
            blp = self.tables["blp"]
            try:
                slope = blp.loc[blp.iloc[:, 0].astype(str).str.contains("2|slope|het", case=False), :]
                if len(slope):
                    num = slope.select_dtypes("number")
                    if len(num.columns):
                        self.metrics["blp_calibration_slope"] = float(num.iloc[0, 0])
            except Exception:
                pass

        self._confounding_demo(te)

        if self.cfg.causal.refute:
            self._refute(te)
        return {"best_learner": best_name}

    def _confounding_demo(self, te: np.ndarray) -> None:
        """Show, per arm, that naive comparison is wrong and adjustment fixes it.

        This is the single most legible piece of evidence the pipeline produces. It puts four
        numbers side by side for each offer:

        * the **true** ATE, known because the simulator computes counterfactuals analytically;
        * the **naive** difference in means, which is what a dashboard would show;
        * the **IPW** estimate using the *estimated* propensity;
        * the **model** estimate from the best CATE learner.

        In this data the confounding is strong enough to flip the sign of the naive estimate,
        so the table doubles as the argument for why any of this machinery is needed.
        """
        try:
            sample, e_hat, best = self.state["sample"], self.state["propensity"], self.state["best_cate"]
            y, arm = self.state["y"], self.state["arm"]
            gt_cols = [f"gt_tau_value_{a}" for a in range(1, N_ARMS)]
            has_gt = all(c in sample.columns for c in gt_cols)

            rows = []
            for a in range(1, N_ARMS):
                treated = te & (arm == a)
                control = te & (arm == 0)
                if treated.sum() < 20 or control.sum() < 20:
                    continue
                naive = float(y[treated].mean() - y[control].mean())

                # Horvitz-Thompson contrast with the estimated propensity
                w_t = np.where(treated, 1.0 / np.clip(e_hat[:, a], 1e-3, 1), 0.0)
                w_c = np.where(control, 1.0 / np.clip(e_hat[:, 0], 1e-3, 1), 0.0)
                ipw = float(((w_t - w_c) * y)[te].mean())

                row = {
                    "arm": a,
                    "arm_name": ARM_NAMES[a],
                    "n_treated": int(treated.sum()),
                    "naive_diff_in_means": naive,
                    "ipw_estimate": ipw,
                    "model_estimate": float(best[te, a - 1].mean()),
                }
                if has_gt:
                    truth = float(sample.loc[te, gt_cols[a - 1]].mean())
                    row["true_ate"] = truth
                    row["naive_error"] = naive - truth
                    row["model_error"] = row["model_estimate"] - truth
                    row["naive_sign_wrong"] = bool(np.sign(naive) != np.sign(truth) and abs(truth) > 1e-9)
                rows.append(row)

            if not rows:
                return
            demo = pd.DataFrame(rows)
            self.tables["confounding_bias"] = demo
            log.info("naive vs adjusted estimates:\n%s", demo.round(3).to_string(index=False))

            if has_gt:
                self.metrics["naive_mean_abs_error"] = float(demo["naive_error"].abs().mean())
                self.metrics["model_mean_abs_error"] = float(demo["model_error"].abs().mean())
                self.metrics["n_arms_naive_sign_wrong"] = int(demo["naive_sign_wrong"].sum())
                self.metrics["bias_reduction_vs_naive_pct"] = (
                    float(
                        100.0
                        * (1.0 - demo["model_error"].abs().mean() / max(demo["naive_error"].abs().mean(), 1e-9))
                    )
                )
                log.info(
                    "naive |error| %.2f -> model |error| %.2f (%.0f%% of the bias removed); "
                    "naive gets the SIGN wrong on %d of %d arms",
                    self.metrics["naive_mean_abs_error"],
                    self.metrics["model_mean_abs_error"],
                    self.metrics["bias_reduction_vs_naive_pct"],
                    self.metrics["n_arms_naive_sign_wrong"],
                    len(demo),
                )
        except Exception as exc:
            log.warning("confounding demonstration skipped: %s", exc)

    def _refute(self, te: np.ndarray) -> None:
        """Run the refutation suite with a cheap estimator (SPEC_PERF section 8)."""
        try:
            from prism.causal.learners import TLearner
            from prism.causal.refute import run_refutation_suite

            X, y, arm = self.state["X"], self.state["y"], self.state["arm"]
            idx = np.where(~te)[0]
            if len(idx) > 8000:
                idx = np.random.default_rng(self.cfg.seed).choice(idx, 8000, replace=False)
                log.info("refutation running on a %d-row subsample", len(idx))

            def factory():
                return TLearner(random_state=self.cfg.seed)

            with timed(log, "refutation suite"):
                suite = run_refutation_suite(
                    factory, X[idx], arm[idx], y[idx], random_state=self.cfg.seed
                )
            self.tables["refutation"] = suite
            if "passed" in suite.columns:
                self.metrics["refutation_passed"] = int(suite["passed"].sum())
                self.metrics["refutation_total"] = int(len(suite))
            log.info("refutation:\n%s", suite.to_string(index=False))
        except Exception as exc:
            log.warning("refutation suite skipped: %s", exc)

        try:
            from prism.causal.refute import e_value

            best = self.state["best_cate"].max(axis=1)
            ate = float(np.mean(best))
            se = float(np.std(best) / np.sqrt(len(best)))
            # The ATE here is on the DIFFERENCE scale (discounted currency), not a risk ratio.
            # e_value() warns loudly that an unscaled difference produces a meaningless number,
            # so hand it the outcome's standard deviation and let it do the standardised-mean
            # -> approximate-risk-ratio conversion properly.
            ev = e_value(
                ate,
                ate - 1.96 * se,
                ate + 1.96 * se,
                scale="difference",
                outcome_sd=float(np.std(self.state["y"])),
            )
            self.metrics["e_value_point"] = _first_float(ev, ("e_value_point", "point", "e_value"))
            self.metrics["e_value_ci"] = _first_float(ev, ("e_value_ci", "ci", "e_value_lower"))
        except Exception as exc:
            log.debug("e-value skipped: %s", exc)

        try:
            from prism.causal.learners import TLearner
            from prism.causal.refute import sensitivity_contour

            X, y, arm = self.state["X"], self.state["y"], self.state["arm"]
            idx = np.where(~te)[0][:4000]
            self.figures_data["sensitivity"] = sensitivity_contour(
                lambda: TLearner(random_state=self.cfg.seed), X[idx], arm[idx], y[idx]
            )
        except Exception as exc:
            log.debug("sensitivity contour skipped: %s", exc)

    # =================================================================================
    # 7. optimize
    # =================================================================================

    def step_optimize(self) -> dict[str, Any]:
        """Turn effects into money, allocate a budget, and prove the policy is worth it."""
        from prism.decision.economics import build_decision_frame, expected_net_value
        from prism.decision.optimize import baseline_policies, efficient_frontier, lagrangian_allocate

        sample, best = self.need("sample", "best_cate")
        churn_risk, clv = self.state["churn_risk"], self.state["clv"]
        econ = self.cfg.economic_config()
        n = len(sample)

        rmst_cate = self.state["cate"].get("survival")
        if rmst_cate is None or rmst_cate.shape != best.shape:
            rmst_cate = np.zeros_like(best)

        net = np.asarray(expected_net_value(best, econ), dtype=float)
        unit_cost = np.asarray(
            [c * r for c, r in zip(econ.arm_costs, econ.offer_redemption_rate)], dtype=float
        )
        cost = np.tile(unit_cost, (n, 1))

        try:
            decisions = build_decision_frame(
                sample["customer_id"].to_numpy(), best, rmst_cate, churn_risk, clv, econ
            )
        except Exception as exc:
            log.warning("build_decision_frame failed (%s); assembling a minimal frame", exc)
            decisions = pd.DataFrame({"customer_id": sample["customer_id"].to_numpy(),
                                      "churn_risk": churn_risk, "clv": clv})
            for a in range(N_ARMS):
                decisions[f"cate_value_{a}"] = 0.0 if a == 0 else best[:, a - 1]
                decisions[f"cate_rmst_{a}"] = 0.0 if a == 0 else rmst_cate[:, a - 1]
                decisions[f"cost_{a}"] = unit_cost[a]
                decisions[f"net_value_{a}"] = net[:, a]
            rec = net.argmax(axis=1)
            rec[net.max(axis=1) <= 0] = 0
            decisions["recommended_arm"] = rec
            decisions["expected_net_value"] = np.maximum(net[np.arange(n), rec], 0.0)

        for a in range(N_ARMS):
            decisions[f"net_value_{a}"] = net[:, a]
            decisions[f"cost_{a}"] = unit_cost[a]
        decisions["split"] = sample["split"].to_numpy()
        for extra in ("gt_segment", "region", "plan_tier"):
            if extra in sample.columns:
                decisions[extra] = sample[extra].to_numpy()

        budget = float(self.cfg.economics.budget)
        max_spend = float(cost.max(axis=1).sum())
        if budget > max_spend:
            budget = 0.2 * max_spend
            log.info("configured budget exceeds the maximum possible spend; using %.0f", budget)
        self.metrics["budget"] = budget

        causal_alloc = lagrangian_allocate(net, cost, budget)
        decisions["assigned_arm"] = np.asarray(causal_alloc.assignment, dtype=int)
        self.state["decisions"] = decisions
        self.state["net"] = net
        self.state["cost"] = cost
        self.state["allocation"] = causal_alloc

        self.metrics.update(
            {
                "policy_n_treated": int(causal_alloc.n_treated),
                "policy_total_cost": float(causal_alloc.total_cost),
                "policy_value_optimised": float(causal_alloc.expected_incremental_value),
                "policy_shadow_price": (
                    float(causal_alloc.dual_lambda)
                    if causal_alloc.dual_lambda is not None and np.isfinite(causal_alloc.dual_lambda)
                    else None
                ),
                "policy_roi": (
                    float(causal_alloc.expected_incremental_value / causal_alloc.total_cost)
                    if causal_alloc.total_cost > 0 else None
                ),
            }
        )

        # -- baselines this must beat -------------------------------------------------
        policies = {"causal (PRISM)": causal_alloc}
        try:
            policies.update(baseline_policies(churn_risk, clv, net, cost, budget))
        except Exception as exc:
            log.warning("baseline policies failed: %s", exc)

        rows = []
        for name, alloc in policies.items():
            assign = np.asarray(getattr(alloc, "assignment", np.zeros(n, dtype=int)), dtype=int)
            realised = float(net[np.arange(n), assign].sum())
            spend = float(cost[np.arange(n), assign].sum())
            rows.append(
                {
                    "policy": name,
                    "value": realised,
                    "n_treated": int((assign > 0).sum()),
                    "total_cost": spend,
                    "roi": (realised / spend) if spend > 0 else np.nan,
                    "within_budget": bool(spend <= budget + 1e-6),
                }
            )
            self.metrics[f"policy_value_{name.replace(' ', '_').replace('(', '').replace(')', '')}"] = realised

        comparison = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
        self.tables["policy_comparison"] = comparison
        self.figures_data["policy_comparison"] = comparison
        log.info("policy comparison:\n%s", comparison.round(2).to_string(index=False))

        causal_v = float(comparison.loc[comparison["policy"] == "causal (PRISM)", "value"].iloc[0])
        for base, key in (
            ("highest_risk", "policy_uplift_vs_risk_targeting"),
            ("treat_all", "policy_uplift_vs_treat_all"),
            ("random", "policy_uplift_vs_random"),
        ):
            m = comparison.loc[comparison["policy"] == base, "value"]
            if len(m):
                self.metrics[key] = causal_v - float(m.iloc[0])
                self.metrics[f"policy_value_{base}"] = float(m.iloc[0])

        # -- off-policy value with confidence intervals -------------------------------
        self._off_policy(policies)

        # -- oracle gap ---------------------------------------------------------------
        gt_cols = [f"gt_tau_value_{a}" for a in range(1, N_ARMS)]
        if all(c in sample.columns for c in gt_cols):
            true_tau = np.nan_to_num(sample[gt_cols].to_numpy(dtype=float))
            true_net = np.column_stack([np.zeros(n), true_tau - unit_cost[1:]])
            assign = np.asarray(causal_alloc.assignment, dtype=int)
            realised = float(true_net[np.arange(n), assign].sum())
            oracle_assign = true_net.argmax(axis=1)
            oracle_assign[true_net.max(axis=1) <= 0] = 0
            oracle_cost = cost[np.arange(n), oracle_assign].sum()
            if oracle_cost > budget:  # oracle under the same budget constraint
                oracle_alloc = lagrangian_allocate(true_net, cost, budget)
                oracle_val = float(true_net[np.arange(n), np.asarray(oracle_alloc.assignment, int)].sum())
            else:
                oracle_val = float(true_net[np.arange(n), oracle_assign].sum())
            self.metrics["oracle_value"] = oracle_val
            self.metrics["realised_true_value"] = realised
            self.metrics["oracle_value_captured_pct"] = (
                float(100.0 * realised / oracle_val) if oracle_val > 0 else None
            )
            self.metrics["policy_regret_total"] = oracle_val - realised
            self.metrics["pct_rows_matching_oracle_arm"] = float(100.0 * (assign == oracle_assign).mean())
            log.info(
                "captured %.1f%% of the oracle's achievable value",
                self.metrics["oracle_value_captured_pct"] or float("nan"),
            )

        # -- frontier and fairness ----------------------------------------------------
        try:
            grid = np.array([b for b in self.cfg.economics.budget_grid if b <= max_spend] or [budget])
            frontier = efficient_frontier(net, cost, grid)
            self.tables["efficient_frontier"] = frontier
            self.figures_data["frontier"] = frontier
        except Exception as exc:
            log.warning("efficient frontier failed: %s", exc)

        self._fairness(decisions, net, cost, budget)
        return {"budget": budget, "value": causal_v}

    def _off_policy(self, policies: dict[str, Any]) -> None:
        """Doubly-robust policy value with bootstrap CIs on the test split."""
        try:
            from prism.decision.policy_eval import compare_policies

            sample, e_hat, is_train = self.state["sample"], self.state["propensity"], self.state["is_train"]
            X = self.state["X"]
            te = ~is_train
            if te.sum() < 100:
                log.info("test split too small for off-policy evaluation; skipping")
                return

            # Score the NET outcome, not gross revenue: a policy that spends must be charged for
            # it, otherwise "treat everyone" is evaluated as if offers were free and treat_none
            # is an unfairly strong baseline.
            panel_test = sample.loc[te, ["customer_id", "arm", "value_h", "offer_cost"]].copy()
            mr = self.state.get("margin_rate", np.full(len(sample), self.cfg.economics.margin_rate))
            panel_test["net_outcome"] = panel_test["value_h"].to_numpy(dtype=float) * mr[te] - panel_test[
                "offer_cost"
            ].to_numpy(dtype=float)
            panel_test = panel_test.reset_index(drop=True)

            # A real DR estimate needs an outcome model mu_a(X). Without it the estimator falls
            # back to SNIPW, whose variance is far higher -- which is exactly the difference
            # between detecting a targeted policy's gain and not detecting it.
            mu_hat = self._outcome_models(X, sample, is_train, te)

            assignments = {
                name: np.asarray(getattr(a, "assignment", np.zeros(len(sample), int)), dtype=int)[te]
                for name, a in policies.items()
            }
            out = compare_policies(assignments, panel_test, e_hat[te], "net_outcome",
                                   mu_hat=mu_hat, n_boot=300, random_state=self.cfg.seed)
            self.tables["off_policy_value"] = out
            log.info("off-policy value:\n%s", out.round(3).to_string(index=False))

            # Keep this table SEPARATE from policy_comparison. They measure different
            # quantities: policy_comparison is total incremental NET value under the model's
            # own effect estimates, while this is the doubly-robust estimate of mean realised
            # outcome E[Y] per customer under each policy. Merging them would invite comparing
            # two different scales in one row. What transfers across is the ranking and whether
            # the paired difference against the baseline clears zero.
            if {"policy", "value"} <= set(out.columns):
                dr = out[out["method"] == "dr"] if "method" in out.columns else out
                dr = dr.drop_duplicates(subset=["policy"], keep="first")
                best_row = dr.sort_values("value", ascending=False).head(1)
                if len(best_row):
                    self.metrics["off_policy_best"] = str(best_row.iloc[0]["policy"])
                    self.metrics["off_policy_best_value"] = float(best_row.iloc[0]["value"])
                causal = dr[dr["policy"].astype(str).str.contains("causal", case=False)]
                if len(causal):
                    r = causal.iloc[0]
                    self.metrics["off_policy_causal_value"] = float(r["value"])
                    for c in ("ci_low", "ci_high", "diff_vs_baseline", "beats_baseline"):
                        if c in dr.columns and pd.notna(r.get(c)):
                            self.metrics[f"off_policy_causal_{c}"] = _jsonable(r[c])
        except Exception as exc:
            log.warning("off-policy evaluation skipped: %s", exc)

    def _outcome_models(
        self, X: np.ndarray, sample: pd.DataFrame, is_train: np.ndarray, te: np.ndarray
    ) -> np.ndarray | None:
        """Fit mu_a(X) per arm on the training split and predict it for every test row.

        Returns an ``(n_test, n_arms)`` matrix of expected net outcomes, which is the ingredient
        that turns the SNIPW fallback into a genuine doubly-robust estimate.
        """
        try:
            from prism.utils.optional import best_gbm

            arm = sample["arm"].to_numpy(dtype=int)
            mr = self.state.get("margin_rate", np.full(len(sample), self.cfg.economics.margin_rate))
            y_net = sample["value_h"].to_numpy(dtype=float) * mr - sample["offer_cost"].to_numpy(dtype=float)
            mu = np.zeros((int(te.sum()), N_ARMS), dtype=float)
            Xte = X[te]
            for a in range(N_ARMS):
                rows = is_train & (arm == a)
                if rows.sum() < 40:
                    mu[:, a] = float(y_net[is_train].mean()) if is_train.any() else 0.0
                    log.debug("arm %d has only %d training rows; using the pooled mean", a, int(rows.sum()))
                    continue
                model = best_gbm(
                    "regression", n_estimators=200, random_state=self.cfg.seed + a
                ).fit(X[rows], y_net[rows])
                mu[:, a] = np.asarray(model.predict(Xte), dtype=float)
            return mu
        except Exception as exc:
            log.warning("outcome models for DR unavailable (%s); estimator will fall back", exc)
            return None

    def _fairness(self, decisions: pd.DataFrame, net: np.ndarray, cost: np.ndarray, budget: float) -> None:
        """Audit who receives the scarce benefit, and price what parity would cost."""
        attr = next((c for c in ("region", "plan_tier") if c in decisions.columns), None)
        if attr is None:
            return
        try:
            from prism.decision.fairness import disparate_impact_ratio, policy_fairness_audit

            assign = decisions["assigned_arm"].to_numpy(dtype=int)
            audit = policy_fairness_audit(
                assign, decisions[attr], net_value=decisions["expected_net_value"].to_numpy(dtype=float)
                if "expected_net_value" in decisions else None,
            )
            self.tables["fairness"] = audit
            self.metrics["disparate_impact_ratio"] = float(disparate_impact_ratio(assign, decisions[attr]))
            self.metrics["fairness_attribute"] = attr

            from prism.decision.fairness import price_of_fairness, reweight_for_parity

            constrained = reweight_for_parity(net, cost, budget, decisions[attr], tolerance=0.1)
            price = price_of_fairness(self.state["allocation"], constrained)
            self.metrics.update({f"fairness_{k}": _jsonable(v) for k, v in price.items()})
            log.info("price of parity: %s", price)
        except Exception as exc:
            log.warning("fairness audit skipped: %s", exc)

    # =================================================================================
    # 8. monitor
    # =================================================================================

    def step_monitor(self) -> dict[str, Any]:
        """Would we notice if this stopped working?"""
        from prism.monitoring.alerts import evaluate_alerts, render_alerts
        from prism.monitoring.drift import cate_stability, feature_drift_report

        sample, best, is_train = self.need("sample", "best_cate", "is_train")
        ref = sample.loc[is_train, list(FEATURES)]
        cur = sample.loc[~is_train, list(FEATURES)]
        if len(cur) < 30 or len(ref) < 30:
            log.info("not enough rows on one side of the split to measure drift")
            return {}

        drift = feature_drift_report(ref, cur, features=list(FEATURES))
        self.tables["drift"] = drift
        self.figures_data["drift"] = drift
        if "severity" in drift.columns:
            self.metrics["n_features_alert"] = int((drift["severity"] == "alert").sum())
            self.metrics["n_features_warn"] = int((drift["severity"] == "warn").sum())
        if "psi" in drift.columns:
            self.metrics["max_psi"] = float(drift["psi"].max())
        log.info("drift (top rows):\n%s", drift.head(8).round(4).to_string(index=False))

        # CATE stability needs MATCHED units: the question is whether the same customer's
        # recommended treatment would change between an early and a late period, not whether
        # two different populations have similar score distributions. So pair each customer's
        # train-period row with their own test-period row.
        ids = sample["customer_id"].to_numpy()
        tr_idx = {cid: i for i, cid in zip(np.where(is_train)[0], ids[is_train])}
        te_idx = {cid: i for i, cid in zip(np.where(~is_train)[0], ids[~is_train])}
        shared = sorted(set(tr_idx) & set(te_idx))
        if len(shared) < 50:
            log.info("only %d customers appear in both periods; skipping CATE stability", len(shared))
            self.metrics["cate_rank_correlation"] = None
            self.metrics["cate_sign_flip_rate"] = None
        else:
            a = np.array([tr_idx[c] for c in shared])
            b = np.array([te_idx[c] for c in shared])
            stability = cate_stability(best[a], best[b])
            self.metrics["cate_rank_correlation"] = _first_float(
                stability, ("spearman_rank_corr", "spearman", "rank_corr")
            )
            self.metrics["cate_sign_flip_rate"] = _first_float(stability, ("sign_flip_rate", "sign_flips"))
            self.metrics["cate_recommended_arm_agreement"] = _first_float(
                stability, ("recommended_arm_agreement",)
            )
            self.metrics["n_customers_tracked"] = len(shared)
            self.state["cate_stability"] = stability
            log.info(
                "CATE stability over %d matched customers: rank corr %s, sign flips %s",
                len(shared),
                _num(self.metrics["cate_rank_correlation"]),
                _pct(self.metrics["cate_sign_flip_rate"]),
            )

        try:
            alerts = evaluate_alerts(drift, None, thresholds=None)
        except Exception:
            alerts = evaluate_alerts(drift, None)
        rendered = render_alerts(alerts)
        (self.cfg.paths.path("reports") / "alerts.txt").write_text(rendered, encoding="utf-8")
        self.metrics["n_alerts"] = len(alerts)
        self.metrics["n_critical_alerts"] = sum(1 for a in alerts if getattr(a, "severity", "") == "critical")
        log.info("alerts:\n%s", rendered)
        return {"n_alerts": len(alerts)}

    # =================================================================================
    # 9. report
    # =================================================================================

    def step_report(self) -> dict[str, Any]:
        """Write every artifact a reader or a downstream service needs."""
        reports = self.cfg.paths.path("reports")
        figdir = self.cfg.paths.path("figures")

        for name, df in self.tables.items():
            try:
                df.to_csv(reports / f"{name}.csv", index=False)
            except Exception as exc:
                log.warning("could not write %s.csv: %s", name, exc)

        decisions = self.state.get("decisions")
        if decisions is not None:
            try:
                decisions.to_parquet(reports / "decision_frame.parquet", index=False)
            except Exception:
                decisions.to_csv(reports / "decision_frame.csv", index=False)

        from prism.pipelines.figures import generate_all

        fd = dict(self.figures_data)
        fd.setdefault("arm_names", list(ARM_NAMES))
        if "survival_curves" in fd and np.ndim(fd["survival_curves"]) == 2:
            model = self.state.get("cate_models", {}).get("survival")
            if model is not None and hasattr(model, "predict_survival_curves"):
                try:
                    Xs = self.state["X"][: min(len(self.state["X"]), 3000)]
                    fd["survival_curves"] = np.asarray(model.predict_survival_curves(Xs), dtype=float)
                except Exception as exc:
                    log.debug("per-arm survival curves unavailable: %s", exc)
        written = generate_all(fd, figdir=figdir)
        self.metrics["n_figures"] = len(written)

        self.metrics["model_version"] = f"prism-{self.cfg.experiment_name}-seed{self.cfg.seed}"
        self.metrics["estimator"] = self.state.get("best_learner")
        self.metrics["run_ok"] = all(r.ok for r in self.results)
        self.metrics["steps_failed"] = [r.name for r in self.results if not r.ok]

        # Write the bundle BEFORE dumping metrics: _write_bundle records whether it succeeded,
        # and that flag has to make it into metrics.json for the CI gate to see it.
        self._write_bundle()

        payload = _jsonable(self.metrics)
        (reports / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        self._write_model_card(reports)
        self._write_decision_memo(reports)

        with self.tracker.run(f"{self.cfg.experiment_name}"):
            self.tracker.log_params(
                {
                    "seed": self.cfg.seed,
                    "n_customers": self.cfg.sim.n_customers,
                    "n_periods": self.cfg.sim.n_periods,
                    "horizon": self.cfg.sim.horizon,
                    "learners": list(self.cfg.causal.learners),
                    "budget": self.cfg.economics.budget,
                    "confounding_strength": self.cfg.sim.confounding_strength,
                }
            )
            self.tracker.log_metrics({k: v for k, v in payload.items() if isinstance(v, (int, float))})
            for name, df in self.tables.items():
                self.tracker.log_dataframe(df, f"{name}.csv")

        log.info("wrote %d tables, %d figures and metrics.json to %s",
                 len(self.tables), len(written), reports)
        return {"figures": len(written), "tables": len(self.tables)}

    def _write_bundle(self) -> None:
        """Persist everything the serving layer needs."""
        try:
            import joblib

            best = self.state.get("best_learner")
            bundle = {
                "encoder": self.state.get("encoder"),
                "feature_names": self.state.get("feature_names"),
                "cate_model": self.state.get("cate_models", {}).get(best),
                "survival_model": self.state.get("survival_model"),
                "clv": None,
                "econ": self.cfg.economic_config(),
                "metadata": {
                    "version": self.metrics.get("model_version"),
                    "trained_at": pd.Timestamp.utcnow().isoformat(),
                    "estimator": best,
                    "training_window": {
                        "train_end_period": self.cfg.split.train_end,
                        "valid_end_period": self.cfg.split.valid_end,
                        "horizon_months": self.cfg.sim.horizon,
                    },
                    "metrics": {
                        k: self.metrics.get(k)
                        for k in ("best_learner_pehe", "best_learner_qini", "survival_c_index",
                                  "oracle_value_captured_pct", "policy_value_optimised")
                    },
                },
            }
            path = self.cfg.paths.path("models") / "bundle.joblib"
            tmp = path.with_suffix(".joblib.tmp")

            # Write atomically and verify by reading back. A partial joblib.dump leaves a
            # truncated file on disk that looks present but raises EOFError on load, so the
            # serving layer would report "degraded" with no clue why. Either the bundle is
            # complete and loadable, or the previous good one is left untouched.
            joblib.dump(bundle, tmp, compress=3)
            reloaded = joblib.load(tmp)
            missing = [k for k in ("encoder", "feature_names", "metadata") if k not in reloaded]
            if missing:
                raise ValueError(f"bundle round-tripped without {missing}")
            tmp.replace(path)

            log.info("wrote serving bundle to %s (%.1f MB)", path, path.stat().st_size / 1e6)
            self.metrics["bundle_path"] = str(path)
            self.metrics["bundle_written"] = True
        except Exception as exc:
            # A missing serving bundle is a real failure, not a cosmetic one: it is the
            # artifact the API loads. Say so loudly and record it in metrics.json.
            log.error("could not write the serving bundle: %s", exc)
            self.metrics["bundle_written"] = False
            self.metrics["bundle_error"] = f"{type(exc).__name__}: {exc}"
            try:
                tmp = self.cfg.paths.path("models") / "bundle.joblib.tmp"
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def _write_model_card(self, reports: Path) -> None:
        """Generate the model card from the run's own numbers."""
        m = self.metrics
        lb = self.tables.get("leaderboard")
        card = f"""# Model card - PRISM Causal Survival Uplift

**Version** `{m.get('model_version')}` · **Estimator** `{m.get('estimator')}` ·
**Trained** {pd.Timestamp.utcnow().date().isoformat()}

## Intended use
Rank customers by the *causal effect* of each retention offer on expected discounted lifetime
value, and allocate a fixed retention budget across offers. It is a decision-support model for a
marketing or retention team.

**Not** intended for: pricing decisions, credit or eligibility decisions, individual-level
guarantees about a customer's behaviour, or any use where an incorrect offer carries material harm
to the customer.

## Training data
| | |
|---|---|
| customers | {m.get('n_customers', '-'):,} |
| panel rows | {m.get('n_rows_panel', '-'):,} |
| decision points used | {m.get('n_decision_points', '-'):,} |
| periods | {m.get('n_periods', '-')} (horizon {m.get('horizon', '-')} months) |
| treated share | {_pct(m.get('treated_share'))} |
| randomised (RCT) share | {_pct(m.get('rct_share'))} |
| censoring rate | {_pct(m.get('censoring_rate'))} |
| split | temporal, by period (train <= {self.cfg.split.train_end}, valid <= {self.cfg.split.valid_end}) |

Features: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical = {len(FEATURES)}.
All point-in-time safe; leakage detected: **{m.get('leakage_detected')}**.

## Performance
| metric | value | reading |
|---|---|---|
| PEHE (vs known truth) | {_num(m.get('best_learner_pehe'))} | lower is better; constant-effect baseline {_num(m.get('baseline_constant_pehe'))} |
| eps_ATE | {_num(m.get('best_learner_eps_ate'))} | absolute error in the average effect |
| Qini | {_num(m.get('best_learner_qini'))} | ranking quality vs random targeting |
| CATE rank corr. vs truth | {_num(m.get('cate_rank_correlation_vs_truth'))} | Spearman |
| survival C-index | {_num(m.get('survival_c_index'))} | discrimination of the hazard model |
| oracle value captured | {_num(m.get('oracle_value_captured_pct'))}% | share of achievable value |

{_leaderboard_md(lb)}

## Causal assumptions
1. **Consistency / SUTVA** - no interference between customers. *Not verified; a known limitation.*
2. **Positivity** - measured. Effective sample size {_num(m.get('effective_sample_size'))}
   ({_pct(m.get('ess_fraction'))} of nominal).
3. **Unconfoundedness** - **known to be violated by construction**: a hidden confounder is
   simulated deliberately. The refutation suite quantifies the consequence
   ({m.get('refutation_passed', '-')}/{m.get('refutation_total', '-')} tests passed);
   E-value {_num(m.get('e_value_point'))}.
4. **Non-informative censoring** given covariates - handled by IPCW.

Balance after weighting: mean |SMD| = {_num(m.get('mean_abs_smd_weighted') or m.get('mean_abs_smd_unweighted'))}
(target < 0.1).

## Fairness
Audited on `{m.get('fairness_attribute', 'n/a')}`; disparate-impact ratio
{_num(m.get('disparate_impact_ratio'))} (four-fifths heuristic: >= 0.8).
Enforcing parity costs {_num(m.get('fairness_value_forgone'))} of incremental value
({_num(m.get('fairness_pct_value_forgone'))}%). See `fairness.csv`.

## Monitoring
Retrain when any of these trip: PSI > 0.25 on a top feature
(currently {m.get('n_features_alert', '-')} features in alert), CATE rank correlation < 0.70
(currently {_num(m.get('cate_rank_correlation'))}), or CATE sign-flip rate > 15%
(currently {_pct(m.get('cate_sign_flip_rate'))}).

## Ethical considerations
The model allocates a scarce *benefit*. A profit-maximising allocation concentrates offers on
high-value customers, which can systematically under-serve lower-value groups. The fairness audit
above makes that visible and the constrained allocator prices the alternative. The decision to
accept the tradeoff is a business one, not a modelling one.

Generated by `prism.pipelines.run_all`. Methodology: `docs/METHODOLOGY.md`.
"""
        (reports / "model_card.md").write_text(card, encoding="utf-8")

    def _write_decision_memo(self, reports: Path) -> None:
        """One page for someone who will never read the code."""
        m = self.metrics
        causal = m.get("policy_value_optimised")
        risk = m.get("policy_value_highest_risk")
        gain = m.get("policy_uplift_vs_risk_targeting")
        lam = m.get("policy_shadow_price")
        # lambda* = d(net value)/d(budget). Net value is ALREADY net of offer cost, so the
        # meaningful threshold is zero, not one: any positive lambda means the next unit of
        # budget still adds profit. A lambda of ~0 means the budget never bound.
        if not isinstance(lam, (int, float)):
            budget_verdict = "could not be estimated for this run"
        elif lam <= 1e-6:
            budget_verdict = (
                "**0** - the budget never binds. Every offer worth making was already funded and "
                "money was left over, so this programme could run on less"
            )
        else:
            budget_verdict = (
                f"**{lam:,.2f}** - each additional unit of budget still returns {lam:,.2f} in "
                "incremental profit, so the programme is under-funded rather than over-funded"
            )
        pct = (100 * gain / abs(risk)) if (gain is not None and risk) else None

        memo = f"""# Retention budget: where it should go

**Recommendation.** Spend the {_money(m.get('budget'))} retention budget on the
{m.get('policy_n_treated', 0):,} customers the causal model identifies, rather than on the
customers most likely to churn.

## Why
Targeting the highest-risk customers is the industry default and it is measurably the wrong rule
here. Risk and *responsiveness* are different orderings of the same customer base:

- Some customers who look risky will leave whatever we do. Spending on them buys nothing.
- Some customers are made *more* likely to leave by being contacted about their price. Spending on
  them actively destroys value. They are invisible to a churn model.

| policy | expected incremental value |
|---|---|
| **causal targeting (recommended)** | **{_money(causal)}** |
| target the highest churn risk | {_money(risk)} |
| treat everyone | {_money(m.get('policy_value_treat_all'))} |
| do nothing | 0 |

Choosing the causal policy over risk targeting is worth **{_money(gain)}**\
{f" ({pct:+.0f}%)" if pct is not None else ""} on this population.

## How much budget?
The marginal return on the next unit of spend at the current budget is
{budget_verdict}.
The efficient frontier (`docs/figures/efficient_frontier.png`) shows the value of every budget
level; the point where that curve flattens is the right size for this programme.

## How confident should we be?
- The estimator recovers {_num(m.get('oracle_value_captured_pct'), 1)}% of the value a
  perfectly-informed targeter could achieve.
- {m.get('refutation_passed', '-')} of {m.get('refutation_total', '-')} refutation tests passed:
  the effect does not appear when treatment is shuffled, and it survives adding noise covariates
  and re-estimating on subsets.
{_confounding_bullet(m)}

## What to do next
1. Run the recommended allocation against a **randomised holdout** for one cycle. The model is
   estimated from observational history, and only an experiment settles it.
2. Watch the monitoring page. Retrain if the customer mix drifts or the estimated effects start
   changing sign ({_pct(m.get('cate_sign_flip_rate'))} of customers currently flip between periods).
3. Decide explicitly whether the fairness tradeoff is acceptable: enforcing equal treatment rates
   across groups costs {_money(m.get('fairness_value_forgone'))}.

*Generated by `prism.pipelines.run_all`. Figures in `docs/figures/`, tables in
`artifacts/reports/`.*
"""
        (reports / "decision_memo.md").write_text(memo, encoding="utf-8")


# =====================================================================================
# formatting helpers
# =====================================================================================


def _num(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    try:
        return f"{float(v):,.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _pct(v: Any) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{float(v) * 100:.1f}%"


def _money(v: Any) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    v = float(v)
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:,.2f}M"
    if a >= 1e3:
        return f"{v / 1e3:,.1f}k"
    return f"{v:,.0f}"


def _first_float(d: Any, keys: tuple[str, ...]) -> float | None:
    """Pull the first present key out of a dict as a float, tolerating naming variation."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            try:
                v = float(d[k])
                return v if np.isfinite(v) else None
            except (TypeError, ValueError):
                continue
    return None


def _to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table.

    ``DataFrame.to_markdown`` needs the optional ``tabulate`` package, and a missing
    formatting dependency must never be the reason a pipeline run produces no model card.
    Falls back to building the table by hand.
    """
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
        return "\n".join(lines)


def _confounding_bullet(m: dict[str, Any]) -> str:
    """Render the unmeasured-confounding bullet, or say plainly it was not computed.

    The E-value is only meaningful once the effect is on a risk-ratio scale. If the run did
    not produce one, say so rather than printing a number whose scale is wrong -- a
    confidently stated 41x that should have been 1.3 is worse than an honest gap.
    """
    ev = m.get("e_value_point")
    if not isinstance(ev, (int, float)) or ev != ev:
        return (
            "- The main risk is confounding we have not measured. The E-value could not be"
            "\n  computed for this run, so treat the estimate as unquantified on that axis."
        )
    if ev < 1.5:
        return (
            "- **The main risk is confounding we have not measured, and this result is"
            "\n  sensitive to it.** An unmeasured factor associated with both the offer and"
            f"\n  churn by only about **{ev:.2f}x** would be enough to explain the effect"
            "\n  away, which is not a demanding bar. Validate against the randomised"
            "\n  holdout before acting at scale."
        )
    return (
        "- The main risk is confounding we have not measured. An unmeasured factor would"
        f"\n  need an association of roughly **{ev:.2f}x** with both the offer and churn"
        "\n  to explain the result away. Compare that against the strongest covariate we"
        "\n  do observe."
    )


def _leaderboard_md(lb: pd.DataFrame | None) -> str:
    if lb is None or lb.empty:
        return ""
    cols = [c for c in ("learner", "pehe", "eps_ate", "rank_corr_vs_truth", "qini", "auuc", "seconds")
            if c in lb.columns]
    return "### Estimator comparison\n\n" + _to_markdown(lb[cols].round(4)) + "\n"


# =====================================================================================
# entry points
# =====================================================================================


def main(config_path: str = "configs/default.yaml", steps: str = "all", verbose: bool = False) -> int:
    """Run the pipeline.

    Parameters
    ----------
    config_path : str
        YAML config, e.g. ``configs/fast.yaml``.
    steps : str
        ``"all"`` or a comma-separated subset of :data:`STEPS`.
    verbose : bool
        Emit debug logging.

    Returns
    -------
    int
        Process exit code: 0 if every requested step succeeded.
    """
    set_level("DEBUG" if verbose else "INFO")
    cfg = load_config(config_path)
    seed_everything(cfg.seed)

    wanted = list(STEPS) if steps.strip().lower() == "all" else [s.strip() for s in steps.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STEPS]
    if unknown:
        log.error("unknown steps %s; valid steps are %s", unknown, list(STEPS))
        return 2

    log.info("PRISM pipeline | config=%s | seed=%d | steps=%s", config_path, cfg.seed, ",".join(wanted))
    t0 = time.perf_counter()

    pipe = Pipeline(cfg)
    dispatch: dict[str, Callable[[], dict[str, Any] | None]] = {
        "simulate": pipe.step_simulate,
        "ingest": pipe.step_ingest,
        "features": pipe.step_features,
        "train": pipe.step_train,
        "causal": pipe.step_causal,
        "evaluate": pipe.step_evaluate,
        "optimize": pipe.step_optimize,
        "monitor": pipe.step_monitor,
        "report": pipe.step_report,
    }

    for name in wanted:
        res = pipe.run_step(name, dispatch[name])
        if not res.ok and name in ("simulate", "features", "causal"):
            log.error("step %s is required by everything downstream; stopping", name)
            if "report" in wanted and name != "report":
                pipe.run_step("report", dispatch["report"])
            break

    elapsed = time.perf_counter() - t0
    failed = [r.name for r in pipe.results if not r.ok]

    log.info("=" * 68)
    log.info("PIPELINE COMPLETE in %.1fs (%.1f min)", elapsed, elapsed / 60)
    for r in pipe.results:
        log.info("  %-9s %-4s %6.1fs%s", r.name, "OK" if r.ok else "FAIL", r.seconds,
                 f"  {r.error}" if r.error else "")
    if failed:
        log.error("failed steps: %s", failed)
    else:
        log.info("all steps succeeded")
        log.info("  reports  -> %s", cfg.paths.path("reports"))
        log.info("  figures  -> %s", cfg.paths.path("figures"))
    log.info("=" * 68)
    return 1 if failed else 0


def cli() -> int:
    """Console-script entry point (``prism ...``)."""
    parser = argparse.ArgumentParser(
        prog="prism",
        description="PRISM - Causal Survival Uplift for budget-constrained retention.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="path to the YAML config")
    parser.add_argument("--steps", default="all", help=f"'all' or a comma-separated subset of: {','.join(STEPS)}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()
    return main(args.config, args.steps, args.verbose)


if __name__ == "__main__":
    sys.exit(cli())
