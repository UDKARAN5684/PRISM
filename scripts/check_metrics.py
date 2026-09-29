"""CI gate: fail the build if the pipeline's headline metrics are missing or implausible.

A green test suite proves the code runs. This proves the *science* still holds: the causal
estimator must still recover the known ground-truth treatment effects, and the optimised
policy must still beat the naive risk-targeting baseline. Those are the two claims the
whole project rests on, so they are enforced in CI rather than trusted.

Usage
-----
    python scripts/check_metrics.py artifacts/reports/metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

#: name -> (required, predicate, human-readable expectation)
#:
#: Note on thresholds: PEHE and eps_ATE are denominated in the outcome's own units (currency
#: here), so an absolute cut-off like "PEHE < 50" means nothing -- it would pass or fail on a
#: change of currency. Every scale-dependent claim is therefore checked RELATIVELY, against
#: the constant-effect baseline or the naive estimator, in CROSS_CHECKS below.
CHECKS: dict[str, tuple[bool, Any, str]] = {
    # --- the causal claim -------------------------------------------------------------
    "best_learner_pehe": (True, lambda v: v > 0.0 and v == v, "PEHE is finite and positive"),
    "best_learner_eps_ate": (True, lambda v: v == v and abs(v) < float("inf"), "|ATE error| is finite"),
    "cate_rank_correlation_vs_truth": (
        True,
        lambda v: v > 0.15,
        "the estimated CATE ranking correlates positively with the true effect",
    ),
    "bias_reduction_vs_naive_pct": (
        True,
        lambda v: v > 50.0,
        "adjustment removes more than half the naive estimator's bias",
    ),
    # --- the money claim --------------------------------------------------------------
    "policy_value_optimised": (True, lambda v: True, "optimised policy value is reported"),
    "policy_value_highest_risk": (True, lambda v: True, "risk-targeting baseline is reported"),
    "policy_uplift_vs_risk_targeting": (
        True,
        lambda v: v > 0.0,
        "the causal policy beats targeting the highest-risk customers",
    ),
    "policy_uplift_vs_treat_all": (False, lambda v: v > 0.0, "the causal policy beats treating everyone"),
    # A policy can genuinely capture NEGATIVE value against the oracle (treating sleeping dogs
    # destroys value), so this is informational rather than a pass/fail bound. It only has to
    # be finite and not exceed the oracle, which by definition it cannot.
    "oracle_value_captured_pct": (False, lambda v: v == v and v <= 100.5, "share of oracle value is reported"),
    # --- hygiene ----------------------------------------------------------------------
    "n_rows_panel": (True, lambda v: v > 0, "the panel is non-empty"),
    "leakage_detected": (True, lambda v: bool(v) is False, "no point-in-time leakage detected"),
    "contract_passed": (True, lambda v: bool(v) is True, "the gold-panel data contract passed"),
}

#: (name, required, predicate over the whole metrics dict, expectation). These are the claims
#: that only make sense as a comparison between two numbers.
CROSS_CHECKS: list[tuple[str, bool, Any, str]] = [
    (
        "pehe_beats_constant_baseline",
        True,
        lambda m: m["best_learner_pehe"] < m["baseline_constant_pehe"],
        "the best learner's PEHE beats a constant-effect baseline "
        "(i.e. it learned real heterogeneity, not just the average)",
    ),
    (
        "model_beats_naive_on_bias",
        True,
        lambda m: m["model_mean_abs_error"] < m["naive_mean_abs_error"],
        "the adjusted estimate is closer to the truth than the naive difference in means",
    ),
    (
        "causal_policy_beats_risk_targeting",
        True,
        lambda m: m["policy_value_optimised"] > m["policy_value_highest_risk"],
        "targeting on causal effect beats targeting on churn risk",
    ),
    (
        "budget_respected",
        True,
        lambda m: m["policy_total_cost"] <= m["budget"] * 1.0001 + 1e-6,
        "the allocation stayed within budget",
    ),
]


def main(argv: list[str]) -> int:
    """Validate a metrics JSON file. Returns a process exit code."""
    path = Path(argv[1] if len(argv) > 1 else "artifacts/reports/metrics.json")
    if not path.exists():
        print(f"FAIL: {path} does not exist -- the pipeline did not finish.")
        return 1

    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"FAIL: {path} is not valid JSON: {exc}")
        return 1

    if not isinstance(metrics, dict) or not metrics:
        print(f"FAIL: {path} is empty or not a JSON object.")
        return 1

    failures: list[str] = []
    skipped: list[str] = []
    passed: list[str] = []

    for key, (required, predicate, expectation) in CHECKS.items():
        if key not in metrics:
            (failures if required else skipped).append(
                f"{key}: MISSING (expected: {expectation})" if required else f"{key}: not reported"
            )
            continue
        value = metrics[key]
        if isinstance(value, (int, float)) and value != value:  # NaN
            failures.append(f"{key}: NaN (expected: {expectation})")
            continue
        try:
            ok = bool(predicate(value))
        except Exception as exc:  # a wrong type reaching the predicate is itself a failure
            failures.append(f"{key}: {value!r} could not be checked ({exc})")
            continue
        (passed if ok else failures).append(f"{key} = {value!r}" if ok else f"{key} = {value!r} (expected: {expectation})")

    for name, required, predicate, expectation in CROSS_CHECKS:
        try:
            ok = bool(predicate(metrics))
        except KeyError as exc:
            (failures if required else skipped).append(
                f"{name}: needs {exc} which is not reported" if required else f"{name}: not computable"
            )
            continue
        except Exception as exc:
            failures.append(f"{name}: could not be checked ({exc})")
            continue
        (passed if ok else failures).append(name if ok else f"{name} FAILED (expected: {expectation})")

    width = 72
    print("=" * width)
    print(f"PRISM metrics gate -- {path}")
    print("=" * width)
    for line in passed:
        print(f"  PASS  {line}")
    for line in skipped:
        print(f"  SKIP  {line}")
    for line in failures:
        print(f"  FAIL  {line}")
    print("-" * width)
    print(f"{len(passed)} passed, {len(skipped)} skipped, {len(failures)} failed")

    if failures:
        print("\nThe pipeline ran but its scientific or economic claims no longer hold.")
        return 1
    print("\nAll headline claims hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
