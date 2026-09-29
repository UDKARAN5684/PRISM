"""Lightweight data contracts.

A hand-rolled alternative to Great Expectations: declarative :class:`Expectation` objects,
an aggregate :class:`DataContract`, and a :class:`ValidationResult` that can either report
or raise. Keeping this in-repo means the contract is reviewable in a PR and adds no
heavyweight dependency to the serving image.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "Expectation",
    "ValidationResult",
    "DataContract",
    "DataContractError",
    "PANEL_CONTRACT",
    "build_panel_contract",
]

_KINDS = {"not_null", "dtype", "in_range", "in_set", "unique", "monotonic", "regex", "row_count", "column_exists"}


class DataContractError(Exception):
    """Raised when a dataset violates an ``error``-severity expectation."""


@dataclass(frozen=True)
class Expectation:
    """A single declarative assertion about a column (or the frame itself).

    Parameters
    ----------
    column : str
        Target column. Use ``"*"`` for frame-level checks such as ``row_count``.
    kind : str
        One of ``not_null``, ``dtype``, ``in_range``, ``in_set``, ``unique``,
        ``monotonic``, ``regex``, ``row_count``, ``column_exists``.
    params : dict
        Kind-specific parameters, e.g. ``{"min": 0, "max": 1}`` for ``in_range``.
    severity : {"error", "warn"}
        ``error`` failures make :meth:`ValidationResult.raise_for_status` raise.
    """

    column: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    severity: str = "error"

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown expectation kind {self.kind!r}; valid: {sorted(_KINDS)}")
        if self.severity not in {"error", "warn"}:
            raise ValueError(f"severity must be 'error' or 'warn', got {self.severity!r}")


@dataclass
class ValidationResult:
    """Outcome of validating one frame against one contract."""

    passed: bool
    failures: list[dict[str, Any]]
    n_rows: int
    contract_name: str = ""

    def summary(self) -> pd.DataFrame:
        """Return failures as a tidy frame (empty frame with the right columns if clean)."""
        cols = ["column", "kind", "severity", "detail", "n_bad", "frac_bad"]
        if not self.failures:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self.failures)[cols]

    @property
    def n_errors(self) -> int:
        """Number of ``error``-severity failures."""
        return sum(1 for f in self.failures if f["severity"] == "error")

    @property
    def n_warnings(self) -> int:
        """Number of ``warn``-severity failures."""
        return sum(1 for f in self.failures if f["severity"] == "warn")

    def raise_for_status(self) -> None:
        """Raise :class:`DataContractError` if any ``error``-severity expectation failed."""
        errs = [f for f in self.failures if f["severity"] == "error"]
        if errs:
            lines = "\n".join(f"  - [{f['column']}] {f['kind']}: {f['detail']}" for f in errs[:20])
            more = "" if len(errs) <= 20 else f"\n  ... and {len(errs) - 20} more"
            raise DataContractError(
                f"contract '{self.contract_name}' failed with {len(errs)} error(s) over {self.n_rows} rows:\n{lines}{more}"
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        status = "PASS" if self.passed else "FAIL"
        return f"<ValidationResult {self.contract_name} {status} rows={self.n_rows} errors={self.n_errors} warnings={self.n_warnings}>"


class DataContract:
    """An ordered collection of :class:`Expectation` objects applied to a DataFrame."""

    def __init__(self, name: str, expectations: list[Expectation]) -> None:
        self.name = name
        self.expectations = list(expectations)

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Check ``df`` against every expectation.

        Parameters
        ----------
        df : pandas.DataFrame
            Frame under test.

        Returns
        -------
        ValidationResult
            Never raises; inspect ``.passed`` or call ``.raise_for_status()``.
        """
        failures: list[dict[str, Any]] = []
        n = len(df)

        def fail(exp: Expectation, detail: str, n_bad: int) -> None:
            failures.append(
                {
                    "column": exp.column,
                    "kind": exp.kind,
                    "severity": exp.severity,
                    "detail": detail,
                    "n_bad": int(n_bad),
                    "frac_bad": float(n_bad / n) if n else 0.0,
                }
            )

        for exp in self.expectations:
            if exp.kind == "row_count":
                lo = exp.params.get("min", 0)
                hi = exp.params.get("max", np.inf)
                if not (lo <= n <= hi):
                    fail(exp, f"row count {n} outside [{lo}, {hi}]", 0)
                continue

            if exp.column != "*" and exp.column not in df.columns:
                fail(exp, "column missing", n)
                continue
            if exp.kind == "column_exists":
                continue

            s = df[exp.column]

            if exp.kind == "not_null":
                bad = int(s.isna().sum())
                tol = exp.params.get("max_frac", 0.0)
                if n and bad / n > tol:
                    fail(exp, f"{bad} nulls (>{tol:.1%} tolerated)", bad)

            elif exp.kind == "dtype":
                want = exp.params["dtype"]
                ok = {
                    "int": pd.api.types.is_integer_dtype,
                    "float": pd.api.types.is_float_dtype,
                    "numeric": pd.api.types.is_numeric_dtype,
                    "str": lambda x: pd.api.types.is_object_dtype(x) or isinstance(x.dtype, pd.CategoricalDtype),
                    "datetime": pd.api.types.is_datetime64_any_dtype,
                    "bool": pd.api.types.is_bool_dtype,
                }[want](s)
                if not ok:
                    fail(exp, f"dtype is {s.dtype}, expected {want}", n)

            elif exp.kind == "in_range":
                lo = exp.params.get("min", -np.inf)
                hi = exp.params.get("max", np.inf)
                num = pd.to_numeric(s, errors="coerce")
                bad = int(((num < lo) | (num > hi)).sum())
                if bad:
                    fail(exp, f"{bad} values outside [{lo}, {hi}]", bad)

            elif exp.kind == "in_set":
                allowed = set(exp.params["values"])
                mask = ~s.isna()
                bad = int((~s[mask].isin(allowed)).sum())
                if bad:
                    observed = sorted(set(s[mask].unique()) - allowed)[:6]
                    fail(exp, f"{bad} values outside allowed set; unexpected: {observed}", bad)

            elif exp.kind == "unique":
                subset = exp.params.get("subset")
                keys = df[subset] if subset else s.to_frame()
                bad = int(keys.duplicated().sum())
                if bad:
                    fail(exp, f"{bad} duplicate keys on {subset or exp.column}", bad)

            elif exp.kind == "monotonic":
                by = exp.params.get("by")
                within = exp.params.get("within")
                frame = df.sort_values([c for c in [within, by] if c]) if (by or within) else df
                if within:
                    bad = int((~frame.groupby(within, sort=False)[exp.column].apply(lambda x: x.is_monotonic_increasing)).sum())
                    if bad:
                        fail(exp, f"{bad} groups not monotonically increasing", bad)
                elif not frame[exp.column].is_monotonic_increasing:
                    fail(exp, "series is not monotonically increasing", n)

            elif exp.kind == "regex":
                pat = re.compile(exp.params["pattern"])
                mask = ~s.isna()
                bad = int((~s[mask].astype(str).str.match(pat)).sum())
                if bad:
                    fail(exp, f"{bad} values do not match /{exp.params['pattern']}/", bad)

        errs = [f for f in failures if f["severity"] == "error"]
        return ValidationResult(passed=not errs, failures=failures, n_rows=n, contract_name=self.name)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataContract:
        """Load a contract from a YAML file with keys ``name`` and ``expectations``."""
        from prism.utils.optional import require

        yaml = require("yaml")
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        exps = [
            Expectation(
                column=e["column"],
                kind=e["kind"],
                params=e.get("params", {}) or {},
                severity=e.get("severity", "error"),
            )
            for e in spec.get("expectations", [])
        ]
        return cls(name=spec.get("name", Path(path).stem), expectations=exps)

    def to_yaml(self, path: str | Path) -> None:
        """Write this contract to YAML."""
        from prism.utils.optional import require

        yaml = require("yaml")
        payload = {
            "name": self.name,
            "expectations": [
                {"column": e.column, "kind": e.kind, "params": e.params, "severity": e.severity}
                for e in self.expectations
            ],
        }
        Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<DataContract {self.name} n_expectations={len(self.expectations)}>"


def build_panel_contract() -> DataContract:
    """Construct the gold-panel contract described in SPEC section 2.2.

    Imports the feature-name tuples lazily so that :mod:`prism.utils.validation` never
    depends on :mod:`prism.data` at import time.
    """
    from prism.data.schema import CATEGORICAL_FEATURES, CATEGORICAL_LEVELS, NUMERIC_FEATURES

    exps: list[Expectation] = [
        Expectation("*", "row_count", {"min": 1}),
        Expectation("customer_id", "not_null", {}),
        Expectation("customer_id", "dtype", {"dtype": "str"}),
        Expectation("period", "dtype", {"dtype": "int"}),
        Expectation("period", "in_range", {"min": 0, "max": 10_000}),
        Expectation("as_of_date", "dtype", {"dtype": "datetime"}),
        Expectation("as_of_date", "not_null", {}),
        Expectation("cohort", "regex", {"pattern": r"^\d{4}-\d{2}$"}),
        Expectation("split", "in_set", {"values": ["train", "valid", "test"]}),
        Expectation("assignment_block", "in_set", {"values": ["observational", "rct"]}),
        Expectation("arm", "in_set", {"values": [0, 1, 2, 3]}),
        Expectation("treated", "in_set", {"values": [0, 1]}),
        Expectation("offer_cost", "in_range", {"min": 0.0, "max": 1_000.0}),
        Expectation("churn_next", "in_set", {"values": [0, 1]}),
        Expectation("event_time", "in_range", {"min": 0.0}),
        Expectation("event_observed", "in_set", {"values": [0, 1]}),
        Expectation("rmst_h", "in_range", {"min": 0.0}),
        Expectation("value_h", "in_range", {"min": -1e6}),
        Expectation("margin_rate", "in_range", {"min": 0.0, "max": 1.0}),
        Expectation("customer_id", "unique", {"subset": ["customer_id", "period"]}),
        Expectation("period", "monotonic", {"within": "customer_id", "by": "period"}, severity="warn"),
    ]
    for col in NUMERIC_FEATURES:
        exps.append(Expectation(col, "dtype", {"dtype": "numeric"}))
        exps.append(Expectation(col, "not_null", {"max_frac": 0.35}, severity="warn"))
    for col in CATEGORICAL_FEATURES:
        exps.append(Expectation(col, "in_set", {"values": list(CATEGORICAL_LEVELS[col])}))
    return DataContract("gold_panel", exps)


class _LazyContract:
    """Defer contract construction until first use (avoids an import cycle with prism.data)."""

    _inner: DataContract | None = None

    def _get(self) -> DataContract:
        if self._inner is None:
            self._inner = build_panel_contract()
        return self._inner

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate ``df`` against the gold-panel contract."""
        return self._get().validate(df)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._get(), item)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return repr(self._get())


PANEL_CONTRACT = _LazyContract()


if __name__ == "__main__":  # pragma: no cover - smoke test
    df = pd.DataFrame(
        {
            "customer_id": ["a", "b", "b"],
            "period": [0, 0, 0],
            "score": [0.1, 1.4, np.nan],
            "tier": ["basic", "gold", "basic"],
        }
    )
    contract = DataContract(
        "demo",
        [
            Expectation("customer_id", "unique", {"subset": ["customer_id", "period"]}),
            Expectation("score", "in_range", {"min": 0.0, "max": 1.0}),
            Expectation("score", "not_null", {}),
            Expectation("tier", "in_set", {"values": ["basic", "plus"]}, severity="warn"),
        ],
    )
    res = contract.validate(df)
    print(res)
    print(res.summary().to_string(index=False))
    try:
        res.raise_for_status()
    except DataContractError as exc:
        print("raised as expected:", str(exc).splitlines()[0])
    print("validation.py OK")
